"""
LLM Test Lab — Web UI (Flask)

Başlatmak için:
    python app.py
Ardından tarayıcıda aç: http://localhost:5000
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import io
import json
import logging
import os
from pathlib import Path
import queue
from collections import defaultdict
import secrets
import sys
import threading
import uuid
import zipfile

# Windows konsolunda Türkçe karakterlerin doğru görünmesi için UTF-8 zorla
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from flask import Flask, Response, jsonify, render_template, request, send_file
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from security.redaction import redact_secrets

load_dotenv()

import config  # noqa: E402
from generators import TraditionalGenerator, GENERATOR_REGISTRY  # noqa: E402
from parsers.curl_parser import parse_curl_collection  # noqa: E402
from parsers.openapi import extract_operations_from_openapi, load_openapi_from_url  # noqa: E402
from reporters.csv_reporter import (  # noqa: E402
    build_comparison_summary,
    compute_generator_metrics,
    save_generator_metrics_csv,
    save_operations_csv,
    save_results_csv,
)
from runner import run_testcases  # noqa: E402

_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = config.MAX_UPLOAD_BYTES


def _resolve_upload_folder() -> str:
    """Yazılabilir upload klasörünü seçer; gerekirse güvenli fallback kullanır."""
    candidates = [Path(config.UPLOAD_DIR), Path("runtime_uploads")]
    for candidate in candidates:
        try:
            candidate.mkdir(exist_ok=True)
            probe = candidate / f".write_probe_{uuid.uuid4().hex}"
            probe.write_text("", encoding="utf-8")
            probe.unlink()
            return str(candidate)
        except OSError:
            continue

    fallback = Path(f"runtime_uploads_{uuid.uuid4().hex}")
    fallback.mkdir()
    return str(fallback)


UPLOAD_FOLDER = _resolve_upload_folder()
if UPLOAD_FOLDER != config.UPLOAD_DIR:
    _logger.warning("UYARI: '%s' klasörü yazılabilir değil; '%s' kullanılacak.", config.UPLOAD_DIR, UPLOAD_FOLDER)

# ── İş durumu ────────────────────────────────────────────────────────────────
_jobs: dict = {}
_jobs_lock = threading.Lock()
_running = threading.Event()  # Aynı anda tek iş


# ── stdout yakalama ──────────────────────────────────────────────────────────
_orig_stdout = sys.stdout


class _LogCapture:
    """sys.stdout yerine geçer; print çıktısını hem konsola hem kuyruğa yazar."""

    def __init__(self, q: queue.Queue):
        self.q = q
        self.buf = ""
        self._lock = threading.Lock()

    def write(self, text: str):
        with self._lock:
            _orig_stdout.write(text)
            _orig_stdout.flush()
            self.buf += text
            while "\n" in self.buf:
                line, self.buf = self.buf.split("\n", 1)
                if line.strip():
                    self.q.put({"type": "log", "text": line})

    def flush(self):
        _orig_stdout.flush()


class _QueueLogHandler(logging.Handler):
    """logging çıktısını SSE kuyruğuna yönlendirir; iş süresi boyunca root logger'a eklenir."""

    def __init__(self, q: queue.Queue):
        super().__init__()
        self.q = q
        self.setFormatter(logging.Formatter("%(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            if msg.strip():
                self.q.put({"type": "log", "text": msg})
        except Exception:
            self.handleError(record)


def _parse_header_lines(header_lines: list[str]) -> dict[str, str]:
    """HTTP header satırlarını toleranslı biçimde sözlüğe çevirir."""
    parsed: dict[str, str] = {}
    for raw in header_lines:
        key, sep, value = raw.partition(":")
        if sep and key.strip():
            parsed[key.strip()] = value.strip()
    return parsed


def _parse_cookie_string(cookie_str: str) -> dict[str, str]:
    """Cookie string'ini `name=value; other=value` formatında ayrıştırır."""
    parsed: dict[str, str] = {}
    for part in cookie_str.split(";"):
        key, _, value = part.partition("=")
        if key.strip():
            parsed[key.strip()] = value.strip()
    return parsed


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_response(message: str, status: int):
    return jsonify({"error": message}), status


def _is_allowed_upload(filename: str) -> bool:
    suffix = Path(filename or "").suffix.lower()
    return suffix in config.ALLOWED_UPLOAD_EXTENSIONS


def _resolve_safe_output_dir(output_dir: str) -> str:
    raw = (output_dir or config.OUTPUT_DIR).strip() or config.OUTPUT_DIR
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = config.PROJECT_ROOT / candidate
    resolved = candidate.resolve()

    allowed_roots = [root.resolve() for root in config.ALLOWED_OUTPUT_ROOTS]
    if not any(resolved == root or root in resolved.parents for root in allowed_roots):
        allowed = ", ".join(str(root) for root in allowed_roots)
        raise ValueError(f"Çıktı klasörü izin verilen köklerin altında olmalı: {allowed}")
    return str(resolved)


def _resolve_uploaded_file(path: str) -> str:
    if not path:
        raise ValueError("Curl kaynağı seçildi ama dosya yolu gönderilmedi.")

    upload_root = Path(UPLOAD_FOLDER).resolve()
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = (config.PROJECT_ROOT / candidate).resolve()
    else:
        candidate = candidate.resolve()

    if not (candidate == upload_root or upload_root in candidate.parents):
        raise ValueError("Curl dosyası sadece uygulama upload klasöründen okunabilir.")
    if not candidate.is_file():
        raise ValueError("Yüklenen curl dosyası bulunamadı.")
    return str(candidate)


def _job_is_cancel_requested(job_id: str | None) -> bool:
    if not job_id:
        return False
    with _jobs_lock:
        return bool(_jobs.get(job_id, {}).get("cancel_requested"))


def _raise_if_cancelled(job_id: str | None) -> None:
    if _job_is_cancel_requested(job_id):
        raise RuntimeError("İş kullanıcı tarafından iptal edildi.")


def _selected_generator_labels(selected_keys: list[str]) -> list[str]:
    return _selected_generator_keys(selected_keys)


def _build_run_metadata(
    *,
    job_id: str | None,
    data: dict,
    source: str,
    base_url: str,
    output_dir: str,
    selected_keys: list[str],
    num_cases: int,
    operations: list,
    no_run: bool,
) -> dict:
    return {
        "job_id": job_id,
        "created_at": _now_iso(),
        "source": source,
        "base_url": base_url,
        "no_run": no_run,
        "output_dir": output_dir,
        "selected_generators": _selected_generator_labels(selected_keys),
        "prompt_variants": dict(config.PROMPT_VARIANTS),
        "num_cases_per_operation": num_cases,
        "operation_count": len(operations),
        "operation_ids": [getattr(op, "op_id", "") for op in operations],
        "config_snapshot": {
            "openai_models": config.OPENAI_MODELS,
            "gemini_models": config.GEMINI_MODELS,
            "claude_models": config.CLAUDE_MODELS,
            "groq_models": config.GROQ_MODELS,
            "request_timeout": config.REQUEST_TIMEOUT,
            "retry_max_attempts": config.RETRY_MAX_ATTEMPTS,
            "retry_backoff_seconds": config.RETRY_BACKOFF_SECONDS,
            "max_parallel_workers": config.MAX_PARALLEL_WORKERS,
        },
        "input_summary": {
            "headers_count": len(data.get("headers", []) or []),
            "has_cookie": bool(data.get("cookie")),
            "has_auth_token": bool(data.get("auth_token")),
            "openapi_url": data.get("openapi_url") if source == "openapi" else "",
        },
    }


def _save_run_metadata(metadata: dict, output_dir: str) -> str:
    path = Path(output_dir) / f"run_info_{metadata.get('job_id') or 'manual'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    _logger.info("Run metadata kaydedildi: %s", path)
    return str(path)


def _authorized_job(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        return None, _json_response("Not found", 404)

    expected = job.get("access_token")
    if expected:
        provided = request.headers.get("X-Job-Token") or request.args.get("token")
        if not secrets.compare_digest(str(provided or ""), str(expected)):
            return None, _json_response("Bu job için erişim token'ı gerekli.", 403)
    return job, None


def _default_generation_quality() -> dict:
    return {
        "requested_cases_per_operation": 0,
        "total_operations": 0,
        "selected_generator_count": 0,
        "expected_total_cases": 0,
        "generated_total_cases": 0,
        "valid_total_cases": 0,
        "invalid_total_cases": 0,
        "repaired_total_cases": 0,
        "fallback_total_cases": 0,
        "per_generator_case_count": [],
        "per_operation_case_count": [],
        "validation_error_summary": {},
    }


def _selected_generator_keys(selected_keys: list[str]) -> list[str]:
    if selected_keys:
        return list(selected_keys)
    keys = ["traditional"]
    keys.extend(f"openai:{model}" for model in config.OPENAI_MODELS)
    keys.extend(f"gemini:{model}" for model in config.GEMINI_MODELS)
    keys.extend(f"claude:{model}" for model in config.CLAUDE_MODELS)
    keys.extend(f"groq:{model}" for model in config.GROQ_MODELS)
    return keys


def _expected_generation_runs(selected_keys: list[str]) -> int:
    runs = 0
    for key in _selected_generator_keys(selected_keys):
        if key == "traditional":
            runs += 1
        else:
            runs += len(config.PROMPT_VARIANTS)
    return runs


def _row_generation_metadata(row: dict) -> dict:
    metadata = row.get("generation_metadata")
    return metadata if isinstance(metadata, dict) else {}


def _build_generation_quality(
    operations: list,
    selected_keys: list[str],
    num_cases: int,
    rows: list[dict],
    generation_summaries: list[dict],
) -> dict:
    quality = _default_generation_quality()
    quality["requested_cases_per_operation"] = num_cases
    quality["total_operations"] = len(operations)
    quality["selected_generator_count"] = len(_selected_generator_keys(selected_keys))
    quality["expected_total_cases"] = len(operations) * num_cases * _expected_generation_runs(selected_keys)
    quality["generated_total_cases"] = len(rows)

    if generation_summaries:
        quality["valid_total_cases"] = sum(int(item.get("valid_cases", 0) or 0) for item in generation_summaries)
        quality["invalid_total_cases"] = sum(int(item.get("invalid_cases", 0) or 0) for item in generation_summaries)
        quality["repaired_total_cases"] = sum(int(item.get("repaired_cases", 0) or 0) for item in generation_summaries)
        quality["fallback_total_cases"] = sum(int(item.get("fallback_cases", 0) or 0) for item in generation_summaries)
    else:
        quality["valid_total_cases"] = sum(1 for row in rows if not row.get("validation_errors"))
        quality["invalid_total_cases"] = max(quality["generated_total_cases"] - quality["valid_total_cases"], 0)
        quality["repaired_total_cases"] = sum(1 for row in rows if _row_generation_metadata(row).get("repaired") is True)
        quality["fallback_total_cases"] = sum(1 for row in rows if _row_generation_metadata(row).get("fallback") is True)

    if generation_summaries:
        grouped: dict[str, dict] = defaultdict(lambda: {
            "generator": "",
            "generated": 0,
            "valid": 0,
            "invalid": 0,
            "repaired": 0,
            "fallback": 0,
        })
        for item in generation_summaries:
            generator = str(item.get("generator", ""))
            bucket = grouped[generator]
            bucket["generator"] = generator
            bucket["generated"] += int(item.get("generated_cases", 0) or 0)
            bucket["valid"] += int(item.get("valid_cases", 0) or 0)
            bucket["invalid"] += int(item.get("invalid_cases", 0) or 0)
            bucket["repaired"] += int(item.get("repaired_cases", 0) or 0)
            bucket["fallback"] += int(item.get("fallback_cases", 0) or 0)
        quality["per_generator_case_count"] = sorted(grouped.values(), key=lambda item: item["generator"])
    else:
        grouped: dict[str, dict] = {}
        for row in rows:
            generator = str(row.get("generator", ""))
            bucket = grouped.setdefault(generator, {
                "generator": generator,
                "generated": 0,
                "valid": 0,
                "invalid": 0,
                "repaired": 0,
                "fallback": 0,
            })
            metadata = _row_generation_metadata(row)
            bucket["generated"] += 1
            bucket["valid"] += 0 if row.get("validation_errors") else 1
            bucket["invalid"] += 1 if row.get("validation_errors") else 0
            bucket["repaired"] += 1 if metadata.get("repaired") is True else 0
            bucket["fallback"] += 1 if metadata.get("fallback") is True else 0
        quality["per_generator_case_count"] = sorted(grouped.values(), key=lambda item: item["generator"])

    if generation_summaries:
        op_grouped: dict[tuple[str, str, str], dict] = defaultdict(lambda: {
            "operation_id": "",
            "method": "",
            "path": "",
            "expected_count": 0,
            "actual_valid_count": 0,
        })
        for item in generation_summaries:
            key = (str(item.get("operation_id", "")), str(item.get("method", "")), str(item.get("path", "")))
            bucket = op_grouped[key]
            bucket["operation_id"], bucket["method"], bucket["path"] = key
            bucket["expected_count"] += int(item.get("requested_cases", 0) or 0)
            bucket["actual_valid_count"] += int(item.get("valid_cases", 0) or 0)
        quality["per_operation_case_count"] = sorted(op_grouped.values(), key=lambda item: (item["operation_id"], item["method"], item["path"]))
    else:
        expected_per_operation = num_cases * _expected_generation_runs(selected_keys)
        op_grouped: dict[tuple[str, str, str], dict] = defaultdict(lambda: {
            "operation_id": "",
            "method": "",
            "path": "",
            "expected_count": expected_per_operation,
            "actual_valid_count": 0,
        })
        for row in rows:
            key = (str(row.get("operation_id", "")), str(row.get("http_method", "")), str(row.get("path", "")))
            bucket = op_grouped[key]
            bucket["operation_id"], bucket["method"], bucket["path"] = key
            if not row.get("validation_errors"):
                bucket["actual_valid_count"] += 1
        quality["per_operation_case_count"] = sorted(op_grouped.values(), key=lambda item: (item["operation_id"], item["method"], item["path"]))

    validation_summary: dict[str, int] = {}
    if generation_summaries:
        for item in generation_summaries:
            for error, count in (item.get("validation_error_summary") or {}).items():
                validation_summary[str(error)] = validation_summary.get(str(error), 0) + int(count or 0)
    for row in rows:
        for error in row.get("validation_errors") or []:
            validation_summary[str(error)] = validation_summary.get(str(error), 0) + 1
    quality["validation_error_summary"] = dict(sorted(validation_summary.items(), key=lambda item: (-item[1], item[0])))
    return quality


# ── Pipeline ─────────────────────────────────────────────────────────────────

def _execute_pipeline(data: dict) -> dict:
    """Test pipeline'ını çalıştırır ve UI için çıktı özetini döner."""

    job_id = data.get("_job_id")
    source = data.get("source", "openapi")
    base_url = data.get("base_url", "").strip()
    auth_token = data.get("auth_token", "").strip() or None
    no_run = bool(data.get("no_run", False))
    output_dir = _resolve_safe_output_dir(data.get("output_dir") or config.OUTPUT_DIR)
    selected_keys: list = data.get("selected_generators", [])
    num_cases = config.normalize_num_cases(data.get("num_cases"))

    # Header / Cookie
    extra_headers = _parse_header_lines(data.get("headers", []))
    cookies = _parse_cookie_string(data.get("cookie") or "")

    # Operasyonlar
    operations = []
    _raise_if_cancelled(job_id)

    if source == "curl":
        curl_path = _resolve_uploaded_file(data.get("curl_file_path", ""))
        with open(curl_path, "r", encoding="utf-8") as f:
            curl_text = f.read()
        parsed_list = parse_curl_collection(curl_text)
        derived_base = None
        for op, op_base, curl_hdrs, curl_ck in parsed_list:
            if derived_base is None:
                derived_base = op_base
            cookies = {**curl_ck, **cookies}
            extra_headers = {**curl_hdrs, **extra_headers}
            operations.append(op)
        base_url = base_url or derived_base
        _logger.info("%d curl operasyonu parse edildi.", len(operations))

    elif source == "openapi":
        openapi_url = (data.get("openapi_url") or "").strip()
        if not openapi_url:
            raise ValueError("OpenAPI kaynağı seçildi ama URL gönderilmedi.")
        spec = load_openapi_from_url(
            openapi_url,
            headers=extra_headers or None,
            cookies=cookies or None,
        )
        operations = extract_operations_from_openapi(spec)
        _logger.info("%d operasyon çıkarıldı.", len(operations))
    else:
        raise ValueError(f"Desteklenmeyen kaynak türü: {source}")

    if not operations:
        raise ValueError("Hiç operasyon bulunamadı.")
    if not base_url and not no_run:
        raise ValueError("Testleri çalıştırmak için base URL gereklidir.")

    _raise_if_cancelled(job_id)
    os.makedirs(output_dir, exist_ok=True)
    save_operations_csv(operations, output_dir)
    run_metadata = _build_run_metadata(
        job_id=job_id,
        data=data,
        source=source,
        base_url=base_url or "",
        output_dir=output_dir,
        selected_keys=selected_keys,
        num_cases=num_cases,
        operations=operations,
        no_run=no_run,
    )
    run_info_path = _save_run_metadata(run_metadata, output_dir)

    # Senaryo üretimi
    all_rows: list = []
    generation_summaries: list = []

    _raise_if_cancelled(job_id)
    if not selected_keys or "traditional" in selected_keys:
        trad = TraditionalGenerator()
        rows = trad.generate(operations, "", "", num_cases)
        all_rows.extend(rows)
        generation_summaries.extend(getattr(trad, "_generation_summaries", []))
        _logger.info("[Geleneksel] %d senaryo üretildi.", len(rows))

    def _run_llm(gen_instance, model_name: str, provider: str):
        # Pre-flight: API anahtarının env'de var olup olmadığını doğrula
        try:
            gen_instance._get_client()
        except RuntimeError as e:
            _logger.warning("[%s %s] ATLANADI — %s", provider, model_name, e)
            return

        for v_name, v_desc in config.PROMPT_VARIANTS.items():
            _raise_if_cancelled(job_id)
            if gen_instance._aborted:
                _logger.warning(
                    "[%s %s / %s] ⚠ Bu model daha önce API hatası aldı ve atlandı. Lütfen %s anahtarınızı kontrol edin.",
                    provider, model_name, v_name, provider,
                )
                continue
            try:
                rows = gen_instance.generate(
                    operations,
                    variant_name=v_name,
                    variant_desc=v_desc,
                    num_cases=num_cases,
                )
                all_rows.extend(rows)
                generation_summaries.extend(getattr(gen_instance, "_generation_summaries", []))
                _logger.info("[%s %s / %s] %d senaryo üretildi.", provider, model_name, v_name, len(rows))
            except RuntimeError as e:
                _logger.warning("UYARI [%s %s]: %s — atlandı.", provider, model_name, e)

    for key, (cls, model, provider) in GENERATOR_REGISTRY.items():
        if key == "traditional":
            continue
        if not selected_keys or key in selected_keys:
            _run_llm(cls(model), model, provider)

    if generation_summaries:
        _logger.info("\n── Üretim Özeti ──")
        for summary in generation_summaries:
            gen = summary.get("generator", "?")
            req = summary.get("requested_cases", 0)
            valid = summary.get("valid_cases", 0)
            fallback = summary.get("fallback_cases", 0)
            status = "✓" if fallback == 0 else f"{fallback} fallback"
            _logger.info("  %s: %d/%d case (%s)", gen, valid, req, status)

    _logger.info("\nToplam %d test senaryosu üretildi.", len(all_rows))

    # Test çalıştırma
    _raise_if_cancelled(job_id)
    if no_run:
        executed_rows = all_rows
        _logger.info("Testler çalıştırılmıyor (no_run aktif).")
    else:
        executed_rows = run_testcases(
            base_url, all_rows,
            auth_token=auth_token,
            extra_headers=extra_headers or None,
            cookies=cookies or None,
        )

    result_path = save_results_csv(executed_rows, output_dir)

    metrics = compute_generator_metrics(executed_rows)
    comparison = build_comparison_summary(executed_rows)
    generation_quality = _build_generation_quality(
        operations=operations,
        selected_keys=selected_keys,
        num_cases=num_cases,
        rows=executed_rows,
        generation_summaries=generation_summaries,
    )

    if not no_run:
        save_generator_metrics_csv(metrics, output_dir)

    _logger.info("\nTamamlandı! Çıktılar: %s/", output_dir)
    return {
        "result_file": result_path,
        "run_info": run_metadata,
        "run_info_file": run_info_path,
        "metrics": metrics,
        "comparison": comparison,
        "generation_quality": generation_quality,
        "output_dir": output_dir,
    }


def _job_thread(job_id: str, data: dict):
    q = _jobs[job_id]["log_queue"]
    orig = sys.stdout
    sys.stdout = _LogCapture(q)
    log_handler = _QueueLogHandler(q)
    logging.root.addHandler(log_handler)
    try:
        data = dict(data)
        data["_job_id"] = job_id
        result = _execute_pipeline(data)
        with _jobs_lock:
            _jobs[job_id]["status"] = "cancelled" if _jobs[job_id].get("cancel_requested") else "done"
            _jobs[job_id]["result_file"] = result["result_file"]
            _jobs[job_id]["run_info"] = result.get("run_info", {})
            _jobs[job_id]["run_info_file"] = result.get("run_info_file")
            _jobs[job_id]["metrics"] = result["metrics"]
            _jobs[job_id]["comparison"] = result["comparison"]
            _jobs[job_id]["generation_quality"] = result["generation_quality"]
            _jobs[job_id]["output_dir"] = result["output_dir"]
    except Exception as e:
        message = redact_secrets(str(e))
        q.put({"type": "error", "text": message})
        with _jobs_lock:
            if _jobs[job_id].get("cancel_requested"):
                _jobs[job_id]["status"] = "cancelled"
            else:
                _jobs[job_id]["status"] = "error"
            _jobs[job_id]["error"] = message
    finally:
        logging.root.removeHandler(log_handler)
        sys.stdout = orig
        _running.clear()
        q.put(None)  # sentinel — SSE akışını bitirir
        # Bellek sızıntısını önlemek için eski işleri temizle (son 20 iş tutulur)
        with _jobs_lock:
            completed = [jid for jid, j in _jobs.items() if j["status"] != "running"]
            for old_id in completed[:-20]:
                del _jobs[old_id]


# ── Flask route'ları ─────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template(
        "index.html",
        openai_models=config.OPENAI_MODELS,
        gemini_models=config.GEMINI_MODELS,
        claude_models=config.CLAUDE_MODELS,
        groq_models=config.GROQ_MODELS,
        default_output=config.OUTPUT_DIR,
        default_num_cases=config.NUM_CASES_PER_OPERATION,
        max_cases=config.MAX_CASES_PER_OPERATION,
    )


@app.errorhandler(413)
def payload_too_large(_error):
    mb = config.MAX_UPLOAD_BYTES / (1024 * 1024)
    return _json_response(f"Dosya çok büyük. Maksimum upload boyutu: {mb:.1f} MB.", 413)


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "running": _running.is_set(),
        "jobs": len(_jobs),
        "max_parallel_jobs": config.MAX_PARALLEL_JOBS,
    })


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return _json_response("Dosya bulunamadı", 400)
    f = request.files["file"]
    original_name = secure_filename(f.filename or "")
    if not original_name:
        return _json_response("Dosya adı boş olamaz", 400)
    if not _is_allowed_upload(original_name):
        allowed = ", ".join(sorted(config.ALLOWED_UPLOAD_EXTENSIONS))
        return _json_response(f"Desteklenmeyen dosya türü. İzin verilenler: {allowed}", 400)
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    safe_name = f"{uuid.uuid4().hex}{Path(original_name).suffix.lower()}"
    path = os.path.join(UPLOAD_FOLDER, safe_name)
    f.save(path)
    return jsonify({"path": path, "name": f.filename, "size": os.path.getsize(path)})


@app.route("/run", methods=["POST"])
def run_job():
    if _running.is_set():
        return _json_response("Zaten bir iş çalışıyor. Lütfen tamamlanmasını bekleyin.", 409)

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return _json_response("Geçerli bir JSON payload gönderilmedi.", 400)
    try:
        safe_output = _resolve_safe_output_dir(data.get("output_dir") or config.OUTPUT_DIR)
    except ValueError as exc:
        return _json_response(redact_secrets(str(exc)), 400)

    job_id = secrets.token_hex(config.JOB_ID_BYTES)
    access_token = secrets.token_urlsafe(config.JOB_TOKEN_BYTES)

    with _jobs_lock:
        _jobs[job_id] = {
            "status": "running",
            "created_at": _now_iso(),
            "cancel_requested": False,
            "access_token": access_token,
            "log_queue": queue.Queue(),
            "result_file": None,
            "run_info": {},
            "run_info_file": None,
            "metrics": [],
            "comparison": {},
            "generation_quality": _default_generation_quality(),
            "output_dir": safe_output,
            "error": "",
        }

    _running.set()
    threading.Thread(target=_job_thread, args=(job_id, data), daemon=True).start()
    return jsonify({"job_id": job_id, "job_token": access_token})


@app.route("/stream/<job_id>")
def stream(job_id: str):
    job, error = _authorized_job(job_id)
    if error:
        return error

    def generate():
        q = job["log_queue"]
        while True:
            try:
                msg = q.get(timeout=60)
                if msg is None:
                    current = _jobs[job_id]
                    payload = json.dumps({
                        "type": "done",
                        "status": current["status"],
                        "result_file": current.get("result_file"),
                    })
                    yield f"data: {payload}\n\n"
                    break
                yield f"data: {json.dumps(msg)}\n\n"
            except queue.Empty:
                yield f"data: {json.dumps({'type': 'ping'})}\n\n"

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/result_data/<job_id>")
def result_data(job_id: str):
    job, error = _authorized_job(job_id)
    if error:
        return error
    result_file = job.get("result_file")
    if not result_file or not os.path.exists(result_file):
        return jsonify({"headers": [], "rows": []})
    with open(result_file, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        headers = next(reader, [])
        rows = list(reader)
    return jsonify({"headers": headers, "rows": rows})


@app.route("/metrics/<job_id>")
def metrics_data(job_id: str):
    """Generator metrik özetini döner (pass rate, vb.)"""
    job, error = _authorized_job(job_id)
    if error:
        return error
    return jsonify(job.get("metrics", []))


@app.route("/comparison/<job_id>")
def comparison_data(job_id: str):
    job, error = _authorized_job(job_id)
    if error:
        return error
    return jsonify(job.get("comparison", {}))


@app.route("/generation_quality/<job_id>")
def generation_quality_data(job_id: str):
    job, error = _authorized_job(job_id)
    if error:
        return error
    return jsonify(job.get("generation_quality") or _default_generation_quality())


@app.route("/run_info/<job_id>")
def run_info_data(job_id: str):
    job, error = _authorized_job(job_id)
    if error:
        return error
    return jsonify({
        "job_id": job_id,
        "status": job.get("status"),
        "created_at": job.get("created_at"),
        "output_dir": job.get("output_dir"),
        "result_file": job.get("result_file"),
        "run_info_file": job.get("run_info_file"),
        "run_info": job.get("run_info") or {},
        "error": job.get("error", ""),
    })


@app.route("/cancel/<job_id>", methods=["POST"])
def cancel_job(job_id: str):
    job, error = _authorized_job(job_id)
    if error:
        return error
    with _jobs_lock:
        if job.get("status") != "running":
            return jsonify({"status": job.get("status"), "cancel_requested": False})
        job["cancel_requested"] = True
        job["status"] = "cancelling"
        q = job.get("log_queue")
        if q is not None:
            q.put({"type": "log", "text": "İptal isteği alındı. Güvenli durma noktası bekleniyor."})
    return jsonify({"status": "cancelling", "cancel_requested": True})


@app.route("/download/<job_id>")
def download(job_id: str):
    job, error = _authorized_job(job_id)
    if error:
        return error
    result_file = job.get("result_file")
    if not result_file or not os.path.exists(result_file):
        return "Dosya bulunamadı", 404
    return send_file(result_file, as_attachment=True)


@app.route("/download_report/<job_id>")
def download_report(job_id: str):
    job, error = _authorized_job(job_id)
    if error:
        return error

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for label, path in {
            "results.csv": job.get("result_file"),
            "run_info.json": job.get("run_info_file"),
        }.items():
            if path and os.path.exists(path):
                zf.write(path, arcname=label)

        output_dir = job.get("output_dir")
        if output_dir:
            for filename in ("operations.csv",):
                path = os.path.join(output_dir, filename)
                if os.path.exists(path):
                    zf.write(path, arcname=filename)

        zf.writestr("metrics.json", json.dumps(job.get("metrics", []), ensure_ascii=False, indent=2))
        zf.writestr("comparison.json", json.dumps(job.get("comparison", {}), ensure_ascii=False, indent=2))
        zf.writestr(
            "generation_quality.json",
            json.dumps(job.get("generation_quality") or _default_generation_quality(), ensure_ascii=False, indent=2),
        )

    buffer.seek(0)
    return send_file(
        buffer,
        mimetype="application/zip",
        as_attachment=True,
        download_name=f"llm_test_lab_report_{job_id}.zip",
    )


if __name__ == "__main__":
    print("=" * 50)
    print("  LLM Test Lab Web UI")
    print("  http://localhost:5000")
    print("=" * 50)
    app.run(debug=False, port=5000, threaded=True)
