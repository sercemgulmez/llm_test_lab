"""
LLM Test Lab — Web UI (Flask)

Başlatmak için:
    python app.py
Ardından tarayıcıda aç: http://localhost:5000
"""

import csv
import json
import os
from pathlib import Path
import queue
from collections import defaultdict
import sys
import threading
import uuid

# Windows konsolunda Türkçe karakterlerin doğru görünmesi için UTF-8 zorla
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from flask import Flask, Response, jsonify, render_template, request, send_file
from dotenv import load_dotenv

load_dotenv()

import config
from generators.claude_gen import ClaudeGenerator
from generators.groq_gen import GroqGenerator
from generators.gemini_gen import GeminiGenerator
from generators.openai_gen import OpenAIGenerator
from generators.traditional import TraditionalGenerator
from parsers.curl_parser import parse_curl_collection
from parsers.openapi import extract_operations_from_openapi, load_openapi_from_url
from reporters.csv_reporter import (
    build_comparison_summary,
    compute_generator_metrics,
    save_generator_metrics_csv,
    save_operations_csv,
    save_results_csv,
)
from runner import run_testcases

app = Flask(__name__)


def _resolve_upload_folder() -> str:
    """Yazılabilir upload klasörünü seçer; gerekirse güvenli fallback kullanır."""
    candidates = [Path("uploads"), Path("runtime_uploads")]
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
if UPLOAD_FOLDER != "uploads":
    print(f"UYARI: 'uploads' klasörü yazılabilir değil; '{UPLOAD_FOLDER}' kullanılacak.")

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

    source = data.get("source", "openapi")
    base_url = data.get("base_url", "").strip()
    auth_token = data.get("auth_token", "").strip() or None
    no_run = bool(data.get("no_run", False))
    output_dir = (data.get("output_dir") or config.OUTPUT_DIR).strip()
    selected_keys: list = data.get("selected_generators", [])
    num_cases = config.normalize_num_cases(data.get("num_cases"))

    # Header / Cookie
    extra_headers = _parse_header_lines(data.get("headers", []))
    cookies = _parse_cookie_string(data.get("cookie") or "")

    # Operasyonlar
    operations = []

    if source == "curl":
        curl_path = data.get("curl_file_path", "")
        if not curl_path:
            raise ValueError("Curl kaynağı seçildi ama dosya yolu gönderilmedi.")
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
        print(f"{len(operations)} curl operasyonu parse edildi.")

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
        print(f"{len(operations)} operasyon çıkarıldı.")
    else:
        raise ValueError(f"Desteklenmeyen kaynak türü: {source}")

    if not operations:
        raise ValueError("Hiç operasyon bulunamadı.")
    if not base_url and not no_run:
        raise ValueError("Testleri çalıştırmak için base URL gereklidir.")

    os.makedirs(output_dir, exist_ok=True)
    save_operations_csv(operations, output_dir)

    # Senaryo üretimi
    all_rows: list = []
    generation_summaries: list = []

    if not selected_keys or "traditional" in selected_keys:
        trad = TraditionalGenerator()
        rows = trad.generate(operations, "", "", num_cases)
        all_rows.extend(rows)
        generation_summaries.extend(getattr(trad, "_generation_summaries", []))
        print(f"[Geleneksel] {len(rows)} senaryo üretildi.")

    def _run_llm(gen_instance, model_name: str, provider: str):
        for v_name, v_desc in config.PROMPT_VARIANTS.items():
            if gen_instance._aborted:
                print(f"[{provider} {model_name} / {v_name}] kredi/auth hatası nedeniyle atlandı.")
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
                print(f"[{provider} {model_name} / {v_name}] {len(rows)} senaryo üretildi.")
            except RuntimeError as e:
                print(f"UYARI [{provider} {model_name}]: {e} — atlandı.")

    for m in config.OPENAI_MODELS:
        if not selected_keys or f"openai:{m}" in selected_keys:
            _run_llm(OpenAIGenerator(m), m, "OpenAI")

    for m in config.GEMINI_MODELS:
        if not selected_keys or f"gemini:{m}" in selected_keys:
            _run_llm(GeminiGenerator(m), m, "Gemini")

    for m in config.CLAUDE_MODELS:
        if not selected_keys or f"claude:{m}" in selected_keys:
            _run_llm(ClaudeGenerator(m), m, "Claude")

    for m in config.GROQ_MODELS:
        if not selected_keys or f"groq:{m}" in selected_keys:
            _run_llm(GroqGenerator(m), m, "Groq")

    print(f"\nToplam {len(all_rows)} test senaryosu üretildi.")

    # Test çalıştırma
    if no_run:
        executed_rows = all_rows
        print("Testler çalıştırılmıyor (no_run aktif).")
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

    print(f"\nTamamlandı! Çıktılar: {output_dir}/")
    return {
        "result_file": result_path,
        "metrics": metrics,
        "comparison": comparison,
        "generation_quality": generation_quality,
        "output_dir": output_dir,
    }


def _job_thread(job_id: str, data: dict):
    q = _jobs[job_id]["log_queue"]
    orig = sys.stdout
    sys.stdout = _LogCapture(q)
    try:
        result = _execute_pipeline(data)
        with _jobs_lock:
            _jobs[job_id]["status"] = "done"
            _jobs[job_id]["result_file"] = result["result_file"]
            _jobs[job_id]["metrics"] = result["metrics"]
            _jobs[job_id]["comparison"] = result["comparison"]
            _jobs[job_id]["generation_quality"] = result["generation_quality"]
            _jobs[job_id]["output_dir"] = result["output_dir"]
    except Exception as e:
        q.put({"type": "error", "text": str(e)})
        with _jobs_lock:
            _jobs[job_id]["status"] = "error"
    finally:
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


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "Dosya bulunamadı"}), 400
    f = request.files["file"]
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    safe_name = uuid.uuid4().hex
    path = os.path.join(UPLOAD_FOLDER, safe_name)
    f.save(path)
    return jsonify({"path": path, "name": f.filename})


@app.route("/run", methods=["POST"])
def run_job():
    if _running.is_set():
        return jsonify({"error": "Zaten bir iş çalışıyor. Lütfen tamamlanmasını bekleyin."}), 409

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Geçerli bir JSON payload gönderilmedi."}), 400
    job_id = uuid.uuid4().hex[:8]

    with _jobs_lock:
        _jobs[job_id] = {
            "status": "running",
            "log_queue": queue.Queue(),
            "result_file": None,
            "metrics": [],
            "comparison": {},
            "generation_quality": _default_generation_quality(),
            "output_dir": data.get("output_dir") or config.OUTPUT_DIR,
        }

    _running.set()
    threading.Thread(target=_job_thread, args=(job_id, data), daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/stream/<job_id>")
def stream(job_id: str):
    if job_id not in _jobs:
        return "Not found", 404

    def generate():
        q = _jobs[job_id]["log_queue"]
        while True:
            try:
                msg = q.get(timeout=60)
                if msg is None:
                    job = _jobs[job_id]
                    payload = json.dumps({
                        "type": "done",
                        "status": job["status"],
                        "result_file": job.get("result_file"),
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
    if job_id not in _jobs:
        return jsonify({"error": "Not found"}), 404
    result_file = _jobs[job_id].get("result_file")
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
    if job_id not in _jobs:
        return jsonify([])
    return jsonify(_jobs[job_id].get("metrics", []))


@app.route("/comparison/<job_id>")
def comparison_data(job_id: str):
    if job_id not in _jobs:
        return jsonify({})
    return jsonify(_jobs[job_id].get("comparison", {}))


@app.route("/generation_quality/<job_id>")
def generation_quality_data(job_id: str):
    if job_id not in _jobs:
        return jsonify(_default_generation_quality())
    return jsonify(_jobs[job_id].get("generation_quality") or _default_generation_quality())


@app.route("/download/<job_id>")
def download(job_id: str):
    if job_id not in _jobs:
        return "Not found", 404
    result_file = _jobs[job_id].get("result_file")
    if not result_file or not os.path.exists(result_file):
        return "Dosya bulunamadı", 404
    return send_file(result_file, as_attachment=True)


if __name__ == "__main__":
    print("=" * 50)
    print("  LLM Test Lab Web UI")
    print("  http://localhost:5000")
    print("=" * 50)
    app.run(debug=False, port=5000, threaded=True)
