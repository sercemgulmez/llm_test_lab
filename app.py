"""
LLM Test Lab — Web UI (Flask)

Başlatmak için:
    python app.py
Ardından tarayıcıda aç: http://localhost:5000
"""

import csv
import json
import os
import queue
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
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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
        _orig_stdout.write(text)
        _orig_stdout.flush()
        with self._lock:
            self.buf += text
            while "\n" in self.buf:
                line, self.buf = self.buf.split("\n", 1)
                if line.strip():
                    self.q.put({"type": "log", "text": line})

    def flush(self):
        _orig_stdout.flush()


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
    extra_headers: dict = {}
    for h in data.get("headers", []):
        k, _, v = h.partition(": ")
        if k:
            extra_headers[k.strip()] = v.strip()

    cookies: dict = {}
    for part in (data.get("cookie") or "").split("; "):
        k, _, v = part.partition("=")
        if k.strip():
            cookies[k.strip()] = v

    # Operasyonlar
    operations = []

    if source == "curl":
        curl_path = data.get("curl_file_path", "")
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
        spec = load_openapi_from_url(
            data["openapi_url"],
            headers=extra_headers or None,
            cookies=cookies or None,
        )
        operations = extract_operations_from_openapi(spec)
        print(f"{len(operations)} operasyon çıkarıldı.")

    if not operations:
        raise ValueError("Hiç operasyon bulunamadı.")

    os.makedirs(output_dir, exist_ok=True)
    save_operations_csv(operations, output_dir)

    # Senaryo üretimi
    all_rows: list = []

    if not selected_keys or "traditional" in selected_keys:
        trad = TraditionalGenerator()
        rows = trad.generate(operations, "", "", 0)
        all_rows.extend(rows)
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

    if not no_run:
        save_generator_metrics_csv(metrics, output_dir)

    print(f"\nTamamlandı! Çıktılar: {output_dir}/")
    return {
        "result_file": result_path,
        "metrics": metrics,
        "comparison": comparison,
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
            _jobs[job_id]["output_dir"] = result["output_dir"]
    except Exception as e:
        q.put({"type": "error", "text": str(e)})
        with _jobs_lock:
            _jobs[job_id]["status"] = "error"
    finally:
        sys.stdout = orig
        _running.clear()
        q.put(None)  # sentinel — SSE akışını bitirir


# ── Flask route'ları ─────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template(
        "index.html",
        openai_models=config.OPENAI_MODELS,
        gemini_models=config.GEMINI_MODELS,
        claude_models=config.CLAUDE_MODELS,
        default_output=config.OUTPUT_DIR,
        default_num_cases=config.NUM_CASES_PER_OPERATION,
        max_cases=config.MAX_CASES_PER_OPERATION,
    )


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "Dosya bulunamadı"}), 400
    f = request.files["file"]
    safe_name = uuid.uuid4().hex
    path = os.path.join(UPLOAD_FOLDER, safe_name)
    f.save(path)
    return jsonify({"path": path, "name": f.filename})


@app.route("/run", methods=["POST"])
def run_job():
    if _running.is_set():
        return jsonify({"error": "Zaten bir iş çalışıyor. Lütfen tamamlanmasını bekleyin."}), 409

    data = request.get_json()
    job_id = uuid.uuid4().hex[:8]

    with _jobs_lock:
        _jobs[job_id] = {
            "status": "running",
            "log_queue": queue.Queue(),
            "result_file": None,
            "metrics": [],
            "comparison": {},
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
