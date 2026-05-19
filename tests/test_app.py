import csv
import io
import queue
import zipfile

import pytest

import app as web_app


@pytest.fixture(autouse=True)
def _reset_app_state():
    web_app._jobs.clear()
    web_app._running.clear()
    yield
    web_app._jobs.clear()
    web_app._running.clear()


def test_index_page_renders():
    client = web_app.app.test_client()

    resp = client.get("/")

    assert resp.status_code == 200
    assert b"LLM Test Lab" in resp.data


def test_run_returns_409_when_another_job_is_running():
    client = web_app.app.test_client()
    web_app._running.set()
    try:
        resp = client.post("/run", json={"source": "openapi"})
    finally:
        web_app._running.clear()

    assert resp.status_code == 409


def test_run_returns_400_for_invalid_json_payload():
    client = web_app.app.test_client()

    resp = client.post("/run", data="not-json", content_type="application/json")

    assert resp.status_code == 400
    assert resp.get_json()["error"]


def test_run_rejects_output_dir_outside_project():
    client = web_app.app.test_client()

    resp = client.post("/run", json={"source": "openapi", "output_dir": "/tmp/outside-llm-test-lab"})

    assert resp.status_code == 400
    assert "izin verilen" in resp.get_json()["error"]


def test_result_metrics_and_comparison_endpoints_return_job_data(tmp_path):
    result_file = tmp_path / "results.csv"
    with result_file.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["generator", "pass"])
        writer.writerow(["A", "True"])

    web_app._jobs["job-test"] = {
        "status": "done",
        "access_token": "",
        "created_at": "2026-01-01T00:00:00+00:00",
        "log_queue": None,
        "result_file": str(result_file),
        "run_info": {"job_id": "job-test"},
        "run_info_file": None,
        "metrics": [{"generator": "A", "pass_rate": 1.0}],
        "comparison": {"overview": {"best_generator": "A"}},
        "generation_quality": {"expected_total_cases": 10, "generated_total_cases": 10},
        "output_dir": str(tmp_path),
    }

    client = web_app.app.test_client()
    result_resp = client.get("/result_data/job-test")
    metrics_resp = client.get("/metrics/job-test")
    comparison_resp = client.get("/comparison/job-test")
    quality_resp = client.get("/generation_quality/job-test")
    run_info_resp = client.get("/run_info/job-test")

    assert result_resp.status_code == 200
    assert result_resp.get_json()["headers"] == ["generator", "pass"]
    assert metrics_resp.get_json()[0]["generator"] == "A"
    assert comparison_resp.get_json()["overview"]["best_generator"] == "A"
    assert quality_resp.get_json()["expected_total_cases"] == 10
    assert run_info_resp.get_json()["run_info"]["job_id"] == "job-test"

    web_app._jobs.pop("job-test", None)


def test_generation_quality_endpoint_returns_404_when_job_missing():
    client = web_app.app.test_client()

    resp = client.get("/generation_quality/missing-job")

    payload = resp.get_json()
    assert resp.status_code == 404
    assert payload["error"]


def test_health_endpoint_reports_ok():
    client = web_app.app.test_client()

    resp = client.get("/health")

    assert resp.status_code == 200
    assert resp.get_json()["status"] == "ok"


def test_upload_accepts_file(tmp_path, monkeypatch):
    client = web_app.app.test_client()
    monkeypatch.setattr(web_app, "UPLOAD_FOLDER", str(tmp_path))

    resp = client.post(
        "/upload",
        data={"file": (io.BytesIO(b"curl https://api.example.com"), "sample.curl")},
        content_type="multipart/form-data",
    )

    payload = resp.get_json()
    assert resp.status_code == 200
    assert payload["name"] == "sample.curl"
    assert payload["path"].endswith(".curl")
    assert (tmp_path / payload["path"].split("/")[-1]).exists()


def test_upload_rejects_unsupported_extension(tmp_path, monkeypatch):
    client = web_app.app.test_client()
    monkeypatch.setattr(web_app, "UPLOAD_FOLDER", str(tmp_path))

    resp = client.post(
        "/upload",
        data={"file": (io.BytesIO(b"curl https://api.example.com"), "sample.exe")},
        content_type="multipart/form-data",
    )

    assert resp.status_code == 400
    assert "Desteklenmeyen" in resp.get_json()["error"]


def test_run_job_executes_with_synchronous_thread(monkeypatch, tmp_path):
    result_file = tmp_path / "result.csv"
    result_file.write_text("generator,pass\nA,True\n", encoding="utf-8")

    def fake_execute_pipeline(data):
        assert data["source"] == "openapi"
        assert data["_job_id"]
        return {
            "result_file": str(result_file),
            "run_info": {"job_id": data["_job_id"]},
            "run_info_file": None,
            "metrics": [{"generator": "A", "pass_rate": 1.0}],
            "comparison": {"overview": {"best_generator": "A"}},
            "generation_quality": {"expected_total_cases": 4, "generated_total_cases": 4},
            "output_dir": str(tmp_path),
        }

    class ImmediateThread:
        def __init__(self, target, args=(), daemon=None):
            self._target = target
            self._args = args

        def start(self):
            self._target(*self._args)

    monkeypatch.setattr(web_app, "_execute_pipeline", fake_execute_pipeline)
    monkeypatch.setattr(web_app.threading, "Thread", ImmediateThread)
    monkeypatch.setattr(web_app.secrets, "token_hex", lambda _n: "job12345deadbeef")
    monkeypatch.setattr(web_app.secrets, "token_urlsafe", lambda _n: "token-123")

    client = web_app.app.test_client()
    resp = client.post("/run", json={"source": "openapi", "output_dir": str(tmp_path)})

    payload = resp.get_json()
    job_id = payload["job_id"]
    assert resp.status_code == 200
    assert job_id == "job12345deadbeef"
    assert payload["job_token"] == "token-123"
    assert web_app._jobs[job_id]["status"] == "done"
    assert web_app._jobs[job_id]["comparison"]["overview"]["best_generator"] == "A"
    assert web_app._jobs[job_id]["generation_quality"]["generated_total_cases"] == 4


def test_protected_job_endpoints_require_token(tmp_path):
    result_file = tmp_path / "result.csv"
    result_file.write_text("generator,pass\nA,True\n", encoding="utf-8")
    web_app._jobs["secure-job"] = {
        "status": "done",
        "access_token": "secret-token",
        "created_at": "2026-01-01T00:00:00+00:00",
        "log_queue": queue.Queue(),
        "result_file": str(result_file),
        "run_info": {},
        "run_info_file": None,
        "metrics": [],
        "comparison": {},
        "generation_quality": web_app._default_generation_quality(),
        "output_dir": str(tmp_path),
        "error": "",
    }

    client = web_app.app.test_client()

    denied = client.get("/metrics/secure-job")
    allowed = client.get("/metrics/secure-job?token=secret-token")

    assert denied.status_code == 403
    assert allowed.status_code == 200


def test_stream_endpoint_emits_log_and_done_messages():
    job_id = "job-stream"
    q = queue.Queue()
    q.put({"type": "log", "text": "step 1"})
    q.put(None)
    web_app._jobs[job_id] = {
        "status": "done",
        "access_token": "",
        "created_at": "2026-01-01T00:00:00+00:00",
        "log_queue": q,
        "result_file": "out.csv",
        "run_info": {},
        "run_info_file": None,
        "metrics": [],
        "comparison": {},
        "generation_quality": web_app._default_generation_quality(),
        "output_dir": "outputs",
        "error": "",
    }

    client = web_app.app.test_client()
    resp = client.get(f"/stream/{job_id}")
    body = b"".join(resp.response).decode("utf-8")

    assert resp.status_code == 200
    assert '"type": "log"' in body
    assert '"type": "done"' in body
    assert '"status": "done"' in body


def test_download_endpoint_returns_csv_file(tmp_path):
    result_file = tmp_path / "result.csv"
    result_file.write_text("generator,pass\nA,True\n", encoding="utf-8")
    web_app._jobs["job-download"] = {
        "status": "done",
        "access_token": "",
        "created_at": "2026-01-01T00:00:00+00:00",
        "log_queue": queue.Queue(),
        "result_file": str(result_file),
        "run_info": {},
        "run_info_file": None,
        "metrics": [],
        "comparison": {},
        "generation_quality": web_app._default_generation_quality(),
        "output_dir": str(tmp_path),
        "error": "",
    }

    client = web_app.app.test_client()
    resp = client.get("/download/job-download")

    assert resp.status_code == 200
    assert "attachment" in resp.headers["Content-Disposition"]
    assert b"generator,pass" in resp.data


def test_cancel_endpoint_marks_running_job():
    q = queue.Queue()
    web_app._jobs["job-cancel"] = {
        "status": "running",
        "access_token": "",
        "created_at": "2026-01-01T00:00:00+00:00",
        "cancel_requested": False,
        "log_queue": q,
        "result_file": None,
        "run_info": {},
        "run_info_file": None,
        "metrics": [],
        "comparison": {},
        "generation_quality": web_app._default_generation_quality(),
        "output_dir": "outputs",
        "error": "",
    }

    client = web_app.app.test_client()
    resp = client.post("/cancel/job-cancel")

    assert resp.status_code == 200
    assert resp.get_json()["cancel_requested"] is True
    assert web_app._jobs["job-cancel"]["status"] == "cancelling"


def test_download_report_returns_zip(tmp_path):
    result_file = tmp_path / "result.csv"
    result_file.write_text("generator,pass\nA,True\n", encoding="utf-8")
    run_info = tmp_path / "run_info.json"
    run_info.write_text('{"job_id":"job-report"}', encoding="utf-8")
    (tmp_path / "operations.csv").write_text("op_id\nlistPets\n", encoding="utf-8")
    web_app._jobs["job-report"] = {
        "status": "done",
        "access_token": "",
        "created_at": "2026-01-01T00:00:00+00:00",
        "log_queue": queue.Queue(),
        "result_file": str(result_file),
        "run_info": {"job_id": "job-report"},
        "run_info_file": str(run_info),
        "metrics": [{"generator": "A"}],
        "comparison": {},
        "generation_quality": web_app._default_generation_quality(),
        "output_dir": str(tmp_path),
        "error": "",
    }

    client = web_app.app.test_client()
    resp = client.get("/download_report/job-report")

    assert resp.status_code == 200
    with zipfile.ZipFile(io.BytesIO(resp.data)) as zf:
        assert "results.csv" in zf.namelist()
        assert "run_info.json" in zf.namelist()
        assert "metrics.json" in zf.namelist()
