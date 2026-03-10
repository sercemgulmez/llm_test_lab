import csv
import io
import json
import queue

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


def test_result_metrics_and_comparison_endpoints_return_job_data(tmp_path):
    result_file = tmp_path / "results.csv"
    with result_file.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["generator", "pass"])
        writer.writerow(["A", "True"])

    web_app._jobs["job-test"] = {
        "status": "done",
        "log_queue": None,
        "result_file": str(result_file),
        "metrics": [{"generator": "A", "pass_rate": 1.0}],
        "comparison": {"overview": {"best_generator": "A"}},
        "output_dir": str(tmp_path),
    }

    client = web_app.app.test_client()
    result_resp = client.get("/result_data/job-test")
    metrics_resp = client.get("/metrics/job-test")
    comparison_resp = client.get("/comparison/job-test")

    assert result_resp.status_code == 200
    assert result_resp.get_json()["headers"] == ["generator", "pass"]
    assert metrics_resp.get_json()[0]["generator"] == "A"
    assert comparison_resp.get_json()["overview"]["best_generator"] == "A"

    web_app._jobs.pop("job-test", None)


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
    assert (tmp_path / payload["path"].split("\\")[-1]).exists()


def test_run_job_executes_with_synchronous_thread(monkeypatch, tmp_path):
    result_file = tmp_path / "result.csv"
    result_file.write_text("generator,pass\nA,True\n", encoding="utf-8")

    def fake_execute_pipeline(data):
        assert data["source"] == "openapi"
        return {
            "result_file": str(result_file),
            "metrics": [{"generator": "A", "pass_rate": 1.0}],
            "comparison": {"overview": {"best_generator": "A"}},
            "output_dir": str(tmp_path),
        }

    class ImmediateThread:
        def __init__(self, target, args=(), daemon=None):
            self._target = target
            self._args = args

        def start(self):
            self._target(*self._args)

    class DummyUuid:
        hex = "job12345deadbeef"

    monkeypatch.setattr(web_app, "_execute_pipeline", fake_execute_pipeline)
    monkeypatch.setattr(web_app.threading, "Thread", ImmediateThread)
    monkeypatch.setattr(web_app.uuid, "uuid4", lambda: DummyUuid())

    client = web_app.app.test_client()
    resp = client.post("/run", json={"source": "openapi", "output_dir": str(tmp_path)})

    payload = resp.get_json()
    job_id = payload["job_id"]
    assert resp.status_code == 200
    assert job_id == "job12345"
    assert web_app._jobs[job_id]["status"] == "done"
    assert web_app._jobs[job_id]["comparison"]["overview"]["best_generator"] == "A"


def test_stream_endpoint_emits_log_and_done_messages():
    job_id = "job-stream"
    q = queue.Queue()
    q.put({"type": "log", "text": "step 1"})
    q.put(None)
    web_app._jobs[job_id] = {
        "status": "done",
        "log_queue": q,
        "result_file": "out.csv",
        "metrics": [],
        "comparison": {},
        "output_dir": "outputs",
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
        "log_queue": queue.Queue(),
        "result_file": str(result_file),
        "metrics": [],
        "comparison": {},
        "output_dir": str(tmp_path),
    }

    client = web_app.app.test_client()
    resp = client.get("/download/job-download")

    assert resp.status_code == 200
    assert "attachment" in resp.headers["Content-Disposition"]
    assert b"generator,pass" in resp.data
