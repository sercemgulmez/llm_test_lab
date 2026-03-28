import requests

from runner import _join_url, run_testcases


class _DummyResponse:
    def __init__(self, status_code):
        self.status_code = status_code


class _DummySession:
    def __init__(self):
        self.headers = {}
        self.cookies = {}
        self.calls = []

    def request(self, method, url, json=None, timeout=None):
        self.calls.append((method, url, json, timeout))
        if url.endswith("/boom"):
            raise requests.RequestException("network error")
        return _DummyResponse(200 if method == "GET" else 201)


def test_join_url_handles_slashes():
    assert _join_url("https://api.example.com", "/users") == "https://api.example.com/users"
    assert _join_url("https://api.example.com/", "/users") == "https://api.example.com/users"
    assert _join_url("https://api.example.com", "users") == "https://api.example.com/users"
    assert _join_url("https://api.example.com/v1", "/users") == "https://api.example.com/v1/users"
    assert _join_url("https://api.example.com/api/web/packages", "/api/web/packages") == (
        "https://api.example.com/api/web/packages"
    )
    assert _join_url("https://api.example.com/api/web/packages", "/api/web/packages?id=42") == (
        "https://api.example.com/api/web/packages?id=42"
    )


def test_run_testcases_executes_requests_and_sets_pass(monkeypatch):
    dummy = _DummySession()
    monkeypatch.setattr("runner.requests.Session", lambda: dummy)

    rows = [
        {
            "tc_id": "TC1",
            "path": "/users",
            "http_method": "GET",
            "request_body": "",
            "expected_status": 200,
        },
        {
            "tc_id": "TC2",
            "path": "/create",
            "http_method": "POST",
            "request_body": '{"name":"Ada"}',
            "expected_status": 201,
        },
        {
            "tc_id": "TC3",
            "path": "/boom",
            "http_method": "GET",
            "request_body": "",
            "expected_status": 200,
        },
    ]

    result = run_testcases(
        "https://api.example.com",
        rows,
        auth_token="secret",
        extra_headers={"X-Env": "test"},
        cookies={"session": "abc"},
    )

    assert dummy.headers["Authorization"] == "Bearer secret"
    assert dummy.headers["X-Env"] == "test"
    assert dummy.cookies["session"] == "abc"
    assert result[0]["pass"] is True
    assert result[1]["actual_status"] == 201
    assert result[2]["actual_status"] == ""
    assert result[2]["pass"] is None
    assert dummy.calls[1][2] == {"name": "Ada"}


def test_run_testcases_stops_after_blocked_network_error(monkeypatch):
    class BlockedSession:
        def __init__(self):
            self.headers = {}
            self.cookies = {}
            self.calls = []

        def request(self, method, url, json=None, timeout=None):
            self.calls.append((method, url, json, timeout))
            raise requests.RequestException(
                "Failed to establish a new connection: [WinError 10013] access permissions"
            )

    dummy = BlockedSession()
    monkeypatch.setattr("runner.requests.Session", lambda: dummy)

    rows = [
        {
            "tc_id": "TC1",
            "path": "/users",
            "http_method": "GET",
            "request_body": "",
            "expected_status": 200,
        },
        {
            "tc_id": "TC2",
            "path": "/users",
            "http_method": "GET",
            "request_body": "",
            "expected_status": 200,
        },
    ]

    result = run_testcases("https://api.example.com", rows)

    assert len(dummy.calls) == 1
    assert result[0]["actual_status"] == ""
    assert result[0]["pass"] is None
    assert result[1]["actual_status"] == ""
    assert result[1]["pass"] is None
