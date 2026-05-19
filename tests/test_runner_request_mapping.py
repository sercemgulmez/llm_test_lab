from runner import run_testcases


class _DummyResponse:
    def __init__(self, status_code, text="", headers=None):
        self.status_code = status_code
        self.text = text
        self.headers = headers or {"Content-Type": "application/json"}


class _DummySession:
    def __init__(self, response=None):
        self.headers = {}
        self.cookies = {}
        self.calls = []
        self._response = response or _DummyResponse(200, text='{"id": 123, "data": {"id": 456}, "items": []}')

    def request(self, method, url, json=None, timeout=None, headers=None, cookies=None):
        self.calls.append(
            {
                "method": method,
                "url": url,
                "json": json,
                "timeout": timeout,
                "headers": headers or {},
                "cookies": cookies or {},
            }
        )
        return self._response


def test_runner_substitutes_path_params(monkeypatch):
    dummy = _DummySession()
    monkeypatch.setattr("runner.requests.Session", lambda: dummy)

    rows = [
        {
            "tc_id": "TC1",
            "path": "/users/{id}",
            "http_method": "GET",
            "request": {"path_params": {"id": 123}},
            "expected": {"status": 200},
        }
    ]

    result = run_testcases("https://api.example.com", rows)

    assert result[0]["url"] == "https://api.example.com/users/123"


def test_runner_adds_query_params_to_url(monkeypatch):
    dummy = _DummySession()
    monkeypatch.setattr("runner.requests.Session", lambda: dummy)

    rows = [
        {
            "tc_id": "TC1",
            "path": "/users",
            "http_method": "GET",
            "request": {"query_params": {"page": 2, "size": 10}},
            "expected": {"status": 200},
        }
    ]

    result = run_testcases("https://api.example.com", rows)

    assert result[0]["url"] == "https://api.example.com/users?page=2&size=10"


def test_runner_merges_testcase_specific_headers(monkeypatch):
    dummy = _DummySession()
    monkeypatch.setattr("runner.requests.Session", lambda: dummy)

    rows = [
        {
            "tc_id": "TC1",
            "path": "/users",
            "http_method": "GET",
            "request": {"headers": {"X-Test": "case-value"}},
            "expected": {"status": 200},
        }
    ]

    run_testcases("https://api.example.com", rows, extra_headers={"X-Env": "global"})

    assert dummy.calls[0]["headers"]["X-Env"] == "global"
    assert dummy.calls[0]["headers"]["X-Test"] == "case-value"


def test_runner_removes_authorization_header_when_null(monkeypatch):
    dummy = _DummySession()
    monkeypatch.setattr("runner.requests.Session", lambda: dummy)

    rows = [
        {
            "tc_id": "TC1",
            "path": "/users",
            "http_method": "GET",
            "request": {"headers": {"Authorization": None}},
            "expected": {"status": 200},
        }
    ]

    run_testcases("https://api.example.com", rows, auth_token="secret")

    assert "Authorization" not in dummy.calls[0]["headers"]


def test_runner_uses_nested_request_body(monkeypatch):
    dummy = _DummySession(response=_DummyResponse(201))
    monkeypatch.setattr("runner.requests.Session", lambda: dummy)

    rows = [
        {
            "tc_id": "TC1",
            "path": "/users",
            "http_method": "POST",
            "request": {"body": {"name": "Ada"}},
            "expected": {"status": 201},
        }
    ]

    run_testcases("https://api.example.com", rows)

    assert dummy.calls[0]["json"] == {"name": "Ada"}


def test_runner_supports_old_request_body(monkeypatch):
    dummy = _DummySession(response=_DummyResponse(201))
    monkeypatch.setattr("runner.requests.Session", lambda: dummy)

    rows = [
        {
            "tc_id": "TC1",
            "path": "/users",
            "http_method": "POST",
            "request_body": '{"name":"Ada"}',
            "expected_status": 201,
        }
    ]

    run_testcases("https://api.example.com", rows)

    assert dummy.calls[0]["json"] == {"name": "Ada"}


def test_runner_passes_with_allowed_statuses(monkeypatch):
    dummy = _DummySession(response=_DummyResponse(202))
    monkeypatch.setattr("runner.requests.Session", lambda: dummy)

    rows = [
        {
            "tc_id": "TC1",
            "path": "/jobs",
            "http_method": "POST",
            "request": {"body": {"name": "Ada"}},
            "expected": {"allowed_statuses": [200, 202]},
        }
    ]

    result = run_testcases("https://api.example.com", rows)

    assert result[0]["pass"] is True


def test_runner_supports_json_path_exists_assertion(monkeypatch):
    dummy = _DummySession(response=_DummyResponse(200, text='{"id": 123, "items": []}'))
    monkeypatch.setattr("runner.requests.Session", lambda: dummy)

    rows = [
        {
            "tc_id": "TC1",
            "path": "/users",
            "http_method": "GET",
            "expected": {
                "status": 200,
                "assertions": [{"type": "json_path_exists", "path": "$.items"}],
            },
        }
    ]

    result = run_testcases("https://api.example.com", rows)

    assert result[0]["assertion_results"][0]["passed"] is True
    assert result[0]["pass"] is True


def test_runner_supports_json_path_equals_assertion(monkeypatch):
    dummy = _DummySession(response=_DummyResponse(200, text='{"data": {"id": 456}}'))
    monkeypatch.setattr("runner.requests.Session", lambda: dummy)

    rows = [
        {
            "tc_id": "TC1",
            "path": "/users",
            "http_method": "GET",
            "expected": {
                "status": 200,
                "assertions": [{"type": "json_path_equals", "path": "$.data.id", "expected": 456}],
            },
        }
    ]

    result = run_testcases("https://api.example.com", rows)

    assert result[0]["assertion_results"][0]["passed"] is True
    assert result[0]["pass"] is True


def test_runner_supports_response_contains_assertion(monkeypatch):
    dummy = _DummySession(response=_DummyResponse(200, text="operation succeeded"))
    monkeypatch.setattr("runner.requests.Session", lambda: dummy)

    rows = [
        {
            "tc_id": "TC1",
            "path": "/users",
            "http_method": "GET",
            "expected": {
                "status": 200,
                "assertions": [{"type": "response_contains", "expected": "succeeded"}],
            },
        }
    ]

    result = run_testcases("https://api.example.com", rows)

    assert result[0]["assertion_results"][0]["passed"] is True
    assert result[0]["pass"] is True
