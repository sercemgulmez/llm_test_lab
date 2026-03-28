import pytest

from generators.base import build_llm_prompt, parse_llm_lines_to_rows
from generators.traditional import TraditionalGenerator
from models import ApiOperation
from parsers.openapi import extract_operations_from_openapi, load_openapi_from_url


class _DummyResponse:
    def __init__(self, text, content_type="application/json"):
        self.text = text
        self.headers = {"Content-Type": content_type}

    def raise_for_status(self):
        return None


def test_extract_operations_from_openapi_filters_valid_methods():
    spec = {
        "paths": {
            "/users": {
                "get": {"operationId": "listUsers", "summary": "List users"},
                "trace": {"operationId": "traceUsers"},
                "post": {"summary": "Create user", "description": "Creates a user"},
            }
        }
    }

    ops = extract_operations_from_openapi(spec)

    assert len(ops) == 2
    assert ops[0].op_id == "listUsers"
    assert ops[1].op_id == "OP2"
    assert ops[1].method == "POST"


def test_load_openapi_from_url_rejects_swagger_ui_html(monkeypatch):
    def fake_get(*args, **kwargs):
        return _DummyResponse("<!doctype html><html><body>Swagger UI</body></html>", "text/html")

    monkeypatch.setattr("parsers.openapi.requests.get", fake_get)

    with pytest.raises(ValueError, match="Swagger UI HTML sayfasi geldi"):
        load_openapi_from_url("https://example.com/swagger")


def test_load_openapi_from_url_rejects_non_spec_json(monkeypatch):
    def fake_get(*args, **kwargs):
        return _DummyResponse('{"urls":[{"url":"/openapi.json"}]}')

    monkeypatch.setattr("parsers.openapi.requests.get", fake_get)

    with pytest.raises(ValueError, match="'paths' bulunamadi"):
        load_openapi_from_url("https://example.com/swagger-config")


def test_load_openapi_from_url_forwards_headers_and_cookies(monkeypatch):
    called = {}

    def fake_get(url, **kwargs):
        called["url"] = url
        called["kwargs"] = kwargs
        return _DummyResponse('{"openapi":"3.0.0","paths":{}}')

    monkeypatch.setattr("parsers.openapi.requests.get", fake_get)

    spec = load_openapi_from_url(
        "https://example.com/openapi.json",
        headers={"Authorization": "Bearer token"},
        cookies={"sessionid": "abc"},
    )

    assert spec["paths"] == {}
    assert called["url"] == "https://example.com/openapi.json"
    assert called["kwargs"]["cookies"] == {"sessionid": "abc"}
    assert called["kwargs"]["headers"]["Authorization"] == "Bearer token"
    assert "application/json" in called["kwargs"]["headers"]["Accept"]


def test_build_llm_prompt_and_parse_lines_round_trip():
    op = ApiOperation(
        op_id="LOGIN",
        method="POST",
        path="/login",
        summary="Login",
        description="Authenticate user",
        example_body='{"phone":"+90555","password":"secret"}',
    )

    prompt = build_llm_prompt(op, num_cases=3, variant_name="basic", variant_desc="happy path")
    lines = [
        'LOGIN_TC1|Valid login|POST /login|{"phone":"+90555","password":"secret"}|200|Token doner',
        'LOGIN_TC2|Invalid login|POST /login|{"phone":"+90555","password":"bad"}|401|Unauthorized',
        "invalid line",
    ]
    rows = parse_llm_lines_to_rows(lines, op, "LLM-X")

    assert "TAM OLARAK 3 SATIR" in prompt
    assert len(rows) == 2
    assert rows[0]["generator"] == "LLM-X"
    assert rows[0]["expected_status"] == 200
    assert rows[1]["request_body"] == '{"phone": "+90555", "password": "bad"}'


def test_traditional_generator_produces_expected_templates():
    op = ApiOperation(
        op_id="USER_GET",
        method="GET",
        path="/users/1",
        summary="Get user",
        description="Fetch one user",
    )

    rows = TraditionalGenerator().generate([op])

    assert len(rows) == 5
    assert rows[0]["expected_status"] == 200
    assert rows[1]["expected_status"] == 400
    assert rows[-1]["expected_status"] == 500
    assert all(row["operation_id"] == "USER_GET" for row in rows)
