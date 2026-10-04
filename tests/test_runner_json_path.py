"""runner._json_path_lookup: "$" koku ve [n] indeksi (ag yok, sahte httpbin govdeleri)."""

import json

import pytest

from runner import _compute_pass, _evaluate_assertions, _json_path_lookup

GET_BODY = {
    "args": {"q": "123"},
    "headers": {"Accept": "*/*", "Host": "httpbin.org"},
    "origin": "203.0.113.1",
    "url": "https://httpbin.org/get?q=123",
}
UUID_BODY = {"uuid": "ae9523ec-409d-4ac1-b9d2-5497e3210bc6"}
ANYTHING_BODY = {
    "args": {},
    "data": '{"message": "smoke"}',
    "files": {},
    "form": {},
    "headers": {"Content-Type": "application/json"},
    "json": {"message": "smoke", "items": [{"id": 1}, {"id": 2}]},
    "method": "POST",
    "url": "https://httpbin.org/anything",
}
LIST_BODY = [{"id": 10}, {"id": 20}]


def _old_json_path_lookup(document, path):
    """Duzeltme oncesi birebir kopya (regresyon karsilastirmasi icin)."""
    if not path.startswith("$."):
        return False, None
    current = document
    for part in path[2:].split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
            continue
        return False, None
    return True, current


@pytest.mark.parametrize(
    "document, path, expected",
    [
        (GET_BODY, "$.args.q", (True, "123")),
        (UUID_BODY, "$.uuid", (True, UUID_BODY["uuid"])),
        (GET_BODY, "$", (True, GET_BODY)),
        (UUID_BODY, "$", (True, UUID_BODY)),
        (ANYTHING_BODY, "$.json.items[1].id", (True, 2)),
        (ANYTHING_BODY, "$.json.items[0]", (True, {"id": 1})),
        (LIST_BODY, "$[0]", (True, {"id": 10})),
        (LIST_BODY, "$[1].id", (True, 20)),
    ],
)
def test_supported_paths_resolve(document, path, expected):
    assert _json_path_lookup(document, path) == expected


@pytest.mark.parametrize(
    "document, path",
    [
        (ANYTHING_BODY, "$.data[0]"),       # data bir string, liste degil
        (GET_BODY, "$.missing"),
        (GET_BODY, "$."),
        (GET_BODY, "$..args"),
        (GET_BODY, "$.args[x]"),
        (GET_BODY, "args.q"),               # $ oneki yok
        (GET_BODY, ""),
        (ANYTHING_BODY, "$.json.items[5]"),  # dizin tasmasi
        (LIST_BODY, "$[2]"),
        (None, "$"),                        # govde JSON degil
        (None, "$.uuid"),
    ],
)
def test_unsupported_or_missing_paths_return_false(document, path):
    assert _json_path_lookup(document, path) == (False, None)


@pytest.mark.parametrize(
    "document, path",
    [
        (GET_BODY, "$.args.q"),
        (GET_BODY, "$.args"),
        (GET_BODY, "$.headers.Host"),
        (GET_BODY, "$.headers.NonExistentHeader"),
        (UUID_BODY, "$.uuid"),
        (ANYTHING_BODY, "$.json.message"),
        (ANYTHING_BODY, "$.json.items"),
        (ANYTHING_BODY, "$.form"),
        ({"json": {}}, "$.json.message"),
        (GET_BODY, "$.a.b.c"),
        (GET_BODY, "$.args."),
        (None, "$.uuid"),
    ],
)
def test_previously_supported_paths_unchanged(document, path):
    assert _json_path_lookup(document, path) == _old_json_path_lookup(document, path)


class _FakeResponse:
    def __init__(self, status_code, body):
        self.status_code = status_code
        self.headers = {"Content-Type": "application/json"}
        self.text = json.dumps(body)


def test_root_exists_assertion_passes_end_to_end():
    response = _FakeResponse(200, UUID_BODY)
    results = _evaluate_assertions(
        [
            {"type": "status_code", "expected": 200},
            {"type": "json_path_exists", "path": "$"},
            {"type": "json_path_equals", "path": "$.uuid", "expected": UUID_BODY["uuid"]},
        ],
        response,
        response.text,
    )
    assert [r["passed"] for r in results] == [True, True, True]
    assert _compute_pass({"status": 200, "allowed_statuses": [200]}, 200, results) is True


def test_root_exists_assertion_fails_for_non_json_body():
    results = _evaluate_assertions([{"type": "json_path_exists", "path": "$"}], None, "<html>not json</html>")
    assert results[0]["passed"] is False


def test_equals_comparison_semantics_unchanged():
    response = _FakeResponse(200, GET_BODY)
    results = _evaluate_assertions(
        [{"type": "json_path_equals", "path": "$.args.q", "expected": "wrong"}], response, response.text
    )
    assert results[0]["passed"] is False
    assert "actual='123'" in results[0]["message"]
