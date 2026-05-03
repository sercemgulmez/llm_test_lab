"""Execute generated test scenarios against the target API."""

import json
import re
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode, urlsplit, urlunsplit

import requests

from config import REQUEST_TIMEOUT


_PATH_PARAM_RE = re.compile(r"\{([^{}]+)\}")


def _join_url(base_url: str, path: str) -> str:
    """Join a base URL and request path without duplicating endpoint paths."""
    parsed = urlsplit(base_url)
    if not parsed.scheme or not parsed.netloc:
        if base_url.endswith("/") and path.startswith("/"):
            return base_url[:-1] + path
        if not base_url.endswith("/") and not path.startswith("/"):
            return base_url + "/" + path
        return base_url + path

    normalized_path = path if path.startswith("/") else f"/{path}"
    base_path = parsed.path.rstrip("/")

    if base_path and (
        normalized_path == base_path
        or normalized_path.startswith(base_path + "/")
        or normalized_path.startswith(base_path + "?")
    ):
        return urlunsplit((parsed.scheme, parsed.netloc, normalized_path, "", ""))

    combined_path = f"{base_path}{normalized_path}" if base_path else normalized_path
    return urlunsplit((parsed.scheme, parsed.netloc, combined_path, "", ""))


def _is_blocked_network_error(exc: Exception) -> bool:
    """Sandbox/OS kaynakli dis ag erisim engellerini tespit eder."""
    message = str(exc).lower()
    return (
        "winerror 10013" in message
        or "permission denied" in message
        or "forbidden by its access permissions" in message
    )


def _session_cookie_dict(session: requests.Session) -> dict:
    if hasattr(session.cookies, "get_dict"):
        return dict(session.cookies.get_dict())
    return dict(session.cookies)


def _resolve_request(row: dict) -> dict:
    request = row.get("request")
    if not isinstance(request, dict):
        request = {}
    return {
        "path_params": request.get("path_params") if isinstance(request.get("path_params"), dict) else {},
        "query_params": request.get("query_params") if isinstance(request.get("query_params"), dict) else {},
        "headers": request.get("headers") if isinstance(request.get("headers"), dict) else {},
        "cookies": request.get("cookies") if isinstance(request.get("cookies"), dict) else {},
        "body": request.get("body"),
    }


def _resolve_expected(row: dict) -> dict:
    expected = row.get("expected")
    if not isinstance(expected, dict):
        expected = {}
    allowed_statuses = expected.get("allowed_statuses")
    if not isinstance(allowed_statuses, list):
        allowed_statuses = []
    return {
        "status": expected.get("status", row.get("expected_status")),
        "allowed_statuses": allowed_statuses,
        "result": expected.get("result", row.get("expected_result", "")),
        "assertions": expected.get("assertions") if isinstance(expected.get("assertions"), list) else [],
        "response_schema_check": bool(expected.get("response_schema_check", False)),
    }


def _render_path(path: str, path_params: dict) -> str:
    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        value = path_params.get(key)
        return str(value) if value is not None else match.group(0)

    return _PATH_PARAM_RE.sub(replace, path)


def _append_query(url: str, query_params: dict) -> str:
    if not query_params:
        return url
    filtered = {key: value for key, value in query_params.items() if value is not None}
    if not filtered:
        return url

    parsed = urlsplit(url)
    query = urlencode(filtered, doseq=True)
    merged_query = "&".join(part for part in [parsed.query, query] if part)
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, merged_query, parsed.fragment))


def _build_url(base_url: str, path: str, request: dict) -> str:
    rendered_path = _render_path(path, request.get("path_params", {}))
    base_joined = _join_url(base_url, rendered_path)
    return _append_query(base_joined, request.get("query_params", {}))


def _build_headers(session: requests.Session, request_headers: dict) -> dict:
    headers = dict(session.headers)
    for key, value in request_headers.items():
        if value is None:
            headers.pop(key, None)
        else:
            headers[key] = value
    return headers


def _build_cookies(session: requests.Session, request_cookies: dict) -> dict:
    cookies = _session_cookie_dict(session)
    for key, value in request_cookies.items():
        if value is None:
            cookies.pop(key, None)
        else:
            cookies[key] = value
    return cookies


def _json_loads_if_possible(value: Any) -> Any:
    if isinstance(value, (dict, list, int, float, bool)) or value is None:
        return value
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def _extract_json_body(row: dict, request: dict, method: str) -> Any:
    if method == "GET":
        return None
    if request.get("body") is not None:
        return request.get("body")
    return _json_loads_if_possible(row.get("request_body") or "")


def _coerce_status_list(values: Any) -> List[int]:
    if not isinstance(values, list):
        return []
    statuses: List[int] = []
    for value in values:
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            statuses.append(value)
        elif isinstance(value, str) and value.isdigit():
            statuses.append(int(value))
    return statuses


def _coerce_status(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _json_path_lookup(document: Any, path: str) -> tuple[bool, Any]:
    if not path.startswith("$."):
        return False, None
    current = document
    for part in path[2:].split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
            continue
        return False, None
    return True, current


def _evaluate_assertions(assertions: list, response: Optional[requests.Response], actual_body: str) -> List[dict]:
    actual_json = _json_loads_if_possible(actual_body)
    content_type = ""
    if response is not None:
        content_type = response.headers.get("Content-Type", "")

    results: List[dict] = []
    for assertion in assertions:
        if not isinstance(assertion, dict):
            results.append({"type": "unknown", "passed": False, "message": "Assertion dict olmali"})
            continue

        assertion_type = assertion.get("type", "unknown")
        passed = False
        message = ""

        if assertion_type == "status_code":
            expected = _coerce_status(assertion.get("expected"))
            actual = response.status_code if response is not None else None
            passed = expected is not None and actual == expected
            message = f"expected={expected} actual={actual}"
        elif assertion_type == "json_path_exists":
            exists, _value = _json_path_lookup(actual_json, str(assertion.get("path", "")))
            passed = exists
            message = f"path={assertion.get('path', '')}"
        elif assertion_type == "json_path_equals":
            exists, value = _json_path_lookup(actual_json, str(assertion.get("path", "")))
            passed = exists and value == assertion.get("expected")
            message = f"path={assertion.get('path', '')} expected={assertion.get('expected')!r} actual={value!r}"
        elif assertion_type == "response_contains":
            expected = str(assertion.get("expected", ""))
            passed = expected in actual_body
            message = f"expected_substring={expected!r}"
        elif assertion_type == "content_type_contains":
            expected = str(assertion.get("expected", ""))
            passed = expected in content_type
            message = f"expected_substring={expected!r} actual={content_type!r}"
        else:
            message = f"Unsupported assertion type: {assertion_type}"

        results.append(
            {
                "type": assertion_type,
                "passed": passed,
                "message": message,
            }
        )

    return results


def _evaluate_schema_check(expected: dict, actual_body: str) -> Optional[dict]:
    if not expected.get("response_schema_check"):
        return None
    parsed = _json_loads_if_possible(actual_body)
    passed = isinstance(parsed, (dict, list))
    return {
        "type": "response_schema_check",
        "passed": passed,
        "message": "Response body JSON parse edilebildi" if passed else "Response body JSON degil",
    }


def _compute_pass(expected: dict, actual_status: Optional[int], assertion_results: list) -> Optional[bool]:
    allowed_statuses = _coerce_status_list(expected.get("allowed_statuses"))
    expected_status = _coerce_status(expected.get("status"))

    status_pass: Optional[bool] = None
    if allowed_statuses:
        status_pass = actual_status in allowed_statuses if actual_status is not None else False
    elif expected_status is not None:
        status_pass = actual_status == expected_status if actual_status is not None else False

    assertion_pass: Optional[bool] = None
    if assertion_results:
        assertion_pass = all(result.get("passed") is True for result in assertion_results)

    decisions = [value for value in [status_pass, assertion_pass] if value is not None]
    if not decisions:
        return None
    return all(decisions)


def run_testcases(
    base_url: str,
    rows: List[Dict],
    auth_token: Optional[str] = None,
    extra_headers: Optional[Dict] = None,
    cookies: Optional[Dict] = None,
) -> List[Dict]:
    """
    Execute generated scenarios against the real API.

    Returns each row with `url`, `actual_status`, `actual_body`, `assertion_results`, and `pass`.
    """
    print("\n=== Test senaryolari calistiriliyor ===")
    session = requests.Session()

    if auth_token:
        session.headers.update({"Authorization": f"Bearer {auth_token}"})

    if extra_headers:
        session.headers.update(extra_headers)

    if cookies:
        session.cookies.update(cookies)

    executed: List[Dict] = []
    skip_remaining_due_to_network_block = False

    for row in rows:
        request = _resolve_request(row)
        expected = _resolve_expected(row)
        full_url = _build_url(base_url, row["path"], request)
        method = row["http_method"].upper()
        request_headers = _build_headers(session, request.get("headers", {}))
        request_cookies = _build_cookies(session, request.get("cookies", {}))
        json_body = _extract_json_body(row, request, method)

        response: Optional[requests.Response] = None
        if skip_remaining_due_to_network_block:
            status = None
            actual_body = ""
            assertion_results: List[dict] = []
            passed = None
        else:
            try:
                request_kwargs: dict = {
                    "headers": request_headers,
                    "cookies": request_cookies,
                    "timeout": REQUEST_TIMEOUT,
                }
                if method != "GET" and json_body is not None:
                    request_kwargs["json"] = json_body
                response = session.request(method, full_url, **request_kwargs)
                status = response.status_code
                actual_body = getattr(response, "text", "") or ""
                assertion_results = _evaluate_assertions(expected.get("assertions", []), response, actual_body)
                schema_check_result = _evaluate_schema_check(expected, actual_body)
                if schema_check_result is not None:
                    assertion_results.append(schema_check_result)
                passed = _compute_pass(expected, status, assertion_results)
            except Exception as exc:
                status = None
                actual_body = ""
                assertion_results = []
                passed = None
                if _is_blocked_network_error(exc):
                    skip_remaining_due_to_network_block = True
                    print(
                        f"  {row['tc_id']} istegi hata verdi: Ortam dis aga erisime izin vermiyor ({exc})."
                    )
                    print("  Kalan testler ayni ag erisim engeli nedeniyle calistirilmadan isaretlenecek.")
                else:
                    print(f"  {row['tc_id']} istegi hata verdi: {exc}")

        new_row = dict(row)
        new_row["url"] = full_url
        new_row["actual_status"] = status if status is not None else ""
        new_row["actual_body"] = actual_body
        new_row["assertion_results"] = assertion_results
        new_row["pass"] = passed
        executed.append(new_row)

    return executed
