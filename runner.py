"""Execute generated test scenarios against the target API."""

import json
from typing import Dict, List, Optional
from urllib.parse import urlsplit, urlunsplit

import requests

from config import REQUEST_TIMEOUT


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

    # If the user already entered the full endpoint in base_url, do not append it twice.
    if base_path and (
        normalized_path == base_path
        or normalized_path.startswith(base_path + "/")
        or normalized_path.startswith(base_path + "?")
    ):
        return urlunsplit((parsed.scheme, parsed.netloc, normalized_path, "", ""))

    combined_path = f"{base_path}{normalized_path}" if base_path else normalized_path
    return urlunsplit((parsed.scheme, parsed.netloc, combined_path, "", ""))


def _is_blocked_network_error(exc: Exception) -> bool:
    """Sandbox/OS kaynaklı dış ağ erişim engellerini tespit eder."""
    message = str(exc).lower()
    return (
        "winerror 10013" in message
        or "permission denied" in message
        or "forbidden by its access permissions" in message
    )


def run_testcases(
    base_url: str,
    rows: List[Dict],
    auth_token: Optional[str] = None,
    extra_headers: Optional[Dict] = None,
    cookies: Optional[Dict] = None,
) -> List[Dict]:
    """
    Execute generated scenarios against the real API.

    Returns each row with `url`, `actual_status`, and `pass` fields added.
    """
    print("\n=== Test senaryoları çalıştırılıyor ===")
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
        full_url = _join_url(base_url, row["path"])
        method = row["http_method"].upper()

        body_str = row.get("request_body") or ""
        json_body = None
        if body_str:
            try:
                json_body = json.loads(body_str)
            except json.JSONDecodeError:
                json_body = None

        expected_status = row.get("expected_status")

        if skip_remaining_due_to_network_block:
            status = None
            actual_body = ""
            passed = None
        else:
            try:
                resp = session.request(method, full_url, json=json_body, timeout=REQUEST_TIMEOUT)
                status = resp.status_code
                actual_body = getattr(resp, "text", "") or ""
                passed: object = (status == expected_status) if isinstance(expected_status, int) else None
            except Exception as e:
                status = None
                actual_body = ""
                passed = None
                if _is_blocked_network_error(e):
                    skip_remaining_due_to_network_block = True
                    print(
                        f"  {row['tc_id']} isteği hata verdi: Ortam dış ağa erişime izin vermiyor ({e})."
                    )
                    print("  Kalan testler aynı ağ erişim engeli nedeniyle çalıştırılmadan işaretlenecek.")
                else:
                    print(f"  {row['tc_id']} isteği hata verdi: {e}")

        new_row = dict(row)
        new_row["url"] = full_url
        new_row["actual_status"] = status if status is not None else ""
        new_row["actual_body"] = actual_body
        new_row["pass"] = passed
        executed.append(new_row)

    return executed
