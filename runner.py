"""Test senaryolarını gerçek API'ye karşı yürütür."""

import json
from typing import Dict, List, Optional

import requests

from config import REQUEST_TIMEOUT


def _join_url(base_url: str, path: str) -> str:
    """Base URL ile path'i düzgün şekilde birleştirir."""
    if base_url.endswith("/") and path.startswith("/"):
        return base_url[:-1] + path
    if not base_url.endswith("/") and not path.startswith("/"):
        return base_url + "/" + path
    return base_url + path


def run_testcases(
    base_url: str,
    rows: List[Dict],
    auth_token: Optional[str] = None,
) -> List[Dict]:
    """
    Üretilen tüm test senaryolarını gerçek API'ye karşı uygular.

    Args:
        base_url:    Hedef API'nin base URL'i (örn: https://api.example.com/v1)
        rows:        generate_* fonksiyonlarından gelen test satırları
        auth_token:  Varsa Bearer token (Authorization: Bearer <token>)

    Returns:
        Her satıra 'url', 'actual_status', 'pass' alanları eklenmiş liste
    """
    print("\n=== Test senaryoları çalıştırılıyor ===")
    session = requests.Session()

    if auth_token:
        session.headers.update({"Authorization": f"Bearer {auth_token}"})

    executed: List[Dict] = []

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

        try:
            resp = session.request(method, full_url, json=json_body, timeout=REQUEST_TIMEOUT)
            status = resp.status_code
            passed: object = (status == expected_status) if isinstance(expected_status, int) else None
        except Exception as e:
            status = None
            passed = None
            print(f"  {row['tc_id']} isteği hata verdi: {e}")

        new_row = dict(row)
        new_row["url"] = full_url
        new_row["actual_status"] = status if status is not None else ""
        new_row["pass"] = passed
        executed.append(new_row)

    return executed
