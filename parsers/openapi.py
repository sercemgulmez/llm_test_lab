"""OpenAPI / Swagger dokümanlarını parse eder."""

import json
from typing import Dict, List

import requests
import yaml

from models import ApiOperation


def load_openapi_from_url(url: str) -> Dict:
    """Verilen URL'den OpenAPI / Swagger JSON veya YAML dokümanı yükler."""
    print(f"OpenAPI/Swagger yükleniyor: {url}")
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    text = resp.text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return yaml.safe_load(text)


def extract_operations_from_openapi(spec: Dict) -> List[ApiOperation]:
    """OpenAPI dokümanından HTTP operasyonlarını çıkarır."""
    VALID_METHODS = {"get", "post", "put", "delete", "patch", "head", "options"}
    ops: List[ApiOperation] = []
    paths = spec.get("paths", {})
    counter = 1

    for path, methods in paths.items():
        if not isinstance(methods, dict):
            continue
        for method, op in methods.items():
            if method.lower() not in VALID_METHODS:
                continue
            if not isinstance(op, dict):
                continue
            op_id = op.get("operationId") or f"OP{counter}"
            ops.append(
                ApiOperation(
                    op_id=op_id,
                    method=method.upper(),
                    path=path,
                    summary=op.get("summary") or "",
                    description=op.get("description") or "",
                )
            )
            counter += 1

    return ops


def manual_operations_input() -> List[ApiOperation]:
    """API operasyonlarını konsoldan manuel olarak alır."""
    ops: List[ApiOperation] = []
    print("\nManuel endpoint girişi. Bitirmek için method kısmını boş bırak.\n")
    idx = 1
    while True:
        method = input(
            f"[{idx}] HTTP method (GET/POST/PUT/DELETE, boş = bitir): "
        ).strip().upper()
        if not method:
            break
        path = input(f"[{idx}] Path (örn: /users, /login): ").strip()
        summary = input(f"[{idx}] Kısa özet: ").strip()
        description = input(f"[{idx}] Detay açıklama (opsiyonel): ").strip()
        ops.append(
            ApiOperation(
                op_id=f"MAN{idx}",
                method=method,
                path=path,
                summary=summary,
                description=description,
            )
        )
        idx += 1
    return ops
