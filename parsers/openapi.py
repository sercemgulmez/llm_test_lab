"""OpenAPI / Swagger document loaders and parsers."""

import json
from typing import Dict, List, Optional

import requests
import yaml

from models import ApiOperation


def _looks_like_html(text: str, content_type: str) -> bool:
    lowered = text.lstrip().lower()
    return (
        "text/html" in content_type.lower()
        or lowered.startswith("<!doctype html")
        or lowered.startswith("<html")
    )


def _parse_openapi_text(text: str) -> Dict:
    try:
        spec = json.loads(text)
    except json.JSONDecodeError:
        try:
            spec = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ValueError("OpenAPI dokumani JSON veya YAML olarak parse edilemedi.") from exc

    if not isinstance(spec, dict):
        raise ValueError("OpenAPI dokumani nesne olarak parse edilemedi.")

    if "paths" not in spec or not isinstance(spec["paths"], dict):
        raise ValueError(
            "OpenAPI dokumani yuklendi ama 'paths' bulunamadi. "
            "Swagger UI sayfasi veya farkli bir JSON/YAML URL'i vermis olabilirsiniz."
        )

    return spec


def load_openapi_from_url(
    url: str,
    headers: Optional[Dict[str, str]] = None,
    cookies: Optional[Dict[str, str]] = None,
) -> Dict:
    """Load an OpenAPI / Swagger JSON or YAML document from a URL."""
    print(f"OpenAPI/Swagger yukleniyor: {url}")
    request_headers = {"Accept": "application/json, application/yaml, text/yaml, */*"}
    if headers:
        request_headers.update(headers)
    resp = requests.get(
        url,
        timeout=15,
        headers=request_headers,
        cookies=cookies,
    )
    resp.raise_for_status()

    text = resp.text
    content_type = resp.headers.get("Content-Type", "")
    if _looks_like_html(text, content_type):
        raise ValueError(
            "Swagger UI HTML sayfasi geldi. Raw OpenAPI JSON/YAML URL'i verin "
            "(ornegin /swagger.json, /openapi.json veya .yaml)."
        )

    return _parse_openapi_text(text)


# ── OpenAPI field extraction helpers ──────────────────────────────────────────

def _extract_request_body_schema(op: dict) -> Optional[dict]:
    for ct_data in op.get("requestBody", {}).get("content", {}).values():
        schema = ct_data.get("schema")
        if schema:
            return schema
    return None


def _extract_request_body_examples(op: dict) -> list[dict]:
    examples: list[dict] = []
    for ct_data in op.get("requestBody", {}).get("content", {}).values():
        if "example" in ct_data and isinstance(ct_data["example"], dict):
            examples.append(ct_data["example"])
        for ex_val in ct_data.get("examples", {}).values():
            if isinstance(ex_val, dict) and isinstance(ex_val.get("value"), dict):
                examples.append(ex_val["value"])
    return examples


def _extract_response_schemas(op: dict) -> dict:
    result: dict = {}
    for status_code, resp in op.get("responses", {}).items():
        key = str(status_code)
        content = resp.get("content", {})
        for ct, ct_data in content.items():
            schema = ct_data.get("schema")
            if schema:
                result[key] = {"content_type": ct, "schema": schema}
                break
        if key not in result:
            result[key] = {"description": resp.get("description", "")}
    return result


def _extract_response_examples(op: dict) -> dict:
    result: dict = {}
    for status_code, resp in op.get("responses", {}).items():
        key = str(status_code)
        for ct_data in resp.get("content", {}).values():
            if "example" in ct_data:
                result[key] = ct_data["example"]
                break
            for ex_val in ct_data.get("examples", {}).values():
                if isinstance(ex_val, dict) and "value" in ex_val:
                    result[key] = ex_val["value"]
                    break
            if key in result:
                break
    return result


def _extract_content_types(op: dict) -> list[str]:
    return list(op.get("requestBody", {}).get("content", {}).keys())


def _extract_servers(spec: dict) -> list[str]:
    return [s["url"] for s in spec.get("servers", []) if isinstance(s, dict) and s.get("url")]


# ── Main extractor ─────────────────────────────────────────────────────────────

def extract_operations_from_openapi(spec: Dict) -> List[ApiOperation]:
    """Extract HTTP operations from an OpenAPI document."""
    valid_methods = {"get", "post", "put", "delete", "patch", "head", "options"}
    ops: List[ApiOperation] = []
    paths = spec.get("paths", {})
    servers = _extract_servers(spec)
    counter = 1

    for path, methods in paths.items():
        if not isinstance(methods, dict):
            continue
        # Path-level parameters shared across all methods
        path_level_params: list[dict] = methods.get("parameters") or []

        for method, op in methods.items():
            if method.lower() not in valid_methods:
                continue
            if not isinstance(op, dict):
                continue

            op_id = op.get("operationId") or f"OP{counter}"
            # Merge path-level and operation-level parameters
            op_params: list[dict] = path_level_params + (op.get("parameters") or [])

            ops.append(
                ApiOperation(
                    op_id=op_id,
                    method=method.upper(),
                    path=path,
                    summary=op.get("summary") or "",
                    description=op.get("description") or "",
                    tags=op.get("tags") or [],
                    parameters=op_params,
                    request_body_schema=_extract_request_body_schema(op),
                    request_body_examples=_extract_request_body_examples(op),
                    response_schemas=_extract_response_schemas(op),
                    response_examples=_extract_response_examples(op),
                    security=op.get("security") or [],
                    content_types=_extract_content_types(op),
                    servers=servers,
                    raw_operation=op,
                )
            )
            counter += 1

    return ops


def manual_operations_input() -> List[ApiOperation]:
    """Collect API operations from stdin."""
    ops: List[ApiOperation] = []
    print("\nManuel endpoint girişi. Bitirmek için method kısmını boş bırakın.\n")
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
