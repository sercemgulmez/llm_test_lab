"""OpenAPI / Swagger document loaders and parsers."""

import json
from typing import Any, Dict, List, Optional, Tuple

import requests
import yaml

from models import ApiOperation


_VALID_METHODS = {"get", "post", "put", "delete", "patch", "head", "options"}
_SWAGGER_SCHEMA_KEYS = (
    "type",
    "format",
    "items",
    "default",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "minLength",
    "maxLength",
    "pattern",
    "minItems",
    "maxItems",
    "uniqueItems",
    "enum",
)


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


def _is_swagger2(spec: dict) -> bool:
    return str(spec.get("swagger", "")).startswith("2.")


def _dedupe_preserve_order(values: List[str]) -> List[str]:
    seen: set[str] = set()
    deduped: List[str] = []
    for value in values:
        if value and value not in seen:
            deduped.append(value)
            seen.add(value)
    return deduped


def _resolve_pointer(spec: dict, ref: str) -> Any:
    if not ref.startswith("#/"):
        return {"$ref": ref}

    node: Any = spec
    for part in ref[2:].split("/"):
        key = part.replace("~1", "/").replace("~0", "~")
        if not isinstance(node, dict) or key not in node:
            return {"$ref": ref}
        node = node[key]
    return node


def _deep_resolve(value: Any, spec: dict, visited: Optional[set[str]] = None) -> Any:
    visited = visited or set()

    if isinstance(value, list):
        return [_deep_resolve(item, spec, visited.copy()) for item in value]

    if not isinstance(value, dict):
        return value

    if "$ref" in value:
        ref = value["$ref"]
        if ref in visited:
            return {"$ref": ref}

        resolved_target = _deep_resolve(_resolve_pointer(spec, ref), spec, visited | {ref})
        siblings = {
            key: _deep_resolve(val, spec, visited | {ref})
            for key, val in value.items()
            if key != "$ref"
        }
        if isinstance(resolved_target, dict):
            merged = dict(resolved_target)
            merged.update(siblings)
            return merged
        return siblings or resolved_target

    return {key: _deep_resolve(val, spec, visited.copy()) for key, val in value.items()}


def _select_preferred_content(content: dict) -> Tuple[str, dict]:
    if not isinstance(content, dict) or not content:
        return "", {}
    if "application/json" in content:
        return "application/json", content["application/json"]
    for content_type, media_type in content.items():
        if "json" in content_type:
            return content_type, media_type
    first_content_type = next(iter(content))
    return first_content_type, content[first_content_type]


def _extract_examples_from_media_type(media_type: dict) -> List[Any]:
    examples: List[Any] = []
    if "example" in media_type:
        examples.append(media_type["example"])
    for example in (media_type.get("examples") or {}).values():
        if isinstance(example, dict) and "value" in example:
            examples.append(example["value"])
        elif example is not None:
            examples.append(example)
    return examples


def _normalize_examples_to_dicts(examples: List[Any]) -> List[dict]:
    normalized: List[dict] = []
    for example in examples:
        if isinstance(example, dict):
            normalized.append(example)
        elif example is not None:
            normalized.append({"value": example})
    return normalized


def _normalize_parameter(param: dict) -> dict:
    schema = param.get("schema")
    if not schema and isinstance(param.get("content"), dict):
        _content_type, media_type = _select_preferred_content(param["content"])
        schema = media_type.get("schema") if isinstance(media_type, dict) else None
    if not schema:
        swagger_schema = {key: param[key] for key in _SWAGGER_SCHEMA_KEYS if key in param}
        schema = swagger_schema or {}

    example = param.get("example")
    if example is None and "x-example" in param:
        example = param.get("x-example")
    if example is None and isinstance(param.get("content"), dict):
        _content_type, media_type = _select_preferred_content(param["content"])
        media_examples = _extract_examples_from_media_type(media_type if isinstance(media_type, dict) else {})
        if media_examples:
            example = media_examples[0]
    if example is None and isinstance(schema, dict):
        example = schema.get("example")

    enum = param.get("enum")
    if enum is None and isinstance(schema, dict):
        enum = schema.get("enum")

    return {
        "name": param.get("name", ""),
        "in": param.get("in", ""),
        "required": bool(param.get("required", False) or param.get("in") == "path"),
        "schema": schema if isinstance(schema, dict) else {},
        "example": example,
        "description": param.get("description", "") or "",
        "enum": enum if isinstance(enum, list) else [],
    }


def _merge_parameters(path_params: list[dict], op_params: list[dict], spec: dict) -> Tuple[List[dict], List[dict]]:
    merged_raw: List[dict] = []
    merged_normalized: List[dict] = []
    index_map: dict[Tuple[str, str], int] = {}

    for raw_param in [*(path_params or []), *(op_params or [])]:
        resolved = _deep_resolve(raw_param, spec)
        if not isinstance(resolved, dict):
            continue

        key = (resolved.get("name", ""), resolved.get("in", ""))
        if not all(key):
            key = (f"__anon__{len(merged_raw)}", resolved.get("in", ""))

        normalized = _normalize_parameter(resolved)
        existing_index = index_map.get(key)
        if existing_index is None:
            index_map[key] = len(merged_raw)
            merged_raw.append(resolved)
            merged_normalized.append(normalized)
        else:
            merged_raw[existing_index] = resolved
            merged_normalized[existing_index] = normalized

    return merged_raw, merged_normalized


def _extract_request_body_openapi3(op: dict, spec: dict) -> Tuple[Optional[dict], List[dict], List[str], dict]:
    request_body = _deep_resolve(op.get("requestBody"), spec)
    if not isinstance(request_body, dict):
        return None, [], [], {}

    content = request_body.get("content") or {}
    content_types = list(content.keys()) if isinstance(content, dict) else []
    selected_content_type, media_type = _select_preferred_content(content)
    schema = _deep_resolve(media_type.get("schema"), spec) if isinstance(media_type, dict) else None

    if isinstance(schema, dict):
        schema = dict(schema)
        schema["x-request-body-required"] = bool(request_body.get("required", False))
        if selected_content_type:
            schema["x-content-type"] = selected_content_type

    all_examples: List[Any] = []
    for media in content.values():
        if isinstance(media, dict):
            all_examples.extend(_extract_examples_from_media_type(media))

    return schema, _normalize_examples_to_dicts(all_examples), content_types, request_body


def _extract_request_body_swagger2(
    merged_raw_parameters: List[dict],
    op: dict,
    spec: dict,
) -> Tuple[Optional[dict], List[dict], List[str]]:
    body_param = next(
        (param for param in merged_raw_parameters if param.get("in") == "body"),
        None,
    )
    if not isinstance(body_param, dict):
        return None, [], _extract_swagger_consumes(op, spec)

    schema = _deep_resolve(body_param.get("schema"), spec)
    if isinstance(schema, dict):
        schema = dict(schema)
        schema["x-request-body-required"] = bool(body_param.get("required", False))

    examples: List[Any] = []
    if isinstance(schema, dict) and "example" in schema:
        examples.append(schema["example"])
    if "x-example" in body_param:
        examples.append(body_param["x-example"])

    return schema, _normalize_examples_to_dicts(examples), _extract_swagger_consumes(op, spec)


def _extract_swagger_consumes(op: dict, spec: dict) -> List[str]:
    consumes = op.get("consumes") or spec.get("consumes") or []
    return [content_type for content_type in consumes if isinstance(content_type, str)]


def _extract_swagger_produces(op: dict, spec: dict) -> List[str]:
    produces = op.get("produces") or spec.get("produces") or []
    return [content_type for content_type in produces if isinstance(content_type, str)]


def _extract_responses_openapi3(op: dict, spec: dict) -> Tuple[dict, dict]:
    response_schemas: dict = {}
    response_examples: dict = {}

    for status_code, response in (op.get("responses") or {}).items():
        resolved_response = _deep_resolve(response, spec)
        if not isinstance(resolved_response, dict):
            continue

        status_key = str(status_code)
        entry = {
            "description": resolved_response.get("description", "") or "",
            "content": {},
        }
        entry_examples: dict = {}

        for content_type, media_type in (resolved_response.get("content") or {}).items():
            if not isinstance(media_type, dict):
                continue
            resolved_media = _deep_resolve(media_type, spec)
            content_entry: dict = {}
            if "schema" in resolved_media:
                content_entry["schema"] = resolved_media.get("schema")
            media_examples = _extract_examples_from_media_type(resolved_media)
            if media_examples:
                content_entry["examples"] = media_examples
                entry_examples[content_type] = media_examples[0] if len(media_examples) == 1 else media_examples
            entry["content"][content_type] = content_entry

        response_schemas[status_key] = entry
        if entry_examples:
            response_examples[status_key] = entry_examples

    return response_schemas, response_examples


def _extract_responses_swagger2(op: dict, spec: dict) -> Tuple[dict, dict]:
    response_schemas: dict = {}
    response_examples: dict = {}
    produces = _extract_swagger_produces(op, spec)

    for status_code, response in (op.get("responses") or {}).items():
        resolved_response = _deep_resolve(response, spec)
        if not isinstance(resolved_response, dict):
            continue

        status_key = str(status_code)
        entry = {
            "description": resolved_response.get("description", "") or "",
            "content": {},
        }
        entry_examples: dict = {}
        response_schema = resolved_response.get("schema")
        response_content_types = produces or list((resolved_response.get("examples") or {}).keys()) or ["application/json"]

        if response_schema is not None:
            resolved_schema = _deep_resolve(response_schema, spec)
            for content_type in response_content_types:
                entry["content"][content_type] = {"schema": resolved_schema}

        for content_type, example in (resolved_response.get("examples") or {}).items():
            entry["content"].setdefault(content_type, {})
            entry["content"][content_type]["examples"] = [example]
            entry_examples[content_type] = example

        response_schemas[status_key] = entry
        if entry_examples:
            response_examples[status_key] = entry_examples

    return response_schemas, response_examples


def _extract_servers(spec: dict, path_item: dict, op: dict) -> List[str]:
    if _is_swagger2(spec):
        host = spec.get("host", "")
        base_path = spec.get("basePath", "")
        schemes = op.get("schemes") or spec.get("schemes") or []
        if host:
            return [f"{scheme}://{host}{base_path}" for scheme in schemes] or [f"https://{host}{base_path}"]
        return []

    urls: List[str] = []
    for container in (op, path_item, spec):
        for server in container.get("servers", []) if isinstance(container, dict) else []:
            if isinstance(server, dict) and server.get("url"):
                urls.append(server["url"])
    return _dedupe_preserve_order(urls)


def _effective_security(spec: dict, op: dict) -> List[dict]:
    if "security" in op and isinstance(op.get("security"), list):
        return op.get("security") or []
    if isinstance(spec.get("security"), list):
        return spec.get("security") or []
    return []


def _build_raw_operation(
    resolved_op: dict,
    merged_raw_parameters: List[dict],
    request_body: Optional[dict],
    response_schemas: dict,
    security: List[dict],
    tags: List[str],
    servers: List[str],
) -> dict:
    raw_operation = dict(resolved_op)
    raw_operation["parameters"] = merged_raw_parameters
    raw_operation["responses"] = raw_operation.get("responses") or {}
    if request_body is not None:
        raw_operation["requestBody"] = request_body
    raw_operation["security"] = security
    raw_operation["tags"] = tags
    if servers:
        raw_operation["servers"] = [{"url": url} for url in servers]
    raw_operation["x-response-schemas"] = response_schemas
    return raw_operation


def extract_operations_from_openapi(spec: Dict) -> List[ApiOperation]:
    """Extract HTTP operations from an OpenAPI or Swagger document."""
    ops: List[ApiOperation] = []
    paths = spec.get("paths", {})
    counter = 1

    for path, path_item in paths.items():
        if not isinstance(path_item, dict):
            continue

        resolved_path_item = _deep_resolve(path_item, spec)
        if not isinstance(resolved_path_item, dict):
            continue

        path_level_params = resolved_path_item.get("parameters") or []

        for method, op in resolved_path_item.items():
            if method.lower() not in _VALID_METHODS or not isinstance(op, dict):
                continue

            merged_raw_parameters, merged_parameters = _merge_parameters(
                path_level_params,
                op.get("parameters") or [],
                spec,
            )

            if _is_swagger2(spec):
                request_body_schema, request_body_examples, content_types = _extract_request_body_swagger2(
                    merged_raw_parameters,
                    op,
                    spec,
                )
                request_body = None
                response_schemas, response_examples = _extract_responses_swagger2(op, spec)
            else:
                request_body_schema, request_body_examples, content_types, request_body = _extract_request_body_openapi3(
                    op,
                    spec,
                )
                response_schemas, response_examples = _extract_responses_openapi3(op, spec)

            tags = op.get("tags") or []
            security = _effective_security(spec, op)
            servers = _extract_servers(spec, resolved_path_item, op)
            op_id = op.get("operationId") or f"OP{counter}"

            ops.append(
                ApiOperation(
                    op_id=op_id,
                    method=method.upper(),
                    path=path,
                    summary=op.get("summary") or "",
                    description=op.get("description") or "",
                    tags=tags,
                    parameters=merged_parameters,
                    request_body_schema=request_body_schema,
                    request_body_examples=request_body_examples,
                    response_schemas=response_schemas,
                    response_examples=response_examples,
                    security=security,
                    content_types=content_types,
                    servers=servers,
                    raw_operation=_build_raw_operation(
                        resolved_op=op,
                        merged_raw_parameters=merged_raw_parameters,
                        request_body=request_body,
                        response_schemas=response_schemas,
                        security=security,
                        tags=tags,
                        servers=servers,
                    ),
                )
            )
            counter += 1

    return ops


def manual_operations_input() -> List[ApiOperation]:
    """Collect API operations from stdin."""
    ops: List[ApiOperation] = []
    print("\nManuel endpoint girisi. Bitirmek icin method kismini bos birakin.\n")
    idx = 1
    while True:
        method = input(
            f"[{idx}] HTTP method (GET/POST/PUT/DELETE, bos = bitir): "
        ).strip().upper()
        if not method:
            break
        path = input(f"[{idx}] Path (orn: /users, /login): ").strip()
        summary = input(f"[{idx}] Kisa ozet: ").strip()
        description = input(f"[{idx}] Detay aciklama (opsiyonel): ").strip()
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
