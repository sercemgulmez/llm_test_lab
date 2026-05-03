"""curl komutlarını parse ederek ApiOperation ve request metadata'sını üretir."""

import json
import re
import shlex
from typing import Dict, List, Tuple
from urllib.parse import urlparse, parse_qs

from models import ApiOperation

# Teste katkı sağlamayan rutin header'lar
_SKIP_HEADERS = {
    "accept", "accept-language", "connection", "content-type",
    "user-agent", "sec-fetch-dest", "sec-fetch-mode", "sec-fetch-site",
    "sec-ch-ua", "sec-ch-ua-mobile", "sec-ch-ua-platform",
    "origin", "referer",
}


def parse_curl(curl_text: str) -> Tuple[ApiOperation, str, Dict[str, str], Dict[str, str]]:
    """
    curl komutunu parse eder.

    Args:
        curl_text: Tek veya çok satırlı ham curl komutu.

    Returns:
        (operation, base_url, headers, cookies)
        - operation : ApiOperation (request_body_examples ve description dolu)
        - base_url  : "https://host" şeklinde
        - headers   : İstek için gerekli ekstra header dict'i
        - cookies   : Cookie dict'i
    """
    # Satır devamlarını temizle
    curl_text = curl_text.replace("\\\r\n", " ").replace("\\\n", " ")

    try:
        tokens = shlex.split(curl_text)
    except ValueError:
        tokens = curl_text.split()

    url = ""
    method = None
    headers: Dict[str, str] = {}
    cookies: Dict[str, str] = {}
    body = None

    i = 0
    while i < len(tokens):
        tok = tokens[i]

        if tok == "curl":
            i += 1
            continue

        if not tok.startswith("-") and not url and tok.startswith("http"):
            url = tok.strip("'\"")
            i += 1
            continue

        if tok in ("-H", "--header") and i + 1 < len(tokens):
            raw = tokens[i + 1]
            key, _, value = raw.partition(": ")
            key = key.strip()
            if key.lower() == "cookie":
                _parse_cookie_string(value, cookies)
            else:
                headers[key] = value.strip()
            i += 2
            continue

        if tok in ("-b", "--cookie") and i + 1 < len(tokens):
            _parse_cookie_string(tokens[i + 1], cookies)
            i += 2
            continue

        if tok in ("-X", "--request") and i + 1 < len(tokens):
            method = tokens[i + 1].upper()
            i += 2
            continue

        if tok in ("-d", "--data", "--data-raw", "--data-binary") and i + 1 < len(tokens):
            body = tokens[i + 1]
            i += 2
            continue

        i += 1

    if not url:
        raise ValueError("curl komutunda geçerli bir URL bulunamadı.")

    if not method:
        method = "POST" if body else "GET"

    parsed_url = urlparse(url)
    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
    path = parsed_url.path or "/"
    if parsed_url.query:
        path += f"?{parsed_url.query}"

    # Request body examples
    request_body_examples: list[dict] = []
    example_body_str = ""
    if body:
        try:
            parsed_body = json.loads(body)
            example_body_str = json.dumps(parsed_body, ensure_ascii=False, indent=2)
            if isinstance(parsed_body, dict):
                request_body_examples = [parsed_body]
        except (json.JSONDecodeError, ValueError):
            example_body_str = body

    # Query params → parameters list
    parameters: list[dict] = []
    if parsed_url.query:
        for k, v in parse_qs(parsed_url.query, keep_blank_values=True).items():
            parameters.append({"name": k, "in": "query", "example": v[0] if v else ""})

    # Content-Type header → content_types
    ct_header = headers.get("Content-Type") or headers.get("content-type") or ""
    content_types = [ct_header.split(";")[0].strip()] if ct_header else []

    # Authorization header → security
    auth_header = headers.get("Authorization") or headers.get("authorization") or ""
    security: list[dict] = []
    if auth_header:
        scheme = auth_header.split()[0].lower() if auth_header.split() else "bearer"
        security = [{scheme: []}]

    # Notable (non-rutin) header'ları description'a ekle
    notable = {k: v for k, v in headers.items() if k.lower() not in _SKIP_HEADERS}
    desc_parts = []
    if notable:
        desc_parts.append(
            "Gerekli header'lar: " + ", ".join(f"{k}: {v}" for k, v in notable.items())
        )
    if example_body_str:
        desc_parts.append(f"Örnek request body:\n{example_body_str}")

    op = ApiOperation(
        op_id="CURL_OP1",
        method=method,
        path=path,
        summary=f"{method} {path}",
        description="\n".join(desc_parts),
        parameters=parameters,
        request_body_examples=request_body_examples,
        content_types=content_types,
        security=security,
        servers=[base_url],
    )

    return op, base_url, headers, cookies


def parse_curl_collection(
    text: str,
) -> List[Tuple[ApiOperation, str, Dict[str, str], Dict[str, str]]]:
    """
    Bir metin içindeki birden fazla curl komutunu parse eder.

    Returns:
        Her curl için (operation, base_url, headers, cookies) tuple listesi.
        op_id'ler CURL_OP1, CURL_OP2, ... şeklinde otomatik atanır.
    """
    blocks = re.split(r"(?m)^(?=curl\s)", text.strip())
    blocks = [b.strip() for b in blocks if b.strip().startswith("curl")]

    if not blocks:
        raise ValueError("Dosyada hiç curl komutu bulunamadı.")

    results: List[Tuple[ApiOperation, str, Dict[str, str], Dict[str, str]]] = []
    for idx, block in enumerate(blocks, start=1):
        op, base_url, headers, cookies = parse_curl(block)
        op.op_id = f"CURL_OP{idx}"
        op.summary = f"[OP{idx}] {op.method} {op.path}"
        results.append((op, base_url, headers, cookies))

    return results


def _parse_cookie_string(cookie_str: str, target: Dict[str, str]) -> None:
    """'name=value; name2=value2' formatındaki string'i dict'e ekler."""
    for part in cookie_str.split("; "):
        k, _, v = part.partition("=")
        k = k.strip()
        if k:
            target[k] = v
