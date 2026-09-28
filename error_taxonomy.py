"""Saglayici hatalarinin ortak siniflandirmasi.

Hem canli smoke test (scripts/check_model_access.py) hem de kosu anindaki
cagri defteri (call_ledger) ayni taksonomiyi kullanir; boylece "429 mi,
kota mi, timeout mu" ayrimi iki yerde farkli tanimlanmaz.
"""

from __future__ import annotations

# Altyapi kaynakli (yeniden denenebilir / gecici) hata siniflari.
INFRASTRUCTURE_ERRORS = frozenset({
    "RATE_LIMIT",
    "TIMEOUT",
    "NETWORK_ERROR",
    "PROVIDER_ERROR",
    "BILLING_QUOTA_ERROR",
    "AUTH_ERROR",
    "MODEL_NOT_FOUND",
    "MODEL_ACCESS_ERROR",
})

# Modelin ciktisindan kaynaklanan (icerik) hata siniflari.
CONTENT_ERRORS = frozenset({
    "PROVIDER_RESPONSE_PARSE_ERROR",
    "MODEL_OUTPUT_FORMAT_ERROR",
    "REQUEST_CONTRACT_ERROR",
})


def classify_error(exc: Exception) -> str:
    status = getattr(exc, "status_code", None)
    code = str(getattr(exc, "code", "") or "").lower()
    name = exc.__class__.__name__.lower()
    message = str(exc).lower()
    combined = f"{code} {name} {message}"
    if "providerresponseparseerror" in combined or "provider response parse error" in combined:
        return "PROVIDER_RESPONSE_PARSE_ERROR"
    if "modeloutputformaterror" in combined or "model output format error" in combined:
        return "MODEL_OUTPUT_FORMAT_ERROR"
    if isinstance(exc, TimeoutError) or "timeout" in combined or "timed out" in combined:
        return "TIMEOUT"
    if any(term in combined for term in ("connection", "network", "dns", "name resolution")):
        return "NETWORK_ERROR"
    if status == 401 or any(term in combined for term in ("invalid api key", "incorrect api key", "authentication_error", "unauthorized")):
        return "AUTH_ERROR"
    if any(term in combined for term in ("quota", "billing", "credit", "insufficient_quota")):
        return "BILLING_QUOTA_ERROR"
    if status == 429 or any(term in combined for term in ("rate limit", "rate_limit", "too many requests")):
        return "RATE_LIMIT"
    if status == 404 or any(term in combined for term in ("model_not_found", "model not found", "does not exist", "no longer available")):
        return "MODEL_NOT_FOUND"
    if status == 403 or any(term in combined for term in ("access denied", "permission", "not authorized", "forbidden")):
        return "MODEL_ACCESS_ERROR"
    if status in (400, 422) or isinstance(exc, (TypeError, ValueError)):
        return "REQUEST_CONTRACT_ERROR"
    if status is not None and int(status) >= 500:
        return "PROVIDER_ERROR"
    if isinstance(exc, (ImportError, AttributeError, NameError, NotImplementedError)):
        return "CODE_ERROR"
    return "UNKNOWN_ERROR"


def failure_origin(error_class: str) -> str:
    """Hata sinifini "altyapi" / "icerik" / "bilinmiyor" olarak etiketler."""
    if error_class in INFRASTRUCTURE_ERRORS:
        return "altyapi"
    if error_class in CONTENT_ERRORS:
        return "icerik"
    return "bilinmiyor"
