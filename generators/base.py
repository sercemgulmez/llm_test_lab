"""Tum generator'lar icin soyut temel sinif ve ortak yardimcilar."""

import json
import logging
import random
import threading
import re
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Tuple

import config
import rate_limiter
from config import MAX_PARALLEL_WORKERS, RETRY_BACKOFF_SECONDS, RETRY_MAX_ATTEMPTS
from error_taxonomy import classify_error, failure_origin
from budget import BudgetExceeded
from models import ApiOperation, TestCase, TokenUsage
from security.redaction import redact_secrets

_logger = logging.getLogger(__name__)


_NON_RETRYABLE_ERROR_MARKERS = (
    "insufficient_quota",
    "credit balance is too low",
    "incorrect api key",
    "invalid api key",
    "no api key provided",
    "invalid x-api-key",
    "authentication_error",
    "provider response parsing error",
    "provider response parse error",
    "model output format error",
)

_LIST_PREFIX_RE = re.compile(r"^\s*(?:\d+[.)]\s+|[-*]\s+)")
_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",(\s*[\]}])")


class ProviderResponseParseError(RuntimeError):
    """Provider response object did not contain extractable final text."""


class ModelOutputFormatError(RuntimeError):
    """Extracted model text did not satisfy the testcase output contract."""


def _is_non_retryable_generation_error(exc: Exception) -> bool:
    """Kredi veya kimlik dogrulama gibi retry ile duzelmeyecek hatalari ayiklar."""
    message = str(exc).lower()
    return any(marker in message for marker in _NON_RETRYABLE_ERROR_MARKERS)


def _infer_test_type(exp_status: Optional[int], title: str = "") -> str:
    """Beklenen status ve basliga gore test tipini cikarir."""
    if exp_status is None:
        return "positive"
    if exp_status in (401, 403):
        return "auth"
    if exp_status == 422:
        return "contract"
    if exp_status >= 500:
        return "error"
    title_lower = title.lower()
    if exp_status == 400:
        if any(kw in title_lower for kw in ("boundary", "sinir", "limit", "max", "min", "edge")):
            return "boundary"
        return "negative"
    if 400 <= exp_status < 500:
        return "negative"
    return "positive"


def _coerce_status(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _coerce_status_list(value: Any) -> List[int]:
    if not isinstance(value, list):
        return []
    statuses: List[int] = []
    for item in value:
        status = _coerce_status(item)
        if status is not None and status not in statuses:
            statuses.append(status)
    return statuses


def _normalize_priority(value: Any) -> str:
    if not value:
        return "medium"
    text = str(value).strip()
    if not text:
        return "medium"
    return text


def _operation_contract(op: ApiOperation) -> dict:
    return {
        "op_id": op.op_id,
        "method": op.method,
        "path": op.path,
        "summary": op.summary,
        "description": op.description,
        "parameters": op.parameters,
        "request_body_schema": op.request_body_schema,
        "request_body_examples": op.request_body_examples,
        "response_schemas": op.response_schemas,
        "security": op.security,
        "content_types": op.content_types,
    }


def _required_path_param_names(op: ApiOperation) -> List[str]:
    return [
        param.get("name", "")
        for param in op.parameters
        if param.get("in") == "path" and param.get("required") and param.get("name")
    ]


def _required_body_fields(op: ApiOperation) -> List[str]:
    if not isinstance(op.request_body_schema, dict):
        return []
    required = op.request_body_schema.get("required")
    if not isinstance(required, list):
        return []
    return [field for field in required if isinstance(field, str) and field]


def _available_response_statuses(op: ApiOperation) -> List[int]:
    statuses: List[int] = []
    for code in op.response_schemas.keys():
        status = _coerce_status(code)
        if status is not None and status not in statuses:
            statuses.append(status)
    return statuses


def _request_template() -> dict:
    return {
        "path_params": {},
        "query_params": {},
        "headers": {},
        "cookies": {},
        "body": None,
    }


def _expected_template() -> dict:
    return {
        "status": None,
        "allowed_statuses": [],
        "result": "",
        "assertions": [],
        "response_schema_check": False,
    }


def build_llm_prompt(op: ApiOperation, num_cases: int, variant_name: str, variant_desc: str) -> str:
    """Contract-aware JSON prompt olusturur."""
    contract_json = json.dumps(_operation_contract(op), ensure_ascii=False, indent=2)
    example_array = json.dumps(
        [
            {
                "tc_id": f"{op.op_id}_TC1",
                "title": "Valid request with required fields",
                "test_type": "positive",
                "priority": "P0",
                "request": _request_template(),
                "expected": {
                    "status": 200,
                    "allowed_statuses": [200],
                    "result": "Request succeeds",
                    "assertions": [{"type": "status_code", "expected": 200}],
                    "response_schema_check": True,
                },
            }
        ],
        ensure_ascii=False,
        indent=2,
    )
    return (
        f"Sen kidemli bir backend QA muhendisisin. Asagidaki API kontrati icin TAM OLARAK {num_cases} adet "
        f"test case uret.\n\n"
        f"Variant: {variant_name}\n"
        f"Strategy: {variant_desc}\n\n"
        f"API_CONTRACT_JSON:\n{contract_json}\n\n"
        f"Cikti kurallari:\n"
        f"- Sadece strict JSON array dondur.\n"
        f"- Markdown, aciklama, code block, onsoz, sonsöz yazma.\n"
        f"- Array icinde TAM OLARAK {num_cases} object olsun.\n"
        f"- Her object alanlari: tc_id, title, test_type, priority, request, expected.\n"
        f"- test_type dagilimi positive, negative, boundary, auth, contract kategorilerini kapsasin.\n"
        f"- GET operasyonlar icin request.body null olsun.\n"
        f"- Path param varsa request.path_params icinde deger uret.\n"
        f"- Query param varsa request.query_params icinde anlamli varyasyon uret.\n"
        f"- Positive case'lerde required body alanlarini doldur.\n"
        f"- Security tanimliysa en az bir auth negative case uret.\n"
        f"- expected.status ve expected.allowed_statuses yalnizca operation response status kodlarindan secilsin.\n"
        f"- expected.assertions listesi en az bir status_code assertion'i icersin.\n"
        f"- Desteklenen assertion type degerleri yalnizca status_code, json_path_exists, json_path_equals, response_contains ve content_type_contains olsun.\n"
        f"- Assertion JSON alan adlari runner sozlesmesine tam uymali; yol alani 'path' olmali, 'json_path' kullanma.\n"
        f'- Assertion ornekleri: {{"type": "status_code", "expected": 200}}; {{"type": "json_path_exists", "path": "$.data"}}; {{"type": "json_path_equals", "path": "$.data.id", "expected": 123}}; {{"type": "response_contains", "expected": "success"}}; {{"type": "content_type_contains", "expected": "application/json"}}.\n'
        f"- response_schema_check yalnizca operasyon kontratinda response_schemas dolu ve beklenen status 2xx ise true olsun; aksi halde false olsun. Bu assertion degil, expected icinde boolean alandir.\n"
        f"- tc_id formatini {op.op_id}_TCn olarak kullan.\n\n"
        f"JSON format ornegi:\n{example_array}"
    )


def build_repair_prompt(
    op: ApiOperation,
    accepted_cases: List[dict],
    validation_errors: List[dict],
    missing_count: int,
    requested_total: int,
) -> str:
    """Eksik veya gecersiz uretimleri onarmak icin ek prompt olusturur."""
    accepted_json = json.dumps(accepted_cases, ensure_ascii=False, indent=2)
    errors_json = json.dumps(validation_errors, ensure_ascii=False, indent=2)
    contract_json = json.dumps(_operation_contract(op), ensure_ascii=False, indent=2)
    return (
        f"Onceki uretim gecersizdi. Asagida kabul edilen case'ler ve validation hatalari var.\n\n"
        f"API_CONTRACT_JSON:\n{contract_json}\n\n"
        f"REQUESTED_TOTAL: {requested_total}\n"
        f"ALREADY_ACCEPTED_COUNT: {len(accepted_cases)}\n"
        f"MISSING_COUNT: {missing_count}\n\n"
        f"ACCEPTED_CASES_JSON:\n{accepted_json}\n\n"
        f"VALIDATION_ERRORS_JSON:\n{errors_json}\n\n"
        f"Kurallar:\n"
        f"- Sadece strict JSON array dondur.\n"
        f"- Yalnizca eksik {missing_count} adet YENI testcase uret.\n"
        f"- Daha once kabul edilen tc_id'leri tekrar kullanma.\n"
        f"- Tum case'ler kontrata uygun olsun.\n"
        f"- Markdown veya aciklama yazma."
    )


def _try_parse_json_array(candidate: str) -> Optional[list]:
    text = candidate.strip()
    if not text:
        return None
    for attempt in (text, _TRAILING_COMMA_RE.sub(r"\1", text)):
        try:
            parsed = json.loads(attempt)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict) and isinstance(parsed.get("cases"), list):
            return parsed["cases"]
    return None


def extract_json_array(text: str) -> list:
    """Metinden JSON array cikarmaya calisir."""
    if not isinstance(text, str):
        return []

    parsed = _try_parse_json_array(text)
    if parsed is not None:
        return parsed

    for block in _JSON_BLOCK_RE.findall(text):
        parsed = _try_parse_json_array(block)
        if parsed is not None:
            return parsed

    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        parsed = _try_parse_json_array(text[start : end + 1])
        if parsed is not None:
            return parsed

    # NESNE-BAZLI KURTARMA
    # Modeller zaman zaman JSON'un icine kod ifadesi yaziyor, ornegin
    # "long_param": "a".repeat(2000). Bu, butun diziyi gecersiz JSON yapar ve
    # json.loads tek bir noktada patlayinca 15 case'in TAMAMI kaybolurdu.
    # Burada dizi tek tek nesnelere ayrilir: yalnizca BOZUK nesne duser,
    # digerleri kurtulur.
    #
    # Bilerek yapilmayan sey: bozuk ifadeyi yorumlamak. "a".repeat(2000) degerini
    # literal'e cevirmek, modelin gecersiz cikti urettigini gizlerdi; o olcum
    # korunsun diye nesne atilir ve sayisi dondurulur (bkz. last_salvage_stats).
    # Yalnizca GERCEKTEN bir dizi aralığı varsa kurtarmayi dene. Aksi halde
    # (ornegin eski boru-isaretli bicim) JSON olmayan metinden nesne ayiklayip
    # kendi ayristiricisinin isini elinden alirdi.
    if start != -1 and end != -1 and end > start:
        return _salvage_json_objects(text[start : end + 1])
    return []


# En son kurtarma denemesinin istatistigi.
# ISPARCACIGI BASINA tutulur: uretim ThreadPoolExecutor ile paralel kosuyor ve
# modul duzeyinde tek bir sozluk, komsu is parcaciginin sayacini okumaya yol
# acardi (defterde yanlis generator'a yazilmis "gecersiz case" demek olurdu).
_SALVAGE_STATE = threading.local()


def _salvage_slot() -> Dict[str, int]:
    slot = getattr(_SALVAGE_STATE, "stats", None)
    if slot is None:
        slot = {"recovered": 0, "discarded": 0}
        _SALVAGE_STATE.stats = slot
    return slot


def last_salvage_stats() -> Dict[str, int]:
    """Bu is parcaciginin EN SON kurtarma denemesi."""
    return dict(_salvage_slot())


def reset_salvage_stats() -> None:
    """Cagri oncesi sifirla; aksi halde onceki cagrinin sayisi deftere tasinir."""
    slot = _salvage_slot()
    slot["recovered"] = 0
    slot["discarded"] = 0


def _split_top_level_objects(text: str) -> List[str]:
    """Bir JSON dizisinin govdesini ust duzey { ... } parcalarina ayirir.

    String icindeki suslu parantezleri ve kacis karakterlerini sayar; bu yuzden
    gecersiz JSON'da da calisir.
    """
    chunks: List[str] = []
    depth = 0
    in_string = False
    escaped = False
    start_index = -1
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start_index = index
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and start_index != -1:
                chunks.append(text[start_index : index + 1])
                start_index = -1
    return chunks


def _salvage_json_objects(text: str) -> list:
    """Bozuk bir JSON dizisinden ayri ayri gecerli nesneleri toplar."""
    recovered: list = []
    discarded = 0
    for chunk in _split_top_level_objects(text or ""):
        try:
            item = json.loads(chunk)
        except (json.JSONDecodeError, ValueError):
            discarded += 1
            continue
        if isinstance(item, dict):
            recovered.append(item)
        else:
            discarded += 1
    slot = _salvage_slot()
    slot["recovered"] = len(recovered)
    slot["discarded"] = discarded
    if recovered or discarded:
        _logger.warning(
            "  [ayristirma] JSON butunuyle gecersiz; nesne-bazli kurtarma: "
            "%d case kurtarildi, %d case ATILDI (gecersiz JSON degeri).",
            len(recovered), discarded,
        )
    return recovered


def normalize_generated_case(case: dict, op: ApiOperation, generator_name: str) -> dict:
    """Tek bir LLM case nesnesini standart row formatina cevirir."""
    raw_request = case.get("request")
    raw_expected = case.get("expected")
    request_was_dict = isinstance(raw_request, dict)
    expected_was_dict = isinstance(raw_expected, dict)

    request = {**_request_template(), **(raw_request if request_was_dict else {})}
    expected = {**_expected_template(), **(raw_expected if expected_was_dict else {})}

    if op.method == "GET":
        request["body"] = None

    expected_status = _coerce_status(case.get("expected_status"))
    if expected_status is None:
        expected_status = _coerce_status(expected.get("status"))
    expected["status"] = expected_status

    allowed_statuses = _coerce_status_list(expected.get("allowed_statuses"))
    if expected_status is not None and not allowed_statuses:
        allowed_statuses = [expected_status]
    expected["allowed_statuses"] = allowed_statuses

    if not isinstance(expected.get("assertions"), list):
        expected["assertions"] = []
    json_path_aliased = 0
    for assertion in expected["assertions"]:
        if isinstance(assertion, dict) and "json_path" in assertion and "path" not in assertion:
            assertion["path"] = assertion.pop("json_path")
            json_path_aliased += 1
    expected["response_schema_check"] = bool(
        expected.get("response_schema_check", False)
        and op.response_schemas.get(str(expected_status), {}).get("content")
    )

    test_type = str(case.get("test_type") or _infer_test_type(expected_status, str(case.get("title") or ""))).strip() or "positive"

    tc = TestCase(
        generator=generator_name,
        operation_id=str(case.get("operation_id") or op.op_id),
        http_method=str(case.get("http_method") or case.get("method") or op.method).upper(),
        path=str(case.get("path") or op.path),
        tc_id=str(case.get("tc_id") or "").strip(),
        title=str(case.get("title") or "").strip(),
        request_body=case.get("request_body"),
        expected_status=expected_status,
        expected_result=str(case.get("expected_result") or expected.get("result") or "").strip(),
        request=request,
        expected=expected,
        test_type=test_type,
        priority=_normalize_priority(case.get("priority")),
        validation_errors=[],
    )
    row = tc.to_dict()
    row["_request_was_dict"] = request_was_dict
    row["_expected_was_dict"] = expected_was_dict
    row["_json_path_aliased"] = json_path_aliased
    return row


def _row_to_contract_case(row: dict) -> dict:
    return {
        "tc_id": row.get("tc_id", ""),
        "title": row.get("title", ""),
        "test_type": row.get("test_type", ""),
        "priority": row.get("priority", ""),
        "request": row.get("request", _request_template()),
        "expected": row.get("expected", _expected_template()),
    }


def _strip_internal_fields(row: dict) -> dict:
    return {key: value for key, value in row.items() if not str(key).startswith("_")}


def parse_llm_lines_to_rows(lines: List[str], op: ApiOperation, generator_name: str) -> List[Dict]:
    """Eski pipe-delimited output icin fallback parser."""
    rows: List[Dict] = []
    for line in lines:
        if line.startswith("```"):
            continue

        line = _LIST_PREFIX_RE.sub("", line).strip()
        if not line:
            continue

        parts = [p.strip() for p in line.split("|", 5)]
        if len(parts) != 6:
            continue

        tc_id, title, method_path, body_str, exp_status_str, exp_result = parts
        mp_parts = method_path.split(maxsplit=1)
        if len(mp_parts) != 2:
            continue
        method, path = mp_parts[0].upper(), mp_parts[1]

        body: Optional[dict]
        if body_str == "-":
            body = None
        else:
            try:
                parsed = json.loads(body_str)
                body = parsed if isinstance(parsed, dict) else None
            except json.JSONDecodeError:
                body = None

        rows.append(
            normalize_generated_case(
                {
                    "tc_id": tc_id,
                    "title": title,
                    "test_type": _infer_test_type(_coerce_status(exp_status_str), title),
                    "request": {
                        "path_params": {},
                        "query_params": {},
                        "headers": {},
                        "cookies": {},
                        "body": body,
                    },
                    "expected": {
                        "status": _coerce_status(exp_status_str),
                        "allowed_statuses": [_coerce_status(exp_status_str)] if _coerce_status(exp_status_str) is not None else [],
                        "result": exp_result,
                        "assertions": [],
                        "response_schema_check": False,
                    },
                    "http_method": method,
                    "path": path,
                },
                op,
                generator_name,
            )
        )
    return rows


def parse_llm_json_to_rows(text: str, op: ApiOperation, generator_name: str) -> List[dict]:
    """JSON array veya gerekirse pipe-delimited LLM ciktisini row formatina cevirir."""
    cases = extract_json_array(text)
    if cases:
        return [
            normalize_generated_case(case, op, generator_name)
            for case in cases
            if isinstance(case, dict)
        ]
    lines = [s for line in text.splitlines() if (s := line.strip())]
    return parse_llm_lines_to_rows(lines, op, generator_name)


def validate_generated_cases(op: ApiOperation, rows: List[dict], num_cases: int) -> Tuple[List[dict], List[dict]]:
    """Uretilen satirlari kontrata gore validate eder."""
    valid_rows: List[dict] = []
    invalid_rows: List[dict] = []
    seen_tc_ids: set[str] = set()
    required_path_params = _required_path_param_names(op)
    required_body_fields = _required_body_fields(op)

    for row in rows[: max(num_cases * 3, num_cases or 1)]:
        errors: List[str] = []
        tc_id = str(row.get("tc_id", "")).strip()
        request = row.get("request")
        expected = row.get("expected")

        if not tc_id:
            errors.append("tc_id bos olamaz")
        elif tc_id in seen_tc_ids:
            errors.append("tc_id duplicate olamaz")

        if row.get("operation_id") != op.op_id:
            errors.append("operation_id op ile ayni olmali")
        if row.get("http_method") != op.method:
            errors.append("http_method op.method ile ayni olmali")
        if row.get("path") != op.path:
            errors.append("path op.path ile ayni olmali")

        if not row.get("_request_was_dict", isinstance(request, dict)):
            errors.append("request dict olmali")
        if not row.get("_expected_was_dict", isinstance(expected, dict)):
            errors.append("expected dict olmali")
        if not isinstance(request, dict):
            errors.append("request dict olmali")
            request = _request_template()
        if not isinstance(expected, dict):
            errors.append("expected dict olmali")
            expected = _expected_template()

        status = _coerce_status(expected.get("status"))
        allowed_statuses = _coerce_status_list(expected.get("allowed_statuses"))
        if status is None and not allowed_statuses:
            errors.append("expected.status veya expected.allowed_statuses olmali")

        path_params = request.get("path_params")
        if not isinstance(path_params, dict):
            errors.append("request.path_params dict olmali")
            path_params = {}
        for param_name in required_path_params:
            if path_params.get(param_name) in ("", None):
                errors.append(f"path param eksik: {param_name}")

        body = request.get("body")
        if row.get("http_method") == "GET" and body not in (None, "", {}, []):
            errors.append("GET operasyonu body icermemeli")

        if row.get("test_type") == "positive" and required_body_fields:
            if not isinstance(body, dict):
                errors.append("positive case required body alanlari icermeli")
            else:
                for field_name in required_body_fields:
                    if body.get(field_name) in ("", None):
                        errors.append(f"required body field eksik: {field_name}")

        try:
            json.dumps(body, ensure_ascii=False)
        except (TypeError, ValueError):
            errors.append("body JSON serializable olmali")

        if errors:
            invalid_rows.append(
                {
                    "tc_id": tc_id or "<empty>",
                    "title": row.get("title", ""),
                    "errors": errors,
                }
            )
            continue

        seen_tc_ids.add(tc_id)
        cleaned_row = _strip_internal_fields(row)
        cleaned_row["validation_errors"] = []
        valid_rows.append(cleaned_row)

    return valid_rows[:num_cases], invalid_rows


def rate_limiter_chars_per_token() -> float:
    """Girdi tahmini icin karakter/token orani — pricing ile AYNI sabit."""
    import pricing
    return pricing.CHARS_PER_TOKEN


def rate_limiter_input_factor() -> float:
    """Girdi tahmini guvenlik katsayisi — pricing ile AYNI sabit."""
    import pricing
    return pricing.INPUT_SAFETY_FACTOR


def _next_available_at(quota_kind: str) -> Optional[str]:
    """Gunluk kota bittiginde modelin yeniden denenebilecegi EN ERKEN an.

    Takvim sifirlanmasi VARSAYILMAZ: kayan 24 saatlik pencere kullanildigi icin
    simdi + 24 saat yazilir. Bu muhafazakar taraftir; saglayici daha erken
    sifirlarsa yalnizca gec baslamis oluruz, kota asilmaz.
    """
    if quota_kind != rate_limiter.QUOTA_DAY:
        return None
    from datetime import datetime, timedelta
    return (datetime.now().astimezone() + timedelta(seconds=rate_limiter.DAY_S)).isoformat(
        timespec="seconds"
    )


def rate_limiter_actual_tokens(usage, limits) -> Optional[int]:
    """Limitorun sayacagi GERCEK token miktari.

    Saglayicilar TPM'i farkli tanimliyor: Groq dokumani toplam (girdi+cikti),
    Gemini dokumani "Tokens per minute (input)". Bu yuzden hangi olcutun
    sayilacagi limit tanimindan gelir.
    """
    if usage is None or limits is None:
        return None
    coerced = TokenUsage.coerce(usage)
    if not coerced.split_available:
        return coerced.total_tokens or None
    if limits.tpm_counts == "input_only":
        return coerced.input_tokens
    return coerced.input_tokens + coerced.billable_output_tokens


def _apply_token_tracking(rows: List[Dict], total_tokens: int) -> None:
    """Operasyonun toplam token'ini satirlara PAYLASTIRIR (K4).

    Eskiden operasyon toplami HER satira ayni ayni yazilirdi; satirlari toplamak
    gercek tuketimi satir sayisi katina cikariyordu. Artik toplam esit bolunur,
    kalan ilk satirlara birer birer dagitilir; boylece
    sum(row["tokens_used"]) == total_tokens her zaman saglanir.

    Tek satirli durumda (smoke test) sonuc degismez: satira toplamin tamami yazilir.

    RAPORLAMA UYARISI: bu deger bir TAHSIStir, olcum degildir. Tek bir test
    senaryosunun gercek token maliyeti bilinemez — olcum cagri duzeyindedir.
    Tek gecerli kaynak cagri defteridir.
    """
    if not rows:
        return
    count = len(rows)
    base, remainder = divmod(int(total_tokens or 0), count)
    for index, row in enumerate(rows):
        row["tokens_used"] = base + (1 if index < remainder else 0)


class BaseGenerator(ABC):
    """Tum test senaryosu ureticileri icin temel sinif."""

    _aborted: bool = False

    def smoke_test(self) -> List[Dict]:
        """Tek provider çağrısıyla production request/parser/normalization yolunu doğrula."""
        op = ApiOperation(
            op_id="SMOKE",
            method="GET",
            path="/smoke",
            summary="Provider smoke test",
            description="Generate exactly one minimal valid testcase.",
            response_schemas={"200": {"description": "OK"}},
        )
        prompt = build_llm_prompt(op, 1, "smoke", "Generate one minimal valid testcase.")
        text, used_tokens = self._request_completion(prompt, 1024, smoke=True)
        generator_name = f"LLM-{getattr(self, '_provider_label', type(self).__name__)}-{getattr(self, 'model', '')}-smoke"
        parsed = parse_llm_json_to_rows(text, op, generator_name)
        valid, invalid = validate_generated_cases(op, parsed, 1)
        if len(valid) != 1:
            detail = invalid[0].get("errors", []) if invalid else ["no valid testcase returned"]
            raise ModelOutputFormatError(
                f"Model output format error: text_length={len(text)}; validation={detail}"
            )
        valid[0]["prompt_variant"] = "smoke"
        _apply_token_tracking(valid, TokenUsage.coerce(used_tokens).total_tokens)
        return valid

    def _request_completion(self, prompt: str, max_tokens: int, smoke: bool = False) -> Tuple[str, int]:
        raise NotImplementedError("Provider does not implement smoke-test request path.")

    def generate(
        self,
        operations: List[ApiOperation],
        variant_name: str,
        variant_desc: str,
        num_cases: int,
        num_cases_by_op: Optional[Dict[str, int]] = None,
    ) -> List[Dict]:
        """Verilen operasyonlar icin test senaryolarini paralel olarak uretir.

        `num_cases_by_op` verilirse her operasyon icin ayri bir testcase sayisi
        kullanilir (generator basina toplami tam tutturmak icin; bkz. K1).
        Verilmezse tum operasyonlar icin `num_cases` gecerlidir.
        """
        if not operations or self._aborted:
            return []
        self._generation_summaries = []
        self._budget_stopped = False

        def _count_for(op: ApiOperation) -> int:
            if num_cases_by_op is None:
                return num_cases
            return num_cases_by_op.get(op.op_id, num_cases)

        workers = min(MAX_PARALLEL_WORKERS, len(operations))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    self._generate_for_operation_with_retry,
                    op, variant_name, variant_desc, _count_for(op),
                )
                for op in operations
            ]
            results: List[List[Dict]] = []
            for future in futures:
                try:
                    results.append(future.result())
                except BudgetExceeded as exc:
                    # Butce asiminda DIGER operasyonlarin satirlari ATILMAZ:
                    # onlarin parasi zaten harcandi. Bayrak kaldirilir, main()
                    # kalan gorevleri iptal eder.
                    self._budget_stopped = True
                    _logger.error("  [butce] %s — bu generator durduruldu.", exc)
        return [row for sublist in results for row in sublist]

    def _next_tc_id(self, op: ApiOperation, used_ids: set[str]) -> str:
        next_index = 1
        pattern = re.compile(rf"^{re.escape(op.op_id)}_TC(\d+)$")
        for tc_id in used_ids:
            match = pattern.match(tc_id)
            if match:
                next_index = max(next_index, int(match.group(1)) + 1)
        while f"{op.op_id}_TC{next_index}" in used_ids:
            next_index += 1
        return f"{op.op_id}_TC{next_index}"

    def _build_fallback_cases(
        self,
        op: ApiOperation,
        generator_name: str,
        missing_count: int,
        existing_rows: List[dict],
    ) -> List[dict]:
        from generators.traditional import TraditionalGenerator

        used_ids = {str(row.get("tc_id", "")).strip() for row in existing_rows if row.get("tc_id")}
        fallback_rows = TraditionalGenerator()._generate_for_operation(op, "", "", 0)
        generated: List[dict] = []

        for seed_row in fallback_rows:
            if len(generated) >= missing_count:
                break
            row = dict(seed_row)
            row["generator"] = generator_name
            row["operation_id"] = op.op_id
            row["http_method"] = op.method
            row["path"] = op.path
            row["tc_id"] = self._next_tc_id(op, used_ids)
            row["generation_metadata"] = {
                "source": "fallback",
                "repaired": False,
                "fallback": True,
                "valid": True,
            }
            used_ids.add(row["tc_id"])
            generated.append(row)

        while len(generated) < missing_count:
            tc_id = self._next_tc_id(op, used_ids)
            used_ids.add(tc_id)
            status_candidates = _available_response_statuses(op) or [200, 400, 401, 404, 500]
            status = status_candidates[len(generated) % len(status_candidates)]
            title = f"Fallback case {len(generated) + 1}"
            generated.append(
                TestCase(
                    generator=generator_name,
                    operation_id=op.op_id,
                    http_method=op.method,
                    path=op.path,
                    tc_id=tc_id,
                    title=title,
                    request=_request_template(),
                    expected={
                        "status": status,
                        "allowed_statuses": [status],
                        "result": "Fallback case generated after invalid LLM output",
                        "assertions": [{"type": "status_code", "expected": status}],
                        "response_schema_check": False,
                    },
                    test_type=_infer_test_type(status, title),
                    priority="medium",
                ).to_dict()
            )
            generated[-1]["generation_metadata"] = {
                "source": "fallback",
                "repaired": False,
                "fallback": True,
                "valid": True,
            }

        return generated[:missing_count]

    def _generate_cases_with_repair(
        self,
        op: ApiOperation,
        variant_name: str,
        variant_desc: str,
        num_cases: int,
        generator_name: str,
        request_completion: Callable[[str], Tuple[str, int]],
    ) -> List[Dict]:
        accepted_rows: List[dict] = []
        invalid_rows: List[dict] = []
        call_usages: List[TokenUsage] = []
        total_parsed_cases = 0
        initial_valid_count = 0
        repair_added = 0
        invalid_case_count = 0
        discarded_total = 0   # gecersiz JSON yuzunden atilan case (Bulgu 1)
        validation_error_summary: dict[str, int] = {}
        json_path_aliased = 0

        ledger = getattr(self, "_call_ledger", None)
        retry_index = getattr(self, "_retry_index", 1)
        previously_accepted = 0  # defterde kumulatif degil ARTIMSAL sayi tutulur

        repeat_index = getattr(self, "_repeat_index", 0)
        had_infrastructure_failure = False

        prompt = build_llm_prompt(op, num_cases, variant_name, variant_desc)
        for attempt in range(3):
            if attempt > 0:
                call_type_for_log = "repair"
            elif retry_index > 1:
                call_type_for_log = "retry"
            else:
                call_type_for_log = "ana"

            # Butce sigortasi cagri ONCESI: sigorta zaten atmissa bu gorev hic
            # para harcamadan durur (kuyrukta bekleyen gorevler icin onemli).
            self._check_budget()
            started_at = time.perf_counter()
            call_meta_extra: dict = {}
            # Onceki cagrinin meta'si (model_returned, response_id, finish_reason,
            # sampling) BU cagriya tasinmamali: istek hic gonderilmezse —ornegin
            # kota yuzunden— defter kaydi eski cagrinin kimligini gosteriyordu.
            self._last_call_meta = None
            try:
                text, used_tokens, call_meta_extra = self._rate_limited_call(
                    request_completion, prompt, num_cases
                )
            except Exception as exc:
                # Yanit alinamadi: defter bunu KAYBETMEMELI, sonra geri getirilemez.
                # Siniflandirma defterden BAGIMSIZ hesaplanir: --no-call-ledger ile
                # kosulsa bile asagidaki karar ayni olmali.
                error_class = classify_error(exc)
                if ledger is not None:
                    ledger.record_failed_call(
                        generator=generator_name,
                        variant=variant_name,
                        operation_id=op.op_id,
                        method=op.method,
                        path=op.path,
                        attempt=attempt,
                        call_type=call_type_for_log,
                        exc=exc,
                        repeat_index=repeat_index,
                        latency_ms=int((time.perf_counter() - started_at) * 1000),
                        call_meta=getattr(self, "_last_call_meta", None),
                        prompt_chars=len(prompt),
                        provider_label=getattr(self, "_provider_label", ""),
                        limiter_meta=call_meta_extra,
                        # _last_call_meta yukarida sifirlandi; istek patlayinca
                        # dolmaz. Model ve tavan generator'dan gelmeli, yoksa
                        # guard tahmini None olur ve butce bu cagriyi gormez.
                        model=getattr(self, "model", ""),
                        max_output_tokens=self._max_tokens_for(num_cases),
                    )
                if failure_origin(error_class) == "altyapi":
                    had_infrastructure_failure = True

                # Ilk denemede elde hicbir sey yok: yukari tasi, generator
                # duzeyindeki retry tum operasyonu yeniden denesin.
                if attempt == 0 or not accepted_rows:
                    raise

                # SONRAKI denemede zaten kabul edilmis satirlar varsa onlari
                # ATMAYIZ. Eskiden burada kosulsuz `raise` vardi: repair cagrisi
                # patlayinca istisna fonksiyondan cikiyor, dongunun ASAGISINDAKI
                # fallback ve `return final_rows` hic calismiyor ve ANA CAGRININ
                # urettigi her sey cope gidiyordu. Olculdu: Groq'ta defterde 206
                # kabul edilmis case varken CSV'ye yalnizca 15 satir girdi.
                # Ucretli fazda bu, parasi odenmis ciktinin atilmasi demek.
                # Hata deftere YAZILDI ve altyapi kokeni ISARETLENDI; burada
                # yalnizca eldeki veri korunur, fallback eksigi tamamlar.
                _logger.warning(
                    "  [%s] %s %d. deneme basarisiz (%s) — ELDEKI %d satir KORUNUYOR, "
                    "eksik %d satir fallback ile tamamlanacak.",
                    generator_name, op.op_id, attempt + 1, error_class,
                    len(accepted_rows), max(0, num_cases - len(accepted_rows)),
                )
                break
            # latency_ms SAF saglayici suresidir: limitor beklemesi ayri alanda
            # tutulur, aksi halde "model ne kadar yavas" sorusu cevaplanamaz.
            latency_ms = max(
                0,
                int((time.perf_counter() - started_at) * 1000)
                - int(call_meta_extra.get("limiter_wait_ms") or 0),
            )
            usage = TokenUsage.coerce(used_tokens)
            call_usages.append(usage)
            # Kurtarma sayaci BU cagri icin olculsun: onceki cagrinin sayisi
            # deftere tasinmamali.
            reset_salvage_stats()
            parsed_rows = parse_llm_json_to_rows(text, op, generator_name)
            json_path_aliased += sum(int(row.get("_json_path_aliased") or 0) for row in parsed_rows)
            salvage = last_salvage_stats()
            call_meta_extra["salvaged_cases"] = salvage["recovered"]
            call_meta_extra["discarded_cases"] = salvage["discarded"]
            discarded_total += salvage["discarded"]
            source_label = "generated" if attempt == 0 else "repaired"
            for row in parsed_rows:
                metadata = row.get("generation_metadata") if isinstance(row.get("generation_metadata"), dict) else {}
                metadata.update({
                    "source": source_label,
                    "repaired": attempt > 0,
                    "fallback": False,
                    "valid": True,
                })
                row["generation_metadata"] = metadata
            total_parsed_cases += len(parsed_rows)

            candidate_rows = accepted_rows + parsed_rows
            accepted_rows, invalid_rows = validate_generated_cases(op, candidate_rows, num_cases)
            llm_valid_count = len(accepted_rows)
            invalid_case_count += len(invalid_rows)
            for invalid in invalid_rows:
                for error in invalid.get("errors", []):
                    validation_error_summary[error] = validation_error_summary.get(error, 0) + 1

            if ledger is not None:
                ledger.record_call(
                    generator=generator_name,
                    variant=variant_name,
                    operation_id=op.op_id,
                    method=op.method,
                    path=op.path,
                    attempt=attempt,
                    call_type=call_type_for_log,
                    usage=usage,
                    repeat_index=repeat_index,
                    latency_ms=latency_ms,
                    call_meta=getattr(self, "_last_call_meta", None),
                    accepted_cases=max(0, llm_valid_count - previously_accepted),
                    rejected_cases=len(invalid_rows),
                    raw_response=text,
                    cases=accepted_rows,
                    validation_errors=invalid_rows,
                    prompt_chars=len(prompt),
                    limiter_meta=call_meta_extra,
                )

            previously_accepted = llm_valid_count

            # Cagri SONRASI: maliyet ancak simdi bilinir. Sert esik asildiysa
            # BudgetExceeded firlar ve kalan denemeler yapilmaz.
            self._check_budget()

            if attempt == 0:
                initial_valid_count = llm_valid_count
            else:
                repair_added = max(repair_added, llm_valid_count - initial_valid_count)

            missing_count = num_cases - llm_valid_count
            if missing_count <= 0:
                break
            if attempt >= 2:
                break

            prompt = build_repair_prompt(
                op=op,
                accepted_cases=[_row_to_contract_case(row) for row in accepted_rows],
                validation_errors=invalid_rows,
                missing_count=missing_count,
                requested_total=num_cases,
            )

        fallback_rows: List[dict] = []
        if len(accepted_rows) < num_cases:
            fallback_rows = self._build_fallback_cases(
                op=op,
                generator_name=generator_name,
                missing_count=num_cases - len(accepted_rows),
                existing_rows=accepted_rows,
            )
            accepted_rows.extend(fallback_rows)
            if ledger is not None and fallback_rows:
                origin = "altyapi" if had_infrastructure_failure else "icerik"
                reason = (
                    "Saglayici cagrisi basarisiz oldu (altyapi)"
                    if had_infrastructure_failure
                    else "LLM yeterli gecerli case uretemedi (icerik)"
                )
                ledger.record_fallback(
                    generator=generator_name,
                    variant=variant_name,
                    operation_id=op.op_id,
                    method=op.method,
                    path=op.path,
                    cases=fallback_rows,
                    reason=reason,
                    origin=origin,
                )

        final_rows = accepted_rows[:num_cases]
        for row in final_rows:
            row["prompt_variant"] = variant_name
        total_tokens = sum(item.total_tokens for item in call_usages)
        if total_tokens:
            _apply_token_tracking(final_rows, total_tokens)

        self._generation_summaries.append(
            {
                "generator": generator_name,
                "operation_id": op.op_id,
                "method": op.method,
                "path": op.path,
                "requested_cases": num_cases,
                "parsed_cases": total_parsed_cases,
                "generated_cases": len(final_rows),
                "valid_cases": len(final_rows),
                "invalid_cases": invalid_case_count,
                "discarded_cases": discarded_total,
                "repaired_cases": repair_added,
                "fallback_cases": len(fallback_rows),
                "json_path_aliased": json_path_aliased,
                "fallback_origin": (
                    ("altyapi" if had_infrastructure_failure else "icerik") if fallback_rows else None
                ),
                "validation_error_summary": validation_error_summary,
            }
        )

        _logger.info(
            "  requested_cases=%d parsed_cases=%d valid_cases=%d repaired_cases=%d fallback_cases=%d",
            num_cases,
            total_parsed_cases,
            len(final_rows) - len(fallback_rows),
            repair_added,
            len(fallback_rows),
        )
        return final_rows

    def _rate_limited_call(self, request_completion, prompt: str, num_cases: int):
        """Limitorden gecerek cagri yapar; 429'da REAKTIF olarak yeniden dener.

        Doner: (text, usage, meta) — meta limitor ve 429 bilgilerini tasir ve
        oldugu gibi deftere yazilir.

        Reaktif 429 denemeleri RETRY_MAX_ATTEMPTS'ten AYRI sayilir: bir hiz
        limiti beklemesi modelin icerik uretme sansini tuketmemeli.
        """
        limiter = getattr(self, "_rate_limiter", None)
        provider = getattr(self, "_provider_label", "")
        model = getattr(self, "model", "")
        meta = {
            "limiter_wait_ms": 0,
            "reserved_input_tokens": 0,
            "reserved_output_tokens": 0,
            "reactive_429_count": 0,
            "retry_after_s": None,
            "quota_kind": None,
            "rate_limit_headers": {},
            # Gecersiz JSON yuzunden kurtarilan / atilan case sayisi (Bulgu 1).
            "salvaged_cases": 0,
            "discarded_cases": 0,
        }
        if limiter is None or not limiter.is_managed(provider, model):
            text, usage = request_completion(prompt)
            return text, usage, meta

        output_ceiling = self._max_tokens_for(num_cases)
        input_estimate = int(
            (len(prompt) / rate_limiter_chars_per_token()) * rate_limiter_input_factor()
        )
        meta["reserved_input_tokens"] = input_estimate
        meta["reserved_output_tokens"] = output_ceiling

        # Hicbir zaman basarili olamayacak istek: beklemek cozmez.
        limiter.preflight(provider, model, input_estimate + output_ceiling)

        attempts = 0
        while True:
            try:
                reservation = limiter.reserve(provider, model, input_estimate, output_ceiling)
            except rate_limiter.QuotaExhausted as exc:
                # Gunluk kota bitti: BEKLEMEYIZ. Gorev birakilir, kosu kisa kalir.
                meta["quota_kind"] = rate_limiter.QUOTA_DAY
                _logger.error("  [limitor] %s", exc)
                raise
            meta["limiter_wait_ms"] += reservation.wait_ms
            try:
                text, usage = request_completion(prompt)
            except rate_limiter.QuotaExhausted:
                # Rezervasyon alinamadi (semafor zaten birakildi); kota bugun bitti.
                raise
            except Exception as exc:
                limiter.settle(reservation, actual_tokens=None)
                if classify_error(exc) != "RATE_LIMIT":
                    raise
                headers = rate_limiter.response_headers(exc)
                kind = rate_limiter.quota_kind(exc, provider)
                meta["reactive_429_count"] += 1
                meta["quota_kind"] = kind
                meta["retry_after_s"] = rate_limiter.retry_after_seconds(exc)
                meta["rate_limit_headers"] = rate_limiter.known_rate_limit_headers(provider, headers)
                limiter.calibrate(provider, model, headers)
                limiter.note_reactive_429(provider, model, kind, _next_available_at(kind))
                attempts += 1
                if kind == rate_limiter.QUOTA_DAY:
                    # Gunluk kota bittiyse AYNI OTURUMDA tekrar denenmez;
                    # gunlerce beklemek yerine durulur ve raporlanir.
                    _logger.error(
                        "  [limitor] %s/%s GUNLUK KOTA bitti — bu oturumda tekrar denenmeyecek.",
                        provider, model,
                    )
                    raise
                if attempts > config.REACTIVE_429_MAX_RETRIES:
                    raise
                wait = meta["retry_after_s"]
                if wait is None:
                    wait = rate_limiter.MINUTE_S
                wait += random.uniform(0.0, config.REACTIVE_429_JITTER_SECONDS)
                _logger.warning(
                    "  [limitor] %s/%s 429 (dakika kotasi) — %.1f sn beklenip tekrar denenecek "
                    "(reaktif %d/%d, icerik denemesi HARCANMADI).",
                    provider, model, wait, attempts, config.REACTIVE_429_MAX_RETRIES,
                )
                meta["limiter_wait_ms"] += int(wait * 1000)
                time.sleep(wait)
                continue

            actual = rate_limiter_actual_tokens(usage, limiter.limits_for(provider, model))
            limiter.settle(reservation, actual_tokens=actual)
            headers = rate_limiter.response_headers(usage)
            if headers:
                limiter.calibrate(provider, model, headers)
                meta["rate_limit_headers"] = rate_limiter.known_rate_limit_headers(provider, headers)
            return text, usage, meta

    def _max_tokens_for(self, num_cases: int) -> int:
        """Bu cagri icin cikti TAVANI.

        Degerler degismedi; yalnizca tek kaynaga tasindi ki limitor de ayni
        tavani rezerve edebilsin. Deney parametresi DEGISTIRILMEDI.
        """
        provider = getattr(self, "_provider_label", "").lower()
        ceiling = config.MAX_TOKENS_BY_PROVIDER.get(provider, 8192)
        return min(ceiling, max(2048, num_cases * 200))

    def _check_budget(self) -> None:
        """Butce sigortasi bagliysa esikleri kontrol eder (K6)."""
        guard = getattr(self, "_budget_guard", None)
        if guard is not None:
            guard.check()

    def _generate_for_operation_with_retry(
        self,
        op: ApiOperation,
        variant_name: str,
        variant_desc: str,
        num_cases: int,
    ) -> List[Dict]:
        """Tek operasyon icin retry mantigiyla senaryo uretir."""
        if self._aborted:
            return []
        for attempt in range(1, RETRY_MAX_ATTEMPTS + 1):
            self._retry_index = attempt
            try:
                return self._generate_for_operation(op, variant_name, variant_desc, num_cases)
            except rate_limiter.QuotaExhausted as exc:
                # Kota bitisi yeniden denenmez: ayni gun icinde cozulmez.
                # ALTYAPI kaynakli sayilir ki resume onu aday olarak gorsun.
                self._infra_failures = getattr(self, "_infra_failures", 0) + 1
                self._aborted = True
                _logger.error("  [ATLANDI] %s — gunluk kota: %s", op.op_id, exc)
                return []

            except BudgetExceeded:
                # Butce asimi bir altyapi hatasi DEGILDIR: yeniden denemek para
                # harcamaya devam etmek olurdu. Generator'i iptal et ve yukari tasi.
                self._aborted = True
                raise
            except Exception as exc:
                safe_exc = redact_secrets(str(exc))
                is_missing_key = isinstance(exc, RuntimeError) and "environment variable is not set" in str(exc)
                if failure_origin(classify_error(exc)) == "altyapi":
                    # Gorev satir uretemeden coktuyse bile bunun ALTYAPI kaynakli
                    # oldugu kaydedilmeli; aksi halde resume onu yeniden kosulabilir
                    # aday olarak goremez (fallback bile uretilememis olur).
                    self._infra_failures = getattr(self, "_infra_failures", 0) + 1
                if is_missing_key or _is_non_retryable_generation_error(exc):
                    self._aborted = True
                    if is_missing_key:
                        # Eksik anahtar bir calisma hatasi degil, konfigurasyon eksigidir:
                        # generator atlanir ve pipeline devam eder.
                        _logger.warning("  [ATLANDI] %s — API anahtari tanimli degil: %s", op.op_id, safe_exc)
                    else:
                        _logger.error("  [HATA] %s — generator iptal edildi: %s", op.op_id, safe_exc)
                    break
                if attempt < RETRY_MAX_ATTEMPTS:
                    wait = RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1))
                    _logger.warning(
                        "  [RETRY %d/%d] %s hata: %s - %.1fs bekleyip tekrar deneniyor...",
                        attempt, RETRY_MAX_ATTEMPTS, op.op_id, safe_exc, wait,
                    )
                    time.sleep(wait)
                else:
                    _logger.error("  [HATA] %s tum denemeler basarisiz: %s", op.op_id, safe_exc)
                    return self._fallback_for_failed_operation(op, variant_name, num_cases, exc)
        return []

    def _fallback_for_failed_operation(
        self,
        op: ApiOperation,
        variant_name: str,
        num_cases: int,
        exc: BaseException,
    ) -> list[dict]:
        """Ana cagri TUM denemelerde basarisiz oldu: operasyon fallback'e duser.

        Eskiden burada [] donuyordu: operasyon 0 satir uretiyor, fallback'e
        ugramiyor ve gorev eksik satirla "tamamlandi" sayiliyordu. Olculdu
        (3 Ekim, gemini-2.5-flash edge_focused): EP1/EP3/EP5 free tier 429'u
        aldi, gorev 30/75 satirla kapandi; kayip 45 satir fallback payinda
        bile gorunmedi. Generator'in IPTAL edildigi durumlar (eksik anahtar,
        yeniden denenemez hata, gunluk kota, butce) buraya gelmez.
        """
        if num_cases <= 0:
            return []
        error_class = classify_error(exc) if isinstance(exc, Exception) else "UNKNOWN_ERROR"
        origin = failure_origin(error_class)
        generator_name = (
            f"LLM-{getattr(self, '_provider_label', type(self).__name__)}-{getattr(self, 'model', '')}"
        )
        rows = self._build_fallback_cases(
            op=op, generator_name=generator_name, missing_count=num_cases, existing_rows=[],
        )
        for row in rows:
            row["prompt_variant"] = variant_name

        ledger = getattr(self, "_call_ledger", None)
        if ledger is not None and rows:
            ledger.record_fallback(
                generator=generator_name,
                variant=variant_name,
                operation_id=op.op_id,
                method=op.method,
                path=op.path,
                cases=rows,
                reason=f"Ana cagri tum denemelerde basarisiz oldu ({error_class})",
                origin=origin,
            )
        if getattr(self, "_generation_summaries", None) is None:
            self._generation_summaries = []
        self._generation_summaries.append(
            {
                "generator": generator_name,
                "operation_id": op.op_id,
                "method": op.method,
                "path": op.path,
                "requested_cases": num_cases,
                "parsed_cases": 0,
                "generated_cases": len(rows),
                "valid_cases": len(rows),
                "invalid_cases": 0,
                "discarded_cases": 0,
                "repaired_cases": 0,
                "fallback_cases": len(rows),
                "json_path_aliased": 0,
                "fallback_origin": origin if rows else None,
                "all_attempts_failed": True,
                "error_class": error_class,
                "validation_error_summary": {},
            }
        )
        _logger.warning(
            "  [%s] %s ana cagri tum denemelerde basarisiz (%s) — %d satir fallback ile uretildi (koken: %s).",
            generator_name, op.op_id, error_class, len(rows), origin,
        )
        return rows

    @abstractmethod
    def _generate_for_operation(
        self,
        op: ApiOperation,
        variant_name: str,
        variant_desc: str,
        num_cases: int,
    ) -> List[Dict]:
        """Alt siniflar bu metodu uygular."""
        ...
