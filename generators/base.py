"""Tum generator'lar icin soyut temel sinif ve ortak yardimcilar."""

import json
import re
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Tuple

from config import MAX_PARALLEL_WORKERS, RETRY_BACKOFF_SECONDS, RETRY_MAX_ATTEMPTS
from models import ApiOperation, TestCase


_NON_RETRYABLE_ERROR_MARKERS = (
    "insufficient_quota",
    "credit balance is too low",
    "incorrect api key",
    "invalid api key",
    "no api key provided",
    "invalid x-api-key",
    "authentication_error",
)

_LIST_PREFIX_RE = re.compile(r"^\s*(?:\d+[.)]\s+|[-*]\s+)")
_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",(\s*[\]}])")


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

    return []


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
    expected["response_schema_check"] = bool(expected.get("response_schema_check", False))

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
    lines = [s for l in text.splitlines() if (s := l.strip())]
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


def _apply_token_tracking(rows: List[Dict], total_tokens: int) -> None:
    """Uretilen satirlara token sayisini yazar."""
    for row in rows:
        row["tokens_used"] = total_tokens


class BaseGenerator(ABC):
    """Tum test senaryosu ureticileri icin temel sinif."""

    _aborted: bool = False

    def generate(
        self,
        operations: List[ApiOperation],
        variant_name: str,
        variant_desc: str,
        num_cases: int,
    ) -> List[Dict]:
        """Verilen operasyonlar icin test senaryolarini paralel olarak uretir."""
        if not operations or self._aborted:
            return []
        self._generation_summaries = []
        workers = min(MAX_PARALLEL_WORKERS, len(operations))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(
                executor.map(
                    lambda op: self._generate_for_operation_with_retry(op, variant_name, variant_desc, num_cases),
                    operations,
                )
            )
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
        total_tokens = 0
        total_parsed_cases = 0
        initial_valid_count = 0
        repair_added = 0
        invalid_case_count = 0
        validation_error_summary: dict[str, int] = {}

        prompt = build_llm_prompt(op, num_cases, variant_name, variant_desc)
        for attempt in range(3):
            text, used_tokens = request_completion(prompt)
            total_tokens += used_tokens
            parsed_rows = parse_llm_json_to_rows(text, op, generator_name)
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

        final_rows = accepted_rows[:num_cases]
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
                "repaired_cases": repair_added,
                "fallback_cases": len(fallback_rows),
                "validation_error_summary": validation_error_summary,
            }
        )

        print(
            f"  requested_cases={num_cases} parsed_cases={total_parsed_cases} "
            f"valid_cases={len(final_rows) - len(fallback_rows)} repaired_cases={repair_added} "
            f"fallback_cases={len(fallback_rows)}"
        )
        return final_rows

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
            try:
                return self._generate_for_operation(op, variant_name, variant_desc, num_cases)
            except Exception as exc:
                if _is_non_retryable_generation_error(exc):
                    self._aborted = True
                    print(f"  [HATA] {op.op_id} kalici provider veya kredi hatasi - generator iptal edildi: {exc}")
                    break
                if attempt < RETRY_MAX_ATTEMPTS:
                    wait = RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1))
                    print(
                        f"  [RETRY {attempt}/{RETRY_MAX_ATTEMPTS}] {op.op_id} hata: {exc} "
                        f"- {wait:.1f}s bekleyip tekrar deneniyor..."
                    )
                    time.sleep(wait)
                else:
                    print(f"  [HATA] {op.op_id} tum denemeler basarisiz: {exc}")
        return []

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
