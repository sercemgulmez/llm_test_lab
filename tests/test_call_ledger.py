"""K13 regresyonu: cagri defteri (kosu aninda kaydedilmezse geri getirilemeyen veri)."""

import json

from call_ledger import CallLedger, summarize_by_generator
from models import TokenUsage


def _case(tc_id, fallback=False, body=None, query=None, headers=None):
    return {
        "tc_id": tc_id,
        "generation_metadata": {"fallback": fallback, "source": "fallback" if fallback else "generated"},
        "expected_status": 200,
        "request": {
            "path_params": {}, "query_params": query or {}, "headers": headers or {},
            "cookies": {}, "body": body,
        },
    }


def _ledger(tmp_path, enabled=True):
    return CallLedger(str(tmp_path), run_id="run_test", enabled=enabled)


def test_call_record_captures_all_required_fields(tmp_path):
    led = _ledger(tmp_path)
    led.record_call(
        generator="LLM-OpenAI-gpt-4.1", variant="basic", operation_id="EP1",
        method="POST", path="/post", attempt=0, call_type="ana",
        usage=TokenUsage(input_tokens=800, output_tokens=400, split_available=True),
        accepted_cases=9, rejected_cases=1, raw_response='[{"tc_id": "EP1_TC1"}]',
        cases=[_case("EP1_TC1", query={"q": "v"}, headers={"X-A": "1"}, body={"f": 1})],
    )
    led.flush()

    record = json.loads(led.path.read_text(encoding="utf-8").strip())
    for field in ("generator", "variant", "operation_id", "attempt", "call_type",
                  "input_tokens", "output_tokens", "accepted_cases", "rejected_cases",
                  "raw_response", "cases"):
        assert field in record, f"{field} defterde yok"
    assert record["input_tokens"] == 800
    assert record["output_tokens"] == 400
    assert record["split_available"] is True


def test_full_request_is_captured_including_query_and_headers(tmp_path):
    """CSV bu alanlari icermiyor; defter icermezse geri getirilemez."""
    led = _ledger(tmp_path)
    led.record_call(
        generator="G", variant="basic", operation_id="EP1", method="GET", path="/get",
        attempt=0, call_type="ana", usage=TokenUsage(total_tokens=10),
        accepted_cases=1, rejected_cases=0, raw_response="[]",
        cases=[_case("EP1_TC1", query={"page": 2}, headers={"X-Trace": "abc"}, body={"k": "v"})],
    )
    led.flush()

    case = json.loads(led.path.read_text(encoding="utf-8").strip())["cases"][0]
    assert case["request"]["query_params"] == {"page": 2}
    assert case["request"]["headers"] == {"X-Trace": "abc"}
    assert case["request"]["body"] == {"k": "v"}


def test_secrets_are_redacted_in_raw_response_and_request(tmp_path):
    secret = "sk-ant-api03-" + "A" * 95
    led = _ledger(tmp_path)
    led.record_call(
        generator="G", variant="basic", operation_id="EP1", method="GET", path="/get",
        attempt=0, call_type="ana", usage=TokenUsage(total_tokens=5),
        accepted_cases=0, rejected_cases=0,
        raw_response=f"error: invalid key {secret}",
        cases=[_case("EP1_TC1", headers={"Authorization": f"Bearer {secret}"})],
    )
    led.flush()

    text = led.path.read_text(encoding="utf-8")
    assert secret not in text
    assert "[REDACTED]" in text


def test_fallback_records_are_separate_and_tokenless(tmp_path):
    led = _ledger(tmp_path)
    led.record_fallback(
        generator="LLM-X", variant="basic", operation_id="EP1", method="GET", path="/get",
        cases=[_case("EP1_TC9", fallback=True)],
    )
    led.flush()

    record = json.loads(led.path.read_text(encoding="utf-8").strip())
    assert record["call_type"] == "fallback"
    assert record["total_tokens"] == 0
    assert record["cases"][0]["is_fallback"] is True


def test_summary_reports_fallback_share_and_acceptance_rate(tmp_path):
    led = _ledger(tmp_path)
    led.record_call(
        generator="LLM-X", variant="basic", operation_id="EP1", method="GET", path="/get",
        attempt=0, call_type="ana", usage=TokenUsage(input_tokens=100, output_tokens=50, split_available=True),
        accepted_cases=6, rejected_cases=4, raw_response="[]",
        cases=[_case(f"EP1_TC{i}") for i in range(1, 7)],
    )
    led.record_fallback(
        generator="LLM-X", variant="basic", operation_id="EP1", method="GET", path="/get",
        cases=[_case(f"EP1_TC{i}", fallback=True) for i in (7, 8, 9, 10)],
    )
    led.flush()

    summary = summarize_by_generator(CallLedger.load(led.path))[0]
    assert summary["accepted_cases"] == 6
    assert summary["rejected_cases"] == 4
    assert summary["acceptance_rate"] == 0.6      # 6 / (6+4)
    assert summary["fallback_cases"] == 4
    assert summary["total_cases"] == 10
    assert summary["fallback_share"] == 0.4       # 4 / 10
    assert summary["token_split_available"] is True


def test_missing_token_split_is_flagged_not_faked(tmp_path):
    """Saglayici ayrim vermediyse uydurma yapilmamali."""
    led = _ledger(tmp_path)
    led.record_call(
        generator="LLM-Y", variant="basic", operation_id="EP1", method="GET", path="/get",
        attempt=0, call_type="ana", usage=TokenUsage(total_tokens=900, split_available=False),
        accepted_cases=1, rejected_cases=0, raw_response="[]", cases=[_case("EP1_TC1")],
    )
    led.flush()

    summary = summarize_by_generator(CallLedger.load(led.path))[0]
    assert summary["token_split_available"] is False
    assert summary["input_tokens"] == 0
    assert summary["output_tokens"] == 0
    assert summary["total_tokens"] == 900


def test_call_types_are_counted_separately(tmp_path):
    led = _ledger(tmp_path)
    for call_type in ("ana", "repair", "repair", "retry"):
        led.record_call(
            generator="G", variant="basic", operation_id="EP1", method="GET", path="/get",
            attempt=0, call_type=call_type, usage=TokenUsage(total_tokens=1),
            accepted_cases=1, rejected_cases=0, raw_response="[]", cases=[],
        )
    led.flush()

    summary = summarize_by_generator(CallLedger.load(led.path))[0]
    assert summary["api_calls"] == 4
    assert summary["main_calls"] == 1
    assert summary["repair_calls"] == 2
    assert summary["retry_calls"] == 1


def test_disabled_ledger_writes_nothing(tmp_path):
    led = _ledger(tmp_path, enabled=False)
    led.record_call(
        generator="G", variant="basic", operation_id="EP1", method="GET", path="/get",
        attempt=0, call_type="ana", usage=TokenUsage(total_tokens=1),
        accepted_cases=1, rejected_cases=0, raw_response="[]", cases=[],
    )
    led.flush()
    assert not led.path.exists()


def test_each_call_is_flushed_immediately(tmp_path):
    """Cokme aninda onceki cagrilar diskte olmali."""
    led = _ledger(tmp_path)
    led.record_call(
        generator="G", variant="basic", operation_id="EP1", method="GET", path="/get",
        attempt=0, call_type="ana", usage=TokenUsage(total_tokens=1),
        accepted_cases=1, rejected_cases=0, raw_response="[]", cases=[],
    )
    # flush() CAGRILMADAN dosya diskte olmali
    assert led.path.is_file()
    assert len(CallLedger.load(led.path)) == 1


def test_ledger_totals_match_produced_rows_not_cumulative_counts(tmp_path):
    """Defterdeki kabul sayisi ARTIMSAL olmali.

    Kumulatif yazilirsa repair dongusundeki her cagri onceki kabulleri tekrar
    sayar ve fallback payi oldugundan DUSUK cikar.
    """
    import json as _json

    from generators.base import BaseGenerator

    class _Gen(BaseGenerator):
        def _generate_for_operation(self, op, variant_name, variant_desc, num_cases):
            return []

    from models import ApiOperation

    op = ApiOperation(op_id="EP1", method="GET", path="/get", response_schemas={"200": {}})
    gen = _Gen()
    gen._generation_summaries = []
    gen._call_ledger = CallLedger(str(tmp_path), run_id="run_inc")

    def _completion(prompt):
        case = {
            "tc_id": "EP1_TC1", "title": "tek case", "test_type": "positive", "priority": "P1",
            "request": {"path_params": {}, "query_params": {}, "headers": {}, "cookies": {}, "body": None},
            "expected": {"status": 200, "allowed_statuses": [200], "result": "ok",
                         "assertions": [{"type": "status_code", "expected": 200}],
                         "response_schema_check": False},
        }
        return _json.dumps([case]), TokenUsage(input_tokens=10, output_tokens=5, split_available=True)

    rows = gen._generate_cases_with_repair(
        op=op, variant_name="basic", variant_desc="d", num_cases=5,
        generator_name="LLM-Test", request_completion=_completion,
    )
    gen._call_ledger.flush()

    summary = summarize_by_generator(CallLedger.load(gen._call_ledger.path))[0]

    assert len(rows) == 5
    assert summary["total_cases"] == len(rows), "defter toplami uretilen satir sayisiyla ayni olmali"
    assert summary["accepted_cases"] == 1, "model tek gecerli case dondu"
    assert summary["fallback_cases"] == 4
    assert summary["fallback_share"] == 0.8


# ── B3 eki: dayaniklilik, basarisiz cagrilar, meta ────────────────────────

def test_ledger_write_failure_does_not_stop_the_run(tmp_path, caplog):
    """Defter yazilamazsa satirlar korunmali, kosu DURMAMALI."""
    import logging

    led = _ledger(tmp_path)

    def _explode(record):
        raise OSError("disk dolu")

    led._log.append = _explode

    with caplog.at_level(logging.ERROR):
        led.record_call(
            generator="G", variant="basic", operation_id="EP1", method="GET", path="/get",
            attempt=0, call_type="ana", usage=TokenUsage(total_tokens=5),
            accepted_cases=1, rejected_cases=0, raw_response="[]", cases=[],
        )

    assert len(led.write_errors) == 1
    assert "disk dolu" in led.write_errors[0]
    assert any("KAYIT YAZILAMADI" in r.getMessage() for r in caplog.records)


def test_failed_call_is_recorded_with_error_class_and_origin(tmp_path):
    class _RateLimited(RuntimeError):
        status_code = 429

    led = _ledger(tmp_path)
    error_class = led.record_failed_call(
        generator="LLM-X", variant="basic", operation_id="EP1", method="GET", path="/get",
        attempt=0, call_type="ana", exc=_RateLimited("rate limit exceeded"), latency_ms=1200,
    )
    led.flush()

    record = json.loads(led.path.read_text(encoding="utf-8").strip())
    assert error_class == "RATE_LIMIT"
    assert record["failed"] is True
    assert record["error_class"] == "RATE_LIMIT"
    assert record["failure_origin"] == "altyapi"
    assert record["latency_ms"] == 1200
    assert record["total_tokens"] == 0


def test_content_failure_is_marked_as_icerik_not_altyapi(tmp_path):
    from generators.base import ModelOutputFormatError

    led = _ledger(tmp_path)
    error_class = led.record_failed_call(
        generator="LLM-X", variant="basic", operation_id="EP1", method="GET", path="/get",
        attempt=1, call_type="repair", exc=ModelOutputFormatError("Model output format error"),
    )
    led.flush()

    record = json.loads(led.path.read_text(encoding="utf-8").strip())
    assert error_class == "MODEL_OUTPUT_FORMAT_ERROR"
    assert record["failure_origin"] == "icerik"


def test_repeat_index_latency_and_model_identity_are_recorded(tmp_path):
    led = _ledger(tmp_path)
    led.record_call(
        generator="LLM-X", variant="basic", operation_id="EP1", method="GET", path="/get",
        attempt=0, call_type="ana", usage=TokenUsage(total_tokens=10),
        accepted_cases=1, rejected_cases=0, raw_response="[]", cases=[],
        repeat_index=2, latency_ms=850,
        call_meta={
            "model_requested": "gpt-4.1",
            "model_returned": "gpt-4.1-2025-04-14",
            "response_id": "resp_123",
            "finish_reason": "stop",
            "sampling": {"max_completion_tokens": 4096},
        },
    )
    led.flush()

    record = json.loads(led.path.read_text(encoding="utf-8").strip())
    assert record["repeat_index"] == 2
    assert record["latency_ms"] == 850
    assert record["model_requested"] == "gpt-4.1"
    assert record["model_returned"] == "gpt-4.1-2025-04-14", "API yanitindaki surum kaydedilmeli"
    assert record["sampling"] == {"max_completion_tokens": 4096}
    assert record["ts"]


def test_repeat_index_defaults_to_zero(tmp_path):
    led = _ledger(tmp_path)
    led.record_call(
        generator="G", variant="basic", operation_id="EP1", method="GET", path="/get",
        attempt=0, call_type="ana", usage=TokenUsage(total_tokens=1),
        accepted_cases=0, rejected_cases=0, raw_response="[]", cases=[],
    )
    led.flush()
    assert json.loads(led.path.read_text(encoding="utf-8").strip())["repeat_index"] == 0


def test_resume_appends_to_existing_ledger_instead_of_overwriting(tmp_path):
    """Ayni run_id ile yeniden acilan defter mevcut kayitlarin USTUNE YAZMAMALI."""
    first = CallLedger(str(tmp_path), run_id="run_resume")
    first.record_call(
        generator="G", variant="basic", operation_id="EP1", method="GET", path="/get",
        attempt=0, call_type="ana", usage=TokenUsage(total_tokens=1),
        accepted_cases=1, rejected_cases=0, raw_response="ilk", cases=[],
    )
    first.flush()

    second = CallLedger(str(tmp_path), run_id="run_resume")   # resume
    second.record_call(
        generator="G", variant="basic", operation_id="EP2", method="GET", path="/get",
        attempt=0, call_type="ana", usage=TokenUsage(total_tokens=1),
        accepted_cases=1, rejected_cases=0, raw_response="ikinci", cases=[],
    )
    second.flush()

    records = CallLedger.load(second.path)
    assert len(records) == 2, "resume mevcut deftere EKLEMELI"
    assert [r["operation_id"] for r in records] == ["EP1", "EP2"]
