"""Bulgu 6: repair cagrisi patlayinca ANA CAGRININ cikisi cope gitmesin.

Olculen hata: _generate_cases_with_repair icindeki deneme dongusunde cagri
hatasi kosulsuz `raise` ediyordu. Istisna fonksiyondan cikinca dongunun
ASAGISINDAKI fallback blogu ve `return final_rows` hic calismiyor, yani ana
cagrinin zaten kabul edilmis satirlari kayboluyordu.

Gercek olcum (2 Ekim, Groq): defterde 206 kabul edilmis case varken CSV'ye
yalnizca 15 satir girdi. Ucretli fazda bu, parasi odenmis ciktinin atilmasidir.
"""

import pytest

import config
from generators.base import BaseGenerator
from models import ApiOperation, TokenUsage
from rate_limiter import ImpossibleRequest


def _op(op_id="EP1"):
    return ApiOperation(
        op_id=op_id, method="POST", path="/post", summary="POST /post",
        description="httpbin", response_schemas={"200": {"description": "OK"}},
    )


def _case(op_id, index, status=200):
    return {
        "tc_id": f"{op_id}_TC{index}", "title": f"senaryo {index}",
        "test_type": "positive" if status < 400 else "negative", "priority": "P1",
        "request": {"path_params": {}, "query_params": {}, "headers": {},
                    "cookies": {}, "body": {"f": index}},
        "expected": {"status": status, "allowed_statuses": [status], "result": "ok",
                     "assertions": [{"type": "status_code", "expected": status}],
                     "response_schema_check": False},
    }


class _Probe(BaseGenerator):
    """Ilk cagri KISMI basarili, repair cagrisi patlar."""

    _provider_label = "Groq"

    def __init__(self, first_batch, repair_exc):
        import json
        self.model = "openai/gpt-oss-20b"
        self._payload = json.dumps(first_batch, ensure_ascii=False)
        self._repair_exc = repair_exc
        self.calls = 0
        self._generation_summaries = []

    def _generate_for_operation(self, op, variant_name, variant_desc, num_cases):
        def request_completion(prompt):
            self.calls += 1
            if self.calls == 1:
                return self._payload, TokenUsage(
                    input_tokens=500, output_tokens=900, split_available=True
                )
            raise self._repair_exc

        return self._generate_cases_with_repair(
            op=op, variant_name=variant_name, variant_desc=variant_desc,
            num_cases=num_cases, generator_name=f"LLM-Groq-{self.model}",
            request_completion=request_completion,
        )


@pytest.fixture(autouse=True)
def _fast_retry(monkeypatch):
    monkeypatch.setattr(config, "RETRY_BACKOFF_SECONDS", 0.01)


def test_main_call_output_survives_a_failed_repair():
    """8 case gelmis, repair patliyor -> 8 satir KORUNUR, eksik fallback'le dolar."""
    op = _op()
    first = [_case("EP1", i) for i in range(1, 9)]          # 8 case, 15 istendi
    gen = _Probe(first, ImpossibleRequest("tek istek 7086 token rezerve ediyor"))

    rows = gen._generate_for_operation(op, "basic", "desc", 15)

    assert len(rows) == 15, "eksik satirlar fallback ile tamamlanmali"
    llm_rows = [r for r in rows if not (r.get("generation_metadata") or {}).get("fallback")]
    assert len(llm_rows) == 8, f"ana cagrinin 8 satiri KORUNMALI, gelen {len(llm_rows)}"
    assert gen.calls == 2, "bir ana + bir repair cagrisi"


def test_failure_is_still_marked_as_infrastructure():
    """Veri korunuyor ama hata GIZLENMIYOR: altyapi kokeni isaretlenir."""
    op = _op()
    gen = _Probe([_case("EP1", i) for i in range(1, 9)],
                 ImpossibleRequest("tek istek 7086 token"))
    gen._generate_for_operation(op, "basic", "desc", 15)

    summary = gen._generation_summaries[0]
    assert summary["fallback_cases"] == 7
    assert summary["fallback_origin"] == "altyapi", (
        "ImpossibleRequest altyapi kaynaklidir; fallback kokeni boyle isaretlenmeli"
    )


def test_first_attempt_failure_still_raises():
    """Ilk denemede elde hicbir sey yok: yukari tasinmali ki generator retry etsin."""
    class _AlwaysFails(_Probe):
        def _generate_for_operation(self, op, variant_name, variant_desc, num_cases):
            def request_completion(prompt):
                self.calls += 1
                raise ImpossibleRequest("tek istek cok buyuk")
            return self._generate_cases_with_repair(
                op=op, variant_name=variant_name, variant_desc=variant_desc,
                num_cases=num_cases, generator_name="LLM-Groq-m",
                request_completion=request_completion,
            )

    gen = _AlwaysFails([], ImpossibleRequest("x"))
    with pytest.raises(ImpossibleRequest):
        gen._generate_for_operation(_op(), "basic", "desc", 15)
    assert gen.calls == 1, "ilk deneme patlayinca repair denenmemeli"


@pytest.mark.parametrize("exc", [
    ImpossibleRequest("tek istek cok buyuk"),
    TimeoutError("read timeout"),
    RuntimeError("Error code: 429 - rate limit exceeded"),
])
def test_any_later_failure_preserves_rows(exc):
    """Sadece ImpossibleRequest degil; 429 / timeout da veriyi atmamali."""
    gen = _Probe([_case("EP1", i) for i in range(1, 11)], exc)
    rows = gen._generate_for_operation(_op(), "basic", "desc", 15)
    llm_rows = [r for r in rows if not (r.get("generation_metadata") or {}).get("fallback")]
    assert len(llm_rows) == 10, f"{type(exc).__name__}: 10 satir korunmali"


def test_no_rows_means_no_silent_success():
    """Ana cagri 0 gecerli case verip repair de patlarsa yukari tasinir."""
    gen = _Probe([], ImpossibleRequest("tek istek cok buyuk"))
    with pytest.raises(ImpossibleRequest):
        gen._generate_for_operation(_op(), "basic", "desc", 15)
