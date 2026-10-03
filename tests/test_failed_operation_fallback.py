"""Ana cagri tum denemelerde basarisiz olursa operasyon fallback'e dussun.

Olculen hata (3 Ekim, gemini-2.5-flash edge_focused): EP1/EP3/EP5 ana cagrilari
her denemede free tier 429'u aldi. _generate_for_operation_with_retry [] dondu;
operasyon 0 satir uretti, fallback'e ugramadi ve gorev 30/75 satirla
"tamamlandi" sayildi. Kayip 45 satir fallback payinda bile gorunmedi.
"""

import pytest

import config
from call_ledger import CallLedger
from generators import base
from generators.base import BaseGenerator, ModelOutputFormatError
from models import ApiOperation


def _op(op_id="EP1"):
    return ApiOperation(
        op_id=op_id, method="GET", path="/get", summary="GET /get",
        description="httpbin", response_schemas={"200": {"description": "OK"}},
    )


class _RateLimited(RuntimeError):
    status_code = 429


class _AlwaysFails(BaseGenerator):
    _provider_label = "Gemini"

    def __init__(self, exc, ledger=None):
        self.model = "gemini-2.5-flash"
        self._exc = exc
        self._call_ledger = ledger
        self.calls = 0
        self._generation_summaries = []

    def _generate_for_operation(self, op, variant_name, variant_desc, num_cases):
        self.calls += 1
        raise self._exc


@pytest.fixture(autouse=True)
def _fast_retry(monkeypatch):
    monkeypatch.setattr(base, "RETRY_BACKOFF_SECONDS", 0.0)
    monkeypatch.setattr(config, "RETRY_BACKOFF_SECONDS", 0.0)


def test_exhausted_retries_fall_back_instead_of_zero_rows(tmp_path):
    ledger = CallLedger(str(tmp_path), run_id="r")
    gen = _AlwaysFails(_RateLimited("429 RESOURCE_EXHAUSTED"), ledger)

    rows = gen._generate_for_operation_with_retry(_op(), "edge_focused", "desc", 15)

    assert gen.calls == base.RETRY_MAX_ATTEMPTS, "once tum denemeler yapilmali"
    assert len(rows) == 15, "operasyon 0 satirla kapanmamali"
    assert all((r.get("generation_metadata") or {}).get("fallback") for r in rows)
    assert {r["prompt_variant"] for r in rows} == {"edge_focused"}
    assert {r["generator"] for r in rows} == {"LLM-Gemini-gemini-2.5-flash"}
    assert len({r["tc_id"] for r in rows}) == 15

    summary = gen._generation_summaries[-1]
    assert summary["fallback_cases"] == 15
    assert summary["fallback_origin"] == "altyapi"
    assert summary["all_attempts_failed"] is True

    ledger.flush()
    fallback = [r for r in CallLedger.load(ledger.path) if r.get("call_type") == "fallback"]
    assert len(fallback) == 1
    assert fallback[0]["failure_origin"] == "altyapi"
    assert fallback[0]["accepted_cases"] == 15


def test_generate_counts_failed_operation_in_task_fallback_total():
    """main.py gorevin fallback_cases'ini summary'lerden toplar; kayip gorunmeli."""
    gen = _AlwaysFails(_RateLimited("429"))
    rows = gen.generate([_op("EP1"), _op("EP2")], "basic", "desc", 15)
    assert len(rows) == 30
    assert sum(s["fallback_cases"] for s in gen._generation_summaries) == 30


@pytest.mark.parametrize("exc", [
    RuntimeError("GEMINI_API_KEY environment variable is not set"),
    RuntimeError("Error code: 429 - insufficient_quota"),
    ModelOutputFormatError("Model output format error"),
])
def test_aborting_errors_still_return_nothing(exc):
    """Generator'i iptal eden hata fallback URETMEZ (eski davranis korunur)."""
    gen = _AlwaysFails(exc)
    rows = gen._generate_for_operation_with_retry(_op(), "basic", "desc", 15)
    assert rows == []
    assert gen._aborted is True
    assert gen.calls == 1
    assert gen._generation_summaries == []
