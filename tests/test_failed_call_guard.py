"""Bulgu 4 yan etkisi: basarisiz cagrinin guard tahmini None olmasin.

Olculen hata: _generate_cases_with_repair her cagridan once _last_call_meta'yi
None'a cekiyor (eski cagrinin kimligi tasinmasin diye). Istek yanit donmeden
patlayinca call_meta None kaliyor; record_failed_call model adini ve cikti
tavanini bilemiyor, pricing.guard_estimate("") None donuyor ve butce sigortasi
parasi odenmis OLABILECEK bu cagriyi hic gormuyordu.
"""

import json

import pytest

import config
import pricing
from budget import BudgetGuard
from call_ledger import CallLedger
from generators.base import BaseGenerator
from models import ApiOperation, TokenUsage

MODEL = "gemini-2.5-flash"


@pytest.fixture(autouse=True)
def _priced(monkeypatch):
    monkeypatch.setattr(
        pricing, "PRICE_TABLE",
        {MODEL: pricing.ModelPrice(1.0, 10.0, True, "u", "2026-10-03")},
    )
    monkeypatch.setattr(config, "RETRY_BACKOFF_SECONDS", 0.01)


def _failed(ledger, **kwargs):
    ledger.record_failed_call(
        generator="G", variant="basic", operation_id="EP1", method="GET", path="/a",
        attempt=0, call_type="ana", exc=TimeoutError("timeout"), prompt_chars=4000,
        **kwargs,
    )
    ledger.flush()
    return CallLedger.load(ledger.path)[-1]


def test_model_from_generator_gives_guard_when_call_meta_is_none(tmp_path):
    ledger = CallLedger(str(tmp_path), run_id="r")
    record = _failed(ledger, call_meta=None, model=MODEL, max_output_tokens=8192)

    expected = pricing.guard_estimate(MODEL, 4000, 8192)
    assert record["cost_usd_guard_estimate"] == pytest.approx(expected)
    assert record["cost_usd_guard_estimate"] > 0
    assert record["model_requested"] == MODEL
    assert record["cost_usd_billed"] is None, "fatura tahmini UYDURULMAZ"
    assert ledger.spend_so_far()["cost_usd_guard_estimate"] == pytest.approx(expected)


def test_call_meta_still_wins_over_generator_fallback(tmp_path):
    ledger = CallLedger(str(tmp_path), run_id="r")
    record = _failed(
        ledger,
        call_meta={"model_requested": MODEL, "sampling": {"max_output_tokens": 1000}},
        model="baska-model", max_output_tokens=8192,
    )
    assert record["model_requested"] == MODEL
    assert record["cost_usd_guard_estimate"] == pytest.approx(pricing.guard_estimate(MODEL, 4000, 1000))


def test_without_model_guard_stays_none(tmp_path):
    """Model gercekten bilinmiyorsa tahmin UYDURULMAZ (eski davranis korunur)."""
    ledger = CallLedger(str(tmp_path), run_id="r")
    record = _failed(ledger, call_meta=None)
    assert record["cost_usd_guard_estimate"] is None


def _op():
    return ApiOperation(
        op_id="EP1", method="POST", path="/post", summary="POST /post",
        description="httpbin", response_schemas={"200": {"description": "OK"}},
    )


def _case(index):
    return {
        "tc_id": f"EP1_TC{index}", "title": f"senaryo {index}",
        "test_type": "positive", "priority": "P1",
        "request": {"path_params": {}, "query_params": {}, "headers": {},
                    "cookies": {}, "body": {"f": index}},
        "expected": {"status": 200, "allowed_statuses": [200], "result": "ok",
                     "assertions": [{"type": "status_code", "expected": 200}],
                     "response_schema_check": False},
    }


class _Probe(BaseGenerator):
    """Ana cagri kismi basarili; repair cagrisi yanit donmeden patlar."""

    _provider_label = "Gemini"

    def __init__(self, ledger):
        self.model = MODEL
        self._call_ledger = ledger
        self._payload = json.dumps([_case(i) for i in range(1, 9)], ensure_ascii=False)
        self.calls = 0
        self.prompts = []
        self._generation_summaries = []

    def _generate_for_operation(self, op, variant_name, variant_desc, num_cases):
        def request_completion(prompt):
            self.calls += 1
            self.prompts.append(prompt)
            if self.calls == 1:
                return self._payload, TokenUsage(input_tokens=500, output_tokens=900, split_available=True)
            raise TimeoutError("read timeout")

        return self._generate_cases_with_repair(
            op=op, variant_name=variant_name, variant_desc=variant_desc,
            num_cases=num_cases, generator_name=f"LLM-Gemini-{self.model}",
            request_completion=request_completion,
        )


def test_generator_passes_model_so_budget_sees_failed_repair(tmp_path):
    ledger = CallLedger(str(tmp_path), run_id="r")
    gen = _Probe(ledger)

    gen._generate_for_operation(_op(), "basic", "desc", 15)
    ledger.flush()

    failed = [r for r in CallLedger.load(ledger.path) if r.get("failed")]
    assert len(failed) == 1
    record = failed[0]
    assert record["model_requested"] == MODEL
    expected = pricing.guard_estimate(MODEL, len(gen.prompts[1]), gen._max_tokens_for(15))
    assert record["cost_usd_guard_estimate"] == pytest.approx(expected)
    assert record["cost_usd_guard_estimate"] > 0

    guard = BudgetGuard(ledger)
    assert guard.spend_breakdown()["cost_usd_guard_estimate"] == pytest.approx(expected)
