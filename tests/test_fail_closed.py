"""Fail-closed regresyonu: olculemeyen harcama yapilmaz."""

import pytest

import main
import pricing
from budget import BudgetGuard
from generators.claude_gen import ClaudeGenerator
from generators.groq_gen import GroqGenerator
from generators.openai_gen import OpenAIGenerator


class _FakeLedger:
    def spend_so_far(self) -> dict:
        return {"cost_usd_billed": 0.0, "cost_usd_guard_estimate": 0.0}


def _gens(*instances):
    return [(gen, "basic", "x") for gen in instances]


@pytest.fixture
def armed_guard(monkeypatch):
    monkeypatch.setattr(
        pricing, "PRICE_TABLE",
        {"gpt-4.1": pricing.ModelPrice(2.0, 8.0, True, "u", "2026-09-28")},
    )
    return BudgetGuard(_FakeLedger())


def test_priced_paid_model_with_armed_guard_passes(armed_guard):
    assert main._fail_closed_check(_gens(OpenAIGenerator("gpt-4.1")), armed_guard) == []


def test_unpriced_paid_model_is_blocked(armed_guard):
    blockers = main._fail_closed_check(_gens(ClaudeGenerator("claude-yok-boyle-bir-model")), armed_guard)
    assert blockers, "fiyati bilinmeyen UCRETLI model kosmamali"
    assert "fiyat tablosunda girdisi olmayan" in blockers[0]
    assert "claude-yok-boyle-bir-model" in blockers[0]


def test_disarmed_guard_blocks_paid_models(monkeypatch):
    monkeypatch.setattr(pricing, "PRICE_TABLE", {})
    guard = BudgetGuard(_FakeLedger())
    assert guard.armed is False
    blockers = main._fail_closed_check(_gens(OpenAIGenerator("gpt-4.1")), guard)
    assert any("ATIL" in item for item in blockers)


def test_free_only_run_is_never_blocked(monkeypatch):
    """Ucretsiz saglayicilar fiyat tablosu bos olsa da kosabilir."""
    monkeypatch.setattr(pricing, "PRICE_TABLE", {})
    guard = BudgetGuard(_FakeLedger())
    assert main._fail_closed_check(_gens(GroqGenerator("openai/gpt-oss-20b")), guard) == []


def test_real_config_passes_fail_closed():
    """Uretimdeki gercek fiyat tablosu ve gercek model listesi engellenmemeli."""
    from generators import GENERATOR_REGISTRY
    import config

    instances = [
        cls(model)
        for _key, (cls, model, provider) in GENERATOR_REGISTRY.items()
        if model is not None and provider in config.PAID_PROVIDERS
    ]
    guard = BudgetGuard(_FakeLedger())
    assert guard.armed is True, "gercek fiyat tablosuyla sigorta silahli olmali"
    assert main._fail_closed_check(_gens(*instances), guard) == []


def test_fail_closed_exit_code_is_nonzero():
    assert main.EXIT_FAIL_CLOSED != 0


# ── Muhafazakar ust tahmin ────────────────────────────────────────────────

def test_guard_estimate_uses_max_tokens_not_actual(monkeypatch):
    monkeypatch.setattr(
        pricing, "PRICE_TABLE",
        {"m": pricing.ModelPrice(1.0, 10.0, True, "u", "2026-09-28")},
    )
    # 4000 karakter prompt -> 1000 token, x1.5 guvenlik = 1500 girdi token
    # 8000 cikti TAVANI (gercek uretim degil)
    estimate = pricing.guard_estimate("m", 4000, 8000)
    assert estimate == pytest.approx((1500 * 1.0 + 8000 * 10.0) / 1e6)


def test_guard_estimate_is_zero_for_free_tier(monkeypatch):
    monkeypatch.setattr(
        pricing, "PRICE_TABLE",
        {"m": pricing.ModelPrice(1.0, 10.0, False, "u", "2026-09-28")},
    )
    assert pricing.guard_estimate("m", 4000, 8000) == 0.0


def test_guard_estimate_is_none_for_unknown_model(monkeypatch):
    monkeypatch.setattr(pricing, "PRICE_TABLE", {})
    assert pricing.guard_estimate("m", 4000, 8000) is None


@pytest.mark.parametrize("sampling,expected", [
    ({"max_tokens": 2048}, 2048),
    ({"max_completion_tokens": 4096}, 4096),
    ({"max_output_tokens": 8192}, 8192),
    ({}, 0),
    (None, 0),
])
def test_max_output_tokens_is_found_across_providers(sampling, expected):
    assert pricing.max_output_tokens_from_sampling(sampling) == expected


def test_failed_call_gets_guard_estimate_not_zero(tmp_path, monkeypatch):
    """Yanit alinamayan cagri icin sifir varsayilmaz."""
    from call_ledger import CallLedger, total_spend

    monkeypatch.setattr(
        pricing, "PRICE_TABLE",
        {"m": pricing.ModelPrice(1.0, 10.0, True, "u", "2026-09-28")},
    )
    ledger = CallLedger(str(tmp_path), run_id="r")
    ledger.record_failed_call(
        generator="G", variant="basic", operation_id="EP1", method="GET", path="/a",
        attempt=0, call_type="ana", exc=TimeoutError("timeout"),
        call_meta={"model_requested": "m", "sampling": {"max_tokens": 8000}},
        prompt_chars=4000,
    )
    ledger.flush()
    record = CallLedger.load(ledger.path)[0]
    assert record["cost_usd_billed"] is None, "fatura tahmini UYDURULMAZ"
    assert record["cost_usd_guard_estimate"] > 0, "ust tahmin sifir olmamali"

    spend = total_spend(CallLedger.load(ledger.path))
    assert spend["cost_usd_billed"] == 0.0
    assert spend["cost_usd_guard_estimate"] == pytest.approx(record["cost_usd_guard_estimate"])
    assert spend["complete"] is False


def test_budget_guard_counts_guard_estimate(monkeypatch):
    """Sigorta esikleri faturalanan + ust tahmin toplamini olcer."""
    monkeypatch.setattr(
        pricing, "PRICE_TABLE",
        {"m": pricing.ModelPrice(1.0, 1.0, True, "u", "2026-09-28")},
    )

    class _L:
        def spend_so_far(self):
            return {"cost_usd_billed": 50.0, "cost_usd_guard_estimate": 35.0}

    guard = BudgetGuard(_L())
    assert guard.spend() == pytest.approx(85.0)
    assert guard.spend_breakdown() == {"cost_usd_billed": 50.0, "cost_usd_guard_estimate": 35.0}
    from budget import BudgetExceeded
    with pytest.raises(BudgetExceeded):
        guard.check()
