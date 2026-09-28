"""B4 regresyonu: token ayrimi, satir basina token dagilimi ve maliyet muhasebesi."""

import pytest

import pricing
from call_ledger import CallLedger, total_spend
from generators.base import _apply_token_tracking
from models import TokenUsage


# ── K4: satir basina token ────────────────────────────────────────────────

def test_row_tokens_sum_to_operation_total():
    rows = [{} for _ in range(7)]
    _apply_token_tracking(rows, 1000)
    assert sum(row["tokens_used"] for row in rows) == 1000, "satir toplami gercek tuketimi vermeli"
    assert max(row["tokens_used"] for row in rows) - min(row["tokens_used"] for row in rows) <= 1


def test_single_row_keeps_full_total():
    """Smoke testi tek satir uretir; oradaki deger degismemeli."""
    rows = [{}]
    _apply_token_tracking(rows, 350)
    assert rows[0]["tokens_used"] == 350


def test_exact_division_distributes_evenly():
    rows = [{} for _ in range(10)]
    _apply_token_tracking(rows, 1000)
    assert [row["tokens_used"] for row in rows] == [100] * 10
    assert sum(row["tokens_used"] for row in rows) == 1000


def test_empty_rows_do_not_crash():
    _apply_token_tracking([], 500)  # bolme hatasi olmamali


# ── Reasoning / thinking token'lari ───────────────────────────────────────

def test_reasoning_inside_output_is_not_double_counted():
    """OpenAI/Groq: reasoning, completion_tokens'in ICINDE."""
    usage = TokenUsage(input_tokens=100, output_tokens=500, split_available=True,
                       reasoning_tokens=200, reasoning_included_in_output=True)
    assert usage.billable_output_tokens == 500


def test_thinking_outside_output_is_added():
    """Gemini: thoughts_token_count, candidates_token_count'un DISINDA."""
    usage = TokenUsage(input_tokens=100, output_tokens=500, total_tokens=800,
                       split_available=True, reasoning_tokens=200,
                       reasoning_included_in_output=False)
    assert usage.billable_output_tokens == 700


# ── Fiyatlandirma ─────────────────────────────────────────────────────────

def test_unknown_model_means_no_cost_is_invented():
    """Tabloda olmayan bir model icin maliyet UYDURULMAZ."""
    usage = TokenUsage(input_tokens=1000, output_tokens=1000, split_available=True)
    cost = pricing.cost_for("bilinmeyen-model-xyz", usage)
    assert cost.cost_usd_billed is None
    assert cost.pricing_available is False
    assert cost.cost_basis == pricing.BASIS_NO_PRICE


def test_missing_split_blocks_cost_even_with_price(monkeypatch):
    price = pricing.ModelPrice(2.0, 8.0, billed=True, source_url="x", fetched_date="2026-09-28")
    monkeypatch.setattr(pricing, "PRICE_TABLE", {"m": price})
    usage = TokenUsage(total_tokens=5000, split_available=False)
    cost = pricing.cost_for("m", usage)
    assert cost.cost_usd_billed is None, "ayrim yoksa maliyet UYDURULMAZ"
    assert cost.cost_basis == pricing.BASIS_NO_SPLIT


def test_billed_model_cost(monkeypatch):
    price = pricing.ModelPrice(2.0, 8.0, billed=True, source_url="x", fetched_date="2026-09-28")
    monkeypatch.setattr(pricing, "PRICE_TABLE", {"m": price})
    usage = TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000, split_available=True)
    cost = pricing.cost_for("m", usage)
    assert cost.cost_usd_billed == pytest.approx(10.0)
    assert cost.cost_usd_list_equivalent == pytest.approx(10.0)
    assert cost.cost_basis == pricing.BASIS_BILLED


def test_free_tier_is_zero_billed_but_keeps_list_equivalent(monkeypatch):
    price = pricing.ModelPrice(0.30, 2.50, billed=False, source_url="x", fetched_date="2026-09-28")
    monkeypatch.setattr(pricing, "PRICE_TABLE", {"m": price})
    usage = TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000, split_available=True)
    cost = pricing.cost_for("m", usage)
    assert cost.cost_usd_billed == 0.0
    assert cost.cost_usd_list_equivalent == pytest.approx(2.80)
    assert cost.cost_basis == pricing.BASIS_FREE_TIER


def test_thinking_tokens_are_priced_as_output(monkeypatch):
    price = pricing.ModelPrice(0.30, 2.50, billed=True, source_url="x", fetched_date="2026-09-28")
    monkeypatch.setattr(pricing, "PRICE_TABLE", {"m": price})
    usage = TokenUsage(input_tokens=0, output_tokens=0, total_tokens=1_000_000,
                       split_available=True, reasoning_tokens=1_000_000,
                       reasoning_included_in_output=False)
    cost = pricing.cost_for("m", usage)
    assert cost.cost_usd_billed == pytest.approx(2.50), "dusunme token'i cikti fiyatindan faturalanir"


# ── Defter toplami ────────────────────────────────────────────────────────

def _priced(model, cost, basis=pricing.BASIS_BILLED):
    return {"call_type": "ana", "model_requested": model, "cost_usd_billed": cost,
            "cost_usd_list_equivalent": cost, "cost_basis": basis, "pricing_available": True}


def test_total_spend_counts_every_record_including_revoked_generations():
    """Iptal edilen nesillerin parasi da harcanmistir; defter append-only oldugu
    icin bu kayitlar silinmez ve toplama DAHIL olmalidir."""
    records = [_priced("m", 1.0), _priced("m", 2.0), _priced("m", 4.0)]
    assert total_spend(records)["cost_usd_billed"] == pytest.approx(7.0)


def test_total_spend_flags_unpriced_failed_calls():
    records = [
        _priced("m", 1.0),
        {"call_type": "ana", "failed": True, "pricing_available": False,
         "cost_basis": "cagri_basarisiz_token_bildirilmedi"},
    ]
    spend = total_spend(records)
    assert spend["complete"] is False, "fiyatlanamayan cagri varsa toplam EKSIK sayilmali"
    assert spend["unpriced_calls"] == 1
    assert spend["unpriced_reasons"]["cagri_basarisiz_token_bildirilmedi"] == 1


def test_total_spend_excludes_fallback_records():
    records = [_priced("m", 1.0), {"call_type": "fallback", "pricing_available": False}]
    spend = total_spend(records)
    assert spend["cost_usd_billed"] == pytest.approx(1.0)
    assert spend["unpriced_calls"] == 0, "fallback saglayiciya cagri degil, fiyatlanamayan sayilmaz"


def test_ledger_live_spend_matches_written_records(tmp_path, monkeypatch):
    price = pricing.ModelPrice(2.0, 8.0, billed=True, source_url="x", fetched_date="2026-09-28")
    monkeypatch.setattr(pricing, "PRICE_TABLE", {"m": price})
    ledger = CallLedger(str(tmp_path), run_id="r1")
    usage = TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000, split_available=True)
    for _ in range(3):
        ledger.record_call(
            generator="LLM-X", variant="basic", operation_id="EP1", method="GET", path="/a",
            attempt=0, call_type="ana", usage=usage, accepted_cases=1, rejected_cases=0,
            raw_response="[]", call_meta={"model_requested": "m"},
        )
    ledger.flush()
    live = ledger.spend_so_far()
    disk = total_spend(CallLedger.load(ledger.path))
    assert live["cost_usd_billed"] == pytest.approx(30.0)
    assert disk["cost_usd_billed"] == pytest.approx(live["cost_usd_billed"])


# ── Fiyat tablosu kapsami ─────────────────────────────────────────────────

def test_every_registry_model_has_a_price_entry():
    """GENERATOR_REGISTRY'deki HER LLM modeli fiyat tablosunda olmali.

    Yeni bir model config'e eklenip fiyati yazilmazsa bu test KIRILIR; aksi
    halde model sessizce fiyatlanamayan olarak kosar ve butce sigortasi o
    generator'in harcamasini goremez.
    """
    from generators import GENERATOR_REGISTRY

    missing = sorted(
        model
        for _key, (_cls, model, _provider) in GENERATOR_REGISTRY.items()
        if model is not None and model not in pricing.PRICE_TABLE
    )
    assert not missing, f"fiyat tablosunda eksik model(ler): {missing}"


def test_price_table_has_no_stale_entries():
    """Tabloda registry'de karsiligi olmayan girdi birikmemeli."""
    from generators import GENERATOR_REGISTRY

    known = {model for _k, (_c, model, _p) in GENERATOR_REGISTRY.items() if model}
    extra = sorted(set(pricing.PRICE_TABLE) - known)
    assert not extra, f"registry'de olmayan fiyat girdisi: {extra}"


def test_every_price_entry_carries_source_and_date():
    for model, price in pricing.PRICE_TABLE.items():
        assert price.source_url.startswith("http"), f"{model}: kaynak linki yok"
        assert price.fetched_date, f"{model}: cekilme tarihi yok"
        assert price.input_usd_per_mtok > 0 and price.output_usd_per_mtok > 0, model


def test_paid_providers_are_marked_billed():
    """config.PAID_PROVIDERS ile tablodaki billed bayragi tutarli olmali."""
    import config
    from generators import GENERATOR_REGISTRY

    for _key, (_cls, model, provider) in GENERATOR_REGISTRY.items():
        if model is None:
            continue
        price = pricing.PRICE_TABLE[model]
        assert price.billed == (provider in config.PAID_PROVIDERS), (
            f"{model}: billed={price.billed} ama saglayici {provider}"
        )


def test_unverified_entries_are_reported():
    """'verified=False' isaretli girdi varsa bu test onu gorunur kilar."""
    unverified = sorted(m for m, p in pricing.PRICE_TABLE.items() if not p.verified)
    assert not unverified, f"DOGRULANMAMIS fiyat girdisi: {unverified}"
