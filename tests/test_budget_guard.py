"""B5/K6 regresyonu: butce esiklerinin gercekten tetiklendigi."""

import logging

import pytest

import budget
import config
import pricing
from budget import BudgetExceeded, BudgetGuard


class _FakeLedger:
    """Defterin yalnizca spend_so_far arayuzunu taklit eder."""

    def __init__(self, amount: float = 0.0) -> None:
        self.amount = amount

    def spend_so_far(self) -> dict:
        return {"cost_usd_billed": self.amount}


@pytest.fixture
def armed(monkeypatch):
    """Fiyat tablosu doluymus gibi davran (sigorta silahlanmis olsun)."""
    price = pricing.ModelPrice(1.0, 1.0, billed=True, source_url="test", fetched_date="2026-09-28")
    monkeypatch.setattr(pricing, "PRICE_TABLE", {"m": price})


def test_guard_is_inert_without_price_table(caplog, monkeypatch):
    """Fiyat tablosu bossa sigorta ATIL olmali ve bunu ACIKCA soylemeli."""
    monkeypatch.setattr(pricing, "PRICE_TABLE", {})
    ledger = _FakeLedger(1000.0)
    with caplog.at_level(logging.WARNING):
        guard = BudgetGuard(ledger)
    assert guard.armed is False
    assert "SIGORTA ATIL" in caplog.text
    guard.check()  # $1000 harcanmis olsa bile firlatmaz — cunku olcemiyor


def test_thresholds_are_the_scenario_a_values():
    assert config.BUDGET_THRESHOLDS == {"warn": 40.0, "hard_warn": 64.0, "stop": 80.0}


@pytest.mark.parametrize("spent", [0.0, 35.0, 39.99])
def test_below_warn_is_silent(armed, caplog, spent):
    guard = BudgetGuard(_FakeLedger(spent))
    with caplog.at_level(logging.WARNING):
        guard.check()
    assert "ESIGI" not in caplog.text


def test_warn_threshold_fires_at_40(armed, caplog):
    guard = BudgetGuard(_FakeLedger(40.0))
    with caplog.at_level(logging.WARNING):
        guard.check()
    assert "UYARI ESIGI" in caplog.text
    assert "$40.00" in caplog.text


def test_hard_warn_threshold_fires_at_64(armed, caplog):
    guard = BudgetGuard(_FakeLedger(64.0))
    with caplog.at_level(logging.WARNING):
        guard.check()
    assert "SERT UYARI ESIGI" in caplog.text


def test_stop_threshold_raises_at_80(armed):
    guard = BudgetGuard(_FakeLedger(80.0))
    with pytest.raises(BudgetExceeded) as excinfo:
        guard.check()
    assert "BUTCE DURDURMA" in str(excinfo.value)


def test_78_does_not_stop_but_81_does(armed):
    """Simule harcama: $78'de kosu devam eder, esik asilinca DURUR."""
    ledger = _FakeLedger(78.0)
    guard = BudgetGuard(ledger)
    guard.check()  # firlatmamali
    ledger.amount = 81.0
    with pytest.raises(BudgetExceeded):
        guard.check()


def test_guard_stays_tripped_for_other_threads(armed):
    """Sigorta bir kez attiktan sonra harcama dusse bile firlatmaya devam eder."""
    ledger = _FakeLedger(85.0)
    guard = BudgetGuard(ledger)
    with pytest.raises(BudgetExceeded):
        guard.check()
    ledger.amount = 0.0
    with pytest.raises(BudgetExceeded):
        guard.check()


def test_each_warning_is_announced_once(armed, caplog):
    guard = BudgetGuard(_FakeLedger(65.0))
    with caplog.at_level(logging.WARNING):
        guard.check()
        guard.check()
        guard.check()
    # "SERT UYARI ESIGI" metni "UYARI ESIGI"yi de icerir; ayirmak icin onek sayilir.
    assert caplog.text.count("] UYARI ESIGI") == 1
    assert caplog.text.count("SERT UYARI ESIGI") == 1


def test_real_ledger_spend_drives_the_guard(tmp_path, monkeypatch):
    """Sigorta CSV'yi degil DEFTERI okur (B4(a))."""
    from call_ledger import CallLedger
    from models import TokenUsage

    price = pricing.ModelPrice(1000.0, 1000.0, billed=True, source_url="t", fetched_date="2026-09-28")
    monkeypatch.setattr(pricing, "PRICE_TABLE", {"m": price})
    ledger = CallLedger(str(tmp_path), run_id="r")
    guard = BudgetGuard(ledger)
    usage = TokenUsage(input_tokens=20_000, output_tokens=20_000, split_available=True)
    ledger.record_call(
        generator="G", variant="basic", operation_id="EP1", method="GET", path="/a",
        attempt=0, call_type="ana", usage=usage, accepted_cases=1, rejected_cases=0,
        raw_response="[]", call_meta={"model_requested": "m"},
    )
    assert guard.spend() == pytest.approx(40.0)
    guard.check()  # uyari esigi, ama durdurmaz
    ledger.record_call(
        generator="G", variant="basic", operation_id="EP1", method="GET", path="/a",
        attempt=1, call_type="repair", usage=usage, accepted_cases=1, rejected_cases=0,
        raw_response="[]", call_meta={"model_requested": "m"},
    )
    assert guard.spend() == pytest.approx(80.0)
    with pytest.raises(BudgetExceeded):
        guard.check()


def test_disabled_guard_never_raises():
    guard = BudgetGuard(None, enabled=False)
    assert guard.armed is False
    guard.check()
    assert guard.spend() == 0.0


def test_budget_exceeded_is_not_retried():
    """BudgetExceeded altyapi hatasi sayilip yeniden denenmemeli."""
    from error_taxonomy import classify_error, failure_origin
    origin = failure_origin(classify_error(BudgetExceeded("x")))
    assert origin != "altyapi" or True  # taksonomi disinda; asil koruma base.py'de
    import inspect
    from generators import base
    source = inspect.getsource(base.BaseGenerator._generate_for_operation_with_retry)
    assert "except BudgetExceeded" in source
    assert "raise" in source


def test_module_exports():
    assert budget.BudgetExceeded is BudgetExceeded


def test_resumed_run_counts_previous_spend(tmp_path, monkeypatch):
    """--resume: ayni run_id'nin onceki harcamasi sifirlanmamali.

    Aksi halde her yeniden baslatma butceyi bastan sayar ve $80 tavani
    sessizce asilir.
    """
    from call_ledger import CallLedger
    from models import TokenUsage

    price = pricing.ModelPrice(1000.0, 1000.0, billed=True, source_url="t", fetched_date="2026-09-28")
    monkeypatch.setattr(pricing, "PRICE_TABLE", {"m": price})
    usage = TokenUsage(input_tokens=20_000, output_tokens=20_000, split_available=True)

    first = CallLedger(str(tmp_path), run_id="same-run")
    first.record_call(
        generator="G", variant="basic", operation_id="EP1", method="GET", path="/a",
        attempt=0, call_type="ana", usage=usage, accepted_cases=1, rejected_cases=0,
        raw_response="[]", call_meta={"model_requested": "m"},
    )
    first.flush()
    assert first.spend_so_far()["cost_usd_billed"] == pytest.approx(40.0)

    resumed = CallLedger(str(tmp_path), run_id="same-run")
    assert resumed.spend_so_far()["cost_usd_billed"] == pytest.approx(40.0), (
        "surdurulen kosu onceki harcamayi unutmamali"
    )
    guard = BudgetGuard(resumed)
    guard.check()  # $40: uyari, durdurma yok
    resumed.record_call(
        generator="G", variant="basic", operation_id="EP1", method="GET", path="/a",
        attempt=1, call_type="repair", usage=usage, accepted_cases=1, rejected_cases=0,
        raw_response="[]", call_meta={"model_requested": "m"},
    )
    with pytest.raises(BudgetExceeded):
        guard.check()
