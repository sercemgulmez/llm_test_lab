"""K1 birim testleri: --tests-per-generator tahsis matematigi."""

import pytest

import main
from models import ApiOperation

VARIANTS = ["basic", "edge_focused"]


def _ops(count: int = 5):
    return [
        ApiOperation(op_id=f"EP{i}", method="GET", path=f"/p{i}")
        for i in range(1, count + 1)
    ]


# ── T = 150: Senaryo A'nin hedef konfigurasyonu ───────────────────────────

def test_t150_gives_llm_15_per_op_per_variant_and_traditional_30_per_op():
    allocation = main._build_balanced_allocation(150, _ops(5), VARIANTS)

    assert allocation["traditional"] == {f"EP{i}": 30 for i in range(1, 6)}
    for variant in VARIANTS:
        assert allocation["llm"][variant] == {f"EP{i}": 15 for i in range(1, 6)}


def test_t150_totals_reach_exactly_150_per_generator_label():
    allocation = main._build_balanced_allocation(150, _ops(5), VARIANTS)

    traditional_total = sum(allocation["traditional"].values())
    llm_model_total = sum(sum(counts.values()) for counts in allocation["llm"].values())

    assert traditional_total == 150
    assert llm_model_total == 150, "iki variant birlikte tek generator etiketi olusturur"


def test_t150_grand_total_is_1350_for_nine_generators():
    allocation = main._build_balanced_allocation(150, _ops(5), VARIANTS)
    llm_model_total = sum(sum(c.values()) for c in allocation["llm"].values())
    traditional_total = sum(allocation["traditional"].values())

    assert 8 * llm_model_total + traditional_total == 1350


# ── T = 151: tam bolunmeyen durum ─────────────────────────────────────────

def test_t151_total_is_exactly_151_despite_uneven_split():
    allocation = main._build_balanced_allocation(151, _ops(5), VARIANTS)

    assert sum(allocation["traditional"].values()) == 151
    assert sum(sum(c.values()) for c in allocation["llm"].values()) == 151


def test_t151_remainder_goes_to_first_operations():
    allocation = main._build_balanced_allocation(151, _ops(5), VARIANTS)

    # 151 / 5 = 30 kalan 1 -> ilk operasyon 31, digerleri 30
    assert allocation["traditional"] == {"EP1": 31, "EP2": 30, "EP3": 30, "EP4": 30, "EP5": 30}
    # 151 -> variantlara 76 / 75; 76/5 = 15 kalan 1, 75/5 = 15 kalan 0
    assert allocation["llm"]["basic"] == {"EP1": 16, "EP2": 15, "EP3": 15, "EP4": 15, "EP5": 15}
    assert allocation["llm"]["edge_focused"] == {f"EP{i}": 15 for i in range(1, 6)}


def test_no_operation_is_starved_when_total_is_uneven():
    allocation = main._build_balanced_allocation(151, _ops(5), VARIANTS)
    for counts in [allocation["traditional"], *allocation["llm"].values()]:
        assert all(value >= 1 for value in counts.values())


# ── T = 5: dagitilamaz, DURMALI ───────────────────────────────────────────

def test_t5_raises_because_it_cannot_be_distributed():
    with pytest.raises(ValueError) as exc_info:
        main._build_balanced_allocation(5, _ops(5), VARIANTS)

    message = str(exc_info.value)
    assert "cok dusuk" in message
    assert "en az 10" in message, "gereken asgari deger mesajda yer almali"


def test_stop_is_silent_free_no_partial_allocation_returned():
    """Sessiz dengesizlik olmamali: hata firlatilir, yarim tahsis donmez."""
    with pytest.raises(ValueError):
        main._build_balanced_allocation(9, _ops(5), VARIANTS)


# ── Yardimci fonksiyonlar ─────────────────────────────────────────────────

@pytest.mark.parametrize("total,parts,expected", [
    (150, 2, [75, 75]),
    (151, 2, [76, 75]),
    (10, 3, [4, 3, 3]),
    (7, 7, [1] * 7),
])
def test_split_total_preserves_sum(total, parts, expected):
    result = main._split_total(total, parts)
    assert result == expected
    assert sum(result) == total


def test_single_variant_run_still_reaches_target():
    """--prompt-variant basic ile tek variant kosulursa hedef yine tam tutmali."""
    allocation = main._build_balanced_allocation(150, _ops(5), ["basic"])

    assert sum(allocation["llm"]["basic"].values()) == 150
    assert allocation["llm"]["basic"] == {f"EP{i}": 30 for i in range(1, 6)}
