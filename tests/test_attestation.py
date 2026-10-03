"""Bolum 2: FREE_ONLY garantisi — beyan kapisi ve fatura kilidi."""

import pytest

import attestation
import config
import main
import pricing
from models import TokenUsage


def _gens(*instances):
    return [(gen, "basic", "x") for gen in instances]


# ── Saglayici listeleri ───────────────────────────────────────────────────

def test_free_only_and_paid_do_not_overlap():
    assert not (config.PAID_PROVIDERS & config.FREE_ONLY_PROVIDERS)


def test_every_registry_provider_is_classified():
    """Her LLM saglayicisi ya ucretli ya yalnizca-ucretsiz olmali; arada kalan olmasin."""
    from generators import GENERATOR_REGISTRY

    known = config.PAID_PROVIDERS | config.FREE_ONLY_PROVIDERS
    unclassified = sorted({
        provider
        for _k, (_c, model, provider) in GENERATOR_REGISTRY.items()
        if model is not None and provider not in known
    })
    assert not unclassified, f"siniflandirilmamis saglayici: {unclassified}"


# 3 Ekim 2026: Gemini projesine faturalandirma acildi -> artik FREE_ONLY DEGIL.
@pytest.mark.parametrize("model,expected", [
    ("openai/gpt-oss-120b", True),
    ("openai/gpt-oss-20b", True),
    ("gemini-2.5-flash", False),
    ("gemini-3.5-flash-lite", False),
    ("gpt-4.1", False),
    ("claude-sonnet-4-5", False),
    ("bilinmeyen", False),
])
def test_is_free_only_model(model, expected):
    assert config.is_free_only_model(model) is expected


# ── 2a: fatura tutari asla hesaplanmasin ──────────────────────────────────

@pytest.mark.parametrize("model", ["openai/gpt-oss-120b", "openai/gpt-oss-20b"])
def test_free_only_model_is_never_billed(model):
    usage = TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000, split_available=True)
    cost = pricing.cost_for(model, usage)
    assert cost.cost_usd_billed == 0.0
    assert cost.cost_basis == pricing.BASIS_FREE_TIER
    assert cost.cost_usd_list_equivalent > 0, "liste-esdegeri yine de hesaplanmali"


def test_free_only_lock_survives_a_wrong_price_table_entry(monkeypatch):
    """Fiyat tablosuna yanlislikla billed=True yazilsa bile fatura cikmamali."""
    wrong = pricing.ModelPrice(0.15, 0.60, billed=True, source_url="x", fetched_date="2026-09-29")
    monkeypatch.setitem(pricing.PRICE_TABLE, "openai/gpt-oss-120b", wrong)
    usage = TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000, split_available=True)
    cost = pricing.cost_for("openai/gpt-oss-120b", usage)
    assert cost.cost_usd_billed == 0.0, "FREE_ONLY kilidi tabloyu gecersiz kilmali"
    assert cost.cost_basis == pricing.BASIS_FREE_TIER


def test_gemini_is_billed_after_moving_to_paid_tier():
    """3 Ekim 2026: Gemini ucretli tier'a gecti; fatura tutari HESAPLANMALI.

    Gemini FREE_ONLY listesinde kalsaydi her cagriya cost_usd_billed=0.0
    yazilirdi, yani gercek para harcanirken defter $0.00 gosterir ve butce
    sigortasi bu harcamayi hic gormezdi.
    """
    usage = TokenUsage(input_tokens=592, output_tokens=3032, split_available=True)
    for model in ("gemini-2.5-flash", "gemini-3.5-flash-lite"):
        cost = pricing.cost_for(model, usage)
        assert cost.cost_basis == pricing.BASIS_BILLED, model
        assert cost.cost_usd_billed > 0, model
        assert cost.cost_usd_billed == cost.cost_usd_list_equivalent, model


def test_free_only_guard_estimate_is_zero(monkeypatch):
    wrong = pricing.ModelPrice(0.075, 0.30, billed=True, source_url="x", fetched_date="2026-09-29")
    monkeypatch.setitem(pricing.PRICE_TABLE, "openai/gpt-oss-20b", wrong)
    assert pricing.guard_estimate("openai/gpt-oss-20b", 4000, 8000) == 0.0


# ── 2b: beyan yoksa kosma ─────────────────────────────────────────────────

def test_no_attestation_means_no_free_only_run(monkeypatch):
    from generators.gemini_gen import GeminiGenerator
    from generators.groq_gen import GroqGenerator

    monkeypatch.delenv(attestation.ENV_VAR, raising=False)
    blockers = main._free_tier_attestation_check(
        _gens(GeminiGenerator("gemini-2.5-flash"), GroqGenerator("openai/gpt-oss-20b"))
    )
    assert blockers, "beyan yokken FREE_ONLY generator kosmamali"
    assert "Gemini" in blockers[0] and "Groq" in blockers[0]


def test_message_names_both_provider_checks(monkeypatch):
    """Rehber metni sartnameden BIREBIR; iki saglayiciyi da anar."""
    from generators.groq_gen import GroqGenerator

    monkeypatch.delenv(attestation.ENV_VAR, raising=False)
    blockers = main._free_tier_attestation_check(_gens(GroqGenerator("openai/gpt-oss-20b")))
    assert "faturalama bağlı mı" in blockers[0]
    assert "plan Free ve ödeme yöntemi yok mu" in blockers[0]


def test_wrong_provider_attestation_does_not_satisfy_groq(monkeypatch):
    """Yalnizca 'gemini' beyan edilmesi Groq'u KARSILAMAZ.

    (3 Ekim 2026 oncesi bu test iki FREE_ONLY saglayicinin kismi beyanini
    sinardi; Gemini ucretli tier'a gectigi icin senaryo tek saglayiciya indi.)
    """
    from generators.groq_gen import GroqGenerator

    monkeypatch.setenv(attestation.ENV_VAR, "gemini")
    blockers = main._free_tier_attestation_check(_gens(GroqGenerator("openai/gpt-oss-20b")))
    assert blockers
    missing_part = blockers[0].split("(")[0]
    assert "Groq" in missing_part


def test_gemini_attestation_is_no_longer_required(monkeypatch):
    """Gemini ucretli oldugu icin beyan kapisi ona bakmaz."""
    from generators.gemini_gen import GeminiGenerator

    monkeypatch.delenv(attestation.ENV_VAR, raising=False)
    assert main._free_tier_attestation_check(_gens(GeminiGenerator("gemini-2.5-flash"))) == []


def test_full_attestation_passes(monkeypatch):
    from generators.groq_gen import GroqGenerator

    monkeypatch.setenv(attestation.ENV_VAR, "gemini,groq")
    assert main._free_tier_attestation_check(
        _gens(GroqGenerator("openai/gpt-oss-120b"), GroqGenerator("openai/gpt-oss-20b"))
    ) == []


def test_paid_only_run_needs_no_attestation(monkeypatch):
    from generators.openai_gen import OpenAIGenerator

    monkeypatch.delenv(attestation.ENV_VAR, raising=False)
    assert main._free_tier_attestation_check(_gens(OpenAIGenerator("gpt-4.1"))) == []


def test_exit_code_is_nonzero_and_distinct():
    assert main.EXIT_FREE_TIER_UNATTESTED != 0
    assert len({main.EXIT_FAIL_CLOSED, main.EXIT_FREE_TIER_UNATTESTED,
                main.EXIT_MISSING_RATE_LIMITS}) == 3


@pytest.mark.parametrize("value,expected", [
    ("gemini,groq", {"Gemini", "Groq"}),
    (" GEMINI ,  Groq ", {"Gemini", "Groq"}),
    ("gemini;groq", {"Gemini", "Groq"}),
    ("gemini", {"Gemini"}),
    ("", set()),
    ("openai,claude", set()),
])
def test_attestation_parsing(monkeypatch, value, expected):
    monkeypatch.setenv(attestation.ENV_VAR, value)
    assert attestation.attested_providers() == expected


def test_attestation_record_has_no_secrets(monkeypatch):
    monkeypatch.setenv(attestation.ENV_VAR, "gemini,groq")
    record = attestation.attestation_record()
    assert record["declared_providers"] == ["Gemini", "Groq"]
    assert record["read_at"]
    blob = str(record).lower()
    for forbidden in ("key", "secret", "token", "aiza", "gsk_"):
        assert forbidden not in blob


# ── 2e: ucretli anahtara/plana gecis yolu yok ─────────────────────────────

def test_free_only_generators_read_only_their_own_env_var():
    """Groq/Gemini baska bir saglayicinin anahtarina duserek kosamamali."""
    from generators.gemini_gen import GeminiGenerator
    from generators.groq_gen import GroqGenerator

    assert GroqGenerator._api_key_env == "GROQ_API_KEY"
    assert GroqGenerator._base_url == "https://api.groq.com/openai/v1"
    # Gemini kendi yukleyicisini kullanir; get_api_key("gemini") tek kaynaktan okur.
    import inspect
    source = inspect.getsource(GeminiGenerator._get_client)
    assert 'get_api_key("gemini")' in source
    assert "OPENAI_API_KEY" not in source and "ANTHROPIC_API_KEY" not in source


def test_no_paid_tier_escalation_knobs_in_code():
    """Ucretli katman/servis seviyesi ayarlari kodda bulunmamali (2c/2e)."""
    import pathlib

    forbidden = ("service_tier", "cached_content", "google_search", "tool_config")
    root = pathlib.Path(__file__).resolve().parents[1]
    hits = []
    for path in list(root.glob("*.py")) + list((root / "generators").glob("*.py")):
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            if token in text:
                hits.append(f"{path.name}:{token}")
    assert not hits, f"ucretli ozellik izi: {hits}"
