"""Bolum 4: ucretli saglayici hatalarinin siniflandirilmasi ve butce esikleri.

Hicbir gercek cagri yok: saglayicilarin GERCEK hata govdeleri taklit edilir.
Olcut su: bakiye ve kota tukenmesi 'altyapi' olmali (yeniden denenebilir,
modelin sucu degil); auth ve model-bulunamadi ise 'icerik' OLMAMALI (bunlar
konfigurasyon hatasidir, modelin kotu cikti uretmesi degil).
"""

import pytest

import budget
import config
from budget import BudgetGuard, resolve_thresholds
from error_taxonomy import classify_error, failure_origin


class _ApiStatusError(Exception):
    """openai/anthropic APIStatusError'un test karsiligi."""

    def __init__(self, message, status_code, code=None, headers=None):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.response = type("R", (), {"headers": headers or {}})()


# Saglayicilarin gercek govde metinlerine yakin ornekler.
CASES = [
    pytest.param(
        _ApiStatusError(
            "Error code: 429 - {'error': {'message': 'You exceeded your current quota, "
            "please check your plan and billing details.', 'type': 'insufficient_quota', "
            "'code': 'insufficient_quota'}}",
            429, code="insufficient_quota",
        ),
        "BILLING_QUOTA_ERROR", "altyapi", id="openai-429-insufficient_quota",
    ),
    pytest.param(
        _ApiStatusError(
            "Error code: 400 - {'type': 'error', 'error': {'type': 'invalid_request_error', "
            "'message': 'Your credit balance is too low to access the Anthropic API. "
            "Please go to Plans & Billing to upgrade or purchase credits.'}}",
            400,
        ),
        "BILLING_QUOTA_ERROR", "altyapi", id="anthropic-400-dusuk-kredi",
    ),
    pytest.param(
        _ApiStatusError(
            "Error code: 401 - {'error': {'message': 'Incorrect API key provided.', "
            "'type': 'invalid_request_error', 'code': 'invalid_api_key'}}",
            401,
        ),
        "AUTH_ERROR", "altyapi", id="401-gecersiz-anahtar",
    ),
    pytest.param(
        _ApiStatusError(
            "Error code: 403 - {'error': {'message': 'Your organization is not authorized "
            "to use this model.', 'type': 'permission_error'}}",
            403,
        ),
        "MODEL_ACCESS_ERROR", "altyapi", id="403-yetkisiz",
    ),
    pytest.param(
        _ApiStatusError(
            "Error code: 404 - {'error': {'message': \"The model 'x' does not exist or you "
            "do not have access to it.\", 'code': 'model_not_found'}}",
            404, code="model_not_found",
        ),
        "MODEL_NOT_FOUND", "altyapi", id="404-model-yok",
    ),
    pytest.param(
        _ApiStatusError(
            "Error code: 429 - {'error': {'message': 'Rate limit reached for gpt-4.1 in "
            "organization org-x on requests per min (RPM).', 'type': 'requests', "
            "'code': 'rate_limit_exceeded'}}",
            429, code="rate_limit_exceeded",
        ),
        "RATE_LIMIT", "altyapi", id="429-hiz-limiti",
    ),
]


@pytest.mark.parametrize("exc,expected_class,expected_origin", CASES)
def test_paid_provider_errors_are_classified(exc, expected_class, expected_origin):
    assert classify_error(exc) == expected_class
    assert failure_origin(classify_error(exc)) == expected_origin


@pytest.mark.parametrize("exc,expected_class,_origin", CASES)
def test_no_paid_provider_error_is_blamed_on_content(exc, expected_class, _origin):
    """Bunlarin hicbiri modelin kotu cikti uretmesi degildir."""
    assert failure_origin(classify_error(exc)) != "icerik"


def test_balance_error_beats_status_code_ordering():
    """Anthropic bakiye hatasi 400 doner; 400 -> REQUEST_CONTRACT_ERROR ('icerik')
    olsaydi bakiye tukenmesi modelin sucuymus gibi gorunurdu."""
    exc = _ApiStatusError("credit balance is too low", 400)
    assert classify_error(exc) == "BILLING_QUOTA_ERROR"
    assert failure_origin("REQUEST_CONTRACT_ERROR") == "icerik"  # karsilastirma icin
    assert failure_origin(classify_error(exc)) == "altyapi"


def test_quota_error_beats_rate_limit_ordering():
    """OpenAI kota tukenmesi de 429 doner; RATE_LIMIT'e dusseydi ayni oturumda
    bosuna yeniden denenirdi."""
    exc = _ApiStatusError("429 insufficient_quota", 429, code="insufficient_quota")
    assert classify_error(exc) == "BILLING_QUOTA_ERROR"


def test_billing_error_is_not_retryable_by_generator():
    """Bakiye/kota hatasi generator'i iptal etmeli; yeniden denemek kredi getirmez."""
    from generators.base import _is_non_retryable_generation_error

    # Gercek OpenAI govdesi 'insufficient_quota' metnini ICERIR.
    assert _is_non_retryable_generation_error(CASES[0].values[0])
    # Gercek Anthropic govdesi 'credit balance is too low' metnini icerir.
    assert _is_non_retryable_generation_error(CASES[1].values[0])


def test_known_gap_retryability_reads_only_the_message():
    """BILINEN ASIMETRI (kod degistirilmedi, raporlandi).

    classify_error, kodu da kapsayan birlesik metne bakar
    (f"{code} {name} {message}"); _is_non_retryable_generation_error ise
    YALNIZCA str(exc)'e bakar. Bir saglayici kota sinyalini sadece `.code`
    alaninda verirse siniflandirma dogru olur (BILLING_QUOTA_ERROR) ama
    generator bosuna yeniden dener.

    Pratikte iki saglayicinin da govdesi metni icerdigi icin bu yol
    tetiklenmiyor; test durumu gorunur kilmak icin var.
    """
    from generators.base import _is_non_retryable_generation_error

    only_in_code = _ApiStatusError("429 Too Many Requests", 429, code="insufficient_quota")
    assert classify_error(only_in_code) == "BILLING_QUOTA_ERROR", "siniflandirma dogru"
    assert not _is_non_retryable_generation_error(only_in_code), (
        "bilinen asimetri: yeniden denenebilir sayiliyor"
    )


# ── 4b: defter redakte edilmis ham hatayi saklıyor mu ────────────────────

def test_ledger_stores_redacted_raw_error(tmp_path):
    from call_ledger import CallLedger

    leaky = _ApiStatusError(
        "Error code: 401 - Incorrect API key provided: sk-proj-" + "A" * 48, 401
    )
    ledger = CallLedger(str(tmp_path), run_id="r")
    ledger.record_failed_call(
        generator="LLM-OpenAI-gpt-4.1", variant="basic", operation_id="EP1",
        method="GET", path="/a", attempt=0, call_type="ana", exc=leaky,
        provider_label="OpenAI",
    )
    ledger.flush()
    record = CallLedger.load(ledger.path)[0]
    assert record["error"], "ham hata mesaji saklanmali"
    assert "Incorrect API key provided" in record["error"]
    assert "sk-proj-" + "A" * 48 not in record["error"], "anahtar REDAKTE edilmeli"
    assert record["error_class"] == "AUTH_ERROR"


# ── 4d: ayarlanabilir butce esikleri ─────────────────────────────────────

def test_defaults_are_40_64_80():
    assert resolve_thresholds() == {"warn": 40.0, "hard_warn": 64.0, "stop": 80.0}


def test_env_overrides_defaults(monkeypatch):
    monkeypatch.setenv("BUDGET_WARN_USD", "5")
    monkeypatch.setenv("BUDGET_HARD_WARN_USD", "12")
    monkeypatch.setenv("BUDGET_STOP_USD", "20")
    effective = resolve_thresholds()
    assert effective == {"warn": 5.0, "hard_warn": 12.0, "stop": 20.0}
    assert config.BUDGET_THRESHOLDS == {"warn": 40.0, "hard_warn": 64.0, "stop": 80.0}, (
        "config varsayilani degistirilmemeli"
    )


def test_cli_overrides_env(monkeypatch):
    monkeypatch.setenv("BUDGET_STOP_USD", "20")
    monkeypatch.setenv("BUDGET_WARN_USD", "5")
    monkeypatch.setenv("BUDGET_HARD_WARN_USD", "12")
    assert resolve_thresholds({"stop": 30.0})["stop"] == 30.0, "komut satiri env'i ezer"


def test_thresholds_must_be_ascending():
    with pytest.raises(ValueError, match="artan sirada"):
        resolve_thresholds({"warn": 90.0})


def test_non_numeric_env_is_refused(monkeypatch):
    monkeypatch.setenv("BUDGET_STOP_USD", "cok")
    with pytest.raises(ValueError, match="sayisal olmali"):
        resolve_thresholds()


def test_guard_uses_the_resolved_thresholds(monkeypatch):
    import pricing

    monkeypatch.setattr(
        pricing, "PRICE_TABLE",
        {"m": pricing.ModelPrice(1.0, 1.0, True, "u", "2026-09-29")},
    )

    class _L:
        def spend_so_far(self):
            return {"cost_usd_billed": 12.0, "cost_usd_guard_estimate": 0.0}

    guard = BudgetGuard(_L(), thresholds=resolve_thresholds({"warn": 5.0, "hard_warn": 8.0, "stop": 10.0}))
    with pytest.raises(budget.BudgetExceeded):
        guard.check()


# ── Kendi istisnalarimiz sinifini ACIKCA bildirmeli ──────────────────────

def test_our_own_exceptions_declare_their_class():
    """Sinif ADINA bakan sezgisel eslesme bizim istisnalarimizda yanlis sonuc verdi.

    QuotaExhausted'in adinda "quota" gectigi icin BILLING_QUOTA_ERROR (fatura
    sorunu) sayiliyordu; ImpossibleRequest ise hicbir sezgiye uymadigi icin
    UNKNOWN_ERROR / 'bilinmiyor' olarak kaydediliyordu. Ikisi de error_class
    bildirir ve 'altyapi' kokenine duser.
    """
    import rate_limiter

    cases = [
        (rate_limiter.QuotaExhausted("kota", "2026-10-03T17:00:00+03:00", 86400.0),
         "QUOTA_EXHAUSTED"),
        (rate_limiter.ImpossibleRequest("tek istek 6655 token rezerve ediyor"),
         "REQUEST_EXCEEDS_LIMIT"),
    ]
    for exc, expected in cases:
        assert classify_error(exc) == expected, type(exc).__name__
        assert failure_origin(classify_error(exc)) == "altyapi", type(exc).__name__


def test_declared_class_must_be_a_known_infrastructure_class():
    """Bildirilen sinif taksonomide tanimli olmali; yoksa koken 'bilinmiyor' kalir."""
    from error_taxonomy import INFRASTRUCTURE_ERRORS

    for name in ("QUOTA_EXHAUSTED", "REQUEST_EXCEEDS_LIMIT"):
        assert name in INFRASTRUCTURE_ERRORS, name


def test_declared_class_wins_over_name_heuristic():
    class _Weird(Exception):
        error_class = "QUOTA_EXHAUSTED"

    exc = _Weird("timeout rate limit quota 429")   # her sezgiyi tetikleyecek metin
    assert classify_error(exc) == "QUOTA_EXHAUSTED"
