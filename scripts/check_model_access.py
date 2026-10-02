"""Observable, minimal real-API access check for every configured LLM model.

PARA HARCAMA KURALI (varsayilan davranis):
  * UCRETSIZ saglayicilar (Gemini, Groq) — tek, kucuk bir URETIM cagrisi
    (`smoke_test`). Free tier'da fatura 0.
  * UCRETLI saglayicilar (config.PAID_PROVIDERS: OpenAI, Claude) — YALNIZCA
    model METADATA cagrisi (GET /models/{id}). Token uretilmez, fatura 0.

`--paid-generation` bayragi acikca verilmedikce ucretli saglayiciya HICBIR
uretim cagrisi yapilmaz. Bayrak verilse bile onay istenir.

Cikti hicbir zaman API anahtarini, bakiye veya kota bilgisini yazdirmaz;
yalnizca saglayici, model, durum ve redakte edilmis hata sinifi basilir.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import argparse
import os
from pathlib import Path
import sys

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import attestation
import config
import rate_limiter
from error_taxonomy import classify_error
from generators import GENERATOR_REGISTRY
from security.redaction import redact_secrets

EXPECTED_MODELS = 8
ENV_BY_PROVIDER = {
    "OpenAI": "OPENAI_API_KEY",
    "Gemini": "GEMINI_API_KEY",
    "Claude": "ANTHROPIC_API_KEY",
    "Groq": "GROQ_API_KEY",
}


@dataclass
class SmokeResult:
    provider: str
    model: str
    status: str
    error_class: str = ""
    detail: str = ""
    attempted: bool = False
    mode: str = ""  # "metadata" (token uretilmedi) | "generation"


def _emit(message: str = "") -> None:
    print(message, flush=True)


def _safe_error(exc: Exception) -> str:
    message = redact_secrets(str(exc)).replace("\n", " ").strip()
    if len(message) > 220:
        message = message[:217] + "..."
    return message or exc.__class__.__name__


_classify_error = classify_error


def _model_specs(registry=None) -> list[tuple[type, str, str]]:
    registry = GENERATOR_REGISTRY if registry is None else registry
    return [
        (generator_class, model, provider)
        for key, (generator_class, model, provider) in registry.items()
        if key != "traditional" and model
    ]


def _select_specs(specs: list[tuple[type, str, str]], only: str | None) -> list[tuple[type, str, str]]:
    if not only:
        return specs
    requested = [item.strip().lower() for item in only.split(",") if item.strip()]
    by_key = {f"{provider.lower()}:{model.lower()}": spec for spec in specs for _cls, model, provider in [spec]}
    unknown = [item for item in requested if item not in by_key]
    if unknown:
        raise ValueError(f"Unknown --only model selection: {', '.join(unknown)}")
    return [by_key[item] for item in requested]


def _probe_metadata(generator_class: type, model: str, provider: str) -> None:
    """Modelin erisilebilirligini URETIM YAPMADAN dogrular.

    Yalnizca saglayicinin model metadata ucunu cagirir (GET /models/{id});
    token uretilmez, dolayisiyla fatura olusmaz.
    """
    client = generator_class(model)._get_client()
    if provider == "Gemini":
        client.models.get(model=model)          # google-genai
    else:
        client.models.retrieve(model)           # openai / anthropic / groq (OpenAI uyumlu)


def _free_only_prerequisites(provider: str, model: str, limiter) -> Optional[str]:
    """FREE_ONLY bir modelde uretim cagrisi yapmanin on kosullari.

    Beyan yoksa ya da yayimlanan limitler girilmemisse cagri YAPILMAZ: biri
    faturalanma riski, digeri kota korlugu demektir. Eksiklik bir HATA degil,
    ERTELEME sebebidir (Bolum 9'daki siraya birakilir).
    """
    if provider not in config.FREE_ONLY_PROVIDERS:
        return None
    if not attestation.is_attested(provider):
        return f"free-tier beyani yok ({attestation.ENV_VAR})"
    if not limiter.is_managed(provider, model):
        return f"{rate_limiter.LIMITS_FILE} icinde yayimlanan limit yok"
    return None


def _run_model(generator_class: type, model: str, provider: str,
               allow_paid_generation: bool = False, limiter=None) -> SmokeResult:
    env_var = ENV_BY_PROVIDER.get(provider)
    if env_var is None:
        return SmokeResult(provider, model, "FAIL", "CODE_ERROR", "Unknown provider mapping")
    if not os.getenv(env_var):
        return SmokeResult(provider, model, "FAIL", "MISSING_CREDENTIAL", f"{env_var} is not set")

    if limiter is not None:
        blocker = _free_only_prerequisites(provider, model, limiter)
        if blocker is not None:
            return SmokeResult(provider, model, "SKIP", "DEFERRED",
                               f"atlandi, beyan/limit yok: {blocker}", mode="skipped")

    metadata_only = provider in config.PAID_PROVIDERS and not allow_paid_generation
    try:
        if metadata_only:
            _probe_metadata(generator_class, model, provider)
        else:
            generator = generator_class(model)
            if limiter is not None and limiter.is_managed(provider, model):
                # Erisim kontrolu de kotadan yer yer; limitorden GECER.
                generator._rate_limiter = limiter
            rows = generator.smoke_test()
            if len(rows) != 1:
                raise RuntimeError("Provider response parsing error: smoke normalization returned no row.")
    except Exception as exc:
        return SmokeResult(provider, model, "FAIL", _classify_error(exc), _safe_error(exc),
                           attempted=True, mode="metadata" if metadata_only else "generation")
    return SmokeResult(provider, model, "PASS", attempted=True,
                       mode="metadata" if metadata_only else "generation")


def _print_header(specs: list[tuple[type, str, str]], expected: int = EXPECTED_MODELS,
                  allow_paid_generation: bool = False) -> None:
    _emit("LLM_TESTLAB REAL API ACCESS CHECK")
    _emit()
    _emit(f"Expected external models: {expected}")
    for _generator_class, model, provider in specs:
        paid = provider in config.PAID_PROVIDERS
        mode = "generation" if (not paid or allow_paid_generation) else "metadata-only (no tokens)"
        _emit(f"- {provider} | {model} | {'PAID' if paid else 'free'} | {mode}")
    _emit()


def _print_result(result: SmokeResult) -> None:
    mode = f" | {result.mode}" if result.mode else ""
    if result.status == "PASS":
        _emit(f"[PASS] {result.provider} | {result.model}{mode}")
        return
    if result.status == "SKIP":
        _emit(f"[SKIP] {result.provider} | {result.model}{mode} | {result.detail}")
        return
    suffix = f"{mode} | {result.error_class}"
    if result.detail:
        suffix += f" | {result.detail}"
    _emit(f"[FAIL] {result.provider} | {result.model}{suffix}")


def _print_summary(results: list[SmokeResult], expected: int = EXPECTED_MODELS, targeted: bool = False) -> bool:
    tested = sum(result.attempted for result in results)
    passed = sum(result.status == "PASS" for result in results)
    failed = sum(result.status == "FAIL" for result in results)
    skipped = sum(result.status == "SKIP" for result in results)
    successful = len(results) == expected and tested == expected and passed == expected and failed == 0 and skipped == 0
    _emit()
    _emit("SMOKE TEST SUMMARY")
    _emit(f"Expected: {expected}")
    _emit(f"Tested: {tested}")
    _emit(f"Passed: {passed}")
    _emit(f"Failed: {failed}")
    _emit(f"Skipped: {skipped}")
    if targeted:
        _emit(f"TARGETED RETEST: {'PASS' if successful else 'FAIL'}")
        _emit("READY FOR FULL EXPERIMENT: NO")
    else:
        _emit(f"READY FOR FULL EXPERIMENT: {'YES' if successful else 'NO'}")
    return successful


def main(registry=None, only: str | None = None, allow_paid_generation: bool = False) -> int:
    load_dotenv(PROJECT_ROOT / ".env")
    all_specs = _model_specs(registry)
    try:
        specs = _select_specs(all_specs, only)
    except ValueError as exc:
        _emit(f"[FAIL] Selection | CODE_ERROR | {_safe_error(exc)}")
        _print_summary([], 0, targeted=True)
        return 1
    targeted = bool(only)
    expected = len(specs) if targeted else EXPECTED_MODELS
    _print_header(specs, expected, allow_paid_generation)
    if not targeted and len(specs) != EXPECTED_MODELS:
        _emit(f"[FAIL] Registry | configured models | NO_MODELS_TESTED | expected {EXPECTED_MODELS}, found {len(specs)}")
        _print_summary([], EXPECTED_MODELS)
        return 1
    limiter = rate_limiter.RateLimiter(rate_limiter.load_limits())
    results: list[SmokeResult] = []
    for generator_class, model, provider in specs:
        _emit(f"[RUN] {provider} | {model}")
        result = _run_model(generator_class, model, provider, allow_paid_generation, limiter)
        results.append(result)
        _print_result(result)
    return 0 if _print_summary(results, expected, targeted=targeted) else 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run minimal real-API model smoke tests.")
    parser.add_argument(
        "--only",
        help="Comma-separated provider:model selections; only those exact configured models are called.",
    )
    parser.add_argument(
        "--paid-generation",
        action="store_true",
        help="UCRETLI saglayicilara da gercek uretim cagrisi yap (PARA HARCAR). "
             "Verilmezse ucretlilerde yalnizca model metadata cagrilir.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    raise SystemExit(main(only=args.only, allow_paid_generation=args.paid_generation))
