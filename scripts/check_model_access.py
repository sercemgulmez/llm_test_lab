"""Observable, minimal real-API smoke test for every configured LLM model."""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import os
from pathlib import Path
import sys

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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


def _emit(message: str = "") -> None:
    print(message, flush=True)


def _safe_error(exc: Exception) -> str:
    message = redact_secrets(str(exc)).replace("\n", " ").strip()
    if len(message) > 220:
        message = message[:217] + "..."
    return message or exc.__class__.__name__


def _classify_error(exc: Exception) -> str:
    status = getattr(exc, "status_code", None)
    code = str(getattr(exc, "code", "") or "").lower()
    name = exc.__class__.__name__.lower()
    message = str(exc).lower()
    combined = f"{code} {name} {message}"
    if "providerresponseparseerror" in combined or "provider response parse error" in combined:
        return "PROVIDER_RESPONSE_PARSE_ERROR"
    if "modeloutputformaterror" in combined or "model output format error" in combined:
        return "MODEL_OUTPUT_FORMAT_ERROR"
    if isinstance(exc, TimeoutError) or "timeout" in combined or "timed out" in combined:
        return "TIMEOUT"
    if any(term in combined for term in ("connection", "network", "dns", "name resolution")):
        return "NETWORK_ERROR"
    if status == 401 or any(term in combined for term in ("invalid api key", "incorrect api key", "authentication_error", "unauthorized")):
        return "AUTH_ERROR"
    if any(term in combined for term in ("quota", "billing", "credit", "insufficient_quota")):
        return "BILLING_QUOTA_ERROR"
    if status == 429 or any(term in combined for term in ("rate limit", "rate_limit", "too many requests")):
        return "RATE_LIMIT"
    if status == 404 or any(term in combined for term in ("model_not_found", "model not found", "does not exist", "no longer available")):
        return "MODEL_NOT_FOUND"
    if status == 403 or any(term in combined for term in ("access denied", "permission", "not authorized", "forbidden")):
        return "MODEL_ACCESS_ERROR"
    if status in (400, 422) or isinstance(exc, (TypeError, ValueError)):
        return "REQUEST_CONTRACT_ERROR"
    if status is not None and int(status) >= 500:
        return "PROVIDER_ERROR"
    if isinstance(exc, (ImportError, AttributeError, NameError, NotImplementedError)):
        return "CODE_ERROR"
    return "UNKNOWN_ERROR"


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


def _run_model(generator_class: type, model: str, provider: str) -> SmokeResult:
    env_var = ENV_BY_PROVIDER.get(provider)
    if env_var is None:
        return SmokeResult(provider, model, "FAIL", "CODE_ERROR", "Unknown provider mapping")
    if not os.getenv(env_var):
        return SmokeResult(provider, model, "FAIL", "MISSING_CREDENTIAL", f"{env_var} is not set")
    try:
        rows = generator_class(model).smoke_test()
        if len(rows) != 1:
            raise RuntimeError("Provider response parsing error: smoke normalization returned no row.")
    except Exception as exc:
        return SmokeResult(provider, model, "FAIL", _classify_error(exc), _safe_error(exc), attempted=True)
    return SmokeResult(provider, model, "PASS", attempted=True)


def _print_header(specs: list[tuple[type, str, str]], expected: int = EXPECTED_MODELS) -> None:
    _emit("LLM_TESTLAB REAL API SMOKE TEST")
    _emit()
    _emit(f"Expected external models: {expected}")
    for _generator_class, model, provider in specs:
        _emit(f"- {provider} | {model}")
    _emit()


def _print_result(result: SmokeResult) -> None:
    if result.status == "PASS":
        _emit(f"[PASS] {result.provider} | {result.model}")
        return
    suffix = f" | {result.error_class}"
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


def main(registry=None, only: str | None = None) -> int:
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
    _print_header(specs, expected)
    if not targeted and len(specs) != EXPECTED_MODELS:
        _emit(f"[FAIL] Registry | configured models | NO_MODELS_TESTED | expected {EXPECTED_MODELS}, found {len(specs)}")
        _print_summary([], EXPECTED_MODELS)
        return 1
    results: list[SmokeResult] = []
    for generator_class, model, provider in specs:
        _emit(f"[RUN] {provider} | {model}")
        result = _run_model(generator_class, model, provider)
        results.append(result)
        _print_result(result)
    return 0 if _print_summary(results, expected, targeted=targeted) else 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run minimal real-API model smoke tests.")
    parser.add_argument(
        "--only",
        help="Comma-separated provider:model selections; only those exact configured models are called.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    raise SystemExit(main(only=args.only))
