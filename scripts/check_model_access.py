"""Smoke-test configured LLM model access without printing API keys."""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import sys
from typing import Callable
import warnings

warnings.simplefilter("ignore")

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from generators.traditional import TraditionalGenerator
from models import ApiOperation
from security.redaction import redact_secrets


PROMPT = "Reply only with OK."
PASS = "PASS"
MISSING_KEY = "MISSING_KEY"
INVALID_KEY = "INVALID_KEY"
MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
MODEL_ACCESS_DENIED = "MODEL_ACCESS_DENIED"
RATE_LIMITED = "RATE_LIMITED"
BILLING_OR_QUOTA = "BILLING_OR_QUOTA"
REQUEST_ERROR = "REQUEST_ERROR"
UNKNOWN_ERROR = "UNKNOWN_ERROR"

@dataclass
class AccessResult:
    provider: str
    model: str
    api_key_present: str
    real_api_call: str
    result: str
    detail: str = ""


def _safe_error(exc: Exception) -> str:
    message = redact_secrets(str(exc)).replace("\n", " ").strip()
    if len(message) > 220:
        message = message[:217] + "..."
    return message or exc.__class__.__name__


def _classify_error(exc: Exception) -> str:
    status_code = getattr(exc, "status_code", None)
    code = str(getattr(exc, "code", "") or "").lower()
    message = str(exc).lower()
    combined = f"{code} {message}"

    if status_code in (401, 403) and any(term in combined for term in ("invalid", "incorrect", "unauthorized", "authentication")):
        return INVALID_KEY
    if any(term in combined for term in ("invalid api key", "incorrect api key", "authentication_error", "unauthorized")):
        return INVALID_KEY
    if status_code == 404 or any(
        term in combined
        for term in ("model_not_found", "not_found", "not found", "does not exist", "not a valid model", "no longer available")
    ):
        return MODEL_NOT_FOUND
    if status_code == 403 or any(term in combined for term in ("access denied", "permission", "not authorized", "forbidden")):
        return MODEL_ACCESS_DENIED
    if status_code == 429 or any(term in combined for term in ("rate limit", "rate_limit", "too many requests")):
        if any(term in combined for term in ("quota", "billing", "credit", "insufficient_quota")):
            return BILLING_OR_QUOTA
        return RATE_LIMITED
    if any(term in combined for term in ("quota", "billing", "credit", "insufficient_quota")):
        return BILLING_OR_QUOTA
    if status_code in (400, 422) or isinstance(exc, (TypeError, ValueError)):
        return REQUEST_ERROR
    return UNKNOWN_ERROR


def _run_with_key(provider: str, model: str, env_var: str, call: Callable[[str, str], None]) -> AccessResult:
    api_key = os.getenv(env_var)
    if not api_key:
        return AccessResult(provider, model, "NO", "FAIL", MISSING_KEY, f"{env_var} is not set")
    try:
        call(api_key, model)
    except Exception as exc:  # noqa: BLE001 - smoke test must classify every provider error.
        return AccessResult(provider, model, "YES", "FAIL", _classify_error(exc), _safe_error(exc))
    return AccessResult(provider, model, "YES", "PASS", PASS)


def _check_openai(api_key: str, model: str) -> None:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": PROMPT}],
        max_tokens=1,
    )


def _check_groq(api_key: str, model: str) -> None:
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
    client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": PROMPT}],
        max_tokens=1,
    )


def _check_gemini(api_key: str, model: str) -> None:
    from google import genai

    client = genai.Client(api_key=api_key)
    try:
        client.models.generate_content(
            model=model,
            contents=PROMPT,
            config={"max_output_tokens": 1},
        )
    except TypeError:
        client.models.generate_content(model=model, contents=PROMPT)


def _check_anthropic(api_key: str, model: str) -> None:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    client.messages.create(
        model=model,
        max_tokens=1,
        messages=[{"role": "user", "content": PROMPT}],
    )


def _check_traditional() -> AccessResult:
    try:
        op = ApiOperation(op_id="SMOKE", method="GET", path="/smoke")
        with redirect_stdout(io.StringIO()):
            rows = TraditionalGenerator().generate([op], "", "", 1)
    except Exception as exc:  # noqa: BLE001 - keep final report complete.
        return AccessResult("Local", "TraditionalGenerator", "N/A", "FAIL", UNKNOWN_ERROR, _safe_error(exc))
    if rows:
        return AccessResult("Local", "TraditionalGenerator", "N/A", "PASS", PASS)
    return AccessResult("Local", "TraditionalGenerator", "N/A", "FAIL", UNKNOWN_ERROR, "No rows generated")


def _collect_results() -> list[AccessResult]:
    results: list[AccessResult] = []
    for model in config.OPENAI_MODELS:
        results.append(_run_with_key("OpenAI", model, "OPENAI_API_KEY", _check_openai))
    for model in config.GEMINI_MODELS:
        results.append(_run_with_key("Gemini", model, "GEMINI_API_KEY", _check_gemini))
    for model in config.GROQ_MODELS:
        results.append(_run_with_key("Groq", model, "GROQ_API_KEY", _check_groq))
    for model in config.CLAUDE_MODELS:
        results.append(_run_with_key("Anthropic", model, "ANTHROPIC_API_KEY", _check_anthropic))
    results.append(_check_traditional())
    return results


def _print_model_configuration() -> None:
    print("MODEL CONFIGURATION")
    print("OpenAI:")
    for model in config.OPENAI_MODELS:
        print(f"- {model}")
    print()
    print("Gemini:")
    for model in config.GEMINI_MODELS:
        print(f"- {model}")
    print()
    print("Groq:")
    for model in config.GROQ_MODELS:
        print(f"- {model}")
    print()
    print("Anthropic:")
    for model in config.CLAUDE_MODELS:
        print(f"- {model}")
    print()
    print("Baseline:")
    print("- TraditionalGenerator")
    print()


def _print_results(results: list[AccessResult]) -> None:
    print("API ACCESS RESULTS")
    print()
    print("| Provider | Model | API Key Present | Real API Call | Result |")
    print("|----------|-------|-----------------|---------------|--------|")
    for item in results:
        detail = f"{item.result}: {item.detail}" if item.detail and item.result != PASS else item.result
        print(f"| {item.provider} | {item.model} | {item.api_key_present} | {item.real_api_call} | {detail} |")
    print()

    failures = [item for item in results if item.result != PASS]
    ready = not failures
    print(f"READY FOR EXPERIMENT: {'YES' if ready else 'NO'}")
    if failures:
        print()
        for item in failures:
            reason = item.detail or item.result
            print(f"- {item.provider} {item.model}: {item.result} ({reason})")


def main() -> int:
    load_dotenv()
    results = _collect_results()
    _print_model_configuration()
    _print_results(results)
    return 0 if all(item.result == PASS for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
