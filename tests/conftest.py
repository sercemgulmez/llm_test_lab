import shutil
import uuid
from pathlib import Path

import pytest


@pytest.fixture
def tmp_path():
    """Windows izin sorunlarından bağımsız, çalışma alanı içi geçici klasör."""
    path = Path.cwd() / f"tmpcase-{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def synthetic_limits(rpm=30, rpd=1000, tpm=200_000, tpd=1_000_000, safety=0.8):
    """Testler icin SENTETIK free-tier limitleri.

    Gercek degerler free_tier_limits.json'a kullanici tarafindan girilir ve
    repoda BOS durur; testler bu yuzden kendi limitlerini uretir. Boylece
    testler saglayicinin gercek limitlerine bagimli olmaz.
    """
    import rate_limiter

    def _limit(provider, model, tpm_counts):
        return rate_limiter.ModelLimits(
            provider=provider, model=model,
            published_rpm=rpm, published_rpd=rpd,
            published_tpm=tpm, published_tpd=tpd,
            safety_factor=safety, tpm_counts=tpm_counts,
        )

    return {
        ("Groq", "openai/gpt-oss-120b"): _limit("Groq", "openai/gpt-oss-120b", "total"),
        ("Groq", "openai/gpt-oss-20b"): _limit("Groq", "openai/gpt-oss-20b", "total"),
        ("Gemini", "gemini-2.5-flash"): _limit("Gemini", "gemini-2.5-flash", "input_only"),
        ("Gemini", "gemini-3.5-flash-lite"): _limit("Gemini", "gemini-3.5-flash-lite", "input_only"),
    }


@pytest.fixture
def limits_ready(monkeypatch):
    """free_tier_limits.json doluymus gibi davranir (sentetik degerlerle)."""
    import rate_limiter

    limits = synthetic_limits()
    monkeypatch.setattr(rate_limiter, "load_limits", lambda *a, **k: dict(limits))
    return limits
