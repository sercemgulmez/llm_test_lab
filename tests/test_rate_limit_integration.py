"""Bolum 3/7: limitorun uretim akisiyla birlikte davranisi ($0, ag yok)."""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest

import config
import main
import rate_limiter
from generators.base import BaseGenerator
from models import TokenUsage
from rate_limiter import ModelLimits, RateLimiter
from tests.conftest import synthetic_limits

KEY = ("Groq", "openai/gpt-oss-20b")


class _Rate429(Exception):
    """Saglayicinin 429'u; basliklariyla birlikte."""

    def __init__(self, headers):
        super().__init__("Error code: 429 - rate limit exceeded, too many requests")
        self.response = type("R", (), {"headers": headers})()


class _Probe(BaseGenerator):
    """Gercek _rate_limited_call yolunu kosan minimal generator."""

    _provider_label = "Groq"

    def __init__(self, model, responses):
        self.model = model
        self._responses = list(responses)
        self.calls = 0

    def _generate_for_operation(self, op, variant_name, variant_desc, num_cases):
        return []

    def request(self, prompt):
        self.calls += 1
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _limits(**kw):
    base = dict(rpm=30, rpd=1000, tpm=200_000, tpd=1_000_000, safety=1.0, counts="total")
    base.update(kw)
    return {KEY: ModelLimits(KEY[0], KEY[1], base["rpm"], base["rpd"],
                             base["tpm"], base["tpd"], base["safety"], base["counts"])}


def _usage():
    return TokenUsage(input_tokens=100, output_tokens=50, split_available=True)


# ── Reaktif 429: içerik denemesi harcanmaz ───────────────────────────────

def test_reactive_429_retries_without_spending_a_content_attempt(monkeypatch):
    """Dakika kotasi 429'u: Retry-After'a uyulur, cagri tekrarlanir.

    Onemli olan, bunun RETRY_MAX_ATTEMPTS (icerik denemesi) sayacini
    TUKETMEMESI; ayri ve dusuk bir sinirla sayilmasi.
    """
    monkeypatch.setattr(time, "sleep", lambda s: None)
    gen = _Probe("openai/gpt-oss-20b", [
        _Rate429({"retry-after": "1", "x-ratelimit-remaining-requests": "500"}),
        ("metin", _usage()),
    ])
    gen._rate_limiter = RateLimiter(_limits())

    text, usage, meta = gen._rate_limited_call(gen.request, "prompt", 15)

    assert text == "metin"
    assert meta["reactive_429_count"] == 1
    assert meta["quota_kind"] == rate_limiter.QUOTA_MINUTE
    assert meta["retry_after_s"] == 1.0
    assert gen.calls == 2, "yeniden deneme limitor katmaninda yapilmali"


def test_reactive_429_has_its_own_low_ceiling(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda s: None)
    too_many = [
        _Rate429({"retry-after": "1", "x-ratelimit-remaining-requests": "500"})
        for _ in range(config.REACTIVE_429_MAX_RETRIES + 2)
    ]
    gen = _Probe("openai/gpt-oss-20b", too_many)
    gen._rate_limiter = RateLimiter(_limits())

    with pytest.raises(Exception):
        gen._rate_limited_call(gen.request, "prompt", 15)

    assert gen.calls == config.REACTIVE_429_MAX_RETRIES + 1, (
        "reaktif sinir RETRY_MAX_ATTEMPTS'ten ayri ve dusuk olmali"
    )
    assert config.REACTIVE_429_MAX_RETRIES < config.RETRY_MAX_ATTEMPTS


def test_daily_quota_is_not_retried_in_the_same_session(monkeypatch):
    """Gunluk kota bittiyse ayni oturumda tekrar denenmez (gunlerce beklenmez)."""
    monkeypatch.setattr(time, "sleep", lambda s: None)
    gen = _Probe("openai/gpt-oss-20b", [
        _Rate429({"x-ratelimit-remaining-requests": "0"}),
        ("asla ulasilmamali", _usage()),
    ])
    limiter = RateLimiter(_limits())
    gen._rate_limiter = limiter

    with pytest.raises(Exception):
        gen._rate_limited_call(gen.request, "prompt", 15)

    assert gen.calls == 1, "gunluk kota bittiyse tekrar denenmemeli"
    assert limiter.daily_quota_exhausted(*KEY) is True
    assert limiter.stats()["daily_quota_exhausted"][f"{KEY[0]}/{KEY[1]}"]


def test_ledger_meta_is_populated(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda s: None)
    gen = _Probe("openai/gpt-oss-20b", [("metin", _usage())])
    gen._rate_limiter = RateLimiter(_limits())
    _t, _u, meta = gen._rate_limited_call(gen.request, "x" * 4000, 15)

    assert meta["reserved_output_tokens"] == 3000, "Groq tavani: min(8192, max(2048, 15*200))"
    assert meta["reserved_input_tokens"] == 1500, "4000 karakter / 4 x 1.5 guvenlik"
    assert meta["limiter_wait_ms"] >= 0


def test_unmanaged_paid_model_bypasses_the_limiter():
    """Ucretli generator'lar limitore hic ugramamali."""
    gen = _Probe("gpt-4.1", [("metin", _usage())])
    gen._provider_label = "OpenAI"
    gen._rate_limiter = RateLimiter(_limits())  # yalnizca Groq modelini yonetiyor
    _t, _u, meta = gen._rate_limited_call(gen.request, "prompt", 15)
    assert meta["limiter_wait_ms"] == 0
    assert meta["reserved_input_tokens"] == 0


# ── Uzun beklemeler future/executor'da takılmıyor ────────────────────────

def test_long_limiter_waits_do_not_trip_futures():
    """Dakikalarca suren bir bekleme as_completed/result()'ta zaman asimina ugramamali.

    main.py bu iki cagriyi da timeout VERMEDEN yapiyor; test bunu kanitlar.
    Gercek dakikalari beklememek icin sure kisaltildi, mekanizma aynidir.
    """
    def slow_task():
        time.sleep(0.6)
        return "bitti"

    with ThreadPoolExecutor(max_workers=config.MAX_PARALLEL_GENERATORS) as executor:
        futures = [executor.submit(slow_task) for _ in range(3)]
        results = [f.result() for f in as_completed(futures)]

    assert results == ["bitti"] * 3


def test_main_calls_as_completed_and_result_without_timeout():
    """Kaynak okumasi: main.py future beklemelerine timeout VERMEZ."""
    import inspect
    source = inspect.getsource(main.main)
    assert "as_completed(future_to_task)" in source
    assert "future.result()" in source
    assert "result(timeout" not in source
    assert "as_completed(future_to_task, timeout" not in source


# ── Günlük kota → resume atlaması ────────────────────────────────────────

def test_resume_skips_a_model_until_its_quota_window_opens():
    from datetime import datetime, timedelta

    future = (datetime.now().astimezone() + timedelta(hours=3)).isoformat(timespec="seconds")
    past = (datetime.now().astimezone() - timedelta(hours=1)).isoformat(timespec="seconds")
    records = {
        "GroqGenerator:m|basic": {"next_available_at": future},
        "GroqGenerator:m|edge_focused": {"next_available_at": past},
        "GeminiGenerator:m|basic": {},
    }
    assert main._quota_block_until(records, "GroqGenerator:m|basic") is not None
    assert main._quota_block_until(records, "GroqGenerator:m|edge_focused") is None, (
        "pencere acildiysa artik atlanmamali"
    )
    assert main._quota_block_until(records, "GeminiGenerator:m|basic") is None


def test_checkpoint_persists_next_available_at(tmp_path):
    from checkpoint import RunCheckpoint

    ckpt = RunCheckpoint(str(tmp_path), flush_every=1)
    ckpt.mark_task_done("GroqGenerator:m|basic", 0, failure_origin="altyapi",
                        next_available_at="2026-09-30T10:00:00+03:00")
    ckpt.flush()
    record = ckpt.task_records()["GroqGenerator:m|basic"]
    assert record["failure_origin"] == "altyapi"
    assert record["next_available_at"] == "2026-09-30T10:00:00+03:00"


# ── Limit girdisi olmayan model koşmaz ───────────────────────────────────

def test_model_without_published_limits_is_blocked():
    from generators.groq_gen import GroqGenerator

    gens = [(GroqGenerator("openai/gpt-oss-20b"), "basic", "x")]
    assert main._rate_limit_check(gens, {}) , "limit yoksa kosmamali"
    assert main._rate_limit_check(gens, synthetic_limits()) == []


def test_paid_model_needs_no_published_limits():
    from generators.openai_gen import OpenAIGenerator

    gens = [(OpenAIGenerator("gpt-4.1"), "basic", "x")]
    assert main._rate_limit_check(gens, {}) == []
