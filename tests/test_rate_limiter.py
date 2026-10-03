"""Bolum 3: limitor. Tum testler SENTETIK limitlerle kosar ($0, ag yok).

Gercek degerler free_tier_limits.json'a kullanicidan gelir ve repoda bostur;
testlerin saglayicinin gercek limitlerine bagimli olmamasi kasitlidir.
"""

import json
import threading
import time

import pytest

import rate_limiter
from rate_limiter import ImpossibleRequest, ModelLimits, RateLimiter, _SlidingWindow

KEY = ("Groq", "openai/gpt-oss-20b")


def _limits(rpm=30, rpd=1000, tpm=200_000, tpd=1_000_000, safety=1.0, counts="total"):
    return {KEY: ModelLimits(KEY[0], KEY[1], rpm, rpd, tpm, tpd, safety, counts)}


# ── Kayan pencere ─────────────────────────────────────────────────────────

def test_window_prunes_old_events():
    w = _SlidingWindow(60.0)
    w.add(0.0, 5)
    w.add(30.0, 5)
    assert w.total(59.0) == 10
    assert w.total(61.0) == 5, "60 sn'den eski olay dusmeli"


def test_window_wait_is_until_oldest_event_leaves():
    w = _SlidingWindow(60.0)
    w.add(0.0, 10)
    assert w.wait_for(10.0, 1, limit=10) == pytest.approx(50.0)
    assert w.wait_for(10.0, 1, limit=20) == 0.0


def test_window_returns_inf_when_request_never_fits():
    w = _SlidingWindow(60.0)
    assert w.wait_for(0.0, 100, limit=10) == float("inf")


def test_effective_limits_apply_safety_factor():
    limit = ModelLimits("Groq", "m", 30, 1000, 8000, 200_000, 0.8, "total")
    assert limit.effective_rpm == 24
    assert limit.effective_rpd == 800
    assert limit.effective_tpm == 6400
    assert limit.effective_tpd == 160_000


# ── free_tier_limits.json ────────────────────────────────────────────────

def test_repo_limits_file_covers_every_free_only_model():
    """Dosya, FREE_ONLY her model icin yayimlanan limitleri tasimali.

    Gemini 3 Ekim 2026'da ucretli tier'a gecti ve dosyadan cikarildi; limitor
    yalnizca FREE_ONLY saglayicilari yonetir, ucretlilerde koruma butce
    sigortasidir.
    """
    import config
    from generators import GENERATOR_REGISTRY

    limits = rate_limiter.load_limits()
    expected = {
        (provider, model)
        for _k, (_c, model, provider) in GENERATOR_REGISTRY.items()
        if model is not None and provider in config.FREE_ONLY_PROVIDERS
    }
    assert set(limits) == expected
    assert rate_limiter.missing_fields() == {}, "zorunlu alanlarin hepsi dolu olmali"


def test_unknown_tpd_is_not_treated_as_unlimited(tmp_path):
    """Yayimlanmamis gunluk token limiti BILINMIYOR demektir, sinirsiz DEGIL.

    (Bu durum Gemini free tier'da gercekti; Gemini 3 Ekim 2026'da ucretli tier'a
    gecip free_tier_limits.json'dan cikarildigi icin test artik sentetik bir
    dosyayla kosuyor — mantik aynen gecerli.)
    """
    path = tmp_path / "limits.json"
    path.write_text(json.dumps({
        "safety_factor_default": 0.8,
        "providers": {"X": {"tpm_counts": "input_only", "models": {
            "tpd-yok": {"published_rpm": 5, "published_rpd": 2,
                        "published_tpm": 250_000, "published_tpd": None},
            "tpd-var": {"published_rpm": 30, "published_rpd": 1000,
                        "published_tpm": 8000, "published_tpd": 200_000},
        }}},
    }), encoding="utf-8")
    limits = rate_limiter.load_limits(path)

    unknown = limits[("X", "tpd-yok")]
    assert unknown.published_tpd is None
    assert unknown.tpd_known is False
    assert unknown.effective_tpd is None, "None = bilinmiyor; buyuk bir sayi UYDURULMAZ"

    known = limits[("X", "tpd-var")]
    assert known.tpd_known is True
    assert known.effective_tpd == int(known.published_tpd * known.safety_factor)


def test_unknown_tpd_does_not_block_reservations():
    """TPD bilinmiyorken limitor TPD penceresi uygulamaz ama calismaya devam eder."""
    key = ("Gemini", "gemini-2.5-flash")
    limits = {key: ModelLimits(*key, 15, 10, 250_000, None, 1.0, "input_only")}
    limiter = RateLimiter(limits)
    reservation = limiter.reserve(*key, 500, 8192)
    assert reservation.tokens == 500
    limiter.settle(reservation, actual_tokens=500)


def test_partially_filled_model_is_not_loaded(tmp_path):
    path = tmp_path / "limits.json"
    path.write_text(json.dumps({
        "safety_factor_default": 0.8,
        "providers": {"Groq": {"tpm_counts": "total", "models": {
            "a": {"published_rpm": 30, "published_rpd": 1000,
                  "published_tpm": None, "published_tpd": 200_000},
            "b": {"published_rpm": 30, "published_rpd": 1000,
                  "published_tpm": 8000, "published_tpd": 200_000},
        }}},
    }), encoding="utf-8")
    loaded = rate_limiter.load_limits(path)
    assert set(loaded) == {("Groq", "b")}
    assert rate_limiter.missing_fields(path) == {("Groq", "a"): ["published_tpm"]}


# ── Ön kontrol ────────────────────────────────────────────────────────────

def test_request_larger_than_tpm_is_refused_not_waited():
    limiter = RateLimiter(_limits(tpm=1000))
    with pytest.raises(ImpossibleRequest) as excinfo:
        limiter.preflight(*KEY, reserved_tokens=5000)
    assert "hicbir zaman basarili olamaz" in str(excinfo.value)


def test_request_within_tpm_passes_preflight():
    limiter = RateLimiter(_limits(tpm=1000))
    limiter.preflight(*KEY, reserved_tokens=900)  # firlatmamali


# ── Eşzamanlılık: model başına tek çağrı ─────────────────────────────────

def test_one_concurrent_call_per_model():
    limiter = RateLimiter(_limits())
    overlap = {"max": 0, "current": 0}
    lock = threading.Lock()

    def worker():
        reservation = limiter.reserve(*KEY, 10, 10)
        with lock:
            overlap["current"] += 1
            overlap["max"] = max(overlap["max"], overlap["current"])
        time.sleep(0.02)
        with lock:
            overlap["current"] -= 1
        limiter.settle(reservation, actual_tokens=20)

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert overlap["max"] == 1, "FREE_ONLY model basina en fazla 1 eszamanli cagri"


def test_different_models_run_in_parallel():
    other = ("Gemini", "gemini-2.5-flash")
    limits = _limits()
    limits[other] = ModelLimits(*other, 30, 1000, 200_000, 1_000_000, 1.0, "input_only")
    limiter = RateLimiter(limits)
    r1 = limiter.reserve(*KEY, 10, 10)
    r2 = limiter.reserve(*other, 10, 10)  # bloklanmamali
    assert r1.wait_ms == 0 and r2.wait_ms == 0
    limiter.settle(r1)
    limiter.settle(r2)


# ── Katı sahte sunucu: limitörle 0 adet 429 ──────────────────────────────

class StrictQuotaServer:
    """RPM/TPM/RPD/TPD'yi KATI uygular; asilirsa 429 sayar."""

    def __init__(self, rpm, rpd, tpm, tpd):
        self.caps = {"rpm": rpm, "rpd": rpd, "tpm": tpm, "tpd": tpd}
        self.windows = {
            "rpm": _SlidingWindow(60.0), "tpm": _SlidingWindow(60.0),
            "rpd": _SlidingWindow(86400.0), "tpd": _SlidingWindow(86400.0),
        }
        self.lock = threading.Lock()
        self.rejections = 0
        self.accepted = 0

    def call(self, tokens, now):
        with self.lock:
            over = (
                self.windows["rpm"].total(now) + 1 > self.caps["rpm"]
                or self.windows["rpd"].total(now) + 1 > self.caps["rpd"]
                or self.windows["tpm"].total(now) + tokens > self.caps["tpm"]
                or self.windows["tpd"].total(now) + tokens > self.caps["tpd"]
            )
            if over:
                self.rejections += 1
                raise RuntimeError("Error code: 429 - rate limit exceeded")
            self.windows["rpm"].add(now, 1)
            self.windows["rpd"].add(now, 1)
            self.windows["tpm"].add(now, tokens)
            self.windows["tpd"].add(now, tokens)
            self.accepted += 1


def test_limiter_prevents_all_429s_under_nine_parallel_workers():
    """9 paralel isciyle, limitor devredeyken sunucu HIC 429 uretmemeli."""
    RPM, TPM, TOKENS, CALLS = 10, 1000, 100, 40
    server = StrictQuotaServer(rpm=RPM, rpd=10_000, tpm=TPM, tpd=1_000_000)

    virtual = {"now": 0.0}
    vlock = threading.Lock()

    def clock():
        return virtual["now"]

    def sleep(seconds):
        # Sanal saat: test gercekten beklemez ama pencere mantigi aynen isler.
        with vlock:
            virtual["now"] += seconds

    limiter = RateLimiter(
        _limits(rpm=RPM, tpm=TPM, rpd=10_000, tpd=1_000_000),
        sleep=sleep, clock=clock,
    )

    errors = []

    def worker():
        for _ in range(CALLS // 9 + 1):
            reservation = limiter.reserve(*KEY, TOKENS // 2, TOKENS // 2)
            try:
                server.call(TOKENS, clock())
            except RuntimeError as exc:
                errors.append(str(exc))
            finally:
                limiter.settle(reservation, actual_tokens=TOKENS)

    threads = [threading.Thread(target=worker) for _ in range(9)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert server.accepted > 0
    assert server.rejections == 0, f"limitor 429 uretti: {errors[:3]}"


def test_miscalibrated_limiter_does_produce_429s():
    """Kontrol grubu: limitor limitleri YANLIS bilirse sunucu 429 uretir.

    Bu test, yukaridaki '0 adet 429' sonucunun limitorden geldigini kanitlar —
    sahte sunucunun zaten hic reddetmiyor olmasindan degil.
    """
    server = StrictQuotaServer(rpm=5, rpd=10_000, tpm=1_000_000, tpd=10_000_000)
    limiter = RateLimiter(_limits(rpm=500, tpm=1_000_000))  # BILEREK yanlis

    for _ in range(20):
        reservation = limiter.reserve(*KEY, 10, 10)
        try:
            server.call(20, 0.0)
        except RuntimeError:
            pass
        finally:
            limiter.settle(reservation, actual_tokens=20)

    assert server.rejections > 0, "yanlis kalibrasyonda 429 beklenir"


# ── Rezervasyon ve düzeltme ──────────────────────────────────────────────

def test_reservation_uses_output_ceiling_then_corrects_to_actual():
    limiter = RateLimiter(_limits(tpm=10_000))
    reservation = limiter.reserve(*KEY, 500, 3000)
    assert reservation.tokens == 3500, "rezervasyon cikti TAVANI uzerinden"
    limiter.settle(reservation, actual_tokens=800)
    state = limiter._states[KEY]
    assert state.tpm.total(limiter._clock()) == 800, "gercek kullanimla duzeltilmeli"


def test_input_only_provider_reserves_input_only():
    key = ("Gemini", "gemini-2.5-flash")
    limits = {key: ModelLimits(*key, 30, 1000, 200_000, 1_000_000, 1.0, "input_only")}
    limiter = RateLimiter(limits)
    reservation = limiter.reserve(*key, 500, 8192)
    assert reservation.tokens == 500, "Gemini dokumani: TPM yalniz GIRDI sayar"
    limiter.settle(reservation)


# ── Kalıcılık: resume sonrası sıfırlanmaz ────────────────────────────────

def test_daily_usage_survives_restart(tmp_path):
    usage = tmp_path / "usage.jsonl"
    first = RateLimiter(_limits(rpd=10), usage_path=usage)
    for _ in range(4):
        reservation = first.reserve(*KEY, 10, 10)
        first.settle(reservation, actual_tokens=20)

    resumed = RateLimiter(_limits(rpd=10), usage_path=usage)
    state = resumed._states[KEY]
    assert state.rpd.total(resumed._clock()) == 4, "gunluk kullanim resume'da sifirlanmamali"
    assert state.tpd.total(resumed._clock()) == 80


def test_usage_older_than_24h_is_dropped(tmp_path):
    usage = tmp_path / "usage.jsonl"
    usage.write_text(json.dumps({
        "wall_ts": time.time() - (25 * 3600),
        "provider": KEY[0], "model": KEY[1], "requests": 5, "tokens": 500,
    }) + "\n", encoding="utf-8")
    limiter = RateLimiter(_limits(), usage_path=usage)
    assert limiter._states[KEY].rpd.total(limiter._clock()) == 0


# ── Kalibrasyon (yalnız dokümante başlıklar) ─────────────────────────────

def test_groq_headers_calibrate_the_windows():
    limiter = RateLimiter(_limits(rpd=1000, tpm=8000))
    limiter.calibrate(*KEY, {"x-ratelimit-remaining-requests": "900",
                             "x-ratelimit-remaining-tokens": "6000"})
    state = limiter._states[KEY]
    now = limiter._clock()
    assert state.rpd.total(now) == 100, "1000 - 900 kullanilmis"
    assert state.tpm.total(now) == 2000, "8000 - 6000 kullanilmis"


def test_gemini_headers_are_not_interpreted():
    """Google hicbir limit basligi belgelemedi; oradan gelen basliga anlam yuklenmez."""
    key = ("Gemini", "gemini-2.5-flash")
    limits = {key: ModelLimits(*key, 30, 1000, 200_000, 1_000_000, 1.0, "input_only")}
    limiter = RateLimiter(limits)
    assert rate_limiter.known_rate_limit_headers(
        "Gemini", {"x-ratelimit-remaining-requests": "5"}
    ) == {}
    limiter.calibrate(*key, {"x-ratelimit-remaining-requests": "5"})
    # Kalibrasyon yine de calisir ama yalnizca bizim bildigimiz alanlarla;
    # onemli olan basliklarin DEFTERE yorumlanmis gibi yazilmamasi.
    assert limiter._states[key] is not None


# ── Bekleme ÜST SINIRI: hiçbir çalıştırma saatlerce asılı kalmaz ─────────

def test_long_wait_raises_instead_of_sleeping_for_hours():
    """RPD tukenince limitor BEKLEMEZ: QuotaExhausted firlatir.

    Aksi halde gunluk kotasi dolan bir model icin ~24 saat sleep edilir ve
    calistirma asili kalir.
    """
    slept = []
    limiter = RateLimiter(_limits(rpd=1), sleep=slept.append, max_wait_s=300.0)
    first = limiter.reserve(*KEY, 10, 10)
    limiter.settle(first, actual_tokens=20)

    with pytest.raises(rate_limiter.QuotaExhausted) as excinfo:
        limiter.reserve(*KEY, 10, 10)

    assert slept == [], "uzun bekleme icin sleep CAGRILMAMALI"
    assert excinfo.value.wait_s > 300.0
    assert excinfo.value.next_available_at
    assert "Gorev birakildi" in str(excinfo.value)


def test_short_wait_still_sleeps():
    """Ust sinirin ALTINDAKI bekleme normal sekilde bloklar (davranis degismedi)."""
    slept = []
    now = {"t": 0.0}
    limiter = RateLimiter(
        _limits(rpm=1), sleep=lambda s: (slept.append(s), now.__setitem__("t", now["t"] + s)),
        clock=lambda: now["t"], max_wait_s=300.0,
    )
    limiter.settle(limiter.reserve(*KEY, 10, 10), actual_tokens=20)
    reservation = limiter.reserve(*KEY, 10, 10)   # RPM=1 -> ~60 sn bekler
    assert slept and 0 < slept[0] <= 60.0
    assert reservation.wait_ms > 0


def test_quota_exhaustion_marks_the_model_and_records_next_available_at():
    limiter = RateLimiter(_limits(rpd=1), sleep=lambda s: None, max_wait_s=1.0)
    limiter.settle(limiter.reserve(*KEY, 10, 10), actual_tokens=20)
    with pytest.raises(rate_limiter.QuotaExhausted):
        limiter.reserve(*KEY, 10, 10)

    assert limiter.daily_quota_exhausted(*KEY) is True
    moment = limiter.stats()["daily_quota_exhausted"][f"{KEY[0]}/{KEY[1]}"]
    assert moment, "next_available_at yazilmali"
    from datetime import datetime
    assert datetime.fromisoformat(moment) > datetime.now().astimezone()


def test_semaphore_is_released_on_quota_exhaustion():
    """Kota bitiminde semafor birakilmali, yoksa model kalici kilitlenir."""
    limiter = RateLimiter(_limits(rpd=1), sleep=lambda s: None, max_wait_s=1.0)
    limiter.settle(limiter.reserve(*KEY, 10, 10), actual_tokens=20)
    for _ in range(3):
        with pytest.raises(rate_limiter.QuotaExhausted):
            limiter.reserve(*KEY, 10, 10)   # kilitlenmeden tekrar tekrar firlatmali


def test_default_max_wait_comes_from_config():
    import config
    assert RateLimiter({})._max_wait_s == config.LIMITER_MAX_WAIT_SECONDS
    assert config.LIMITER_MAX_WAIT_SECONDS < 24 * 3600, "24 saatlik bekleme asla"
