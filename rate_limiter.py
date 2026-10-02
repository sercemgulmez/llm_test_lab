"""FREE_ONLY saglayicilar icin proaktif hiz/kota limitoru.

NEDEN PROAKTIF: free tier'da 429 yemek yalnizca gecikme degil, kayip demektir —
gunluk kota (RPD/TPD) tukenirse o model o gun bir daha kosmaz. Bu yuzden limitor
istegi GONDERMEDEN once yer ayirir ve gerekiyorsa bekler; 429'a tepki vermek
ikincil savunmadir.

PENCERE SECIMI: RPM/TPM icin kayan 60 saniye, RPD/TPD icin kayan 24 saat.
Takvim sifirlanmasi VARSAYILMAZ. Gemini dokumaninda RPD'nin "midnight Pacific
time"da sifirlandigi yaziyor, Groq'ta boyle bir kural dokumante degil; kayan
pencere ikisinde de MUHAFAZAKAR taraftir (hicbir zaman olmayan kotayi var
gostermez), bu yuzden varsayilan odur.

TOKEN REZERVASYONU: cagri oncesinde gercek kullanim bilinemez, bu yuzden
girdi tahmini (karakter/4 x guvenlik katsayisi) + cikti TAVANI (max_tokens)
rezerve edilir. Yanit gelince gercek kullanimla duzeltilir. Gemini'de dusunme
token'lari max_output_tokens ICINDE sayildigi icin (resmi dokuman) rezervasyona
ayrica eklenmez.

Limitor HATA FIRLATMAZ: bekleme bloklayicidir (sleep).
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, Optional, Tuple

import config

_logger = logging.getLogger(__name__)

LIMITS_FILE = "free_tier_limits.json"
USAGE_DIR_NAME = ".limits"
MINUTE_S = 60.0
DAY_S = 24 * 60 * 60.0

# published_tpd BILEREK zorunlu DEGIL: bazi saglayicilar gunluk token limitini
# yayimlamiyor. Bilinmiyorsa None kalir ve SINIRSIZ SAYILMAZ (bkz. ModelLimits).
REQUIRED_FIELDS = ("published_rpm", "published_rpd", "published_tpm")
OPTIONAL_FIELDS = ("published_tpd",)


class MissingRateLimits(RuntimeError):
    """FREE_ONLY bir model icin yayimlanan limit degerleri eksik."""


class ImpossibleRequest(RuntimeError):
    """Tek bir istek bile TPM limitine sigmiyor; hicbir zaman basarili olamaz."""


class QuotaExhausted(RuntimeError):
    """Bugun bu modelde kota kalmadi; beklemek yerine gorev birakilir.

    Limitor saatlerce ya da gunlerce bloklamaz: bekleme suresi
    config.LIMITER_MAX_WAIT_SECONDS'i asarsa bu firlatilir. Cagiran taraf gorevi
    'altyapi' isaretler, `next_available_at` yazar ve kosu biter; zamanlayici
    sonraki firsatta devam eder.
    """

    def __init__(self, message: str, next_available_at: str, wait_s: float) -> None:
        super().__init__(message)
        self.next_available_at = next_available_at
        self.wait_s = wait_s


@dataclass(frozen=True)
class ModelLimits:
    provider: str
    model: str
    published_rpm: int
    published_rpd: int
    published_tpm: int
    published_tpd: Optional[int]   # None = saglayici yayimlamamis, BILINMIYOR
    safety_factor: float
    tpm_counts: str  # "total" | "input_only"

    def _effective(self, published: int) -> int:
        return max(1, int(math.floor(published * self.safety_factor)))

    @property
    def effective_rpm(self) -> int:
        return self._effective(self.published_rpm)

    @property
    def effective_rpd(self) -> int:
        return self._effective(self.published_rpd)

    @property
    def effective_tpm(self) -> int:
        return self._effective(self.published_tpm)

    @property
    def tpd_known(self) -> bool:
        return self.published_tpd is not None

    @property
    def effective_tpd(self) -> Optional[int]:
        """Gunluk token limiti. BILINMIYORSA None doner — SINIRSIZ DEMEK DEGILDIR.

        Limitor bu durumda TPD penceresi uygulayamaz; saglayici yayimlanmamis
        bir gunluk limit uygularsa bunu ancak 429 ile ogreniriz. O yol da
        muhafazakar davranir (kota turu ayirt edilemezse GUN kabul edilir).
        """
        if self.published_tpd is None:
            return None
        return self._effective(self.published_tpd)

    def to_dict(self) -> dict:
        return {
            "provider": self.provider,
            "model": self.model,
            "published_rpm": self.published_rpm,
            "published_rpd": self.published_rpd,
            "published_tpm": self.published_tpm,
            "published_tpd": self.published_tpd,
            "tpd_known": self.tpd_known,
            "safety_factor": self.safety_factor,
            "tpm_counts": self.tpm_counts,
            "effective_rpm": self.effective_rpm,
            "effective_rpd": self.effective_rpd,
            "effective_tpm": self.effective_tpm,
            "effective_tpd": self.effective_tpd,  # None = bilinmiyor, sinirsiz DEGIL
        }


def load_limits(path: str | Path = LIMITS_FILE) -> Dict[Tuple[str, str], ModelLimits]:
    """free_tier_limits.json -> {(provider, model): ModelLimits}.

    Eksik (null) alani olan modeller ATLANIR; hangi alanlarin eksik oldugu
    `missing_fields()` ile sorulur. Boylece dosya sema olarak hazir ama
    degerleri henuz girilmemis olabilir.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    default_factor = float(data.get("safety_factor_default") or 0.8)
    result: Dict[Tuple[str, str], ModelLimits] = {}
    for provider, block in (data.get("providers") or {}).items():
        tpm_counts = str(block.get("tpm_counts") or "total")
        for model, values in (block.get("models") or {}).items():
            if any(values.get(field_name) is None for field_name in REQUIRED_FIELDS):
                continue
            tpd = values.get("published_tpd")
            factor = values.get("safety_factor")
            result[(provider, model)] = ModelLimits(
                provider=provider,
                model=model,
                published_rpm=int(values["published_rpm"]),
                published_rpd=int(values["published_rpd"]),
                published_tpm=int(values["published_tpm"]),
                published_tpd=None if tpd is None else int(tpd),
                safety_factor=float(default_factor if factor is None else factor),
                tpm_counts=tpm_counts,
            )
    return result


def missing_fields(path: str | Path = LIMITS_FILE) -> Dict[Tuple[str, str], list]:
    """Hangi (saglayici, model) icin hangi alanlarin bos oldugu."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    gaps: Dict[Tuple[str, str], list] = {}
    for provider, block in (data.get("providers") or {}).items():
        for model, values in (block.get("models") or {}).items():
            empty = [f for f in REQUIRED_FIELDS if values.get(f) is None]
            if empty:
                gaps[(provider, model)] = empty
    return gaps


# ── Kayan pencere ─────────────────────────────────────────────────────────


class _SlidingWindow:
    """Belirli bir sure icindeki toplami tutan kayan pencere."""

    def __init__(self, duration_s: float) -> None:
        self.duration_s = duration_s
        self._events: Deque[Tuple[float, int]] = deque()
        self._total = 0

    def prune(self, now: float) -> None:
        cutoff = now - self.duration_s
        while self._events and self._events[0][0] <= cutoff:
            self._total -= self._events.popleft()[1]

    def total(self, now: float) -> int:
        self.prune(now)
        return self._total

    def add(self, now: float, amount: int) -> None:
        if amount <= 0:
            return
        self._events.append((now, amount))
        self._total += amount

    def adjust_last(self, delta: int) -> None:
        """Rezervasyonu gercek kullanimla duzeltir (son kaydi degistirir)."""
        if not self._events or delta == 0:
            return
        timestamp, amount = self._events[-1]
        corrected = max(0, amount + delta)
        self._events[-1] = (timestamp, corrected)
        self._total += corrected - amount

    def wait_for(self, now: float, amount: int, limit: int) -> float:
        """`amount` kadarini eklemek icin beklenmesi gereken sure (saniye)."""
        self.prune(now)
        if self._total + amount <= limit:
            return 0.0
        freed = 0
        for timestamp, event_amount in self._events:
            freed += event_amount
            if self._total - freed + amount <= limit:
                # Bu olay pencereden cikinca yer acilir.
                return max(0.0, timestamp + self.duration_s - now)
        # Tum pencere bosalsa bile sigmiyor -> cagiran ImpossibleRequest'e bakar.
        return float("inf")


@dataclass
class Reservation:
    key: Tuple[str, str]
    requests: int
    tokens: int
    wait_ms: int
    reserved_input: int
    reserved_output: int
    settled: bool = False


@dataclass
class _ModelState:
    limits: ModelLimits
    rpm: _SlidingWindow = field(init=False)
    rpd: _SlidingWindow = field(init=False)
    tpm: _SlidingWindow = field(init=False)
    tpd: _SlidingWindow = field(init=False)
    semaphore: threading.Semaphore = field(default_factory=lambda: threading.Semaphore(1))
    reactive_429: int = 0
    daily_quota_exhausted_at: Optional[float] = None
    next_available_at: Optional[str] = None

    def __post_init__(self) -> None:
        self.rpm = _SlidingWindow(MINUTE_S)
        self.rpd = _SlidingWindow(DAY_S)
        self.tpm = _SlidingWindow(MINUTE_S)
        self.tpd = _SlidingWindow(DAY_S)


class RateLimiter:
    """(saglayici, model) basina proaktif hiz/kota yonetimi.

    Eszamanlilik: FREE_ONLY her (saglayici, model) icin en fazla 1 eszamanli
    cagri (semafor). Farkli modeller/saglayicilar paralel kosmaya devam eder;
    ucretli generator'lar limitore hic ugramaz.
    """

    def __init__(
        self,
        limits: Dict[Tuple[str, str], ModelLimits],
        usage_path: Optional[Path] = None,
        sleep=time.sleep,
        clock=time.monotonic,
        max_wait_s: Optional[float] = None,
    ) -> None:
        self._limits = dict(limits)
        self._states: Dict[Tuple[str, str], _ModelState] = {
            key: _ModelState(limits=value) for key, value in self._limits.items()
        }
        self._lock = threading.Lock()
        self._sleep = sleep
        self._clock = clock
        self._usage_path = Path(usage_path) if usage_path else None
        self._max_wait_s = (
            config.LIMITER_MAX_WAIT_SECONDS if max_wait_s is None else float(max_wait_s)
        )
        self.total_wait_ms = 0
        if self._usage_path is not None:
            self._load_usage()

    # ── Kalicilik ────────────────────────────────────────────────────────

    def _load_usage(self) -> None:
        """--resume: ayni kosunun 24 SAATLIK kullanimi sifirlanmamali.

        Gunluk kota kosuyla degil hesapla ilgilidir; yeniden baslatinca sifirdan
        saymak RPD/TPD'yi sessizce iki katina cikarir.
        """
        if self._usage_path is None or not self._usage_path.is_file():
            return
        now = self._clock()
        wall_now = time.time()
        loaded = 0
        for line in self._usage_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue  # cokme kalintisi yarim satir
            key = (record.get("provider", ""), record.get("model", ""))
            state = self._states.get(key)
            if state is None:
                continue
            age = wall_now - float(record.get("wall_ts") or 0.0)
            if age < 0 or age >= DAY_S:
                continue  # 24 saatlik pencerenin disinda kalmis
            timestamp = now - age
            state.rpd.add(timestamp, int(record.get("requests") or 0))
            state.tpd.add(timestamp, int(record.get("tokens") or 0))
            if age < MINUTE_S:
                state.rpm.add(timestamp, int(record.get("requests") or 0))
                state.tpm.add(timestamp, int(record.get("tokens") or 0))
            loaded += 1
        if loaded:
            _logger.info("  [limitor] onceki kullanim yuklendi: %d kayit", loaded)

    def _append_usage(self, key: Tuple[str, str], requests: int, tokens: int) -> None:
        if self._usage_path is None:
            return
        try:
            self._usage_path.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "wall_ts": time.time(),
                "provider": key[0],
                "model": key[1],
                "requests": requests,
                "tokens": tokens,
            }
            with self._usage_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                import os
                os.fsync(handle.fileno())
        except Exception as exc:  # noqa: BLE001 - kullanim kaydi kosuyu durdurmamali
            _logger.error("  [limitor] kullanim kaydi yazilamadi: %s", type(exc).__name__)

    # ── Sorgular ─────────────────────────────────────────────────────────

    def is_managed(self, provider: str, model: str) -> bool:
        return (provider, model) in self._states

    def limits_for(self, provider: str, model: str) -> Optional[ModelLimits]:
        return self._limits.get((provider, model))

    def effective_limits(self) -> dict:
        return {f"{p}/{m}": limit.to_dict() for (p, m), limit in sorted(self._limits.items())}

    def preflight(self, provider: str, model: str, reserved_tokens: int) -> None:
        """Tek bir istegin TPM'e sigip sigmadigi.

        Sigmiyorsa bu istek HICBIR ZAMAN basarili olamaz; beklemek anlamsizdir.
        """
        limits = self._limits.get((provider, model))
        if limits is None:
            return
        if reserved_tokens > limits.effective_tpm:
            raise ImpossibleRequest(
                f"{provider}/{model}: tek istek {reserved_tokens} token rezerve ediyor ama "
                f"efektif TPM {limits.effective_tpm} (yayimlanan {limits.published_tpm} x "
                f"guvenlik payi {limits.safety_factor}). Bu istek hicbir zaman basarili olamaz."
            )

    def stats(self) -> dict:
        with self._lock:
            return {
                "total_wait_ms": self.total_wait_ms,
                "reactive_429": {
                    f"{p}/{m}": state.reactive_429
                    for (p, m), state in sorted(self._states.items())
                },
                "daily_quota_exhausted": {
                    f"{p}/{m}": state.next_available_at
                    for (p, m), state in sorted(self._states.items())
                    if state.next_available_at
                },
            }

    # ── Rezervasyon ──────────────────────────────────────────────────────

    def reserve(
        self,
        provider: str,
        model: str,
        input_tokens_estimate: int,
        output_tokens_ceiling: int,
    ) -> Reservation:
        """Yer acilana kadar BLOKLAR, sonra rezervasyonu kaydeder.

        Hata firlatmaz (ImpossibleRequest on kontrolde, reserve'den once bakilir).
        """
        key = (provider, model)
        state = self._states.get(key)
        if state is None:
            # Yonetilmeyen (ucretli) model: limitor devrede degil.
            return Reservation(key, 0, 0, 0, 0, 0, settled=True)

        limits = state.limits
        tokens = (
            input_tokens_estimate
            if limits.tpm_counts == "input_only"
            else input_tokens_estimate + output_tokens_ceiling
        )

        state.semaphore.acquire()  # model basina tek eszamanli cagri
        waited_s = 0.0
        try:
            while True:
                with self._lock:
                    now = self._clock()
                    waits = [
                        state.rpm.wait_for(now, 1, limits.effective_rpm),
                        state.rpd.wait_for(now, 1, limits.effective_rpd),
                        state.tpm.wait_for(now, tokens, limits.effective_tpm),
                    ]
                    if limits.effective_tpd is not None:
                        waits.append(state.tpd.wait_for(now, tokens, limits.effective_tpd))
                    wait = max(waits)
                    if wait <= 0.0:
                        state.rpm.add(now, 1)
                        state.rpd.add(now, 1)
                        state.tpm.add(now, tokens)
                        state.tpd.add(now, tokens)
                        break
                if wait == float("inf"):
                    # Gunluk pencere bosalsa bile sigmiyor: beklemek cozmez.
                    state.semaphore.release()
                    raise ImpossibleRequest(
                        f"{provider}/{model}: {tokens} token gunluk limite sigmiyor "
                        f"(efektif TPD {limits.effective_tpd})."
                    )
                if wait > self._max_wait_s:
                    # Saatlerce/gunlerce BEKLEMEYIZ. Bu, "bugun bu modelde kota
                    # kalmadi" demektir; gorev birakilir ve zamanlayici devam eder.
                    state.semaphore.release()
                    moment = _wall_clock_after(wait)
                    self.note_reactive_429(provider, model, QUOTA_DAY, moment)
                    raise QuotaExhausted(
                        f"{provider}/{model}: kota icin {wait / 3600:.1f} saat beklemek "
                        f"gerekiyor (ust sinir {self._max_wait_s / 60:.0f} dakika). "
                        f"Gorev birakildi; en erken {moment}.",
                        next_available_at=moment,
                        wait_s=wait,
                    )
                self._sleep(wait)
                waited_s += wait

            wait_ms = int(waited_s * 1000)
            with self._lock:
                self.total_wait_ms += wait_ms
            if wait_ms:
                _logger.info("  [limitor] %s/%s %.1f sn beklendi", provider, model, waited_s)
            return Reservation(
                key=key,
                requests=1,
                tokens=tokens,
                wait_ms=wait_ms,
                reserved_input=input_tokens_estimate,
                reserved_output=output_tokens_ceiling,
            )
        except BaseException:
            # reserve() icinde patlarsak semaforu birakmadan cikmayalim.
            try:
                state.semaphore.release()
            except ValueError:
                pass
            raise

    def settle(self, reservation: Reservation, actual_tokens: Optional[int] = None) -> None:
        """Cagri bitti: semafor birakilir, rezervasyon gercek kullanimla duzeltilir."""
        if reservation.settled:
            return
        reservation.settled = True
        state = self._states.get(reservation.key)
        if state is None:
            return
        try:
            if actual_tokens is not None:
                delta = int(actual_tokens) - reservation.tokens
                with self._lock:
                    state.tpm.adjust_last(delta)
                    state.tpd.adjust_last(delta)
            self._append_usage(
                reservation.key,
                reservation.requests,
                int(actual_tokens) if actual_tokens is not None else reservation.tokens,
            )
        finally:
            state.semaphore.release()

    # ── 429 tepkisi ──────────────────────────────────────────────────────

    def note_reactive_429(self, provider: str, model: str, quota_kind: str,
                          next_available_at: Optional[str] = None) -> None:
        state = self._states.get((provider, model))
        if state is None:
            return
        with self._lock:
            state.reactive_429 += 1
            if quota_kind == "gun":
                state.daily_quota_exhausted_at = self._clock()
                state.next_available_at = next_available_at

    def daily_quota_exhausted(self, provider: str, model: str) -> bool:
        state = self._states.get((provider, model))
        return bool(state and state.daily_quota_exhausted_at is not None)

    def calibrate(self, provider: str, model: str, headers: dict) -> None:
        """Saglayicinin bildirdigi KALAN kota ile pencereleri hizalar.

        Yalnizca DOKUMANTE basliklar kullanilir (Groq). Gemini icin Google
        hicbir limit basligi belgelemedi, bu yuzden orada kalibrasyon yapilmaz.
        """
        state = self._states.get((provider, model))
        if state is None or not headers:
            return
        limits = state.limits
        remaining_requests = _header_int(headers, "x-ratelimit-remaining-requests")
        remaining_tokens = _header_int(headers, "x-ratelimit-remaining-tokens")
        with self._lock:
            now = self._clock()
            if remaining_requests is not None:
                # Groq dokumani: bu basligi RPD'yi gosterir.
                used = max(0, limits.published_rpd - remaining_requests)
                _reset_window(state.rpd, now, used, DAY_S)
            if remaining_tokens is not None:
                # Groq dokumani: bu baslik TPM'i gosterir.
                used = max(0, limits.published_tpm - remaining_tokens)
                _reset_window(state.tpm, now, used, MINUTE_S)


def _reset_window(window: _SlidingWindow, now: float, used: int, duration_s: float) -> None:
    window._events.clear()
    window._total = 0
    window.add(now, used)


def _header_int(headers: dict, name: str) -> Optional[int]:
    for key, value in (headers or {}).items():
        if str(key).lower() == name:
            try:
                return int(float(str(value).strip()))
            except (TypeError, ValueError):
                return None
    return None


# ── 429 gövdesi ve basliklarindan bilgi cikarma ──────────────────────────

QUOTA_MINUTE = "dakika"
QUOTA_DAY = "gun"
QUOTA_UNKNOWN = "bilinmiyor"

# Groq'un DOKUMANTE ettigi basliklar (console.groq.com/docs/rate-limits).
# Gemini icin Google hicbir limit basligi belgelemedi; oradan okunan basliklar
# yorumlanmaz.
GROQ_DOCUMENTED_HEADERS = (
    "retry-after",
    "x-ratelimit-limit-requests",
    "x-ratelimit-limit-tokens",
    "x-ratelimit-remaining-requests",
    "x-ratelimit-remaining-tokens",
    "x-ratelimit-reset-requests",
    "x-ratelimit-reset-tokens",
)


def response_headers(exc_or_resp) -> dict:
    """Istisnadan ya da yanittan HTTP basliklarini cikarir; bulamazsa {}."""
    response = getattr(exc_or_resp, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        sdk_response = getattr(exc_or_resp, "sdk_http_response", None)
        headers = getattr(sdk_response, "headers", None)
    if headers is None:
        headers = getattr(exc_or_resp, "headers", None)
    if headers is None:
        return {}
    try:
        return {str(k).lower(): str(v) for k, v in dict(headers).items()}
    except Exception:  # noqa: BLE001 - baslik okunamiyorsa bos don
        return {}


def known_rate_limit_headers(provider: str, headers: dict) -> dict:
    """Yalnizca DOKUMANTE baslik adlarini suzer.

    Belgelenmemis basliklara anlam yuklenmez; Gemini icin bos doner.
    """
    if provider != "Groq":
        return {}
    return {name: headers[name] for name in GROQ_DOCUMENTED_HEADERS if name in headers}


def retry_after_seconds(exc) -> Optional[float]:
    """Dokumante `retry-after` basligi (saniye). Yoksa None."""
    headers = response_headers(exc)
    value = headers.get("retry-after")
    if value is None:
        return None
    try:
        return max(0.0, float(str(value).strip()))
    except (TypeError, ValueError):
        return None  # HTTP-date bicimi: yorumlamiyoruz, uydurmuyoruz


def quota_kind(exc, provider: str) -> str:
    """429'un DAKIKA kotasi mi GUNLUK kota mi oldugu.

    Groq: dokumante `x-ratelimit-remaining-requests` basligi RPD'yi gosterir;
    0 ise gunluk kota bitmistir.
    Gemini: baslik dokumante degil, bu yuzden hata govdesindeki metne bakilir.
    AYIRT EDILEMEZSE MUHAFAZAKAR davranilir ve GUN kabul edilir — dakika sanip
    beklemek gunluk kotasi bitmis bir modeli bosuna denemek olurdu.
    """
    headers = response_headers(exc)
    if provider == "Groq":
        remaining = _header_int(headers, "x-ratelimit-remaining-requests")
        if remaining is not None:
            return QUOTA_DAY if remaining <= 0 else QUOTA_MINUTE
    message = str(exc).lower()
    if "perday" in message.replace("_", "").replace(" ", "") or "per day" in message:
        return QUOTA_DAY
    if "perminute" in message.replace("_", "").replace(" ", "") or "per minute" in message:
        return QUOTA_MINUTE
    if retry_after_seconds(exc) is not None:
        # Saglayici bekleyecegimiz sureyi soyluyorsa bu dakika kotasidir.
        return QUOTA_MINUTE
    return QUOTA_DAY  # muhafazakar


def _wall_clock_after(seconds: float) -> str:
    """Duvar saatine gore `seconds` sonrasi (ISO). next_available_at icin."""
    from datetime import datetime, timedelta

    return (datetime.now().astimezone() + timedelta(seconds=seconds)).isoformat(
        timespec="seconds"
    )
