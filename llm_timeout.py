"""LLM istemcileri icin HTTP timeout yardimcilari.

runner.py'nin httpbin timeout'u (config.REQUEST_TIMEOUT = 10 sn) ile KARISTIRILMAZ
ve limitorle (rate_limiter.py) de ilgisi yoktur: bu modul yalnizca saglayici
istemcisinin ag zaman asimini kurar ve bir zaman asimi gerceklestiginde hangi
asamada (baglanti/okuma) oldugunu soyler.
"""

from __future__ import annotations

import httpx

import config

# httpx'in zaman asimi istisnalari -> asama adi
_PHASE_BY_EXCEPTION = (
    (httpx.ConnectTimeout, "connect"),
    (httpx.ReadTimeout, "read"),
    (httpx.WriteTimeout, "write"),
    (httpx.PoolTimeout, "pool"),
)


def timeout_seconds_for(provider: str) -> dict[str, float]:
    """Saglayicinin efektif timeout degerleri (saniye), deftere yazilmak icin."""
    return config.llm_timeout_for(provider)


def httpx_timeout_for(provider: str) -> httpx.Timeout:
    """openai / anthropic istemcilerine verilecek httpx.Timeout nesnesi."""
    values = timeout_seconds_for(provider)
    return httpx.Timeout(
        connect=values.get("connect"),
        read=values.get("read"),
        write=values.get("write"),
        pool=values.get("pool"),
    )


def genai_timeout_ms_for(provider: str) -> int:
    """google-genai HttpOptions.timeout MILISANIYE bekler (_api_client.py:208-210).

    genai tek bir deger alir; en kisitlayici asama olan `read` kullanilir —
    `connect` daha kucuk oldugu icin onu kullanmak uretim cagrilarini erken keser.
    """
    values = timeout_seconds_for(provider)
    seconds = values.get("read") or values.get("connect") or 0.0
    return int(seconds * 1000)


def timeout_phase(exc: BaseException) -> str:
    """Zaman asiminin hangi asamada oldugu; ayirt edilemezse 'bilinmiyor'.

    Saglayici SDK'lari httpx istisnasini kendi tipleriyle sarmalayabilir, bu
    yuzden __cause__/__context__ zinciri de taranir.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        for exception_type, phase in _PHASE_BY_EXCEPTION:
            if isinstance(current, exception_type):
                return phase
        current = current.__cause__ or current.__context__
    return "bilinmiyor"
