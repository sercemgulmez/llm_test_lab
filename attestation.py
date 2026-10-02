"""ATTEST_FREE_TIER beyani.

Kod, bir hesabin gercekten free tier'da olup olmadigini GUVENILIR bicimde
oleremez: Gemini'de bunu belirleyen sey projeye faturalama bagli olup olmadigi,
Groq'ta ise planin Free olmasi ve odeme yontemi bulunmamasidir. Ikisi de yalniz
saglayici arayuzunden goruluyor.

Bu yuzden mekanizma bir BEYANDIR: kullanici arayuzden dogrular ve .env dosyasina
ATTEST_FREE_TIER=gemini,groq satirini ELLE ekler. Beyan yoksa FREE_ONLY
generator'lar hic kosmaz (bkz. main._free_tier_attestation_check).

Beyan bir kanit degil, bilincli bir onaydir; bu yuzden degeri ve okundugu an
run_info'ya ve cagri defterine yazilir ki kosu sonrasi neyin beyan edildigi
belli olsun.
"""

from __future__ import annotations

import os
from datetime import datetime

ENV_VAR = "ATTEST_FREE_TIER"

# Kullaniciya gosterilecek metin; birebir sartnameden.
GUIDANCE = (
    "Gemini: projede faturalama bağlı mı? Groq: plan Free ve ödeme yöntemi yok mu?"
)


def _raw_value() -> str:
    return (os.getenv(ENV_VAR) or "").strip()


def attested_providers() -> set[str]:
    """Beyan edilen saglayicilar, kanonik etiketleriyle ('Gemini', 'Groq')."""
    canonical = {"gemini": "Gemini", "groq": "Groq"}
    result: set[str] = set()
    for part in _raw_value().replace(";", ",").split(","):
        key = part.strip().lower()
        if key in canonical:
            result.add(canonical[key])
    return result


def is_attested(provider: str) -> bool:
    return provider in attested_providers()


def missing_for(providers: set[str]) -> list[str]:
    """Beyani eksik olan saglayicilar (sirali)."""
    return sorted(providers - attested_providers())


def attestation_record() -> dict:
    """run_info ve defter icin beyan kaydi.

    Anahtar veya baska bir sir ICERMEZ; yalnizca saglayici adlari yazilir.
    """
    return {
        "env_var": ENV_VAR,
        "declared_providers": sorted(attested_providers()),
        "declared_raw_present": bool(_raw_value()),
        "read_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
