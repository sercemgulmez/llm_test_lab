"""Kosu ici butce sigortasi (K6).

KAPSAM VE SINIR — once bunu okuyun:

1. Bu bir STOP-LOSS'tur, on-yetkilendirme DEGILDIR. Bir cagrinin maliyeti
   ancak yanit geldikten sonra bilinir; dolayisiyla esik, esigi asan cagri
   YAPILDIKTAN SONRA tetiklenir. Esikten sonra en fazla birkac cagrilik
   (paralellik kadar) asim mumkundur.

2. Harcamanin TEK kaynagi cagri defteridir (B4(a)): iptal edilen nesillerin
   ve yeniden kosulan gorevlerin parasi da sayilir.

3. Fiyat tablosu (pricing.PRICE_TABLE) BOSSA bu sigorta ATIL'dir: defter
   maliyet goremez, dolayisiyla hicbir esik tetiklenmez. Bu durumda
   `BudgetGuard` acik bir UYARI verir ve durumu `armed=False` olarak bildirir.
"""

from __future__ import annotations

import logging
import threading

import config
import pricing

_logger = logging.getLogger(__name__)


ENV_BY_LEVEL = {
    "warn": "BUDGET_WARN_USD",
    "hard_warn": "BUDGET_HARD_WARN_USD",
    "stop": "BUDGET_STOP_USD",
}


def resolve_thresholds(overrides: dict | None = None) -> dict:
    """Efektif butce esikleri: config varsayilani < env < komut satiri.

    Esikler artan sirada olmali; degilse ValueError. Boylece "stop"u yanlislikla
    "warn"in altina indirip kosuyu hemen durduran bir yazim hatasi sessizce
    gecmez.
    """
    import os

    effective = dict(config.BUDGET_THRESHOLDS)
    for level, env_var in ENV_BY_LEVEL.items():
        raw = os.getenv(env_var)
        if raw is None or not str(raw).strip():
            continue
        try:
            effective[level] = float(raw)
        except ValueError as exc:
            raise ValueError(f"{env_var} sayisal olmali, gelen: {raw!r}") from exc
    for level, value in (overrides or {}).items():
        if value is not None:
            effective[level] = float(value)

    ordered = (effective["warn"], effective["hard_warn"], effective["stop"])
    if not (0 < ordered[0] <= ordered[1] <= ordered[2]):
        raise ValueError(
            f"Butce esikleri artan sirada ve pozitif olmali: warn={ordered[0]}, "
            f"hard_warn={ordered[1]}, stop={ordered[2]}"
        )
    return effective


class BudgetExceeded(RuntimeError):
    """Sert esik asildi: kosu durdurulmali."""


class BudgetGuard:
    """Defterdeki toplam harcamayi izler ve esiklerde mudahale eder.

    Esikler `config.BUDGET_THRESHOLDS`ten gelir: warn / hard_warn / stop.
    Her esik en fazla BIR KEZ loglanir; `stop` esiginde BudgetExceeded firlatir
    ve bundan sonraki her `check()` cagrisi da firlatir (yaris durumunda diger
    thread'ler de durur).
    """

    def __init__(self, ledger, thresholds: dict | None = None, enabled: bool = True) -> None:
        self._ledger = ledger
        self._thresholds = dict(thresholds or config.BUDGET_THRESHOLDS)
        self._lock = threading.Lock()
        self._announced: set[str] = set()
        self._tripped = False
        self.enabled = enabled and ledger is not None
        self.armed = self.enabled and pricing.price_table_ready()
        if self.enabled and not self.armed:
            _logger.warning(
                "  [butce] SIGORTA ATIL: fiyat tablosu bos, harcama olculemiyor. "
                "$%.0f / $%.0f / $%.0f esikleri TETIKLENMEYECEK.",
                self._thresholds.get("warn", 0), self._thresholds.get("hard_warn", 0),
                self._thresholds.get("stop", 0),
            )

    def spend(self) -> float:
        """Esiklerin olctugu tutar: FATURALANAN + muhafazakar UST TAHMIN.

        Fiyatlanamayan ve yanit alinamayan cagrilar icin sifir varsaymak
        sigortayi kor ederdi. Ust tahmin ayri bir alanda tutulur ve fatura
        tahminiyle KARISTIRILMAZ; burada yalnizca muhafazakar tarafta kalmak
        icin toplanir.
        """
        if not self.enabled:
            return 0.0
        snapshot = self._ledger.spend_so_far()
        return (
            float(snapshot.get("cost_usd_billed") or 0.0)
            + float(snapshot.get("cost_usd_guard_estimate") or 0.0)
        )

    def spend_breakdown(self) -> dict:
        """Raporlama icin: faturalanan ve ust tahmin AYRI."""
        if not self.enabled:
            return {"cost_usd_billed": 0.0, "cost_usd_guard_estimate": 0.0}
        snapshot = self._ledger.spend_so_far()
        return {
            "cost_usd_billed": float(snapshot.get("cost_usd_billed") or 0.0),
            "cost_usd_guard_estimate": float(snapshot.get("cost_usd_guard_estimate") or 0.0),
        }

    def check(self) -> None:
        """Her cagridan sonra cagrilir. Sert esikte BudgetExceeded firlatir."""
        if not self.armed:
            return
        spent = self.spend()
        stop = self._thresholds.get("stop")
        with self._lock:
            if self._tripped:
                raise BudgetExceeded(self._stop_message(spent, stop))
            for level in ("warn", "hard_warn"):
                limit = self._thresholds.get(level)
                if limit is not None and spent >= limit and level not in self._announced:
                    self._announced.add(level)
                    _logger.warning(
                        "  [butce] %s ESIGI: defter toplami $%.2f, esik $%.2f. Kosu devam ediyor.",
                        "UYARI" if level == "warn" else "SERT UYARI", spent, limit,
                    )
            if stop is not None and spent >= stop:
                self._tripped = True
                message = self._stop_message(spent, stop)
                _logger.error("  [butce] %s", message)
                raise BudgetExceeded(message)

    @staticmethod
    def _stop_message(spent: float, stop: float | None) -> str:
        return (
            f"BUTCE DURDURMA: defterdeki toplam harcama ${spent:.2f}, sert esik "
            f"${(stop or 0):.2f}. Kalan tum uretim gorevleri IPTAL edildi; o ana kadar "
            f"uretilen satirlar yine de diske yazilir."
        )
