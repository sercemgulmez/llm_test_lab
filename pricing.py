"""Token maliyeti ve fiyat tablosu.

B4(c) KURALI: fiyatlar resmi saglayici sayfalarindan cekilip raporlandi, ancak
KULLANICININ ACIK ONAYI OLMADAN BU DOSYAYA SABITLENMEZ. `PRICE_TABLE` bos
oldugu surece maliyet alanlari None doner ve defter `pricing_available=False`
yazar. Maliyet HICBIR KOSULDA tahmin edilmez.

B4(d) KURALI: her cagri icin iki ayri tutar tutulur.
  * cost_usd_billed              — gercekten FATURALANAN tutar. Free-tier'da 0.0.
  * cost_usd_list_equivalent     — ayni kullanimin liste fiyatiyla karsiligi.
`cost_basis` hangisinin gecerli oldugunu soyler.

Girdi/cikti ayrimi yoksa (TokenUsage.split_available=False) maliyet
HESAPLANMAZ: tek bir toplam token sayisini girdi/cikti fiyatlarina bolmek
uydurma olurdu.
"""

from __future__ import annotations

from dataclasses import dataclass

from models import TokenUsage

# Maliyet dayanaklari
BASIS_BILLED = "billed"
BASIS_FREE_TIER = "free_tier_list_equivalent"
BASIS_NO_PRICE = "fiyat_tablosu_yok"
BASIS_NO_SPLIT = "token_ayrimi_yok"


@dataclass(frozen=True)
class ModelPrice:
    """Bir model kimligi icin 1M token basina USD fiyat.

    `billed=False` ise kosu free-tier ile yapiliyor demektir: gercek fatura 0,
    ama liste-esdeger tutar yine de hesaplanir (tezde karsilastirma icin).
    """

    input_usd_per_mtok: float
    output_usd_per_mtok: float
    billed: bool
    source_url: str
    fetched_on: str  # ISO tarih — fiyatin cekildigi gun


@dataclass(frozen=True)
class CallCost:
    cost_usd_billed: float | None
    cost_usd_list_equivalent: float | None
    cost_basis: str
    pricing_available: bool

    def to_dict(self) -> dict:
        return {
            "cost_usd_billed": self.cost_usd_billed,
            "cost_usd_list_equivalent": self.cost_usd_list_equivalent,
            "cost_basis": self.cost_basis,
            "pricing_available": self.pricing_available,
        }


UNPRICED = CallCost(None, None, BASIS_NO_PRICE, False)

# ─────────────────────────────────────────────────────────────────────────────
# FIYAT TABLOSU — BOS. Kullanici onayi gelene kadar doldurulmaz (B4(c)).
# Anahtar: config'teki TAM model kimligi (ornegin "gpt-4.1", "claude-haiku-4-5").
# ─────────────────────────────────────────────────────────────────────────────
PRICE_TABLE_APPROVED_ON: str | None = None
PRICE_TABLE: dict[str, ModelPrice] = {}


def price_table_ready() -> bool:
    return bool(PRICE_TABLE)


def cost_for(model: str, usage: TokenUsage) -> CallCost:
    """Bir cagrinin maliyeti. Bilinmeyeni UYDURMAZ."""
    price = PRICE_TABLE.get(model or "")
    if price is None:
        return UNPRICED
    if not usage.split_available:
        # Girdi ve cikti ayri fiyatlandirilir; ayrim yoksa hesap yapilamaz.
        return CallCost(None, None, BASIS_NO_SPLIT, False)

    list_equivalent = (
        usage.input_tokens * price.input_usd_per_mtok
        + usage.billable_output_tokens * price.output_usd_per_mtok
    ) / 1_000_000.0
    list_equivalent = round(list_equivalent, 8)
    if price.billed:
        return CallCost(list_equivalent, list_equivalent, BASIS_BILLED, True)
    return CallCost(0.0, list_equivalent, BASIS_FREE_TIER, True)
