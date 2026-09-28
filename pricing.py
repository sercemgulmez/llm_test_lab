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
    fetched_date: str           # ISO tarih — fiyatin resmi sayfadan cekildigi gun
    second_source_url: str = ""  # bagimsiz dogrulama (bos = yalnizca tek kaynak)
    verified: bool = True        # False ise tutar DOGRULANMAMIS sayilir


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
# FIYAT TABLOSU
#
# Anahtar: config'teki TAM model kimligi. Tutarlar 1M token basina USD, STANDART
# (batch/flex/priority degil) fiyatlandirma. 28.09.2026'da resmi sayfalardan
# cekildi; iki suphe uyandiran girdi bagimsiz ikinci kaynakla dogrulandi.
#
# billed=False olanlar free-tier ile kosuluyor: gercek fatura 0, ama
# cost_usd_list_equivalent yine hesaplanir (tezdeki karsilastirma icin).
# ─────────────────────────────────────────────────────────────────────────────
PRICE_TABLE_APPROVED_ON: str = "2026-09-28"
_FETCHED = "2026-09-28"

_OPENAI_PRICING = "https://developers.openai.com/api/docs/pricing"
_ANTHROPIC_PRICING = "https://platform.claude.com/docs/en/about-claude/pricing"
_GEMINI_PRICING = "https://ai.google.dev/gemini-api/docs/pricing"
_GROQ_PRICING = "https://console.groq.com/docs/models"
_VERTEX_PRICING = "https://cloud.google.com/vertex-ai/generative-ai/pricing"
_OPENROUTER_MODELS = "https://openrouter.ai/api/v1/models"

PRICE_TABLE: dict[str, ModelPrice] = {
    # ── OpenAI (UCRETLI) ────────────────────────────────────────────────────
    # gpt-4.1'in STANDART satiri OpenAI fiyat tablosunda JS ile geliyor ve ham
    # HTML'de yok; tutar model sayfasindan alindi, OpenRouter'in canli model
    # API'siyle ve ayni sayfadaki Batch satiriyla ($1/$4 = %50) dogrulandi.
    "gpt-4.1": ModelPrice(
        2.00, 8.00, billed=True,
        source_url="https://developers.openai.com/api/docs/models/gpt-4.1",
        fetched_date=_FETCHED, second_source_url=_OPENROUTER_MODELS,
    ),
    "gpt-4o-mini": ModelPrice(
        0.15, 0.60, billed=True,
        source_url=_OPENAI_PRICING, fetched_date=_FETCHED,
        second_source_url=_OPENROUTER_MODELS,
    ),

    # ── Anthropic (UCRETLI) ─────────────────────────────────────────────────
    # Sayfa gorunen adi kullanir ("Claude Sonnet 4.5"); API kimligiyle eslesme
    # model-deprecations sayfasindaki claude-sonnet-4-5-20250929 satiri uzerinden.
    "claude-sonnet-4-5": ModelPrice(
        3.00, 15.00, billed=True,
        source_url=_ANTHROPIC_PRICING, fetched_date=_FETCHED,
        second_source_url=_OPENROUTER_MODELS,
    ),
    "claude-haiku-4-5": ModelPrice(
        1.00, 5.00, billed=True,
        source_url=_ANTHROPIC_PRICING, fetched_date=_FETCHED,
        second_source_url=_OPENROUTER_MODELS,
    ),

    # ── Google Gemini (FREE TIER ile kosuluyor) ─────────────────────────────
    # Cikti satirinin basligi birebir: "Output price (including thinking tokens)".
    # Vertex AI sayfasi ayni tutari "Text output (response and reasoning)" diye
    # verir — dusunme token'inin cikti fiyatindan faturalandiginin ikinci teyidi.
    "gemini-2.5-flash": ModelPrice(
        0.30, 2.50, billed=False,
        source_url=_GEMINI_PRICING, fetched_date=_FETCHED,
        second_source_url=_VERTEX_PRICING,
    ),
    "gemini-3.5-flash-lite": ModelPrice(
        0.30, 2.50, billed=False,
        source_url=_GEMINI_PRICING, fetched_date=_FETCHED,
        second_source_url=_VERTEX_PRICING,
    ),

    # ── Groq (FREE TIER ile kosuluyor) ──────────────────────────────────────
    # gpt-oss acik agirlikli modeller: liste fiyati SAGLAYICIYA GORE DEGISIR.
    # Burada Groq'un kendi yayinladigi oran kullanilir; baska saglayicilarin
    # ayni modeli daha ucuza sunmasi bu tutari gecersiz kilmaz.
    "openai/gpt-oss-120b": ModelPrice(
        0.15, 0.60, billed=False,
        source_url=_GROQ_PRICING, fetched_date=_FETCHED,
    ),
    "openai/gpt-oss-20b": ModelPrice(
        0.075, 0.30, billed=False,
        source_url=_GROQ_PRICING, fetched_date=_FETCHED,
    ),
}


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
