"""Merkezi yapılandırma sabitleri."""

# OpenAI (GPT) modelleri
OPENAI_MODELS = [
    "gpt-4.1-mini",
    "gpt-4.1",
    "gpt-4o-mini",
]

# Google Gemini modelleri
GEMINI_MODELS = [
    "gemini-2.5-flash",
]

# Anthropic Claude modelleri
CLAUDE_MODELS = [
    "claude-sonnet-4-6",
    "claude-haiku-4-5-20251001",
]

# Prompt stratejileri (farklı test üretim tarzları)
PROMPT_VARIANTS: dict[str, str] = {
    "basic": "Temel fonksiyonel senaryolar üret; mutlu path ve birkaç hata durumu ekle.",
    "edge_focused": (
        "Negatif, sınır değeri ve kimlik doğrulama odaklı senaryolar üret; "
        "farklı hata kodlarını da kapsa."
    ),
}

# Operasyon başına üretilecek LLM test senaryosu sayısı
NUM_CASES_PER_OPERATION: int = 10

# Üst sınır yok; UI ve backend pozitif sayı kabul eder.
MAX_CASES_PER_OPERATION: int | None = None

# Varsayılan çıktı klasörü
OUTPUT_DIR: str = "outputs"

# HTTP istek zaman aşımı (saniye)
REQUEST_TIMEOUT: int = 10

# LLM API retry ayarları
# Backoff: 8s → 16s → (max deneme). Gemini free tier "retry in 6s" için yeterli.
RETRY_MAX_ATTEMPTS: int = 3
RETRY_BACKOFF_SECONDS: float = 8.0

# LLM operasyonları için paralel thread sayısı (I/O-bound API çağrıları)
MAX_PARALLEL_WORKERS: int = 5


def normalize_num_cases(value: object, default: int = NUM_CASES_PER_OPERATION) -> int:
    """Pozitif testcase sayısı döner; geçersiz girişte default kullanır."""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(1, parsed)
