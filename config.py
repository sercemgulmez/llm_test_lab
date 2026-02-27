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

# Güvenlik üst sınırı
MAX_CASES_PER_OPERATION: int = 10

# Varsayılan çıktı klasörü
OUTPUT_DIR: str = "outputs"

# HTTP istek zaman aşımı (saniye)
REQUEST_TIMEOUT: int = 10

# LLM API retry ayarları
RETRY_MAX_ATTEMPTS: int = 3
RETRY_BACKOFF_SECONDS: float = 2.0
