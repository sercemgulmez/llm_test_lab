"""Merkezi yapılandırma sabitleri - Tez Uyumlu (9 Generator + 2 Prompt)"""

from __future__ import annotations
from pathlib import Path

# ============= 9 GENERATOR (8 LLM + 1 Traditional) =============

# OpenAI (2 models)
OPENAI_MODELS = [
    "gpt-4.1",
    "gpt-4o-mini",
]

# Google Gemini (2 models)
GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-3.5-flash-lite",
]

# Groq (2 models)
GROQ_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
]

# Anthropic Claude (2 models)
CLAUDE_MODELS = [
    "claude-sonnet-4-5",
    "claude-haiku-4-5",
]

# Traditional Template Baseline (1 model)
TRADITIONAL_MODELS = [
    "template-baseline",
]

# Validation
TOTAL_LLM_MODELS = len(OPENAI_MODELS) + len(GEMINI_MODELS) + len(GROQ_MODELS) + len(CLAUDE_MODELS)
TOTAL_GENERATORS = TOTAL_LLM_MODELS + len(TRADITIONAL_MODELS)
assert TOTAL_GENERATORS == 9, f"Expected 9 generators, got {TOTAL_GENERATORS}"
assert TOTAL_LLM_MODELS == 8, f"Expected 8 LLM, got {TOTAL_LLM_MODELS}"

# Ucretli saglayicilar. Groq ve Gemini free-tier ile kosuluyor; bir yeniden
# kosu bu ikisinde bedelsiz, asagidakilerde PARA HARCAR ve acik onay ister.
PAID_PROVIDERS: set[str] = {"OpenAI", "Claude"}

# Yalnizca FREE TIER ile kosulan saglayicilar. Bu listedeki saglayicilarda
# fatura tutari HICBIR KOSULDA hesaplanmaz (pricing.cost_for), cost_usd_billed
# her zaman 0.0 ve cost_basis free_tier_list_equivalent olur. Fiyat tablosuna
# yanlislikla billed=True yazilsa bile bu liste onu gecersiz kilar.
FREE_ONLY_PROVIDERS: set[str] = {"Gemini", "Groq"}

assert not (PAID_PROVIDERS & FREE_ONLY_PROVIDERS), (
    "Bir saglayici hem ucretli hem yalnizca-ucretsiz olamaz: "
    f"{sorted(PAID_PROVIDERS & FREE_ONLY_PROVIDERS)}"
)


def provider_for_model(model: str) -> str:
    """Model kimliginden saglayici etiketi; bilinmiyorsa bos string.

    GENERATOR_REGISTRY'yi import etmeden calisir (dairesel import olmasin diye
    model listelerine bakar).
    """
    if model in OPENAI_MODELS:
        return "OpenAI"
    if model in GEMINI_MODELS:
        return "Gemini"
    if model in GROQ_MODELS:
        return "Groq"
    if model in CLAUDE_MODELS:
        return "Claude"
    return ""


def is_free_only_model(model: str) -> bool:
    return provider_for_model(model) in FREE_ONLY_PROVIDERS

# ============= BUTCE SIGORTASI (K6) =============
# Senaryo A butcesi $80. Esikler DEFTERDEKI toplam faturalanan harcamaya gore
# degerlendirilir (bkz. budget.py). "stop" asildiginda kalan uretim gorevleri
# iptal edilir; o ana kadar uretilen satirlar yine de CSV'ye yazilir.
BUDGET_THRESHOLDS: dict[str, float] = {
    "warn": 40.0,       # butcenin %50'si
    "hard_warn": 64.0,  # butcenin %80'i
    "stop": 80.0,       # sert tavan
}

# ============= 2 PROMPT STRATEGIES =============
PROMPT_VARIANTS: dict[str, dict] = {
    "basic": {
        "name": "Basic Functional Testing",
        "description": "Positive scenarios, happy path, expected flows",
        "focus": "Temel fonksiyonel senaryolar üret; mutlu path ve sözleşmeye uygun davranışları test et.",
        "tests_per_model": 10,
    },
    "edge_focused": {
        "name": "Edge Case & Error Testing",
        "description": "Boundary values, errors, edge cases, auth failures",
        "focus": "Negatif, sınır değeri ve kimlik doğrulama odaklı senaryolar üret; farklı hata kodlarını da kapsa.",
        "tests_per_model": 10,
    },
}

# ============= TEST GENERATION BUDGET =============
GENERATION_BUDGET = {
    "llm_tests_per_prompt": 10,           # 10 basic + 10 edge_focused = 20 per model
    "llm_models_count": 8,
    "llm_total_tests": 160,               # 8 × 20
    "traditional_tests": 5,
    "total_tests": 165,                   # 160 + 5
}

NUM_CASES_PER_OPERATION: int = 10
MAX_CASES_PER_OPERATION: int | None = None

# ============= EVALUATION METRICS PARAMETERS =============
DIVERSITY_LAMBDA = 0.6  # Balance: 60% repeat minimization, 40% unique signatures
SPECTRAL_CONVERGENCE_EPSILON = 1e-6
SPECTRAL_MAX_ITERATIONS = 100

SPECTRAL_METRIC_WEIGHTS = {
    "executability": 0.25,
    "pass_rate": 0.25,
    "coverage": 0.25,
    "diversity": 0.15,
    "similarity": 0.10,
}

# ============= CSV OUTPUT SCHEMAS =============
EXECUTED_TESTCASES_FIELDS = [
    'generator', 'prompt_variant', 'operation_id', 'http_method', 'path', 'tc_id', 'title',
    'request_body', 'expected_status', 'expected_result', 'url',
    'actual_status', 'pass', 'tokens_used',
]

GENERATOR_METRICS_FIELDS = [
    'generator', 'total_tests', 'pass_count', 'fail_count', 'pass_rate',
    'total_tokens', 'avg_tokens_per_tc',
    'expected_status_distribution', 'actual_status_distribution',
]

# ============= PATHS & RUNTIME =============
OUTPUT_DIR: str = "outputs"
PROJECT_ROOT: Path = Path(__file__).resolve().parent
UPLOAD_DIR: str = "uploads"
ALLOWED_UPLOAD_EXTENSIONS: set[str] = {".txt", ".curl", ".http"}
MAX_UPLOAD_BYTES: int = 1 * 1024 * 1024
JOB_ID_BYTES: int = 16
JOB_TOKEN_BYTES: int = 24
ALLOWED_OUTPUT_ROOTS: tuple[Path, ...] = (PROJECT_ROOT,)

# ============= WEB & THREADING =============
MAX_PARALLEL_JOBS: int = 1
REQUEST_TIMEOUT: int = 10
RETRY_MAX_ATTEMPTS: int = 3
RETRY_BACKOFF_SECONDS: float = 8.0
MAX_PARALLEL_WORKERS: int = 9
MAX_PARALLEL_GENERATORS: int = 3

# ============= LLM ISTEMCI TIMEOUT'U =============
# REQUEST_TIMEOUT (10 sn) runner.py'nin HTTPBIN test cagrilari icindir ve oyle
# kalir. LLM uretim cagrilari bambaska bir is: 3000 token uretmek 10 saniyeye
# sigmaz ve sigmadiginda TIMEOUT -> 'altyapi' hatasi uretir. Bu yuzden LLM
# istemcileri AYRI bir timeout kullanir.
#
# Kurulu SDK'larin kendi varsayilanlari (kaynak koddan, 29.09.2026):
#   openai 1.109.1      -> connect=5, read=600, write=600, pool=600
#   anthropic 0.120.2   -> connect=5, read=600, write=600, pool=600
#   google-genai 1.47.0 -> HttpOptions.timeout=None -> httpx'te SONSUZ
#   groq                -> ayri paket yok; OpenAI SDK + base_url kullaniliyor
# Asagidaki degerler bu varsayilanlarin en dusugunun altina inmez.
LLM_REQUEST_TIMEOUT: dict[str, float] = {
    "connect": 10.0,
    "read": 600.0,
    "write": 600.0,
    "pool": 600.0,
}

# Saglayici bazinda override (bos = LLM_REQUEST_TIMEOUT gecerli).
# Anahtar: generator'in _provider_label degeri kucuk harfle ("openai", "groq",
# "gemini", "claude").
LLM_REQUEST_TIMEOUT_BY_PROVIDER: dict[str, dict[str, float]] = {}


def llm_timeout_for(provider: str) -> dict[str, float]:
    """Bir saglayici icin efektif LLM timeout sozlugu."""
    override = LLM_REQUEST_TIMEOUT_BY_PROVIDER.get((provider or "").lower())
    return dict(override or LLM_REQUEST_TIMEOUT)

MAX_TOKENS_BY_PROVIDER: dict[str, int] = {
    "openai": 16384,
    "gemini": 8192,
    "claude": 8192,
    "groq": 8192,
}

def normalize_num_cases(value: object, default: int = NUM_CASES_PER_OPERATION) -> int:
    """Pozitif testcase sayısı döner; geçersiz girişte default kullanır."""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(1, parsed)

print("✓ Config loaded: 9 generators, 2 prompt strategies, 165 test capacity")
