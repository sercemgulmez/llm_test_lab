"""Merkezi yapılandırma sabitleri - Tez Uyumlu (10 Generator + 2 Prompt)"""

from __future__ import annotations
from pathlib import Path

# ============= 10 GENERATOR (9 LLM + 1 Traditional) =============

# OpenAI (3 models)
OPENAI_MODELS = [
    "gpt-4.1",
    "gpt-4o-mini",
    "gpt-4-turbo",  # ← NEW
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
assert TOTAL_GENERATORS == 10, f"Expected 10 generators, got {TOTAL_GENERATORS}"
assert TOTAL_LLM_MODELS == 9, f"Expected 9 LLM, got {TOTAL_LLM_MODELS}"

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
    "llm_models_count": 9,
    "llm_total_tests": 180,               # 9 × 20
    "traditional_tests": 5,
    "total_tests": 185,                   # 180 + 5
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
    'generator', 'operation_id', 'http_method', 'path', 'tc_id', 'title',
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

print("✓ Config loaded: 10 generators, 2 prompt strategies, 185 test capacity")
