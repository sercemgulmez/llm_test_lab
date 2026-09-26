import config
import main
from generators import GENERATOR_REGISTRY
from generators.traditional import TraditionalGenerator
from models import ApiOperation


EXPECTED_OPENAI_MODELS = ["gpt-4.1", "gpt-4o-mini"]
EXPECTED_GEMINI_MODELS = ["gemini-2.5-flash", "gemini-3.5-flash-lite"]
EXPECTED_GROQ_MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]
EXPECTED_CLAUDE_MODELS = ["claude-sonnet-4-5", "claude-haiku-4-5"]
DEPRECATED_MODELS = {
    "gpt-4.1-mini",
    "gemini-2.0-flash",
    "gemini-2.5-flash-lite",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "claude-sonnet-4-6",
    "claude-haiku-4-5-20251001",
}


def test_parse_cli_headers_accepts_lines_without_space_after_colon():
    headers = main._parse_cli_headers([
        "Authorization:Bearer token",
        "X-Trace-Id: trace-1",
    ])

    assert headers == {
        "Authorization": "Bearer token",
        "X-Trace-Id": "trace-1",
    }


def test_parse_cli_cookies_accepts_semicolon_without_space():
    cookies = main._parse_cli_cookies("session=abc;theme=dark")

    assert cookies == {
        "session": "abc",
        "theme": "dark",
    }


def test_build_llm_generators_includes_selected_groq_model():
    selected_model = config.GROQ_MODELS[0]

    generators = main._build_llm_generators([f"groq:{selected_model}"])

    assert len(generators) == len(config.PROMPT_VARIANTS)
    assert all(generator.__class__.__name__ == "GroqGenerator" for generator, _, _ in generators)
    assert all(generator.model == selected_model for generator, _, _ in generators)


def test_model_config_matches_target_experiment_set():
    assert config.OPENAI_MODELS == EXPECTED_OPENAI_MODELS
    assert config.GEMINI_MODELS == EXPECTED_GEMINI_MODELS
    assert config.GROQ_MODELS == EXPECTED_GROQ_MODELS
    assert config.CLAUDE_MODELS == EXPECTED_CLAUDE_MODELS

    configured_models = (
        config.OPENAI_MODELS
        + config.GEMINI_MODELS
        + config.GROQ_MODELS
        + config.CLAUDE_MODELS
    )
    assert len(configured_models) == 8
    assert not DEPRECATED_MODELS.intersection(configured_models)


def test_generator_registry_has_8_llms_and_1_traditional_baseline():
    assert len(GENERATOR_REGISTRY) == 9
    assert GENERATOR_REGISTRY["traditional"][0] is TraditionalGenerator

    llm_keys = [key for key in GENERATOR_REGISTRY if key != "traditional"]
    assert len(llm_keys) == 8
    assert not any(model in key for model in DEPRECATED_MODELS for key in GENERATOR_REGISTRY)


def test_traditional_generator_does_not_require_api_key(monkeypatch):
    for env_var in ("OPENAI_API_KEY", "GEMINI_API_KEY", "GROQ_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(env_var, raising=False)

    op = ApiOperation(op_id="PING", method="GET", path="/ping")

    rows = TraditionalGenerator().generate([op], "", "", 1)

    assert len(rows) == 1
    assert rows[0]["generator"] == TraditionalGenerator.GENERATOR_NAME


def test_build_llm_generators_excludes_traditional_to_prevent_double_run():
    """K2 regresyonu: Traditional main() icinde ayrica kosuyor.

    _build_llm_generators onu tekrar dondururse ayni operasyonlar icin iki kez
    calisir ve birebir ayni tc_id'leri uretir.
    """
    generators = main._build_llm_generators(None)

    assert generators, "En az bir LLM generator beklenir"
    assert not any(
        isinstance(generator, TraditionalGenerator) for generator, _, _ in generators
    ), "Traditional _build_llm_generators tarafindan dondurulmemeli (K2)"
    assert not any(v_name == "traditional" for _, v_name, _ in generators)

    # 8 LLM modeli × 2 prompt variant = 16 tuple; Traditional bunlara dahil degil.
    llm_model_count = len(GENERATOR_REGISTRY) - 1
    assert len(generators) == llm_model_count * len(config.PROMPT_VARIANTS)


def test_build_llm_generators_ignores_traditional_even_when_explicitly_selected():
    """'traditional' acikca secilse bile LLM listesine sizmamali (K2)."""
    generators = main._build_llm_generators(["traditional"])

    assert generators == []


def test_traditional_rows_are_not_duplicated_end_to_end():
    """K2: tek gecisten uretilen Traditional satirlarinda duplicate tc_id olmamali."""
    ops = [
        ApiOperation(op_id="EP1", method="GET", path="/get"),
        ApiOperation(op_id="EP2", method="POST", path="/post"),
    ]

    rows = TraditionalGenerator().generate(ops, "", "", 3)

    identities = [
        (row["generator"], row.get("prompt_variant", ""), row["tc_id"]) for row in rows
    ]
    assert len(identities) == len(set(identities)), f"Duplicate tc_id: {identities}"
    assert len(rows) == 6  # 2 operasyon × 3 case
