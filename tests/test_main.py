import pytest

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


def test_missing_api_keys_are_skipped_with_warning_not_error(monkeypatch, caplog, tmp_path):
    """K8: anahtarsiz generator'lar pre-flight'ta atlanir.

    Orijinal denetim iddiasi ("main() cokuyor") ampirik olarak yanlisti; gercek
    kusur log seviyesiydi: her gorev ERROR uretiyordu. Pre-flight sonrasi hicbir
    ERROR olmamali, gorev basina tek WARNING olmali ve Traditional devam etmeli.
    """
    import logging

    for env_var in ("OPENAI_API_KEY", "GEMINI_API_KEY", "GROQ_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(env_var, raising=False)
    monkeypatch.setattr(main, "load_dotenv", lambda *a, **kw: False)
    monkeypatch.setattr(
        "sys.argv",
        [
            "main.py",
            "--endpoints", "GET /get,POST /post",
            "--base-url", "https://httpbin.org",
            "--num-cases", "2",
            "--no-run",
            "--no-checkpoint",
            "--output-dir", str(tmp_path),
        ],
    )

    with caplog.at_level(logging.WARNING):
        main.main()  # cokmemeli

    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    skipped = [r for r in caplog.records if "ATLANDI" in r.getMessage()]

    assert not errors, f"Eksik anahtar ERROR uretmemeli: {[r.getMessage() for r in errors]}"
    # 8 LLM modeli × 2 prompt variant = 16 gorev, her biri bir kez atlanir.
    assert len(skipped) == (len(GENERATOR_REGISTRY) - 1) * len(config.PROMPT_VARIANTS)
    # Traditional anahtar gerektirmedigi icin uretim devam etmis olmali.
    assert list(tmp_path.glob("executed_testcases_*.csv")), "Traditional satirlari yazilmali"


def test_non_runtime_error_in_generator_does_not_lose_other_results(monkeypatch, tmp_path):
    """K9: bir generator'in RuntimeError OLMAYAN istisnasi tum kosuyu oldurmemeli.

    Fix oncesi main.py yalnizca RuntimeError yakaliyordu; ValueError as_completed
    dongusunden disari tasiyor ve save_results_csv'ye HIC ULASILMIYORDU.
    """
    from generators.openai_gen import OpenAIGenerator

    monkeypatch.setattr(OpenAIGenerator, "_get_client", lambda self: object())

    def _boom(self, *args, **kwargs):
        raise ValueError("beklenmedik istisna")

    monkeypatch.setattr(OpenAIGenerator, "generate", _boom)
    monkeypatch.setattr(main, "load_dotenv", lambda *a, **kw: False)
    monkeypatch.setattr(
        "sys.argv",
        [
            "main.py",
            "--endpoints", "GET /get,POST /post",
            "--base-url", "https://httpbin.org",
            "--generators", "traditional,openai:gpt-4.1",
            "--num-cases", "2", "--no-run", "--no-checkpoint",
            "--output-dir", str(tmp_path),
        ],
    )

    main.main()  # cokmemeli

    written = list(tmp_path.glob("executed_testcases_*.csv"))
    assert written, "Diger generator'larin satirlari yine de yazilmali"


def test_unexpected_failure_still_writes_partial_results_and_exits_nonzero(monkeypatch, tmp_path):
    """K9 dis guvenlik agi: generator dongusu disindaki istisnada bile CSV yazilmali."""
    def _boom(*args, **kwargs):
        raise OSError("yurutme fazinda beklenmedik istisna")

    monkeypatch.setattr(main, "run_testcases", _boom)
    monkeypatch.setattr(main, "load_dotenv", lambda *a, **kw: False)
    monkeypatch.setattr(
        "sys.argv",
        [
            "main.py",
            "--endpoints", "GET /get,POST /post",
            "--base-url", "https://httpbin.org",
            "--generators", "traditional",
            "--num-cases", "2", "--no-checkpoint",
            "--output-dir", str(tmp_path),
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        main.main()

    assert exc_info.value.code == 1, "Hatali kosu sifir-disi cikis kodu dondurmeli"
    written = list(tmp_path.glob("executed_testcases_*.csv"))
    assert written, "Cokmede bile uretilen satirlar diske yazilmali"
