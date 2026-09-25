import json

import pytest

import config
import metrics
import runner
from generators.claude_gen import ClaudeGenerator
from generators.gemini_gen import GeminiGenerator
from generators.groq_gen import GroqGenerator
from generators.openai_gen import OpenAIGenerator
from generators.traditional import TraditionalGenerator
from models import ApiOperation
from reporters.csv_reporter import (
    build_comparison_summary,
    compute_generator_metrics,
    save_results_csv,
)


def _operation():
    return ApiOperation(
        op_id="PING",
        method="GET",
        path="/ping",
        summary="Ping",
        description="Health check",
        response_schemas={"200": {"description": "OK"}},
    )


def _payload():
    return "PING_TC1|Valid ping|GET /ping|-|200|OK"


def test_all_providers_use_production_generators_for_both_variants(monkeypatch):
    calls = {}

    class OpenAICompletions:
        def create(self, **kwargs):
            calls["openai_request"] = kwargs
            return type(
                "Response",
                (),
                {
                    "choices": [type("Choice", (), {"message": type("Message", (), {"content": _payload()})()})()],
                    "usage": type("Usage", (), {"total_tokens": 3})(),
                },
            )()

    class OpenAIClient:
        def __init__(self, **kwargs):
            calls["openai_client"] = kwargs
            self.chat = type("Chat", (), {"completions": OpenAICompletions()})()

    class GeminiModels:
        def generate_content(self, **kwargs):
            calls["gemini_request"] = kwargs
            return type("Response", (), {"text": _payload(), "usage_metadata": None})()

    class GeminiClient:
        def __init__(self, **kwargs):
            calls["gemini_client"] = kwargs
            self.models = GeminiModels()

    class ClaudeMessages:
        def create(self, **kwargs):
            calls["claude_request"] = kwargs
            return type(
                "Response",
                (),
                {
                    "content": [type("Block", (), {"text": _payload()})()],
                    "usage": type("Usage", (), {"input_tokens": 1, "output_tokens": 2})(),
                },
            )()

    class ClaudeClient:
        def __init__(self, **kwargs):
            calls["claude_client"] = kwargs
            self.messages = ClaudeMessages()

    class GeminiModule:
        Client = GeminiClient

    class AnthropicModule:
        Anthropic = ClaudeClient

    monkeypatch.setenv("OPENAI_API_KEY", "dummy-openai")
    monkeypatch.setenv("GEMINI_API_KEY", "dummy-gemini")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-anthropic")
    monkeypatch.setenv("GROQ_API_KEY", "dummy-groq")
    monkeypatch.setattr("generators.openai_gen.OpenAI", OpenAIClient)
    monkeypatch.setattr("generators.gemini_gen.genai", GeminiModule)
    monkeypatch.setattr("generators.claude_gen.anthropic", AnthropicModule)

    operation = [_operation()]
    generators = [
        OpenAIGenerator("gpt-test"),
        GeminiGenerator("gemini-test"),
        ClaudeGenerator("claude-test"),
        GroqGenerator("groq-test"),
    ]

    rows = []
    for generator in generators:
        for variant in ("basic", "edge_focused"):
            rows.extend(generator.generate(operation, variant, "deterministic", 1))

    rows.extend(TraditionalGenerator().generate(operation, "", "", 1))

    assert len(rows) == 9
    assert {row["prompt_variant"] for row in rows} >= {"basic", "edge_focused", "traditional"}
    assert calls["openai_request"]["max_completion_tokens"] >= 2048
    assert calls["openai_client"]["timeout"] == config.REQUEST_TIMEOUT
    assert calls["gemini_request"]["model"] == "gemini-test"
    assert calls["gemini_request"]["config"]["max_output_tokens"] == 8192
    assert calls["claude_request"]["max_tokens"] >= 2048
    assert calls["claude_client"]["timeout"] == config.REQUEST_TIMEOUT

    # Groq shares the production OpenAI adapter but must retain its own routing.
    assert calls["openai_client"]["base_url"] == "https://api.groq.com/openai/v1"

    class DummySession:
        def __init__(self):
            self.headers = {}
            self.cookies = {}

        def request(self, method, url, **kwargs):
            return type(
                "Response",
                (),
                {
                    "status_code": 200,
                    "text": json.dumps({"ok": True}),
                    "headers": {"Content-Type": "application/json"},
                },
            )()

    monkeypatch.setattr(runner.requests, "Session", DummySession)
    executed = runner.run_testcases("https://example.test", rows)
    assert len(executed) == len(rows)
    assert all(row["actual_status"] == 200 for row in executed)


def test_full_pipeline_dry_run_persists_metrics_ranks_and_exports(monkeypatch, tmp_path):
    class DummySession:
        def __init__(self):
            self.headers = {}
            self.cookies = {}

        def request(self, method, url, **kwargs):
            return type(
                "Response",
                (),
                {
                    "status_code": 200,
                    "text": json.dumps({"ok": True}),
                    "headers": {"Content-Type": "application/json"},
                    "json": lambda self: {"ok": True},
                },
            )()

    monkeypatch.setattr(runner.requests, "Session", DummySession)
    rows = TraditionalGenerator().generate([_operation()], "", "", 2)
    executed = runner.run_testcases("https://example.test", rows)

    path = save_results_csv(executed, str(tmp_path))
    metrics_rows = compute_generator_metrics(executed)
    comparison = build_comparison_summary(executed)
    ranking = metrics.spectral_ranking(
        [row["generator"] for row in metrics_rows],
        executed,
    )

    assert path
    assert (tmp_path / path.split("/")[-1]).exists()
    assert metrics_rows[0]["total_tests"] == len(executed)
    assert comparison["generator_rankings"]
    assert len(ranking["ranking"]) == 1
    assert ranking["ranking"][0][0] == TraditionalGenerator.GENERATOR_NAME


@pytest.mark.parametrize("bad_value", [[], [{"generator": "A", "pass": True}]])
def test_spectral_ranking_handles_small_and_missing_inputs(bad_value):
    generators = ["A"] if bad_value else []
    result = metrics.spectral_ranking(generators, bad_value)
    assert result["ranking"] == ([ ("A", pytest.approx(1.0)) ] if bad_value else [])


def test_spectral_ranking_prefers_higher_pass_rate():
    rows = [
        {"generator": "A", "http_method": "GET", "path": "/a", "expected_status": 200, "pass": True},
        {"generator": "B", "http_method": "GET", "path": "/b", "expected_status": 500, "pass": False},
    ]
    ranking = metrics.spectral_ranking(["A", "B"], rows)["ranking"]
    assert ranking[0][0] == "A"
