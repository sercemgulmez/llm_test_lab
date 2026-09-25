import pytest

from generators.gemini_gen import GeminiGenerator
from generators.groq_gen import GroqGenerator
from generators.base import ProviderResponseParseError
from scripts import check_model_access as smoke


class _PassGenerator:
    def __init__(self, model):
        self.model = model

    def smoke_test(self):
        return [{"generator": self.model}]


class _FailGenerator(_PassGenerator):
    def smoke_test(self):
        raise RuntimeError("invalid api key sk-proj-" + "A" * 48)


def _registry(generator_class=_PassGenerator):
    providers = [
        ("openai", "OpenAI", "OPENAI_API_KEY"),
        ("gemini", "Gemini", "GEMINI_API_KEY"),
        ("groq", "Groq", "GROQ_API_KEY"),
        ("claude", "Claude", "ANTHROPIC_API_KEY"),
    ]
    result = {"traditional": (object, None, "Traditional")}
    for prefix, provider, _env in providers:
        for index in range(2):
            result[f"{prefix}:model-{index}"] = (generator_class, f"model-{index}", provider)
    return result


def _set_dummy_credentials(monkeypatch):
    for env_var in smoke.ENV_BY_PROVIDER.values():
        monkeypatch.setenv(env_var, "dummy-value")


def test_main_prints_observable_success_summary(monkeypatch, capsys):
    _set_dummy_credentials(monkeypatch)
    monkeypatch.setattr(smoke, "load_dotenv", lambda *args, **kwargs: False)
    assert smoke.main(_registry()) == 0
    output = capsys.readouterr().out
    assert "LLM_TESTLAB REAL API SMOKE TEST" in output
    assert output.count("[RUN]") == 8
    assert output.count("[PASS]") == 8
    assert "Expected: 8" in output
    assert "Tested: 8" in output
    assert "READY FOR FULL EXPERIMENT: YES" in output


def test_empty_registry_is_failure(monkeypatch, capsys):
    monkeypatch.setattr(smoke, "load_dotenv", lambda *args, **kwargs: False)
    assert smoke.main({}) == 1
    output = capsys.readouterr().out
    assert "NO_MODELS_TESTED" in output
    assert "Tested: 0" in output
    assert "READY FOR FULL EXPERIMENT: NO" in output


def test_missing_credentials_do_not_call_network_boundary(monkeypatch, capsys):
    calls = []

    class MustNotRun(_PassGenerator):
        def smoke_test(self):
            calls.append(self.model)
            raise AssertionError("network boundary reached")

    for env_var in smoke.ENV_BY_PROVIDER.values():
        monkeypatch.delenv(env_var, raising=False)
    monkeypatch.setattr(smoke, "load_dotenv", lambda *args, **kwargs: False)
    assert smoke.main(_registry(MustNotRun)) == 1
    output = capsys.readouterr().out
    assert not calls
    assert output.count("MISSING_CREDENTIAL") == 8
    assert "Tested: 0" in output


def test_failure_is_nonzero_and_secret_is_redacted(monkeypatch, capsys):
    _set_dummy_credentials(monkeypatch)
    monkeypatch.setattr(smoke, "load_dotenv", lambda *args, **kwargs: False)
    assert smoke.main(_registry(_FailGenerator)) == 1
    output = capsys.readouterr().out
    assert "AUTH_ERROR" in output
    assert "[REDACTED]" in output
    assert "sk-proj-" not in output
    assert "READY FOR FULL EXPERIMENT: NO" in output


def test_error_taxonomy():
    class StatusError(RuntimeError):
        def __init__(self, message, status_code):
            super().__init__(message)
            self.status_code = status_code

    assert smoke._classify_error(TimeoutError("timed out")) == "TIMEOUT"
    assert smoke._classify_error(ConnectionError("connection failed")) == "NETWORK_ERROR"
    assert smoke._classify_error(StatusError("unauthorized", 401)) == "AUTH_ERROR"
    assert smoke._classify_error(StatusError("insufficient_quota", 429)) == "BILLING_QUOTA_ERROR"
    assert smoke._classify_error(StatusError("rate limit", 429)) == "RATE_LIMIT"
    assert smoke._classify_error(StatusError("model not found", 404)) == "MODEL_NOT_FOUND"
    assert smoke._classify_error(StatusError("forbidden", 403)) == "MODEL_ACCESS_ERROR"
    assert smoke._classify_error(StatusError("bad request", 400)) == "REQUEST_CONTRACT_ERROR"
    assert smoke._classify_error(StatusError("server failure", 503)) == "PROVIDER_ERROR"
    assert smoke._classify_error(ProviderResponseParseError("Provider response parse error")) == "PROVIDER_RESPONSE_PARSE_ERROR"
    assert smoke._classify_error(RuntimeError("Model output format error")) == "MODEL_OUTPUT_FORMAT_ERROR"


def test_targeted_selection_calls_only_requested_models(monkeypatch, capsys):
    _set_dummy_credentials(monkeypatch)
    monkeypatch.setattr(smoke, "load_dotenv", lambda *args, **kwargs: False)
    selection = "gemini:model-0,groq:model-1"
    assert smoke.main(_registry(), only=selection) == 0
    output = capsys.readouterr().out
    assert output.count("[RUN]") == 2
    assert "Gemini | model-0" in output
    assert "Groq | model-1" in output
    assert "OpenAI | model-0" not in output
    assert "Claude | model-0" not in output
    assert "TARGETED RETEST: PASS" in output
    assert "READY FOR FULL EXPERIMENT: NO" in output


def test_gemini_25_smoke_uses_valid_budget_and_disables_thinking(monkeypatch):
    captured = {}

    class Models:
        def generate_content(self, **kwargs):
            captured.update(kwargs)
            return type("Response", (), {"text": "ok", "usage_metadata": None})()

    generator = GeminiGenerator("gemini-2.5-flash")
    generator._client = type("Client", (), {"models": Models()})()
    text, _tokens = generator._request_completion("prompt", 1024, smoke=True)
    assert text == "ok"
    assert captured["config"]["max_output_tokens"] == 1024
    assert captured["config"]["thinking_config"] == {"thinking_budget": 0}


def test_groq_120b_smoke_excludes_reasoning_without_exposing_it():
    captured = {}

    class Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            message = type("Message", (), {"content": "", "reasoning": "private-reasoning-value", "tool_calls": None, "refusal": None})()
            choice = type("Choice", (), {"message": message, "finish_reason": "length"})()
            return type("Response", (), {"choices": [choice]})()

    generator = GroqGenerator("openai/gpt-oss-120b")
    generator._client = type("Client", (), {"chat": type("Chat", (), {"completions": Completions()})()})()
    with pytest.raises(ProviderResponseParseError) as exc_info:
        generator._request_completion("prompt", 1024, smoke=True)
    assert captured["max_completion_tokens"] == 1024
    assert captured["reasoning_effort"] == "low"
    assert captured["extra_body"] == {"include_reasoning": False}
    assert "private-reasoning-value" not in str(exc_info.value)
    assert "reasoning_length=23" in str(exc_info.value)
