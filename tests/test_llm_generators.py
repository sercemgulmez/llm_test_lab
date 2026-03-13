from models import ApiOperation
from generators.base import BaseGenerator
from generators.openai_gen import OpenAIGenerator
from generators.gemini_gen import GeminiGenerator
from generators.claude_gen import ClaudeGenerator


def _sample_operation():
    return ApiOperation(
        op_id="LOGIN",
        method="POST",
        path="/login",
        summary="Login",
        description="Authenticate user",
    )


def test_openai_generator_uses_mocked_client(monkeypatch):
    class DummyCompletions:
        def create(self, model, messages):
            assert model == "gpt-test"
            assert messages[0]["role"] == "user"
            return type(
                "Resp",
                (),
                {
                    "choices": [
                        type(
                            "Choice",
                            (),
                            {"message": type("Msg", (), {"content": 'LOGIN_TC1|Valid|POST /login|{"email":"a"}|200|OK'})()},
                        )()
                    ]
                },
            )()

    class DummyClient:
        def __init__(self, api_key):
            assert api_key == "openai-key"
            self.chat = type("Chat", (), {"completions": DummyCompletions()})()

    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setattr("generators.openai_gen.OpenAI", DummyClient)

    rows = OpenAIGenerator("gpt-test").generate([_sample_operation()], "basic", "happy path", 1)

    assert len(rows) == 1
    assert rows[0]["generator"] == "LLM-OpenAI-gpt-test-basic"
    assert rows[0]["expected_status"] == 200


def test_gemini_generator_uses_mocked_client(monkeypatch):
    class DummyModels:
        def generate_content(self, model, contents):
            assert model == "gemini-test"
            assert "LOGIN" in contents
            return type("Resp", (), {"text": 'LOGIN_TC1|Valid|POST /login|-|200|OK'})()

    class DummyClient:
        def __init__(self, api_key):
            assert api_key == "gemini-key"
            self.models = DummyModels()

    class DummyGenAI:
        Client = DummyClient

    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    monkeypatch.setattr("generators.gemini_gen.genai", DummyGenAI)

    rows = GeminiGenerator("gemini-test").generate([_sample_operation()], "basic", "happy path", 1)

    assert len(rows) == 1
    assert rows[0]["generator"] == "LLM-Gemini-gemini-test-basic"
    assert rows[0]["request_body"] == ""


def test_claude_generator_uses_mocked_client(monkeypatch):
    class DummyMessages:
        def create(self, model, max_tokens, messages):
            assert model == "claude-test"
            assert max_tokens == 2048
            assert messages[0]["role"] == "user"
            return type(
                "Resp",
                (),
                {
                    "content": [type("Block", (), {"text": 'LOGIN_TC1|Valid|POST /login|-|200|OK'})()]
                },
            )()

    class DummyAnthropicClient:
        def __init__(self, api_key):
            assert api_key == "claude-key"
            self.messages = DummyMessages()

    class DummyAnthropicModule:
        Anthropic = DummyAnthropicClient

    monkeypatch.setenv("ANTHROPIC_API_KEY", "claude-key")
    monkeypatch.setattr("generators.claude_gen.anthropic", DummyAnthropicModule)

    rows = ClaudeGenerator("claude-test").generate([_sample_operation()], "basic", "happy path", 1)

    assert len(rows) == 1
    assert rows[0]["generator"] == "LLM-Claude-claude-test-basic"
    assert rows[0]["expected_result"] == "OK"


def test_non_retryable_quota_errors_do_not_retry():
    class DummyGenerator(BaseGenerator):
        def __init__(self):
            self.calls = 0

        def _generate_for_operation(self, op, variant_name, variant_desc, num_cases):
            self.calls += 1
            raise RuntimeError(
                "Error code: 429 - {'error': {'message': 'You exceeded your current quota', "
                "'code': 'insufficient_quota'}}"
            )

    gen = DummyGenerator()
    rows = gen.generate([_sample_operation()], "basic", "happy path", 1)

    assert rows == []
    assert gen.calls == 1
