import config
import main


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
