"""Generator kayıt merkezi — tüm sağlayıcılar buradan import edilir."""

from generators.traditional import TraditionalGenerator
from generators.openai_gen import OpenAIGenerator
from generators.gemini_gen import GeminiGenerator
from generators.claude_gen import ClaudeGenerator
from generators.groq_gen import GroqGenerator
import config

__all__ = [
    "TraditionalGenerator",
    "OpenAIGenerator",
    "GeminiGenerator",
    "ClaudeGenerator",
    "GroqGenerator",
    "GENERATOR_REGISTRY",
]

# key → (GeneratorClass, model_str | None, provider_label)
# "traditional" modelsiz; diğerleri cls(model) ile örneklenir.
GENERATOR_REGISTRY: dict[str, tuple] = {
    "traditional": (TraditionalGenerator, None, "Traditional"),
}
for _m in config.OPENAI_MODELS:
    GENERATOR_REGISTRY[f"openai:{_m}"] = (OpenAIGenerator, _m, "OpenAI")
for _m in config.GEMINI_MODELS:
    GENERATOR_REGISTRY[f"gemini:{_m}"] = (GeminiGenerator, _m, "Gemini")
for _m in config.CLAUDE_MODELS:
    GENERATOR_REGISTRY[f"claude:{_m}"] = (ClaudeGenerator, _m, "Claude")
for _m in config.GROQ_MODELS:
    GENERATOR_REGISTRY[f"groq:{_m}"] = (GroqGenerator, _m, "Groq")
