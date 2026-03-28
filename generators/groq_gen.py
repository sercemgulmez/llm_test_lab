"""Groq (OpenAI-uyumlu API) tabanlı test senaryosu üreticisi."""

from generators.openai_gen import OpenAIGenerator


class GroqGenerator(OpenAIGenerator):
    """Groq API'si ile test senaryosu üretir (OpenAI-uyumlu endpoint)."""

    _base_url = "https://api.groq.com/openai/v1"
    _api_key_env = "GROQ_API_KEY"
    _provider_label = "Groq"
