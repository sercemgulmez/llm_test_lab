"""Groq (OpenAI-uyumlu API) tabanlı test senaryosu üreticisi."""

import config
from generators.openai_gen import OpenAIGenerator


class GroqGenerator(OpenAIGenerator):
    """Groq API'si ile test senaryosu üretir (OpenAI-uyumlu endpoint)."""

    _base_url = "https://api.groq.com/openai/v1"
    _api_key_env = "GROQ_API_KEY"
    _provider_label = "Groq"

    def _max_tokens_for(self, num_cases: int) -> int:
        """Groq'ta cikti tavani SABIT; case sayisiyla olceklenmez.

        gpt-oss modelleri reasoning uretiyor ve reasoning AYNI butceden harcaniyor;
        olculdu: 3000 tavanda ~1500 token reasoning'e gidiyor ve yanit
        finish_reason='length' ile kesiliyor. Tavan config.GROQ_MAX_OUTPUT_TOKENS'tan
        gelir ve ust siniri Groq'un TPM'i belirler (gerekce orada yazili).

        CASE SAYISI DEGISTIRILMEDI.
        """
        return config.GROQ_MAX_OUTPUT_TOKENS
