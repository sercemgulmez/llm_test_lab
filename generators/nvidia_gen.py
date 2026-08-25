"""NVIDIA NIM (OpenAI-uyumlu API) tabanlı test senaryosu üreticisi."""

from generators.openai_gen import OpenAIGenerator


class NvidiaGenerator(OpenAIGenerator):
    """NVIDIA NIM API'si ile test senaryosu üretir (OpenAI-uyumlu endpoint, ücretsiz tier)."""

    _base_url = "https://integrate.api.nvidia.com/v1"
    _api_key_env = "NVIDIA_API_KEY"
    _provider_label = "Nvidia"
