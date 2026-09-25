"""Anthropic Claude tabanlı test senaryosu üreticisi."""

import logging
from typing import Dict, List

import config
from models import ApiOperation
from generators.base import BaseGenerator, ProviderResponseParseError
from security.secret_loader import get_api_key

_logger = logging.getLogger(__name__)

try:
    import anthropic  # type: ignore
except ImportError:
    anthropic = None  # type: ignore


class ClaudeGenerator(BaseGenerator):
    """Anthropic Claude API'si ile test senaryosu üretir."""

    def __init__(self, model: str) -> None:
        self.model = model
        self._client = None

    def _get_client(self):
        if anthropic is None:
            raise RuntimeError("'anthropic' paketi yüklü değil. pip install anthropic")
        api_key = get_api_key("claude")
        if self._client is None:
            self._client = anthropic.Anthropic(api_key=api_key, timeout=config.REQUEST_TIMEOUT)
        return self._client

    def _request_completion(self, prompt: str, max_tokens: int, smoke: bool = False) -> tuple[str, int]:
        message = self._get_client().messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        content = getattr(message, "content", None) or []
        if not content:
            raise ProviderResponseParseError("Provider response parse error: content_blocks=0")
        text = getattr(content[0], "text", None)
        if not isinstance(text, str) or not text.strip():
            raise ProviderResponseParseError(
                f"Provider response parse error: content_blocks={len(content)}; first_block_type={type(content[0]).__name__}"
            )
        usage = getattr(message, "usage", None)
        total_tokens = 0
        if usage:
            total_tokens = (getattr(usage, "input_tokens", 0) or 0) + (getattr(usage, "output_tokens", 0) or 0)
        return text, total_tokens

    def _generate_for_operation(
        self,
        op: ApiOperation,
        variant_name: str,
        variant_desc: str,
        num_cases: int,
    ) -> List[Dict]:
        client = self._get_client()
        generator_name = f"LLM-Claude-{self.model}-{variant_name}"
        _logger.info("[Claude - %s - %s] %s (%s %s) üretiliyor...", self.model, variant_name, op.op_id, op.method, op.path)

        def request_completion(prompt: str) -> tuple[str, int]:
            token_ceiling = config.MAX_TOKENS_BY_PROVIDER.get("claude", 8192)
            max_tokens = min(token_ceiling, max(2048, num_cases * 200))
            return self._request_completion(prompt, max_tokens)

        return self._generate_cases_with_repair(
            op=op,
            variant_name=variant_name,
            variant_desc=variant_desc,
            num_cases=num_cases,
            generator_name=generator_name,
            request_completion=request_completion,
        )
