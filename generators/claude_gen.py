"""Anthropic Claude tabanlı test senaryosu üreticisi."""

import logging
from typing import Dict, List

import llm_timeout
from models import ApiOperation, TokenUsage
from generators.base import BaseGenerator, ProviderResponseParseError
from security.secret_loader import get_api_key

_logger = logging.getLogger(__name__)

try:
    import anthropic  # type: ignore
except ImportError:
    anthropic = None  # type: ignore


class ClaudeGenerator(BaseGenerator):
    """Anthropic Claude API'si ile test senaryosu üretir."""

    _provider_label: str = "Claude"

    def __init__(self, model: str) -> None:
        self.model = model
        self._client = None

    def _get_client(self):
        if anthropic is None:
            raise RuntimeError("'anthropic' paketi yüklü değil. pip install anthropic")
        api_key = get_api_key("claude")
        if self._client is None:
            # config.REQUEST_TIMEOUT DEGIL (bkz. llm_timeout modulu).
            self._client = anthropic.Anthropic(
                api_key=api_key,
                timeout=llm_timeout.httpx_timeout_for(self._provider_label),
            )
        return self._client

    def _request_completion(self, prompt: str, max_tokens: int, smoke: bool = False) -> tuple[str, TokenUsage]:
        message = self._get_client().messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        self._last_call_meta = {
            "model_requested": self.model,
            "model_returned": getattr(message, "model", None),
            "response_id": getattr(message, "id", None),
            "finish_reason": getattr(message, "stop_reason", None),
            "sampling": {"max_tokens": max_tokens},
        }
        content = getattr(message, "content", None) or []
        if not content:
            raise ProviderResponseParseError("Provider response parse error: content_blocks=0")
        text = getattr(content[0], "text", None)
        if not isinstance(text, str) or not text.strip():
            raise ProviderResponseParseError(
                f"Provider response parse error: content_blocks={len(content)}; first_block_type={type(content[0]).__name__}"
            )
        # Anthropic usage'inda ayri bir "thinking token" alani YOKTUR ve bu kosuda
        # extended thinking istenmiyor; dusunme token'i uretilse bile output_tokens
        # icinde gelir ve cikti fiyatindan faturalanir. Bu yuzden reasoning_tokens
        # bos birakilir — sahte bir ayristirma yapilmaz.
        usage = getattr(message, "usage", None)
        if usage is None:
            return text, TokenUsage()
        input_tokens = getattr(usage, "input_tokens", None)
        output_tokens = getattr(usage, "output_tokens", None)
        if input_tokens is None or output_tokens is None:
            return text, TokenUsage(split_available=False)
        return text, TokenUsage(
            input_tokens=int(input_tokens or 0),
            output_tokens=int(output_tokens or 0),
            split_available=True,
        )

    def _generate_for_operation(
        self,
        op: ApiOperation,
        variant_name: str,
        variant_desc: str,
        num_cases: int,
    ) -> List[Dict]:
        client = self._get_client()
        generator_name = f"LLM-Claude-{self.model}"
        _logger.info("[Claude - %s - %s] %s (%s %s) üretiliyor...", self.model, variant_name, op.op_id, op.method, op.path)

        def request_completion(prompt: str) -> tuple[str, int]:
            return self._request_completion(prompt, self._max_tokens_for(num_cases))

        return self._generate_cases_with_repair(
            op=op,
            variant_name=variant_name,
            variant_desc=variant_desc,
            num_cases=num_cases,
            generator_name=generator_name,
            request_completion=request_completion,
        )
