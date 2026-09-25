"""OpenAI Chat Completions API ve uyumlu endpoint'ler için test senaryosu üreticisi."""

from __future__ import annotations

import logging
from typing import Dict, List

import config
from models import ApiOperation
from generators.base import BaseGenerator, ProviderResponseParseError
from security.secret_loader import get_api_key_from_env

_logger = logging.getLogger(__name__)

try:
    from openai import OpenAI  # type: ignore
except ImportError:
    OpenAI = None  # type: ignore


class OpenAIGenerator(BaseGenerator):
    """OpenAI Chat Completions API (ve uyumlu endpoint'ler) ile test senaryosu üretir."""

    _base_url: str | None = None
    _api_key_env: str = "OPENAI_API_KEY"
    _provider_label: str = "OpenAI"

    def __init__(self, model: str) -> None:
        self.model = model
        self._client = None

    def _get_client(self):
        if OpenAI is None:
            raise RuntimeError("'openai' paketi yüklü değil. pip install openai")
        api_key = get_api_key_from_env(self._api_key_env)
        if self._client is None:
            kwargs: dict = {
                "api_key": api_key,
                "timeout": config.REQUEST_TIMEOUT,
            }
            if self._base_url:
                kwargs["base_url"] = self._base_url
            self._client = OpenAI(**kwargs)
        return self._client

    @staticmethod
    def _response_metadata(resp) -> str:
        choices = getattr(resp, "choices", None) or []
        choice = choices[0] if choices else None
        message = getattr(choice, "message", None) if choice is not None else None
        content = getattr(message, "content", None) if message is not None else None
        reasoning = getattr(message, "reasoning", None) if message is not None else None
        tool_calls = getattr(message, "tool_calls", None) if message is not None else None
        refusal = getattr(message, "refusal", None) if message is not None else None
        return (
            f"response_type={type(resp).__name__}; choices={len(choices)}; "
            f"finish_reason={getattr(choice, 'finish_reason', None)}; message={message is not None}; "
            f"content_type={type(content).__name__}; content_length={len(content) if isinstance(content, str) else 0}; "
            f"reasoning_present={bool(reasoning)}; reasoning_length={len(reasoning) if isinstance(reasoning, str) else 0}; "
            f"tool_calls_present={bool(tool_calls)}; refusal_present={bool(refusal)}"
        )

    def _request_completion(self, prompt: str, max_tokens: int, smoke: bool = False) -> tuple[str, int]:
        request_kwargs = {
            "model": self.model,
            "max_completion_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if smoke and self._provider_label == "Groq" and self.model.startswith("openai/gpt-oss-"):
            request_kwargs["reasoning_effort"] = "low"
            request_kwargs["extra_body"] = {"include_reasoning": False}
        resp = self._get_client().chat.completions.create(
            **request_kwargs,
        )
        choices = getattr(resp, "choices", None) or []
        if not choices:
            raise ProviderResponseParseError(f"Provider response parse error: {self._response_metadata(resp)}")
        message = getattr(choices[0], "message", None)
        text = getattr(message, "content", None) if message is not None else None
        if not isinstance(text, str) or not text.strip():
            raise ProviderResponseParseError(f"Provider response parse error: {self._response_metadata(resp)}")
        usage = getattr(resp, "usage", None)
        return text, (getattr(usage, "total_tokens", 0) or 0)

    def _generate_for_operation(
        self,
        op: ApiOperation,
        variant_name: str,
        variant_desc: str,
        num_cases: int,
    ) -> List[Dict]:
        client = self._get_client()
        generator_name = f"LLM-{self._provider_label}-{self.model}-{variant_name}"
        _logger.info("[%s - %s - %s] %s (%s %s) üretiliyor...", self._provider_label, self.model, variant_name, op.op_id, op.method, op.path)

        def request_completion(prompt: str) -> tuple[str, int]:
            token_ceiling = config.MAX_TOKENS_BY_PROVIDER.get(self._provider_label.lower(), 16384)
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
