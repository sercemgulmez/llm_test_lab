"""OpenAI Chat Completions API ve uyumlu endpoint'ler için test senaryosu üreticisi."""

from __future__ import annotations

from typing import Dict, List

from models import ApiOperation
from generators.base import BaseGenerator
from security.secret_loader import get_api_key_from_env

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
            kwargs: dict = {"api_key": api_key}
            if self._base_url:
                kwargs["base_url"] = self._base_url
            self._client = OpenAI(**kwargs)
        return self._client

    def _generate_for_operation(
        self,
        op: ApiOperation,
        variant_name: str,
        variant_desc: str,
        num_cases: int,
    ) -> List[Dict]:
        client = self._get_client()
        generator_name = f"LLM-{self._provider_label}-{self.model}-{variant_name}"
        print(f"[{self._provider_label} - {self.model} - {variant_name}] {op.op_id} ({op.method} {op.path}) üretiliyor...")

        def request_completion(prompt: str) -> tuple[str, int]:
            max_tokens = min(16384, max(2048, num_cases * 200))
            resp = client.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            text: str = resp.choices[0].message.content or ""
            usage = getattr(resp, "usage", None)
            return text, (getattr(usage, "total_tokens", 0) or 0)

        return self._generate_cases_with_repair(
            op=op,
            variant_name=variant_name,
            variant_desc=variant_desc,
            num_cases=num_cases,
            generator_name=generator_name,
            request_completion=request_completion,
        )
