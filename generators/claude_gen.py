"""Anthropic Claude tabanlı test senaryosu üreticisi."""

from typing import Dict, List

from models import ApiOperation
from generators.base import BaseGenerator
from security.secret_loader import get_api_key

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
            self._client = anthropic.Anthropic(api_key=api_key)
        return self._client

    def _generate_for_operation(
        self,
        op: ApiOperation,
        variant_name: str,
        variant_desc: str,
        num_cases: int,
    ) -> List[Dict]:
        client = self._get_client()
        generator_name = f"LLM-Claude-{self.model}-{variant_name}"
        print(f"[Claude - {self.model} - {variant_name}] {op.op_id} ({op.method} {op.path}) üretiliyor...")

        def request_completion(prompt: str) -> tuple[str, int]:
            max_tokens = min(8192, max(2048, num_cases * 200))
            message = client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            text: str = message.content[0].text if message.content else ""
            usage = getattr(message, "usage", None)
            total_tokens = 0
            if usage:
                total_tokens = (getattr(usage, "input_tokens", 0) or 0) + (getattr(usage, "output_tokens", 0) or 0)
            return text, total_tokens

        return self._generate_cases_with_repair(
            op=op,
            variant_name=variant_name,
            variant_desc=variant_desc,
            num_cases=num_cases,
            generator_name=generator_name,
            request_completion=request_completion,
        )
