"""Anthropic Claude tabanlı test senaryosu üreticisi."""

import os
from typing import Dict, List

from models import ApiOperation
from generators.base import BaseGenerator, build_llm_prompt, parse_llm_lines_to_rows

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
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY ortam değişkeni tanımlı değil.")
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

        prompt = build_llm_prompt(op, num_cases, variant_name, variant_desc)
        message = client.messages.create(
            model=self.model,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        text: str = message.content[0].text if message.content else ""
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        return parse_llm_lines_to_rows(lines, op, generator_name)
