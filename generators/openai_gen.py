"""OpenAI (GPT) tabanlı test senaryosu üreticisi."""

import os
from typing import Dict, List

from models import ApiOperation
from generators.base import BaseGenerator, _apply_token_tracking, build_llm_prompt, parse_llm_lines_to_rows

try:
    from openai import OpenAI  # type: ignore
except ImportError:
    OpenAI = None  # type: ignore


class OpenAIGenerator(BaseGenerator):
    """OpenAI Chat Completions API'si ile test senaryosu üretir."""

    def __init__(self, model: str) -> None:
        self.model = model
        self._client = None

    def _get_client(self):
        if OpenAI is None:
            raise RuntimeError("'openai' paketi yüklü değil. pip install openai")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY ortam değişkeni tanımlı değil.")
        if self._client is None:
            self._client = OpenAI(api_key=api_key)
        return self._client

    def _generate_for_operation(
        self,
        op: ApiOperation,
        variant_name: str,
        variant_desc: str,
        num_cases: int,
    ) -> List[Dict]:
        client = self._get_client()
        generator_name = f"LLM-OpenAI-{self.model}-{variant_name}"
        print(f"[OpenAI - {self.model} - {variant_name}] {op.op_id} ({op.method} {op.path}) üretiliyor...")

        prompt = build_llm_prompt(op, num_cases, variant_name, variant_desc)
        max_tokens = min(16384, max(2048, num_cases * 150))
        resp = client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        text: str = resp.choices[0].message.content or ""
        lines = [s for l in text.splitlines() if (s := l.strip())]
        rows = parse_llm_lines_to_rows(lines, op, generator_name)
        if resp.usage:
            _apply_token_tracking(rows, resp.usage.total_tokens)
        return rows
