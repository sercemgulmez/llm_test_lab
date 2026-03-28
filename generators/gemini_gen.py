"""Google Gemini tabanlı test senaryosu üreticisi."""

import os
from typing import Dict, List

from models import ApiOperation
from generators.base import BaseGenerator, build_llm_prompt, parse_llm_lines_to_rows

try:
    from google import genai  # type: ignore
except ImportError:
    genai = None  # type: ignore


class GeminiGenerator(BaseGenerator):
    """Google Gemini API'si ile test senaryosu üretir."""

    def __init__(self, model: str) -> None:
        self.model = model
        self._client = None

    def _get_client(self):
        if genai is None:
            raise RuntimeError("'google-genai' paketi yüklü değil. pip install google-genai")
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY ortam değişkeni tanımlı değil.")
        if self._client is None:
            self._client = genai.Client(api_key=api_key)
        return self._client

    def _generate_for_operation(
        self,
        op: ApiOperation,
        variant_name: str,
        variant_desc: str,
        num_cases: int,
    ) -> List[Dict]:
        client = self._get_client()
        generator_name = f"LLM-Gemini-{self.model}-{variant_name}"
        print(f"[Gemini - {self.model} - {variant_name}] {op.op_id} ({op.method} {op.path}) üretiliyor...")

        prompt = build_llm_prompt(op, num_cases, variant_name, variant_desc)
        resp = client.models.generate_content(model=self.model, contents=prompt)
        text: str = getattr(resp, "text", "") or ""
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        rows = parse_llm_lines_to_rows(lines, op, generator_name)
        usage = getattr(resp, "usage_metadata", None)
        if usage:
            total_tokens = getattr(usage, "total_token_count", 0) or 0
            for row in rows:
                row["tokens_used"] = total_tokens
        return rows
