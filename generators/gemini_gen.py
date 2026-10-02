"""Google Gemini tabanlı test senaryosu üreticisi."""

import logging
from typing import Dict, List

import llm_timeout
from models import ApiOperation, TokenUsage
from generators.base import BaseGenerator, ProviderResponseParseError
from security.secret_loader import get_api_key

_logger = logging.getLogger(__name__)

try:
    from google import genai  # type: ignore
    from google.genai import types as genai_types  # type: ignore
except ImportError:
    genai = None  # type: ignore
    genai_types = None  # type: ignore


class GeminiGenerator(BaseGenerator):
    """Google Gemini API'si ile test senaryosu üretir."""

    _provider_label: str = "Gemini"

    def __init__(self, model: str) -> None:
        self.model = model
        self._client = None

    def _get_client(self):
        if genai is None:
            raise RuntimeError("'google-genai' paketi yüklü değil. pip install google-genai")
        api_key = get_api_key("gemini")
        if self._client is None:
            # Once hic timeout verilmiyordu: google-genai HttpOptions.timeout=None
            # birakiyor ve httpx bunu SONSUZ olarak yorumluyor, yani asili kalan
            # bir istek kosuyu sureklilesirebiliyordu.
            self._client = genai.Client(
                api_key=api_key,
                http_options=genai_types.HttpOptions(
                    timeout=llm_timeout.genai_timeout_ms_for(self._provider_label),
                    # google-genai varsayilani zaten retry YAPMIYOR
                    # (retry_options=None -> stop_after_attempt(1)); burada
                    # ACIKCA sabitleniyor ki bir surum yukseltmesi sessizce
                    # gizli yeniden deneme getirmesin.
                    retry_options=genai_types.HttpRetryOptions(attempts=1),
                ),
            )
        return self._client

    @staticmethod
    def _response_metadata(resp) -> str:
        candidates = getattr(resp, "candidates", None) or []
        candidate = candidates[0] if candidates else None
        content = getattr(candidate, "content", None) if candidate is not None else None
        parts = getattr(content, "parts", None) or []
        part_types = []
        text_parts = 0
        for part in parts:
            fields = []
            for name in ("text", "function_call", "function_response", "thought_signature"):
                if getattr(part, name, None) is not None:
                    fields.append(name)
            part_types.append("+".join(fields) or type(part).__name__)
            if isinstance(getattr(part, "text", None), str) and getattr(part, "text").strip():
                text_parts += 1
        feedback = getattr(resp, "prompt_feedback", None)
        return (
            f"response_type={type(resp).__name__}; candidates={len(candidates)}; "
            f"finish_reason={getattr(candidate, 'finish_reason', None)}; content={content is not None}; "
            f"parts={len(parts)}; part_types={part_types}; text_parts={text_parts}; "
            f"block_reason={getattr(feedback, 'block_reason', None)}"
        )

    def _request_completion(self, prompt: str, max_tokens: int, smoke: bool = False) -> tuple[str, TokenUsage]:
        generation_config = {"max_output_tokens": max_tokens}
        if smoke and self.model == "gemini-2.5-flash":
            generation_config["thinking_config"] = {"thinking_budget": 0}
        resp = self._get_client().models.generate_content(
            model=self.model,
            contents=prompt,
            config=generation_config,
        )
        self._last_call_meta = {
            "model_requested": self.model,
            "model_returned": getattr(resp, "model_version", None),
            "response_id": getattr(resp, "response_id", None),
            "finish_reason": getattr((getattr(resp, "candidates", None) or [None])[0], "finish_reason", None),
            "sampling": dict(generation_config),
        }
        try:
            text = getattr(resp, "text", None)
        except Exception as exc:
            raise ProviderResponseParseError(
                f"Provider response parse error: {self._response_metadata(resp)}"
            ) from exc
        if not isinstance(text, str) or not text.strip():
            raise ProviderResponseParseError(f"Provider response parse error: {self._response_metadata(resp)}")
        usage = getattr(resp, "usage_metadata", None)
        prompt_tokens = getattr(usage, "prompt_token_count", None)
        output_tokens = getattr(usage, "candidates_token_count", None)
        total = getattr(usage, "total_token_count", 0) or 0
        # Gemini'de dusunme token'lari AYRI alandadir: candidates_token_count'un
        # ICINDE DEGILDIR ama cikti fiyatindan faturalanir (fiyat sayfasi:
        # "Output price (including thinking tokens)"). Bu yuzden ayri kaydedilir
        # ve faturalanabilir cikti = candidates + thoughts olarak hesaplanir.
        thoughts = getattr(usage, "thoughts_token_count", None) if usage is not None else None
        if prompt_tokens is None or output_tokens is None:
            return text, TokenUsage(total_tokens=total, split_available=False)
        return text, TokenUsage(
            input_tokens=int(prompt_tokens or 0),
            output_tokens=int(output_tokens or 0),
            total_tokens=total,
            split_available=True,
            reasoning_tokens=int(thoughts or 0),
            reasoning_included_in_output=False,
        )

    def _max_tokens_for(self, num_cases: int) -> int:
        """Gemini'de cikti tavani SABIT 8192; case sayisiyla olceklenmez.

        Deney parametresi DEGISTIRILMEDI; yalnizca limitorun ayni tavani
        rezerve edebilmesi icin tek kaynaga tasindi.
        """
        return 8192

    def _generate_for_operation(
        self,
        op: ApiOperation,
        variant_name: str,
        variant_desc: str,
        num_cases: int,
    ) -> List[Dict]:
        client = self._get_client()
        generator_name = f"LLM-Gemini-{self.model}"
        _logger.info("[Gemini - %s - %s] %s (%s %s) üretiliyor...", self.model, variant_name, op.op_id, op.method, op.path)

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
