"""Tüm generator'lar için soyut temel sınıf ve ortak yardımcılar."""

import json
import re
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

from models import ApiOperation
from config import MAX_PARALLEL_WORKERS, RETRY_MAX_ATTEMPTS, RETRY_BACKOFF_SECONDS


_NON_RETRYABLE_ERROR_MARKERS = (
    # Fatura/kota hataları — tekrar denemeyle düzelmez
    "insufficient_quota",
    "credit balance is too low",
    # OpenAI kimlik doğrulama hataları
    "incorrect api key",
    "invalid api key",
    "no api key provided",
    # Anthropic kimlik doğrulama hataları
    "invalid x-api-key",
    "authentication_error",
)

# LLM'in satır başına eklediği numara / madde işareti öneklerini temizler
_LIST_PREFIX_RE = re.compile(r"^\s*(?:\d+[.)]\s+|[-*•]\s+)")


def _is_non_retryable_generation_error(exc: Exception) -> bool:
    """Kredi/kota gibi tekrar denemeyle düzelmeyecek provider hatalarini ayiklar."""
    message = str(exc).lower()
    return any(marker in message for marker in _NON_RETRYABLE_ERROR_MARKERS)


def build_llm_prompt(op: ApiOperation, num_cases: int, variant_name: str, variant_desc: str) -> str:
    """Bir API operasyonu için LLM'e gönderilecek token-optimized prompt."""
    extra_parts = []
    if op.description:
        extra_parts.append(op.description)
    if op.example_body:
        extra_parts.append(f"Örnek body: {op.example_body}")
    extra_section = ("\n" + "\n".join(extra_parts)) if extra_parts else ""

    return f"""Kıdemli Backend QA mühendisi olarak aşağıdaki API operasyonu için TAM OLARAK {num_cases} test senaryosu üret.

Operasyon: {op.op_id} | {op.method} {op.path} | {op.summary}{extra_section}
Strateji: [{variant_name}] {variant_desc}

FORMAT (her satır tam 6 pipe-ayrımlı alan):
TC_ID|Başlık|METHOD PATH|JSON_body veya -|HTTP_kodu|Beklenen_sonuç

ÖRNEK:
{op.op_id}_TC1|Başarılı istek|{op.method} {op.path}|-|200|İşlem başarıyla tamamlanır

KURALLAR:
- TC_ID: {op.op_id}_TC1, {op.op_id}_TC2, ... formatında
- Body yoksa "-", varsa geçerli JSON nesnesi
- Satır başına ASLA numara, tire, yıldız veya ``` ekleme
- Sadece ham satırlar, başka hiçbir şey yazma

TAM OLARAK {num_cases} SATIR:""".strip()


def parse_llm_lines_to_rows(
    lines: List[str],
    op: ApiOperation,
    generator_name: str,
) -> List[Dict]:
    """LLM çıktısındaki pipe-delimited satırları sözlük listesine dönüştürür."""
    rows: List[Dict] = []
    for line in lines:
        # Markdown kod bloğu sınırlarını atla (``` veya ```python vb.)
        if line.startswith("```"):
            continue

        # LLM'in eklediği "1. ", "2) ", "- ", "* " gibi önekleri temizle
        line = _LIST_PREFIX_RE.sub("", line).strip()
        if not line:
            continue

        # maxsplit=5 → JSON body içindeki | karakterlerini korur
        parts = [p.strip() for p in line.split("|", 5)]
        if len(parts) != 6:
            continue

        tc_id, title, method_path, body_str, exp_status_str, exp_result = parts

        mp_parts = method_path.split(maxsplit=1)
        if len(mp_parts) != 2:
            continue
        method, path = mp_parts[0].upper(), mp_parts[1]

        try:
            exp_status: object = int(exp_status_str)
        except ValueError:
            exp_status = ""

        if body_str == "-":
            body: Optional[dict] = None
        else:
            try:
                body = json.loads(body_str)
            except json.JSONDecodeError:
                body = None

        rows.append(
            {
                "generator": generator_name,
                "operation_id": op.op_id,
                "http_method": method,
                "path": path,
                "tc_id": tc_id,
                "title": title,
                "request_body": json.dumps(body) if body is not None else "",
                "expected_status": exp_status,
                "expected_result": exp_result,
            }
        )
    return rows


def _apply_token_tracking(rows: List[Dict], total_tokens: int) -> None:
    """Üretilen satırların tümüne token sayısını yazar."""
    for row in rows:
        row["tokens_used"] = total_tokens


class BaseGenerator(ABC):
    """Tüm test senaryosu üreticileri için temel sınıf."""

    # Non-retryable hata gelince True; kalan tüm operasyonlar hemen atlanır.
    _aborted: bool = False

    def generate(
        self,
        operations: List[ApiOperation],
        variant_name: str,
        variant_desc: str,
        num_cases: int,
    ) -> List[Dict]:
        """Verilen operasyonlar için test senaryoları paralel olarak üretir."""
        if not operations or self._aborted:
            return []
        workers = min(MAX_PARALLEL_WORKERS, len(operations))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(
                lambda op: self._generate_for_operation_with_retry(
                    op, variant_name, variant_desc, num_cases
                ),
                operations,
            ))
        return [row for sublist in results for row in sublist]

    def _generate_for_operation_with_retry(
        self,
        op: ApiOperation,
        variant_name: str,
        variant_desc: str,
        num_cases: int,
    ) -> List[Dict]:
        """Tek operasyon için retry mantığıyla senaryo üretir."""
        if self._aborted:
            return []
        for attempt in range(1, RETRY_MAX_ATTEMPTS + 1):
            try:
                return self._generate_for_operation(op, variant_name, variant_desc, num_cases)
            except Exception as e:
                if _is_non_retryable_generation_error(e):
                    self._aborted = True
                    print(f"  [HATA] {op.op_id} kalıcı provider/kredi hatası — generator iptal edildi: {e}")
                    break
                if attempt < RETRY_MAX_ATTEMPTS:
                    wait = RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1))
                    print(
                        f"  [RETRY {attempt}/{RETRY_MAX_ATTEMPTS}] {op.op_id} hata: {e} "
                        f"— {wait:.1f}s bekleyip tekrar deneniyor..."
                    )
                    time.sleep(wait)
                else:
                    print(f"  [HATA] {op.op_id} tüm denemeler başarısız: {e}")
        return []

    @abstractmethod
    def _generate_for_operation(
        self,
        op: ApiOperation,
        variant_name: str,
        variant_desc: str,
        num_cases: int,
    ) -> List[Dict]:
        """Alt sınıflar bu metodu uygular."""
        ...
