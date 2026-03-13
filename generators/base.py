"""Tüm generator'lar için soyut temel sınıf ve ortak yardımcılar."""

import json
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from models import ApiOperation
from config import RETRY_MAX_ATTEMPTS, RETRY_BACKOFF_SECONDS


_NON_RETRYABLE_ERROR_MARKERS = (
    "insufficient_quota",
    "credit balance is too low",
    "plans & billing",
    "check your plan and billing",
)


def _is_non_retryable_generation_error(exc: Exception) -> bool:
    """Kredi/kota gibi tekrar denemeyle düzelmeyecek provider hatalarini ayiklar."""
    message = str(exc).lower()
    return any(marker in message for marker in _NON_RETRYABLE_ERROR_MARKERS)


def build_llm_prompt(op: ApiOperation, num_cases: int, variant_name: str, variant_desc: str) -> str:
    """Bir API operasyonu için LLM'e gönderilecek prompt'u hazırlar."""
    example_body_section = (
        f"\nÖrnek Request Body:\n{op.example_body}\n"
        if op.example_body else ""
    )
    return f"""
Kıdemli bir Backend QA mühendisi gibi davran.
Aşağıdaki API operasyonu için fonksiyonel test senaryoları üret.

Operasyon ID: {op.op_id}
HTTP Method: {op.method}
Path: {op.path}
Özet: {op.summary}
Açıklama: {op.description}{example_body_section}
Test stratejisi (varyant):
- {variant_name}: {variant_desc}

Her test senaryosu için:
- Gerçekçi bir istek örneği kurgula (query parametreleri / JSON body vb.).
- Beklenen HTTP durum kodunu belirt.
- Senaryonun amacını kısa şekilde özetle.

ÇIKTI FORMATIN:
Her test tek satır olacak ve şu formatta yazılacak (aralarda | karakteri):

TC_ID|Kısa Başlık|HTTP_METHOD PATH|Request JSON Body (yoksa - yaz)|Beklenen HTTP Status Kodu (sayı)|Beklenen Sonuç (kısa açıklama)

ÖRNEK:
{op.op_id}_TC1|Başarılı giriş|POST /login|{{"phone":"+905xxxxxxxxx","password":"GecerliSifre1"}}|200|Kullanıcı başarıyla giriş yapar ve token döner

Kurallar:
- TC_ID şu formda olmalı: {op.op_id}_TC1, {op.op_id}_TC2, ...
- Request body varsa geçerli bir JSON nesnesi ({{ }}) olmalı.
- Body yoksa aynen "-" yaz.
- HTTP method ve path kısmında boşlukla ayrılmış method ve path kullan (örn: GET /users/1).
- Ekstra açıklama yazma, sadece bu formatta satırlar üret.

Şimdi tam olarak {num_cases} satır üret.
""".strip()


def parse_llm_lines_to_rows(
    lines: List[str],
    op: ApiOperation,
    generator_name: str,
) -> List[Dict]:
    """LLM çıktısındaki pipe-delimited satırları sözlük listesine dönüştürür."""
    rows: List[Dict] = []
    for line in lines:
        parts = [p.strip() for p in line.split("|")]
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


class BaseGenerator(ABC):
    """Tüm test senaryosu üreticileri için temel sınıf."""

    def generate(
        self,
        operations: List[ApiOperation],
        variant_name: str,
        variant_desc: str,
        num_cases: int,
    ) -> List[Dict]:
        """Verilen operasyonlar için test senaryoları üretir."""
        rows: List[Dict] = []
        for op in operations:
            result = self._generate_for_operation_with_retry(
                op, variant_name, variant_desc, num_cases
            )
            rows.extend(result)
        return rows

    def _generate_for_operation_with_retry(
        self,
        op: ApiOperation,
        variant_name: str,
        variant_desc: str,
        num_cases: int,
    ) -> List[Dict]:
        """Tek operasyon için retry mantığıyla senaryo üretir."""
        for attempt in range(1, RETRY_MAX_ATTEMPTS + 1):
            try:
                return self._generate_for_operation(op, variant_name, variant_desc, num_cases)
            except Exception as e:
                if _is_non_retryable_generation_error(e):
                    print(f"  [HATA] {op.op_id} kalici provider/kredi hatasi: {e}")
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
