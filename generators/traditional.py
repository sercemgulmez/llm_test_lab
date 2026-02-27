"""Şablona dayalı geleneksel test senaryosu üreticisi."""

import json
from typing import Dict, List

from models import ApiOperation
from generators.base import BaseGenerator


class TraditionalGenerator(BaseGenerator):
    """Sabit şablonlara dayalı test senaryoları üretir. LLM gerektirmez."""

    GENERATOR_NAME = "Traditional-Template"

    # HTTP metoduna göre varsayılan başarı kodu
    _SUCCESS_CODES = {
        "GET": 200,
        "POST": 201,
        "PUT": 200,
        "PATCH": 200,
        "DELETE": 204,
        "HEAD": 200,
        "OPTIONS": 200,
    }

    def generate(self, operations: List[ApiOperation], *args, **kwargs) -> List[Dict]:
        """Varyant ve retry gerekmez, direkt üretir."""
        rows: List[Dict] = []
        for op in operations:
            print(
                f"[Traditional] {op.op_id} ({op.method} {op.path}) şablon senaryolar üretiliyor..."
            )
            rows.extend(self._generate_for_operation(op, "", "", 0))
        return rows

    def _generate_for_operation(
        self, op: ApiOperation, variant_name: str, variant_desc: str, num_cases: int
    ) -> List[Dict]:
        success_code = self._SUCCESS_CODES.get(op.method, 200)
        templates = [
            (
                f"{op.op_id}_TC1",
                f"{success_code} - Mutlu senaryo",
                None,
                success_code,
                "İşlem başarıyla tamamlanmalı.",
            ),
            (
                f"{op.op_id}_TC2",
                "400 - Validasyon hatası",
                {"dummy": "invalid"},
                400,
                "Hatalı istek için validasyon hatası dönmeli.",
            ),
            (
                f"{op.op_id}_TC3",
                "401 - Yetkisiz erişim",
                None,
                401,
                "Yetkisiz kullanıcı için 401 dönmeli.",
            ),
            (
                f"{op.op_id}_TC4",
                "404 - Kaynak bulunamadı",
                None,
                404,
                "Var olmayan kaynak için 404 dönmeli.",
            ),
            (
                f"{op.op_id}_TC5",
                "500 - Sunucu hatası",
                None,
                500,
                "Beklenmedik hata durumunda 500 dönmeli.",
            ),
        ]

        rows = []
        for tc_id, title, body, exp_status, exp_result in templates:
            rows.append(
                {
                    "generator": self.GENERATOR_NAME,
                    "operation_id": op.op_id,
                    "http_method": op.method,
                    "path": op.path,
                    "tc_id": tc_id,
                    "title": title,
                    "request_body": json.dumps(body) if body is not None else "",
                    "expected_status": exp_status,
                    "expected_result": exp_result,
                }
            )
        return rows
