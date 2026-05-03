"""Şablona dayalı geleneksel test senaryosu üreticisi."""

from typing import Dict, List

from models import ApiOperation, TestCase
from generators.base import BaseGenerator, _infer_test_type


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

    def generate(self, operations: List[ApiOperation], *_args, **_kwargs) -> List[Dict]:
        """Varyant ve retry gerekmez, direkt üretir."""
        rows: List[Dict] = []
        for op in operations:
            print(
                f"[Traditional] {op.op_id} ({op.method} {op.path}) şablon senaryolar üretiliyor..."
            )
            rows.extend(self._generate_for_operation(op, "", "", 0))
        return rows

    def _generate_for_operation(
        self, op: ApiOperation, _variant_name: str, _variant_desc: str, _num_cases: int
    ) -> List[Dict]:
        success_code = self._SUCCESS_CODES.get(op.method, 200)
        # (tc_id_suffix, title, body_dict, exp_status)
        templates = [
            (1, f"{success_code} - Mutlu senaryo",      None,             success_code),
            (2, "400 - Validasyon hatası",               {"dummy": "invalid"}, 400),
            (3, "401 - Yetkisiz erişim",                 None,             401),
            (4, "404 - Kaynak bulunamadı",               None,             404),
            (5, "500 - Sunucu hatası",                   None,             500),
        ]

        _results = [
            "İşlem başarıyla tamamlanmalı.",
            "Hatalı istek için validasyon hatası dönmeli.",
            "Yetkisiz kullanıcı için 401 dönmeli.",
            "Var olmayan kaynak için 404 dönmeli.",
            "Beklenmedik hata durumunda 500 dönmeli.",
        ]

        rows = []
        for (suffix, title, body, exp_status), exp_result in zip(templates, _results):
            tc = TestCase(
                generator=self.GENERATOR_NAME,
                operation_id=op.op_id,
                http_method=op.method,
                path=op.path,
                tc_id=f"{op.op_id}_TC{suffix}",
                title=title,
                test_type=_infer_test_type(exp_status, title),
                request={
                    "path_params": {},
                    "query_params": {},
                    "headers": {},
                    "cookies": {},
                    "body": body,
                },
                expected={
                    "status": exp_status,
                    "allowed_statuses": [exp_status],
                    "result": exp_result,
                    "assertions": [],
                    "response_schema_check": False,
                },
            )
            rows.append(tc.to_dict())
        return rows
