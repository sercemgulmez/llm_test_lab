"""Proje genelinde kullanılan veri modelleri."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ApiOperation:
    """OpenAPI'den veya manuel girişten alınan bir API operasyonunu temsil eder."""
    op_id: str
    method: str
    path: str
    summary: str
    description: str
    example_body: str = ""   # curl'den veya manuel girişten gelen örnek request body


@dataclass
class TestCase:
    """Üretilen bir test senaryosunu temsil eder."""
    generator: str
    operation_id: str
    http_method: str
    path: str
    tc_id: str
    title: str
    request_body: str          # JSON string veya ""
    expected_status: object    # int veya ""
    expected_result: str
    # Yürütme sonrası doldurulur
    url: str = ""
    actual_status: object = ""  # int veya ""
    passed: object = None       # True / False / None

    def to_dict(self) -> dict:
        return {
            "generator": self.generator,
            "operation_id": self.operation_id,
            "http_method": self.http_method,
            "path": self.path,
            "tc_id": self.tc_id,
            "title": self.title,
            "request_body": self.request_body,
            "expected_status": self.expected_status,
            "expected_result": self.expected_result,
            "url": self.url,
            "actual_status": self.actual_status,
            "pass": self.passed,
        }
