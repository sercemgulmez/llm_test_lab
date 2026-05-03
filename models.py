"""Proje genelinde kullanılan veri modelleri."""

import json
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
    tags: list[str] = field(default_factory=list)
    parameters: list[dict] = field(default_factory=list)
    request_body_schema: Optional[dict] = None
    request_body_examples: list[dict] = field(default_factory=list)
    response_schemas: dict = field(default_factory=dict)
    response_examples: dict = field(default_factory=dict)
    security: list[dict] = field(default_factory=list)
    content_types: list[str] = field(default_factory=list)
    servers: list[str] = field(default_factory=list)
    raw_operation: dict = field(default_factory=dict)


@dataclass
class TestCase:
    """Üretilen bir test senaryosunu temsil eder."""
    generator: str
    operation_id: str
    http_method: str
    path: str
    tc_id: str
    title: str
    test_type: str = "positive"   # positive, negative, boundary, auth, contract, schema, error
    priority: str = "medium"      # high, medium, low
    request: dict = field(default_factory=lambda: {
        "path_params": {},
        "query_params": {},
        "headers": {},
        "cookies": {},
        "body": None,
    })
    expected: dict = field(default_factory=lambda: {
        "status": None,
        "allowed_statuses": [],
        "result": "",
        "assertions": [],
        "response_schema_check": False,
    })
    url: str = ""
    actual_status: object = ""
    actual_body: str = ""
    passed: object = None
    validation_errors: list = field(default_factory=list)

    def to_dict(self) -> dict:
        body = self.request.get("body")
        exp_status = self.expected.get("status")
        return {
            # Yapısal alanlar
            "generator": self.generator,
            "operation_id": self.operation_id,
            "http_method": self.http_method,
            "path": self.path,
            "tc_id": self.tc_id,
            "title": self.title,
            "test_type": self.test_type,
            "priority": self.priority,
            "request": self.request,
            "expected": self.expected,
            "url": self.url,
            "actual_status": self.actual_status,
            "actual_body": self.actual_body,
            "pass": self.passed,
            "validation_errors": self.validation_errors,
            # Geriye dönük uyumluluk (CSV / runner)
            "request_body": json.dumps(body, ensure_ascii=False) if body is not None else "",
            "expected_status": exp_status if exp_status is not None else "",
            "expected_result": self.expected.get("result", ""),
        }
