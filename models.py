"""Proje genelinde kullanilan veri modelleri."""

import json
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class TokenUsage:
    """Bir LLM cagrisinin token tuketimi.

    `split_available=False` ise saglayici girdi/cikti ayrimini vermemistir ve
    yalnizca toplam bilinir. Bu durumda AYRISTIRMA UYDURULMAZ; maliyet hesabi
    bu cagriyi "ayristirilamiyor" olarak isaretler.

    `reasoning_tokens`: saglayicinin AYRI raporladigi dusunme/akil yurutme
    token'lari (Gemini `thoughts_token_count`, OpenAI uyumlu API'lerde
    `completion_tokens_details.reasoning_tokens`). Her iki saglayici da bunlari
    CIKTI fiyatindan faturalandirir; fark, sayinin `output_tokens`in ICINDE
    olup olmadigidir:
      * OpenAI/Groq: reasoning, completion_tokens'in ICINDEDIR  -> included=True
      * Gemini:      thoughts, candidates_token_count'un DISINDADIR -> included=False
    Bu yuzden faturalanabilir cikti `billable_output_tokens` ile hesaplanir.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    split_available: bool = False
    reasoning_tokens: int = 0
    reasoning_included_in_output: bool = True

    def __post_init__(self) -> None:
        if not self.total_tokens:
            self.total_tokens = (self.input_tokens or 0) + (self.output_tokens or 0)

    @property
    def billable_output_tokens(self) -> int:
        """Cikti fiyatindan faturalanacak token sayisi."""
        if self.reasoning_included_in_output:
            return self.output_tokens or 0
        return (self.output_tokens or 0) + (self.reasoning_tokens or 0)

    @classmethod
    def coerce(cls, value: Any) -> "TokenUsage":
        """int ya da TokenUsage kabul eder (geri uyumluluk)."""
        if isinstance(value, cls):
            return value
        try:
            return cls(total_tokens=int(value or 0))
        except (TypeError, ValueError):
            return cls()

    def to_dict(self) -> dict:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "split_available": self.split_available,
            "reasoning_tokens": self.reasoning_tokens,
            "reasoning_included_in_output": self.reasoning_included_in_output,
            "billable_output_tokens": self.billable_output_tokens,
        }


@dataclass
class ApiOperation:
    """OpenAPI'den veya manuel giristen alinan bir API operasyonunu temsil eder."""

    op_id: str = ""
    method: str = ""
    path: str = ""
    summary: str = ""
    description: str = ""
    example_body: str = ""
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

    def __post_init__(self) -> None:
        if not self.example_body and self.request_body_examples:
            self.example_body = self._serialize_example(self.request_body_examples[0])
        elif self.example_body and not self.request_body_examples:
            parsed = self._parse_example_body(self.example_body)
            if isinstance(parsed, dict):
                self.request_body_examples = [parsed]

    @staticmethod
    def _serialize_example(example: Any) -> str:
        if isinstance(example, str):
            return example
        try:
            return json.dumps(example, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(example)

    @staticmethod
    def _parse_example_body(example_body: str) -> Any:
        try:
            return json.loads(example_body)
        except (TypeError, ValueError, json.JSONDecodeError):
            return None


@dataclass
class TestCase:
    """Uretilen bir test senaryosunu temsil eder."""

    generator: str = ""
    operation_id: str = ""
    http_method: str = ""
    path: str = ""
    tc_id: str = ""
    title: str = ""
    request_body: Any = None
    expected_status: Any = None
    expected_result: str = ""
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
    test_type: str = "positive"   # positive, negative, boundary, auth, contract, schema, error
    priority: str = "medium"      # high, medium, low
    url: str = ""
    actual_status: object = ""
    actual_body: str = ""
    passed: object = None
    validation_errors: list = field(default_factory=list)

    def __post_init__(self) -> None:
        request = {
            "path_params": {},
            "query_params": {},
            "headers": {},
            "cookies": {},
            "body": None,
            **(self.request or {}),
        }
        expected = {
            "status": None,
            "allowed_statuses": [],
            "result": "",
            "assertions": [],
            "response_schema_check": False,
            **(self.expected or {}),
        }

        if request.get("body") is None and self.request_body is not None:
            request["body"] = self.request_body
        elif self.request_body is None:
            self.request_body = request.get("body")

        if expected.get("status") is None and self.expected_status is not None:
            expected["status"] = self.expected_status
        elif self.expected_status is None:
            self.expected_status = expected.get("status")

        if not expected.get("allowed_statuses") and self.expected_status is not None:
            expected["allowed_statuses"] = [self.expected_status]

        if not expected.get("result") and self.expected_result:
            expected["result"] = self.expected_result
        elif not self.expected_result:
            self.expected_result = expected.get("result", "")

        self.request = request
        self.expected = expected

    def to_dict(self) -> dict:
        body = self.request.get("body", self.request_body)
        exp_status = self.expected.get("status", self.expected_status)
        exp_result = self.expected.get("result", self.expected_result)
        return {
            "generator": self.generator,
            "operation_id": self.operation_id,
            "http_method": self.http_method,
            "path": self.path,
            "tc_id": self.tc_id,
            "title": self.title,
            "request_body": json.dumps(body, ensure_ascii=False) if body is not None else "",
            "expected_status": exp_status if exp_status is not None else "",
            "expected_result": exp_result,
            "request": self.request,
            "expected": self.expected,
            "test_type": self.test_type,
            "priority": self.priority,
            "url": self.url,
            "actual_status": self.actual_status,
            "actual_body": self.actual_body,
            "pass": self.passed,
            "validation_errors": self.validation_errors,
        }
