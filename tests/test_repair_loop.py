from __future__ import annotations

import json

from generators.base import BaseGenerator
from models import ApiOperation


def _repair_operation():
    return ApiOperation(
        op_id="ORDERS",
        method="POST",
        path="/orders/{orderId}",
        summary="Create order",
        description="Creates an order",
        parameters=[
            {"name": "orderId", "in": "path", "required": True, "schema": {"type": "string"}},
            {"name": "locale", "in": "query", "required": False, "schema": {"type": "string"}},
        ],
        request_body_schema={
            "type": "object",
            "required": ["sku"],
            "properties": {"sku": {"type": "string"}, "qty": {"type": "integer"}},
        },
        response_schemas={"201": {"description": "Created"}, "400": {"description": "Bad request"}},
        security=[{"bearerAuth": []}],
    )


def _case(op: ApiOperation, index: int, *, tc_id: str | None = None, method: str | None = None) -> dict:
    status = 201 if index % 2 else 400
    return {
        "tc_id": tc_id or f"{op.op_id}_TC{index}",
        "title": f"Case {index}",
        "test_type": "positive" if status == 201 else "negative",
        "priority": "P1",
        "http_method": method or op.method,
        "path": op.path,
        "request": {
            "path_params": {"orderId": f"ord-{index}"},
            "query_params": {"locale": "tr-TR"},
            "headers": {},
            "cookies": {},
            "body": {"sku": f"sku-{index}", "qty": index},
        },
        "expected": {
            "status": status,
            "allowed_statuses": [status],
            "result": "Result",
            "assertions": [{"type": "status_code", "expected": status}],
            "response_schema_check": True,
        },
    }


class DummyRepairGenerator(BaseGenerator):
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def _generate_for_operation(self, op, variant_name, variant_desc, num_cases):
        generator_name = "LLM-Dummy-basic"

        def request_completion(prompt: str):
            self.prompts.append(prompt)
            response = self.responses.pop(0)
            return response, 25

        return self._generate_cases_with_repair(
            op=op,
            variant_name=variant_name,
            variant_desc=variant_desc,
            num_cases=num_cases,
            generator_name=generator_name,
            request_completion=request_completion,
        )


def test_repair_and_fallback_fill_missing_cases(caplog):
    import logging
    op = _repair_operation()
    first_batch = json.dumps([_case(op, index) for index in range(1, 7)])
    repair_batch = json.dumps(
        [
            _case(op, 7),
            _case(op, 8),
            _case(op, 1, tc_id="ORDERS_TC1"),
            _case(op, 9, method="GET"),
        ]
    )
    final_empty_repair = "[]"

    gen = DummyRepairGenerator([first_batch, repair_batch, final_empty_repair])
    with caplog.at_level(logging.INFO, logger="generators.base"):
        rows = gen.generate([op], "basic", "repair", 10)
    log_text = caplog.text

    assert len(rows) == 10
    assert len({row["tc_id"] for row in rows}) == 10
    assert rows[0]["generator"] == "LLM-Dummy-basic"
    assert "requested_cases=10" in log_text
    assert "fallback_cases=2" in log_text


def test_fallback_completes_requested_missing_count():
    op = _repair_operation()
    gen = DummyRepairGenerator(["[]", "[]", "[]"])

    rows = gen.generate([op], "basic", "repair", 3)

    assert len(rows) == 3
    assert all(row["generator"] == "LLM-Dummy-basic" for row in rows)
    assert len({row["tc_id"] for row in rows}) == 3
