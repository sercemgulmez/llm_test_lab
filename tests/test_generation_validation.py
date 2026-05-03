from generators.base import normalize_generated_case, validate_generated_cases
from models import ApiOperation


def _contract_operation():
    return ApiOperation(
        op_id="USER_CREATE",
        method="POST",
        path="/users/{userId}",
        summary="Create user",
        description="Creates a user for the given path id",
        parameters=[
            {"name": "userId", "in": "path", "required": True, "schema": {"type": "string"}},
        ],
        request_body_schema={
            "type": "object",
            "required": ["name"],
            "properties": {"name": {"type": "string"}, "email": {"type": "string"}},
        },
    )


def test_duplicate_tc_id_is_invalid():
    op = _contract_operation()
    case = {
        "tc_id": "USER_CREATE_TC1",
        "title": "Valid create",
        "test_type": "positive",
        "request": {
            "path_params": {"userId": "u1"},
            "query_params": {},
            "headers": {},
            "cookies": {},
            "body": {"name": "Ada"},
        },
        "expected": {"status": 201, "allowed_statuses": [201], "result": "Created"},
    }
    rows = [
        normalize_generated_case(case, op, "LLM-X"),
        normalize_generated_case(case, op, "LLM-X"),
    ]

    valid, invalid = validate_generated_cases(op, rows, 2)

    assert len(valid) == 1
    assert any("duplicate" in error for item in invalid for error in item["errors"])


def test_method_and_path_mismatch_are_invalid():
    op = _contract_operation()
    row = normalize_generated_case(
        {
            "tc_id": "USER_CREATE_TC1",
            "title": "Wrong method",
            "http_method": "GET",
            "path": "/wrong",
            "request": {
                "path_params": {"userId": "u1"},
                "query_params": {},
                "headers": {},
                "cookies": {},
                "body": None,
            },
            "expected": {"status": 200, "allowed_statuses": [200], "result": "OK"},
        },
        op,
        "LLM-X",
    )

    valid, invalid = validate_generated_cases(op, [row], 1)

    assert valid == []
    assert any("http_method" in error for item in invalid for error in item["errors"])
    assert any("path op.path" in error for item in invalid for error in item["errors"])


def test_missing_required_path_param_is_invalid():
    op = _contract_operation()
    row = normalize_generated_case(
        {
            "tc_id": "USER_CREATE_TC2",
            "title": "Missing path param",
            "test_type": "positive",
            "request": {
                "path_params": {},
                "query_params": {},
                "headers": {},
                "cookies": {},
                "body": {"name": "Ada"},
            },
            "expected": {"status": 201, "allowed_statuses": [201], "result": "Created"},
        },
        op,
        "LLM-X",
    )

    valid, invalid = validate_generated_cases(op, [row], 1)

    assert valid == []
    assert any("path param eksik: userId" in error for item in invalid for error in item["errors"])


def test_positive_case_requires_required_body_fields():
    op = _contract_operation()
    row = normalize_generated_case(
        {
            "tc_id": "USER_CREATE_TC3",
            "title": "Missing body field",
            "test_type": "positive",
            "request": {
                "path_params": {"userId": "u1"},
                "query_params": {},
                "headers": {},
                "cookies": {},
                "body": {"email": "ada@example.com"},
            },
            "expected": {"status": 201, "allowed_statuses": [201], "result": "Created"},
        },
        op,
        "LLM-X",
    )

    valid, invalid = validate_generated_cases(op, [row], 1)

    assert valid == []
    assert any("required body field eksik: name" in error for item in invalid for error in item["errors"])
