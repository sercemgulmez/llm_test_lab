from generators.traditional import TraditionalGenerator
from models import ApiOperation


def _schema_aware_operation():
    return ApiOperation(
        op_id="USER_CREATE",
        method="POST",
        path="/users/{userId}",
        summary="Create user",
        description="Creates a user",
        parameters=[
            {"name": "userId", "in": "path", "required": True, "schema": {"type": "string"}},
            {"name": "locale", "in": "query", "required": True, "schema": {"type": "string"}},
            {"name": "X-Tenant", "in": "header", "required": False, "schema": {"type": "string"}},
        ],
        request_body_schema={
            "type": "object",
            "required": ["name", "role"],
            "properties": {
                "name": {"type": "string", "minLength": 2},
                "age": {"type": "integer", "minimum": 18, "maximum": 99},
                "role": {"type": "string", "enum": ["admin", "user"]},
                "active": {"type": "boolean"},
            },
        },
        response_schemas={
            "201": {"description": "Created"},
            "400": {"description": "Bad Request"},
            "401": {"description": "Unauthorized"},
            "404": {"description": "Not Found"},
            "415": {"description": "Unsupported Media Type"},
        },
        security=[{"bearerAuth": []}],
    )


def _find_case(rows, phrase: str):
    return next(row for row in rows if phrase in row["title"])


def test_traditional_generator_builds_valid_body_from_schema():
    rows = TraditionalGenerator().generate([_schema_aware_operation()], "", "", 10)

    positive = _find_case(rows, "Valid request")

    assert positive["request"]["body"]["name"] == "xx"
    assert positive["request"]["body"]["age"] == 18
    assert positive["request"]["body"]["role"] == "admin"
    assert positive["request"]["body"]["active"] is True
    assert positive["expected_status"] == 201


def test_traditional_generator_produces_required_field_missing_case():
    rows = TraditionalGenerator().generate([_schema_aware_operation()], "", "", 10)

    missing_case = _find_case(rows, "Missing required body field")

    assert "name" not in missing_case["request"]["body"]
    assert missing_case["expected_status"] == 400


def test_traditional_generator_produces_invalid_type_case():
    rows = TraditionalGenerator().generate([_schema_aware_operation()], "", "", 10)

    invalid_type = _find_case(rows, "Invalid body field type")

    assert invalid_type["request"]["body"]["name"] == 999
    assert invalid_type["expected_status"] == 400


def test_traditional_generator_produces_invalid_enum_case():
    rows = TraditionalGenerator().generate([_schema_aware_operation()], "", "", 10)

    enum_case = _find_case(rows, "Invalid enum value")

    assert enum_case["request"]["body"]["role"] == "__invalid_enum__"
    assert enum_case["expected_status"] == 400


def test_traditional_generator_produces_invalid_path_param_case():
    rows = TraditionalGenerator().generate([_schema_aware_operation()], "", "", 10)

    path_case = _find_case(rows, "Invalid path param")

    assert path_case["request"]["path_params"]["userId"] == 999
    assert path_case["expected_status"] == 404


def test_traditional_generator_produces_query_param_cases():
    rows = TraditionalGenerator().generate([_schema_aware_operation()], "", "", 10)

    positive = _find_case(rows, "Valid request")
    missing_query = _find_case(rows, "Missing required query param")
    invalid_query = _find_case(rows, "Invalid query param type")

    assert positive["request"]["query_params"]["locale"] == "x"
    assert "locale" not in missing_query["request"]["query_params"]
    assert invalid_query["request"]["query_params"]["locale"] == 999


def test_traditional_generator_produces_auth_negative_cases():
    rows = TraditionalGenerator().generate([_schema_aware_operation()], "", "", 10)

    missing_auth = _find_case(rows, "Missing auth header")
    invalid_auth = _find_case(rows, "Invalid auth token")

    assert "Authorization" not in missing_auth["request"]["headers"]
    assert invalid_auth["request"]["headers"]["Authorization"] == "Bearer invalid-token"
    assert missing_auth["expected_status"] == 401
    assert invalid_auth["expected_status"] == 401


def test_traditional_generator_honors_num_cases():
    rows = TraditionalGenerator().generate([_schema_aware_operation()], "", "", 8)

    assert len(rows) == 8
