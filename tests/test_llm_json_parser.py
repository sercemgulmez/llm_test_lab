from generators.base import extract_json_array, parse_llm_json_to_rows
from models import ApiOperation


def _sample_operation():
    return ApiOperation(
        op_id="OP1",
        method="POST",
        path="/users",
        summary="Create user",
        description="Creates a user",
    )


def test_extract_json_array_parses_direct_json():
    text = """
    [
      {
        "tc_id": "OP1_TC1",
        "title": "Valid create",
        "test_type": "positive",
        "priority": "P0",
        "request": {
          "path_params": {},
          "query_params": {},
          "headers": {},
          "cookies": {},
          "body": {"name": "Ada"}
        },
        "expected": {
          "status": 201,
          "allowed_statuses": [201],
          "result": "Created",
          "assertions": [{"type": "status_code", "expected": 201}],
          "response_schema_check": true
        }
      }
    ]
    """

    parsed = extract_json_array(text)
    rows = parse_llm_json_to_rows(text, _sample_operation(), "LLM-X")

    assert len(parsed) == 1
    assert len(rows) == 1
    assert rows[0]["tc_id"] == "OP1_TC1"
    assert rows[0]["expected_status"] == 201


def test_extract_json_array_parses_markdown_code_block():
    text = """```json
    [
      {
        "tc_id": "OP1_TC2",
        "title": "Unauthorized request",
        "test_type": "auth",
        "priority": "P1",
        "request": {
          "path_params": {},
          "query_params": {},
          "headers": {},
          "cookies": {},
          "body": {"name": "Ada"}
        },
        "expected": {
          "status": 401,
          "allowed_statuses": [401],
          "result": "Unauthorized",
          "assertions": [{"type": "status_code", "expected": 401}],
          "response_schema_check": false
        }
      }
    ]
    ```"""

    rows = parse_llm_json_to_rows(text, _sample_operation(), "LLM-X")

    assert len(rows) == 1
    assert rows[0]["tc_id"] == "OP1_TC2"
    assert rows[0]["expected"]["allowed_statuses"] == [401]
