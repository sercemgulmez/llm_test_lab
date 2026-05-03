from parsers.openapi import extract_operations_from_openapi


def _build_oas3_spec():
    return {
        "openapi": "3.0.3",
        "servers": [{"url": "https://api.example.com/v1"}],
        "components": {
            "parameters": {
                "UserId": {
                    "name": "userId",
                    "in": "path",
                    "required": True,
                    "description": "User identifier",
                    "schema": {"type": "string"},
                },
            },
            "schemas": {
                "UserCreateRequest": {
                    "type": "object",
                    "required": ["name"],
                    "properties": {
                        "name": {"type": "string"},
                        "email": {"type": "string", "format": "email"},
                    },
                },
                "UserResponse": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "name": {"type": "string"},
                    },
                },
            },
            "requestBodies": {
                "CreateUserBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/UserCreateRequest"},
                            "example": {"name": "Ada", "email": "ada@example.com"},
                        }
                    },
                }
            },
            "responses": {
                "CreatedUser": {
                    "description": "Created",
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/UserResponse"},
                            "examples": {
                                "default": {"value": {"id": "u1", "name": "Ada"}}
                            },
                        }
                    },
                }
            },
            "securitySchemes": {
                "bearerAuth": {"type": "http", "scheme": "bearer"},
            },
        },
        "paths": {
            "/users/{userId}": {
                "parameters": [{"$ref": "#/components/parameters/UserId"}],
                "post": {
                    "operationId": "createUser",
                    "summary": "Create user",
                    "description": "Creates a user",
                    "tags": ["users"],
                    "parameters": [
                        {
                            "name": "include",
                            "in": "query",
                            "required": False,
                            "description": "Related entities",
                            "schema": {"type": "string", "enum": ["roles", "teams"]},
                            "example": "roles",
                        },
                        {
                            "name": "X-Tenant",
                            "in": "header",
                            "required": True,
                            "description": "Tenant key",
                            "schema": {"type": "string"},
                            "example": "tenant-a",
                        },
                    ],
                    "requestBody": {"$ref": "#/components/requestBodies/CreateUserBody"},
                    "responses": {
                        "201": {"$ref": "#/components/responses/CreatedUser"},
                    },
                    "security": [{"bearerAuth": []}],
                },
            }
        },
    }


def test_extracts_path_param():
    op = extract_operations_from_openapi(_build_oas3_spec())[0]

    path_param = next(p for p in op.parameters if p["in"] == "path")
    assert path_param["name"] == "userId"
    assert path_param["required"] is True
    assert path_param["schema"]["type"] == "string"


def test_extracts_query_param():
    op = extract_operations_from_openapi(_build_oas3_spec())[0]

    query_param = next(p for p in op.parameters if p["in"] == "query")
    assert query_param["name"] == "include"
    assert query_param["example"] == "roles"
    assert query_param["enum"] == ["roles", "teams"]


def test_extracts_header_param():
    op = extract_operations_from_openapi(_build_oas3_spec())[0]

    header_param = next(p for p in op.parameters if p["in"] == "header")
    assert header_param["name"] == "X-Tenant"
    assert header_param["required"] is True
    assert header_param["description"] == "Tenant key"


def test_extracts_request_body_schema():
    op = extract_operations_from_openapi(_build_oas3_spec())[0]

    assert op.request_body_schema is not None
    assert op.request_body_schema["type"] == "object"
    assert "name" in op.request_body_schema["properties"]
    assert op.content_types == ["application/json"]


def test_preserves_required_body_fields():
    op = extract_operations_from_openapi(_build_oas3_spec())[0]

    assert op.request_body_schema["required"] == ["name"]
    assert op.request_body_schema["x-request-body-required"] is True


def test_extracts_response_schema():
    op = extract_operations_from_openapi(_build_oas3_spec())[0]

    response_schema = op.response_schemas["201"]["content"]["application/json"]["schema"]
    assert op.response_schemas["201"]["description"] == "Created"
    assert response_schema["properties"]["id"]["type"] == "string"


def test_resolves_refs_recursively():
    op = extract_operations_from_openapi(_build_oas3_spec())[0]

    assert "$ref" not in op.request_body_schema
    assert "$ref" not in op.response_schemas["201"]["content"]["application/json"]["schema"]
    assert op.raw_operation["requestBody"]["content"]["application/json"]["schema"]["properties"]["email"]["format"] == "email"


def test_extracts_security_information():
    op = extract_operations_from_openapi(_build_oas3_spec())[0]

    assert op.security == [{"bearerAuth": []}]
    assert op.tags == ["users"]
    assert op.servers == ["https://api.example.com/v1"]


def test_minimal_openapi_spec_keeps_backward_compatibility():
    spec = {
        "openapi": "3.0.0",
        "paths": {
            "/health": {
                "get": {
                    "summary": "Health",
                }
            }
        },
    }

    op = extract_operations_from_openapi(spec)[0]

    assert op.op_id == "OP1"
    assert op.method == "GET"
    assert op.summary == "Health"
    assert op.description == ""
    assert op.example_body == ""
    assert op.parameters == []
    assert op.request_body_schema is None
    assert op.response_schemas == {}
    assert op.security == []
    assert op.content_types == []


def test_swagger2_extracts_body_schema_and_definitions():
    spec = {
        "swagger": "2.0",
        "host": "api.example.com",
        "basePath": "/v1",
        "schemes": ["https"],
        "consumes": ["application/json"],
        "produces": ["application/json"],
        "definitions": {
            "CreatePet": {
                "type": "object",
                "required": ["name"],
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                },
            },
            "Pet": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "name": {"type": "string"},
                },
            },
        },
        "paths": {
            "/pets": {
                "post": {
                    "operationId": "createPet",
                    "parameters": [
                        {
                            "name": "body",
                            "in": "body",
                            "required": True,
                            "schema": {"$ref": "#/definitions/CreatePet"},
                        }
                    ],
                    "responses": {
                        "201": {
                            "description": "Created",
                            "schema": {"$ref": "#/definitions/Pet"},
                        }
                    },
                }
            }
        },
    }

    op = extract_operations_from_openapi(spec)[0]

    assert op.request_body_schema["properties"]["name"]["type"] == "string"
    assert op.request_body_schema["x-request-body-required"] is True
    assert op.response_schemas["201"]["content"]["application/json"]["schema"]["properties"]["id"]["type"] == "integer"
    assert op.content_types == ["application/json"]
    assert op.servers == ["https://api.example.com/v1"]
