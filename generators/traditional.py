"""Schema-aware geleneksel test senaryosu ureticisi."""

import copy
from typing import Any, Dict, List, Optional

from generators.base import BaseGenerator, _infer_test_type
from models import ApiOperation, TestCase


class TraditionalGenerator(BaseGenerator):
    """OpenAPI kontratina dayali baseline test senaryolari uretir."""

    GENERATOR_NAME = "Traditional-Template"

    _SUCCESS_CODES = {
        "GET": 200,
        "POST": 201,
        "PUT": 200,
        "PATCH": 200,
        "DELETE": 204,
        "HEAD": 200,
        "OPTIONS": 200,
    }

    _DEFAULT_ERROR_CODES = [400, 401, 404, 415, 422, 500]

    def generate(self, operations: List[ApiOperation], *_args, **_kwargs) -> List[Dict]:
        """LLM kullanmadan senaryolari uretir ve num_cases parametresini dikkate alir."""
        num_cases = 5
        self._generation_summaries = []
        if len(_args) >= 3 and isinstance(_args[2], int):
            num_cases = _args[2]
        elif "num_cases" in _kwargs and isinstance(_kwargs["num_cases"], int):
            num_cases = _kwargs["num_cases"]
        if num_cases <= 0:
            num_cases = 5

        rows: List[Dict] = []
        for op in operations:
            print(f"[Traditional] {op.op_id} ({op.method} {op.path}) schema-aware senaryolar uretiliyor...")
            rows.extend(self._generate_for_operation(op, "", "", num_cases))
        return rows

    def _generate_for_operation(
        self, op: ApiOperation, _variant_name: str, _variant_desc: str, _num_cases: int
    ) -> List[Dict]:
        if not hasattr(self, "_generation_summaries"):
            self._generation_summaries = []
        num_cases = _num_cases if _num_cases and _num_cases > 0 else 5
        base_request = self._build_positive_request(op)
        positive_body = copy.deepcopy(base_request["body"])
        cases = []

        positive_status = self._pick_status(op, [200, 201, 202, 204], self._SUCCESS_CODES.get(op.method, 200))
        cases.append(
            self._build_case(
                op=op,
                title=f"{positive_status} - Valid request",
                request=base_request,
                status=positive_status,
                result="Gecerli istek kontrata uygun sekilde basarili olur.",
                test_type="positive",
                priority="P0",
                response_schema_check=bool(op.response_schemas),
            )
        )

        missing_required_body = self._missing_required_body_case(op, base_request)
        if missing_required_body:
            cases.append(missing_required_body)

        invalid_body_type = self._invalid_body_type_case(op, base_request)
        if invalid_body_type:
            cases.append(invalid_body_type)

        invalid_enum_body = self._invalid_enum_case(op, base_request)
        if invalid_enum_body:
            cases.append(invalid_enum_body)

        cases.append(self._boundary_case(op, base_request))

        missing_required_query = self._missing_required_query_case(op, base_request)
        if missing_required_query:
            cases.append(missing_required_query)

        invalid_query_type = self._invalid_query_type_case(op, base_request)
        if invalid_query_type:
            cases.append(invalid_query_type)

        invalid_path = self._invalid_path_param_case(op, base_request)
        if invalid_path:
            cases.append(invalid_path)

        auth_cases = self._auth_negative_cases(op, base_request)
        cases.extend(auth_cases)

        cases.append(self._error_expectation_case(op, base_request))

        if len(cases) < num_cases:
            cases.extend(self._extra_fallback_cases(op, base_request, num_cases - len(cases), positive_body))

        rows = []
        for index, case in enumerate(cases[:num_cases], start=1):
            tc = TestCase(
                generator=self.GENERATOR_NAME,
                operation_id=op.op_id,
                http_method=op.method,
                path=op.path,
                tc_id=f"{op.op_id}_TC{index}",
                title=case["title"],
                request=case["request"],
                expected=case["expected"],
                test_type=case["test_type"],
                priority=case["priority"],
            )
            row = tc.to_dict()
            row["generation_metadata"] = {
                "source": "traditional",
                "repaired": False,
                "fallback": False,
                "valid": True,
            }
            rows.append(row)
        self._generation_summaries.append(
            {
                "generator": self.GENERATOR_NAME,
                "operation_id": op.op_id,
                "method": op.method,
                "path": op.path,
                "requested_cases": num_cases,
                "parsed_cases": len(rows),
                "generated_cases": len(rows),
                "valid_cases": len(rows),
                "invalid_cases": 0,
                "repaired_cases": 0,
                "fallback_cases": 0,
                "validation_error_summary": {},
            }
        )
        return rows

    def _responses_available(self, op: ApiOperation) -> List[int]:
        statuses: List[int] = []
        for status_code in op.response_schemas.keys():
            try:
                status = int(str(status_code))
            except (TypeError, ValueError):
                continue
            if status not in statuses:
                statuses.append(status)
        return statuses

    def _pick_status(self, op: ApiOperation, preferred: List[int], fallback: int) -> int:
        available = self._responses_available(op)
        for status in preferred:
            if status in available:
                return status
        if available:
            return available[0]
        return fallback

    def _parameters_by_location(self, op: ApiOperation, location: str) -> List[dict]:
        return [param for param in op.parameters if param.get("in") == location]

    def _required_parameters(self, op: ApiOperation, location: str) -> List[dict]:
        return [param for param in self._parameters_by_location(op, location) if param.get("required")]

    def _valid_scalar_for_schema(self, schema: Optional[dict]) -> Any:
        schema = schema or {}
        if schema.get("enum"):
            return schema["enum"][0]
        if "example" in schema:
            return schema["example"]
        if "default" in schema:
            return schema["default"]

        schema_type = schema.get("type")
        if schema_type == "string":
            min_length = schema.get("minLength")
            return "x" * max(int(min_length or 0), 1)
        if schema_type == "integer":
            if "minimum" in schema:
                return int(schema["minimum"])
            if "maximum" in schema:
                return int(schema["maximum"])
            return 1
        if schema_type == "number":
            if "minimum" in schema:
                return schema["minimum"]
            if "maximum" in schema:
                return schema["maximum"]
            return 1.5
        if schema_type == "boolean":
            return True
        if schema_type == "array":
            item_schema = schema.get("items") if isinstance(schema.get("items"), dict) else {}
            return [self._valid_value_from_schema(item_schema)]
        if schema_type == "object" or schema.get("properties"):
            return self._valid_body_from_schema(schema)
        return "sample"

    def _valid_value_from_schema(self, schema: Optional[dict]) -> Any:
        schema = schema or {}
        if schema.get("oneOf") and isinstance(schema["oneOf"], list):
            return self._valid_value_from_schema(schema["oneOf"][0] if schema["oneOf"] else {})
        if schema.get("anyOf") and isinstance(schema["anyOf"], list):
            return self._valid_value_from_schema(schema["anyOf"][0] if schema["anyOf"] else {})
        if schema.get("allOf") and isinstance(schema["allOf"], list):
            merged: dict = {"type": "object", "properties": {}, "required": []}
            for part in schema["allOf"]:
                if not isinstance(part, dict):
                    continue
                merged["properties"].update(part.get("properties", {}))
                merged["required"].extend(part.get("required", []))
            return self._valid_value_from_schema(merged)
        return self._valid_scalar_for_schema(schema)

    def _valid_body_from_schema(self, schema: Optional[dict]) -> Optional[dict]:
        if not isinstance(schema, dict):
            return None
        properties = schema.get("properties")
        if not isinstance(properties, dict):
            return None

        body: dict = {}
        for field_name, field_schema in properties.items():
            if not isinstance(field_schema, dict):
                continue
            body[field_name] = self._valid_value_from_schema(field_schema)
        return body

    def _invalid_value_for_schema(self, schema: Optional[dict]) -> Any:
        schema = schema or {}
        schema_type = schema.get("type")
        if schema.get("enum"):
            return "__invalid_enum__"
        if schema_type == "string":
            return 999
        if schema_type in ("integer", "number"):
            return "not-a-number"
        if schema_type == "boolean":
            return "not-a-boolean"
        if schema_type == "array":
            return "not-an-array"
        if schema_type == "object" or schema.get("properties"):
            return "not-an-object"
        return None

    def _boundary_value_for_schema(self, schema: Optional[dict]) -> Any:
        schema = schema or {}
        schema_type = schema.get("type")
        if schema.get("enum"):
            return schema["enum"][0]
        if schema_type == "string":
            if "minLength" in schema:
                return "x" * max(int(schema["minLength"]), 1)
            if "maxLength" in schema:
                return "x" * max(int(schema["maxLength"]), 1)
            return ""
        if schema_type == "integer":
            if "minimum" in schema:
                return int(schema["minimum"])
            if "maximum" in schema:
                return int(schema["maximum"])
            return 0
        if schema_type == "number":
            if "minimum" in schema:
                return schema["minimum"]
            if "maximum" in schema:
                return schema["maximum"]
            return 0
        if schema_type == "array":
            item_schema = schema.get("items") if isinstance(schema.get("items"), dict) else {}
            min_items = int(schema.get("minItems", 1) or 1)
            return [self._valid_value_from_schema(item_schema) for _ in range(max(min_items, 1))]
        return self._valid_value_from_schema(schema)

    def _build_positive_request(self, op: ApiOperation) -> dict:
        request = {
            "path_params": {},
            "query_params": {},
            "headers": {},
            "cookies": {},
            "body": None,
        }

        for location in ("path", "query", "header"):
            for param in self._parameters_by_location(op, location):
                name = param.get("name")
                if not name:
                    continue
                if not param.get("required") and location == "header" and str(name).lower() == "authorization":
                    continue
                value = param.get("example")
                if value is None:
                    value = self._valid_value_from_schema(param.get("schema") if isinstance(param.get("schema"), dict) else {})
                if location == "path":
                    request["path_params"][name] = value
                elif location == "query":
                    request["query_params"][name] = value
                elif location == "header":
                    request["headers"][name] = value

        if op.security:
            request["headers"].setdefault("Authorization", "Bearer valid-token")

        if op.method != "GET":
            if op.request_body_examples:
                request["body"] = copy.deepcopy(op.request_body_examples[0])
            else:
                request["body"] = self._valid_body_from_schema(op.request_body_schema)

        return request

    def _build_case(
        self,
        op: ApiOperation,
        title: str,
        request: dict,
        status: int,
        result: str,
        test_type: str,
        priority: str,
        response_schema_check: bool = False,
    ) -> dict:
        return {
            "title": title,
            "request": copy.deepcopy(request),
            "expected": {
                "status": status,
                "allowed_statuses": [status],
                "result": result,
                "assertions": [{"type": "status_code", "expected": status}],
                "response_schema_check": response_schema_check,
            },
            "test_type": test_type or _infer_test_type(status, title),
            "priority": priority,
        }

    def _first_body_field(self, op: ApiOperation) -> tuple[Optional[str], Optional[dict]]:
        schema = op.request_body_schema if isinstance(op.request_body_schema, dict) else {}
        properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        for field_name, field_schema in properties.items():
            if isinstance(field_schema, dict):
                return field_name, field_schema
        return None, None

    def _missing_required_body_case(self, op: ApiOperation, base_request: dict) -> Optional[dict]:
        if not isinstance(base_request.get("body"), dict):
            return None
        required = op.request_body_schema.get("required") if isinstance(op.request_body_schema, dict) else []
        if not isinstance(required, list) or not required:
            return None

        request = copy.deepcopy(base_request)
        request["body"].pop(required[0], None)
        status = self._pick_status(op, [400, 422], 400)
        return self._build_case(
            op=op,
            title=f"{status} - Missing required body field",
            request=request,
            status=status,
            result=f"Zorunlu body alani '{required[0]}' olmadiginda dogrulama hatasi donmeli.",
            test_type="contract",
            priority="P0",
        )

    def _invalid_body_type_case(self, op: ApiOperation, base_request: dict) -> Optional[dict]:
        if not isinstance(base_request.get("body"), dict):
            return None
        field_name, field_schema = self._first_body_field(op)
        if not field_name or not isinstance(field_schema, dict):
            return None

        request = copy.deepcopy(base_request)
        request["body"][field_name] = self._invalid_value_for_schema(field_schema)
        status = self._pick_status(op, [400, 422], 400)
        return self._build_case(
            op=op,
            title=f"{status} - Invalid body field type",
            request=request,
            status=status,
            result=f"Body alani '{field_name}' yanlis tipte oldugunda istek reddedilmeli.",
            test_type="negative",
            priority="P1",
        )

    def _find_enum_in_body_schema(self, schema: Optional[dict], prefix: str = "") -> tuple[Optional[str], Optional[dict]]:
        if not isinstance(schema, dict):
            return None, None
        properties = schema.get("properties")
        if not isinstance(properties, dict):
            return None, None

        for field_name, field_schema in properties.items():
            if not isinstance(field_schema, dict):
                continue
            path = f"{prefix}.{field_name}" if prefix else field_name
            if field_schema.get("enum"):
                return path, field_schema
            nested_path, nested_schema = self._find_enum_in_body_schema(field_schema, path)
            if nested_path:
                return nested_path, nested_schema
        return None, None

    def _set_nested_value(self, body: dict, dotted_path: str, value: Any) -> None:
        target = body
        parts = dotted_path.split(".")
        for part in parts[:-1]:
            if not isinstance(target.get(part), dict):
                target[part] = {}
            target = target[part]
        target[parts[-1]] = value

    def _invalid_enum_case(self, op: ApiOperation, base_request: dict) -> Optional[dict]:
        request = copy.deepcopy(base_request)
        if isinstance(request.get("body"), dict):
            enum_path, enum_schema = self._find_enum_in_body_schema(op.request_body_schema)
            if enum_path and enum_schema:
                self._set_nested_value(request["body"], enum_path, "__invalid_enum__")
                status = self._pick_status(op, [400, 422], 400)
                return self._build_case(
                    op=op,
                    title=f"{status} - Invalid enum value",
                    request=request,
                    status=status,
                    result=f"Enum disi deger gonderildiginde '{enum_path}' icin dogrulama hatasi donmeli.",
                    test_type="negative",
                    priority="P1",
                )

        for param in op.parameters:
            if param.get("enum"):
                request = copy.deepcopy(base_request)
                location = param.get("in")
                if location == "query":
                    request["query_params"][param["name"]] = "__invalid_enum__"
                elif location == "path":
                    request["path_params"][param["name"]] = "__invalid_enum__"
                elif location == "header":
                    request["headers"][param["name"]] = "__invalid_enum__"
                else:
                    continue
                status = self._pick_status(op, [400, 422], 400)
                return self._build_case(
                    op=op,
                    title=f"{status} - Invalid enum value",
                    request=request,
                    status=status,
                    result=f"Enum disi parametre degeri '{param['name']}' icin reddedilmeli.",
                    test_type="negative",
                    priority="P1",
                )
        return None

    def _boundary_case(self, op: ApiOperation, base_request: dict) -> dict:
        request = copy.deepcopy(base_request)
        title = "Boundary value case"
        status = self._pick_status(op, [200, 201, 202, 204, 400, 422], self._SUCCESS_CODES.get(op.method, 200))
        result = "Sinir deger kontrata uygun sekilde islenmeli."

        if isinstance(request.get("body"), dict) and isinstance(op.request_body_schema, dict):
            properties = op.request_body_schema.get("properties")
            if isinstance(properties, dict):
                for field_name, field_schema in properties.items():
                    if isinstance(field_schema, dict) and any(
                        key in field_schema for key in ("minimum", "maximum", "minLength", "maxLength", "minItems", "maxItems")
                    ):
                        request["body"][field_name] = self._boundary_value_for_schema(field_schema)
                        return self._build_case(
                            op=op,
                            title=title,
                            request=request,
                            status=status,
                            result=result,
                            test_type="boundary",
                            priority="P1",
                            response_schema_check=200 <= status < 300 and bool(op.response_schemas),
                        )

        for param in op.parameters:
            schema = param.get("schema") if isinstance(param.get("schema"), dict) else {}
            if any(key in schema for key in ("minimum", "maximum", "minLength", "maxLength")):
                value = self._boundary_value_for_schema(schema)
                if param.get("in") == "query":
                    request["query_params"][param["name"]] = value
                elif param.get("in") == "path":
                    request["path_params"][param["name"]] = value
                elif param.get("in") == "header":
                    request["headers"][param["name"]] = value
                return self._build_case(
                    op=op,
                    title=title,
                    request=request,
                    status=status,
                    result=result,
                    test_type="boundary",
                    priority="P1",
                    response_schema_check=200 <= status < 300 and bool(op.response_schemas),
                )

        return self._build_case(
            op=op,
            title=title,
            request=request,
            status=status,
            result=result,
            test_type="boundary",
            priority="P2",
            response_schema_check=200 <= status < 300 and bool(op.response_schemas),
        )

    def _missing_required_query_case(self, op: ApiOperation, base_request: dict) -> Optional[dict]:
        required_query_params = self._required_parameters(op, "query")
        if not required_query_params:
            return None
        request = copy.deepcopy(base_request)
        param_name = required_query_params[0]["name"]
        request["query_params"].pop(param_name, None)
        status = self._pick_status(op, [400, 422], 400)
        return self._build_case(
            op=op,
            title=f"{status} - Missing required query param",
            request=request,
            status=status,
            result=f"Zorunlu query parametresi '{param_name}' olmadiginda hata donmeli.",
            test_type="contract",
            priority="P1",
        )

    def _invalid_query_type_case(self, op: ApiOperation, base_request: dict) -> Optional[dict]:
        query_params = self._parameters_by_location(op, "query")
        if not query_params:
            return None
        request = copy.deepcopy(base_request)
        param = query_params[0]
        request["query_params"][param["name"]] = self._invalid_value_for_schema(
            param.get("schema") if isinstance(param.get("schema"), dict) else {}
        )
        status = self._pick_status(op, [400, 422], 400)
        return self._build_case(
            op=op,
            title=f"{status} - Invalid query param type",
            request=request,
            status=status,
            result=f"Query parametresi '{param['name']}' yanlis tipte ise hata donmeli.",
            test_type="negative",
            priority="P2",
        )

    def _invalid_path_param_case(self, op: ApiOperation, base_request: dict) -> Optional[dict]:
        path_params = self._parameters_by_location(op, "path")
        if not path_params:
            return None
        request = copy.deepcopy(base_request)
        param = path_params[0]
        schema = param.get("schema") if isinstance(param.get("schema"), dict) else {}
        invalid_value = self._invalid_value_for_schema(schema)
        if invalid_value in (None, ""):
            invalid_value = "invalid-path"
        request["path_params"][param["name"]] = invalid_value
        status = self._pick_status(op, [404, 400, 422], 404)
        return self._build_case(
            op=op,
            title=f"{status} - Invalid path param",
            request=request,
            status=status,
            result=f"Path parametresi '{param['name']}' gecersiz oldugunda hata donmeli.",
            test_type="negative",
            priority="P1",
        )

    def _auth_negative_cases(self, op: ApiOperation, base_request: dict) -> List[dict]:
        if not op.security:
            return []

        status = self._pick_status(op, [401, 403], 401)
        missing_auth = copy.deepcopy(base_request)
        missing_auth["headers"].pop("Authorization", None)
        invalid_auth = copy.deepcopy(base_request)
        invalid_auth["headers"]["Authorization"] = "Bearer invalid-token"

        return [
            self._build_case(
                op=op,
                title=f"{status} - Missing auth header",
                request=missing_auth,
                status=status,
                result="Authorization header olmadiginda erisim reddedilmeli.",
                test_type="auth",
                priority="P0",
            ),
            self._build_case(
                op=op,
                title=f"{status} - Invalid auth token",
                request=invalid_auth,
                status=status,
                result="Gecersiz token kullanildiginda erisim reddedilmeli.",
                test_type="auth",
                priority="P0",
            ),
        ]

    def _error_expectation_case(self, op: ApiOperation, base_request: dict) -> dict:
        request = copy.deepcopy(base_request)
        status = self._pick_status(op, [415, 500, 400], 500)
        result = "Desteklenmeyen icerik veya beklenmeyen hata durumunda uygun hata kodu donmeli."
        if op.method != "GET":
            request["headers"]["Content-Type"] = "application/xml"
        return self._build_case(
            op=op,
            title=f"{status} - Unsupported or error expectation",
            request=request,
            status=status,
            result=result,
            test_type="error",
            priority="P2",
        )

    def _extra_fallback_cases(
        self,
        op: ApiOperation,
        base_request: dict,
        missing_count: int,
        positive_body: Any,
    ) -> List[dict]:
        cases: List[dict] = []
        status_pool = self._responses_available(op) or [self._SUCCESS_CODES.get(op.method, 200), *self._DEFAULT_ERROR_CODES]
        for index in range(missing_count):
            request = copy.deepcopy(base_request)
            status = status_pool[index % len(status_pool)]
            if op.method != "GET" and isinstance(positive_body, dict) and request.get("body") is None:
                request["body"] = copy.deepcopy(positive_body)
            cases.append(
                self._build_case(
                    op=op,
                    title=f"{status} - Additional contract baseline {index + 1}",
                    request=request,
                    status=status,
                    result="Kontrat bazli ek baseline senaryosu.",
                    test_type=_infer_test_type(status, ""),
                    priority="P2",
                    response_schema_check=200 <= status < 300 and bool(op.response_schemas),
                )
            )
        return cases
