from parsers.curl_parser import parse_curl, parse_curl_collection


def test_parse_curl_extracts_request_metadata():
    curl_text = (
        "curl 'https://api.example.com/v1/users?id=42' "
        "-X POST "
        "-H 'Authorization: Bearer abc' "
        "-H 'X-Trace-Id: trace-1' "
        "-H 'Cookie: session=xyz; theme=dark' "
        "--data-raw '{\"name\":\"Ada\"}'"
    )

    op, base_url, headers, cookies = parse_curl(curl_text)

    assert base_url == "https://api.example.com"
    assert op.method == "POST"
    assert op.path == "/v1/users?id=42"
    assert op.request_body_examples == [{"name": "Ada"}]
    assert op.parameters == [{"name": "id", "in": "query", "example": "42"}]
    assert op.security == [{"bearer": []}]
    assert op.servers == ["https://api.example.com"]
    assert headers["Authorization"] == "Bearer abc"
    assert headers["X-Trace-Id"] == "trace-1"
    assert cookies == {"session": "xyz", "theme": "dark"}
    assert "X-Trace-Id: trace-1" in op.description


def test_parse_curl_collection_assigns_incremental_operation_ids():
    collection = """
curl 'https://api.example.com/users' -X GET
curl 'https://api.example.com/users' -X POST --data-raw '{"name":"Ada"}'
""".strip()

    parsed = parse_curl_collection(collection)

    assert len(parsed) == 2
    assert parsed[0][0].op_id == "CURL_OP1"
    assert parsed[1][0].op_id == "CURL_OP2"
    assert parsed[0][0].summary.startswith("[OP1]")
    assert parsed[1][0].method == "POST"
