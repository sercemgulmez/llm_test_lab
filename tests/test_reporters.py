from reporters.csv_reporter import build_comparison_summary, compute_generator_metrics


def _sample_rows():
    return [
        {
            "generator": "A",
            "operation_id": "op1",
            "http_method": "GET",
            "path": "/users",
            "title": "happy path",
            "request_body": "",
            "expected_status": 200,
            "expected_result": "ok",
            "actual_status": 200,
            "pass": True,
        },
        {
            "generator": "A",
            "operation_id": "op1",
            "http_method": "GET",
            "path": "/users",
            "title": "missing token",
            "request_body": "",
            "expected_status": 401,
            "expected_result": "unauthorized",
            "actual_status": 500,
            "pass": False,
        },
        {
            "generator": "B",
            "operation_id": "op1",
            "http_method": "GET",
            "path": "/users",
            "title": "happy path variant",
            "request_body": "",
            "expected_status": 200,
            "expected_result": "ok",
            "actual_status": 200,
            "pass": True,
        },
        {
            "generator": "B",
            "operation_id": "op2",
            "http_method": "POST",
            "path": "/users",
            "title": "invalid payload",
            "request_body": '{"name":null}',
            "expected_status": 400,
            "expected_result": "validation error",
            "actual_status": 400,
            "pass": True,
        },
    ]


def test_compute_generator_metrics_aggregates_pass_rates():
    metrics = compute_generator_metrics(_sample_rows())
    by_generator = {row["generator"]: row for row in metrics}

    assert by_generator["A"]["pass_count"] == 1
    assert by_generator["A"]["fail_count"] == 1
    assert by_generator["A"]["pass_rate"] == 0.5
    assert by_generator["B"]["pass_rate"] == 1.0


def test_build_comparison_summary_exposes_complex_and_semantic_signals():
    summary = build_comparison_summary(_sample_rows())

    assert summary["overview"]["best_generator"] == "B"
    assert len(summary["generator_rankings"]) == 2
    assert "complex_matrix" in summary
    assert summary["complex_matrix"]["labels"] == ["B", "A"]
    assert summary["complex_matrix"]["pairwise_similarity"]["A"]["B"] >= 0.0
    assert summary["generator_rankings"][0]["diversity_score"] >= 0.0
    assert summary["operation_comparison"][0]["generator_stats"]["A"]["total"] >= 1
