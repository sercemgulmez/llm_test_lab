"""Bolum 6: kosu basligi. Sonradan geri getirilemeyen baglam o an kaydedilmeli."""

import json

import pytest

import attestation
import config
import rate_limiter
import run_header
from tests.conftest import synthetic_limits


@pytest.fixture
def limiter():
    return rate_limiter.RateLimiter(synthetic_limits())


def _build(limiter, armed=True):
    return run_header.build("run_x", limiter, {"warn": 40.0, "hard_warn": 64.0, "stop": 80.0}, armed)


def test_header_carries_git_state(limiter):
    git = _build(limiter)["git"]
    assert git["commit"] and len(git["commit"]) == 40
    assert git["branch"]
    assert isinstance(git["working_tree_clean"], bool)


def test_header_carries_effective_limits_and_safety_factor(limiter):
    header = _build(limiter)
    assert header["safety_factor_default"] == 0.8
    entry = header["rate_limits"]["Groq/openai/gpt-oss-20b"]
    assert entry["safety_factor"] == 0.8
    assert entry["effective_rpm"] == int(entry["published_rpm"] * 0.8)
    assert entry["tpm_counts"] == "total"
    assert header["rate_limits"]["Gemini/gemini-2.5-flash"]["tpm_counts"] == "input_only"


def test_header_carries_attestation_with_timestamp(monkeypatch, limiter):
    monkeypatch.setenv(attestation.ENV_VAR, "gemini,groq")
    record = _build(limiter)["attestation"]
    assert record["declared_providers"] == ["Gemini", "Groq"]
    assert record["read_at"]


def test_header_carries_budget_state(limiter):
    header = _build(limiter, armed=False)
    assert header["budget"] == {
        "armed": False,
        "thresholds": {"warn": 40.0, "hard_warn": 64.0, "stop": 80.0},
    }


def test_header_carries_llm_timeout(limiter):
    assert _build(limiter)["llm_request_timeout"] == dict(config.LLM_REQUEST_TIMEOUT)


def test_header_contains_no_secrets(monkeypatch, limiter):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_" + "Z" * 48)
    monkeypatch.setenv(attestation.ENV_VAR, "groq")
    blob = json.dumps(_build(limiter), ensure_ascii=False)
    assert "gsk_" + "Z" * 48 not in blob
    for forbidden in ("api_key", "GROQ_API_KEY", "balance", "credit"):
        assert forbidden not in blob


def test_missing_git_does_not_crash(monkeypatch, limiter):
    monkeypatch.setattr(run_header, "_git", lambda *a: None)
    git = _build(limiter)["git"]
    assert git == {"commit": None, "branch": None, "working_tree_clean": None, "dirty_paths": []}


def test_dirty_tree_is_flagged(monkeypatch, caplog, limiter):
    import logging

    monkeypatch.setattr(
        run_header, "_git",
        lambda *a: "abc123" * 6 + "abcd" if a[0] == "rev-parse" and a[1] == "HEAD"
        else ("main" if "--abbrev-ref" in a else " M main.py"),
    )
    with caplog.at_level(logging.WARNING):
        run_header.log(_build(limiter))
    assert "KIRLI" in caplog.text


def test_ledger_stores_the_header_as_its_first_record(tmp_path, limiter):
    from call_ledger import CallLedger

    ledger = CallLedger(str(tmp_path), run_id="run_x")
    ledger.record_run_header(_build(limiter))
    ledger.flush()
    records = CallLedger.load(ledger.path)
    assert records[0]["record_type"] == "run_header"
    assert records[0]["run_id"] == "run_x"
    assert records[0]["git"]["commit"]
