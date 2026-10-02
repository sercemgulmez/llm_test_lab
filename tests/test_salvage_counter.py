"""Bulgu 1b: gecersiz JSON yuzunden atilan case sayisi SESSIZ kalmasin.

Onceden bozuk bir JSON dizisi sessizce 0 case'e dusuyordu; defterde "model
hicbir sey dondurmedi" gibi gorunuyordu. Artik kurtarilan/atilan sayisi cagri
basina deftere yaziliyor ve raporlaniyor.
"""

import threading

import pytest

from call_ledger import CallLedger, summarize_by_generator
from generators.base import (
    extract_json_array,
    last_salvage_stats,
    reset_salvage_stats,
)
from models import TokenUsage

GOOD = '{"tc_id": "TC%d", "title": "ok"}'
BROKEN = '{"tc_id": "BAD", "v": "a".repeat(2000)}'


def _array(*chunks):
    return "[\n" + ",\n".join(chunks) + "\n]"


def test_counter_reports_recovered_and_discarded():
    reset_salvage_stats()
    extract_json_array(_array(GOOD % 1, GOOD % 2, BROKEN))
    assert last_salvage_stats() == {"recovered": 2, "discarded": 1}


def test_reset_clears_previous_call():
    extract_json_array(_array(GOOD % 1, BROKEN))
    assert last_salvage_stats()["discarded"] == 1
    reset_salvage_stats()
    assert last_salvage_stats() == {"recovered": 0, "discarded": 0}


def test_counter_is_per_thread():
    """Paralel uretimde komsu is parcaciginin sayaci okunmamali.

    Sayac modul duzeyinde global olsaydi, bir generator'in atilan case'i baska
    bir generator'in defter kaydina yazilabilirdi.
    """
    seen: dict = {}
    barrier = threading.Barrier(2)

    def worker(name, text, expected_discarded):
        reset_salvage_stats()
        extract_json_array(text)
        barrier.wait()          # iki is parcacigi da ayristirmayi bitirsin
        seen[name] = last_salvage_stats()["discarded"]
        assert seen[name] == expected_discarded

    threads = [
        threading.Thread(target=worker, args=("bozuk", _array(GOOD % 1, BROKEN, BROKEN), 2)),
        threading.Thread(target=worker, args=("saglam", _array(GOOD % 1, GOOD % 2), 0)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert seen == {"bozuk": 2, "saglam": 0}


def test_ledger_records_the_counter(tmp_path):
    ledger = CallLedger(str(tmp_path), run_id="r")
    ledger.record_call(
        generator="LLM-Groq-m", variant="basic", operation_id="EP1", method="GET",
        path="/a", attempt=0, call_type="ana",
        usage=TokenUsage(input_tokens=10, output_tokens=5, split_available=True),
        accepted_cases=14, rejected_cases=0, raw_response="[]",
        call_meta={"model_requested": "m"},
        limiter_meta={"salvaged_cases": 14, "discarded_cases": 1},
    )
    ledger.flush()
    record = CallLedger.load(ledger.path)[0]
    assert record["salvaged_cases"] == 14
    assert record["discarded_cases"] == 1


def test_summary_computes_discarded_share(tmp_path):
    records = [
        {"generator": "G", "call_type": "ana", "accepted_cases": 14,
         "rejected_cases": 0, "discarded_cases": 1, "salvaged_cases": 14,
         "split_available": True},
        {"generator": "G", "call_type": "ana", "accepted_cases": 6,
         "rejected_cases": 0, "discarded_cases": 0, "salvaged_cases": 6,
         "split_available": True},
    ]
    item = summarize_by_generator(records)[0]
    assert item["discarded_cases"] == 1
    assert item["salvaged_cases"] == 20
    assert item["discarded_share"] == pytest.approx(1 / 21, abs=1e-4)


def test_no_discard_means_no_share_noise():
    records = [{"generator": "G", "call_type": "ana", "accepted_cases": 15,
                "rejected_cases": 0, "split_available": True}]
    item = summarize_by_generator(records)[0]
    assert item["discarded_cases"] == 0
    assert item["discarded_share"] == 0.0


def test_validator_warns_about_discarded_cases(tmp_path):
    """Dogrulayici atilan case'leri gorunur kilmali."""
    import sys
    sys.path.insert(0, "scripts")
    from scripts.validate_run_output import Findings, check_call_ledger

    ledger = CallLedger(str(tmp_path), run_id="r")
    ledger.record_call(
        generator="LLM-Groq-m", variant="basic", operation_id="EP1", method="GET",
        path="/a", attempt=0, call_type="ana",
        usage=TokenUsage(input_tokens=10, output_tokens=5, split_available=True),
        accepted_cases=10, rejected_cases=0, raw_response="[]",
        call_meta={"model_requested": "m"},
        limiter_meta={"salvaged_cases": 10, "discarded_cases": 5},
    )
    ledger.flush()
    findings = Findings()
    check_call_ledger(str(tmp_path), None, findings)
    warnings = " ".join(findings.warnings)
    assert "GECERSIZ JSON" in warnings
    assert "5 case" in warnings
