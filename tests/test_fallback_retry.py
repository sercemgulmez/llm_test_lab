"""B3b regresyonu: altyapi kaynakli fallback iceren gorevlerin yeniden kosulmasi."""

import pytest

import main
from checkpoint import RunCheckpoint


def _row(generator, tc_id, variant="basic"):
    return {"generator": generator, "prompt_variant": variant, "tc_id": tc_id}


# ── Checkpoint semantigi ──────────────────────────────────────────────────

def test_task_with_fallback_is_still_marked_completed(tmp_path):
    """Tamamlanma semantigi DEGISMEMELI: fallback'li gorev de tamamlanmis sayilir."""
    ckpt = RunCheckpoint(str(tmp_path), flush_every=1)
    ckpt.record_generated([_row("LLM-X", "EP1_TC1")], "OpenAIGenerator:gpt-4.1|basic")
    ckpt.mark_task_done(
        "OpenAIGenerator:gpt-4.1|basic", 15, fallback_cases=15, failure_origin="altyapi"
    )
    ckpt.flush()

    assert "OpenAIGenerator:gpt-4.1|basic" in ckpt.completed_tasks()
    record = ckpt.task_records()["OpenAIGenerator:gpt-4.1|basic"]
    assert record["fallback_cases"] == 15
    assert record["failure_origin"] == "altyapi"


def test_revoked_task_drops_out_of_completed_tasks(tmp_path):
    ckpt = RunCheckpoint(str(tmp_path), flush_every=1)
    ckpt.mark_task_done("T1", 5, fallback_cases=5, failure_origin="altyapi")
    ckpt.mark_task_done("T2", 5)
    ckpt.flush()
    assert ckpt.completed_tasks() == {"T1", "T2"}

    ckpt.revoke_task("T1", reason="altyapi")
    ckpt.flush()

    resumed = RunCheckpoint(str(tmp_path), run_id=ckpt.run_id)
    assert resumed.completed_tasks() == {"T2"}


def test_revoked_task_rows_are_dropped_so_no_duplicates(tmp_path):
    """Iptal edilen gorevin satirlari yetim kalir ve resume'da yuklenmez."""
    ckpt = RunCheckpoint(str(tmp_path), flush_every=1)
    ckpt.record_generated([_row("LLM-X", "EP1_TC1"), _row("LLM-X", "EP1_TC2")], "T1")
    ckpt.mark_task_done("T1", 2, fallback_cases=2, failure_origin="altyapi")
    ckpt.record_generated([_row("LLM-Y", "EP1_TC1")], "T2")
    ckpt.mark_task_done("T2", 1)
    ckpt.flush()
    assert len(ckpt.load_generated_rows()) == 3

    ckpt.revoke_task("T1")
    ckpt.flush()

    resumed = RunCheckpoint(str(tmp_path), run_id=ckpt.run_id)
    rows = resumed.load_generated_rows()
    assert len(rows) == 1, "iptal edilen gorevin satirlari TAMAMEN dusmeli"
    assert rows[0]["generator"] == "LLM-Y"


def test_rerun_replaces_all_rows_without_duplicate_tc_ids(tmp_path):
    """Yeniden kosu = TAM DEGISTIRME; kismi birlestirme olmamali."""
    ckpt = RunCheckpoint(str(tmp_path), flush_every=1)
    old_rows = [_row("LLM-X", f"EP1_TC{i}") for i in (1, 2, 3)]
    ckpt.record_generated(old_rows, "T1")
    ckpt.mark_task_done("T1", 3, fallback_cases=3, failure_origin="altyapi")
    ckpt.flush()

    ckpt.revoke_task("T1")
    new_rows = [_row("LLM-X", f"EP1_TC{i}") for i in (1, 2, 3)]   # AYNI tc_id'ler
    ckpt.record_generated(new_rows, "T1", epoch=1)
    ckpt.mark_task_done("T1", 3, fallback_cases=0, failure_origin=None, retry_count=1)
    ckpt.flush()

    rows = RunCheckpoint(str(tmp_path), run_id=ckpt.run_id).load_generated_rows()
    identities = [(r["generator"], r["prompt_variant"], r["tc_id"]) for r in rows]
    assert len(rows) == 3, "eski satirlar degistirilmeli, eklenmemeli"
    assert len(identities) == len(set(identities)), f"duplicate tc_id: {identities}"


# ── Aday secimi ───────────────────────────────────────────────────────────

def test_only_infrastructure_fallback_is_retried_not_content():
    records = {
        "OpenAIGenerator:gpt-4.1|basic": {"failure_origin": "altyapi", "retry_count": 0},
        "GroqGenerator:m|basic": {"failure_origin": "icerik", "retry_count": 0},
        "GeminiGenerator:m|basic": {"failure_origin": "karma", "retry_count": 0},
        "ClaudeGenerator:m|basic": {"failure_origin": None, "retry_count": 0},
    }
    keys = [key for key, _ in main._infra_fallback_candidates(records)]

    assert "OpenAIGenerator:gpt-4.1|basic" in keys
    assert "GeminiGenerator:m|basic" in keys, "karma da altyapi icerir"
    assert "GroqGenerator:m|basic" not in keys, "icerik kaynakli fallback'e DOKUNULMAMALI"
    assert "ClaudeGenerator:m|basic" not in keys


def test_second_rerun_is_refused():
    records = {"OpenAIGenerator:m|basic": {"failure_origin": "altyapi", "retry_count": 1}}
    assert main._infra_fallback_candidates(records) == [], "gorev basina en fazla 1 yeniden kosu"


@pytest.mark.parametrize("task_key,expected_paid", [
    ("OpenAIGenerator:gpt-4.1|basic", True),
    ("ClaudeGenerator:claude-haiku-4-5|basic", True),
    ("GroqGenerator:openai/gpt-oss-20b|basic", False),
    ("GeminiGenerator:gemini-2.5-flash|basic", False),
])
def test_paid_provider_detection(task_key, expected_paid):
    assert main._task_is_paid(task_key) is expected_paid


def test_task_provider_parsing():
    assert main._task_provider("OpenAIGenerator:gpt-4.1|basic") == "OpenAI"
    assert main._task_provider("GroqGenerator:m|basic") == "Groq"


def test_paid_retry_requires_explicit_confirmation(tmp_path, monkeypatch):
    """Ucretli generator onay bayragi olmadan yeniden kosulmamali; kosu durmali."""
    ckpt = RunCheckpoint(str(tmp_path / "out"), flush_every=1)
    ckpt.record_generated([_row("LLM-OpenAI-gpt-4.1", "EP1_TC1")], "OpenAIGenerator:gpt-4.1|basic")
    ckpt.mark_task_done(
        "OpenAIGenerator:gpt-4.1|basic", 1, fallback_cases=1, failure_origin="altyapi"
    )
    ckpt.flush()

    monkeypatch.setattr(main, "load_dotenv", lambda *a, **kw: False)
    monkeypatch.setattr("sys.argv", [
        "main.py", "--endpoints", "GET /get", "--base-url", "https://httpbin.org",
        "--generators", "traditional", "--num-cases", "1", "--no-run",
        "--resume", ckpt.run_id, "--retry-infra-fallback",
        "--output-dir", str(tmp_path / "out"),
    ])

    with pytest.raises(SystemExit) as exc_info:
        main.main()

    assert exc_info.value.code == 2, "onaysiz ucretli yeniden kosu kosuyu DURDURMALI"


def test_retry_flag_requires_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "load_dotenv", lambda *a, **kw: False)
    monkeypatch.setattr("sys.argv", [
        "main.py", "--endpoints", "GET /get", "--base-url", "https://httpbin.org",
        "--generators", "traditional", "--num-cases", "1", "--no-run",
        "--retry-infra-fallback", "--output-dir", str(tmp_path),
    ])

    with pytest.raises(SystemExit) as exc_info:
        main.main()

    assert exc_info.value.code == 1
