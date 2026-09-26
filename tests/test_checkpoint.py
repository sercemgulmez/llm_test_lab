"""K7 regresyonu: checkpoint / resume mekanizmasi."""

from checkpoint import RunCheckpoint, row_identity


def _row(generator, variant, tc_id, **extra):
    row = {"generator": generator, "prompt_variant": variant, "tc_id": tc_id}
    row.update(extra)
    return row


def test_row_identity_is_not_tc_id_alone():
    """Ayni tc_id farkli generator'larda tekrar eder; kimlik uclu olmali."""
    a = _row("LLM-A", "basic", "EP1_TC1")
    b = _row("LLM-B", "basic", "EP1_TC1")
    c = _row("LLM-A", "edge_focused", "EP1_TC1")

    assert row_identity(a) != row_identity(b)
    assert row_identity(a) != row_identity(c)
    assert row_identity(a) == row_identity(dict(a))


def test_generated_rows_and_tasks_survive_reopen(tmp_path):
    ckpt = RunCheckpoint(str(tmp_path), flush_every=2)
    rows = [_row("Traditional-Template", "", f"EP1_TC{i}") for i in range(1, 4)]
    ckpt.record_generated(rows, "traditional")
    ckpt.mark_task_done("traditional", len(rows))
    ckpt.flush()

    resumed = RunCheckpoint(str(tmp_path), run_id=ckpt.run_id)

    assert resumed.resumed is True
    assert resumed.completed_tasks() == {"traditional"}
    assert len(resumed.load_generated_rows()) == 3


def test_crash_between_rows_and_task_marker(tmp_path):
    ckpt = RunCheckpoint(str(tmp_path), flush_every=1)
    run_id = ckpt.run_id

    ckpt.record_generated([_row("LLM-X", "basic", "EP1_TC1")], "OpenAIGenerator:gpt-x|basic")
    ckpt.mark_task_done("OpenAIGenerator:gpt-x|basic", 1)
    # Ikinci gorevin satirlari yazildi ama gorev isaretlenmeden cokme oldu.
    ckpt.record_generated([_row("LLM-Y", "basic", "EP2_TC1")], "OpenAIGenerator:gpt-y|basic")
    ckpt.flush()

    resumed = RunCheckpoint(str(tmp_path), run_id=run_id)

    assert resumed.completed_tasks() == {"OpenAIGenerator:gpt-x|basic"}
    assert "OpenAIGenerator:gpt-y|basic" not in resumed.completed_tasks()


def test_truncated_last_line_is_skipped(tmp_path):
    """Cokme aninda yarim kalan son satir okumayi bozmamali."""
    ckpt = RunCheckpoint(str(tmp_path), flush_every=1)
    ckpt.record_generated([_row("LLM-X", "basic", "EP1_TC1")], "t1")
    ckpt.mark_task_done("t1", 1)
    ckpt.flush()

    path = ckpt.dir / "generation.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"generator": "LLM-X", "prompt_vari')  # yarim satir

    resumed = RunCheckpoint(str(tmp_path), run_id=ckpt.run_id)
    rows = resumed.load_generated_rows()

    assert len(rows) == 1
    assert rows[0]["tc_id"] == "EP1_TC1"


def test_executed_rows_are_checkpointed_and_skipped_on_resume(tmp_path):
    ckpt = RunCheckpoint(str(tmp_path), flush_every=1)
    executed = [_row("LLM-X", "basic", f"EP1_TC{i}", actual_status=200) for i in (1, 2)]
    for row in executed:
        ckpt.record_executed(row)
    ckpt.flush()

    resumed = RunCheckpoint(str(tmp_path), run_id=ckpt.run_id)
    done = resumed.executed_identities()

    all_rows = executed + [_row("LLM-X", "basic", "EP1_TC3")]
    pending = [row for row in all_rows if row_identity(row) not in done]

    assert len(done) == 2
    assert [row["tc_id"] for row in pending] == ["EP1_TC3"]


def test_disabled_checkpoint_writes_nothing(tmp_path):
    ckpt = RunCheckpoint(str(tmp_path), flush_every=1, enabled=False)
    ckpt.record_generated([_row("LLM-X", "basic", "EP1_TC1")], "t")
    ckpt.mark_task_done("t", 1)
    ckpt.record_executed(_row("LLM-X", "basic", "EP1_TC1"))
    ckpt.flush()

    assert ckpt.load_generated_rows() == []
    assert ckpt.completed_tasks() == set()
    assert not ckpt.dir.exists()


def test_buffer_flushes_only_after_threshold(tmp_path):
    ckpt = RunCheckpoint(str(tmp_path), flush_every=3)
    path = ckpt.dir / "generation.jsonl"

    ckpt.record_generated([_row("LLM-X", "basic", "EP1_TC1")], "t")
    assert not path.exists(), "esik dolmadan diske yazilmamali"

    ckpt.record_generated([_row("LLM-X", "basic", f"EP1_TC{i}") for i in (2, 3)], "t")
    ckpt.mark_task_done("t", 3)
    assert path.is_file(), "esik dolunca diske yazilmali"
    assert len(ckpt.load_generated_rows()) == 3


def test_nested_row_structures_round_trip(tmp_path):
    """Gercek satirlar ic ice dict iceriyor (request/expected); bozulmadan donmeli."""
    ckpt = RunCheckpoint(str(tmp_path), flush_every=1)
    row = _row(
        "LLM-X", "basic", "EP1_TC1",
        request={"path_params": {}, "query_params": {"a": 1}, "body": {"name": "sercem"}},
        expected={"status": 200, "allowed_statuses": [200], "assertions": [{"type": "status_code"}]},
    )
    ckpt.record_generated([row], "t")
    ckpt.mark_task_done("t", 1)
    ckpt.flush()

    loaded = RunCheckpoint(str(tmp_path), run_id=ckpt.run_id).load_generated_rows()

    assert loaded[0]["request"]["body"] == {"name": "sercem"}
    assert loaded[0]["expected"]["allowed_statuses"] == [200]


def test_orphaned_rows_from_incomplete_task_are_not_reloaded(tmp_path):
    """record_generated ile mark_task_done arasindaki cokme duplicate uretmemeli.

    Yetim satirlar resume'da yuklenirse, gorev yeniden kostugunda ayni tc_id'ler
    ikinci kez eklenir (K2 ile ayni siniftan hata).
    """
    ckpt = RunCheckpoint(str(tmp_path), flush_every=1)
    run_id = ckpt.run_id

    # Tamamlanmis gorev
    ckpt.record_generated([_row("LLM-X", "basic", "EP1_TC1")], "task-done")
    ckpt.mark_task_done("task-done", 1)

    # Satirlari yazdi, gorev isaretlenmeden coktu
    ckpt.record_generated(
        [_row("LLM-Y", "basic", "EP2_TC1"), _row("LLM-Y", "basic", "EP2_TC2")],
        "task-crashed",
    )
    ckpt.flush()

    resumed = RunCheckpoint(str(tmp_path), run_id=run_id)
    rows = resumed.load_generated_rows()

    assert [row["tc_id"] for row in rows] == ["EP1_TC1"], "yetim satirlar yuklenmemeli"
    assert "task-crashed" not in resumed.completed_tasks(), "yarim gorev yeniden kosmali"
    # Dahili alan disari sizmamali (CSV kolonlarina karismasin)
    assert all("_checkpoint_task" not in row for row in rows)


def test_reloaded_rows_plus_rerun_produce_no_duplicate_identities(tmp_path):
    """Yetim koruma + yeniden kosum sonrasi kimlikler tekil kalmali."""
    ckpt = RunCheckpoint(str(tmp_path), flush_every=1)
    run_id = ckpt.run_id
    ckpt.record_generated([_row("LLM-X", "basic", "EP1_TC1")], "task-done")
    ckpt.mark_task_done("task-done", 1)
    ckpt.record_generated([_row("LLM-Y", "basic", "EP2_TC1")], "task-crashed")
    ckpt.flush()

    resumed = RunCheckpoint(str(tmp_path), run_id=run_id)
    all_rows = resumed.load_generated_rows()
    # Yarim kalan gorev yeniden kosuyor ve ayni satirlari uretiyor
    rerun_rows = [_row("LLM-Y", "basic", "EP2_TC1")]
    resumed.record_generated(rerun_rows, "task-crashed")
    resumed.mark_task_done("task-crashed", 1)
    all_rows.extend(rerun_rows)

    identities = [row_identity(row) for row in all_rows]
    assert len(identities) == len(set(identities)), f"duplicate: {identities}"
    assert len(all_rows) == 2
