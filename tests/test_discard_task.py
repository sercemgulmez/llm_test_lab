"""--discard-task: operator 'bu veri kullanilamaz' dediginde satirlar GERCEKTEN gitsin.

Kritik nokta: generation.jsonl append-only. Yalnizca revoke etmek, o kosu icinde
satirlari yetim yapar (iyi) ama gorev ayni nesille yeniden tamamlaninca SONRAKI
--resume hem eski hem yeni satirlari yukler (kotu). Bu yuzden epoch da artar.
"""

import pytest

from checkpoint import RunCheckpoint


def _row(tc_id, generator="LLM-X", variant="basic"):
    return {"generator": generator, "prompt_variant": variant, "tc_id": tc_id}


def test_revoke_without_epoch_bump_would_duplicate(tmp_path):
    """Hatanin kendisini belgeleyen test: epoch artmazsa satirlar ikiye katlanir."""
    ckpt = RunCheckpoint(str(tmp_path), flush_every=1)
    ckpt.record_generated([_row("TC1"), _row("TC2")], "G:m|basic", epoch=0)
    ckpt.mark_task_done("G:m|basic", 2, retry_count=0)
    ckpt.revoke_task("G:m|basic", reason="test")
    # AYNI nesille yeniden tamamla (yanlis kullanim)
    ckpt.record_generated([_row("TC1"), _row("TC2")], "G:m|basic", epoch=0)
    ckpt.mark_task_done("G:m|basic", 2, retry_count=0)
    ckpt.flush()

    resumed = RunCheckpoint(str(tmp_path), run_id=ckpt.run_id, flush_every=1)
    assert len(resumed.load_generated_rows()) == 4, (
        "bu testin amaci: epoch artmazsa 4 satir yuklenir (2 olmali)"
    )


def test_epoch_bump_replaces_instead_of_appending(tmp_path):
    """Dogru kullanim: revoke + epoch artimi -> TAM DEGISTIRME."""
    ckpt = RunCheckpoint(str(tmp_path), flush_every=1)
    ckpt.record_generated([_row("TC1"), _row("TC2")], "G:m|basic", epoch=0)
    ckpt.mark_task_done("G:m|basic", 2, retry_count=0)

    ckpt.revoke_task("G:m|basic", reason="operator karariyla atildi")
    ckpt.record_generated([_row("TC9")], "G:m|basic", epoch=1)
    ckpt.mark_task_done("G:m|basic", 1, retry_count=1)
    ckpt.flush()

    resumed = RunCheckpoint(str(tmp_path), run_id=ckpt.run_id, flush_every=1)
    rows = resumed.load_generated_rows()
    assert [r["tc_id"] for r in rows] == ["TC9"], f"eski nesil atilmali: {rows}"


def test_other_tasks_are_untouched(tmp_path):
    """Atilan gorev DISINDAKI gorevler (ornegin kota bekleyen Gemini) korunur."""
    ckpt = RunCheckpoint(str(tmp_path), flush_every=1)
    ckpt.record_generated([_row("G1")], "Groq:m|basic", epoch=0)
    ckpt.mark_task_done("Groq:m|basic", 1, retry_count=0)
    ckpt.record_generated([_row("M1")], "Gemini:m|basic", epoch=0)
    ckpt.mark_task_done("Gemini:m|basic", 1, failure_origin="altyapi",
                        next_available_at="2026-10-03T17:00:46+03:00", retry_count=0)

    ckpt.revoke_task("Groq:m|basic", reason="operator")
    ckpt.record_generated([_row("G2")], "Groq:m|basic", epoch=1)
    ckpt.mark_task_done("Groq:m|basic", 1, retry_count=1)
    ckpt.flush()

    resumed = RunCheckpoint(str(tmp_path), run_id=ckpt.run_id, flush_every=1)
    rows = sorted(r["tc_id"] for r in resumed.load_generated_rows())
    assert rows == ["G2", "M1"], f"Gemini satiri korunmali: {rows}"
    gemini = resumed.task_records()["Gemini:m|basic"]
    assert gemini["next_available_at"] == "2026-10-03T17:00:46+03:00"
    assert gemini["failure_origin"] == "altyapi"


def test_discard_requires_resume(monkeypatch, tmp_path, caplog):
    import logging
    import main

    monkeypatch.setattr(main, "load_dotenv", lambda *a, **k: False)
    monkeypatch.setenv("ATTEST_FREE_TIER", "gemini,groq")
    monkeypatch.setattr("sys.argv", [
        "main.py", "--endpoints", "GET /get", "--base-url", "https://httpbin.org",
        "--generators", "traditional", "--num-cases", "2", "--no-run",
        "--discard-task", "Groq:m|basic", "--output-dir", str(tmp_path),
    ])
    with caplog.at_level(logging.ERROR), pytest.raises(SystemExit) as excinfo:
        main.main()
    assert excinfo.value.code == 1
    assert "yalnizca --resume ile" in caplog.text


def test_discard_unknown_task_is_refused(monkeypatch, tmp_path, caplog):
    """Yanlis yazilmis gorev adi sessizce yutulmamali."""
    import logging
    import main

    ckpt = RunCheckpoint(str(tmp_path), flush_every=1)
    ckpt.mark_task_done("traditional", 5, retry_count=0)
    ckpt.flush()

    monkeypatch.setattr(main, "load_dotenv", lambda *a, **k: False)
    monkeypatch.setenv("ATTEST_FREE_TIER", "gemini,groq")
    monkeypatch.setattr("sys.argv", [
        "main.py", "--endpoints", "GET /get", "--base-url", "https://httpbin.org",
        "--generators", "traditional", "--num-cases", "2", "--no-run",
        "--resume", ckpt.run_id, "--discard-task", "YOK:boyle|gorev",
        "--output-dir", str(tmp_path),
    ])
    with caplog.at_level(logging.ERROR), pytest.raises(SystemExit) as excinfo:
        main.main()
    assert excinfo.value.code == 1
    assert "boyle tamamlanmis bir gorev yok" in caplog.text


def test_traditional_also_honours_the_epoch(monkeypatch, tmp_path):
    """Traditional dali da nesil damgasi yazmali.

    Olculen hata: Traditional dali record_generated/mark_task_done'a epoch ve
    retry_count GECIRMIYORDU. --discard-task traditional ile kullanildiginda yeni
    satirlar da epoch 0'a yaziliyor ve eski 150 satirla birlikte yukleniyordu
    (generation.jsonl append-only) -> 300 satir.
    """
    import main

    monkeypatch.setattr(main, "load_dotenv", lambda *a, **k: False)
    monkeypatch.setenv("ATTEST_FREE_TIER", "gemini,groq")
    argv = [
        "main.py", "--endpoints", "GET /get,POST /post",
        "--base-url", "https://httpbin.org", "--generators", "traditional",
        "--num-cases", "3", "--no-run", "--output-dir", str(tmp_path),
    ]
    monkeypatch.setattr("sys.argv", list(argv))
    main.main()

    ckpt = RunCheckpoint(str(tmp_path), enabled=True)
    run_id = sorted(p.name for p in (tmp_path / ".checkpoints").iterdir())[-1]
    first = RunCheckpoint(str(tmp_path), run_id=run_id, enabled=True)
    before = len(first.load_generated_rows())
    assert before > 0

    # Ayni run_id'yi surdurerek traditional'i AT ve yeniden kos
    monkeypatch.setattr("sys.argv", argv + ["--resume", run_id,
                                            "--discard-task", "traditional"])
    main.main()

    after = RunCheckpoint(str(tmp_path), run_id=run_id, enabled=True)
    rows = after.load_generated_rows()
    assert len(rows) == before, (
        f"yeniden kosu DEGISTIRMELI, eklemememeli: once {before}, sonra {len(rows)}"
    )
    identities = [(r["generator"], r["prompt_variant"], r["tc_id"]) for r in rows]
    assert len(identities) == len(set(identities)), "mukerrer kimlik olmamali"
    assert after.task_records()["traditional"]["retry_count"] == 1
