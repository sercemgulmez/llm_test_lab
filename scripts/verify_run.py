"""Tez deneyi run klasoru dogrulayicisi (SALT OKUMA).

validate_run_output.py'nin KAPSAMADIGI kontroller burada:
  1. 450 satir / tam 9 etiket / generator basina 50 / generator x operasyon 10 /
     LLM'lerde prompt_variant 25-25
  2. "_" ile baslayan CSV kolonu
  3. fallback_cases toplami (tasks.jsonl) + defter ve execution capraz kontrolu
  4. Cagri defteri: her LLM icin cagri, ucretli saglayicida maliyet, failed sayisi
  5. run_info: commit ve calisma agaci
  6. 9 generator icin pass ozeti

Kullanim:
    python3 scripts/verify_run.py outputs/run_1

Cikis kodu: 0 = GECERLI, 1 = GECERSIZ, 2 = kullanim hatasi.
Yalnizca stdlib (+ report_views) kullanir; venv disinda da calisir.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import report_views as rv

EXPECTED_TASKS_PER_LLM = len(rv.LLM_VARIANTS)


class Report:
    def __init__(self) -> None:
        self.results: list[tuple[str, str]] = []

    def add(self, status: str, name: str, evidence: str) -> None:
        self.results.append((status, name))
        print(f"[{status}] {name} — {evidence}")

    def info(self, text: str = "") -> None:
        print(f"        {text}" if text else "")

    def blocking(self) -> list[str]:
        return [f"{status}:{name}" for status, name in self.results if status in ("FAIL", "YOK")]

    def warnings(self) -> int:
        return sum(1 for status, _ in self.results if status == "UYARI")


def _fmt_rate(rate) -> str:
    return "yok" if rate is None else f"{rate:.1%}"


def check_counts(run: rv.RunData, report: Report, per_gen: int, per_op: int) -> None:
    expected = rv.expected_labels()
    total_expected = len(expected) * per_gen
    rows = run.rows
    report.add("PASS" if len(rows) == total_expected else "FAIL",
               "1a Toplam satir", f"beklenen {total_expected}, bulunan {len(rows)}")

    grouped = rv.rows_by_generator(rows)
    present = set(grouped)
    missing = [label for label in expected if label not in present]
    unexpected = sorted(present - set(expected))
    ok = not missing and not unexpected
    report.add("PASS" if ok else "FAIL", "1b Generator etiketleri",
               f"beklenen {len(expected)}, bulunan {len(present)}; eksik {len(missing)}, beklenmeyen {len(unexpected)}")
    for label in missing:
        report.info(f"eksik: {label}")
    for label in unexpected:
        report.info(f"beklenmeyen: {label} ({len(grouped[label])} satir)")

    bad_gen = {label: len(g) for label, g in grouped.items() if len(g) != per_gen}
    report.add("PASS" if not bad_gen else "FAIL", "1c Generator basina satir",
               f"{len(grouped) - len(bad_gen)}/{len(grouped)} mevcut generator {per_gen} satir")
    for label, count in sorted(bad_gen.items()):
        report.info(f"{label}: {count}")

    ops = list((run.run_info or {}).get("operation_ids") or []) or sorted(
        {str(r.get("operation_id", "")) for r in rows}
    )
    bad_cells = []
    for label, gen_rows in sorted(grouped.items()):
        per = Counter(str(r.get("operation_id", "")) for r in gen_rows)
        for op in sorted(set(ops) | set(per)):
            if per.get(op, 0) != per_op:
                bad_cells.append(f"{label} x {op}: {per.get(op, 0)}")
    cells = len(grouped) * len(set(ops))
    report.add("PASS" if not bad_cells else "FAIL", "1d Generator x operasyon",
               f"{cells - len(bad_cells)}/{cells} hucre {per_op} satir ({len(set(ops))} operasyon)")
    for cell in bad_cells[:20]:
        report.info(cell)

    targets = rv.variant_targets(per_gen)
    bad_variants = []
    llm_labels = [label for label in grouped if rv.is_llm(label)]
    for label in sorted(llm_labels):
        dist = Counter(str(r.get("prompt_variant", "")) for r in grouped[label])
        if dict(dist) != targets:
            bad_variants.append(f"{label}: {dict(dist)}")
    report.add("PASS" if not bad_variants and llm_labels else ("YOK" if not llm_labels else "FAIL"),
               "1e LLM prompt_variant dagilimi",
               f"{len(llm_labels) - len(bad_variants)}/{len(llm_labels)} LLM generator {targets}")
    for line in bad_variants:
        report.info(line)


def check_columns(run: rv.RunData, report: Report) -> None:
    internal = [name for name in run.fieldnames if name.startswith("_")]
    report.add("PASS" if not internal else "FAIL", "2 '_' ile baslayan kolon",
               f"{len(internal)} adet" + (f": {', '.join(internal)}" if internal else ""))


def check_fallback(run: rv.RunData, report: Report) -> None:
    if run.tasks is None:
        report.add("YOK", "3a fallback_cases (tasks.jsonl)", "tasks.jsonl bulunamadi")
    else:
        per_gen = rv.tasks_by_generator(run.tasks)
        total = sum(item["fallback_cases"] for item in per_gen.values())
        report.add("PASS" if total == 0 else "FAIL", "3a fallback_cases (tasks.jsonl)", f"toplam {total}")
        for label, item in sorted(per_gen.items()):
            report.info(f"{label}: {item['fallback_cases']}")

        expected_tasks = 1 + EXPECTED_TASKS_PER_LLM * (len(rv.expected_labels()) - 1)
        origins = [o for item in per_gen.values() for o in item["failure_origins"]]
        pending = sum(item["pending"] for item in per_gen.values())
        ok = len(run.tasks) == expected_tasks and not origins and not pending
        report.add("PASS" if ok else "FAIL", "3b Gorev kayitlari",
                   f"beklenen {expected_tasks}, bulunan {len(run.tasks)} (dosyada {run.task_records} satir); "
                   f"failure_origin dolu {len(origins)}, kota bekleyen {pending}")
        for origin in origins:
            report.info(origin)

    ledger_fb = sum(s["fallback_cases"] for s in rv.ledger_by_generator(run.ledger).values()) \
        if run.ledger is not None else None
    exec_fb = rv.execution_fallback_by_generator(run.execution)
    exec_total = sum(exec_fb.values()) if exec_fb is not None else None
    if ledger_fb is None and exec_total is None:
        report.add("YOK", "3c Fallback capraz kontrol", "defter ve execution.jsonl yok")
    else:
        values = [v for v in (ledger_fb, exec_total) if v is not None]
        report.add("PASS" if all(v == 0 for v in values) else "FAIL", "3c Fallback capraz kontrol",
                   f"defter fallback case {ledger_fb if ledger_fb is not None else 'yok'}, "
                   f"execution generation_metadata.fallback {exec_total if exec_total is not None else 'yok'}")


def check_ledger(run: rv.RunData, report: Report) -> None:
    if run.ledger is None:
        report.add("YOK", "4 Cagri defteri", ".calls/<run_id>/calls.jsonl bulunamadi")
        return
    stats = rv.ledger_by_generator(run.ledger)
    llm_expected = [label for label in rv.expected_labels() if rv.is_llm(label)]
    no_calls = [label for label in llm_expected if stats.get(label, {}).get("calls", 0) == 0]
    report.add("PASS" if not no_calls else "FAIL", "4a Her LLM icin cagri kaydi",
               f"{len(llm_expected) - len(no_calls)}/{len(llm_expected)} LLM generator'da >=1 cagri")
    for label in no_calls:
        report.info(f"cagri yok: {label}")

    paid = [label for label in llm_expected if rv.provider_of(label) in rv.PAID_PROVIDERS]
    unverified = [label for label in paid if rv.cost_status(label, stats.get(label)) != "ok"]
    report.add("PASS" if not unverified else "FAIL", "4b Ucretli saglayici maliyeti",
               f"{len(paid) - len(unverified)}/{len(paid)} ucretli generator'da cost_usd_billed > 0"
               + ("; MALIYET DOGRULANMADI" if unverified else ""))
    for label in unverified:
        item = stats.get(label)
        detail = "defter kaydi yok" if not item else (
            f"billed {item['cost_usd_billed']}, fiyatlanan {item['priced_calls']} cagri")
        report.info(f"MALIYET DOGRULANMADI: {label} ({detail})")

    failed = sum(item["failed"] for item in stats.values())
    report.add("PASS" if failed == 0 else "UYARI", "4c failed=true cagri", f"{failed} kayit")
    unpriced = sum(item["unpriced_calls"] for item in stats.values())
    guard = sum(item["cost_usd_guard_estimate"] for item in stats.values())
    report.add("PASS" if unpriced == 0 else "UYARI", "4d Fiyatlanamayan cagri",
               f"{unpriced} cagri, ust tahmin ${guard:.4f}")

    billed = sum(item["cost_usd_billed"] for item in stats.values())
    thresholds = ((run.ledger_header or {}).get("budget") or {}).get("thresholds") or {}
    stop = thresholds.get("stop")
    spend = billed + guard
    if stop is None:
        report.add("YOK", "4e Butce esikleri", f"defterde run_header/budget yok; harcama ${billed:.4f}")
    else:
        status = "FAIL" if spend >= stop else ("UYARI" if spend >= thresholds.get("warn", stop) else "PASS")
        report.add(status, "4e Butce esikleri",
                   f"faturalanan ${billed:.4f} + ust tahmin ${guard:.4f}; esikler "
                   f"{thresholds.get('warn')}/{thresholds.get('hard_warn')}/{stop}")


def check_run_info(run: rv.RunData, report: Report) -> None:
    if run.run_info is None:
        report.add("YOK", "5 run_info", "run_info_cli_*.json bulunamadi")
        return
    state = rv.git_state(run.run_info)
    report.add("PASS" if state["clean"] is True else "FAIL", "5a Calisma agaci",
               f"{rv.clean_label(state['clean'])}; commit {state['commit'] or 'yok'}; dal {state['branch'] or 'yok'}")
    for path in state["dirty_paths"]:
        report.info(f"kirli: {path}")
    multi = run.run_info_count > 1 or run.csv_count > 1
    report.add("UYARI" if multi else "PASS", "5b Tek kosu izi",
               f"{run.run_info_count} run_info, {run.csv_count} executed_testcases CSV")


def check_execution(run: rv.RunData, report: Report) -> None:
    unexecuted = sum(1 for r in run.rows if str(r.get("actual_status", "")).strip() == "")
    report.add("PASS" if unexecuted == 0 else "FAIL", "6a Istek atilmamis satir",
               f"actual_status bos {unexecuted} satir")
    truncated = rv.truncated_json_rows(run)
    if truncated is None:
        report.add("YOK", "6b Kesilmis govde + JSON assertion", "execution.jsonl yok")
    else:
        report.add("PASS" if truncated == 0 else "UYARI", "6b Kesilmis govde + JSON assertion",
                   f"{truncated} satir")


def print_tables(run: rv.RunData) -> None:
    summary = rv.generator_summary(run)
    breakdown = rv.failure_breakdown(run)
    width = max([len(item["generator"]) for item in summary] + [9])
    print()
    print("PASS OZETI (pass orani = pass / (pass+fail))")
    print(f"  {'Generator':<{width}}  {'Satir':>5} {'Pass':>5} {'Fail':>5} {'Bos':>4} {'Oran':>7}  "
          f"{'Tekrar':>7}  {'FAIL: status/json_path/diger':<30}")
    for item in summary:
        rep = "yok" if item["repeated_rows"] is None else f"{item['repeated_rows']}/{item['rows']}"
        br = breakdown.get(item["generator"])
        br_text = f"{br['status']}/{br['json_path']}/{br['other']} ({br['source']})" if br else "-"
        print(f"  {item['generator']:<{width}}  {item['rows']:>5} {item['passed']:>5} {item['failed']:>5} "
              f"{item['empty']:>4} {_fmt_rate(item['rate']):>7}  {rep:>7}  {br_text:<30}")
    totals = rv.pass_counts(run.rows)
    print(f"  {'TOPLAM':<{width}}  {totals['rows']:>5} {totals['passed']:>5} {totals['failed']:>5} "
          f"{totals['empty']:>4} {_fmt_rate(totals['rate']):>7}")

    print()
    print("CAGRI DEFTERI (maliyetin tek kaynagi)")
    if run.ledger is None:
        print("  YOK")
        return
    stats = rv.ledger_by_generator(run.ledger)
    print(f"  {'Generator':<{width}}  {'Cagri':>5} {'ana':>4} {'rep':>4} {'ret':>4} {'fail':>4}  "
          f"{'Girdi':>9} {'Cikti':>9}  {'Billed $':>10}  Maliyet")
    for label in sorted(stats, key=lambda lbl: (rv.expected_labels().index(lbl)
                                               if lbl in rv.expected_labels() else 99, lbl)):
        s = stats[label]
        print(f"  {label:<{width}}  {s['calls']:>5} {s['ana']:>4} {s['repair']:>4} {s['retry']:>4} "
              f"{s['failed']:>4}  {s['input_tokens']:>9} {s['output_tokens']:>9}  "
              f"{s['cost_usd_billed']:>10.4f}  {rv.cost_status(label, s)}")
    tot = {key: sum(s[key] for s in stats.values())
           for key in ("calls", "ana", "repair", "retry", "failed", "input_tokens", "output_tokens", "cost_usd_billed")}
    print(f"  {'TOPLAM':<{width}}  {tot['calls']:>5} {tot['ana']:>4} {tot['repair']:>4} {tot['retry']:>4} "
          f"{tot['failed']:>4}  {tot['input_tokens']:>9} {tot['output_tokens']:>9}  {tot['cost_usd_billed']:>10.4f}")
    print(f"  NOT: {rv.TOKENS_NOTE}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run klasoru dogrulamasi (salt okuma).")
    parser.add_argument("run_dir", help="orn. outputs/run_1")
    parser.add_argument("--expected-per-generator", type=int, default=50)
    parser.add_argument("--expected-per-op", type=int, default=10)
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        print(f"[HATA] klasor yok: {run_dir}")
        return 2

    print("=" * 78)
    print(f"RUN DOGRULAMA: {run_dir}")
    print("=" * 78)
    run = rv.load_run(run_dir)
    report = Report()
    if run.csv_path is None:
        report.add("YOK", "CSV", "executed_testcases_*.csv bulunamadi (kosu bitmemis olabilir)")
    else:
        print(f"CSV: {run.csv_path.name} ({len(run.rows)} satir) | run_id: {run.run_id or 'yok'}")
        bad = {k: v for k, v in run.bad_lines.items() if v}
        if bad:
            print(f"Bozuk JSONL satirlari atlandi: {bad}")
        print()
        check_counts(run, report, args.expected_per_generator, args.expected_per_op)
        check_columns(run, report)
    check_fallback(run, report)
    check_ledger(run, report)
    check_run_info(run, report)
    if run.csv_path is not None:
        check_execution(run, report)
        print_tables(run)
    print(f"  NOT: {rv.PASS_RATE_NOTE}")

    print()
    print("Ek denetim (venv ile): .venv/bin/python scripts/validate_run_output.py "
          f"--output-dir {run_dir} --expected-rows {len(rv.expected_labels()) * args.expected_per_generator} "
          f"--expected-generators {len(rv.expected_labels())} "
          f"--expected-per-generator {args.expected_per_generator} --executed")
    blocking = report.blocking()
    print("=" * 78)
    if blocking:
        print(f"GENEL SONUÇ: GEÇERSİZ ({len(blocking)} engelleyici: {', '.join(blocking)}; "
              f"{report.warnings()} UYARI)")
        return 1
    print(f"GENEL SONUÇ: GEÇERLİ ({report.warnings()} UYARI)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
