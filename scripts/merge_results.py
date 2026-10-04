"""run_N klasorlerini birlestirir (kaynaklari SALT OKUR, yalnizca <root>/final/ altina yazar).

  1. outputs/run_[0-9]*/executed_testcases_*.csv (en yenisi) -> final/all_results.csv
     (basa "run" kolonu eklenir, utf-8-sig)
  2. .calls/<run_id>/calls.jsonl -> final/cost_by_generator.csv (run x generator)
  3. .checkpoints/<run_id>/tasks.jsonl -> run basina fallback_cases toplami (ekrana)

Kullanim:
    python3 scripts/merge_results.py
    python3 scripts/merge_results.py --outputs-root /tmp/deneme   # test icin
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import report_views as rv

COST_FIELDS = [
    "run", "generator", "provider", "calls", "ana", "repair", "retry", "failed",
    "input_tokens", "output_tokens", "reasoning_tokens", "billable_output_tokens",
    "cost_usd_billed", "cost_usd_list_equivalent", "unpriced_calls", "cost_usd_guard_estimate",
]


def _run_number(run_dir: Path) -> str:
    return rv.RUN_DIR_RE.match(run_dir.name).group(1)


def merge_results(runs: list[rv.RunData], numbers: dict[str, str], final_dir: Path) -> None:
    print("── 1) all_results.csv ──")
    header: list[str] = []
    merged: list[dict] = []
    for run in runs:
        if run.csv_path is None:
            print(f"  {run.name}: YOK (executed_testcases CSV yok)")
            continue
        if not header:
            header = list(run.fieldnames)
        elif run.fieldnames != header:
            extra = [name for name in run.fieldnames if name not in header]
            print(f"  UYARI: {run.name} basliklari ilk run'dan farkli; fazla kolonlar sona eklendi: {extra}")
            header.extend(extra)
        for row in run.rows:
            merged.append({"run": numbers[run.name], **row})
        print(f"  {run.name}: {run.csv_path.name} ({len(run.rows)} satir)")
        for label, count in sorted(Counter(r.get("generator", "") for r in run.rows).items()):
            print(f"      {label}: {count}")
    if not merged:
        print("  YOK: birlestirilecek satir yok, all_results.csv yazilmadi.")
        return
    path = final_dir / rv.ALL_RESULTS_NAME
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run"] + header, restval="", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(merged)
    print(f"  Yazildi: {path} ({len(merged)} satir)")


def merge_costs(runs: list[rv.RunData], numbers: dict[str, str], final_dir: Path) -> None:
    print("── 2) cost_by_generator.csv ──")
    lines: list[dict] = []
    per_provider: dict[str, float] = defaultdict(float)
    for run in runs:
        if run.ledger is None:
            print(f"  {run.name}: YOK (calls.jsonl yok)")
            continue
        stats = rv.ledger_by_generator(run.ledger)
        for label, s in sorted(stats.items()):
            if not s["calls"]:
                continue  # yalnizca fallback kaydi olan generator: API cagrisi yok
            provider = rv.provider_of(label)
            lines.append({"run": numbers[run.name], "generator": label, "provider": provider,
                          **{key: s[key] for key in COST_FIELDS if key in s}})
            per_provider[provider] += s["cost_usd_billed"]
        print(f"  {run.name}: {sum(s['calls'] for s in stats.values())} cagri, "
              f"faturalanan ${sum(s['cost_usd_billed'] for s in stats.values()):.4f}")
    if not lines:
        print("  YOK: defter kaydi yok, cost_by_generator.csv yazilmadi.")
        return
    path = final_dir / "cost_by_generator.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COST_FIELDS, restval="")
        writer.writeheader()
        writer.writerows(lines)
    print(f"  Yazildi: {path} ({len(lines)} satir)")
    for provider, total in sorted(per_provider.items()):
        print(f"      {provider}: ${total:.4f}")


def report_fallback(runs: list[rv.RunData]) -> None:
    print("── 3) fallback_cases (tasks.jsonl) ──")
    for run in runs:
        if run.tasks is None:
            print(f"  {run.name}: YOK (tasks.jsonl yok)")
            continue
        per_gen = rv.tasks_by_generator(run.tasks)
        total = sum(item["fallback_cases"] for item in per_gen.values())
        nonzero = {label: item["fallback_cases"] for label, item in per_gen.items() if item["fallback_cases"]}
        print(f"  {run.name}: toplam {total}" + (f" | {nonzero}" if nonzero else ""))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="run_N sonuclarini outputs/final/ altinda birlestirir.")
    parser.add_argument("--outputs-root", default=str(rv.OUTPUTS_ROOT),
                        help="run_N klasorlerinin bulundugu kok (varsayilan: outputs)")
    args = parser.parse_args(argv)

    root = Path(args.outputs_root)
    run_dirs = rv.numbered_run_dirs(root)
    if not run_dirs:
        print(f"YOK: {root} altinda executed_testcases CSV iceren run_N klasoru yok.")
        return 1
    final_dir = root / rv.FINAL_DIR_NAME
    final_dir.mkdir(parents=True, exist_ok=True)
    print(f"Kok: {root} | run'lar: {', '.join(d.name for d in run_dirs)} | hedef: {final_dir}")

    runs = [rv.load_run(d) for d in run_dirs]
    numbers = {d.name: _run_number(d) for d in run_dirs}
    merge_results(runs, numbers, final_dir)
    merge_costs(runs, numbers, final_dir)
    report_fallback(runs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
