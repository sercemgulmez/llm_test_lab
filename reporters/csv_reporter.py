"""CSV formatında test raporu ve metrik üretici."""

import csv
import json
import os
from collections import defaultdict
from datetime import datetime
from typing import Dict, List

from models import ApiOperation


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)


# ── Operasyon listesi ──────────────────────────────────────────────────────────

def save_operations_csv(operations: List[ApiOperation], output_dir: str) -> str:
    if not operations:
        return ""
    path = os.path.join(output_dir, "operations.csv")
    _ensure_dir(path)
    fieldnames = ["op_id", "method", "path", "summary", "description"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for op in operations:
            w.writerow({
                "op_id": op.op_id,
                "method": op.method,
                "path": op.path,
                "summary": op.summary,
                "description": op.description,
            })
    print(f"Operasyon listesi kaydedildi: {path}")
    return path


# ── Test sonuçları ─────────────────────────────────────────────────────────────

RESULT_FIELDNAMES = [
    "generator", "operation_id", "http_method", "path",
    "tc_id", "title", "request_body", "expected_status", "expected_result",
    "url", "actual_status", "pass",
]


def save_results_csv(rows: List[Dict], output_dir: str) -> str:
    if not rows:
        print("Kaydedilecek test sonucu yok.")
        return ""
    path = os.path.join(output_dir, f"executed_testcases_{_timestamp()}.csv")
    _ensure_dir(path)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_FIELDNAMES, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Test sonuçları kaydedildi: {path}")
    return path


# ── Generator metrikleri ───────────────────────────────────────────────────────

METRICS_FIELDNAMES = [
    "generator", "total_tests", "pass_count", "fail_count", "pass_rate",
    "expected_status_distribution", "actual_status_distribution",
]


def compute_generator_metrics(rows: List[Dict]) -> List[Dict]:
    """Her generator için toplu metrikler hesaplar."""
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        groups[r["generator"]].append(r)

    metrics: List[Dict] = []
    for gen, gen_rows in groups.items():
        total = len(gen_rows)
        num_pass = sum(1 for r in gen_rows if r.get("pass") is True)
        num_fail = sum(1 for r in gen_rows if r.get("pass") is False)

        exp_dist: Dict[str, int] = defaultdict(int)
        act_dist: Dict[str, int] = defaultdict(int)
        for r in gen_rows:
            es = r.get("expected_status")
            as_ = r.get("actual_status")
            if es not in ("", None):
                exp_dist[str(es)] += 1
            if as_ not in ("", None):
                act_dist[str(as_)] += 1

        metrics.append({
            "generator": gen,
            "total_tests": total,
            "pass_count": num_pass,
            "fail_count": num_fail,
            "pass_rate": round(num_pass / total, 3) if total else "",
            "expected_status_distribution": json.dumps(dict(exp_dist), ensure_ascii=False),
            "actual_status_distribution": json.dumps(dict(act_dist), ensure_ascii=False),
        })
    return metrics


def save_generator_metrics_csv(rows: List[Dict], output_dir: str) -> str:
    if not rows:
        print("Generator metrikleri için satır yok.")
        return ""
    path = os.path.join(output_dir, f"generator_metrics_{_timestamp()}.csv")
    _ensure_dir(path)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=METRICS_FIELDNAMES)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Generator metrikleri kaydedildi: {path}")
    return path


# ── Konsol özet tablosu ────────────────────────────────────────────────────────

def print_summary_table(rows: List[Dict]) -> None:
    """Konsola generator bazlı özet tablo basar."""
    if not rows:
        print("Hiç test sonucu yok.")
        return

    metrics = compute_generator_metrics(rows)
    total_all = len(rows)
    pass_all = sum(1 for r in rows if r.get("pass") is True)
    fail_all = sum(1 for r in rows if r.get("pass") is False)

    col_gen = max(len(m["generator"]) for m in metrics)
    col_gen = max(col_gen, 9)  # "Generator" başlığı

    header = (
        f"{'Generator':<{col_gen}}  {'Total':>6}  {'PASS':>6}  {'FAIL':>6}  {'Rate':>6}"
    )
    sep = "-" * len(header)

    print(f"\n{'=' * len(header)}")
    print("GENERATOR ÖZET TABLOSU")
    print(sep)
    print(header)
    print(sep)
    for m in metrics:
        rate = f"{m['pass_rate']:.1%}" if m["pass_rate"] != "" else "N/A"
        print(
            f"{m['generator']:<{col_gen}}  "
            f"{m['total_tests']:>6}  "
            f"{m['pass_count']:>6}  "
            f"{m['fail_count']:>6}  "
            f"{rate:>6}"
        )
    print(sep)
    rate_all = f"{pass_all / total_all:.1%}" if total_all else "N/A"
    print(
        f"{'TOPLAM':<{col_gen}}  "
        f"{total_all:>6}  "
        f"{pass_all:>6}  "
        f"{fail_all:>6}  "
        f"{rate_all:>6}"
    )
    print("=" * len(header))
