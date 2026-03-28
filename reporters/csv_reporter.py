"""CSV formatında test raporu ve metrik üretici."""

import cmath
import csv
import json
import os
import re
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
    "url", "actual_status", "pass", "tokens_used",
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
    "total_tokens", "avg_tokens_per_tc",
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
        evaluated_total = num_pass + num_fail

        exp_dist: Dict[str, int] = defaultdict(int)
        act_dist: Dict[str, int] = defaultdict(int)
        for r in gen_rows:
            es = r.get("expected_status")
            as_ = r.get("actual_status")
            if es not in ("", None):
                exp_dist[str(es)] += 1
            if as_ not in ("", None):
                act_dist[str(as_)] += 1

        total_tokens = sum(r.get("tokens_used") or 0 for r in gen_rows)
        metrics.append({
            "generator": gen,
            "total_tests": total,
            "pass_count": num_pass,
            "fail_count": num_fail,
            "pass_rate": round(num_pass / evaluated_total, 3) if evaluated_total else "",
            "total_tokens": total_tokens or "",
            "avg_tokens_per_tc": round(total_tokens / total, 1) if total and total_tokens else "",
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


def _tokenize_testcase(row: Dict) -> set[str]:
    """Testcase'i kaba semantik imza için token setine çevirir."""
    parts = [
        str(row.get("operation_id", "")),
        str(row.get("http_method", "")),
        str(row.get("path", "")),
        str(row.get("title", "")),
        str(row.get("request_body", "")),
        str(row.get("expected_status", "")),
        str(row.get("expected_result", "")),
    ]
    text = " ".join(parts).lower()
    return set(re.findall(r"[a-z0-9_:/.-]+", text))


def _jaccard_similarity(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _compute_semantic_profiles(rows: List[Dict]) -> Dict:
    """Generator bazlı çeşitlilik ve generatorlar arası benzerlik profili üretir."""
    by_generator: Dict[str, List[Dict]] = defaultdict(list)
    for row in rows:
        by_generator[row["generator"]].append(row)

    all_operation_ids = {
        row.get("operation_id", "") for row in rows if row.get("operation_id", "")
    }
    all_expected_statuses = {
        str(row.get("expected_status", "")) for row in rows if row.get("expected_status") not in ("", None)
    }

    token_rows = {
        generator: [_tokenize_testcase(row) for row in gen_rows]
        for generator, gen_rows in by_generator.items()
    }

    diversity: Dict[str, Dict] = {}
    for generator, gen_rows in by_generator.items():
        tokens = token_rows[generator]
        total = len(tokens)
        if total <= 1:
            mean_similarity = 0.0
        else:
            pair_sum = 0.0
            pair_count = 0
            for i in range(total):
                for j in range(i + 1, total):
                    pair_sum += _jaccard_similarity(tokens[i], tokens[j])
                    pair_count += 1
            mean_similarity = pair_sum / pair_count if pair_count else 0.0

        operation_counts: Dict[str, int] = defaultdict(int)
        status_counts: Dict[str, int] = defaultdict(int)
        for row in gen_rows:
            operation_counts[row.get("operation_id", "")] += 1
            status_counts[str(row.get("expected_status", ""))] += 1

        op_coverage = len([k for k in operation_counts if k]) / max(1, len(all_operation_ids))
        status_variety = len([k for k in status_counts if k]) / max(1, len(all_expected_statuses))
        diversity_score = max(0.0, min(1.0, 0.55 * (1.0 - mean_similarity) + 0.30 * op_coverage + 0.15 * status_variety))
        diversity[generator] = {
            "intra_similarity": round(mean_similarity, 6),
            "diversity_score": round(diversity_score, 6),
            "operation_coverage": round(op_coverage, 6),
            "status_variety": round(status_variety, 6),
        }

    pairwise_similarity: Dict[str, Dict[str, float]] = defaultdict(dict)
    generators = sorted(by_generator.keys())
    for left in generators:
        for right in generators:
            if left == right:
                pairwise_similarity[left][right] = 1.0
                continue
            if right in pairwise_similarity and left in pairwise_similarity[right]:
                pairwise_similarity[left][right] = pairwise_similarity[right][left]
                continue

            left_tokens = token_rows[left]
            right_tokens = token_rows[right]
            if not left_tokens or not right_tokens:
                similarity = 0.0
            else:
                # Her testcase için karşı tarafta en yakın senaryoyu bulup ortalıyoruz.
                left_to_right = sum(
                    max(_jaccard_similarity(tokens, other) for other in right_tokens)
                    for tokens in left_tokens
                ) / len(left_tokens)
                right_to_left = sum(
                    max(_jaccard_similarity(tokens, other) for other in left_tokens)
                    for tokens in right_tokens
                ) / len(right_tokens)
                similarity = (left_to_right + right_to_left) / 2

            pairwise_similarity[left][right] = round(similarity, 6)
            pairwise_similarity[right][left] = round(similarity, 6)

    return {
        "diversity": diversity,
        "pairwise_similarity": pairwise_similarity,
    }


def _build_complex_pairwise_matrix(metrics_sorted: List[Dict], semantic_profiles: Dict) -> Dict:
    """Generator'lar için Hermitian kompleks karşılaştırma matrisi kurar."""
    size = len(metrics_sorted)
    if size == 0:
        return {"labels": [], "matrix": [], "spectral_scores": []}

    max_tests = max((m["total_tests"] for m in metrics_sorted), default=1) or 1
    matrix: List[List[complex]] = [[0j for _ in range(size)] for _ in range(size)]

    for i, left in enumerate(metrics_sorted):
        matrix[i][i] = 1 + 0j
        for j in range(i + 1, size):
            right = metrics_sorted[j]
            left_rate = left["pass_rate"] if left["pass_rate"] != "" else 0.0
            right_rate = right["pass_rate"] if right["pass_rate"] != "" else 0.0
            left_diversity = semantic_profiles["diversity"].get(left["generator"], {}).get("diversity_score", 0.0)
            right_diversity = semantic_profiles["diversity"].get(right["generator"], {}).get("diversity_score", 0.0)
            inter_similarity = semantic_profiles["pairwise_similarity"].get(left["generator"], {}).get(right["generator"], 0.0)

            # Genlik: başarı, çeşitlilik ve düşük kopyacılığı birlikte temsil eder.
            rate_delta = left_rate - right_rate
            coverage_delta = (left["total_tests"] - right["total_tests"]) / max_tests
            diversity_delta = left_diversity - right_diversity
            novelty_bonus = 1.0 - inter_similarity
            magnitude = max(
                0.25,
                1.0
                + 0.65 * rate_delta
                + 0.20 * coverage_delta
                + 0.50 * diversity_delta
                + 0.25 * novelty_bonus
            )

            # Faz: fail/load ve semantik yakınlığı encode eder.
            fail_balance = (left["fail_count"] - right["fail_count"]) / max_tests
            phase = 0.45 * fail_balance + 0.35 * (inter_similarity - 0.5) + 0.20 * diversity_delta
            value = cmath.rect(magnitude, phase)
            matrix[i][j] = value
            matrix[j][i] = value.conjugate()

    vector = [1 + 0j for _ in range(size)]
    for _ in range(24):
        next_vector: List[complex] = []
        for row in matrix:
            total = 0j
            for idx, cell in enumerate(row):
                total += cell * vector[idx]
            next_vector.append(total)

        norm = sum(abs(v) for v in next_vector) or 1.0
        vector = [v / norm for v in next_vector]

    spectral_scores = []
    for idx, item in enumerate(metrics_sorted):
        score = abs(vector[idx])
        phase = cmath.phase(vector[idx])
        spectral_scores.append({
            "generator": item["generator"],
            "score": round(score, 6),
            "phase": round(phase, 6),
            "diversity_score": semantic_profiles["diversity"].get(item["generator"], {}).get("diversity_score", 0.0),
            "intra_similarity": semantic_profiles["diversity"].get(item["generator"], {}).get("intra_similarity", 0.0),
        })

    score_map = {row["generator"]: row["score"] for row in spectral_scores}
    phase_map = {row["generator"]: row["phase"] for row in spectral_scores}
    encoded_matrix = []
    for row in matrix:
        encoded_matrix.append([
            {
                "real": round(cell.real, 6),
                "imag": round(cell.imag, 6),
                "magnitude": round(abs(cell), 6),
                "phase": round(cmath.phase(cell), 6),
            }
            for cell in row
        ])

    return {
        "labels": [m["generator"] for m in metrics_sorted],
        "matrix": encoded_matrix,
        "score_map": score_map,
        "phase_map": phase_map,
        "pairwise_similarity": semantic_profiles["pairwise_similarity"],
        "diversity": semantic_profiles["diversity"],
        "spectral_scores": sorted(spectral_scores, key=lambda row: row["score"], reverse=True),
    }


def build_comparison_summary(rows: List[Dict]) -> Dict:
    """UI için generator ve operasyon bazlı karşılaştırma özeti üretir."""
    metrics = compute_generator_metrics(rows)
    semantic_profiles = _compute_semantic_profiles(rows)
    base_sorted = sorted(
        metrics,
        key=lambda m: (
            m["pass_rate"] if m["pass_rate"] != "" else -1,
            m["pass_count"],
            -m["fail_count"],
            m["generator"],
        ),
        reverse=True,
    )
    matrix_summary = _build_complex_pairwise_matrix(base_sorted, semantic_profiles)
    metrics_sorted = sorted(
        base_sorted,
        key=lambda m: (
            matrix_summary["score_map"].get(m["generator"], 0.0),
            semantic_profiles["diversity"].get(m["generator"], {}).get("diversity_score", 0.0),
            m["pass_rate"] if m["pass_rate"] != "" else -1,
            m["pass_count"],
            -m["fail_count"],
            m["generator"],
        ),
        reverse=True,
    )

    total_tests = len(rows)
    total_pass = sum(1 for r in rows if r.get("pass") is True)
    total_fail = sum(1 for r in rows if r.get("pass") is False)
    generators = [m["generator"] for m in metrics_sorted]
    operation_groups: Dict[tuple, List[Dict]] = defaultdict(list)
    for row in rows:
        key = (row.get("operation_id", ""), row.get("http_method", ""), row.get("path", ""))
        operation_groups[key].append(row)

    operation_rows: List[Dict] = []
    for (operation_id, method, path), op_rows in sorted(operation_groups.items()):
        per_generator = {}
        for generator in generators:
            gen_rows = [r for r in op_rows if r.get("generator") == generator]
            total = len(gen_rows)
            passed = sum(1 for r in gen_rows if r.get("pass") is True)
            failed = sum(1 for r in gen_rows if r.get("pass") is False)
            per_generator[generator] = {
                "total": total,
                "pass": passed,
                "fail": failed,
                "pass_rate": round(passed / total, 3) if total else None,
            }

        operation_rows.append({
            "operation_id": operation_id,
            "method": method,
            "path": path,
            "generator_stats": per_generator,
        })

    best_generator = metrics_sorted[0]["generator"] if metrics_sorted else ""
    enriched_rankings = []
    for item in metrics_sorted:
        enriched = dict(item)
        enriched["matrix_score"] = matrix_summary["score_map"].get(item["generator"], 0.0)
        enriched["matrix_phase"] = matrix_summary["phase_map"].get(item["generator"], 0.0)
        enriched["diversity_score"] = semantic_profiles["diversity"].get(item["generator"], {}).get("diversity_score", 0.0)
        enriched["intra_similarity"] = semantic_profiles["diversity"].get(item["generator"], {}).get("intra_similarity", 0.0)
        enriched["operation_coverage"] = semantic_profiles["diversity"].get(item["generator"], {}).get("operation_coverage", 0.0)
        enriched_rankings.append(enriched)

    return {
        "overview": {
            "total_tests": total_tests,
            "total_pass": total_pass,
            "total_fail": total_fail,
            "generator_count": len(metrics_sorted),
            "operation_count": len(operation_rows),
            "best_generator": best_generator,
        },
        "generator_rankings": enriched_rankings,
        "operation_comparison": operation_rows,
        "complex_matrix": matrix_summary,
    }


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
        f"{'Generator':<{col_gen}}  {'Total':>6}  {'PASS':>6}  {'FAIL':>6}  {'Rate':>6}  {'Tokens':>8}  {'Tok/TC':>6}"
    )
    sep = "-" * len(header)

    print(f"\n{'=' * len(header)}")
    print("GENERATOR ÖZET TABLOSU")
    print(sep)
    print(header)
    print(sep)
    for m in metrics:
        rate = f"{m['pass_rate']:.1%}" if m["pass_rate"] != "" else "N/A"
        tok_total = f"{m['total_tokens']:,}" if m.get("total_tokens") not in ("", None) else "-"
        tok_avg = f"{m['avg_tokens_per_tc']}" if m.get("avg_tokens_per_tc") not in ("", None) else "-"
        print(
            f"{m['generator']:<{col_gen}}  "
            f"{m['total_tests']:>6}  "
            f"{m['pass_count']:>6}  "
            f"{m['fail_count']:>6}  "
            f"{rate:>6}  "
            f"{tok_total:>8}  "
            f"{tok_avg:>6}"
        )
    print(sep)
    rate_all = f"{pass_all / total_all:.1%}" if total_all else "N/A"
    print(
        f"{'TOPLAM':<{col_gen}}  "
        f"{total_all:>6}  "
        f"{pass_all:>6}  "
        f"{fail_all:>6}  "
        f"{rate_all:>6}  "
        f"{'':>8}  "
        f"{'':>6}"
    )
    print("=" * len(header))
