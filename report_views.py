"""Kosu ciktilarinin SALT OKUNUR veri katmani.

verify_run.py, merge_results.py ve /reports sayfasi ayni okuyuculari kullanir;
boylece uc yer ayni sayilari uretir. Yalnizca stdlib + reporters.csv_reporter
+ config kullanir (Flask ve SDK import etmez): venv disindaki python3 ile de
calisir.

Hicbir fonksiyon outputs/ disindan okumaz ve hicbir sey yazmaz. .env okunmaz.

Kaynaklar (bir kosu klasoru icinde):
  executed_testcases_*.csv              satirlar (en yenisi esas alinir)
  run_info_cli_*.json                   commit / calisma agaci (run_header)
  .checkpoints/<run_id>/tasks.jsonl     gorev basina rows / fallback_cases
  .checkpoints/<run_id>/execution.jsonl assertion sonuclari, generation_metadata
  .calls/<run_id>/calls.jsonl           cagri defteri: token ve maliyetin TEK kaynagi
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import config
from reporters.csv_reporter import compute_repetition_stats

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
FINAL_DIR_NAME = "final"
ALL_RESULTS_NAME = "all_results.csv"

CSV_GLOB = "executed_testcases_*.csv"
RUN_INFO_GLOB = "run_info_cli_*.json"
RUN_DIR_RE = re.compile(r"^run_(\d+)$")
TRUNCATION_MARKER = "...(truncated)"
STATUS_PATH = "/status/{code}"

TRADITIONAL_LABEL = "Traditional-Template"
PROVIDER_BY_CLASS = {
    "OpenAIGenerator": "OpenAI",
    "GeminiGenerator": "Gemini",
    "ClaudeGenerator": "Claude",
    "GroqGenerator": "Groq",
}
PAID_PROVIDERS = frozenset(config.PAID_PROVIDERS)
LLM_VARIANTS = tuple(config.PROMPT_VARIANTS)
LLM_CALL_TYPES = ("ana", "repair", "retry")

PASS_RATE_NOTE = (
    "Pass oranı generator'lar arasında doğrudan karşılaştırılamaz: Traditional yalnızca "
    "status_code assertion'ı üretir, LLM'ler içerik assertion'ı da yazar."
)
TOKENS_NOTE = (
    "tokens_used satır bazında paylaştırılmış değerdir; maliyet için tek geçerli kaynak "
    "çağrı defteridir."
)


# ── Etiketler ────────────────────────────────────────────────────────────────

def expected_labels() -> list[str]:
    """run_all.sh'in 9 generator'unun CSV etiketleri (generator dosyalarindaki bicim)."""
    labels = [TRADITIONAL_LABEL]
    for provider, models in (
        ("OpenAI", config.OPENAI_MODELS),
        ("Gemini", config.GEMINI_MODELS),
        ("Claude", config.CLAUDE_MODELS),
        ("Groq", config.GROQ_MODELS),
    ):
        labels.extend(f"LLM-{provider}-{model}" for model in models)
    return labels


def provider_of(label: str) -> str:
    if label == TRADITIONAL_LABEL:
        return "Traditional"
    parts = str(label).split("-", 2)
    if len(parts) == 3 and parts[0] == "LLM":
        return parts[1]
    return ""


def is_llm(label: str) -> bool:
    return provider_of(label) not in ("", "Traditional")


def task_to_label(task: str) -> str:
    """'GroqGenerator:openai/gpt-oss-20b|basic' -> 'LLM-Groq-openai/gpt-oss-20b'."""
    if task == "traditional":
        return TRADITIONAL_LABEL
    head = str(task).split("|", 1)[0]
    cls, _, model = head.partition(":")
    provider = PROVIDER_BY_CLASS.get(cls, cls.replace("Generator", ""))
    return f"LLM-{provider}-{model}"


def variant_targets(per_generator: int) -> dict[str, int]:
    """main._split_total ile ayni dagitim: kalan ilk variant'lara eklenir."""
    base, remainder = divmod(per_generator, len(LLM_VARIANTS))
    return {v: base + (1 if i < remainder else 0) for i, v in enumerate(LLM_VARIANTS)}


# ── Klasor secimi (path traversal'a kapali) ─────────────────────────────────

def _root(root: Path | None) -> Path:
    return Path(root) if root is not None else OUTPUTS_ROOT


def list_run_dirs(root: Path | None = None) -> list[Path]:
    """outputs/* altinda, gizli olmayan ve executed_testcases_*.csv iceren klasorler."""
    base = _root(root)
    if not base.is_dir():
        return []
    return sorted(
        child for child in base.iterdir()
        if child.is_dir() and not child.name.startswith(".") and any(child.glob(CSV_GLOB))
    )


def resolve_run_dir(name: Any, root: Path | None = None) -> Path | None:
    """Ad, mevcut kosu klasorlerinden biriyle BIREBIR eslesmiyorsa None."""
    if not isinstance(name, str) or not name:
        return None
    base = _root(root)
    for candidate in list_run_dirs(base):
        if candidate.name == name and candidate.resolve().parent == base.resolve():
            return candidate
    return None


def download_path(run_name: Any, kind: str, root: Path | None = None) -> Path | None:
    """Indirmeye izinli yalnizca iki sabit kalip var."""
    base = _root(root)
    if kind == "executed":
        run_dir = resolve_run_dir(run_name, base)
        return latest_csv(run_dir) if run_dir is not None else None
    if kind == "all_results":
        path = base / FINAL_DIR_NAME / ALL_RESULTS_NAME
        return path if path.is_file() else None
    return None


# ── Dosya okuyucular ────────────────────────────────────────────────────────

def csv_files(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob(CSV_GLOB))


def latest_csv(run_dir: Path) -> Path | None:
    files = csv_files(run_dir)
    return files[-1] if files else None


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(line.replace("\x00", "") for line in handle)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    return fieldnames, rows


def read_jsonl(path: Path | None) -> tuple[list[dict] | None, int]:
    """(kayitlar, bozuk satir sayisi). Dosya yoksa (None, 0)."""
    if path is None or not path.is_file():
        return None, 0
    records: list[dict] = []
    bad = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            if isinstance(value, dict):
                records.append(value)
            else:
                bad += 1
    return records, bad


def run_info_files(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob(RUN_INFO_GLOB))


def latest_run_info(run_dir: Path) -> dict | None:
    files = run_info_files(run_dir)
    if not files:
        return None
    try:
        return json.loads(files[-1].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _state_dir(run_dir: Path, kind: str, run_id: str | None) -> Path | None:
    base = run_dir / kind
    if run_id and (base / run_id).is_dir():
        return base / run_id
    if not base.is_dir():
        return None
    candidates = sorted(child for child in base.iterdir() if child.is_dir())
    return candidates[-1] if candidates else None


def git_state(run_info: dict | None) -> dict[str, Any]:
    header = (run_info or {}).get("run_header") or {}
    git = header.get("git") or {}
    return {
        "commit": git.get("commit"),
        "branch": git.get("branch"),
        "clean": git.get("working_tree_clean"),
        "dirty_paths": git.get("dirty_paths") or [],
    }


def clean_label(clean: Any) -> str:
    if clean is True:
        return "TEMIZ"
    if clean is False:
        return "KIRLI"
    return "yok"


def csv_timestamp(csv_path: Path | None) -> str:
    if csv_path is None:
        return "yok"
    match = re.search(r"(\d{8})_(\d{6})", csv_path.name)
    if not match:
        return "yok"
    d, t = match.groups()
    return f"{d[:4]}-{d[4:6]}-{d[6:]} {t[:2]}:{t[2:4]}:{t[4:]}"


# ── Kosu verisi ─────────────────────────────────────────────────────────────

@dataclass
class RunData:
    name: str
    path: Path
    csv_path: Path | None
    csv_count: int
    fieldnames: list[str]
    rows: list[dict[str, str]]
    run_info: dict | None
    run_info_count: int
    run_id: str | None
    tasks: dict[str, dict] | None          # gorev -> SON kayit
    task_records: int
    ledger: list[dict] | None              # run_header haric
    ledger_header: dict | None
    execution: list[dict] | None
    bad_lines: dict[str, int] = field(default_factory=dict)


def load_run(run_dir: Path) -> RunData:
    csv_path = latest_csv(run_dir)
    fieldnames, rows = read_csv(csv_path) if csv_path else ([], [])
    info = latest_run_info(run_dir)
    run_id = ((info or {}).get("run_header") or {}).get("run_id")

    checkpoint_dir = _state_dir(run_dir, ".checkpoints", run_id)
    ledger_dir = _state_dir(run_dir, ".calls", run_id)
    if run_id is None and checkpoint_dir is not None:
        run_id = checkpoint_dir.name

    task_list, bad_tasks = read_jsonl(checkpoint_dir / "tasks.jsonl" if checkpoint_dir else None)
    tasks = None
    if task_list is not None:
        tasks = {}
        for record in task_list:
            tasks[str(record.get("task", ""))] = record

    ledger_all, bad_ledger = read_jsonl(ledger_dir / "calls.jsonl" if ledger_dir else None)
    ledger, header = None, None
    if ledger_all is not None:
        ledger = [r for r in ledger_all if r.get("record_type") != "run_header"]
        header = next((r for r in ledger_all if r.get("record_type") == "run_header"), None)

    execution, bad_exec = read_jsonl(checkpoint_dir / "execution.jsonl" if checkpoint_dir else None)
    if execution is not None:
        # resume'da ayni satir tekrar yazilabilir: kimlik basina SON kayit.
        dedup: dict[tuple[str, str, str], dict] = {}
        for record in execution:
            key = (str(record.get("generator", "")), str(record.get("prompt_variant", "")),
                   str(record.get("tc_id", "")))
            dedup[key] = record
        execution = list(dedup.values())

    return RunData(
        name=run_dir.name,
        path=run_dir,
        csv_path=csv_path,
        csv_count=len(csv_files(run_dir)),
        fieldnames=fieldnames,
        rows=rows,
        run_info=info,
        run_info_count=len(run_info_files(run_dir)),
        run_id=run_id,
        tasks=tasks,
        task_records=len(task_list or []),
        ledger=ledger,
        ledger_header=header,
        execution=execution,
        bad_lines={"tasks": bad_tasks, "ledger": bad_ledger, "execution": bad_exec},
    )


# ── Ozetler ─────────────────────────────────────────────────────────────────

def _is_true(value: Any) -> bool:
    return value is True or str(value) == "True"


def _is_false(value: Any) -> bool:
    return value is False or str(value) == "False"


def pass_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    passed = sum(1 for r in rows if _is_true(r.get("pass")))
    failed = sum(1 for r in rows if _is_false(r.get("pass")))
    evaluated = passed + failed
    return {
        "rows": len(rows),
        "passed": passed,
        "failed": failed,
        "empty": len(rows) - evaluated,
        "rate": (passed / evaluated) if evaluated else None,
    }


def rows_by_generator(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("generator", ""))].append(row)
    return dict(groups)


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def ledger_by_generator(records: list[dict] | None) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for record in records or []:
        label = str(record.get("generator", ""))
        stats = out.setdefault(label, {
            "calls": 0, "ana": 0, "repair": 0, "retry": 0, "failed": 0,
            "fallback_records": 0, "fallback_cases": 0,
            "input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0,
            "billable_output_tokens": 0,
            "cost_usd_billed": 0.0, "priced_calls": 0, "unpriced_calls": 0,
            "cost_usd_list_equivalent": 0.0, "cost_usd_guard_estimate": 0.0,
        })
        call_type = record.get("call_type")
        if call_type == "fallback":
            stats["fallback_records"] += 1
            stats["fallback_cases"] += _int(record.get("accepted_cases"))
            continue
        stats["calls"] += 1
        if call_type in LLM_CALL_TYPES:
            stats[call_type] += 1
        if record.get("failed"):
            stats["failed"] += 1
        for key in ("input_tokens", "output_tokens", "reasoning_tokens", "billable_output_tokens"):
            stats[key] += _int(record.get(key))
        billed = _float(record.get("cost_usd_billed"))
        if record.get("pricing_available") and billed is not None:
            stats["priced_calls"] += 1
            stats["cost_usd_billed"] += billed
            stats["cost_usd_list_equivalent"] += _float(record.get("cost_usd_list_equivalent")) or 0.0
        else:
            stats["unpriced_calls"] += 1
            stats["cost_usd_guard_estimate"] += _float(record.get("cost_usd_guard_estimate")) or 0.0
    for stats in out.values():
        for key in ("cost_usd_billed", "cost_usd_list_equivalent", "cost_usd_guard_estimate"):
            stats[key] = round(stats[key], 6)
    return out


def tasks_by_generator(tasks: dict[str, dict] | None) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for task, record in (tasks or {}).items():
        stats = out.setdefault(task_to_label(task), {
            "tasks": 0, "rows": 0, "fallback_cases": 0, "failure_origins": [], "pending": 0,
        })
        stats["tasks"] += 1
        stats["rows"] += _int(record.get("rows"))
        stats["fallback_cases"] += _int(record.get("fallback_cases"))
        if record.get("failure_origin"):
            stats["failure_origins"].append(f"{task}={record.get('failure_origin')}")
        if record.get("next_available_at"):
            stats["pending"] += 1
    return out


def execution_fallback_by_generator(execution: list[dict] | None) -> dict[str, int] | None:
    if execution is None:
        return None
    counts: Counter = Counter()
    for record in execution:
        metadata = record.get("generation_metadata")
        if isinstance(metadata, dict) and metadata.get("fallback"):
            counts[str(record.get("generator", ""))] += 1
    return dict(counts)


def cost_status(label: str, ledger_stats: dict[str, Any] | None) -> str:
    """'ucretsiz' | 'ok' | 'dogrulanmadi' (ucretli ama billed yok/0)."""
    provider = provider_of(label)
    if provider not in PAID_PROVIDERS:
        return "ücretsiz"
    if not ledger_stats or not ledger_stats.get("priced_calls") or not ledger_stats.get("cost_usd_billed"):
        return "doğrulanmadı"
    return "ok"


def generator_summary(run: RunData) -> list[dict[str, Any]]:
    grouped = rows_by_generator(run.rows)
    repetition = {s["generator"]: s for s in compute_repetition_stats(run.rows)}
    ledger = ledger_by_generator(run.ledger) if run.ledger is not None else None
    tasks = tasks_by_generator(run.tasks) if run.tasks is not None else None

    labels = list(grouped)
    for extra in list((ledger or {}).keys()) + list((tasks or {}).keys()):
        if extra and extra not in labels:
            labels.append(extra)
    order = {label: i for i, label in enumerate(expected_labels())}
    labels.sort(key=lambda label: (order.get(label, len(order)), label))

    summary = []
    for label in labels:
        counts = pass_counts(grouped.get(label, []))
        rep = repetition.get(label)
        lstats = (ledger or {}).get(label) if ledger is not None else None
        tstats = (tasks or {}).get(label) if tasks is not None else None
        summary.append({
            "generator": label,
            "provider": provider_of(label),
            **counts,
            "repeated_rows": rep["repeated_rows"] if rep else None,
            "repetition_rate": rep["repetition_rate"] if rep else None,
            "fallback": tstats["fallback_cases"] if tstats else (0 if tasks is not None else None),
            "ledger": lstats,
            "has_ledger": ledger is not None,
            "cost_status": cost_status(label, lstats),
        })
    return summary


def failure_breakdown(run: RunData) -> dict[str, dict[str, Any]]:
    """Basarisiz satirlarin nedeni: status / json_path / diger.

    execution.jsonl varsa assertion sonuclarindan (allowed_statuses dahil) kesin
    hesaplanir; yoksa CSV'den expected_status != actual_status ile YAKLASIK.
    """
    out: dict[str, dict[str, Any]] = {}
    if run.execution is not None:
        for record in run.execution:
            if not _is_false(record.get("pass")):
                continue
            label = str(record.get("generator", ""))
            stats = out.setdefault(label, {"status": 0, "json_path": 0, "other": 0, "source": "execution"})
            expected = record.get("expected") if isinstance(record.get("expected"), dict) else {}
            allowed = [s for s in (expected.get("allowed_statuses") or []) if isinstance(s, int)]
            if not allowed and isinstance(expected.get("status"), int):
                allowed = [expected["status"]]
            actual = record.get("actual_status")
            status_ok = (not allowed) or (actual in allowed)
            failing = [a for a in (record.get("assertion_results") or [])
                       if isinstance(a, dict) and a.get("passed") is not True]
            if not status_ok or any(a.get("type") == "status_code" for a in failing):
                stats["status"] += 1
            elif any(str(a.get("type", "")).startswith("json_path") for a in failing):
                stats["json_path"] += 1
            else:
                stats["other"] += 1
        return out
    for row in run.rows:
        if not _is_false(row.get("pass")):
            continue
        label = str(row.get("generator", ""))
        stats = out.setdefault(label, {"status": 0, "json_path": 0, "other": 0, "source": "yaklaşık"})
        if str(row.get("expected_status", "")) != str(row.get("actual_status", "")):
            stats["status"] += 1
        else:
            stats["other"] += 1
    return out


def truncated_json_rows(run: RunData) -> int | None:
    """Govdesi kesilmis ve JSON tabanli assertion tasiyan satir sayisi."""
    if run.execution is None:
        return None
    count = 0
    for record in run.execution:
        if not str(record.get("actual_body") or "").endswith(TRUNCATION_MARKER):
            continue
        types = {str(a.get("type", "")) for a in (record.get("assertion_results") or []) if isinstance(a, dict)}
        if any(t.startswith("json_path") or t == "response_schema_check" for t in types):
            count += 1
    return count


def op_matrix(rows: list[dict[str, Any]]) -> tuple[list[str], dict[str, dict[str, dict[str, Any]]]]:
    ops = sorted({str(r.get("operation_id", "")) for r in rows})
    matrix: dict[str, dict[str, dict[str, Any]]] = {}
    for label, gen_rows in rows_by_generator(rows).items():
        per_op: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in gen_rows:
            per_op[str(row.get("operation_id", ""))].append(row)
        matrix[label] = {op: pass_counts(per_op.get(op, [])) for op in ops}
    return ops, matrix


def op_paths(rows: list[dict[str, Any]]) -> dict[str, str]:
    paths: dict[str, str] = {}
    for row in rows:
        paths.setdefault(str(row.get("operation_id", "")), f"{row.get('http_method', '')} {row.get('path', '')}")
    return paths


def status_mismatches(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row for row in rows
        if row.get("path") == STATUS_PATH
        and str(row.get("expected_status", "")) != str(row.get("actual_status", ""))
    ]


def numbered_run_dirs(root: Path | None = None) -> list[Path]:
    dirs = [d for d in list_run_dirs(root) if RUN_DIR_RE.match(d.name)]
    return sorted(dirs, key=lambda d: int(RUN_DIR_RE.match(d.name).group(1)))


def run_comparison(root: Path | None = None) -> tuple[list[str], list[dict[str, Any]]]:
    """run_N klasorleri yan yana: generator basina pass orani."""
    runs = numbered_run_dirs(root)
    names = [d.name for d in runs]
    rates: dict[str, dict[str, Any]] = defaultdict(dict)
    for run_dir in runs:
        csv_path = latest_csv(run_dir)
        _fields, rows = read_csv(csv_path) if csv_path else ([], [])
        for label, gen_rows in rows_by_generator(rows).items():
            rates[label][run_dir.name] = pass_counts(gen_rows)
    order = {label: i for i, label in enumerate(expected_labels())}
    table = [
        {"generator": label, "runs": rates[label]}
        for label in sorted(rates, key=lambda label: (order.get(label, len(order)), label))
    ]
    return names, table


# ── /reports baglami ────────────────────────────────────────────────────────

def run_list_entry(run_dir: Path) -> dict[str, Any]:
    """Run listesi icin hafif ozet (yalnizca CSV + run_info okunur)."""
    csv_path = latest_csv(run_dir)
    try:
        _fields, rows = read_csv(csv_path) if csv_path else ([], [])
    except (OSError, csv.Error, UnicodeDecodeError):
        rows = []
    state = git_state(latest_run_info(run_dir))
    counts = pass_counts(rows)
    return {
        "name": run_dir.name,
        "date": csv_timestamp(csv_path),
        "csv_name": csv_path.name if csv_path else "",
        "commit": (state["commit"] or "")[:7] or "yok",
        "tree": clean_label(state["clean"]),
        "rows": counts["rows"],
        "rate": counts["rate"],
    }


def default_run_dir(root: Path | None = None) -> Path | None:
    """Secim yoksa en yeni CSV'ye sahip kosu klasoru."""
    dirs = list_run_dirs(root)
    if not dirs:
        return None
    return max(dirs, key=lambda d: (latest_csv(d).name if latest_csv(d) else "", d.name))


def build_reports_context(selected: Path | None, root: Path | None = None,
                          expected_per_generator: int = 50) -> dict[str, Any]:
    base = _root(root)
    runs = [run_list_entry(d) for d in list_run_dirs(base)]
    runs.sort(key=lambda entry: entry["date"], reverse=True)
    context: dict[str, Any] = {
        "runs": runs,
        "selected": None,
        "pass_rate_note": PASS_RATE_NOTE,
        "tokens_note": TOKENS_NOTE,
        "expected_per_generator": expected_per_generator,
        "all_results_available": (base / FINAL_DIR_NAME / ALL_RESULTS_NAME).is_file(),
    }
    comparison_runs, comparison = run_comparison(base)
    context["comparison_runs"] = comparison_runs
    context["comparison"] = comparison
    if selected is None:
        return context

    run = load_run(selected)
    state = git_state(run.run_info)
    ops, matrix = op_matrix(run.rows)
    breakdown = failure_breakdown(run)
    summary = generator_summary(run)
    for item in summary:
        item["breakdown"] = breakdown.get(item["generator"])
        item["warn_rows"] = item["rows"] != expected_per_generator
        item["warn_fallback"] = item["fallback"] not in (0, None)
    context["selected"] = {
        "name": run.name,
        "csv_name": run.csv_path.name if run.csv_path else "yok",
        "csv_count": run.csv_count,
        "date": csv_timestamp(run.csv_path),
        "commit": state["commit"] or "yok",
        "branch": state["branch"] or "yok",
        "tree": clean_label(state["clean"]),
        "dirty_paths": state["dirty_paths"],
        "run_id": run.run_id or "yok",
        "has_tasks": run.tasks is not None,
        "has_ledger": run.ledger is not None,
        "totals": pass_counts(run.rows),
        "summary": summary,
        "ops": ops,
        "op_paths": op_paths(run.rows),
        "matrix": matrix,
        "rows": run.rows,
        "generators": sorted({str(r.get("generator", "")) for r in run.rows}),
        "variants": sorted({str(r.get("prompt_variant", "")) for r in run.rows}),
        "status_mismatches": status_mismatches(run.rows),
    }
    return context
