"""Ucretsiz fazin tek satirlik durum ozeti (launchd her calistirmada yazar).

Kaynak: checkpoint gorev kayitlari + free_tier_limits.json. Hicbir API cagrisi
yapmaz, hicbir sir yazdirmaz.

Cikti bicimi (tek satir):
  2026-10-02T17:30:00+03:00 | tamam 5/9 | bekleyen: Gemini/gemini-2.5-flash(9 cagri, en erken 2026-10-03T17:05) | tahmini kalan: 9 gun
"""

from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
import rate_limiter
from checkpoint import RunCheckpoint

OPERATION_COUNT = 5          # GET/POST/PUT/PATCH/DELETE
VARIANTS = ("basic", "edge_focused")

FREE_PHASE_MODELS = [
    ("Groq", "openai/gpt-oss-120b", "GroqGenerator"),
    ("Groq", "openai/gpt-oss-20b", "GroqGenerator"),
    ("Gemini", "gemini-2.5-flash", "GeminiGenerator"),
    ("Gemini", "gemini-3.5-flash-lite", "GeminiGenerator"),
]


def _task_keys() -> list:
    keys = ["traditional"]
    for _provider, model, cls in FREE_PHASE_MODELS:
        for variant in VARIANTS:
            keys.append(f"{cls}:{model}|{variant}")
    return keys


def _pending_until(record: dict):
    value = record.get("next_available_at")
    if not value:
        return None
    try:
        moment = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    now = datetime.now(moment.tzinfo) if moment.tzinfo else datetime.now()
    return moment if moment > now else None


def build_status(output_dir: str, run_id: str) -> str:
    checkpoint = RunCheckpoint(output_dir, run_id=run_id, enabled=True)
    records = checkpoint.task_records()
    limits = rate_limiter.load_limits()

    all_keys = _task_keys()
    done, waiting = [], {}
    for key in all_keys:
        record = records.get(key)
        if record is None:
            waiting.setdefault(key, None)
            continue
        until = _pending_until(record)
        if until is not None:
            waiting[key] = until
        elif int(record.get("rows") or 0) > 0:
            done.append(key)
        else:
            # Tamamlanmis isaretli ama satir uretmemis: kota/altyapi nedeniyle bos.
            waiting[key] = None

    parts = [datetime.now().astimezone().isoformat(timespec="seconds")]
    parts.append(f"tamam {len(done)}/{len(all_keys)}")

    if waiting:
        per_model: dict = {}
        for key, until in waiting.items():
            if key == "traditional":
                per_model.setdefault("Traditional", {"tasks": 0, "until": None})["tasks"] += 1
                continue
            model = key.split(":", 1)[1].split("|")[0]
            provider = next((p for p, m, _c in FREE_PHASE_MODELS if m == model), "?")
            entry = per_model.setdefault(f"{provider}/{model}", {"tasks": 0, "until": None})
            entry["tasks"] += 1
            if until is not None and (entry["until"] is None or until < entry["until"]):
                entry["until"] = until

        chunks = []
        max_days = 0
        for name, entry in sorted(per_model.items()):
            calls = entry["tasks"] * OPERATION_COUNT
            limit = next(
                (L for (p, m), L in limits.items() if f"{p}/{m}" == name), None
            )
            if limit is not None and limit.effective_rpd:
                days = math.ceil(calls / limit.effective_rpd)
                max_days = max(max_days, days)
                chunks.append(f"{name}({calls} cagri, ~{days} gun)")
            else:
                chunks.append(f"{name}({calls} cagri)")
            if entry["until"] is not None:
                chunks[-1] = chunks[-1][:-1] + f", en erken {entry['until'].strftime('%Y-%m-%dT%H:%M')})"
        parts.append("bekleyen: " + " ".join(chunks))
        parts.append(f"tahmini kalan: {max_days} gun" if max_days else "tahmini kalan: bugun")
    else:
        parts.append("bekleyen: yok")
        parts.append("tahmini kalan: 0 gun — UCRETSIZ FAZ TAMAM")
    return " | ".join(parts)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Ucretsiz faz durum satiri")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--append-to", default=None, help="Satirin eklenecegi dosya")
    args = parser.parse_args(argv)

    line = build_status(args.output_dir, args.run_id)
    print(line)
    if args.append_to:
        path = Path(args.append_to)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
