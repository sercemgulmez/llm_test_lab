"""Kosu anindaki TEK CAGRI DEFTERI (JSONL).

Paid kosu sirasinda yalnizca o an var olan bilgiyi kalici hale getirir:
ham yanit, tam istek, token ayrimi, kabul/red sayilari ve fallback bayragi.
Bunlarin hicbiri CSV'den geriye dogru uretilemez.

CSV semasina DOKUNULMAZ; defter ayri bir JSONL dosyasidir ve outputs/
altinda (gitignored) tutulur. run_id K7 checkpoint ile paylasilir.

Her kayit diske yazilirken flush + fsync yapilir: cokme aninda en fazla
yazilmakta olan tek satir kaybolur.

Tum serbest metin alanlari redact_secrets'tan gecirilir.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from checkpoint import _JsonlLog
from models import TokenUsage
from security.redaction import redact_secrets

_logger = logging.getLogger(__name__)

LEDGER_DIR_NAME = ".calls"
MAX_RAW_RESPONSE_CHARS = 20000


def _clean(value: Any) -> Any:
    """Serbest metinleri redakte eder; yapiyi bozmadan derinlemesine gezer."""
    if isinstance(value, str):
        return redact_secrets(value)
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clean(item) for item in value]
    return value


def _full_request(row: dict) -> dict:
    """Satirin TAM istegi: path/query/header/cookie/body birlikte."""
    request = row.get("request") if isinstance(row.get("request"), dict) else {}
    return _clean({
        "path_params": request.get("path_params", {}),
        "query_params": request.get("query_params", {}),
        "headers": request.get("headers", {}),
        "cookies": request.get("cookies", {}),
        "body": request.get("body"),
    })


def _case_entry(row: dict) -> dict:
    metadata = row.get("generation_metadata") if isinstance(row.get("generation_metadata"), dict) else {}
    return {
        "tc_id": row.get("tc_id", ""),
        "is_fallback": bool(metadata.get("fallback", False)),
        "source": metadata.get("source", ""),
        "expected_status": row.get("expected_status"),
        "request": _full_request(row),
    }


class CallLedger:
    """Bir kosunun tum LLM cagrilarini tek dosyada tutar."""

    def __init__(self, output_dir: str, run_id: str, enabled: bool = True) -> None:
        self.enabled = enabled
        self.run_id = run_id
        self.dir = Path(output_dir) / LEDGER_DIR_NAME / run_id
        self.path = self.dir / "calls.jsonl"
        self._log = _JsonlLog(self.path, flush_every=1)  # her cagridan sonra diske

    def record_call(
        self,
        generator: str,
        variant: str,
        operation_id: str,
        method: str,
        path: str,
        attempt: int,
        call_type: str,
        usage: TokenUsage,
        accepted_cases: int,
        rejected_cases: int,
        raw_response: str,
        cases: list[dict] | None = None,
        validation_errors: list[dict] | None = None,
    ) -> None:
        if not self.enabled:
            return
        raw = redact_secrets(raw_response or "")
        truncated = len(raw) > MAX_RAW_RESPONSE_CHARS
        record = {
            "ts": datetime.now().astimezone().isoformat(timespec="seconds"),
            "run_id": self.run_id,
            "generator": generator,
            "variant": variant,
            "operation_id": operation_id,
            "method": method,
            "path": path,
            "attempt": attempt,
            "call_type": call_type,
            **usage.to_dict(),
            "accepted_cases": accepted_cases,
            "rejected_cases": rejected_cases,
            "raw_response": raw[:MAX_RAW_RESPONSE_CHARS],
            "raw_response_truncated": truncated,
            "raw_response_length": len(raw),
            "validation_errors": _clean(validation_errors or []),
            "cases": [_case_entry(row) for row in (cases or [])],
        }
        self._log.append(record)

    def record_fallback(
        self,
        generator: str,
        variant: str,
        operation_id: str,
        method: str,
        path: str,
        cases: list[dict],
        reason: str = "LLM yeterli gecerli case uretemedi",
    ) -> None:
        """Fallback satirlari API cagrisi OLMADAN uretilir; ayri kayit turu."""
        if not self.enabled:
            return
        self._log.append({
            "ts": datetime.now().astimezone().isoformat(timespec="seconds"),
            "run_id": self.run_id,
            "generator": generator,
            "variant": variant,
            "operation_id": operation_id,
            "method": method,
            "path": path,
            "attempt": -1,
            "call_type": "fallback",
            **TokenUsage().to_dict(),
            "accepted_cases": len(cases),
            "rejected_cases": 0,
            "raw_response": "",
            "raw_response_truncated": False,
            "raw_response_length": 0,
            "reason": redact_secrets(reason),
            "cases": [_case_entry(row) for row in cases],
        })

    def flush(self) -> None:
        if self.enabled:
            self._log.flush()

    # ── Okuma tarafi (validate_run_output.py kullanir) ────────────────────

    @staticmethod
    def load(path: Path) -> list[dict]:
        return _JsonlLog(path).load()

    @staticmethod
    def latest_path(output_dir: str) -> Path | None:
        root = Path(output_dir) / LEDGER_DIR_NAME
        if not root.is_dir():
            return None
        candidates = sorted(root.glob("*/calls.jsonl"))
        return candidates[-1] if candidates else None


def summarize_by_generator(records: list[dict]) -> list[dict]:
    """Generator basina fallback payi ve kabul orani."""
    groups: dict[str, list[dict]] = {}
    for record in records:
        groups.setdefault(str(record.get("generator", "")), []).append(record)

    summary: list[dict] = []
    for generator, items in sorted(groups.items()):
        api_calls = [r for r in items if r.get("call_type") != "fallback"]
        fallback_calls = [r for r in items if r.get("call_type") == "fallback"]
        accepted = sum(int(r.get("accepted_cases") or 0) for r in api_calls)
        rejected = sum(int(r.get("rejected_cases") or 0) for r in api_calls)
        fallback_cases = sum(int(r.get("accepted_cases") or 0) for r in fallback_calls)
        produced = accepted + fallback_cases
        offered = accepted + rejected
        summary.append({
            "generator": generator,
            "api_calls": len(api_calls),
            "main_calls": sum(1 for r in api_calls if r.get("call_type") == "ana"),
            "repair_calls": sum(1 for r in api_calls if r.get("call_type") == "repair"),
            "retry_calls": sum(1 for r in api_calls if r.get("call_type") == "retry"),
            "accepted_cases": accepted,
            "rejected_cases": rejected,
            "acceptance_rate": round(accepted / offered, 4) if offered else None,
            "fallback_cases": fallback_cases,
            "total_cases": produced,
            "fallback_share": round(fallback_cases / produced, 4) if produced else None,
            "input_tokens": sum(int(r.get("input_tokens") or 0) for r in api_calls),
            "output_tokens": sum(int(r.get("output_tokens") or 0) for r in api_calls),
            "total_tokens": sum(int(r.get("total_tokens") or 0) for r in api_calls),
            "token_split_available": all(bool(r.get("split_available")) for r in api_calls) if api_calls else False,
        })
    return summary
