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
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import pricing
from checkpoint import _JsonlLog
from error_taxonomy import classify_error, failure_origin
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
        self.write_errors: list[str] = []
        # B4(a): harcamanin TEK kaynagi defterdir. Kosu ICINDE de ayni toplam
        # kullanilsin diye burada canli birikim tutulur (K6 esikleri bunu okur).
        self._spend_lock = threading.Lock()
        self._spend_billed = 0.0
        self._spend_list_equivalent = 0.0
        self._unpriced_calls = 0
        self._seed_spend_from_disk()

    def _seed_spend_from_disk(self) -> None:
        """--resume: ayni run_id'nin ONCEKI harcamasi da butceye sayilir.

        Defter dosyasi run_id ile paylasildigi icin surdurulen bir kosuda
        dosya zaten doludur. Bu tutar yuklenmezse butce sigortasi her
        yeniden baslatmada sifirdan sayar ve $80 tavani asilabilir.
        """
        if not self.enabled or not self.path.is_file():
            return
        previous = total_spend(self.load(self.path))
        self._spend_billed = previous["cost_usd_billed"]
        self._spend_list_equivalent = previous["cost_usd_list_equivalent"]
        self._unpriced_calls = previous["unpriced_calls"]
        if self._spend_billed or self._unpriced_calls:
            _logger.info(
                "  [defter] onceki harcama yuklendi: $%.4f (%d fiyatlanamayan cagri)",
                self._spend_billed, self._unpriced_calls,
            )

    def _safe_append(self, record: dict) -> None:
        """Defter yazimi BASARISIZ olsa bile kosu devam etmeli.

        Defter bir tani kaydidir; kaybi can sikicidir ama uretilen satirlari
        ve harcanan parayi bosa cikarmaz. Bu yuzden hata yutulmaz (ERROR ile
        loglanir ve kosu ozetinde gorunur) ama yukari da firlatilmaz.
        """
        try:
            self._log.append(record)
        except Exception as exc:  # noqa: BLE001 - defter hatasi kosuyu durdurmamali
            message = f"{type(exc).__name__}: {redact_secrets(str(exc))}"
            self.write_errors.append(message)
            _logger.error("  [defter] KAYIT YAZILAMADI (kosu devam ediyor): %s", message)

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
        repeat_index: int = 0,
        latency_ms: int | None = None,
        call_meta: dict | None = None,
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
            "repeat_index": repeat_index,
            "call_type": call_type,
            "latency_ms": latency_ms,
            "model_requested": (call_meta or {}).get("model_requested"),
            "model_returned": (call_meta or {}).get("model_returned"),
            "response_id": (call_meta or {}).get("response_id"),
            "finish_reason": (call_meta or {}).get("finish_reason"),
            "sampling": _clean((call_meta or {}).get("sampling") or {}),
            **usage.to_dict(),
            **self._cost_fields(usage, (call_meta or {}).get("model_requested")),
            "accepted_cases": accepted_cases,
            "rejected_cases": rejected_cases,
            "raw_response": raw[:MAX_RAW_RESPONSE_CHARS],
            "raw_response_truncated": truncated,
            "raw_response_length": len(raw),
            "validation_errors": _clean(validation_errors or []),
            "cases": [_case_entry(row) for row in (cases or [])],
        }
        self._safe_append(record)

    def record_failed_call(
        self,
        generator: str,
        variant: str,
        operation_id: str,
        method: str,
        path: str,
        attempt: int,
        call_type: str,
        exc: BaseException,
        repeat_index: int = 0,
        latency_ms: int | None = None,
        call_meta: dict | None = None,
    ) -> str:
        """Yanit alinamayan cagriyi (429, kota, timeout, ag) deftere yazar.

        Doner: hata sinifi (cagiran taraf fallback nedenini belirlemek icin kullanir).
        """
        error_class = classify_error(exc) if isinstance(exc, Exception) else "UNKNOWN_ERROR"
        if not self.enabled:
            return error_class
        self._safe_append({
            "ts": datetime.now().astimezone().isoformat(timespec="seconds"),
            "run_id": self.run_id,
            "generator": generator,
            "variant": variant,
            "operation_id": operation_id,
            "method": method,
            "path": path,
            "attempt": attempt,
            "repeat_index": repeat_index,
            "call_type": call_type,
            "latency_ms": latency_ms,
            "model_requested": (call_meta or {}).get("model_requested"),
            "model_returned": (call_meta or {}).get("model_returned"),
            "sampling": _clean((call_meta or {}).get("sampling") or {}),
            **TokenUsage().to_dict(),
            # Yanit alinamadi: saglayici token bildirmedi, dolayisiyla bu cagrinin
            # para harcayip harcamadigi BILINMIYOR. Sifir YAZILMAZ; kayit
            # fiyatlandirilamamis sayilir ve toplam "eksik" olarak raporlanir.
            "cost_usd_billed": None,
            "cost_usd_list_equivalent": None,
            "cost_basis": "cagri_basarisiz_token_bildirilmedi",
            "pricing_available": False,
            "accepted_cases": 0,
            "rejected_cases": 0,
            "raw_response": "",
            "raw_response_truncated": False,
            "raw_response_length": 0,
            "failed": True,
            "error_class": error_class,
            "failure_origin": failure_origin(error_class),
            "error": redact_secrets(str(exc)),
            "cases": [],
        })
        return error_class

    def record_fallback(
        self,
        generator: str,
        variant: str,
        operation_id: str,
        method: str,
        path: str,
        cases: list[dict],
        reason: str = "LLM yeterli gecerli case uretemedi",
        origin: str = "icerik",
    ) -> None:
        """Fallback satirlari API cagrisi OLMADAN uretilir; ayri kayit turu."""
        if not self.enabled:
            return
        self._safe_append({
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
            "failure_origin": origin,
            "cases": [_case_entry(row) for row in cases],
        })

    def _cost_fields(self, usage: TokenUsage, model: str | None) -> dict:
        """Cagri basina maliyet alanlari; ayni anda canli toplami da gunceller.

        Fiyat tablosu bos oldugu surece tutarlar None kalir ve cagri
        `unpriced_calls` icinde sayilir — maliyet ASLA tahmin edilmez.
        """
        cost = pricing.cost_for(model or "", usage)
        with self._spend_lock:
            if cost.pricing_available:
                self._spend_billed += cost.cost_usd_billed or 0.0
                self._spend_list_equivalent += cost.cost_usd_list_equivalent or 0.0
            else:
                self._unpriced_calls += 1
        return cost.to_dict()

    def spend_so_far(self) -> dict:
        """Kosu ICINDEKI canli harcama — butce esikleri bunu kullanir (K6).

        Kapsam: bu defterin yazdigi TUM cagrilar. Iptal edilen (revoke edilen)
        nesillerin cagrilari da dahildir, cunku para yine harcanmistir; defter
        append-only oldugu icin bu kendiliginden saglanir.
        """
        with self._spend_lock:
            return {
                "cost_usd_billed": round(self._spend_billed, 6),
                "cost_usd_list_equivalent": round(self._spend_list_equivalent, 6),
                "unpriced_calls": self._unpriced_calls,
                "pricing_available": pricing.price_table_ready(),
            }

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
            "reasoning_tokens": sum(int(r.get("reasoning_tokens") or 0) for r in api_calls),
            "billable_output_tokens": sum(int(r.get("billable_output_tokens") or 0) for r in api_calls),
            "total_tokens": sum(int(r.get("total_tokens") or 0) for r in api_calls),
            "token_split_available": all(bool(r.get("split_available")) for r in api_calls) if api_calls else False,
            **{f"spend_{key}": value for key, value in total_spend(items).items()},
        })
    return summary


def total_spend(records: list[dict]) -> dict:
    """Defterdeki TUM kayitlar uzerinden toplam harcama (B4(a)).

    CSV degil DEFTER esas alinir: CSV yalnizca kosunun sonunda hayatta kalan
    satirlari icerir, oysa para iptal edilen nesiller ve yanit alinamayan
    cagrilar icin de harcanmis olabilir. Bu yuzden hicbir kayit turu elenmez.

    `unpriced_calls` > 0 ise toplam EKSIKTIR ve oyle raporlanmalidir.
    """
    billed = 0.0
    list_equivalent = 0.0
    priced = 0
    unpriced = 0
    reasons: dict[str, int] = {}
    for record in records:
        if record.get("call_type") == "fallback":
            continue  # sablon uretimi, saglayiciya cagri degil
        if record.get("pricing_available"):
            billed += float(record.get("cost_usd_billed") or 0.0)
            list_equivalent += float(record.get("cost_usd_list_equivalent") or 0.0)
            priced += 1
        else:
            unpriced += 1
            basis = str(record.get("cost_basis") or "bilinmiyor")
            reasons[basis] = reasons.get(basis, 0) + 1
    return {
        "cost_usd_billed": round(billed, 6),
        "cost_usd_list_equivalent": round(list_equivalent, 6),
        "priced_calls": priced,
        "unpriced_calls": unpriced,
        "unpriced_reasons": reasons,
        "complete": unpriced == 0,
    }
