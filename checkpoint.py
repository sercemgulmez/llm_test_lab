"""Uzun kosulari cokmeye karsi koruyan checkpoint / resume mekanizmasi.

Senaryo A gibi saatler suren paid kosularda main.py tum sonuclari yalnizca
en sonda diske yaziyordu; ortada bir cokme tum uretilen veriyi ve harcanan
parayi yok ediyordu. Bu modul iki fazi ayri ayri kalici hale getirir:

  generation.jsonl  - uretilen testcase satirlari (pahali olan taraf)
  tasks.jsonl       - tamamlanmis uretim gorevleri (generator x prompt variant)
  execution.jsonl   - calistirilmis testcase satirlari

Yazim append-only JSONL'dir: her flush'ta write + flush + fsync yapilir, bu
yuzden cokme aninda en fazla son (henuz flush edilmemis) tampon kaybolur.
Okuma sirasinda yarim kalmis son satir sessizce atlanir.

NOT: CSV'nin atomik yazilmasi ayri bir bulgudur (ORTA) ve bu modulun kapsami
disindadir; burada yalnizca checkpoint/resume ele alinir.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

_logger = logging.getLogger(__name__)

CHECKPOINT_DIR_NAME = ".checkpoints"
DEFAULT_FLUSH_EVERY = 25

# validate_run_output.py ile ayni kimlik tanimi: tc_id tek basina tekil degil.
IDENTITY_COLUMNS: Tuple[str, ...] = ("generator", "prompt_variant", "tc_id")

# Satirin hangi uretim gorevine ait oldugunu tutan dahili alan (CSV'ye sizmaz).
TASK_FIELD = "_checkpoint_task"
# Gorevin kacinci kosusunda uretildigi. Yeniden kosulan gorevin ESKI satirlari
# append-only dosyada durmaya devam eder; bu damga olmadan gorev yeniden
# tamamlandiginda eski satirlar da geri yuklenir ve duplicate uretir.
EPOCH_FIELD = "_checkpoint_epoch"


def row_identity(row: Dict[str, Any]) -> Tuple[str, ...]:
    """Bir satirin checkpoint kimligi."""
    return tuple(str(row.get(column, "")) for column in IDENTITY_COLUMNS)


def new_run_id() -> str:
    """Yeni bir kosu kimligi uretir."""
    return datetime.now().strftime("run_%Y%m%d_%H%M%S")


class _JsonlLog:
    """Tamponlu, fsync'li append-only JSONL kaydi."""

    def __init__(self, path: Path, flush_every: int = DEFAULT_FLUSH_EVERY) -> None:
        self.path = path
        self.flush_every = max(1, int(flush_every))
        self._buffer: List[str] = []
        self._lock = threading.Lock()

    def load(self) -> List[dict]:
        """Diskteki kayitlari okur; bozuk/yarim satirlari atlar."""
        if not self.path.is_file():
            return []
        records: List[dict] = []
        skipped = 0
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    skipped += 1  # cokme aninda yarim kalmis son satir
                    continue
                if isinstance(record, dict):
                    records.append(record)
        if skipped:
            _logger.warning(
                "  [checkpoint] %s: %d bozuk satir atlandi (cokme kalintisi).",
                self.path.name, skipped,
            )
        return records

    def append(self, record: dict) -> None:
        with self._lock:
            self._buffer.append(json.dumps(record, ensure_ascii=False, default=str))
            should_flush = len(self._buffer) >= self.flush_every
        if should_flush:
            self.flush()

    def extend(self, records: Iterable[dict]) -> None:
        for record in records:
            self.append(record)

    def flush(self) -> None:
        with self._lock:
            if not self._buffer:
                return
            pending, self._buffer = self._buffer, []
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(pending) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


class RunCheckpoint:
    """Bir kosunun uretim ve yurutme fazlarini kalici hale getirir."""

    def __init__(
        self,
        output_dir: str,
        run_id: Optional[str] = None,
        flush_every: int = DEFAULT_FLUSH_EVERY,
        enabled: bool = True,
    ) -> None:
        self.enabled = enabled
        self.run_id = run_id or new_run_id()
        self.resumed = bool(run_id)
        self.dir = Path(output_dir) / CHECKPOINT_DIR_NAME / self.run_id
        self._generation = _JsonlLog(self.dir / "generation.jsonl", flush_every)
        self._tasks = _JsonlLog(self.dir / "tasks.jsonl", 1)  # gorevler hemen yazilir
        self._execution = _JsonlLog(self.dir / "execution.jsonl", flush_every)

    # ── Uretim fazi ──────────────────────────────────────────────────────

    def load_generated_rows(self) -> List[dict]:
        """Yalnizca TAMAMLANMIS gorevlere ait satirlari doner.

        record_generated() ile mark_task_done() arasinda cokme olursa satirlar
        diske inmis ama gorev tamamlanmamis olur. O gorev resume'da yeniden
        kosacagi icin yetim satirlarin yuklenmesi duplicate uretirdi; bu yuzden
        gorev anahtari tamamlananlar arasinda olmayan satirlar atilir.
        """
        if not self.enabled:
            return []
        task_records = self.task_records()
        rows: List[dict] = []
        orphaned = 0
        stale = 0
        for record in self._generation.load():
            task_key = str(record.get(TASK_FIELD, ""))
            task = task_records.get(task_key)
            if task is None:
                orphaned += 1
                continue
            # Yeniden kosulan gorevin ONCEKI nesil satirlari atilir: yeniden kosu
            # kismi birlestirme degil TAM DEGISTIRMEDIR.
            if int(record.get(EPOCH_FIELD, 0) or 0) != int(task.get("retry_count") or 0):
                stale += 1
                continue
            row = {
                key: value for key, value in record.items()
                if key not in (TASK_FIELD, EPOCH_FIELD)
            }
            rows.append(row)
        if stale:
            _logger.info(
                "  [checkpoint] %d eski nesil satir atlandi (yeniden kosulan gorevler).", stale,
            )
        if orphaned:
            _logger.warning(
                "  [checkpoint] %d yetim satir atlandi (gorevi tamamlanmamis); "
                "o gorev yeniden kosacak.", orphaned,
            )
        return rows

    def task_records(self) -> Dict[str, dict]:
        """Gorev anahtari -> son gecerli kayit. Iptal edilen gorevler DUSER."""
        if not self.enabled:
            return {}
        records: Dict[str, dict] = {}
        for record in self._tasks.load():
            key = str(record.get("task", ""))
            if not key:
                continue
            if record.get("revoked"):
                records.pop(key, None)
            else:
                records[key] = record
        return records

    def completed_tasks(self) -> Set[str]:
        return set(self.task_records())

    def revoke_task(self, task_key: str, reason: str = "") -> None:
        """Gorevi 'tamamlanmadi' haline getirir.

        Satirlari silmeye gerek yoktur: load_generated_rows() yalnizca
        TAMAMLANMIS gorevlerin satirlarini yukledigi icin iptal edilen gorevin
        satirlari yetim kalir ve resume'da otomatik olarak atilir. Boylece
        yeniden kosu KISMI BIRLESTIRME degil TAM DEGISTIRME olur.
        """
        if self.enabled:
            self._tasks.append({"task": task_key, "revoked": True, "reason": reason})

    def record_generated(self, rows: Iterable[dict], task_key: str, epoch: int = 0) -> None:
        """Satirlari gorev anahtari VE nesil damgasiyla etiketleyerek yazar."""
        if self.enabled:
            self._generation.extend(
                {**row, TASK_FIELD: task_key, EPOCH_FIELD: epoch} for row in rows
            )

    def mark_task_done(
        self,
        task_key: str,
        row_count: int,
        fallback_cases: int = 0,
        failure_origin: Optional[str] = None,
        retry_count: int = 0,
        next_available_at: Optional[str] = None,
    ) -> None:
        """Gorevi tamamlanmis olarak isaretler.

        TAMAMLANMA SEMANTIGI DEGISMEZ: fallback iceren gorev de tamamlanmis
        sayilir. Ek alanlar yalnizca sonradan "hangi gorev altyapi hatasi
        yuzunden kirlendi" sorusunu cevaplayabilmek icindir.
        """
        if self.enabled:
            self._generation.flush()  # gorev tamamlandi olarak isaretlenmeden once satirlar diskte olsun
            self._tasks.append({
                "task": task_key,
                "rows": row_count,
                "fallback_cases": fallback_cases,
                "failure_origin": failure_origin,
                "retry_count": retry_count,
                # Gunluk kota bittiyse bu andan once yeniden denemek anlamsizdir;
                # --resume o zamana kadar bu gorevi atlar.
                "next_available_at": next_available_at,
            })

    # ── Yurutme fazi ─────────────────────────────────────────────────────

    def load_executed_rows(self) -> List[dict]:
        return self._execution.load() if self.enabled else []

    def executed_identities(self) -> Set[Tuple[str, ...]]:
        return {row_identity(row) for row in self.load_executed_rows()}

    def record_executed(self, row: dict) -> None:
        if self.enabled:
            self._execution.append(row)

    # ── Ortak ────────────────────────────────────────────────────────────

    def flush(self) -> None:
        if not self.enabled:
            return
        self._generation.flush()
        self._tasks.flush()
        self._execution.flush()
