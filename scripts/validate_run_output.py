"""Run sonrasi CSV cikti dogrulamasi.

Senaryo A (9 generator x 150 test) gibi uzun kosulardan sonra uretilen
executed_testcases_*.csv dosyasini denetler:

  * satir sayisi beklenen degerle uyusuyor mu
  * kritik kolonlarda null/bos deger var mi
  * duplicate test kimligi var mi  (generator + prompt_variant + tc_id)
  * generator basina test dagilimi dengeli mi
  * token toplamlari tutarli mi (satir basi tekrar eden toplam token tespiti)
  * maliyet kolonu var mi ve satir bazinda tutarli mi

Veri henuz yoksa hatasiz sekilde "dogrulanacak dosya yok" raporu dondurur.

Kullanim:
    python scripts/validate_run_output.py
    python scripts/validate_run_output.py --expected-rows 1350 --expected-generators 9
    python scripts/validate_run_output.py --csv outputs/executed_testcases_20260926_120000.csv

Cikis kodlari:
    0 = bulgu yok (veya denetlenecek dosya yok)
    1 = en az bir bulgu var
    2 = kullanim / okuma hatasi
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Bos olmamasi gereken kolonlar. actual_status / pass yalnizca --executed ile zorunlu.
CRITICAL_COLUMNS = ("generator", "operation_id", "http_method", "path", "tc_id", "title")
EXECUTION_COLUMNS = ("url", "actual_status", "pass")

# Satir kimligi: tc_id yalnizca (generator, prompt_variant) icinde tekil olmali.
IDENTITY_COLUMNS = ("generator", "prompt_variant", "tc_id")


class Findings:
    """Bulgulari siniflandirarak toplar."""

    def __init__(self) -> None:
        self.items: list[tuple[str, str]] = []

    def add(self, severity: str, message: str) -> None:
        self.items.append((severity, message))

    @property
    def critical(self) -> list[str]:
        return [m for s, m in self.items if s == "KRITIK"]

    @property
    def warnings(self) -> list[str]:
        return [m for s, m in self.items if s == "UYARI"]

    @property
    def info(self) -> list[str]:
        return [m for s, m in self.items if s == "BILGI"]

    def __bool__(self) -> bool:
        return any(s in ("KRITIK", "UYARI") for s, _ in self.items)


def _latest_results_csv(output_dir: Path) -> Path | None:
    if not output_dir.is_dir():
        return None
    candidates = sorted(output_dir.glob("executed_testcases_*.csv"))
    return candidates[-1] if candidates else None


def _read_rows(csv_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    return fieldnames, rows


def _is_blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _to_float(value: Any) -> float | None:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    parsed = _to_float(value)
    if parsed is None:
        return None
    return int(parsed)


# ── Denetimler ────────────────────────────────────────────────────────────────

def check_row_count(rows: list[dict], expected: int | None, findings: Findings) -> None:
    findings.add("BILGI", f"Toplam satir sayisi: {len(rows)}")
    if expected is None:
        return
    if len(rows) != expected:
        findings.add(
            "KRITIK",
            f"Satir sayisi beklenenden farkli: beklenen={expected} bulunan={len(rows)} "
            f"(fark={len(rows) - expected:+d})",
        )


def check_nulls(rows: list[dict], fieldnames: list[str], executed: bool, findings: Findings) -> None:
    columns = list(CRITICAL_COLUMNS)
    if executed:
        columns += list(EXECUTION_COLUMNS)

    for column in columns:
        if column not in fieldnames:
            findings.add("KRITIK", f"Beklenen kolon CSV'de yok: {column}")
            continue
        blanks = [idx for idx, row in enumerate(rows, start=2) if _is_blank(row.get(column))]
        if blanks:
            preview = ", ".join(str(line) for line in blanks[:5])
            suffix = " ..." if len(blanks) > 5 else ""
            findings.add(
                "KRITIK",
                f"'{column}' kolonunda {len(blanks)} bos deger (CSV satirlari: {preview}{suffix})",
            )


def check_duplicate_ids(rows: list[dict], fieldnames: list[str], findings: Findings) -> None:
    keys = [column for column in IDENTITY_COLUMNS if column in fieldnames]
    if "tc_id" not in keys:
        findings.add("KRITIK", "tc_id kolonu yok, duplicate denetimi yapilamadi.")
        return
    if keys != list(IDENTITY_COLUMNS):
        missing = sorted(set(IDENTITY_COLUMNS) - set(keys))
        findings.add("UYARI", f"Kimlik kolonlari eksik ({', '.join(missing)}); denetim {keys} ile yapildi.")

    counter = Counter(tuple(str(row.get(key, "")) for key in keys) for row in rows)
    duplicates = {key: count for key, count in counter.items() if count > 1}
    if duplicates:
        sample = list(duplicates.items())[:5]
        detail = "; ".join(f"{'|'.join(key)} x{count}" for key, count in sample)
        suffix = " ..." if len(duplicates) > 5 else ""
        findings.add(
            "KRITIK",
            f"{len(duplicates)} duplicate kimlik ({'+'.join(keys)}): {detail}{suffix}",
        )
    else:
        findings.add("BILGI", f"Duplicate kimlik yok ({'+'.join(keys)} bazinda).")

    # tc_id'nin tek basina tekil OLMAMASI beklenen davranis; bilgi olarak raporla.
    bare = Counter(str(row.get("tc_id", "")) for row in rows)
    repeated = sum(1 for count in bare.values() if count > 1)
    if repeated:
        findings.add(
            "BILGI",
            f"{repeated} tc_id degeri birden fazla generator'da tekrar ediyor "
            f"(beklenen: ayni op_id_TCn tum generator'larda uretilir).",
        )


def check_generator_balance(
    rows: list[dict],
    expected_generators: int | None,
    expected_per_generator: int | None,
    tolerance: float,
    findings: Findings,
) -> None:
    per_generator = Counter(str(row.get("generator", "")) for row in rows)
    findings.add("BILGI", f"Farkli generator etiketi sayisi: {len(per_generator)}")
    for generator, count in sorted(per_generator.items()):
        findings.add("BILGI", f"  {generator}: {count}")

    if expected_generators is not None and len(per_generator) != expected_generators:
        findings.add(
            "KRITIK",
            f"Generator etiketi sayisi beklenenden farkli: beklenen={expected_generators} "
            f"bulunan={len(per_generator)}",
        )

    if expected_per_generator is not None:
        low = max(1, int(expected_per_generator * (1 - tolerance)))
        high = int(expected_per_generator * (1 + tolerance))
        for generator, count in sorted(per_generator.items()):
            if not low <= count <= high:
                findings.add(
                    "KRITIK",
                    f"Dengesiz dagilim: '{generator}' {count} test "
                    f"(beklenen {expected_per_generator}, kabul araligi {low}-{high})",
                )
    elif per_generator:
        low, high = min(per_generator.values()), max(per_generator.values())
        if low and high / low > 1 + tolerance:
            findings.add(
                "UYARI",
                f"Generator'lar arasi dagilim dengesiz: en az={low}, en cok={high} "
                f"(oran {high / low:.2f}x, tolerans {1 + tolerance:.2f}x)",
            )


def check_tokens(rows: list[dict], fieldnames: list[str], findings: Findings) -> None:
    if "tokens_used" not in fieldnames:
        findings.add("UYARI", "tokens_used kolonu yok; token tutarliligi denetlenemedi.")
        return

    by_generator: dict[str, list[int]] = defaultdict(list)
    negatives = 0
    for row in rows:
        value = _to_int(row.get("tokens_used"))
        if value is None:
            continue
        if value < 0:
            negatives += 1
        by_generator[str(row.get("generator", ""))].append(value)

    if negatives:
        findings.add("KRITIK", f"{negatives} satirda negatif tokens_used degeri var.")

    total_tokens = sum(sum(values) for values in by_generator.values())
    findings.add("BILGI", f"tokens_used toplami (satir bazinda): {total_tokens}")

    # base.py::_apply_token_tracking her satira operasyonun TOPLAM tokenini yazar.
    # Bu durumda ayni (generator, operation_id) grubunda tum degerler ayni olur ve
    # satirlari toplamak gercek token sayisini ~num_cases kati sisirir.
    suspicious: list[str] = []
    grouped: dict[tuple[str, str], set[int]] = defaultdict(set)
    for row in rows:
        value = _to_int(row.get("tokens_used"))
        if value is None or value == 0:
            continue
        grouped[(str(row.get("generator", "")), str(row.get("operation_id", "")))].add(value)
    for (generator, operation_id), values in grouped.items():
        group_size = sum(
            1
            for row in rows
            if str(row.get("generator", "")) == generator
            and str(row.get("operation_id", "")) == operation_id
        )
        if group_size > 1 and len(values) == 1:
            suspicious.append(f"{generator}/{operation_id} ({group_size} satir, tek deger {values.pop()})")

    if suspicious:
        preview = "; ".join(suspicious[:3])
        suffix = " ..." if len(suspicious) > 3 else ""
        findings.add(
            "KRITIK",
            f"{len(suspicious)} (generator, operation_id) grubunda tum satirlar AYNI tokens_used "
            f"degerini tasiyor. Bu, operasyon toplaminin her satira kopyalandigini gosterir; "
            f"satirlari toplamak token/maliyet sayisini sisirir: {preview}{suffix}",
        )


def check_cost(rows: list[dict], fieldnames: list[str], findings: Findings) -> None:
    cost_columns = [name for name in fieldnames if "cost" in name.lower()]
    if not cost_columns:
        findings.add(
            "KRITIK",
            "CSV'de maliyet kolonu (cost_usd) YOK. Harcama, uretilen veriden dogrulanamaz; "
            "saglayici faturasi ile karsilastirma yapilamaz.",
        )
        return

    for column in cost_columns:
        values = [_to_float(row.get(column)) for row in rows]
        parsed = [value for value in values if value is not None]
        missing = len(values) - len(parsed)
        if missing:
            findings.add("KRITIK", f"'{column}' kolonunda {missing} satir sayisal degil/bos.")
        negatives = sum(1 for value in parsed if value < 0)
        if negatives:
            findings.add("KRITIK", f"'{column}' kolonunda {negatives} negatif deger var.")
        if parsed:
            findings.add("BILGI", f"{column} toplami: {sum(parsed):.6f}")
            per_generator: dict[str, float] = defaultdict(float)
            for row in rows:
                value = _to_float(row.get(column))
                if value is not None:
                    per_generator[str(row.get("generator", ""))] += value
            for generator, total in sorted(per_generator.items()):
                findings.add("BILGI", f"  {generator}: {total:.6f}")


def check_status_sanity(rows: list[dict], fieldnames: list[str], executed: bool, findings: Findings) -> None:
    if "expected_status" in fieldnames:
        distribution = Counter(str(row.get("expected_status", "")).strip() for row in rows)
        findings.add("BILGI", f"expected_status dagilimi: {json.dumps(dict(distribution), ensure_ascii=False)}")
    if executed and "actual_status" in fieldnames:
        distribution = Counter(str(row.get("actual_status", "")).strip() for row in rows)
        findings.add("BILGI", f"actual_status dagilimi: {json.dumps(dict(distribution), ensure_ascii=False)}")
        unexecuted = distribution.get("", 0)
        if unexecuted:
            findings.add("UYARI", f"{unexecuted} satirda actual_status bos (istek atilmamis olabilir).")
    if executed and "pass" in fieldnames:
        distribution = Counter(str(row.get("pass", "")).strip() for row in rows)
        findings.add("BILGI", f"pass dagilimi: {json.dumps(dict(distribution), ensure_ascii=False)}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sonrasi CSV cikti dogrulamasi.")
    parser.add_argument("--csv", metavar="PATH", default=None, help="Denetlenecek CSV (varsayilan: outputs/ icindeki en yeni).")
    parser.add_argument("--output-dir", metavar="DIR", default=str(PROJECT_ROOT / "outputs"))
    parser.add_argument("--expected-rows", metavar="N", type=int, default=None, help="Beklenen toplam satir sayisi (orn. 1350).")
    parser.add_argument("--expected-generators", metavar="N", type=int, default=None, help="Beklenen generator etiketi sayisi (orn. 9).")
    parser.add_argument("--expected-per-generator", metavar="N", type=int, default=None, help="Generator basina beklenen test sayisi (orn. 150).")
    parser.add_argument("--tolerance", metavar="R", type=float, default=0.02, help="Dengeleme toleransi (varsayilan 0.02 = %%2).")
    parser.add_argument("--executed", action="store_true", help="url/actual_status/pass kolonlarini da zorunlu dene.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.csv:
        csv_path: Path | None = Path(args.csv).expanduser()
        if csv_path and not csv_path.is_file():
            print(f"[HATA] CSV bulunamadi: {csv_path}")
            return 2
    else:
        csv_path = _latest_results_csv(Path(args.output_dir).expanduser())

    print("=" * 72)
    print("RUN SONRASI VERI DOGRULAMA")
    print("=" * 72)

    if csv_path is None:
        print(f"Denetlenecek dosya yok: {args.output_dir}/executed_testcases_*.csv bulunamadi.")
        print("Sonuc: DENETLENECEK VERI YOK (script saglikli calisti).")
        return 0

    print(f"Dosya: {csv_path}")
    try:
        fieldnames, rows = _read_rows(csv_path)
    except (OSError, csv.Error, UnicodeDecodeError) as exc:
        print(f"[HATA] CSV okunamadi: {exc}")
        return 2

    print(f"Kolonlar ({len(fieldnames)}): {', '.join(fieldnames)}")
    print()

    if not rows:
        print("CSV bos (yalnizca baslik satiri).")
        print("Sonuc: VERI YOK (script saglikli calisti).")
        return 0

    findings = Findings()
    check_row_count(rows, args.expected_rows, findings)
    check_nulls(rows, fieldnames, args.executed, findings)
    check_duplicate_ids(rows, fieldnames, findings)
    check_generator_balance(
        rows, args.expected_generators, args.expected_per_generator, args.tolerance, findings
    )
    check_tokens(rows, fieldnames, findings)
    check_cost(rows, fieldnames, findings)
    check_status_sanity(rows, fieldnames, args.executed, findings)

    for label, items in (("KRITIK", findings.critical), ("UYARI", findings.warnings), ("BILGI", findings.info)):
        if not items:
            continue
        print(f"--- {label} ({len(items)}) ---")
        for item in items:
            print(f"  [{label}] {item}" if label != "BILGI" else f"  {item}")
        print()

    print("=" * 72)
    print(f"SONUC: KRITIK={len(findings.critical)}  UYARI={len(findings.warnings)}")
    print("=" * 72)
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
