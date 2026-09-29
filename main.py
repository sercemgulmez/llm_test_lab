"""
LLM Test Lab — Otomatik API Test Senaryosu Üretim ve Yürütme Aracı

Kullanım:
    python main.py                          # İnteraktif wizard
    python main.py --curl-file turkcell.txt # Doğrudan argümanlar
    python main.py --help                   # Tüm seçenekler
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import sys
import time

# pricing ortam degiskenine bagli degil; load_dotenv oncesinde guvenle import edilir.
import attestation
import budget as budget_module
import pricing
import rate_limiter

from dotenv import load_dotenv

# Load project-local secrets without overriding explicitly exported variables.
load_dotenv(Path(__file__).resolve().parent / ".env")

import config
from call_ledger import CallLedger
from checkpoint import DEFAULT_FLUSH_EVERY, RunCheckpoint, row_identity
from models import ApiOperation
from parsers.openapi import load_openapi_from_url, extract_operations_from_openapi, manual_operations_input
from parsers.curl_parser import parse_curl_collection
from security.redaction import redact_secrets
from generators import TraditionalGenerator, GENERATOR_REGISTRY
from runner import run_testcases
from reporters.csv_reporter import (
    compute_repetition_stats,
    format_repetition_table,
    save_operations_csv,
    save_results_csv,
    compute_generator_metrics,
    save_generator_metrics_csv,
    print_summary_table,
)

_logger = logging.getLogger(__name__)


def _configure_logging(verbose: bool = False) -> None:
    """Root logger'ı stdout'a yönlendirir; main.py CLI akışı için."""
    level = logging.DEBUG if verbose else logging.INFO
    root = logging.getLogger()
    root.setLevel(level)
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        root.addHandler(handler)


# ── Wizard yardımcı fonksiyonları ────────────────────────────────────────────

def _wi(prompt: str, default: str = "") -> str:
    """Tek satır input. Boş bırakılırsa default döner."""
    display = f"  {prompt} [{default}]: " if default else f"  {prompt}: "
    try:
        val = input(display).strip()
        return val if val else default
    except (KeyboardInterrupt, EOFError):
        print("\n\nÇıkılıyor...")
        sys.exit(0)


def _wc(prompt: str, choices: list, default: str = None) -> str:
    """Belirli seçeneklerden birini seçtirir (büyük/küçük harf duyarsız)."""
    cs = "/".join(choices)
    display = f"  {prompt} ({cs}) [{default}]: " if default else f"  {prompt} ({cs}): "
    while True:
        try:
            val = input(display).strip().lower()
            if not val and default:
                return default.lower()
            if val in [c.lower() for c in choices]:
                return val
            print(f"    Geçersiz seçim. Lütfen {cs} arasından birini girin.")
        except (KeyboardInterrupt, EOFError):
            print("\n\nÇıkılıyor...")
            sys.exit(0)


def _wms(prompt: str, options: list) -> list:
    """Çoklu seçim. Seçilen indekslerin listesini (0 tabanlı) döner."""
    print()
    for i, opt in enumerate(options, 1):
        print(f"    {i:2}) {opt}")
    print("     A) Tümünü seç")
    print()
    while True:
        try:
            val = input(f"  {prompt} [virgülle ayırın, örn: 1,3,4  veya A]: ").strip().upper()
            if not val:
                print("    En az bir seçenek seçmelisiniz.")
                continue
            if val == "A":
                return list(range(len(options)))
            selected = []
            valid = True
            for part in val.split(","):
                part = part.strip()
                if part.isdigit():
                    idx = int(part) - 1
                    if 0 <= idx < len(options):
                        if idx not in selected:
                            selected.append(idx)
                    else:
                        print(f"    Geçersiz numara: {part} (1-{len(options)} arasında olmalı)")
                        valid = False
                        break
                else:
                    print(f"    Geçersiz giriş: '{part}'")
                    valid = False
                    break
            if valid and selected:
                return selected
            elif valid:
                print("    En az bir seçenek seçmelisiniz.")
        except (KeyboardInterrupt, EOFError):
            print("\n\nÇıkılıyor...")
            sys.exit(0)


def _separator(title: str = "") -> None:
    if title:
        pad = (58 - len(title)) // 2
        print(f"\n{'─' * pad}  {title}  {'─' * pad}")
    else:
        print("─" * 60)


# ── İnteraktif Wizard ────────────────────────────────────────────────────────

def interactive_wizard() -> argparse.Namespace:
    """Adım adım soru soran interaktif wizard. argparse.Namespace döner."""

    print()
    print("=" * 60)
    print("          LLM TEST LAB — Hoş Geldiniz!")
    print("      Otomatik API Test Senaryosu Üretici")
    print("=" * 60)
    print()
    print("  Adım adım yönlendirileceksiniz.")
    print("  İstediğiniz zaman Ctrl+C ile çıkabilirsiniz.")
    print()

    # ── Adım 1: Operasyon kaynağı ────────────────────────────────────────
    _separator("ADIM 1 — Operasyon Kaynağı")
    print()
    print("    1) curl dosyasından oku")
    print("    2) OpenAPI / Swagger URL'den çek")
    print("    3) Manuel operasyon girişi")
    print()
    source = _wc("Kaynağı seçin", ["1", "2", "3"], default="1")

    curl_file = None
    openapi_url = None
    base_url = None

    if source == "1":
        while True:
            paths = _wi("curl dosya yolu(ları) [birden fazlaysa boşlukla ayırın]")
            if not paths:
                print("    Dosya yolu boş olamaz, lütfen tekrar girin.")
                continue
            file_list = paths.split()
            missing = [f for f in file_list if not os.path.isfile(f)]
            if missing:
                print(f"    Bulunamayan dosya(lar): {', '.join(missing)}")
                print("    Lütfen geçerli bir dosya yolu girin.")
                continue
            curl_file = file_list
            break
        base_url_inp = _wi("Base URL [curl'den otomatik alınır, opsiyonel — Enter geç]")
        base_url = base_url_inp or None

    elif source == "2":
        openapi_url = _wi("OpenAPI/Swagger dokümanının URL'i")
        if not openapi_url:
            print("  HATA: URL boş olamaz.")
            sys.exit(1)
        base_url = _wi("API Base URL (örn: https://api.example.com/v1)")
        if not base_url:
            print("  HATA: Base URL boş olamaz.")
            sys.exit(1)

    else:  # manuel
        base_url = _wi("API Base URL (örn: https://api.example.com/v1)")
        if not base_url:
            print("  HATA: Base URL boş olamaz.")
            sys.exit(1)

    # ── Adım 2: Kimlik doğrulama ─────────────────────────────────────────
    _separator("ADIM 2 — Kimlik Doğrulama")
    print()
    auth_token = _wi("Bearer Token [yoksa Enter geç]") or None

    extra_headers_raw: list = []
    add_hdr = _wc("Ekstra HTTP header eklemek ister misiniz?", ["e", "h"], default="h")
    if add_hdr == "e":
        print("  Her satıra bir header. Bitirmek için boş bırakın.")
        while True:
            hdr = _wi("  Header (örn: App-Channel-Type: WEB) [bitirmek için Enter]")
            if not hdr:
                break
            extra_headers_raw.append(hdr)

    cookie_str = _wi("Cookie string [yoksa Enter geç]") or None

    # ── Adım 3: Generator seçimi ─────────────────────────────────────────
    _separator("ADIM 3 — Generator Seçimi")
    print()
    print("  Hangi generator'ları kullanmak istersiniz?")

    gen_options = ["Geleneksel şablon  (API anahtarı gerektirmez)"]
    gen_keys = ["traditional"]

    for m in config.OPENAI_MODELS:
        gen_options.append(f"OpenAI            {m}")
        gen_keys.append(f"openai:{m}")
    for m in config.GEMINI_MODELS:
        gen_options.append(f"Google Gemini     {m}")
        gen_keys.append(f"gemini:{m}")
    for m in config.CLAUDE_MODELS:
        gen_options.append(f"Anthropic Claude  {m}")
        gen_keys.append(f"claude:{m}")
    for m in config.GROQ_MODELS:
        gen_options.append(f"Groq              {m}")
        gen_keys.append(f"groq:{m}")

    selected_indices = _wms("Generator seçin", gen_options)
    selected_keys = [gen_keys[i] for i in selected_indices]

    # ── Adım 4: Senaryo sayısı ────────────────────────────────────────────
    _separator("ADIM 4 — Senaryo Sayısı")
    print()
    print("  Her operasyon için kaç LLM test senaryosu üretilsin?")
    num_cases_str = _wi("Senaryo sayısı", str(config.NUM_CASES_PER_OPERATION))
    num_cases = config.normalize_num_cases(num_cases_str)

    # ── Adım 5: Test çalıştırma ───────────────────────────────────────────
    _separator("ADIM 5 — Test Çalıştırma")
    print()
    run_str = _wc("Üretilen testler API'ye karşı gerçekten çalıştırılsın mı?", ["e", "h"], default="e")
    no_run = run_str == "h"

    # ── Adım 6: Çıktı klasörü ─────────────────────────────────────────────
    _separator("ADIM 6 — Çıktı Ayarları")
    print()
    output_dir = _wi("Çıktı klasörü", config.OUTPUT_DIR)

    # ── Özet ──────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  ÖZET — Başlamadan önce kontrol edin")
    print("─" * 60)
    if curl_file:
        print(f"  Kaynak       : curl ({', '.join(curl_file)})")
    elif openapi_url:
        print(f"  Kaynak       : OpenAPI ({openapi_url})")
    else:
        print("  Kaynak       : Manuel giriş")

    base_display = base_url or "(curl'den otomatik alınacak)"
    print(f"  Base URL     : {base_display}")
    print(f"  Auth Token   : {'Var' if auth_token else 'Yok'}")
    print(f"  Ekstra Header: {len(extra_headers_raw)} adet")
    print(f"  Cookie       : {'Var' if cookie_str else 'Yok'}")
    print("  Generator    :")
    for i in selected_indices:
        print(f"               - {gen_options[i].strip()}")
    print(f"  Senaryo/op   : {num_cases}")
    print(f"  Test çalıştır: {'Hayır' if no_run else 'Evet'}")
    print(f"  Çıktı        : {output_dir}/")
    print("=" * 60)
    print()

    confirm = _wc("Başlamak istiyor musunuz?", ["e", "h"], default="e")
    if confirm != "e":
        print("\nİptal edildi.")
        sys.exit(0)

    print()

    return argparse.Namespace(
        curl_file=curl_file,
        openapi_url=openapi_url,
        base_url=base_url,
        auth_token=auth_token,
        no_run=no_run,
        headers=extra_headers_raw,
        cookie=cookie_str,
        output_dir=output_dir,
        selected_generators=selected_keys,
        num_cases=num_cases,
    )


# ── Argparse (doğrudan argüman kullanımı) ────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="llm_test_lab",
        description="LLM ile API test senaryosu üretir ve yürütür.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--curl-file",
        metavar="FILE",
        nargs="+",
        default=None,
        help="curl komutlarını içeren dosya(lar).",
    )
    parser.add_argument("--openapi-url", metavar="URL")
    parser.add_argument("--base-url", metavar="URL", default=None)
    parser.add_argument("--auth-token", metavar="TOKEN", default=None)
    parser.add_argument("--no-run", action="store_true")
    parser.add_argument(
        "--header", metavar="KEY: VALUE", action="append",
        dest="headers", default=[],
    )
    parser.add_argument("--cookie", metavar="COOKIE_STRING", default=None)
    parser.add_argument(
        "--output-dir", metavar="DIR",
        default=config.OUTPUT_DIR,
    )
    parser.add_argument(
        "--num-cases",
        metavar="N",
        type=int,
        default=config.NUM_CASES_PER_OPERATION,
        help="Her operasyon için üretilecek testcase sayısı.",
    )
    parser.add_argument(
        "--max-tests",
        metavar="N",
        type=int,
        default=None,
        help="Üretilen toplam testcase sayısını sınırlar (opsiyonel).",
    )
    parser.add_argument(
        "--generators",
        metavar="LIST",
        default=None,
        help=(
            "Kullanılacak generator'lar, virgülle ayrılmış "
            "(örn. 'traditional,groq' veya 'openai:gpt-4.1,claude:claude-haiku-4-5'). "
            "Sağlayıcı adı tek başına verilirse o sağlayıcının tüm modelleri seçilir. "
            "Belirtilmezse tüm generator'lar kullanılır."
        ),
    )
    parser.add_argument(
        "--endpoints",
        metavar="LIST",
        default=None,
        help=(
            "--curl-file / --openapi-url yerine doğrudan 'METHOD /path' çiftleri, "
            "virgülle ayrılmış (örn. 'GET /status/200,POST /post'). "
            "--base-url ile birlikte kullanılmalıdır."
        ),
    )
    parser.add_argument(
        "--tests-per-generator",
        metavar="N",
        type=int,
        default=None,
        help=(
            "Her generator'in TOPLAM uretecegi test sayisi (orn. 150). Verilirse "
            "--num-cases yok sayilir; sayi operasyonlara (ve LLM'lerde prompt "
            "variant'lara) bolunur, boylece her generator etiketi tam N satir uretir. "
            "Tam bolunmezse kalan operasyonlara dagitilir; dagitilamazsa kosu durur."
        ),
    )
    parser.add_argument(
        "--resume",
        metavar="RUN_ID",
        default=None,
        help=(
            "Yarida kalmis bir kosuyu surdurur. RUN_ID, <output-dir>/.checkpoints/ "
            "altindaki klasor adidir (orn. run_20260926_174530). Tamamlanmis uretim "
            "gorevleri ve calistirilmis testler tekrarlanmaz."
        ),
    )
    parser.add_argument(
        "--checkpoint-every",
        metavar="N",
        type=int,
        default=DEFAULT_FLUSH_EVERY,
        help=f"Kac satirda bir checkpoint diske yazilsin (varsayilan {DEFAULT_FLUSH_EVERY}).",
    )
    parser.add_argument(
        "--retry-infra-fallback",
        action="store_true",
        help=(
            "--resume ile birlikte: ALTYAPI kaynakli fallback iceren gorevleri "
            "(429, kota, timeout, ag) bir kez yeniden kosar. Icerik kaynakli "
            "fallback'e DOKUNMAZ. Gorev basina en fazla 1 yeniden kosu yapilir "
            "ve gorevin tum eski satirlari degistirilir (kismi birlestirme yok)."
        ),
    )
    parser.add_argument(
        "--confirm-paid-retry",
        action="store_true",
        help=(
            "Ucretli saglayicilarda (OpenAI, Claude) yeniden kosuya ACIK ONAY. "
            "Bu bayrak verilmezse ucretli gorevler yeniden kosulmaz; tahmin "
            "gosterilir ve kosu durur."
        ),
    )
    parser.add_argument(
        "--no-call-ledger",
        action="store_true",
        help="Cagri defterini kapatir (ham yanit/istek/token kaydi yazilmaz).",
    )
    parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Checkpoint yazmayi tamamen kapatir (kisa/deneme kosulari icin).",
    )
    parser.add_argument(
        "--prompt-variant",
        choices=list(config.PROMPT_VARIANTS.keys()) + ["both"],
        default="both",
        help="Kullanılacak prompt variant(lar)ı: 'basic', 'edge_focused', veya 'both' (varsayılan).",
    )
    ns = parser.parse_args()
    if ns.generators:
        try:
            ns.selected_generators = _parse_cli_generators(ns.generators)
        except ValueError as e:
            parser.error(str(e))
    else:
        ns.selected_generators = None  # Tümünü kullan
    ns.num_cases = config.normalize_num_cases(ns.num_cases)
    return ns


def _parse_cli_generators(spec: str) -> list[str]:
    """'traditional,groq' ya da 'openai:gpt-4.1' gibi virgülle ayrılmış generator
    seçimini GENERATOR_REGISTRY anahtar listesine çevirir. Sağlayıcı adı tek
    başına verilirse (örn. 'groq') o sağlayıcının tüm modelleri eklenir.
    """
    known_providers = {"traditional", "openai", "gemini", "claude", "groq"}
    resolved: list[str] = []
    unknown: list[str] = []
    for raw in spec.split(","):
        token = raw.strip()
        if not token:
            continue
        if token in GENERATOR_REGISTRY:
            if token not in resolved:
                resolved.append(token)
            continue
        provider = token.lower()
        if provider in known_providers:
            matches = [k for k in GENERATOR_REGISTRY if k == provider or k.startswith(f"{provider}:")]
            if not matches:
                unknown.append(token)
                continue
            resolved.extend(k for k in matches if k not in resolved)
            continue
        unknown.append(token)

    if unknown:
        valid_keys = ", ".join(sorted(GENERATOR_REGISTRY.keys()))
        raise ValueError(
            f"Bilinmeyen generator(lar): {', '.join(unknown)}. "
            f"Geçerli sağlayıcı adları: traditional, openai, gemini, claude, groq "
            f"(tek başına verilirse tüm modelleri seçer) veya tam anahtar (örn. openai:gpt-4.1). "
            f"Kayıtlı anahtarlar: {valid_keys}"
        )
    if not resolved:
        raise ValueError("--generators boş bir seçim üretti; en az bir generator belirtin.")
    return resolved


def _parse_cli_endpoints(spec: str) -> list[ApiOperation]:
    """'GET /status/200,POST /post' gibi virgülle ayrılmış 'METHOD /path' çiftlerinden
    minimal ApiOperation listesi üretir (şema/parametre bilgisi olmadan).
    """
    ops: list[ApiOperation] = []
    for idx, raw in enumerate(spec.split(","), start=1):
        entry = raw.strip()
        if not entry:
            continue
        parts = entry.split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(
                f"Geçersiz endpoint tanımı: '{entry}'. Beklenen format: 'METHOD /path' (örn. 'GET /status/200')."
            )
        method, path = parts[0].upper(), parts[1].strip()
        if not path.startswith("/"):
            raise ValueError(f"Geçersiz path: '{path}' — '/' ile başlamalı.")
        ops.append(
            ApiOperation(
                op_id=f"EP{idx}",
                method=method,
                path=path,
                summary=f"{method} {path}",
                description="",
            )
        )
    if not ops:
        raise ValueError("--endpoints boş bir liste üretti; en az bir 'METHOD /path' girin.")
    return ops


# ── Generator builder ────────────────────────────────────────────────────────

def _build_llm_generators(selected_keys: list = None, variant_filter: str = "both") -> list:
    """
    Yalnizca LLM generator tuple'lari doner: (instance, variant_name, variant_desc)
    - LLM: her model × seçili prompt_variant(lar)ı (basic + edge_focused, ya da tek biri)
    - Traditional BURADA URETILMEZ: main() icinde ayrica calistiriliyor. Buraya da
      eklenirse ayni operasyonlar icin iki kez kosar ve birebir ayni tc_id'leri
      uretir (duplicate satir + bozuk diversity metrikleri).
    """
    generators = []
    for key, (cls, model, _provider) in GENERATOR_REGISTRY.items():
        if selected_keys is not None and key not in selected_keys:
            continue
        if key == "traditional":
            continue
        for v_name, v_desc in config.PROMPT_VARIANTS.items():
            if variant_filter != "both" and v_name != variant_filter:
                continue
            generators.append((cls(model), v_name, v_desc["focus"]))
    return generators


def _task_provider(task_key: str) -> str:
    """'OpenAIGenerator:gpt-4.1|basic' -> 'OpenAI'"""
    class_name = task_key.split(":", 1)[0]
    return class_name[:-len("Generator")] if class_name.endswith("Generator") else class_name


def _task_is_paid(task_key: str) -> bool:
    return _task_provider(task_key) in config.PAID_PROVIDERS


def _infra_fallback_candidates(task_records: dict) -> list:
    """YALNIZCA altyapi kaynakli fallback iceren ve henuz yeniden kosulmamis gorevler.

    'karma' (hem altyapi hem icerik) gorevler BILEREK DISARIDA BIRAKILIR: icerik
    kaynakli fallback'e ikinci sans vermek, modelin kotu ciktisini eleyip iyisini
    saklamak demektir — bu bir SECILIM YANLILIGIDIR. Bkz. _mixed_origin_tasks.
    """
    candidates = []
    for task_key, record in sorted(task_records.items()):
        if record.get("failure_origin") != "altyapi":
            continue
        if int(record.get("retry_count") or 0) >= 1:
            continue  # gorev basina en fazla 1 yeniden kosu
        candidates.append((task_key, record))
    return candidates


def _paid_retry_cost_note(call_ledger, paid_task_keys: list, extra_calls: int) -> str:
    """Yeniden kosunun DOLAR tahmini (B4(f)).

    Tahmin, onceki kosunun DEFTERINDEKI olculmus ortalama cagri maliyetinden
    turetilir — uydurma bir birim fiyattan degil. Fiyat tablosu onaylanmadiysa
    ya da defterde fiyatlanmis cagri yoksa bunu acikca soyler.
    """
    if not pricing.price_table_ready():
        return "(Dolar tahmini YOK: fiyat tablosu henuz onaylanmadi.)"
    try:
        records = CallLedger.load(call_ledger.path)
    except Exception:  # noqa: BLE001 - tahmin yoklugu kosuyu bozmamali
        records = []
    costs = [
        float(r.get("cost_usd_billed") or 0.0)
        for r in records
        if r.get("pricing_available") and r.get("call_type") != "fallback"
    ]
    if not costs:
        return "(Dolar tahmini YOK: defterde fiyatlanmis cagri bulunamadi.)"
    mean = sum(costs) / len(costs)
    return (
        f"Tahmini ek maliyet: {extra_calls} cagri x ~${mean:.4f} (onceki kosunun "
        f"{len(costs)} cagrisindan olculen ortalama) = ~${extra_calls * mean:.2f}. "
        f"Repair cagrilari bu tahminin DISINDADIR; gercek tutar daha yuksek olabilir."
    )


EXIT_FAIL_CLOSED = 3
EXIT_FREE_TIER_UNATTESTED = 4
EXIT_MISSING_RATE_LIMITS = 5


class _FailClosed(RuntimeError):
    """Ucretli kosu on kosullari saglanmadi; kosu baslatilmaz."""


class _FreeTierUnattested(RuntimeError):
    """FREE_ONLY saglayici icin free-tier beyani yok; kosu baslatilmaz."""


def _free_only_providers_in(llm_generators: list) -> set:
    """Secili generator'lar arasindaki FREE_ONLY saglayicilar."""
    return {
        type(gen).__name__.removesuffix("Generator")
        for gen, _v, _d in llm_generators
        if type(gen).__name__.removesuffix("Generator") in config.FREE_ONLY_PROVIDERS
    }


def _quota_block_until(task_records: dict, task_key: str):
    """Gorev gunluk kota yuzunden hala bekliyorsa bitis ani, degilse None.

    Gunlerce beklenmez: kosu bu gorevleri ATLAR ve sonunda raporlar.
    """
    record = task_records.get(task_key) or {}
    value = record.get("next_available_at")
    if not value:
        return None
    try:
        moment = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    now = datetime.now(moment.tzinfo) if moment.tzinfo else datetime.now()
    return moment if moment > now else None


class _MissingRateLimits(RuntimeError):
    """FREE_ONLY model icin yayimlanan limit degerleri yok; kosu baslatilmaz."""


def _rate_limit_check(llm_generators: list, limits: dict) -> list:
    """FREE_ONLY her modelin limit girdisi olmali.

    Limit bilinmeden free tier'da kosmak, kotanin ne zaman bitecegini bilmeden
    kosmaktir: gunluk kota tukenirse o model o gun bir daha uretmez.
    """
    blockers: list[str] = []
    for gen, _v, _d in llm_generators:
        provider = type(gen).__name__.removesuffix("Generator")
        if provider not in config.FREE_ONLY_PROVIDERS:
            continue
        model = getattr(gen, "model", "")
        if (provider, model) not in limits:
            blockers.append(f"{provider}/{model}")
    if not blockers:
        return []
    gaps = rate_limiter.missing_fields()
    detail = "; ".join(
        f"{p}/{m}: {', '.join(fields)}" for (p, m), fields in sorted(gaps.items())
    )
    return [
        f"free_tier_limits.json'da limit girdisi eksik olan FREE_ONLY model(ler): "
        f"{', '.join(sorted(set(blockers)))}. Bos alanlar -> {detail or 'yok'}"
    ]


def _free_tier_attestation_check(llm_generators: list) -> list:
    """FREE_ONLY generator'lar icin beyan on kontrolu.

    Kod, bir hesabin free tier'da olup olmadigini olcemez (bkz. attestation
    modulu). Beyan yoksa bu generator'lar hic kosmaz — yanlislikla faturali
    bir projeyle kosmak sessizce para harcatirdi.
    """
    selected = _free_only_providers_in(llm_generators)
    if not selected:
        return []
    missing = attestation.missing_for(selected)
    if not missing:
        return []
    return [
        f"free-tier beyani eksik: {', '.join(missing)} "
        f"({attestation.ENV_VAR} degiskeninde bildirilmeli). {attestation.GUIDANCE}"
    ]


def _fail_closed_check(llm_generators: list, budget_guard) -> list:
    """UCRETLI generator'lar icin FAIL-CLOSED on kontrol.

    Para harcayan bir generator, harcamasi OLCULEMEYECEK durumdayken
    kosmamalidir. Iki kosul da engelleyicidir:
      (a) modelin pricing.PRICE_TABLE'da girdisi yok  -> maliyet hesaplanamaz,
      (b) butce sigortasi armed=False                 -> esikler tetiklenmez.
    Ucretsiz generator'lar bu kontrolden etkilenmez.

    Doner: engelleyici sebeplerin listesi (bos ise kosulabilir).
    """
    blockers: list[str] = []
    paid_models = sorted({
        getattr(gen, "model", "")
        for gen, _v, _d in llm_generators
        if type(gen).__name__.removesuffix("Generator") in config.PAID_PROVIDERS
    })
    if not paid_models:
        return blockers

    missing = [model for model in paid_models if model not in pricing.PRICE_TABLE]
    if missing:
        blockers.append(
            f"fiyat tablosunda girdisi olmayan UCRETLI model(ler): {', '.join(missing)}"
        )
    if not budget_guard.armed:
        blockers.append(
            "butce sigortasi ATIL (armed=False): $%.0f/$%.0f/$%.0f esikleri tetiklenmez"
            % (config.BUDGET_THRESHOLDS["warn"], config.BUDGET_THRESHOLDS["hard_warn"],
               config.BUDGET_THRESHOLDS["stop"])
        )
    return blockers


def _mixed_origin_tasks(task_records: dict) -> list:
    """Hem altyapi hem icerik kaynakli basarisizlik iceren gorevler.

    Yeniden KOSULMAZ; yalnizca raporlanir ki kosu sonrasi analizde bu gorevlerin
    fallback payinin bir kismi altyapi kaynakli oldugu bilinsin.
    """
    return [(k, r) for k, r in sorted(task_records.items()) if r.get("failure_origin") == "karma"]


def _split_total(total: int, parts: int) -> list[int]:
    """`total`'i `parts` parcaya bolerek dagitir; toplam TAM `total` kalir.

    Tam bolunmedigi durumda kalan, ilk parcalara birer birer eklenir.
    """
    if parts <= 0:
        raise ValueError("Bolunecek parca sayisi pozitif olmali.")
    base, remainder = divmod(total, parts)
    return [base + (1 if index < remainder else 0) for index in range(parts)]


def _allocate_cases_per_operation(total: int, operations: list) -> dict:
    """Bir generator gorevinin `total` testini operasyonlara dagitir."""
    op_count = len(operations)
    if op_count == 0:
        raise ValueError("Dagitilacak operasyon yok.")
    if total < op_count:
        raise ValueError(
            f"Gorev basina {total} test {op_count} operasyona dagitilamiyor: "
            f"her operasyona en az 1 test dusmeli."
        )
    shares = _split_total(total, op_count)
    return {op.op_id: share for op, share in zip(operations, shares)}


def _build_balanced_allocation(tests_per_generator: int, operations: list, variants: list) -> dict:
    """K1: her generator etiketi TAM `tests_per_generator` satir uretsin.

    LLM generator'lari her prompt variant icin ayri bir gorev kosar ama CSV'de
    tek etikette birlesir; bu yuzden hedef once variant'lara, sonra
    operasyonlara bolunur. Traditional'in variant'i yoktur.

    Doner: {"traditional": {op_id: n}, "llm": {variant_name: {op_id: n}}}
    """
    variant_count = len(variants)
    if variant_count == 0:
        raise ValueError("En az bir prompt variant secilmeli.")
    op_count = len(operations)
    minimum = variant_count * op_count
    if tests_per_generator < minimum:
        raise ValueError(
            f"--tests-per-generator {tests_per_generator} cok dusuk: "
            f"{variant_count} prompt variant x {op_count} operasyon = en az {minimum} gerekir "
            f"(her variant-operasyon ciftine en az 1 test dusmeli)."
        )

    variant_totals = _split_total(tests_per_generator, variant_count)
    return {
        "traditional": _allocate_cases_per_operation(tests_per_generator, operations),
        "llm": {
            variant: _allocate_cases_per_operation(variant_total, operations)
            for variant, variant_total in zip(variants, variant_totals)
        },
    }


def _generation_task_key(gen_instance, variant_name: str) -> str:
    """Bir uretim gorevi icin checkpoint'te kullanilan kararli anahtar."""
    model = getattr(gen_instance, "model", "")
    return f"{type(gen_instance).__name__}:{model}|{variant_name}"


def _parse_cli_headers(header_list: list) -> dict:
    result = {}
    for h in header_list:
        key, sep, value = h.partition(":")
        if sep and key.strip():
            result[key.strip()] = value.strip()
    return result


def _parse_cli_cookies(cookie_str: str) -> dict:
    result = {}
    for part in cookie_str.split(";"):
        k, _, v = part.partition("=")
        if k.strip():
            result[k.strip()] = v.strip()
    return result


def _resolve_safe_output_dir(output_dir: str) -> str:
    raw = (output_dir or config.OUTPUT_DIR).strip() or config.OUTPUT_DIR
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = config.PROJECT_ROOT / candidate
    resolved = candidate.resolve()
    allowed_roots = [root.resolve() for root in config.ALLOWED_OUTPUT_ROOTS]
    if not any(resolved == root or root in resolved.parents for root in allowed_roots):
        allowed = ", ".join(str(root) for root in allowed_roots)
        raise ValueError(f"Çıktı klasörü izin verilen köklerin altında olmalı: {allowed}")
    return str(resolved)


def _save_cli_run_info(args: argparse.Namespace, operations: list, output_dir: str, selected_keys: list | None) -> str:
    metadata = {
        "job_id": "cli",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": (
            "curl" if args.curl_file
            else "openapi" if args.openapi_url
            else "endpoints" if getattr(args, "endpoints", None)
            else "manual"
        ),
        "base_url": args.base_url,
        "no_run": bool(args.no_run),
        "output_dir": output_dir,
        "selected_generators": selected_keys or ["all"],
        "prompt_variants": dict(config.PROMPT_VARIANTS),
        "num_cases_per_operation": getattr(args, "num_cases", config.NUM_CASES_PER_OPERATION),
        "operation_count": len(operations),
        "operation_ids": [getattr(op, "op_id", "") for op in operations],
        "config_snapshot": {
            "openai_models": config.OPENAI_MODELS,
            "gemini_models": config.GEMINI_MODELS,
            "claude_models": config.CLAUDE_MODELS,
            "groq_models": config.GROQ_MODELS,
            "request_timeout": config.REQUEST_TIMEOUT,
            "retry_max_attempts": config.RETRY_MAX_ATTEMPTS,
            "retry_backoff_seconds": config.RETRY_BACKOFF_SECONDS,
            "max_parallel_workers": config.MAX_PARALLEL_WORKERS,
        },
    }
    path = Path(output_dir) / f"run_info_cli_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    _logger.info("Run metadata kaydedildi: %s", path)
    return str(path)


# ── Ana akış ─────────────────────────────────────────────────────────────────

def main() -> None:
    _configure_logging()
    load_dotenv()

    # Argüman yoksa interaktif wizard
    if len(sys.argv) == 1:
        args = interactive_wizard()
    else:
        args = parse_args()

    try:
        args.output_dir = _resolve_safe_output_dir(args.output_dir)
    except ValueError as e:
        _logger.error("HATA: %s", e)
        sys.exit(1)

    # ── Header / Cookie hazırlığı ────────────────────────────────────────
    extra_headers = _parse_cli_headers(args.headers)
    cookies = _parse_cli_cookies(args.cookie) if args.cookie else {}

    # ── Operasyonları al ─────────────────────────────────────────────────
    base_url = args.base_url

    if args.curl_file:
        operations = []
        derived_base_url = None
        for filepath in args.curl_file:
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    curl_text = f.read()
            except OSError as e:
                _logger.error("HATA: curl dosyası okunamadı (%s): %s", filepath, e)
                sys.exit(1)
            try:
                parsed_list = parse_curl_collection(curl_text)
            except ValueError as e:
                _logger.error("HATA: curl parse edilemedi (%s): %s", filepath, e)
                sys.exit(1)
            for op, op_base_url, curl_headers, curl_cookies in parsed_list:
                if derived_base_url is None:
                    derived_base_url = op_base_url
                elif op_base_url != derived_base_url:
                    _logger.warning("UYARI: %s farklı host (%s), base URL olarak %s kullanılıyor.", op.op_id, op_base_url, derived_base_url)
                cookies = {**curl_cookies, **cookies}
                extra_headers = {**curl_headers, **extra_headers}
                operations.append(op)
        base_url = base_url or derived_base_url
        _logger.info("%d curl operasyonu parse edildi: %s", len(operations), ", ".join(f"{op.method} {op.path}" for op in operations))

    elif args.openapi_url:
        if not base_url:
            _logger.error("HATA: --openapi-url ile birlikte --base-url da verilmeli.")
            sys.exit(1)
        try:
            spec = load_openapi_from_url(
                args.openapi_url,
                headers=extra_headers or None,
                cookies=cookies or None,
            )
        except Exception as e:
            _logger.error("HATA: OpenAPI dokümanı yüklenemedi: %s", e)
            sys.exit(1)
        operations = extract_operations_from_openapi(spec)
        _logger.info("%d operasyon çıkarıldı.", len(operations))

    elif getattr(args, "endpoints", None):
        if not base_url:
            _logger.error("HATA: --endpoints ile birlikte --base-url da verilmeli.")
            sys.exit(1)
        try:
            operations = _parse_cli_endpoints(args.endpoints)
        except ValueError as e:
            _logger.error("HATA: %s", e)
            sys.exit(1)
        _logger.info(
            "%d endpoint tanımlandı: %s",
            len(operations),
            ", ".join(f"{op.method} {op.path}" for op in operations),
        )

    else:
        if not base_url:
            _logger.error("HATA: --base-url veya --curl-file argümanlarından biri zorunludur.")
            sys.exit(1)
        operations = manual_operations_input()
        _logger.info("%d operasyon girildi.", len(operations))

    if not operations:
        _logger.info("Operasyon bulunamadı, program sonlandırılıyor.")
        sys.exit(0)

    print()
    print("=" * 60)
    print("  LLM TEST LAB — Başlıyor")
    print(f"  Base URL : {base_url}")
    print(f"  Çıktı    : {args.output_dir}")
    print("=" * 60)
    print()

    os.makedirs(args.output_dir, exist_ok=True)
    save_operations_csv(operations, args.output_dir)

    # ── Test senaryolarını üret ──────────────────────────────────────────
    all_rows: list = []
    selected_keys = getattr(args, "selected_generators", None)
    num_cases = getattr(args, "num_cases", config.NUM_CASES_PER_OPERATION)
    _save_cli_run_info(args, operations, args.output_dir, selected_keys)

    # ── Checkpoint / resume ──────────────────────────────────────────────
    run_checkpoint = RunCheckpoint(
        args.output_dir,
        run_id=getattr(args, "resume", None),
        flush_every=getattr(args, "checkpoint_every", DEFAULT_FLUSH_EVERY),
        enabled=not getattr(args, "no_checkpoint", False),
    )
    # Cagri defteri: kosu anindaki ham yanit/istek/token bilgisi (CSV'den uretilemez).
    call_ledger = CallLedger(
        args.output_dir,
        run_id=run_checkpoint.run_id,
        enabled=not getattr(args, "no_call_ledger", False),
    )
    if call_ledger.enabled:
        _logger.info("  [defter] cagri kaydi: %s", call_ledger.path)

    # FREE_ONLY hiz/kota limitoru. Limit girdisi olmayan model KOSMAZ.
    limiter_limits = rate_limiter.load_limits()
    run_limiter = rate_limiter.RateLimiter(
        limiter_limits,
        usage_path=Path(args.output_dir) / rate_limiter.USAGE_DIR_NAME
        / run_checkpoint.run_id / "usage.jsonl",
    )
    if limiter_limits:
        _logger.info("  [limitor] %d FREE_ONLY model yonetiliyor:", len(limiter_limits))
        for name, values in run_limiter.effective_limits().items():
            _logger.info(
                "      %s | efektif rpm=%s rpd=%s tpm=%s tpd=%s (guvenlik payi %s, tpm=%s)",
                name, values["effective_rpm"], values["effective_rpd"],
                values["effective_tpm"], values["effective_tpd"],
                values["safety_factor"], values["tpm_counts"],
            )
    else:
        _logger.warning(
            "  [limitor] free_tier_limits.json BOS — FREE_ONLY generator secilirse kosu durur."
        )

    # K6 butce sigortasi — harcamayi DEFTERDEN okur (B4(a)).
    budget_guard = budget_module.BudgetGuard(call_ledger, enabled=call_ledger.enabled)
    if budget_guard.armed:
        _logger.info(
            "  [butce] sigorta aktif — uyari $%.0f / sert uyari $%.0f / DURDURMA $%.0f",
            config.BUDGET_THRESHOLDS["warn"], config.BUDGET_THRESHOLDS["hard_warn"],
            config.BUDGET_THRESHOLDS["stop"],
        )

    task_records = run_checkpoint.task_records()
    retry_counts: dict = {}

    if getattr(args, "retry_infra_fallback", False):
        if not run_checkpoint.resumed:
            _logger.error("HATA: --retry-infra-fallback yalnizca --resume ile kullanilir.")
            sys.exit(1)
        candidates = _infra_fallback_candidates(task_records)
        for key, rec in sorted(task_records.items()):
            if rec.get("failure_origin") == "altyapi" and int(rec.get("retry_count") or 0) >= 1:
                _logger.warning("  [yeniden kosu] %s daha once bir kez yeniden kosuldu, ATLANDI.", key)

        mixed = _mixed_origin_tasks(task_records)
        if mixed:
            _logger.warning(
                "  [yeniden kosu] %d gorev KARMA (altyapi + icerik) kaynakli, YENIDEN KOSULMADI "
                "(icerik kaynakli fallback'e ikinci sans vermek secilim yanliligi olurdu):",
                len(mixed),
            )
            for key, record in mixed:
                _logger.warning(
                    "      %s | karma, yeniden kosulmadi | fallback=%s satir",
                    key, record.get("fallback_cases"),
                )

        if not candidates:
            _logger.info("  [yeniden kosu] altyapi kaynakli fallback iceren yeniden kosulabilir gorev yok.")
        else:
            paid = [(k, r) for k, r in candidates if _task_is_paid(k)]
            free = [(k, r) for k, r in candidates if not _task_is_paid(k)]
            _logger.info("  [yeniden kosu] aday gorev: %d (ucretsiz %d, UCRETLI %d)",
                         len(candidates), len(free), len(paid))
            for key, record in candidates:
                _logger.info(
                    "      %s | saglayici=%s | ucretli=%s | fallback=%s satir",
                    key, _task_provider(key), "EVET" if _task_is_paid(key) else "hayir",
                    record.get("fallback_cases"),
                )
            if paid and not getattr(args, "confirm_paid_retry", False):
                extra_calls = len(paid) * len(operations)
                _logger.error(
                    "\n  UCRETLI yeniden kosu ONAY BEKLIYOR: %d gorev PARA HARCAR.\n"
                    "  Tahmini ek cagri: %d gorev x %d operasyon = %d ana cagri.\n"
                    "  %s\n"
                    "  Onaylamak icin --confirm-paid-retry ekleyin; onaysiz kosu DURDURULDU.",
                    len(paid), len(paid), len(operations), extra_calls,
                    _paid_retry_cost_note(call_ledger, [key for key, _ in paid], extra_calls),
                )
                sys.exit(2)
            for key, record in candidates:
                retry_counts[key] = int(record.get("retry_count") or 0) + 1
                run_checkpoint.revoke_task(key, reason="altyapi kaynakli fallback yeniden kosuluyor")
                task_records.pop(key, None)
            _logger.info("  [yeniden kosu] %d gorev iptal edildi, satirlari degistirilecek.", len(candidates))

    completed_tasks = set(task_records)
    if run_checkpoint.resumed:
        resumed_rows = run_checkpoint.load_generated_rows()
        all_rows.extend(resumed_rows)
        _logger.info(
            "  [resume] %s: %d satir, %d tamamlanmis uretim gorevi yuklendi.",
            run_checkpoint.run_id, len(resumed_rows), len(completed_tasks),
        )
    elif run_checkpoint.enabled:
        _logger.info("  [checkpoint] run_id=%s — surdurmek icin: --resume %s",
                     run_checkpoint.run_id, run_checkpoint.run_id)

    # ── K1: generator basina dengeli tahsis ────────────────────────────
    prompt_variant_filter = getattr(args, "prompt_variant", "both")
    active_variants = [
        name for name in config.PROMPT_VARIANTS
        if prompt_variant_filter == "both" or name == prompt_variant_filter
    ]
    tests_per_generator = getattr(args, "tests_per_generator", None)
    allocation = None
    if tests_per_generator:
        try:
            allocation = _build_balanced_allocation(tests_per_generator, operations, active_variants)
        except ValueError as exc:
            _logger.error("HATA: dengeli tahsis kurulamadi — %s", exc)
            sys.exit(1)
        llm_per_variant = {v: sum(counts.values()) for v, counts in allocation["llm"].items()}
        _logger.info(
            "  [denge] generator basina hedef %d | Traditional %d | LLM variant dagilimi %s (toplam %d)",
            tests_per_generator,
            sum(allocation["traditional"].values()),
            llm_per_variant,
            sum(llm_per_variant.values()),
        )

    generation_started_at = time.perf_counter()
    budget_stopped = False
    quota_blocked: list = []
    executed_rows: list = []
    run_failure: Exception | None = None
    failed_generations: list[dict] = []

    try:
        # Geleneksel şablon
        if selected_keys is None or "traditional" in selected_keys:
            if "traditional" in completed_tasks:
                _logger.info("  [Geleneksel] checkpoint'te tamamlanmis, atlandi.")
            else:
                trad_gen = TraditionalGenerator()
                trad_rows = trad_gen.generate(
                    operations, "", "", num_cases,
                    num_cases_by_op=(allocation["traditional"] if allocation else None),
                )
                all_rows.extend(trad_rows)
                run_checkpoint.record_generated(trad_rows, "traditional")
                run_checkpoint.mark_task_done("traditional", len(trad_rows), fallback_cases=0)
                _logger.info("  [Geleneksel] %d senaryo üretildi.", len(trad_rows))

        # LLM tabanlı generator'lar — dış döngü paralelliği (generator başına bir thread)
        llm_generators = _build_llm_generators(selected_keys, prompt_variant_filter)

        # FAIL-CLOSED: olculemeyen harcama yapilmaz. Kontrol uretim baslamadan
        # once, yani tek kurus harcanmadan once yapilir.
        unattested = _free_tier_attestation_check(llm_generators)
        if unattested:
            for reason in unattested:
                _logger.error("  [FREE-TIER BEYAN] %s", reason)
            _logger.error(
                "  FREE_ONLY generator'lar KOSULMADI. Saglayici arayuzunden dogrulayip "
                "%s degiskenini .env dosyaniza ekleyin.", attestation.ENV_VAR,
            )
            raise _FreeTierUnattested("; ".join(unattested))

        rate_limit_blockers = _rate_limit_check(llm_generators, limiter_limits)
        if rate_limit_blockers:
            for reason in rate_limit_blockers:
                _logger.error("  [LIMIT EKSIK] %s", reason)
            _logger.error(
                "  FREE_ONLY generator'lar KOSULMADI. Degerleri saglayicinin kendi "
                "limit sayfasindan alip %s dosyasina girin.", rate_limiter.LIMITS_FILE,
            )
            raise _MissingRateLimits("; ".join(rate_limit_blockers))

        fail_closed = _fail_closed_check(llm_generators, budget_guard)
        if fail_closed:
            for reason in fail_closed:
                _logger.error("  [FAIL-CLOSED] %s", reason)
            _logger.error(
                "  UCRETLI generator'lar KOSULMADI. Duzeltip tekrar deneyin, ya da "
                "--generators ile yalnizca ucretsiz saglayicilari secin."
            )
            raise _FailClosed("; ".join(fail_closed))
        with ThreadPoolExecutor(max_workers=config.MAX_PARALLEL_GENERATORS) as executor:
            future_to_task = {}
            for gen_instance, v_name, v_desc in llm_generators:
                gen_label = f"{type(gen_instance).__name__} ({v_name})"
                task_key = _generation_task_key(gen_instance, v_name)
                if task_key in completed_tasks:
                    _logger.info("  [%s] checkpoint'te tamamlanmis, atlandi.", gen_label)
                    continue
                blocked_until = _quota_block_until(task_records, task_key)
                if blocked_until is not None:
                    _logger.error(
                        "  [%s] ATLANDI — gunluk kota penceresi henuz acilmadi (%s).",
                        gen_label, blocked_until.isoformat(timespec="seconds"),
                    )
                    quota_blocked.append((gen_label, blocked_until))
                    continue
                # Pre-flight: anahtar yoksa gorevi hic thread'e verme (app.py ile ayni desen).
                # Aksi halde her operasyon ayri ayri _get_client()'ta patlar ve log dolar.
                try:
                    gen_instance._get_client()
                except RuntimeError as exc:
                    _logger.warning("  [%s] ATLANDI — %s", gen_label, redact_secrets(str(exc)))
                    continue
                gen_instance._call_ledger = call_ledger
                gen_instance._budget_guard = budget_guard
                gen_instance._rate_limiter = run_limiter
                if budget_stopped:
                    _logger.error("  [butce] %s ATLANDI — sert esik asildi.", gen_label)
                    continue
                _logger.info("  [%s] üretiliyor...", gen_label)
                future = executor.submit(
                    gen_instance.generate,
                    operations,
                    variant_name=v_name,
                    variant_desc=v_desc,
                    num_cases=num_cases,
                    num_cases_by_op=(allocation["llm"].get(v_name) if allocation else None),
                )
                # v_name ve gen_instance dongu degiskeni; as_completed sirasinda
                # SON degerleri degil, bu gorevin kendi degerleri gerekir.
                future_to_task[future] = (gen_label, task_key, gen_instance, v_name)

            for future in as_completed(future_to_task):
                gen_label, task_key, gen_instance, task_variant = future_to_task[future]
                try:
                    rows = future.result()
                    all_rows.extend(rows)
                    run_checkpoint.record_generated(rows, task_key, epoch=retry_counts.get(task_key, 0))
                    summaries = getattr(gen_instance, "_generation_summaries", []) or []
                    fallback_total = sum(int(item.get("fallback_cases") or 0) for item in summaries)
                    origins = {item.get("fallback_origin") for item in summaries if item.get("fallback_origin")}
                    if getattr(gen_instance, "_infra_failures", 0):
                        # Gorev hic satir uretemeden coktuyse fallback de yoktur;
                        # yine de ALTYAPI kaynakli sayilir ve yeniden kosulabilir.
                        origins.add("altyapi")
                    origin = "karma" if len(origins) > 1 else next(iter(origins), None)
                    # Gunluk kota bittiyse bu gorev 'altyapi' sayilir ve
                    # yeniden denenebilecegi EN ERKEN an kaydedilir.
                    provider = type(gen_instance).__name__.removesuffix("Generator")
                    model = getattr(gen_instance, "model", "")
                    next_available = None
                    if run_limiter.daily_quota_exhausted(provider, model):
                        origin = "altyapi"
                        next_available = run_limiter.stats()["daily_quota_exhausted"].get(
                            f"{provider}/{model}"
                        )
                        _logger.error(
                            "  [%s] GUNLUK KOTA bitti — yeniden denenebilecegi an: %s",
                            gen_label, next_available or "bilinmiyor",
                        )
                    run_checkpoint.mark_task_done(
                        task_key, len(rows),
                        fallback_cases=fallback_total,
                        failure_origin=origin,
                        retry_count=retry_counts.get(task_key, 0),
                        next_available_at=next_available,
                    )
                    if getattr(gen_instance, "_budget_stopped", False):
                        budget_stopped = True
                    _logger.info("  [%s] %d senaryo üretildi.", gen_label, len(rows))
                except budget_module.BudgetExceeded as exc:
                    # Butce asimi tek bir generator'in hatasi degil, kosu genelinde
                    # bir DURDURMA karari: diger gorevler de calistirilmaz.
                    budget_stopped = True
                    _logger.error("  [butce] %s", exc)
                except Exception as exc:  # noqa: BLE001 - tek generator tum kosuyu oldurmemeli
                    # Bu gorevin uretmesi beklenen satir sayisi = kaybedilen satir sayisi.
                    if allocation and task_variant in allocation["llm"]:
                        lost_rows = sum(allocation["llm"][task_variant].values())
                    else:
                        lost_rows = num_cases * len(operations)
                    failed_generations.append({
                        "generator": gen_label,
                        "error_type": type(exc).__name__,
                        "error": redact_secrets(str(exc)),
                        "lost_rows": lost_rows,
                    })
                    _logger.error(
                        "  [%s] BASARISIZ — %s: %s | ~%d satir uretilemedi "
                        "(diger generator'lar devam ediyor)",
                        gen_label, type(exc).__name__, redact_secrets(str(exc)), lost_rows,
                    )

        if quota_blocked:
            _logger.error(
                "\n── GUNLUK KOTA BEKLEMESI ── %d gorev atlandi; gunlerce beklenmedi. "
                "Pencere acildiktan sonra --resume ile devam edin:", len(quota_blocked),
            )
            for label, moment in quota_blocked:
                _logger.error("  %s | en erken %s", label, moment.isoformat(timespec="seconds"))

        if budget_stopped:
            _logger.error(
                "\n── BUTCE DURDURMASI ── Defter toplami $%.2f, sert esik $%.2f. "
                "Uretim erken kesildi; o ana kadarki satirlar CSV'ye yaziliyor.",
                budget_guard.spend(), config.BUDGET_THRESHOLDS["stop"],
            )

        if failed_generations:
            total_lost = sum(item["lost_rows"] for item in failed_generations)
            _logger.error(
                "\n── BASARISIZ GENERATOR OZETI — %d gorev, ~%d satir eksik ──",
                len(failed_generations), total_lost,
            )
            for item in failed_generations:
                _logger.error(
                    "  %s | %s | ~%d satir | %s",
                    item["generator"], item["error_type"], item["lost_rows"], item["error"],
                )

        generation_elapsed = time.perf_counter() - generation_started_at
        _logger.info("  Üretim süresi: %.1f saniye (%.1f dakika).", generation_elapsed, generation_elapsed / 60)

        max_tests = getattr(args, "max_tests", None)
        if max_tests and len(all_rows) > max_tests:
            all_rows = all_rows[:max_tests]
            _logger.info("  Test sayısı --max-tests ile %d'e sınırlandı.", max_tests)

        _logger.info("\nToplam %d test senaryosu üretildi.", len(all_rows))

        # ── Testleri çalıştır ───────────────────────────────────────────────
        if args.no_run:
            _logger.info("Testler çalıştırılmıyor (--no-run / wizard seçimi).")
            executed_rows = all_rows
        else:
            already_executed = run_checkpoint.load_executed_rows()
            done_identities = {row_identity(row) for row in already_executed}
            pending_rows = [row for row in all_rows if row_identity(row) not in done_identities]
            if already_executed:
                _logger.info(
                    "  [resume] %d test zaten çalıştırılmış, %d test kaldı.",
                    len(already_executed), len(pending_rows),
                )
            executed_rows = already_executed + run_testcases(
                base_url,
                pending_rows,
                auth_token=args.auth_token,
                extra_headers=extra_headers or None,
                cookies=cookies or None,
                on_result=run_checkpoint.record_executed,
            )

    except Exception as exc:  # noqa: BLE001 - kismi sonuc her halukarda diske yazilmali
        run_failure = exc
        _logger.error(
            "  [KRITIK] Kosu beklenmedik sekilde sonlandi: %s: %s",
            type(exc).__name__, redact_secrets(str(exc)),
        )
    finally:
        run_checkpoint.flush()
        call_ledger.flush()

    # Yurutmeye hic gelinemediyse en azindan uretilen satirlari raporla.
    if not executed_rows:
        executed_rows = all_rows

    # ── Raporla ────────────────────────────────────────────────────────
    save_results_csv(executed_rows, args.output_dir)

    # Icerik tekrari (K1): ayni istegi ayni beklentiyle tekrar eden satirlar.
    if executed_rows:
        _logger.info("\n── ICERIK TEKRARI (tc_id ve title haric imza) ──")
        for line in format_repetition_table(compute_repetition_stats(executed_rows)):
            _logger.info("  %s", line)

    if not args.no_run:
        metrics = compute_generator_metrics(executed_rows)
        save_generator_metrics_csv(metrics, args.output_dir)
        print_summary_table(executed_rows)

    limiter_stats = run_limiter.stats()
    if limiter_stats["total_wait_ms"] or any(limiter_stats["reactive_429"].values()):
        _logger.info(
            "\n── LIMITOR ── toplam bekleme %.1f sn · reaktif 429: %s",
            limiter_stats["total_wait_ms"] / 1000.0,
            {k: v for k, v in limiter_stats["reactive_429"].items() if v} or "yok",
        )

    if call_ledger.enabled and call_ledger.write_errors:
        _logger.error(
            "\nUYARI: cagri defterine %d kayit YAZILAMADI (kosu etkilenmedi, "
            "uretilen satirlar korundu). Ilk hata: %s",
            len(call_ledger.write_errors), call_ledger.write_errors[0],
        )

    if failed_generations:
        total_lost = sum(item["lost_rows"] for item in failed_generations)
        _logger.error(
            "\nUYARI: %d generator basarisiz oldu, ~%d satir eksik. Detay icin "
            "yukaridaki BASARISIZ GENERATOR OZETI bolumune bakin.",
            len(failed_generations), total_lost,
        )

    if isinstance(run_failure, (_MissingRateLimits, rate_limiter.ImpossibleRequest)):
        _logger.error(
            "\nKOSU DURDURULDU — hiz/kota limiti sorunu (%s). %d satir diske yazildi: %s/",
            run_failure, len(executed_rows), args.output_dir,
        )
        sys.exit(EXIT_MISSING_RATE_LIMITS)

    if isinstance(run_failure, _FreeTierUnattested):
        _logger.error(
            "\nKOSU DURDURULDU — free-tier beyani yok (%s). %d satir diske yazildi: %s/",
            run_failure, len(executed_rows), args.output_dir,
        )
        sys.exit(EXIT_FREE_TIER_UNATTESTED)

    if isinstance(run_failure, _FailClosed):
        # Ucretli on kosullar saglanmadi: ucretsiz/sablon satirlar yazildi ama
        # kosu BASARISIZ sayilir ve ayri bir cikis koduyla biter.
        _logger.error(
            "\nKOSU FAIL-CLOSED ile durduruldu (%s). %d satir diske yazildi: %s/",
            run_failure, len(executed_rows), args.output_dir,
        )
        sys.exit(EXIT_FAIL_CLOSED)

    if run_failure is not None:
        _logger.error(
            "\nKosu HATAYLA bitti, ancak %d satir diske yazildi: %s/",
            len(executed_rows), args.output_dir,
        )
        sys.exit(1)

    _logger.info("\nTamamlandı. Çıktılar: %s/", args.output_dir)


if __name__ == "__main__":
    main()
