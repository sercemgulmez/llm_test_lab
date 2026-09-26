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

from dotenv import load_dotenv

# Load project-local secrets without overriding explicitly exported variables.
load_dotenv(Path(__file__).resolve().parent / ".env")

import config
from checkpoint import DEFAULT_FLUSH_EVERY, RunCheckpoint, row_identity
from models import ApiOperation
from parsers.openapi import load_openapi_from_url, extract_operations_from_openapi, manual_operations_input
from parsers.curl_parser import parse_curl_collection
from security.redaction import redact_secrets
from generators import TraditionalGenerator, GENERATOR_REGISTRY
from runner import run_testcases
from reporters.csv_reporter import (
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
    completed_tasks = run_checkpoint.completed_tasks()
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

    generation_started_at = time.perf_counter()
    executed_rows: list = []
    run_failure: Exception | None = None

    try:
        # Geleneksel şablon
        if selected_keys is None or "traditional" in selected_keys:
            if "traditional" in completed_tasks:
                _logger.info("  [Geleneksel] checkpoint'te tamamlanmis, atlandi.")
            else:
                trad_gen = TraditionalGenerator()
                trad_rows = trad_gen.generate(operations, "", "", num_cases)
                all_rows.extend(trad_rows)
                run_checkpoint.record_generated(trad_rows, "traditional")
                run_checkpoint.mark_task_done("traditional", len(trad_rows))
                _logger.info("  [Geleneksel] %d senaryo üretildi.", len(trad_rows))

        # LLM tabanlı generator'lar — dış döngü paralelliği (generator başına bir thread)
        prompt_variant_filter = getattr(args, "prompt_variant", "both")
        llm_generators = _build_llm_generators(selected_keys, prompt_variant_filter)
        with ThreadPoolExecutor(max_workers=config.MAX_PARALLEL_GENERATORS) as executor:
            future_to_task = {}
            for gen_instance, v_name, v_desc in llm_generators:
                gen_label = f"{type(gen_instance).__name__} ({v_name})"
                task_key = _generation_task_key(gen_instance, v_name)
                if task_key in completed_tasks:
                    _logger.info("  [%s] checkpoint'te tamamlanmis, atlandi.", gen_label)
                    continue
                # Pre-flight: anahtar yoksa gorevi hic thread'e verme (app.py ile ayni desen).
                # Aksi halde her operasyon ayri ayri _get_client()'ta patlar ve log dolar.
                try:
                    gen_instance._get_client()
                except RuntimeError as exc:
                    _logger.warning("  [%s] ATLANDI — %s", gen_label, redact_secrets(str(exc)))
                    continue
                _logger.info("  [%s] üretiliyor...", gen_label)
                future = executor.submit(
                    gen_instance.generate,
                    operations,
                    variant_name=v_name,
                    variant_desc=v_desc,
                    num_cases=num_cases,
                )
                future_to_task[future] = (gen_label, task_key)

            for future in as_completed(future_to_task):
                gen_label, task_key = future_to_task[future]
                try:
                    rows = future.result()
                    all_rows.extend(rows)
                    run_checkpoint.record_generated(rows, task_key)
                    run_checkpoint.mark_task_done(task_key, len(rows))
                    _logger.info("  [%s] %d senaryo üretildi.", gen_label, len(rows))
                except Exception as exc:  # noqa: BLE001 - tek generator tum kosuyu oldurmemeli
                    _logger.error(
                        "  [%s] BASARISIZ — %s: %s (diger generator'lar devam ediyor)",
                        gen_label, type(exc).__name__, redact_secrets(str(exc)),
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

    # Yurutmeye hic gelinemediyse en azindan uretilen satirlari raporla.
    if not executed_rows:
        executed_rows = all_rows

    # ── Raporla ────────────────────────────────────────────────────────
    save_results_csv(executed_rows, args.output_dir)

    if not args.no_run:
        metrics = compute_generator_metrics(executed_rows)
        save_generator_metrics_csv(metrics, args.output_dir)
        print_summary_table(executed_rows)

    if run_failure is not None:
        _logger.error(
            "\nKosu HATAYLA bitti, ancak %d satir diske yazildi: %s/",
            len(executed_rows), args.output_dir,
        )
        sys.exit(1)

    _logger.info("\nTamamlandı. Çıktılar: %s/", args.output_dir)


if __name__ == "__main__":
    main()
