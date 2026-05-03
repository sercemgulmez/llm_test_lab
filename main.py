"""
LLM Test Lab — Otomatik API Test Senaryosu Üretim ve Yürütme Aracı

Kullanım:
    python main.py                          # İnteraktif wizard
    python main.py --curl-file turkcell.txt # Doğrudan argümanlar
    python main.py --help                   # Tüm seçenekler
"""

import argparse
import os
import sys

from dotenv import load_dotenv

import config
from parsers.openapi import load_openapi_from_url, extract_operations_from_openapi, manual_operations_input
from parsers.curl_parser import parse_curl_collection
from generators.traditional import TraditionalGenerator
from generators.openai_gen import OpenAIGenerator
from generators.gemini_gen import GeminiGenerator
from generators.claude_gen import ClaudeGenerator
from generators.groq_gen import GroqGenerator
from runner import run_testcases
from reporters.csv_reporter import (
    save_operations_csv,
    save_results_csv,
    compute_generator_metrics,
    save_generator_metrics_csv,
    print_summary_table,
)


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
    print(f"     A) Tümünü seç")
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
    print(f"  Her operasyon için kaç LLM test senaryosu üretilsin?")
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
        print(f"  Kaynak       : Manuel giriş")

    base_display = base_url or "(curl'den otomatik alınacak)"
    print(f"  Base URL     : {base_display}")
    print(f"  Auth Token   : {'Var' if auth_token else 'Yok'}")
    print(f"  Ekstra Header: {len(extra_headers_raw)} adet")
    print(f"  Cookie       : {'Var' if cookie_str else 'Yok'}")
    print(f"  Generator    :")
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
    ns = parser.parse_args()
    ns.selected_generators = None  # Tümünü kullan
    ns.num_cases = config.normalize_num_cases(ns.num_cases)
    return ns


# ── Generator builder ────────────────────────────────────────────────────────

def _build_llm_generators(selected_keys: list = None) -> list:
    """
    LLM generator tuple listesi döner: (instance, variant_name, variant_desc)
    selected_keys: ["openai:gpt-4.1-mini", "gemini:...", "claude:...", "groq:..."]
                   None ise tümü dahil edilir.
    """
    def _include(key: str) -> bool:
        return selected_keys is None or key in selected_keys

    generators = []
    for model in config.OPENAI_MODELS:
        if _include(f"openai:{model}"):
            for v_name, v_desc in config.PROMPT_VARIANTS.items():
                generators.append((OpenAIGenerator(model), v_name, v_desc))
    for model in config.GEMINI_MODELS:
        if _include(f"gemini:{model}"):
            for v_name, v_desc in config.PROMPT_VARIANTS.items():
                generators.append((GeminiGenerator(model), v_name, v_desc))
    for model in config.CLAUDE_MODELS:
        if _include(f"claude:{model}"):
            for v_name, v_desc in config.PROMPT_VARIANTS.items():
                generators.append((ClaudeGenerator(model), v_name, v_desc))
    for model in config.GROQ_MODELS:
        if _include(f"groq:{model}"):
            for v_name, v_desc in config.PROMPT_VARIANTS.items():
                generators.append((GroqGenerator(model), v_name, v_desc))
    return generators


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


# ── Ana akış ─────────────────────────────────────────────────────────────────

def main() -> None:
    load_dotenv()

    # Argüman yoksa interaktif wizard
    if len(sys.argv) == 1:
        args = interactive_wizard()
    else:
        args = parse_args()

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
                print(f"HATA: curl dosyası okunamadı ({filepath}): {e}", file=sys.stderr)
                sys.exit(1)
            try:
                parsed_list = parse_curl_collection(curl_text)
            except ValueError as e:
                print(f"HATA: curl parse edilemedi ({filepath}): {e}", file=sys.stderr)
                sys.exit(1)
            for op, op_base_url, curl_headers, curl_cookies in parsed_list:
                if derived_base_url is None:
                    derived_base_url = op_base_url
                elif op_base_url != derived_base_url:
                    print(f"UYARI: {op.op_id} farklı host ({op_base_url}), "
                          f"base URL olarak {derived_base_url} kullanılıyor.")
                cookies = {**curl_cookies, **cookies}
                extra_headers = {**curl_headers, **extra_headers}
                operations.append(op)
        base_url = base_url or derived_base_url
        print(f"{len(operations)} curl operasyonu parse edildi: "
              + ", ".join(f"{op.method} {op.path}" for op in operations))

    elif args.openapi_url:
        if not base_url:
            print("HATA: --openapi-url ile birlikte --base-url da verilmeli.", file=sys.stderr)
            sys.exit(1)
        try:
            spec = load_openapi_from_url(
                args.openapi_url,
                headers=extra_headers or None,
                cookies=cookies or None,
            )
        except Exception as e:
            print(f"HATA: OpenAPI dokümanı yüklenemedi: {e}", file=sys.stderr)
            sys.exit(1)
        operations = extract_operations_from_openapi(spec)
        print(f"{len(operations)} operasyon çıkarıldı.")

    else:
        if not base_url:
            print("HATA: --base-url veya --curl-file argümanlarından biri zorunludur.", file=sys.stderr)
            sys.exit(1)
        operations = manual_operations_input()
        print(f"{len(operations)} operasyon girildi.")

    if not operations:
        print("Operasyon bulunamadı, program sonlandırılıyor.")
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

    # Geleneksel şablon
    if selected_keys is None or "traditional" in selected_keys:
        trad_gen = TraditionalGenerator()
        trad_rows = trad_gen.generate(operations, "", "", num_cases)
        all_rows.extend(trad_rows)
        print(f"  [Geleneksel] {len(trad_rows)} senaryo üretildi.")

    # LLM tabanlı generator'lar
    for gen_instance, v_name, v_desc in _build_llm_generators(selected_keys):
        gen_label = f"{type(gen_instance).__name__} ({v_name})"
        print(f"  [{gen_label}] üretiliyor...", end=" ", flush=True)
        try:
            rows = gen_instance.generate(
                operations,
                variant_name=v_name,
                variant_desc=v_desc,
                num_cases=num_cases,
            )
            all_rows.extend(rows)
            print(f"{len(rows)} senaryo.")
        except RuntimeError as e:
            print(f"ATILDI — {e}")

    print()
    print(f"Toplam {len(all_rows)} test senaryosu üretildi.")

    # ── Testleri çalıştır ───────────────────────────────────────────────
    if args.no_run:
        print("Testler çalıştırılmıyor (--no-run / wizard seçimi).")
        executed_rows = all_rows
    else:
        executed_rows = run_testcases(
            base_url,
            all_rows,
            auth_token=args.auth_token,
            extra_headers=extra_headers or None,
            cookies=cookies or None,
        )

    # ── Raporla ────────────────────────────────────────────────────────
    save_results_csv(executed_rows, args.output_dir)

    if not args.no_run:
        metrics = compute_generator_metrics(executed_rows)
        save_generator_metrics_csv(metrics, args.output_dir)
        print_summary_table(executed_rows)

    print()
    print(f"Tamamlandı. Çıktılar: {args.output_dir}/")


if __name__ == "__main__":
    main()
