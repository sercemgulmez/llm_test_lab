"""
LLM Test Lab — Otomatik API Test Senaryosu Üretim ve Yürütme Aracı

Kullanım:
    python main.py --base-url https://api.example.com [SEÇENEKLER]

Örnekler:
    # curl dosyasından otomatik parse
    python main.py --curl-file turkcell.txt

    # OpenAPI spec URL'den operasyonları al, testleri çalıştır
    python main.py \\
        --openapi-url https://petstore.swagger.io/v2/swagger.json \\
        --base-url    https://petstore.swagger.io/v2

    # Manuel operasyon girişi, auth token ile
    python main.py \\
        --base-url   https://api.example.com \\
        --auth-token eyJhbGciOiJIUzI1NiJ9...

    # Sadece senaryo üret, testleri çalıştırma
    python main.py \\
        --curl-file turkcell.txt \\
        --no-run
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
from runner import run_testcases
from reporters.csv_reporter import (
    save_operations_csv,
    save_results_csv,
    compute_generator_metrics,
    save_generator_metrics_csv,
    print_summary_table,
)


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
        help="curl komutlarını içeren dosya(lar). Tek dosyada birden fazla curl olabilir. "
             "Verilirse --base-url ve --openapi-url'ye gerek kalmaz. "
             "Örn: --curl-file a.txt b.txt",
    )
    parser.add_argument(
        "--openapi-url",
        metavar="URL",
        help="OpenAPI / Swagger JSON veya YAML dokümanının URL'i.",
    )
    parser.add_argument(
        "--base-url",
        metavar="URL",
        default=None,
        help="Test edilecek API'nin base URL'i (örn: https://api.example.com/v1). "
             "--curl-file verilmişse opsiyoneldir.",
    )
    parser.add_argument(
        "--auth-token",
        metavar="TOKEN",
        default=None,
        help="Bearer token (Authorization: Bearer <token> başlığına eklenir).",
    )
    parser.add_argument(
        "--no-run",
        action="store_true",
        help="Test senaryolarını üret ama API'ye karşı çalıştırma.",
    )
    parser.add_argument(
        "--header",
        metavar="KEY: VALUE",
        action="append",
        dest="headers",
        default=[],
        help="Ekstra HTTP header (tekrarlanabilir). Örn: --header 'App-Channel-Type: WEB'",
    )
    parser.add_argument(
        "--cookie",
        metavar="COOKIE_STRING",
        default=None,
        help="Ham cookie string. Örn: --cookie 'sessionId=abc; token=xyz'",
    )
    parser.add_argument(
        "--output-dir",
        metavar="DIR",
        default=config.OUTPUT_DIR,
        help=f"Çıktıların kaydedileceği klasör (varsayılan: {config.OUTPUT_DIR}).",
    )
    return parser.parse_args()


def _build_llm_generators() -> list:
    """
    Yapılandırılmış LLM generator'larının listesini döner.
    Her öğe (generator_instance, variant_name, variant_desc) tuple'ıdır.
    """
    generators = []
    for model in config.OPENAI_MODELS:
        for v_name, v_desc in config.PROMPT_VARIANTS.items():
            generators.append((OpenAIGenerator(model), v_name, v_desc))
    for model in config.GEMINI_MODELS:
        for v_name, v_desc in config.PROMPT_VARIANTS.items():
            generators.append((GeminiGenerator(model), v_name, v_desc))
    for model in config.CLAUDE_MODELS:
        for v_name, v_desc in config.PROMPT_VARIANTS.items():
            generators.append((ClaudeGenerator(model), v_name, v_desc))
    return generators


def _parse_cli_headers(header_list: list) -> dict:
    """--header 'Key: Value' listesini dict'e çevirir."""
    result = {}
    for h in header_list:
        key, _, value = h.partition(": ")
        if key:
            result[key.strip()] = value.strip()
    return result


def _parse_cli_cookies(cookie_str: str) -> dict:
    """'name=val; name2=val2' string'ini dict'e çevirir."""
    result = {}
    for part in cookie_str.split("; "):
        k, _, v = part.partition("=")
        if k:
            result[k.strip()] = v
    return result


def main() -> None:
    load_dotenv()
    args = parse_args()

    # ── 0. Header / Cookie hazırlığı ────────────────────────────────────────
    extra_headers = _parse_cli_headers(args.headers)
    cookies = _parse_cli_cookies(args.cookie) if args.cookie else {}

    # ── 1. Operasyonları al ─────────────────────────────────────────────────
    base_url = args.base_url

    if args.curl_file:
        # Bir veya birden fazla curl dosyasını parse et
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
                # İlk curl'den base_url al
                if derived_base_url is None:
                    derived_base_url = op_base_url
                elif op_base_url != derived_base_url:
                    print(f"UYARI: {op.op_id} farklı host ({op_base_url}), "
                          f"base URL olarak {derived_base_url} kullanılıyor.")
                # curl header/cookie'lerini biriktir (CLI argümanları override eder)
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
            spec = load_openapi_from_url(args.openapi_url)
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

    print("=" * 60)
    print("LLM TEST LAB")
    print(f"Base URL : {base_url}")
    print(f"Çıktı    : {args.output_dir}")
    print("=" * 60)

    os.makedirs(args.output_dir, exist_ok=True)
    save_operations_csv(operations, args.output_dir)

    # ── 2. Test senaryolarını üret ──────────────────────────────────────────
    all_rows: list = []

    # Geleneksel şablon
    trad_gen = TraditionalGenerator()
    all_rows.extend(trad_gen.generate(operations, "", "", 0))

    # LLM tabanlı generator'lar
    for gen_instance, v_name, v_desc in _build_llm_generators():
        try:
            rows = gen_instance.generate(
                operations,
                variant_name=v_name,
                variant_desc=v_desc,
                num_cases=config.NUM_CASES_PER_OPERATION,
            )
            all_rows.extend(rows)
        except RuntimeError as e:
            gen_label = type(gen_instance).__name__
            print(f"UYARI [{gen_label}]: {e} — bu generator atlandı.")

    print(f"\nToplam {len(all_rows)} test senaryosu üretildi.")

    # ── 3. Testleri çalıştır (opsiyonel) ───────────────────────────────────
    if args.no_run:
        print("--no-run aktif: testler çalıştırılmıyor.")
        executed_rows = all_rows
    else:
        executed_rows = run_testcases(
            base_url,
            all_rows,
            auth_token=args.auth_token,
            extra_headers=extra_headers or None,
            cookies=cookies or None,
        )

    # ── 4. Raporla ─────────────────────────────────────────────────────────
    save_results_csv(executed_rows, args.output_dir)

    if not args.no_run:
        metrics = compute_generator_metrics(executed_rows)
        save_generator_metrics_csv(metrics, args.output_dir)
        print_summary_table(executed_rows)

    print(f"\nTamamlandı. Çıktılar: {args.output_dir}/")


if __name__ == "__main__":
    main()
