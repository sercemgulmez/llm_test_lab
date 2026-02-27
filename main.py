"""
LLM Test Lab — Otomatik API Test Senaryosu Üretim ve Yürütme Aracı

Kullanım:
    python main.py --base-url https://api.example.com [SEÇENEKLER]

Örnekler:
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
        --openapi-url https://petstore.swagger.io/v2/swagger.json \\
        --base-url    https://petstore.swagger.io/v2 \\
        --no-run
"""

import argparse
import os
import sys

from dotenv import load_dotenv

import config
from parsers.openapi import load_openapi_from_url, extract_operations_from_openapi, manual_operations_input
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
        "--openapi-url",
        metavar="URL",
        help="OpenAPI / Swagger JSON veya YAML dokümanının URL'i. "
             "Verilmezse manuel giriş modu devreye girer.",
    )
    parser.add_argument(
        "--base-url",
        metavar="URL",
        required=True,
        help="Test edilecek API'nin base URL'i (örn: https://api.example.com/v1).",
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


def main() -> None:
    load_dotenv()
    args = parse_args()

    print("=" * 60)
    print("LLM TEST LAB")
    print(f"Base URL : {args.base_url}")
    print(f"Çıktı    : {args.output_dir}")
    print("=" * 60)

    # ── 1. Operasyonları al ─────────────────────────────────────────────────
    if args.openapi_url:
        try:
            spec = load_openapi_from_url(args.openapi_url)
        except Exception as e:
            print(f"HATA: OpenAPI dokümanı yüklenemedi: {e}", file=sys.stderr)
            sys.exit(1)
        operations = extract_operations_from_openapi(spec)
        print(f"{len(operations)} operasyon çıkarıldı.")
    else:
        operations = manual_operations_input()
        print(f"{len(operations)} operasyon girildi.")

    if not operations:
        print("Operasyon bulunamadı, program sonlandırılıyor.")
        sys.exit(0)

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
            # API anahtarı eksik veya paket yok — uyar, devam et
            gen_label = type(gen_instance).__name__
            print(f"UYARI [{gen_label}]: {e} — bu generator atlandı.")

    print(f"\nToplam {len(all_rows)} test senaryosu üretildi.")

    # ── 3. Testleri çalıştır (opsiyonel) ───────────────────────────────────
    if args.no_run:
        print("--no-run aktif: testler çalıştırılmıyor.")
        executed_rows = all_rows
    else:
        executed_rows = run_testcases(
            args.base_url, all_rows, auth_token=args.auth_token
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
