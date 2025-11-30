"""
URL tabanlı otomatik test senaryosu üretim ve yürütme aracı.

Özellikler:
- OpenAPI / Swagger URL'inden veya manuel girişten API operasyonlarını okur
- Her operasyon için:
    - Şablon (geleneksel) test senaryoları üretir
    - OpenAI (GPT) ile test senaryoları üretir (birden fazla model)
    - Google Gemini ile test senaryoları üretir
- Tüm senaryoları gerçek API'ye karşı çalıştırır
- Sonuçları CSV formatında dışa aktarır

Bu proje, farklı test senaryosu üretim yaklaşımlarının
karşılaştırmalı analizine temel olacak şekilde tasarlanmıştır.
"""

import csv
import os
import json
from dataclasses import dataclass
from typing import List, Dict, Optional
from collections import defaultdict

import requests
import yaml
from dotenv import load_dotenv
from openai import OpenAI
from google import genai


# ================== AYARLAR ==================

# OpenAI (GPT) için kullanılacak modeller
# İstersen bu listeyi ihtiyaçlarına göre düzenleyebilirsin.
OPENAI_MODELS = [
    "gpt-4.1-mini",
    "gpt-4.1",
    "gpt-4o-mini",
]

# Google Gemini için kullanılacak modeller
GEMINI_MODELS = [
    "gemini-2.5-flash",
]

# Prompt stratejileri (farklı test üretim tarzları)
PROMPT_VARIANTS = {
    "basic": "Temel fonksiyonel senaryolar üret; mutlu path ve birkaç hata durumu ekle.",
    "edge_focused": "Negatif, sınır değeri ve kimlik doğrulama odaklı senaryolar üret; farklı hata kodlarını da kapsa.",
}

# Her API operasyonu için üretilecek LLM test senaryosu sayısı (hedef)
NUM_CASES_PER_OPERATION = 10

# Güvenlik için üst sınır (operasyon başına maksimum LLM senaryosu)
MAX_CASES_PER_OPERATION = 10

# Çıktıların yazılacağı klasör
OUTPUT_DIR = "outputs"

# ============================================

load_dotenv()  # .env içindeki anahtarları yükler


@dataclass
class ApiOperation:
    op_id: str
    method: str
    path: str
    summary: str
    description: str


# ---------- Yardımcı fonksiyonlar ----------

def join_url(base_url: str, path: str) -> str:
    """Base URL ile path'i düzgün şekilde birleştirir."""
    if base_url.endswith("/") and path.startswith("/"):
        return base_url[:-1] + path
    if (not base_url.endswith("/")) and (not path.startswith("/")):
        return base_url + "/" + path
    return base_url + path


def load_openapi_from_url(url: str) -> Dict:
    """Verilen URL'den OpenAPI / Swagger JSON veya YAML dokümanı yükler."""
    print(f"OpenAPI/Swagger yükleniyor: {url}")
    resp = requests.get(url)
    resp.raise_for_status()
    text = resp.text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return yaml.safe_load(text)


def extract_operations_from_openapi(spec: Dict) -> List[ApiOperation]:
    """OpenAPI dokümanından HTTP operasyonlarını çıkarır."""
    ops: List[ApiOperation] = []
    paths = spec.get("paths", {})
    counter = 1
    for path, methods in paths.items():
        if not isinstance(methods, dict):
            continue
        for method, op in methods.items():
            if method.lower() not in [
                "get",
                "post",
                "put",
                "delete",
                "patch",
                "head",
                "options",
            ]:
                continue
            if not isinstance(op, dict):
                continue
            summary = op.get("summary") or ""
            description = op.get("description") or ""
            op_id = op.get("operationId") or f"OP{counter}"
            ops.append(
                ApiOperation(
                    op_id=op_id,
                    method=method.upper(),
                    path=path,
                    summary=summary,
                    description=description,
                )
            )
            counter += 1
    return ops


def manual_operations_input() -> List[ApiOperation]:
    """İstenirse API operasyonlarını manuel olarak konsoldan alır."""
    ops: List[ApiOperation] = []
    print("\nManuel endpoint girişi. Bitirmek için method kısmını boş bırak.\n")
    idx = 1
    while True:
        method = input(
            f"[{idx}] HTTP method (GET/POST/PUT/DELETE, boş = bitir): "
        ).strip().upper()
        if not method:
            break
        path = input(f"[{idx}] Path (örn: /users, /login): ").strip()
        summary = input(f"[{idx}] Kısa özet: ").strip()
        description = input(f"[{idx}] Detay açıklama (opsiyonel): ").strip()
        ops.append(
            ApiOperation(
                op_id=f"MAN{idx}",
                method=method,
                path=path,
                summary=summary,
                description=description,
            )
        )
        idx += 1
    return ops


def save_operations_csv(operations: List[ApiOperation], path: str) -> None:
    """Çıkarılan/girilen API operasyonlarını CSV olarak kaydeder."""
    if not operations:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = ["op_id", "method", "path", "summary", "description"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for op in operations:
            w.writerow(
                {
                    "op_id": op.op_id,
                    "method": op.method,
                    "path": op.path,
                    "summary": op.summary,
                    "description": op.description,
                }
            )
    print("Operasyon listesi kaydedildi:", path)


def build_llm_prompt_for_operation(
    op: ApiOperation, num_cases: int, variant_name: str, variant_desc: str
) -> str:
    """Bir API operasyonu için LLM'e gönderilecek metni hazırlar."""
    return f"""
Kıdemli bir Backend QA mühendisi gibi davran.
Aşağıdaki API operasyonu için fonksiyonel test senaryoları üret.

Operasyon ID: {op.op_id}
HTTP Method: {op.method}
Path: {op.path}
Özet: {op.summary}
Açıklama: {op.description}

Test stratejisi (varyant):
- {variant_name}: {variant_desc}

Her test senaryosu için:
- Gerçekçi bir istek örneği kurgula (query parametreleri / JSON body vb.).
- Beklenen HTTP durum kodunu belirt.
- Senaryonun amacını kısa şekilde özetle.

ÇIKTI FORMATIN:
Her test tek satır olacak ve şu formatta yazılacak (aralarda | karakteri):

TC_ID|Kısa Başlık|HTTP_METHOD PATH|Request JSON Body (yoksa - yaz)|Beklenen HTTP Status Kodu (sayı)|Beklenen Sonuç (kısa açıklama)

ÖRNEK:
{op.op_id}_TC1|Başarılı giriş|POST /login|{{"phone":"+905xxxxxxxxx","password":"GecerliSifre1"}}|200|Kullanıcı başarıyla giriş yapar ve token döner

Kurallar:
- TC_ID şu formda olmalı: {op.op_id}_TC1, {op.op_id}_TC2, ...
- Request body varsa geçerli bir JSON nesnesi ({{ }}) olmalı.
- Body yoksa aynen "-" yaz.
- HTTP method ve path kısmında boşlukla ayrılmış method ve path kullan (örn: GET /users/1).
- Ekstra açıklama yazma, sadece bu formatta satırlar üret.

Şimdi tam olarak {num_cases} satır üret.
""".strip()


# ---------- LLM üreticileri ----------

def generate_llm_cases_openai(
    operations: List[ApiOperation],
    model: str,
    variant_name: str,
    variant_desc: str,
    num_cases: int,
) -> List[Dict]:
    """OpenAI modelleri ile test senaryosu üretir."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("UYARI: OPENAI_API_KEY tanımlı değil, OpenAI üretimi atlandı.")
        return []

    client = OpenAI(api_key=api_key)
    num_cases = min(num_cases, MAX_CASES_PER_OPERATION)
    rows: List[Dict] = []

    for op in operations:
        print(
            f"[OpenAI - {model} - {variant_name}] {op.op_id} ({op.method} {op.path}) için senaryo üretiliyor..."
        )
        prompt = build_llm_prompt_for_operation(
            op, num_cases, variant_name, variant_desc
        )
        try:
            resp = client.responses.create(
                model=model,
                input=prompt,
            )
            # Resmi OpenAI Python SDK'da önerilen yol: output_text kullanmak
            text = resp.output_text or ""
        except Exception as e:
            print("OpenAI cevabı okunamadı:", e)
            continue

        lines = [l.strip() for l in text.splitlines() if l.strip()]
        rows.extend(
            _parse_llm_lines_to_rows(
                lines, op, f"LLM-OpenAI-{model}-{variant_name}"
            )
        )

    return rows


def generate_llm_cases_gemini(
    operations: List[ApiOperation],
    model: str,
    variant_name: str,
    variant_desc: str,
    num_cases: int,
) -> List[Dict]:
    """Google Gemini modelleri ile test senaryosu üretir."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("UYARI: GEMINI_API_KEY tanımlı değil, Gemini üretimi atlandı.")
        return []

    client = genai.Client(api_key=api_key)
    num_cases = min(num_cases, MAX_CASES_PER_OPERATION)
    rows: List[Dict] = []

    for op in operations:
        print(
            f"[Gemini - {model} - {variant_name}] {op.op_id} ({op.method} {op.path}) için senaryo üretiliyor..."
        )
        prompt = build_llm_prompt_for_operation(
            op, num_cases, variant_name, variant_desc
        )
        try:
            resp = client.models.generate_content(
                model=model,
                contents=prompt,
            )
            text = resp.text or ""
        except Exception as e:
            print("Gemini cevabı okunamadı:", e)
            continue

        lines = [l.strip() for l in text.splitlines() if l.strip()]
        rows.extend(
            _parse_llm_lines_to_rows(
                lines, op, f"LLM-Gemini-{model}-{variant_name}"
            )
        )

    return rows


def _parse_llm_lines_to_rows(
    lines: List[str],
    op: ApiOperation,
    generator_name: str,
) -> List[Dict]:
    """LLM çıktısındaki satırları ortak satır formatına dönüştürür."""
    rows: List[Dict] = []
    for line in lines:
        parts = [p.strip() for p in line.split("|")]
        if len(parts) != 6:
            continue

        tc_id, title, method_path, body_str, exp_status_str, exp_result = parts

        mp_parts = method_path.split(maxsplit=1)
        if len(mp_parts) != 2:
            continue
        method, path = mp_parts[0].upper(), mp_parts[1]

        try:
            exp_status = int(exp_status_str)
        except ValueError:
            exp_status = None

        if body_str == "-":
            body: Optional[dict] = None
        else:
            try:
                body = json.loads(body_str)
            except json.JSONDecodeError:
                body = None

        rows.append(
            {
                "generator": generator_name,
                "operation_id": op.op_id,
                "http_method": method,
                "path": path,
                "tc_id": tc_id,
                "title": title,
                "request_body": json.dumps(body) if body is not None else "",
                "expected_status": exp_status if exp_status is not None else "",
                "expected_result": exp_result,
            }
        )
    return rows


# ---------- Geleneksel (template) üretici ----------

def generate_traditional_cases(operations: List[ApiOperation]) -> List[Dict]:
    """
    Basit şablonlara dayalı geleneksel test senaryosu üretimi.
    Örnek senaryo tipleri:
    - 200 mutlu senaryo
    - 400 validasyon hatası
    - 401 yetkisiz erişim
    """
    rows: List[Dict] = []

    for op in operations:
        print(
            f"[Traditional] {op.op_id} ({op.method} {op.path}) için şablon senaryolar üretiliyor..."
        )
        base = op.op_id
        base_path = op.path

        templates = [
            (
                f"{base}_TC1",
                "200 - Mutlu senaryo",
                op.method,
                base_path,
                None,
                200,
                "İşlem başarıyla tamamlanmalı.",
            ),
            (
                f"{base}_TC2",
                "400 - Validasyon hatası",
                op.method,
                base_path,
                {"dummy": "invalid"},
                400,
                "Hatalı istek için validasyon hatası dönmeli.",
            ),
            (
                f"{base}_TC3",
                "401 - Yetkisiz erişim",
                op.method,
                base_path,
                None,
                401,
                "Yetkisiz kullanıcı için 401/403 dönmeli.",
            ),
        ]

        for tc_id, title, method, path, body, exp_status, exp_result in templates:
            rows.append(
                {
                    "generator": "Traditional-Template",
                    "operation_id": op.op_id,
                    "http_method": method,
                    "path": path,
                    "tc_id": tc_id,
                    "title": title,
                    "request_body": json.dumps(body) if body is not None else "",
                    "expected_status": exp_status,
                    "expected_result": exp_result,
                }
            )

    return rows


# ---------- Testleri yürütme ve raporlama ----------

def run_testcases(base_url: str, rows: List[Dict]) -> List[Dict]:
    """Üretilen tüm test senaryolarını gerçek API'ye karşı uygular."""
    print("\n=== Test senaryoları çalıştırılıyor ===")
    session = requests.Session()
    executed: List[Dict] = []

    for row in rows:
        full_url = join_url(base_url, row["path"])
        method = row["http_method"].upper()

        body_str = row.get("request_body") or ""
        json_body = None
        if body_str:
            try:
                json_body = json.loads(body_str)
            except json.JSONDecodeError:
                json_body = None

        expected_status = row.get("expected_status")

        try:
            resp = session.request(method, full_url, json=json_body, timeout=10)
            status = resp.status_code
            passed = (status == expected_status) if isinstance(expected_status, int) else None
        except Exception as e:
            status = None
            passed = None
            print(f"{row['tc_id']} isteği hata verdi: {e}")

        new_row = dict(row)
        new_row["url"] = full_url
        new_row["actual_status"] = status if status is not None else ""
        new_row["pass"] = passed
        executed.append(new_row)

    return executed


def save_results_csv(rows: List[Dict], path: str) -> None:
    """Çalıştırılan testlerin sonuçlarını CSV dosyasına yazar."""
    if not rows:
        print("Kaydedilecek satır yok:", path)
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        "generator",
        "operation_id",
        "http_method",
        "path",
        "tc_id",
        "title",
        "request_body",
        "expected_status",
        "expected_result",
        "url",
        "actual_status",
        "pass",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print("Kaydedildi:", path)


def compute_generator_metrics(rows: List[Dict]) -> List[Dict]:
    """Her generator için toplu metrikler üretir."""
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        groups[r["generator"]].append(r)

    metrics_rows: List[Dict] = []
    for gen, gen_rows in groups.items():
        total = len(gen_rows)
        num_pass = sum(1 for r in gen_rows if r.get("pass") is True)
        num_fail = sum(1 for r in gen_rows if r.get("pass") is False)

        exp_status_counts: Dict[str, int] = defaultdict(int)
        act_status_counts: Dict[str, int] = defaultdict(int)
        for r in gen_rows:
            es = r.get("expected_status")
            as_ = r.get("actual_status")
            if es != "" and es is not None:
                exp_status_counts[str(es)] += 1
            if as_ != "" and as_ is not None:
                act_status_counts[str(as_)] += 1

        metrics_rows.append(
            {
                "generator": gen,
                "total_tests": total,
                "pass_count": num_pass,
                "fail_count": num_fail,
                "pass_rate": round(num_pass / total, 3) if total else "",
                "expected_status_distribution": json.dumps(exp_status_counts, ensure_ascii=False),
                "actual_status_distribution": json.dumps(act_status_counts, ensure_ascii=False),
            }
        )

    return metrics_rows


def save_generator_metrics_csv(rows: List[Dict], path: str) -> None:
    """Generator bazlı metrikleri CSV dosyasına yazar."""
    if not rows:
        print("Generator metrikleri için satır yok.")
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        "generator",
        "total_tests",
        "pass_count",
        "fail_count",
        "pass_rate",
        "expected_status_distribution",
        "actual_status_distribution",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print("Kaydedildi:", path)


def simple_overall_metrics(rows: List[Dict]) -> None:
    """Genel PASS/FAIL sayıları."""
    print("\n===== GENEL ÖZET =====")
    if not rows:
        print("Hiç satır yok.")
        return

    total = len(rows)
    num_pass = sum(1 for r in rows if r.get("pass") is True)
    num_fail = sum(1 for r in rows if r.get("pass") is False)

    print("Toplam çalıştırılan test:", total)
    print("PASS:", num_pass)
    print("FAIL:", num_fail)


# ----------------- main -----------------

def main() -> None:
    print("=== LLM Test Aracı (Geleneksel + 3xGPT + Gemini, max 10 case/operasyon) ===\n")
    print("Çalışma modu:")
    print("1) OpenAPI / Swagger URL'den operasyonları çıkar")
    print("2) Operasyonları manuel gir")
    choice = input("Seçim (1/2): ").strip()

    if choice == "1":
        url = input("OpenAPI / Swagger JSON veya YAML URL'i: ").strip()
        spec = load_openapi_from_url(url)
        operations = extract_operations_from_openapi(spec)
        print(f"{len(operations)} adet operasyon çıkarıldı.")
    else:
        operations = manual_operations_input()
        print(f"{len(operations)} adet operasyon girildi.")

    if not operations:
        print("Operasyon bulunamadı, program sonlandırılıyor.")
        return

    # Operasyon listesini de kayıt altına al (tez için faydalı)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_operations_csv(operations, os.path.join(OUTPUT_DIR, "operations.csv"))

    base_url = input("Testler için base URL (örn: https://api.ornek.com): ").strip()

    all_rows: List[Dict] = []

    # 1) Geleneksel şablon senaryolar
    traditional_rows = generate_traditional_cases(operations)
    all_rows.extend(traditional_rows)

    # 2) OpenAI (GPT) senaryoları - çoklu model + prompt varyantı
    for model in OPENAI_MODELS:
        for variant_name, variant_desc in PROMPT_VARIANTS.items():
            llm_rows = generate_llm_cases_openai(
                operations,
                model=model,
                variant_name=variant_name,
                variant_desc=variant_desc,
                num_cases=NUM_CASES_PER_OPERATION,
            )
            all_rows.extend(llm_rows)

    # 3) Google Gemini senaryoları
    for model in GEMINI_MODELS:
        for variant_name, variant_desc in PROMPT_VARIANTS.items():
            llm_rows = generate_llm_cases_gemini(
                operations,
                model=model,
                variant_name=variant_name,
                variant_desc=variant_desc,
                num_cases=NUM_CASES_PER_OPERATION,
            )
            all_rows.extend(llm_rows)

    # 4) Tüm senaryoları çalıştır
    executed_rows = run_testcases(base_url, all_rows)

    # 5) Sonuçları kaydet
    save_results_csv(executed_rows, os.path.join(OUTPUT_DIR, "executed_testcases.csv"))

    # 6) Generator bazlı metrikleri hesaplayıp kaydet
    gen_metrics = compute_generator_metrics(executed_rows)
    save_generator_metrics_csv(
        gen_metrics, os.path.join(OUTPUT_DIR, "generator_metrics.csv")
    )

    # 7) Konsolda genel özet
    simple_overall_metrics(executed_rows)

    print(
        "\nTamamlandı.\n"
        f"- Operasyon listesi: {OUTPUT_DIR}/operations.csv\n"
        f"- Ayrıntılı sonuçlar: {OUTPUT_DIR}/executed_testcases.csv\n"
        f"- Generator bazlı özet metrikler: {OUTPUT_DIR}/generator_metrics.csv\n"
    )


if __name__ == "__main__":
    main()
