# LLM Test Lab

LLM Test Lab, API operasyonlarını OpenAPI/Swagger veya `curl` girdilerinden alıp çoklu test senaryosu üreten, bu senaryoları gerçek API'ye karşı çalıştırabilen ve üreten modelleri karşılaştırmalı olarak analiz eden bir laboratuvar uygulamasıdır.

Proje iki çalışma modu sunar:

- Web UI: Flask tabanlı arayüz ile yükleme, çalıştırma, log izleme ve sonuç karşılaştırma
- CLI: `main.py` üzerinden etkileşimli wizard veya doğrudan parametre ile çalışma

## Ne Yapar?

- OpenAPI/Swagger URL'den operasyon çıkarır
- `curl` komut koleksiyonundan operasyon ve request metadata üretir
- Geleneksel şablon bazlı baseline test seti oluşturur
- OpenAI, Gemini ve Claude tabanlı generator'larla test case üretir
- Üretilen case'leri gerçek API'ye karşı çalıştırır
- Sonuçları CSV olarak kaydeder
- Model bazlı performans, çeşitlilik ve benzerlik kıyasını UI üzerinde gösterir

## Akademik ve Analitik Kavramlar

Bu projede sıradan bir pass/fail raporlamasının ötesine geçen birkaç analitik kavram kullanılır.

### 1. Baseline ve Çoklu LLM Karşılaştırması

Sadece LLM'leri birbiriyle kıyaslamak yerine önce sabit bir geleneksel baseline üretilir. Böylece:

- LLM'lerin klasik şablon yaklaşımına göre ne kattığı görülebilir
- Aynı operasyon için hangi modelin daha güçlü veya daha zayıf olduğu anlaşılır
- Üretim kalitesi sadece sezgisel değil, ölçülebilir hale gelir

### 2. Prompt Varyantları

Her LLM aynı operasyon için tek bir prompt ile değil, farklı test üretim stratejileri ile çalıştırılır.

- `basic`: temel fonksiyonel senaryolar
- `edge_focused`: negatif, sınır değeri ve auth odaklı senaryolar

Bu yaklaşım, modelin sadece tek bir prompt altında değil, farklı test stratejilerinde nasıl davrandığını incelemek için önemlidir.

### 3. Generator Metrikleri

Her generator için temel performans metrikleri hesaplanır:

- `total_tests`
- `pass_count`
- `fail_count`
- `pass_rate`
- beklenen durum kodu dağılımı
- gerçekleşen durum kodu dağılımı

Bu katman, klasik test raporlama bakışıdır. Ancak projedeki akademik fark bunun üstüne inşa edilen anlamsal analiz katmanıdır.

### 4. Testcase Semantik İmzası

Her testcase aşağıdaki alanlardan token tabanlı bir semantik imza üretir:

- `operation_id`
- `http_method`
- `path`
- `title`
- `request_body`
- `expected_status`
- `expected_result`

Amaç, iki testin sadece metin olarak değil, anlamca ne kadar benzer olduğunu yaklaşık olarak ölçmektir.

### 5. Jaccard Benzerliği

İki testcase arasındaki benzerlik, token kümeleri üzerinden Jaccard benzerliği ile hesaplanır:

`J(A, B) = |A ∩ B| / |A ∪ B|`

Bu sayede:

- birbirine çok benzeyen tekrar senaryolar tespit edilir
- iki farklı modelin aslında aynı case'leri üretip üretmediği anlaşılır
- yüksek çeşitlilik ile yüksek kopyacılık ayrıştırılır

### 6. Intra-Generator Similarity

Aynı generator tarafından üretilen testcase'lerin birbirine ortalama ne kadar benzediği hesaplanır.

Yorum:

- yüksek değer: model kendi içinde tekrar eden veya birbirine çok yakın testler üretiyor olabilir
- düşük değer: model daha çeşitli senaryolar üretiyor olabilir

### 7. Pairwise Generator Similarity

Her generator çifti için karşılıklı benzerlik hesaplanır. Bunun için her testcase, karşı taraftaki en yakın testcase ile eşleştirilir ve çift yönlü ortalama alınır.

Bu tablo şu soruya cevap verir:

"Bu iki model gerçekten farklı test düşünüyor mu, yoksa aynı fikri başka cümlelerle mi yazıyor?"

### 8. Diversity Score

Çeşitlilik skoru tek bir sayı ile ifade edilir. Aşağıdaki sinyaller birlikte kullanılır:

- `1 - intra_similarity`
- operasyon kapsama oranı
- beklenen status çeşitliliği

Bu skor, test üretiminin ne kadar geniş bir uzaya yayıldığını yaklaşık olarak temsil eder.

### 9. Hermitian Kompleks Karşılaştırma Matrisi

Model karşılaştırmasının merkezinde kompleks değerli bir Hermitian pairwise matrix bulunur.

Bu matriste:

- genlik (`|z|`): başarı oranı, çeşitlilik, coverage ve düşük kopyacılık etkisini taşır
- faz (`∠z`): fail dengesizliği, semantik yakınlık ve çeşitlilik farkını taşır

Hermitian yapı sayesinde:

- `M[i][j] = conjugate(M[j][i])`
- köşegen elemanlar özdeşlik/öz referans gibi davranır
- pairwise üstünlük ilişkileri yönlü ama matematiksel olarak dengeli temsil edilir

Bu yaklaşım klasik "tek skorla sıralama" yönteminden daha zengindir; çünkü sadece büyüklüğü değil yön bilgisini de saklar.

### 10. Spektral Sıralama ve Power Iteration

Kompleks karşılaştırma matrisinden baskın özvektör yaklaşık olarak `power iteration` ile çıkarılır. Elde edilen özvektörün büyüklüğü (`abs`) modelin spektral skorunu verir.

Bu skor:

- sadece pass rate'e dayanmaz
- çeşitlilik ve semantik farklılaşmayı da dikkate alır
- çoklu model sistemlerinde daha dengeli bir global sıralama üretir

### 11. Operasyon Bazlı Karşılaştırma Matrisi

UI'da ayrıca operasyon bazında şu kıyas yapılır:

- her generator ilgili operasyon için kaç case üretti
- kaç tanesi geçti
- operasyon bazlı pass rate nedir

Bu görünüm, toplamda iyi gözüken bir modelin hangi endpoint türlerinde zayıf kaldığını yakalamak için önemlidir.

## Proje Yapısı

```text
.
|-- app.py                          # Flask Web UI
|-- main.py                         # CLI / wizard
|-- config.py                       # Merkezi ayarlar
|-- runner.py                       # Test case execution
|-- models.py                       # Veri modelleri
|-- generators/
|   |-- traditional.py              # Baseline generator
|   |-- openai_gen.py               # OpenAI adapter
|   |-- gemini_gen.py               # Gemini adapter
|   |-- claude_gen.py               # Claude adapter
|   |-- groq_gen.py                 # Groq adapter
|   `-- base.py                     # Ortak LLM yardımcıları
|-- parsers/
|   |-- openapi.py                  # OpenAPI parser
|   `-- curl_parser.py              # curl parser
|-- reporters/
|   `-- csv_reporter.py             # CSV ve analitik özetler
|-- security/
|   |-- redaction.py                # Secret redaction utility
|   `-- secret_loader.py            # Safe API key loader
|-- scripts/
|   `-- scan_secrets.py             # Local secret scanner (exit 1 on hit)
|-- docs/
|   `-- security.md                 # Security policy & key rotation guide
|-- templates/
|   `-- index.html                  # Web arayüzü
|-- tests/                          # Unit ve integration testleri
|-- .env.example                    # Placeholder env file (committed)
`-- .github/workflows/
    |-- ci.yml                      # Test suite CI
    `-- security-check.yml          # Secret scan CI
```

## Gereksinimler

- Python 3.13 önerilir
- Windows PowerShell komutları aşağıda verilmiştir
- OpenAI / Gemini / Anthropic API anahtarları gerekiyorsa `.env` içine yazılmalıdır

## Ortam Değişkenleri

Gerçek anahtarları doğrudan repoya yazmayın. Bu repo için örnek dosya hazırlandı:

- [.env.example](.env.example)

Kullanılan değişkenler:

- `OPENAI_API_KEY`
- `GEMINI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GROQ_API_KEY`

Not:

- Eğer bir API anahtarı geçmişte commit edildiyse, sadece `.gitignore` yetmez; anahtar mutlaka rotate edilmelidir.

## Security & Secret Management

This is a public academic/research prototype. The following rules apply to all contributors.

**Never commit real API keys.** `.env.example` contains placeholder values only. Real keys must be stored locally in `.env`, which is git-ignored and must never be committed.

### Local Setup

```bash
cp .env.example .env
# Open .env and fill in your own keys
```

### If a real key was accidentally committed

Treat it as **compromised immediately** — being in git history is the same as being public. Steps:

1. Revoke/rotate the key at the provider dashboard:
   - OpenAI: <https://platform.openai.com/api-keys>
   - Anthropic: <https://console.anthropic.com/settings/keys>
   - Groq: <https://console.groq.com/keys>
   - Google AI / Gemini: <https://aistudio.google.com/app/apikey>
2. Purge the key from git history with `git filter-repo` or BFG Repo Cleaner.
3. Force-push the cleaned history and notify collaborators.

### Secret scanning

A local scanner is included. Run it before every push:

```bash
python scripts/scan_secrets.py
```

Exits with code `0` if clean, `1` if potential secrets are detected. File paths and pattern names are reported — actual values are never printed.

GitHub Actions runs this scanner automatically on every push and pull request via [.github/workflows/security-check.yml](.github/workflows/security-check.yml).

### How API keys are handled

- Keys are loaded from environment variables at runtime via `security/secret_loader.py`.
- Error messages include only the variable **name**, never the value.
- Exception messages and HTTP error responses are passed through `security/redaction.py` before being logged or returned, masking any accidentally embedded key values.

See [docs/security.md](docs/security.md) for the full policy, rotation checklist, and incident response steps.

## Kurulum: Komut Komut

### 1. Proje klasörüne gir

```powershell
cd "c:\Sercem\İlk Deneme projesi\llm_test_lab"
```

### 2. Sanal ortam oluştur

```powershell
py -3.13 -m venv .venv
```

### 3. Sanal ortamı aktifleştir

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. `pip` güncelle

```powershell
python -m pip install --upgrade pip
```

### 5. Bağımlılıkları kur

```powershell
pip install -r requirements.txt
```

Alternatif olarak proje metadata'sı üzerinden geliştirme bağımlılıklarıyla kurulum:

```powershell
pip install -e ".[dev]"
```

### 6. Ortam dosyasını oluştur

```powershell
Copy-Item .env.example .env
```

### 7. `.env` içine API anahtarlarını yaz

```powershell
notepad .env
```

## Web UI Nasıl Çalıştırılır?

### 1. Uygulamayı başlat

```powershell
python app.py
```

### 2. Tarayıcıda aç

```text
http://localhost:5000
```

### 3. UI üzerinden akış

1. Kaynak seç:
   - OpenAPI / Swagger URL
   - veya `curl` dosyası
2. Gerekirse Bearer token, header ve cookie bilgilerini gir
3. Generator seç
4. Operasyon başına testcase sayısını gir
5. İstersen gerçek API execution'ı aç/kapat
6. `ÇALIŞTIR` butonuna bas
7. Loglar ve sonuç karşılaştırma ekranını izle

## CLI Nasıl Çalıştırılır?

### Etkileşimli mod

```powershell
python main.py
```

Bu komut seni adım adım wizard içine alır.

### Doğrudan OpenAPI ile çalışma

```powershell
python main.py --openapi-url "https://example.com/openapi.json" --base-url "https://api.example.com/v1" --no-run
```

### `curl` dosyası ile çalışma

```powershell
python main.py --curl-file .\sample.txt --no-run
```

### Gerçek API'ye karşı çalıştırma

```powershell
python main.py --openapi-url "https://example.com/openapi.json" --base-url "https://api.example.com/v1"
```

## Testler Nasıl Çalıştırılır?

### Tüm testleri çalıştır

```powershell
pytest -q
```

### Belirli bir test dosyasını çalıştır

```powershell
pytest .\tests\test_reporters.py -q
```

### Belirli bir test fonksiyonunu çalıştır

```powershell
pytest .\tests\test_llm_generators.py -q -k openai
```

## Çıktılar Nerede?

Varsayılan çıktı klasörü:

```text
outputs/
```

Üretilen başlıca dosyalar:

- `operations.csv`
- `executed_testcases_<timestamp>.csv`
- `generator_metrics_<timestamp>.csv`
- `run_info_<job_id>_<timestamp>.json`

`run_info` dosyası seçilen modelleri, prompt varyantlarını, operasyon sayısını, senaryo sayısını ve çalışma zamanı config özetini saklar. Bu dosya deneylerin daha sonra aynı koşullarla tekrarlanabilmesi için eklenmiştir.

## Web UI Güvenlik ve İş Yönetimi

- Upload dosyaları yalnızca `.txt`, `.curl` ve `.http` uzantılarıyla kabul edilir.
- Varsayılan upload limiti 1 MB'dir.
- Web UI job sonuçları job token ile korunur; UI bu token'ı otomatik kullanır.
- Çalışan job için iptal isteği gönderilebilir; iş en yakın güvenli durma noktasında `cancelled` durumuna geçer.
- `/health` endpoint'i uygulama durumunu döner.
- `/download_report/<job_id>` endpoint'i CSV sonuçları, metadata ve JSON özetleri içeren ZIP rapor paketi üretir.

## CI Nasıl Çalışır?

Bu projede GitHub Actions CI tanımlıdır:

- [.github/workflows/ci.yml](.github/workflows/ci.yml)

Tetiklenme koşulları:

- her `push`
- her `pull_request`

CI adımları:

1. repository checkout
2. Python 3.13 kurulumu
3. `requirements.txt` ile bağımlılıkların kurulumu
4. `pytest -q` çalıştırılması

### GitHub üzerinde CI'yı tetiklemek için

```powershell
git add .
git commit -m "Update README and CI docs"
git push origin <branch-adi>
```

Push sonrası:

1. GitHub repo sayfasına git
2. `Actions` sekmesini aç
3. `CI` workflow'unu seç
4. `Run test suite` adımını incele

### Localde CI ile aynı mantığı çalıştırmak için

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest -q
```

## Mevcut Test Kapsamı

Projede şu alanlar test altındadır:

- parser testleri
- generator yardımcı fonksiyonları
- traditional generator
- OpenAI / Gemini / Claude mock testleri
- runner testleri
- reporter ve kompleks matris testleri
- Flask route ve job lifecycle integration testleri

## Sık Kullanılan Komutlar

### Uygulamayı başlat

```powershell
python app.py
```

### CLI wizard

```powershell
python main.py
```

### Testler

```powershell
pytest -q
```

### CI öncesi hızlı kontrol

```powershell
python -m pytest -q
```

## Kısa Özet

Bu proje, çoklu LLM tabanlı API test üretimini sadece "hangi model daha iyi?" seviyesinde değil, aşağıdaki daha akademik seviyelerde de ele alır:

- doğruluk
- kapsama
- çeşitlilik
- tekrar oranı
- modeller arası semantik yakınlık
- kompleks pairwise üstünlük ilişkisi
- spektral/global sıralama

Bu nedenle LLM Test Lab, hem pratik bir QA aracı hem de karşılaştırmalı test üretimi için analitik bir araştırma ortamı olarak kullanılabilir.
