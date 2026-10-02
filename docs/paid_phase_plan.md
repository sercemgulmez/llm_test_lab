# Ücretli Faz Planı — canary ve tam koşu

> **Durum: HAZIR, ÇALIŞTIRILMADI.** Bu belgenin hiçbir adımı,
> "ONAYLIYORUM, ücretli canary'yi başlat" cümlesi gelmeden uygulanmaz.
>
> Hazırlanma tarihi: 2 Ekim 2026. Fiyatlar 28–29 Eylül 2026'da resmî
> sayfalardan çekildi (`pricing.py`, kaynak linkleri orada).

## Neden canary

Defterdeki `cost_usd_billed` bizim hesabımızdır: sağlayıcının bildirdiği
token sayıları × onaylanmış fiyat tablosu. Bu hesabın sağlayıcının **kendi
faturasıyla** örtüştüğü hiç doğrulanmadı. 600 satırlık ücretli faza girmeden
önce tek bir küçük koşuyla bu örtüşme sınanır.

## ⚠ Canary'nin ölçülebilirlik sorunu

Plandaki "1 model × 1 operasyon × 1 case" koşusunun maliyeti **ölçülemeyecek
kadar küçük**:

| Model | 1 case'lik canary | Panelde görünür mü |
|---|---|---|
| gpt-4o-mini | **$0.000139** | HAYIR (<$0.01) |
| gpt-4.1 | $0.001853 | HAYIR |
| claude-haiku-4-5 | $0.001037 | HAYIR |
| claude-sonnet-4-5 | $0.003109 | HAYIR |

Sağlayıcı panelleri maliyeti sente yuvarlar. $0.000139'luk bir çağrıyı panelde
**$0.00** olarak görürüz; "sapma ≤%10" ölçütü uygulanamaz — %10'u $0.0000139'dur.

### Çözüm: iki ölçütü ayır (kullanıcı kararı, 2 Ekim 2026)

**Ölçüt A — token mutabakatı: BİRİNCİL ölçüt.**
Sağlayıcı panelleri **token sayısını** model bazında raporlar. Defterdeki
`input_tokens` / `output_tokens` toplamı ile panelin token sayısı karşılaştırılır.
Sapma ≤%10 → PASS. Bu, fiyat tablosundan bağımsız olarak *ölçümün* doğruluğunu
sınar ve asıl risk buradadır (bizim token kaydı yanlışsa her şey yanlış olur).

**Ölçüt B — dolar mutabakatı (ikincil, panelde görünür tutar gerektirir).**
Panelin sente yuvarlamasını aşmak için canary, tutarı ≥$0.01 yapacak kadar
çağrı içerir:

| Model | 15-case çağrı başına | $0.01 için gereken çağrı |
|---|---|---|
| gpt-4o-mini | $0.001005 | 10 |
| gpt-4.1 | $0.013398 | 1 |
| claude-haiku-4-5 | $0.008252 | 2 |
| claude-sonnet-4-5 | $0.024756 | 1 |

**Canary (ONAYLANDI):** `gpt-4o-mini` × 2 variant × 5 operasyon × 15 case =
**10 çağrı**, yani modelin normal bir tam turu. Maliyet **~$0.010**, panelde
görünür, ve gerçek koşu yolunu (repair, limitör, defter, bütçe sigortası) birebir
çalıştırır.

**Ürettiği 150 satır GERÇEK VERİ olarak saklanır, atılmaz.** Yani bu koşu hem
ölçüm hem de `gpt-4o-mini`'nin ücretli fazdaki ilk üretim turudur; para iki kez
harcanmaz ve ücretli fazın kalan işi 3 modele (gpt-4.1, claude-haiku-4-5,
claude-sonnet-4-5) iner.

Bu, ilk plandaki "1 case" tanımından sapar; gerekçe yukarıdaki görünürlük
tablosudur ve sapma onaylandı.

## Canary adımları

1. Ön kontrol: `ATTEST` gerekmez (ücretli), fiyat tablosu dolu, bütçe sigortası
   `armed` (ikisi de doğrulandı), `--paid-generation` **kullanılmaz** (o yalnızca
   `check_model_access.py` bayrağıdır).
2. Koşu:
   ```bash
   python main.py --resume run_20261002_170046 \
     --endpoints "GET /get,POST /post,PUT /put,PATCH /patch,DELETE /delete" \
     --base-url https://httpbin.org \
     --generators "openai:gpt-4o-mini" \
     --tests-per-generator 150 --no-run \
     --output-dir outputs/free_phase
   ```
3. Koşu sonrası, defterden:
   - `model_returned` == `gpt-4o-mini` mi (sağlayıcı başka bir sürüme yönlendirmiş mi)
   - `input_tokens` / `output_tokens` toplamı
   - `cost_usd_billed` toplamı
   - `discarded_cases` (geçersiz JSON) ve `fallback_share`
4. **24 saat beklenir** (panel gecikmeli).
5. OpenAI panelinde (platform.openai.com → Usage) `gpt-4o-mini` satırının token
   ve maliyet değerleri okunur.
6. Karar (**token mutabakatı birincil**):
   - **Token sapması ≤%10 → canary PASS.** Ücretli faza devam edilir.
   - Token sapması >%10 → **DUR.** Defterin token kaydı yanlıştır; bütçe
     sigortası da bu kayda dayandığı için ücretli faza girilmez, sebep bulunur.
   - Token tutuyor ama dolar >%10 sapıyor → PASS engellenmez, ama fiyat tablosu
     güncellenmiş olabilir; fiyatlar yeniden çekilir ve tablo onaya sunulur.
     (Dolar ölçütü ikincildir çünkü panel yuvarlaması ve faturalama gecikmesi
     bizim hesabımızdan bağımsız hata kaynaklarıdır.)

## Tam ücretli faz

4 model × 150 test = **600 satır**. Sağlayıcı sırayla: önce OpenAI, sonra Claude.

`gpt-4o-mini`'nin 150 satırı **canary'den gelir ve saklanır**, yani canary PASS
olduktan sonra geriye **3 model × 150 = 450 satır** kalır (~$0.46, repair dahil
~$0.93).

### Tahmini maliyet

Ölçülmüş girdilerle (15-case prompt ~487 token girdi, ~1553 token çıktı):

| Model | 10 ana çağrı | repair dahil (~2×) |
|---|---|---|
| gpt-4o-mini | $0.0100 | $0.0201 |
| gpt-4.1 | $0.1340 | $0.2680 |
| claude-haiku-4-5 | $0.0825 | $0.1650 |
| claude-sonnet-4-5 | $0.2476 | $0.4951 |
| **TOPLAM** | **$0.4741** | **$0.9482** |

Yani ücretli fazın tamamı **yaklaşık 1 dolar** — $80 bütçenin ~%1'i. B4'te
hesaplanan $1.03–$1.92 aralığıyla tutarlı (o hesap daha muhafazakâr çıktı
tahminiyle yapılmıştı).

### Aktif güvenlikler

- **Bütçe sigortası**: $40 uyarı / $64 sert uyarı / $80 DURDURMA. Harcama
  defterden okunur ve fiyatlanamayan çağrılar için muhafazakâr üst tahmin de
  toplanır. Beklenen harcama ~$1 olduğundan eşiklerin tetiklenmesi sürpriz olur
  ve tetiklenirse **gerçek bir sorun** demektir.
- **Fail-closed**: fiyat tablosunda girdisi olmayan ücretli model koşmaz (exit 3).
- **Checkpoint**: her görev bitince diske yazılır; çökmede tek görev kaybedilir.
- **Defter**: her çağrının ham yanıtı, tam isteği, token ayrımı, maliyeti.

### ZORUNLU kontrol: defter-kabul ile CSV satırı karşılaştırması

**Her ücretli generator bittiğinde**, bir sonraki modele geçmeden önce:

```bash
python - <<'PY'
import csv, glob, json
from collections import Counter
csvp = sorted(glob.glob("outputs/free_phase/executed_testcases_*.csv"))[-1]
rows = list(csv.DictReader(open(csvp)))
csv_count = Counter(r["generator"] for r in rows)
led = glob.glob("outputs/free_phase/.calls/*/calls.jsonl")[0]
acc = Counter()
for line in open(led):
    r = json.loads(line)
    if r.get("record_type") == "run_header" or r.get("failed"): continue
    if r.get("call_type") == "fallback": continue
    acc[r.get("generator", "?")] += int(r.get("accepted_cases") or 0)
for g in sorted(acc):
    print(g, "defter", acc[g], "-> CSV", csv_count.get(g, 0))
PY
```

**Fark varsa KOŞU DURUR ve bildirilir.** Gerekçe: 2 Ekim'de ücretsiz fazda tam
bu sessiz kayıp yaşandı — repair çağrısı patlayınca ana çağrının çıktısı
atılıyordu ve defterde 206 kabul edilmiş case varken CSV'ye yalnızca 15 satır
girmişti. Hata düzeltildi (`generators/base.py`, Bulgu 6) ama ücretli fazda
**sessizce tekrar etmemesi** için bu karşılaştırma her modelden sonra zorunlu
kontroldür: orada kayıp, parası ödenmiş çıktının atılması demektir.

Beklenen fark kaynakları (kayıp sayılmaz, ama açıklanmalı):
- `num_cases` üstü kırpma: defter kabul > CSV satırı olabilir (hedef 150'ye
  kırpılır). Bu normaldir; **CSV < 150 ise** sorun vardır.
- Fallback satırları defterde `accepted_cases` değil `fallback` kaydı olarak
  görünür; `fallback_share` ile birlikte okunmalı.

### Durma koşulları

| Koşul | Davranış |

|---|---|
| Bütçe eşiği ($80) | `BudgetExceeded` → kalan görevler iptal, üretilen satırlar yazılır |
| Bakiye/kota hatası | **Devre kesici** (ölçüldü): ilk kalıcı hatada `_aborted=True`, o modelin kalan operasyonları hiç denenmez, fallback üretilmez, görev boş döner ve 'altyapi' işaretlenir |
| Defter-CSV farkı | **Manuel kontrol** (yukarıdaki blok). Kodda otomatik değil; her modelden sonra operatör çalıştırır ve fark varsa durur |
| 3 ardışık hata | **Şu an kodda YOK.** `RETRY_MAX_ATTEMPTS=3` operasyon başına çalışır, ama "3 ardışık başarısız operasyon → generator'ı durdur" kuralı yok. Gerekirse eklenir — ayrı onay |
| Kullanıcı iptali | `Ctrl-C` → `finally` bloğu checkpoint ve defteri flush eder, CSV yazılır |

### Model kimliği riski

`claude-sonnet-4-5`'in emeklilik tabanı ("not sooner than September 29, 2026")
29 Eylül'de doldu. Model hâlâ erişilebilir (5 Ekim'deki erişim kontrolünde PASS)
ve *deprecated* değil. Canary'de ve tam fazda `model_returned` alanı kontrol
edilir; sağlayıcı başka bir sürüm döndürürse bu bir **sapma olarak belgelenir**,
sessizce kabul edilmez.

## Koşu sonrası

- `scripts/validate_run_output.py --expected-rows 1350 --expected-generators 9
  --expected-per-generator 150`
- Defter mutabakatı: CSV satır toplamı ile defter token toplamı; boşa giden
  üretim ölçülür.
- `tokens_used` satır başına **tahsistir, ölçüm değildir** — raporlarda bu not
  otomatik basılır.

---

## Kabul edilen riskler (final rapora girecek)

**Karanlık uyanma belirsizliği (kullanıcı kararı: kabul edilebilir risk).**
`man launchd.plist`: launchd uyuyan makineyi uyandırmaz; `StartCalendarInterval`
işi **bir sonraki uyanışta** koşar ve kaçırılan tetikleyiciler **tek olaya
indirgenir**. Power Nap bu makinede etkin (`powernap 1`) ama Power Nap'in
karanlık uyanmasında `StartCalendarInterval` ajanlarının koştuğuna dair resmî bir
kaynak **bulunamadı** — doğrulanmadı, varsayılmadı.

Azaltma: kullanıcı `sudo pmset repeat wake MTWRFSU 11:05:00` komutunu kendisi
çalıştırır (tek tekrarlayan olay; `man pmset`: "you may only have one pair of
repeating events scheduled"). Bu komut çalıştırılmazsa ücretsiz faz durmaz,
yalnızca günlük turun zamanı kapak açılışına bağlı hale gelir. Gemini'nin RPD'si
gece yarısı Pasifik'te sıfırlandığı için gün içinde bir kez açılması yeterlidir.

**16:10 güvenlik kontrolü kaldırıldı (kullanıcı kararı).** RPM/TPM zaten
limitörün reaktif mekanizmasıyla korunuyor: `Retry-After` + jitter ve
`RETRY_MAX_ATTEMPTS`'ten ayrı, düşük bir reaktif deneme sınırı
(`REACTIVE_429_MAX_RETRIES=2`). İkinci bir uyanma olayı hem gereksiz hem de
`pmset` tek tekrarlayan çift desteklediği için mümkün değildi.
