# Metodoloji notları — tez metnine girecek sapmalar

Bu dosya, deney sırasında alınan ve **tezde belgelenmesi gereken** sapmaları
tutar. Tez metni bu repoda değildir; aşağıdaki maddeler oraya elle taşınmalıdır.

---

## 1. Gemini'nin ücretli tier'a geçişi (3 Ekim 2026)

**Tez metnindeki "PHASE 1 ücretsiz" ifadesine eklenecek dipnot — birebir:**

> Gemini, hız kısıtı nedeniyle sonradan ücretli tier'a geçirildi, gerçek maliyet
> ~$0,20-0,30 (ledger'dan doğrulanacak).

### Gerekçe (dipnotun arkasındaki ölçüm)

Google AI Studio'nun free tier'ında bu projenin Gemini modelleri için yayımlanan
günlük istek limitleri şunlardı:

| Model | RPM | RPD | Güvenlik payı 0.8 ile efektif RPD |
|---|---|---|---|
| `gemini-2.5-flash` | 5 | **2** | **1 istek/gün** |
| `gemini-3.5-flash-lite` | 15 | **5** | 4 istek/gün |

Model başına 10 ana çağrı gerektiğinden (2 prompt variant × 5 operasyon),
`gemini-2.5-flash` tek başına **~10 gün** sürecekti. 2 Ekim'de ölçülen gerçek
davranış bunu doğruladı: modelin o günkü tek isteği Google'ın
`503 UNAVAILABLE — "This model is currently experiencing high demand"` yanıtına
gitti ve gün boşa geçti.

3 Ekim'de projeye faturalandırma açıldı ve $5 bütçe **uyarısı** (enforcement
değil, yalnızca alert) kuruldu.

### Kod tarafındaki sonuçları

Bu, yalnızca bir ödeme kararı değil; kodda sınıflandırma değişikliği gerektirdi:

- `config.FREE_ONLY_PROVIDERS`: `{Gemini, Groq}` → `{Groq}`
- `config.PAID_PROVIDERS`: `{OpenAI, Claude}` → `{OpenAI, Claude, Gemini}`
- `pricing.PRICE_TABLE`: Gemini girdileri `billed=False` → `billed=True`
- `free_tier_limits.json`: Gemini bloğu kaldırıldı (limitör yalnızca FREE_ONLY
  sağlayıcıları yönetir; ücretlilerde koruma bütçe sigortasıdır)

**Neden kritik:** Gemini `FREE_ONLY_PROVIDERS` içinde kalsaydı `pricing.cost_for`
her Gemini çağrısına `cost_usd_billed = 0.0` yazardı — yani gerçek para
harcanırken defter $0.00 gösterir ve bütçe sigortası bu harcamayı **hiç
görmezdi**. Fiyat tablosundaki `billed` bayrağı tek başına yetmez; FREE_ONLY
listesi onu bilerek ezer (bu ikinci kilit, yanlış fiyat girdisine karşı
konulmuştu).

### Karşılaştırılabilirlik üzerindeki etkisi

Gemini artık ücretli tier'da koştuğu için **hız kısıtı kalktı**, ama üretim
parametreleri (prompt, `max_output_tokens=8192`, case sayısı 15) **değişmedi**.
Yani üretilen veri free tier'da üretilecek veriyle aynı niteliktedir; değişen
yalnızca çağrıların ne kadar hızlı yapılabildiğidir.

Tez tablolarında Gemini'nin maliyeti artık **gerçek fatura** olarak raporlanır;
Groq ise free tier'da kaldığı için `cost_usd_billed = 0` ve yalnızca
`cost_usd_list_equivalent` taşır. Bu asimetri tabloda belirtilmelidir.

### Otomasyon üzerindeki etkisi

`scripts/free_phase_tick.sh` (günlük launchd tetikleyicisi) Gemini modellerini
listesinden **çıkardı**. Gerekçe: zamanlanmış bir iş, kimse başında olmadan her
gün para harcamamalı. Gemini'nin kalan işi elle, tek seferde koşuldu.

---

## 2. Groq'ta `max_tokens` artırımı (2 Ekim 2026)

Groq reasoning modellerinde (`gpt-oss-120b`, `gpt-oss-20b`) reasoning bütçesi
output'u tüketiyordu: `max_tokens=3000` tavanında çıktının ~1500 token'ı
reasoning'e gidiyor ve yanıt `finish_reason='length'` ile kesiliyordu.

`max_tokens` **3000 → 3700**'e çıkarıldı. **Case sayısı (15) değişmedi.**

Üst sınırı Groq'un TPM'i belirledi: limitör TPM'e karşı
`girdi tahmini + max_tokens` rezerve ediyor; yayımlanan TPM 8000, güvenlik payı
0.8 ile efektif 6400. Ana çağrılar için tavan 5639, repair çağrıları da çalışsın
diye 3736 — bu yüzden 3700 seçildi. (İlk istenen 8192 matematiksel olarak
imkânsızdı: 761 + 8192 = 8953, yayımlanan 8000'i bile aşıyor.)

## 3. Groq'ta repair mekanizması fiilen devre dışı

Groq'ta repair prompt'ları 6655–7422 token rezerve ediyor ve efektif TPM 6400'ü
aşıyor; `build_repair_prompt` kabul edilmiş case'leri prompt'a gömdüğü için
prompt kabul edilen case sayısıyla büyüyor.

Ana çağrı kurtarma mekanizması (geçersiz JSON'dan nesne-bazlı kurtarma) yeterli
olduğu için şu an veri kaybına yol açmıyor (0 fallback, 158/158 case kabul).
İleride fallback oranı yükselirse bu kısıt yeniden değerlendirilmeli.
