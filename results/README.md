# Deney sonuç özeti (tez)

- Deney: 9 üretici (8 LLM + şablon tabanlı baseline) x 5 operasyon x 10 senaryo = tekrar başına 450 satır, 5 tekrar (planlanan 2.250 satır).
- Kod sürümü: commit 594fde1 (etiket: deney-run1-5). Beş tekrar aynı kod, aynı OpenAPI tanımı (specs/httpbin_5ops.json) ve aynı ayarlarla koşuldu.
- Üreticiler: gpt-4.1, gpt-4o-mini, gemini-2.5-flash, gemini-3.5-flash-lite, claude-sonnet-4-5, claude-haiku-4-5, gpt-oss-120b ve gpt-oss-20b (Groq), Traditional-Template (şablon tabanlı baseline).
- Operasyonlar (httpbin.org): GET /status/{code}, GET /get, POST /anything, GET /headers, GET /uuid.
- Satır durumları: saglam (LLM üretimi ve çalıştırılmış), fallback (şablonla tamamlanmış), bos_istek (istekten sonuç alınamamış: çoğunlukla HTTP istemcisinin reddettiği geçersiz header değerleri, bir satırda zaman aşımı), uretilemedi (sağlayıcı hatası nedeniyle üretilmemiş).
- Pass oranı pass/(pass+fail); yalnızca sağlam satırlardan hesaplanır. Traditional yalnızca status_code assertion'ı üretir, LLM'ler içerik assertion'ı da yazar; bu yüzden pass oranları üreticiler arasında doğrudan karşılaştırılamaz.
- Dosyalar: ozet_durum.csv (üretici özeti), siniflandirma.csv (satır bazlı durum), cost_by_generator.csv (çağrı defterinden maliyet ve token; çıktı token'ı için billable_output_tokens sütunu kullanılmalı, Gemini'nin düşünme token'ları dahildir), pass_ort_sd.csv (5 tekrar ortalama ve standart sapma).
- Ham veri (yanıt gövdeleri, çağrı defteri) repoda yok; yanıtlarda genel IP adresi bulunabildiği için paylaşılmadı.
