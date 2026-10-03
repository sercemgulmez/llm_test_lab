#!/bin/bash
# Ucretsiz fazin KISA tetiklenisi (launchd her gun calistirir).
#
# Tasarim:
#   * Kisa kalir: o an kotasi acik gorevleri isler, kotasi bitenleri atlar, biter.
#     Limitor LIMITER_MAX_WAIT_SECONDS'i asan beklemede QuotaExhausted firlatir,
#     yani burada 24 saat sleep YOKTUR.
#   * caffeinate YALNIZCA aktif kosu boyunca sarar; bekleme saatlerinde degil.
#   * YALNIZCA FREE_ONLY generator'lari secer. Ucretli bir modele kazara
#     dokunmanin yolu yok.
#   * HER ZAMAN 0 ile cikar. Kota tukenmesi normal bir sonuctur; sifir-disi cikis
#     launchd'nin KeepAlive(SuccessfulExit=false) kuralini tetikleyip gereksiz
#     yeniden denemeye yol acardi. Gercek hata log'a ve status.txt'e yazilir.
set -u

ROOT="/Users/suleymansercemgulmez/Projeler/llm_test_lab"
RUN_ID="run_20261002_170046"
OUT_DIR="$ROOT/outputs/free_phase"
PROGRESS_DIR="$ROOT/outputs/.progress"
STATUS_FILE="$PROGRESS_DIR/status.txt"
TICK_LOG="$PROGRESS_DIR/tick.log"
PYTHON="$ROOT/.venv/bin/python"

# YALNIZCA ucretsiz generator'''lar. Gemini 3 Ekim 2026'''da ucretli tier'''a gecti ve
# BU LISTEDEN CIKARILDI: zamanlanmis bir is, kimse basinda olmadan her gun para
# harcamamali. Gemini'''nin kalan isi elle, tek seferde kosuluyor.
FREE_GENERATORS="traditional,groq:openai/gpt-oss-120b,groq:openai/gpt-oss-20b"
ENDPOINTS="GET /get,POST /post,PUT /put,PATCH /patch,DELETE /delete"

mkdir -p "$PROGRESS_DIR"
cd "$ROOT" || exit 0

STAMP="$(date -Iseconds)"
echo "=== $STAMP tick basladi ===" >> "$TICK_LOG"

# Ayni anda iki tetikleyici (11:10 ve 16:10, ya da uyku sonrasi birlesmis olay)
# ust uste binmesin: kilit alinamazsa sessizce cik.
LOCK="$PROGRESS_DIR/tick.lock"
if ! mkdir "$LOCK" 2>/dev/null; then
    echo "$STAMP | ATLANDI: baska bir tick calisiyor ($LOCK)" >> "$TICK_LOG"
    exit 0
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT

# caffeinate: -i bos beklemeyle uykuyu engeller, komut bitince caffeinate de biter.
caffeinate -i "$PYTHON" main.py \
    --resume "$RUN_ID" \
    --endpoints "$ENDPOINTS" \
    --base-url https://httpbin.org \
    --generators "$FREE_GENERATORS" \
    --tests-per-generator 150 \
    --no-run \
    --output-dir "$OUT_DIR" >> "$TICK_LOG" 2>&1
RUN_EXIT=$?
echo "$STAMP | main.py exit=$RUN_EXIT" >> "$TICK_LOG"

# Durum satiri: tarih, model basina tamamlanan/bekleyen, tahmini kalan gun.
"$PYTHON" scripts/free_phase_status.py \
    --output-dir "$OUT_DIR" --run-id "$RUN_ID" \
    --append-to "$STATUS_FILE" >> "$TICK_LOG" 2>&1 \
    || echo "$STAMP | UYARI: durum satiri yazilamadi" >> "$STATUS_FILE"

if [ "$RUN_EXIT" -ne 0 ]; then
    echo "$STAMP | NOT: main.py sifir-disi cikti ($RUN_EXIT); ayrinti icin tick.log" \
        >> "$STATUS_FILE"
fi

exit 0
