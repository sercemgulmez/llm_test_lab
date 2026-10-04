#!/usr/bin/env bash
# kullanim: scripts/run_all.sh <1|2|3>
set -euo pipefail
cd "$(dirname "$0")/.."
RUN="${1:?kullanim: scripts/run_all.sh <1|2|3>}"
SPEC="http://127.0.0.1:8000/httpbin_5ops.json"
[ "$(curl -s -o /dev/null -w '%{http_code}' "$SPEC")" = "200" ] || { echo "Spec sunucusu kapali"; exit 1; }
[ -z "$(git status --short)" ] || { echo "Calisma agaci kirli: once commit et"; exit 1; }
PY=".venv/bin/python"
[ -x "$PY" ] || { echo ".venv/bin/python yok"; exit 1; }
OUT="outputs/run_${RUN}"
[ ! -e "$OUT" ] || { echo "$OUT zaten var"; exit 1; }
GENS="traditional,openai:gpt-4.1,openai:gpt-4o-mini,gemini:gemini-2.5-flash,gemini:gemini-3.5-flash-lite,claude:claude-sonnet-4-5,claude:claude-haiku-4-5,groq:openai/gpt-oss-120b,groq:openai/gpt-oss-20b"
"$PY" main.py --base-url "https://httpbin.org" --openapi-url "$SPEC" \
  --generators "$GENS" --tests-per-generator 50 \
  --budget-warn 4 --budget-hard-warn 6 --budget-stop 8 \
  --output-dir "$OUT"
