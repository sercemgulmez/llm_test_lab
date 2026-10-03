#!/bin/sh
set -eu

python3 -m http.server 8000 --directory specs >/tmp/httpbin-spec-server.$$.log 2>&1 &
SPEC_SERVER_PID=$!
trap 'kill "$SPEC_SERVER_PID" 2>/dev/null || true' EXIT INT TERM
sleep 2

python main.py \
  --base-url "https://httpbin.org" \
  --openapi-url "http://localhost:8000/httpbin_5ops.json" \
  --generators "traditional,groq:openai/gpt-oss-20b" \
  --tests-per-generator 50 \
  --budget-warn 4 \
  --budget-hard-warn 6 \
  --budget-stop 8 \
  --output-dir outputs/smoke3
