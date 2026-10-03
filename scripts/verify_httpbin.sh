#!/bin/sh
set -eu

OUT="outputs/verify_httpbin.txt"
mkdir -p "$(dirname "$OUT")"

{
  for c in 200 400 404 500; do
    curl -s -o /dev/null -w "status/$c -> %{http_code} bytes=%{size_download}\n" "https://httpbin.org/status/$c"
  done

  curl -s -o /dev/null -w "GET /get?q=smoke -> %{http_code} content_type=%{content_type}\n" \
    "https://httpbin.org/get?q=smoke"

  curl -s -o /dev/null -w "POST /anything valid -> %{http_code} content_type=%{content_type}\n" \
    -H "Content-Type: application/json" \
    -d '{"message":"smoke"}' \
    "https://httpbin.org/anything"

  curl -s -o /dev/null -w "POST /anything malformed -> %{http_code}\n" \
    -H "Content-Type: application/json" \
    -d '{bozuk' \
    "https://httpbin.org/anything"

  curl -s -o /dev/null -w "GET /headers -> %{http_code} content_type=%{content_type}\n" \
    "https://httpbin.org/headers"

  curl -s -o /dev/null -w "GET /uuid -> %{http_code} content_type=%{content_type}\n" \
    "https://httpbin.org/uuid"
} | tee "$OUT"
