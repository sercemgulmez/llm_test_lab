# Web UI REST API Reference

All endpoints are served by `app.py` on `http://localhost:5000`.

Job endpoints (all except `/health`, `/upload`, `/run`) require the `X-Job-Token` header set to the token returned by `/run`.

## Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/health` | — | App liveness check. Returns `{"status": "ok"}`. |
| `POST` | `/upload` | — | Upload a curl/OpenAPI file. Accepts `.txt`, `.curl`, `.http`; max 1 MB. Returns `{"path": "<server-side path>"}`. |
| `POST` | `/run` | — | Start a background job. Returns `{"job_id": "...", "token": "..."}`. |
| `GET` | `/stream/<job_id>` | Token | Server-Sent Events log stream. Each event is `{"type": "log"|"error", "text": "..."}`. Ends with a `null` sentinel. |
| `GET` | `/result_data/<job_id>` | Token | Raw executed test case rows as JSON array. |
| `GET` | `/metrics/<job_id>` | Token | Per-generator pass rate metrics. |
| `GET` | `/comparison/<job_id>` | Token | Pairwise similarity + spectral ranking summary. |
| `GET` | `/generation_quality/<job_id>` | Token | Repair-loop stats: parsed/valid/repaired/fallback counts per generator+operation. |
| `GET` | `/run_info/<job_id>` | Token | Experiment config snapshot (models, prompt variants, operation count, config values). |
| `POST` | `/cancel/<job_id>` | Token | Request graceful job cancellation. Job stops at next `_raise_if_cancelled` checkpoint. |
| `GET` | `/download/<job_id>` | Token | Download result CSV directly. |
| `GET` | `/download_report/<job_id>` | Token | ZIP archive containing: result CSV, metrics CSV, `run_info.json`, `comparison.json`, `generation_quality.json`. |

## `/run` Request Body

```json
{
  "source": "curl | openapi",
  "curl_file_path": "<path returned by /upload>",
  "openapi_url": "https://...",
  "base_url": "https://api.example.com/v1",
  "auth_token": "Bearer ...",
  "headers": ["X-Custom: value"],
  "cookie": "name=value; other=value",
  "selected_generators": ["traditional", "openai:gpt-4o-mini", "claude:claude-sonnet-4-6"],
  "num_cases": 10,
  "no_run": false,
  "output_dir": "outputs"
}
```

`selected_generators` keys follow the pattern `provider:model` (e.g. `openai:gpt-4o-mini`, `gemini:gemini-2.0-flash`, `claude:claude-sonnet-4-6`, `groq:llama-3.3-70b-versatile`). Use `traditional` for the template baseline. Omit the field (or send an empty list) to include all generators.

## Job Lifecycle

```
POST /run  →  status: "running"
GET /stream/<id>  →  live log lines
  (job finishes)  →  status: "done" | "error" | "cancelled"
GET /result_data/<id>  →  results
GET /download_report/<id>  →  ZIP
```
