# Security Policy — LLM Test Lab

## Overview

LLM Test Lab is a public academic/research repository. Because the repository is publicly visible, strict rules apply to how API keys and other secrets are handled.

---

## API Key Handling Policy

| Rule | Detail |
|---|---|
| Keys in environment only | All API keys are read from environment variables at runtime. |
| No keys in source code | Never hardcode a key in `.py`, `.yml`, `.json`, `.md`, or any other file. |
| No keys in `.env.example` | `.env.example` contains placeholder values only and is safe to commit. |
| `.env` is git-ignored | `.env` (and `.env.*`, `.env.local`) is listed in `.gitignore` and must never be committed. |
| Safe error messages | `security/secret_loader.py` raises errors that include only the variable **name**, never the value. |
| Redacted logs | `security/redaction.py` masks known key patterns before they appear in logs or HTTP responses. |

---

## Environment Variables

| Variable | Provider |
|---|---|
| `OPENAI_API_KEY` | OpenAI |
| `ANTHROPIC_API_KEY` | Anthropic / Claude |
| `GROQ_API_KEY` | Groq |
| `GEMINI_API_KEY` | Google Gemini |

### Local setup

```bash
cp .env.example .env
# Fill in your keys — never commit this file
```

---

## Secret Scanning

A scanner script checks the repository for potential secrets:

```bash
python scripts/scan_secrets.py
```

- Reports: `[POTENTIAL SECRET] <file>:<line> — <pattern-name>` (value never printed)
- Exit code `0`: repository is clean
- Exit code `1`: potential secrets detected — review and act immediately

GitHub Actions runs this scanner on every push and pull request via `.github/workflows/security-check.yml`.

---

## Key Rotation Checklist

Run through this checklist whenever a key may have been exposed:

- [ ] Identify which key(s) were exposed and in which files/commits
- [ ] Revoke the key at the provider dashboard immediately (see links below)
- [ ] Generate a replacement key
- [ ] Update your local `.env` with the new key
- [ ] Verify the application works with the new key
- [ ] Purge the old key from git history (see incident response below)
- [ ] Notify all collaborators to pull the rewritten history

### Provider dashboards

- OpenAI: <https://platform.openai.com/api-keys>
- Anthropic: <https://console.anthropic.com/settings/keys>
- Groq: <https://console.groq.com/keys>
- Google AI / Gemini: <https://aistudio.google.com/app/apikey>

---

## Incident Response: Key Accidentally Committed

A key in git history is **public**, even if you delete it in the next commit.

### Step 1 — Revoke immediately

Go to the provider dashboard and revoke the exposed key before doing anything else.

### Step 2 — Purge from history

Using `git filter-repo` (recommended):

```bash
pip install git-filter-repo
git filter-repo --path .env --invert-paths
```

Or with BFG Repo Cleaner:

```bash
java -jar bfg.jar --delete-files .env
git reflog expire --expire=now --all
git gc --prune=now --aggressive
```

### Step 3 — Force-push

```bash
git push origin --force --all
git push origin --force --tags
```

### Step 4 — Notify collaborators

All clones of the repository still contain the old history. Every collaborator must:

```bash
git fetch origin
git reset --hard origin/main
```

### Step 5 — Rotate again

Even after purging history, treat the revoked key as permanently compromised. Use only the newly generated replacement key.

---

## Security Utilities Reference

| File | Purpose |
|---|---|
| `security/redaction.py` | `redact_secrets(text) -> str` — masks known key patterns |
| `security/secret_loader.py` | `get_api_key(provider)` and `get_api_key_from_env(env_var)` — safe loaders |
| `scripts/scan_secrets.py` | Repository-wide secret scanner |
| `.github/workflows/security-check.yml` | CI enforcement of scanner on every push/PR |
