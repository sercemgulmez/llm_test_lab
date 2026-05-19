#!/usr/bin/env python3
"""Scan the repository for potential leaked secrets.

Prints only the file path, line number, and pattern name — never the
actual secret value.  Exits with code 1 if any potential secret is found,
0 if the repository is clean.

Usage:
    python scripts/scan_secrets.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# ── Patterns ────────────────────────────────────────────────────────────────

_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("anthropic-key", re.compile(r"sk-ant-api03-[A-Za-z0-9_\-]{80,}")),
    ("openai-key-proj", re.compile(r"sk-proj-[A-Za-z0-9_\-]{40,}")),
    ("openai-key", re.compile(r"sk-[A-Za-z0-9]{48}")),
    ("groq-key", re.compile(r"gsk_[A-Za-z0-9]{52}")),
    ("gemini-key", re.compile(r"AIza[A-Za-z0-9_\-]{35}")),
    ("bearer-token", re.compile(r"Bearer\s+[A-Za-z0-9\-_\.]{20,}", re.IGNORECASE)),
    (
        "env-assignment",
        re.compile(
            r"(OPENAI_API_KEY|ANTHROPIC_API_KEY|GROQ_API_KEY|GEMINI_API_KEY)"
            r"\s*=\s*(?!your_|<|#|\s*$)\S+",
            re.IGNORECASE,
        ),
    ),
]

# ── Directories and files to skip ───────────────────────────────────────────

_SKIP_DIRS: set[str] = {
    ".git",
    "venv",
    ".venv",
    "env",
    "__pycache__",
    "node_modules",
    "outputs",
    "uploads",
    ".pytest_cache",
}

_SKIP_DIR_PREFIXES: tuple[str, ...] = ("runtime_uploads",)

_SCAN_EXTENSIONS: set[str] = {
    ".py",
    ".env",
    ".yml",
    ".yaml",
    ".json",
    ".txt",
    ".md",
    ".sh",
    ".cfg",
    ".toml",
    ".ini",
    ".example",
}

# ── Files whose placeholders look like real keys (safe to skip) ─────────────
_ALLOWLISTED_FILES: set[str] = {
    ".env.example",
    "security/redaction.py",
    "scripts/scan_secrets.py",
}


def _should_skip_dir(part: str) -> bool:
    if part in _SKIP_DIRS:
        return True
    return any(part.startswith(prefix) for prefix in _SKIP_DIR_PREFIXES)


def scan(root: Path) -> list[tuple[Path, int, str]]:
    hits: list[tuple[Path, int, str]] = []

    for path in sorted(root.rglob("*")):
        # Skip directories we don't want to descend into
        if any(_should_skip_dir(part) for part in path.parts):
            continue

        if not path.is_file():
            continue

        rel = path.relative_to(root)

        # Skip allowlisted files (they intentionally contain pattern text)
        if str(rel) in _ALLOWLISTED_FILES:
            continue

        # .env and .env.local are gitignored private files expected to hold real keys.
        # We only scan committed files; skip these private env files.
        if path.name in {".env", ".env.local"}:
            continue

        if path.suffix not in _SCAN_EXTENSIONS:
            continue

        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue

        for lineno, line in enumerate(lines, start=1):
            for pattern_name, pattern in _PATTERNS:
                if pattern.search(line):
                    hits.append((rel, lineno, pattern_name))
                    break  # one hit per line is enough

    return hits


def main() -> int:
    root = Path(__file__).parent.parent.resolve()
    hits = scan(root)

    if not hits:
        print("Secret scan complete — no potential secrets found.")
        return 0

    print(f"[SECRET SCAN] {len(hits)} potential secret(s) detected:\n")
    for filepath, lineno, pattern_name in hits:
        print(f"  [POTENTIAL SECRET] {filepath}:{lineno} — {pattern_name}")

    print(
        "\nAction required: review the files above and rotate any real keys "
        "that may have been exposed."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
