"""Utility for redacting secrets from strings before logging or displaying."""

from __future__ import annotations

import re

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
            r"\s*=\s*\S+",
            re.IGNORECASE,
        ),
    ),
]


def redact_secrets(text: str) -> str:
    """Replace any detected secrets in *text* with ``[REDACTED]``."""
    for _name, pattern in _PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    return text
