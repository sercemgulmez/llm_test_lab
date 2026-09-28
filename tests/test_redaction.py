"""K10 regresyonu: API anahtarlarinin loglardan redakte edilmesi.

Gercek anahtar bu testlerde KULLANILMAZ; tum ornekler sentetiktir.
"""

import re

from security.redaction import _PATTERNS, redact_secrets

SYNTHETIC = {
    "anthropic-key": "sk-ant-api03-" + "A" * 95,
    "openai-key-proj": "sk-proj-" + "B" * 156,
    "groq-key": "gsk_" + "C" * 52,
    "gemini-key": "AIza" + "D" * 35,
    "gemini-key-aq": "AQ.Ab8RN6" + "E" * 44,   # Google AI Studio yeni format
}


def _pattern(name: str) -> re.Pattern:
    return next(p for label, p in _PATTERNS if label == name)


def test_new_gemini_aq_format_is_matched_by_regex():
    """K10: AQ. onekli anahtar regex ile ESLESMELI (onceden eslesmiyordu)."""
    assert _pattern("gemini-key-aq").match(SYNTHETIC["gemini-key-aq"])


def test_old_aiza_pattern_does_not_match_aq_format():
    """Hatanin kok nedeni: eski desen AQ. formatini hic yakalamiyor."""
    assert _pattern("gemini-key").match(SYNTHETIC["gemini-key-aq"]) is None


def test_both_gemini_formats_are_redacted():
    for key in ("gemini-key", "gemini-key-aq"):
        value = SYNTHETIC[key]
        redacted = redact_secrets(f"Gemini hatasi: api_key={value} gecersiz")
        assert value not in redacted
        assert "[REDACTED]" in redacted


def test_all_configured_key_formats_are_redacted():
    for name, value in SYNTHETIC.items():
        redacted = redact_secrets(f"hata olustu: {value}")
        assert value not in redacted, f"{name} redakte edilmedi"


def test_aq_pattern_does_not_swallow_ordinary_text():
    """Kisa 'AQ.' benzeri metinler yanlislikla redakte edilmemeli."""
    harmless = "Sorgu AQ.x ile basliyor ve kisa."
    assert redact_secrets(harmless) == harmless


def test_redaction_is_idempotent():
    once = redact_secrets(f"key={SYNTHETIC['gemini-key-aq']}")
    assert redact_secrets(once) == once
