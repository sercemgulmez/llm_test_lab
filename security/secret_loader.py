"""Centralised, safe API key loader.

Keys are always read from environment variables.
Error messages include only the variable *name*, never the value.
"""

from __future__ import annotations

import os

_ENV_VARS: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "claude": "ANTHROPIC_API_KEY",
    "groq": "GROQ_API_KEY",
    "gemini": "GEMINI_API_KEY",
}


def get_api_key(provider: str) -> str:
    """Return the API key for *provider* from the environment.

    Raises
    ------
    KeyError
        If *provider* is not a recognised provider name.
    RuntimeError
        If the environment variable is not set or empty.
    """
    env_var = _ENV_VARS.get(provider.lower())
    if env_var is None:
        supported = ", ".join(sorted(_ENV_VARS))
        raise KeyError(f"Unknown provider '{provider}'. Supported: {supported}")
    return get_api_key_from_env(env_var)


def get_api_key_from_env(env_var: str) -> str:
    """Return the value of *env_var*, raising RuntimeError if absent.

    Use this when you already know the environment variable name (e.g. when
    reading a class-level ``_api_key_env`` attribute in a generator).

    Error messages include only the variable *name*, never the value.
    """
    key = os.getenv(env_var)
    if not key:
        raise RuntimeError(f"{env_var} environment variable is not set.")
    return key
