"""Kosu basligi: hangi kodun, hangi ayarlarla ve hangi beyanla kostugu.

Kosu sonrasi "bu satirlar hangi surumle uretildi" sorusu ancak o an kaydedilirse
cevaplanabilir; sonradan geri getirilemez. Bu yuzden basligi hem run_info
JSON'una hem cagri defterinin ilk kaydina yaziyoruz.

Hicbir alan sir ICERMEZ: anahtar, bakiye veya kota degeri yazilmaz.
"""

from __future__ import annotations

import logging
import subprocess
from datetime import datetime
from pathlib import Path

import attestation
import config

_logger = logging.getLogger(__name__)


def _git(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(config.PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def git_state() -> dict:
    """Commit, dal ve calisma agacinin temiz olup olmadigi.

    Git yoksa ya da burasi bir depo degilse alanlar None doner ve cagiran taraf
    UYARI verir — sessizce "temiz" denmez.
    """
    commit = _git("rev-parse", "HEAD")
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    status = _git("status", "--porcelain")
    return {
        "commit": commit,
        "branch": branch,
        "working_tree_clean": (status == "") if status is not None else None,
        "dirty_paths": (
            [line[3:] for line in status.splitlines()][:20] if status else []
        ),
    }


def build(run_id: str, limiter, budget_thresholds: dict, budget_armed: bool) -> dict:
    """Kosu basligi sozlugu."""
    git = git_state()
    return {
        "record_type": "run_header",
        "run_id": run_id,
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "git": git,
        "rate_limits": limiter.effective_limits(),
        "safety_factor_default": _safety_factor_default(),
        "attestation": attestation.attestation_record(),
        "budget": {"armed": budget_armed, "thresholds": dict(budget_thresholds)},
        "llm_request_timeout": dict(config.LLM_REQUEST_TIMEOUT),
        "free_only_providers": sorted(config.FREE_ONLY_PROVIDERS),
        "paid_providers": sorted(config.PAID_PROVIDERS),
    }


def _safety_factor_default() -> float | None:
    import json

    import rate_limiter

    try:
        data = json.loads(Path(rate_limiter.LIMITS_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    value = data.get("safety_factor_default")
    return float(value) if value is not None else None


def log(header: dict) -> None:
    """Basligi kosu ciktisina yazar (sir icermez)."""
    git = header["git"]
    if git["commit"] is None:
        _logger.warning("  [baslik] git bilgisi alinamadi — commit/dal kaydedilemedi.")
    else:
        clean = git["working_tree_clean"]
        _logger.info(
            "  [baslik] commit=%s dal=%s calisma_agaci=%s",
            git["commit"][:12], git["branch"],
            "TEMIZ" if clean else "KIRLI",
        )
        if clean is False:
            _logger.warning(
                "  [baslik] Calisma agaci KIRLI: uretilen satirlar tam olarak bu "
                "commit'e karsilik gelmiyor. Degisen yollar: %s",
                ", ".join(git["dirty_paths"]) or "(bilinmiyor)",
            )
    declared = header["attestation"]["declared_providers"]
    _logger.info("  [baslik] free-tier beyani: %s", ", ".join(declared) or "YOK")
