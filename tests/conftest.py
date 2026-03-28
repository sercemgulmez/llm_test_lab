import shutil
import uuid
from pathlib import Path

import pytest


@pytest.fixture
def tmp_path():
    """Windows izin sorunlarından bağımsız, çalışma alanı içi geçici klasör."""
    path = Path.cwd() / f"tmpcase-{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
