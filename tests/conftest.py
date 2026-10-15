import os
import shutil

import pytest


@pytest.fixture
def ccx_path() -> str:
    """Path to the ccx executable, from env var (never hardcoded in source)."""
    path = os.environ.get("SOUNDBOARD_LAB_CCX_PATH")
    if not path or not shutil.which(path) and not os.path.isfile(path):
        pytest.skip(
            "Set SOUNDBOARD_LAB_CCX_PATH to the ccx executable to run solver tests"
        )
    return path
