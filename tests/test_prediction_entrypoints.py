from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PREDICTION_SCRIPTS = sorted((REPO_ROOT / "predictions").glob("*.py"))


@pytest.mark.parametrize("script", PREDICTION_SCRIPTS, ids=lambda path: path.name)
def test_prediction_script_exits_successfully(script: Path) -> None:
    env = os.environ.copy()
    env["MPLCONFIGDIR"] = "/tmp"
    proc = subprocess.run(
        [sys.executable, "-B", str(script.relative_to(REPO_ROOT))],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
