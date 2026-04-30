from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from audits import test_cross_validation
from predictions import desi_bao_test

REPO_ROOT = Path(__file__).resolve().parents[1]


def run_script(path: str, *, input_text: str | None = None) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["MPLCONFIGDIR"] = "/tmp"
    return subprocess.run(
        [sys.executable, "-B", path],
        cwd=REPO_ROOT,
        env=env,
        input=input_text,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_cross_validation_main_reports_success_when_assertions_pass() -> None:
    assert test_cross_validation.main() is True


def test_audit_scripts_run_directly_from_repo_root() -> None:
    for path in (
        "audits/test_synthetic_validation.py",
        "audits/test_verification_ladder.py",
    ):
        proc = run_script(path)
        assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]


def test_sdss_util_defaults_to_no_download_without_stdin() -> None:
    proc = run_script("sdss_util.py", input_text="")
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    assert "No download performed." in proc.stdout


def test_desi_two_sigma_tension_is_not_a_process_failure() -> None:
    assert desi_bao_test.exit_code_for_p_value(0.0442) == 0
    assert desi_bao_test.exit_code_for_p_value(0.0031) == 0
    assert desi_bao_test.exit_code_for_p_value(0.0030) == 1
