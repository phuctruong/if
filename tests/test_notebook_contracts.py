from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DARK_MATTER_NOTEBOOKS = (
    "dark_matter_sdss.ipynb",
    "dark_matter_desi.ipynb",
    "dark_matter_euclid.ipynb",
)


def executed_notebook_output(notebook_name: str) -> str:
    with tempfile.TemporaryDirectory(prefix="if-notebook-contract-") as tmpdir:
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "jupyter",
                "nbconvert",
                "--execute",
                "--to",
                "notebook",
                "--output-dir",
                tmpdir,
                notebook_name,
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]

        executed_path = Path(tmpdir) / notebook_name
        notebook = json.loads(executed_path.read_text())
        output_parts: list[str] = []
        for cell in notebook["cells"]:
            if cell.get("cell_type") != "code":
                continue
            for output in cell.get("outputs", []):
                if output.get("output_type") == "stream":
                    text = output.get("text", "")
                    output_parts.append("".join(text) if isinstance(text, list) else text)
                elif output.get("output_type") == "error":
                    output_parts.append(f"{output.get('ename')}: {output.get('evalue')}")
        return "".join(output_parts)


def test_dark_matter_notebooks_are_driver_bearing_and_honest() -> None:
    """Contract updated 2026-06-12 (referee loop).

    The previous contract executed the notebooks under a 60s timeout and
    asserted the exact-kernel cell ran. That contract only worked because
    the notebooks were stubs (config + a verification cell that returned
    VALIDATED unconditionally — review Finding N1). The working
    driver-bearing versions (restored from 12473c8) run real multi-minute
    analyses and cannot complete in CI; their EXECUTION is covered by:
      - evidence/historical_rerun/*/ (sealed era-faithful reruns), and
      - adversarial/survey_clustering_replication.py (fast standalone).
    This contract now statically requires each notebook to (a) carry the
    real driver, (b) carry the provenance/σ-semantics banner, and
    (c) contain no always-pass verification theater.
    """
    import json

    for notebook_name in DARK_MATTER_NOTEBOOKS:
        nb = json.loads((REPO_ROOT / notebook_name).read_text())
        first = "".join(nb["cells"][0].get("source", []))
        assert "REFEREE BANNER" in first or "RESTORED WORKING VERSION" in first, (
            f"{notebook_name}: provenance banner missing")
        code = "\n".join("".join(c["source"]) for c in nb["cells"]
                          if c["cell_type"] == "code")
        assert "TEST_TYPE" in code and "jackknife" in code.lower(), (
            f"{notebook_name}: analysis driver missing — stub regression")
        assert '"status": "VALIDATED"' not in code and "'status': 'VALIDATED'" not in code, (
            f"{notebook_name}: unconditional-VALIDATED theater regression")
