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


def test_dark_matter_notebooks_enable_exact_kernel() -> None:
    for notebook_name in DARK_MATTER_NOTEBOOKS:
        output = executed_notebook_output(notebook_name)
        assert "Exact Kernel: ENABLED" in output
        assert "will use float computations" not in output
        assert "float_contamination" in output
