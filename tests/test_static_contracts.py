from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_python_sources_do_not_catch_base_exception() -> None:
    offenders: list[str] = []
    for path in sorted(REPO_ROOT.glob("**/*.py")):
        if any(part.startswith(".") or part == "__pycache__" for part in path.parts):
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler) and isinstance(node.type, ast.Name):
                if node.type.id in {"Exception", "BaseException"}:
                    offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")

    assert offenders == []
