#!/usr/bin/env python3
"""Lock the LSS/BAO prime-field correlation prediction before new data.

This artifact intentionally forbids amplitude or r0 re-normalization for future
survey tests. Existing BOSS data may be used as context, but the locked future
rule is zero-parameter: C_XI=62 and r0=0.6594900863537677 kpc.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from prime_field_util import C_XI_CANONICAL, R0_KPC_CANONICAL  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent.parent / "evidence" / "lss_bao_locked_prediction"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def xi_locked(r_mpc: float) -> float:
    """Zero-parameter locked correlation prediction."""
    r0_mpc = R0_KPC_CANONICAL / 1000.0
    phi = 1.0 / math.log(r_mpc / r0_mpc + 1.0)
    return C_XI_CANONICAL * phi * phi


def table() -> list[dict[str, float]]:
    radii = [5, 8, 10, 20, 30, 50, 75, 100, 120, 150, 200]
    return [{"r_mpc": float(r), "xi_locked": xi_locked(float(r))} for r in radii]


def main() -> int:
    rows = table()
    model_spec = {
        "formula": "xi(r)=C_XI/[log(r/r0+1)]^2",
        "C_XI": C_XI_CANONICAL,
        "r0_kpc": R0_KPC_CANONICAL,
        "free_parameters_allowed_on_future_data": 0,
    }
    spec_hash = hashlib.sha256(json.dumps(model_spec, sort_keys=True).encode()).hexdigest()
    out = {
        "artifact": "IF Theory LSS/BAO locked zero-parameter prediction",
        "status": "PREDICTION_LOCKED_FOR_FUTURE_SURVEYS",
        "model_spec": model_spec,
        "model_spec_sha256": spec_hash,
        "future_data_rule": {
            "allowed": "Apply the exact model_spec to new DESI/Euclid/Roman correlation data.",
            "forbidden": "No amplitude fit, no r0 fit, no bin-range tuning after seeing the new data.",
            "pass": "Pearson r(log xi) >= 0.93 and predeclared chi2/dof threshold for the survey covariance.",
            "fail": "Shape Pearson r(log xi) < 0.80 on a predeclared survey range, or post-hoc renormalization required.",
        },
        "locked_table": rows,
    }
    path = OUT_DIR / "lss_bao_locked_prediction.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"model_spec_sha256 = {spec_hash}")
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
