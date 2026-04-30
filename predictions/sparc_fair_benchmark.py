#!/usr/bin/env python3
"""Fair SPARC benchmark harness for IF, MOND, and NFW-like halo fits.

Default mode runs a deterministic smoke subset so CI stays fast. Use
`--max-galaxies 0` to run every local SPARC rotmod file.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.optimize import minimize, minimize_scalar

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from predictions.sparc_corrected_log_potential import (  # noqa: E402
    SPARC_DIR,
    SPARC_TABLE,
    G_kpc_kms_msun,
    load_rotmod,
    parse_sparc_table,
)
from predictions.sparc_per_galaxy_ml import evaluate_galaxy as evaluate_if  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent.parent / "evidence" / "sparc_fair_benchmark"
OUT_DIR.mkdir(parents=True, exist_ok=True)

A0_KM2_S2_PER_KPC = 1.2e-10 / (1e6 / 3.085677581491367e19)
H0_KM_S_KPC = 0.0674


def _prepared_arrays(path: Path, min_floor_err: float = 1.0) -> tuple[np.ndarray, ...] | None:
    data = load_rotmod(path)
    radius = data["R"]
    keep = radius > 0
    radius = radius[keep]
    if len(radius) < 3:
        return None
    return (
        radius,
        data["Vobs"][keep],
        np.maximum(data["errV"][keep], min_floor_err),
        data["Vgas"][keep],
        data["Vdisk"][keep],
        data["Vbul"][keep],
    )


def _chi2(v_model: np.ndarray, v_obs: np.ndarray, err_v: np.ndarray) -> float:
    return float(np.sum(((v_obs - v_model) / err_v) ** 2))


def _bic(chi2: float, n_points: int, n_params: int) -> float:
    return chi2 + n_params * math.log(n_points)


def evaluate_mond(path: Path) -> Optional[dict]:
    arrays = _prepared_arrays(path)
    if arrays is None:
        return None
    radius, v_obs, err_v, v_gas, v_disk, v_bul = arrays

    def objective(y: float) -> float:
        vbar_sq = np.maximum(v_gas ** 2 + y * (v_disk ** 2 + v_bul ** 2), 0.0)
        g_bar = np.maximum(vbar_sq / radius, 1e-12)
        x = np.sqrt(g_bar / A0_KM2_S2_PER_KPC)
        g_mond = g_bar / np.maximum(1.0 - np.exp(-x), 1e-12)
        v_model = np.sqrt(np.maximum(g_mond * radius, 0.0))
        return _chi2(v_model, v_obs, err_v)

    res = minimize_scalar(objective, bounds=(0.1, 1.0), method="bounded", options={"xatol": 1e-3})
    chi2 = float(res.fun)
    dof = max(len(radius) - 1, 1)
    return {
        "chi2": chi2,
        "dof": dof,
        "chi2_per_dof": chi2 / dof,
        "n_params": 1,
        "bic": _bic(chi2, len(radius), 1),
        "Y_fitted": float(res.x),
    }


def _nfw_v2(radius: np.ndarray, v200: float, concentration: float) -> np.ndarray:
    r200 = v200 / (10.0 * H0_KM_S_KPC)
    x = np.maximum(radius / r200, 1e-8)
    cx = concentration * x
    numerator = np.log1p(cx) - cx / (1.0 + cx)
    denominator = x * (np.log1p(concentration) - concentration / (1.0 + concentration))
    return np.maximum(v200 * v200 * numerator / np.maximum(denominator, 1e-12), 0.0)


def evaluate_nfw(path: Path) -> Optional[dict]:
    arrays = _prepared_arrays(path)
    if arrays is None:
        return None
    radius, v_obs, err_v, v_gas, v_disk, v_bul = arrays

    def objective(params: np.ndarray) -> float:
        y, log_v200, log_c = params
        v200 = math.exp(log_v200)
        concentration = math.exp(log_c)
        vbar_sq = np.maximum(v_gas ** 2 + y * (v_disk ** 2 + v_bul ** 2), 0.0)
        v_model = np.sqrt(vbar_sq + _nfw_v2(radius, v200, concentration))
        return _chi2(v_model, v_obs, err_v)

    best = None
    starts = [
        (0.5, math.log(80.0), math.log(8.0)),
        (0.5, math.log(150.0), math.log(10.0)),
        (0.5, math.log(250.0), math.log(12.0)),
    ]
    bounds = [(0.1, 1.0), (math.log(20.0), math.log(500.0)), (math.log(1.0), math.log(50.0))]
    for start in starts:
        res = minimize(objective, np.array(start), method="L-BFGS-B", bounds=bounds)
        if best is None or res.fun < best.fun:
            best = res
    if best is None:
        return None
    y, log_v200, log_c = best.x
    chi2 = float(best.fun)
    dof = max(len(radius) - 3, 1)
    return {
        "chi2": chi2,
        "dof": dof,
        "chi2_per_dof": chi2 / dof,
        "n_params": 3,
        "bic": _bic(chi2, len(radius), 3),
        "Y_fitted": float(y),
        "v200_km_s": float(math.exp(log_v200)),
        "concentration": float(math.exp(log_c)),
    }


def run(max_galaxies: int) -> dict:
    table = parse_sparc_table(SPARC_TABLE)
    files = sorted(SPARC_DIR.glob("*_rotmod.dat"))
    if max_galaxies > 0:
        files = files[:max_galaxies]

    rows = []
    for fp in files:
        name = fp.stem.replace("_rotmod", "")
        if_result = evaluate_if(name, fp, table)
        mond_result = evaluate_mond(fp)
        nfw_result = evaluate_nfw(fp)
        if if_result and mond_result and nfw_result:
            rows.append({"name": name, "IF": if_result, "MOND": mond_result, "NFW": nfw_result})

    summary = {}
    for key in ["IF", "MOND", "NFW"]:
        values = np.array([row[key]["chi2_per_dof"] for row in rows], dtype=float)
        bics = np.array([row[key].get("bic", row[key]["chi2"]) for row in rows], dtype=float)
        summary[key] = {
            "median_chi2_per_dof": float(np.median(values)) if len(values) else None,
            "mean_chi2_per_dof": float(np.mean(values)) if len(values) else None,
            "median_bic": float(np.median(bics)) if len(bics) else None,
        }

    return {
        "artifact": "SPARC fair benchmark harness",
        "status": "SMOKE_SUBSET" if max_galaxies > 0 else "FULL_LOCAL_SPARK_RUN",
        "max_galaxies": max_galaxies,
        "n_evaluated": len(rows),
        "fairness_rules": {
            "IF": "one fitted stellar M/L Y per galaxy; r0 and Freeman factor fixed",
            "MOND": "one fitted stellar M/L Y per galaxy; a0 fixed at 1.2e-10 m/s^2",
            "NFW": "stellar M/L Y plus V200 and concentration per galaxy",
        },
        "summary": summary,
        "per_galaxy": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-galaxies", type=int, default=25, help="0 means all local SPARC files")
    args = parser.parse_args()
    out = run(args.max_galaxies)
    path = OUT_DIR / ("sparc_fair_benchmark_full.json" if args.max_galaxies == 0 else "sparc_fair_benchmark_smoke.json")
    path.write_text(json.dumps(out, indent=2))
    print(f"evaluated={out['n_evaluated']} status={out['status']}")
    for model, stats in out["summary"].items():
        print(f"{model}: median chi2/dof={stats['median_chi2_per_dof']}")
    print(f"Wrote {path}")
    return 0 if out["n_evaluated"] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
