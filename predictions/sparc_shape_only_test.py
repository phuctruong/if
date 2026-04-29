#!/usr/bin/env python3
"""
sparc_shape_only_test.py — IF Theory rotation-curve SHAPE test
isolated from amplitude derivation.

The SPARC test with per-galaxy M/L gave χ²/dof = 7.13 median. To
isolate the SHAPE claim from the v_0 amplitude derivation, we use the
SPARC-table-published V_flat as the anchor for the prime-field
asymptotic velocity:

    v_prime²(R) = V_flat² · R / (R + r_0)
    v_total(R)  = √(V_baryon² + v_prime²)

with V_baryon as before (V_gas² + Y · (V_disk² + V_bul²)) and Y per
galaxy. This tests whether the FUNCTIONAL FORM v² = R/(R+r_0) — i.e.,
the integrated logarithmic potential — describes rotation-curve
shapes given the correct asymptotic value, regardless of how V_flat
itself is derived.

If shape is right: median χ²/dof should drop substantially (close to
MOND's ~2-5 on SPARC).

Per-galaxy free parameters:
  - Y (M/L_3.6) — fitted, [0.1, 1.0]
  - V_flat — from SPARC table (NOT fitted)
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
from scipy.optimize import minimize_scalar

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from prime_field_util import R0_KPC_CANONICAL  # noqa: E402
from predictions.sparc_corrected_log_potential import (  # noqa: E402
    parse_sparc_table, load_rotmod, SPARC_DIR, SPARC_TABLE,
)

OUT_DIR = Path(_ROOT, "evidence", "sparc_shape_only")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def galaxy_chi2_at_Y(Y: float, V_flat: float, R: np.ndarray, Vobs: np.ndarray,
                     errV: np.ndarray, Vgas: np.ndarray, Vdisk: np.ndarray,
                     Vbul: np.ndarray, r_0: float) -> float:
    v_prime_sq = V_flat ** 2 * R / (R + r_0)
    v_baryon_sq = Vgas ** 2 + Y * (Vdisk ** 2 + Vbul ** 2)
    v_total = np.sqrt(np.maximum(v_baryon_sq + v_prime_sq, 0.0))
    return float(np.sum(((Vobs - v_total) / errV) ** 2))


def evaluate_galaxy(name: str, path: Path, table: dict,
                    r_0_strategy: str = "R_disk",  # "canonical" or "R_disk"
                    Y_min: float = 0.1, Y_max: float = 1.0,
                    min_floor_err: float = 1.0) -> Optional[dict]:
    if name not in table:
        return None
    info = table[name]
    V_flat = info["Vflat_kms"]
    if V_flat <= 0:
        return None  # need a measured V_flat to anchor

    if r_0_strategy == "canonical":
        r_0_kpc = R0_KPC_CANONICAL
    elif r_0_strategy == "R_disk":
        r_0_kpc = info["Rdisk_kpc"]
        if r_0_kpc <= 0:
            return None
    else:
        raise ValueError(r_0_strategy)

    d = load_rotmod(path)
    R = d["R"]; keep = R > 0
    R = R[keep]; Vobs = d["Vobs"][keep]
    errV = np.maximum(d["errV"][keep], min_floor_err)
    Vgas = d["Vgas"][keep]; Vdisk = d["Vdisk"][keep]; Vbul = d["Vbul"][keep]
    if len(R) < 3:
        return None

    res = minimize_scalar(
        galaxy_chi2_at_Y,
        args=(V_flat, R, Vobs, errV, Vgas, Vdisk, Vbul, r_0_kpc),
        bounds=(Y_min, Y_max), method="bounded",
        options=dict(xatol=1e-3),
    )
    Y_opt = float(res.x)
    chi2 = float(res.fun)
    dof = max(len(R) - 1, 1)  # 1 free parameter per galaxy (Y); V_flat is from table

    return {
        "name": name,
        "n_points": len(R),
        "Y_fitted": Y_opt,
        "V_flat_table_kms": V_flat,
        "chi2": chi2,
        "dof": dof,
        "chi2_per_dof": chi2 / dof,
    }


def run_strategy(table: dict, files: list, strategy: str) -> dict:
    results: List[dict] = []
    skipped: List[str] = []
    for fp in files:
        name = fp.stem.replace("_rotmod", "")
        try:
            r = evaluate_galaxy(name, fp, table, r_0_strategy=strategy)
            if r is None:
                skipped.append(name)
            else:
                results.append(r)
        except Exception as e:
            skipped.append(f"{name} ({e})")
    return {"strategy": strategy, "results": results, "skipped": skipped}


def main() -> int:
    table = parse_sparc_table(SPARC_TABLE)
    files = sorted(SPARC_DIR.glob("*_rotmod.dat"))

    print("=" * 78)
    print("SPARC SHAPE-ONLY: V_flat from table; comparing r_0 strategies")
    print("=" * 78)

    overall = {}
    for strategy in ("canonical", "R_disk"):
        outcome = run_strategy(table, files, strategy)
        results = outcome["results"]
        if not results:
            print(f"  [{strategy}] no galaxies evaluated")
            continue
        chi2_per_dof = np.array([r["chi2_per_dof"] for r in results])
        Y_fitted = np.array([r["Y_fitted"] for r in results])
        print(f"\n--- r_0 = {strategy} (n = {len(results)}) ---")
        print(f"  χ²/dof   median = {np.median(chi2_per_dof):8.2f}")
        print(f"           25th   = {np.percentile(chi2_per_dof, 25):8.2f}")
        print(f"           75th   = {np.percentile(chi2_per_dof, 75):8.2f}")
        print(f"  fraction χ²/dof < 1   : {np.mean(chi2_per_dof < 1):.1%}")
        print(f"  fraction χ²/dof < 5   : {np.mean(chi2_per_dof < 5):.1%}")
        print(f"  fraction χ²/dof < 10  : {np.mean(chi2_per_dof < 10):.1%}")
        print(f"  fraction χ²/dof < 50  : {np.mean(chi2_per_dof < 50):.1%}")
        print(f"  Y median = {np.median(Y_fitted):.3f}")
        print(f"  Y in [0.3, 0.7] (Schombert): "
              f"{np.mean((Y_fitted >= 0.3) & (Y_fitted <= 0.7)):.1%}")
        overall[strategy] = {
            "n_evaluated": len(results),
            "n_skipped": len(outcome["skipped"]),
            "chi2_per_dof_median": float(np.median(chi2_per_dof)),
            "chi2_per_dof_mean": float(np.mean(chi2_per_dof)),
            "frac_under_5": float(np.mean(chi2_per_dof < 5)),
            "frac_under_10": float(np.mean(chi2_per_dof < 10)),
            "frac_under_50": float(np.mean(chi2_per_dof < 50)),
            "Y_median": float(np.median(Y_fitted)),
            "Y_in_schombert_range": float(np.mean((Y_fitted >= 0.3) & (Y_fitted <= 0.7))),
            "per_galaxy": results,
        }
    with open(OUT_DIR / "sparc_shape_only_results.json", "w") as f:
        json.dump(overall, f, indent=2)
    print(f"\nWrote {OUT_DIR / 'sparc_shape_only_results.json'}")
    return 0


def _legacy_main_replaced() -> int:
    """Kept for reference; the new main() runs both strategies."""
    table = parse_sparc_table(SPARC_TABLE)
    files = sorted(SPARC_DIR.glob("*_rotmod.dat"))
    results: List[dict] = []
    skipped: List[str] = []
    for fp in files:
        name = fp.stem.replace("_rotmod", "")
        try:
            r = evaluate_galaxy(name, fp, table)
            if r is None:
                skipped.append(name)
            else:
                results.append(r)
        except Exception as e:
            skipped.append(f"{name} ({e})")
    if not results:
        return 1

    chi2_per_dof = np.array([r["chi2_per_dof"] for r in results])
    Y_fitted = np.array([r["Y_fitted"] for r in results])

    print("=" * 78)
    print("SPARC SHAPE-ONLY TEST: V_flat from SPARC table, Y per galaxy")
    print(f"  r_0 = {R0_KPC_CANONICAL:.4f} kpc canonical")
    print("=" * 78)
    print(f"  Galaxies evaluated      : {len(results)} (require V_flat measured)")
    print(f"  Skipped (no V_flat)     : {len(skipped)}")
    print()
    print(f"  χ²/dof   median         : {np.median(chi2_per_dof):8.2f}")
    print(f"           mean           : {np.mean(chi2_per_dof):8.2f}")
    print(f"           25th pct       : {np.percentile(chi2_per_dof, 25):8.2f}")
    print(f"           75th pct       : {np.percentile(chi2_per_dof, 75):8.2f}")
    print()
    print(f"  Fraction χ²/dof < 1     : {np.mean(chi2_per_dof < 1):.1%}")
    print(f"  Fraction χ²/dof < 5     : {np.mean(chi2_per_dof < 5):.1%}")
    print(f"  Fraction χ²/dof < 10    : {np.mean(chi2_per_dof < 10):.1%}")
    print(f"  Fraction χ²/dof < 50    : {np.mean(chi2_per_dof < 50):.1%}")
    print()
    print(f"  Y_fitted distribution:")
    print(f"    median = {np.median(Y_fitted):.3f}")
    print(f"    25th   = {np.percentile(Y_fitted, 25):.3f}")
    print(f"    75th   = {np.percentile(Y_fitted, 75):.3f}")
    print(f"    fraction in [0.3, 0.7]: {np.mean((Y_fitted >= 0.3) & (Y_fitted <= 0.7)):.1%}")

    out = {
        "r_0_kpc": R0_KPC_CANONICAL,
        "n_evaluated": len(results),
        "n_skipped": len(skipped),
        "summary": {
            "chi2_per_dof_median": float(np.median(chi2_per_dof)),
            "chi2_per_dof_mean": float(np.mean(chi2_per_dof)),
            "frac_under_1": float(np.mean(chi2_per_dof < 1)),
            "frac_under_5": float(np.mean(chi2_per_dof < 5)),
            "frac_under_10": float(np.mean(chi2_per_dof < 10)),
            "frac_under_50": float(np.mean(chi2_per_dof < 50)),
            "Y_median": float(np.median(Y_fitted)),
            "Y_in_schombert_range": float(np.mean((Y_fitted >= 0.3) & (Y_fitted <= 0.7))),
        },
        "per_galaxy": results,
    }
    with open(OUT_DIR / "sparc_shape_only_results.json", "w") as f:
        json.dump(out, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
