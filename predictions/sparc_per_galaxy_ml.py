#!/usr/bin/env python3
"""
sparc_per_galaxy_ml.py — IF Theory + SPARC with one free parameter per
galaxy: the stellar mass-to-light ratio Y = M/L_3.6.

This is the standard SPARC analysis convention (Lelli et al. 2016).
SPARC's V_disk and V_bul columns in rotmod.dat are computed at M/L = 1;
the actual baryonic rotation contribution is

    v_baryon²(R) = V_gas²(R) + Y · [V_disk²(R) + V_bul²(R)]

where Y is allowed to vary per galaxy in the physically motivated
range [0.3, 0.7] (Schombert et al. 2014; spans young to old stellar
populations at 3.6 μm).

The IF Theory's prime-field contribution depends on M_baryon (via the
baryon-virial v_0):

    M_baryon(Y) = M_HI + Y · L_3.6 · 1e9   [M_sun]
    v_0(Y)      = √(0.62 · G · M_baryon(Y) / R_disk)
    v_prime²(R) = v_0(Y)² · R / (R + r_0_canonical)

Combined:
    v_total²(R) = V_gas²(R) + Y · [V_disk²(R) + V_bul²(R)] + v_prime²(R, Y)

Minimize χ² over Y for each galaxy (bounded [0.1, 1.0] to allow some
deviation from the Schombert range while excluding unphysical values).
This is 1 free parameter per galaxy, matching MOND's standard SPARC
analysis. Compare median χ²/dof to MOND's reported ~2-5.
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

from predictions.sparc_corrected_log_potential import (  # noqa: E402
    SPARC_DIR,
    SPARC_TABLE,
    G_kpc_kms_msun,
    load_rotmod,
    parse_sparc_table,
)
from prime_field_util import R0_KPC_CANONICAL  # noqa: E402

OUT_DIR = Path(_ROOT, "evidence", "sparc_per_galaxy_ml")
OUT_DIR.mkdir(parents=True, exist_ok=True)
FREEMAN_FACTOR = 0.62


def galaxy_chi2_at_Y(Y: float, info: dict, R: np.ndarray, Vobs: np.ndarray,
                     errV: np.ndarray, Vgas: np.ndarray, Vdisk: np.ndarray,
                     Vbul: np.ndarray, r_0: float) -> float:
    """χ² of v_total at given Y = M/L_3.6."""
    M_baryon = (Y * info["L_3_6_1e9_Lsun"] + info["MHI_1e9_Msun"]) * 1e9  # Msun
    if M_baryon <= 0 or info["Rdisk_kpc"] <= 0:
        return 1e18
    G = G_kpc_kms_msun()
    v_0 = math.sqrt(FREEMAN_FACTOR * G * M_baryon / info["Rdisk_kpc"])
    v_prime_sq = (v_0 ** 2) * R / (R + r_0)
    v_baryon_sq = Vgas ** 2 + Y * (Vdisk ** 2 + Vbul ** 2)
    v_total = np.sqrt(np.maximum(v_baryon_sq + v_prime_sq, 0.0))
    return float(np.sum(((Vobs - v_total) / errV) ** 2))


def evaluate_galaxy(name: str, path: Path, table: dict,
                    r_0_kpc: float = R0_KPC_CANONICAL,
                    min_floor_err: float = 1.0,
                    Y_min: float = 0.1,
                    Y_max: float = 1.0) -> Optional[dict]:
    if name not in table:
        return None
    info = table[name]
    if info["L_3_6_1e9_Lsun"] <= 0 and info["MHI_1e9_Msun"] <= 0:
        return None
    if info["Rdisk_kpc"] <= 0:
        return None

    d = load_rotmod(path)
    R = d["R"]
    keep = R > 0
    R = R[keep]
    Vobs = d["Vobs"][keep]
    errV = np.maximum(d["errV"][keep], min_floor_err)
    Vgas = d["Vgas"][keep]
    Vdisk = d["Vdisk"][keep]
    Vbul = d["Vbul"][keep]
    if len(R) < 3:
        return None

    res = minimize_scalar(
        galaxy_chi2_at_Y,
        args=(info, R, Vobs, errV, Vgas, Vdisk, Vbul, r_0_kpc),
        bounds=(Y_min, Y_max), method="bounded",
        options=dict(xatol=1e-3),
    )
    Y_opt = float(res.x)
    chi2 = float(res.fun)
    dof = max(len(R) - 1, 1)  # 1 free parameter (Y)

    M_baryon = (Y_opt * info["L_3_6_1e9_Lsun"] + info["MHI_1e9_Msun"]) * 1e9
    v_0 = math.sqrt(FREEMAN_FACTOR * G_kpc_kms_msun() * M_baryon / info["Rdisk_kpc"])

    return {
        "name": name,
        "n_points": len(R),
        "Y_fitted": Y_opt,
        "M_baryon_Msun": M_baryon,
        "R_disk_kpc": info["Rdisk_kpc"],
        "v_0_predicted_kms": v_0,
        "V_flat_observed_kms": info["Vflat_kms"],
        "chi2": chi2,
        "dof": dof,
        "chi2_per_dof": chi2 / dof,
    }


def main() -> int:
    table = parse_sparc_table(SPARC_TABLE)
    files = sorted(SPARC_DIR.glob("*_rotmod.dat"))
    print(f"Loaded {len(table)} SPARC table entries; {len(files)} rotmod files")

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
        except (OSError, ValueError, RuntimeError, KeyError, IndexError, TypeError, AttributeError, ArithmeticError, ImportError) as e:
            skipped.append(f"{name} ({e})")
    if not results:
        return 1

    chi2_per_dof = np.array([r["chi2_per_dof"] for r in results])
    Y_fitted = np.array([r["Y_fitted"] for r in results])
    v0_pred = np.array([r["v_0_predicted_kms"] for r in results])
    Vflat = np.array([r["V_flat_observed_kms"] for r in results])

    print("=" * 78)
    print("SPARC per-galaxy M/L fit (Y ∈ [0.1, 1.0]); IF Theory: corrected log Φ")
    print("=" * 78)
    print(f"  Galaxies evaluated      : {len(results)}")
    print(f"  Skipped                 : {len(skipped)}")
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
    print("  Y_fitted distribution:")
    print(f"    median = {np.median(Y_fitted):.3f}")
    print(f"    25th   = {np.percentile(Y_fitted, 25):.3f}")
    print(f"    75th   = {np.percentile(Y_fitted, 75):.3f}")
    print(f"    fraction in physical [0.3, 0.7] (Schombert): "
          f"{np.mean((Y_fitted >= 0.3) & (Y_fitted <= 0.7)):.1%}")

    mask = (v0_pred > 0) & (Vflat > 0)
    if mask.sum() > 5:
        lp = np.log10(v0_pred[mask])
        lo = np.log10(Vflat[mask])
        slope, intercept = np.polyfit(lo, lp, 1)
        r_pearson = float(np.corrcoef(lp, lo)[0, 1])
        print()
        print("  Tully-Fisher: log(v_0_pred) vs log(V_flat_obs):")
        print(f"    slope = {slope:+.3f}")
        print(f"    intercept = {intercept:+.3f}")
        print(f"    Pearson r = {r_pearson:+.3f}")
        print(f"    n = {int(mask.sum())} galaxies")

    out = {
        "r_0_kpc": R0_KPC_CANONICAL,
        "n_evaluated": len(results),
        "summary": {
            "chi2_per_dof_median": float(np.median(chi2_per_dof)),
            "chi2_per_dof_mean": float(np.mean(chi2_per_dof)),
            "frac_under_5": float(np.mean(chi2_per_dof < 5)),
            "frac_under_10": float(np.mean(chi2_per_dof < 10)),
            "frac_under_50": float(np.mean(chi2_per_dof < 50)),
            "Y_median": float(np.median(Y_fitted)),
            "Y_in_schombert_range": float(np.mean((Y_fitted >= 0.3) & (Y_fitted <= 0.7))),
            "tully_fisher_slope": float(slope) if mask.sum() > 5 else None,
            "tully_fisher_pearson_r": r_pearson if mask.sum() > 5 else None,
        },
        "per_galaxy": results,
    }
    with open(OUT_DIR / "sparc_per_galaxy_ml_results.json", "w") as f:
        json.dump(out, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
