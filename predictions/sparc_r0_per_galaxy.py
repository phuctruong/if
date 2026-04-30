#!/usr/bin/env python3
"""
sparc_r0_per_galaxy.py — SPARC variant where r_0 scales per galaxy
with the galaxy's own characteristic scale (R_disk).

Motivation: in the corrected log-potential test (sparc_corrected_log_potential.py),
the universal r_0 = 0.6595 kpc means the prime-field transition happens at
~0.66 kpc for all galaxies. But real galaxies have transition radii from
rising to flat that scale with galaxy size. A dwarf with R_disk = 1 kpc
should have its prime-field transition at ~1 kpc; a giant with R_disk = 10 kpc
at ~10 kpc.

Hypothesis: r_0_galaxy = R_disk (galaxy's own disk scale length).
This makes the IF Theory's transition radius track the galaxy's structure.
The amplitude v_0 is unchanged (baryon virial).

Resulting prediction:
  v_prime²(R) = v_0_galaxy² · R / (R + R_disk)
  v_total(R)  = √(v_baryon(R)² + v_prime²(R))

Per-galaxy free parameter count: still ZERO in the IF Theory part — both
v_0 and r_0 are derived from SPARC table entries (M_b, R_disk).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from predictions.sparc_corrected_log_potential import (  # noqa: E402
    M_TO_L_RATIO_3_6,
    SPARC_DIR,
    SPARC_TABLE,
    G_kpc_kms_msun,
    load_rotmod,
    parse_sparc_table,
)

OUT_DIR = Path(_ROOT, "evidence", "sparc_r0_per_galaxy")
OUT_DIR.mkdir(parents=True, exist_ok=True)
FREEMAN_FACTOR = 0.62


def evaluate_galaxy(name: str, path: Path, table: dict,
                    min_floor_err: float = 1.0) -> Optional[dict]:
    if name not in table:
        return None
    info = table[name]
    L_3_6 = info["L_3_6_1e9_Lsun"]
    M_HI = info["MHI_1e9_Msun"]
    Rdisk = info["Rdisk_kpc"]
    if Rdisk <= 0:
        return None
    M_baryon = (M_TO_L_RATIO_3_6 * L_3_6 + M_HI) * 1e9  # Msun
    if M_baryon <= 0:
        return None

    G = G_kpc_kms_msun()
    v_0_galaxy = math.sqrt(FREEMAN_FACTOR * G * M_baryon / Rdisk)
    r_0_galaxy = Rdisk  # galaxy-specific: prime-field transition matches disk scale

    d = load_rotmod(path)
    R = d["R"]
    keep = R > 0
    R = R[keep]
    Vobs = d["Vobs"][keep]
    errV = np.maximum(d["errV"][keep], min_floor_err)
    Vbar = np.sqrt(d["Vgas"][keep] ** 2 + d["Vdisk"][keep] ** 2 + d["Vbul"][keep] ** 2)
    if len(R) < 3:
        return None

    v_prime_sq = (v_0_galaxy ** 2) * R / (R + r_0_galaxy)
    v_prime = np.sqrt(v_prime_sq)
    v_total = np.sqrt(Vbar ** 2 + v_prime ** 2)

    residuals = Vobs - v_total
    chi2 = float(np.sum((residuals / errV) ** 2))
    dof = max(len(R) - 0, 1)
    return {
        "name": name,
        "n_points": len(R),
        "M_baryon_Msun": M_baryon,
        "R_disk_kpc": Rdisk,
        "v_0_predicted_kms": v_0_galaxy,
        "V_flat_observed_kms": info["Vflat_kms"],
        "chi2": chi2,
        "dof": dof,
        "chi2_per_dof": chi2 / dof,
        "rms_residual_kms": float(np.sqrt(np.mean(residuals ** 2))),
    }


def main() -> int:
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
        except (OSError, ValueError, RuntimeError, KeyError, IndexError, TypeError, AttributeError, ArithmeticError, ImportError) as e:
            skipped.append(f"{name} ({e})")
    if not results:
        return 1

    chi2_per_dof = np.array([r["chi2_per_dof"] for r in results])
    v0_pred = np.array([r["v_0_predicted_kms"] for r in results])
    Vflat = np.array([r["V_flat_observed_kms"] for r in results])

    print(f"Evaluated {len(results)} galaxies; r_0 = R_disk per galaxy, v_0 from baryon virial\n")
    print(f"χ²/dof   median = {np.median(chi2_per_dof):.2f}")
    print(f"         mean   = {np.mean(chi2_per_dof):.2f}")
    print(f"         25th   = {np.percentile(chi2_per_dof, 25):.2f}")
    print(f"         75th   = {np.percentile(chi2_per_dof, 75):.2f}")
    print(f"  fraction χ²/dof < 1   : {np.mean(chi2_per_dof < 1):.1%}")
    print(f"  fraction χ²/dof < 5   : {np.mean(chi2_per_dof < 5):.1%}")
    print(f"  fraction χ²/dof < 10  : {np.mean(chi2_per_dof < 10):.1%}")
    print(f"  fraction χ²/dof < 50  : {np.mean(chi2_per_dof < 50):.1%}")

    mask = (v0_pred > 0) & (Vflat > 0)
    if mask.sum() > 5:
        lp = np.log10(v0_pred[mask])
        lo = np.log10(Vflat[mask])
        slope, intercept = np.polyfit(lo, lp, 1)
        r_pearson = float(np.corrcoef(lp, lo)[0, 1])
        print()
        print("Tully-Fisher: log(v_0_pred) vs log(V_flat_obs):")
        print(f"  slope = {slope:+.3f}  (theoretical 1.00 expected)")
        print(f"  intercept = {intercept:+.3f}")
        print(f"  Pearson r = {r_pearson:+.3f}")
        print(f"  n = {int(mask.sum())} galaxies with V_flat measured")

    out = {
        "n_evaluated": len(results),
        "summary": {
            "chi2_per_dof_median": float(np.median(chi2_per_dof)),
            "chi2_per_dof_mean": float(np.mean(chi2_per_dof)),
            "frac_under_5": float(np.mean(chi2_per_dof < 5)),
            "frac_under_10": float(np.mean(chi2_per_dof < 10)),
            "frac_under_50": float(np.mean(chi2_per_dof < 50)),
            "tully_fisher_slope": float(slope) if mask.sum() > 5 else None,
            "tully_fisher_pearson_r": r_pearson if mask.sum() > 5 else None,
        },
        "per_galaxy": results,
    }
    with open(OUT_DIR / "sparc_r0_per_galaxy_results.json", "w") as f:
        json.dump(out, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
