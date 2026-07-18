#!/usr/bin/env python3
"""
sparc_175_per_galaxy_v0.py — IF Theory rotation-curve SHAPE test
across the 175-galaxy SPARC database, with ONE free parameter per
galaxy (v_0 scaling).

The earlier sparc_175_validation.py showed that the universal
v_0 = 397 km/s catastrophically over-predicts dwarf galaxy rotation
curves (median χ²/dof ≈ 1082). The "zero free parameters across all
galaxies" framing is therefore *false* at SPARC scales — at minimum,
v_0 must scale with galaxy mass (this is empirically the Tully-Fisher
relation: v_flat ∝ M_baryon^¼).

This script is the more modest test: fix r_0 at the canonical
0.6595 kpc, fit v_0 as one free parameter per galaxy, and ask whether
the SHAPE of v_prime(R) given by

    v_prime(R) = v_0 · √(R · |dΦ/dR|),  Φ(r) = 1/log(r/r_0 + 1)

reproduces the observed rotation curve when added in quadrature to the
SPARC-published baryonic decomposition v_baryon = √(V_gas² + V_disk² + V_bul²).

If most galaxies fit at χ²/dof < 5 with one parameter (v_0), and the
fitted v_0 values track Tully-Fisher (v_0 ∝ M_b^¼), the SHAPE claim
survives even though the strong "universal v_0" claim does not.

Run with:
    python3 predictions/sparc_175_per_galaxy_v0.py
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
from scipy.optimize import minimize_scalar

logging.basicConfig(level=logging.WARNING)

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from prime_field_util import R0_KPC_CANONICAL  # noqa: E402

SPARC_DIR = Path("/home/phuc/Downloads/if/data/sparc/Rotmod_LTG")
R0_MPC = R0_KPC_CANONICAL / 1000.0


def v_prime_unit(R_mpc: np.ndarray) -> np.ndarray:
    """v_prime(R) at v_0 = 1 km/s. Multiply by v_0 to scale.

    v_prime(R) = √(R · |dΦ/dR|) where Φ(r) = 1/log(r/r_0 + 1).
    Closed form gradient: dΦ/dR = -1 / [r_0 · (R/r_0 + 1) · log²(R/r_0 + 1)].
    |dΦ/dR| therefore equals 1/[r_0 · (R/r_0 + 1) · log²(R/r_0 + 1)].
    """
    x = R_mpc / R0_MPC + 1.0
    log_x = np.log(x)
    grad = 1.0 / (R0_MPC * x * log_x ** 2)
    return np.sqrt(R_mpc * grad)


def load_rotmod(path: Path) -> dict:
    R, Vobs, errV, Vgas, Vdisk, Vbul = [], [], [], [], [], []
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            R.append(float(parts[0]))
            Vobs.append(float(parts[1]))
            errV.append(float(parts[2]))
            Vgas.append(float(parts[3]))
            Vdisk.append(float(parts[4]))
            Vbul.append(float(parts[5]))
    return dict(
        R=np.asarray(R), Vobs=np.asarray(Vobs), errV=np.asarray(errV),
        Vgas=np.asarray(Vgas), Vdisk=np.asarray(Vdisk), Vbul=np.asarray(Vbul),
    )


@dataclass
class Result:
    name: str
    n_points: int
    v0_fitted_kms: float
    chi2: float
    dof: int
    chi2_per_dof: float
    v_obs_outer_kms: float  # v at largest radius — proxy for v_flat
    rms_residual_kms: float
    fraction_within_1sigma: float


def fit_galaxy(name: str, path: Path, min_floor_err: float = 1.0) -> Optional[Result]:
    d = load_rotmod(path)
    R = d["R"]
    keep = R > 0
    R = R[keep]
    Vobs = d["Vobs"][keep]
    errV = np.maximum(d["errV"][keep], min_floor_err)
    Vbar = np.sqrt(d["Vgas"][keep] ** 2 + d["Vdisk"][keep] ** 2 + d["Vbul"][keep] ** 2)
    if len(R) < 3:
        return None

    R_mpc = R * 1e-3
    vp_unit = v_prime_unit(R_mpc)  # shape only

    def chi2(v0_kms: float) -> float:
        v_total = np.sqrt(Vbar ** 2 + (v0_kms * vp_unit) ** 2)
        return float(np.sum(((Vobs - v_total) / errV) ** 2))

    res = minimize_scalar(chi2, bounds=(0.0, 5000.0), method="bounded",
                          options=dict(xatol=0.1))
    v0_opt = float(res.x)
    chi2_min = float(res.fun)
    dof = max(len(R) - 1, 1)  # one free parameter (v_0)

    v_total_opt = np.sqrt(Vbar ** 2 + (v0_opt * vp_unit) ** 2)
    residuals = Vobs - v_total_opt
    rms = float(np.sqrt(np.mean(residuals ** 2)))
    f1 = float(np.mean(np.abs(residuals / errV) < 1.0))

    return Result(
        name=name,
        n_points=len(R),
        v0_fitted_kms=v0_opt,
        chi2=chi2_min,
        dof=dof,
        chi2_per_dof=chi2_min / dof,
        v_obs_outer_kms=float(Vobs[-1]),
        rms_residual_kms=rms,
        fraction_within_1sigma=f1,
    )


def main() -> int:
    files = sorted(SPARC_DIR.glob("*_rotmod.dat"))
    print(f"Found {len(files)} SPARC galaxies; r_0 = {R0_KPC_CANONICAL:.4f} kpc; "
          f"fitting v_0 per galaxy (one free parameter)\n")

    results: List[Result] = []
    skipped: List[str] = []
    for fp in files:
        name = fp.stem.replace("_rotmod", "")
        try:
            r = fit_galaxy(name, fp)
            if r is None:
                skipped.append(name)
            else:
                results.append(r)
        except (OSError, ValueError, RuntimeError, KeyError, IndexError, TypeError, AttributeError, ArithmeticError, ImportError) as e:
            skipped.append(f"{name} (error: {e})")

    if not results:
        print("No galaxies evaluable.")
        return 1

    chi2_per_dof = np.array([r.chi2_per_dof for r in results])
    v0_fitted = np.array([r.v0_fitted_kms for r in results])
    v_outer = np.array([r.v_obs_outer_kms for r in results])
    f1 = np.array([r.fraction_within_1sigma for r in results])

    print("=" * 78)
    print("SPARC 175 — IF Theory shape test, ONE free parameter per galaxy (v_0)")
    print("=" * 78)
    print(f"  Galaxies fitted        : {len(results)} / {len(files)}")
    print(f"  Skipped                : {len(skipped)}")
    print()
    print(f"  χ²/dof  median         : {np.median(chi2_per_dof):.2f}")
    print(f"          mean           : {np.mean(chi2_per_dof):.2f}")
    print(f"          25th pct       : {np.percentile(chi2_per_dof, 25):.2f}")
    print(f"          75th pct       : {np.percentile(chi2_per_dof, 75):.2f}")
    print()
    print(f"  Fraction χ²/dof < 1     : {np.mean(chi2_per_dof < 1):.1%}")
    print(f"  Fraction χ²/dof < 5     : {np.mean(chi2_per_dof < 5):.1%}")
    print(f"  Fraction χ²/dof < 10    : {np.mean(chi2_per_dof < 10):.1%}")
    print(f"  Fraction χ²/dof < 50    : {np.mean(chi2_per_dof < 50):.1%}")
    print()
    print(f"  Fitted v_0:  median = {np.median(v0_fitted):.0f} km/s, "
          f"min = {np.min(v0_fitted):.0f}, max = {np.max(v0_fitted):.0f}")
    print(f"  v_obs (outer): median = {np.median(v_outer):.0f} km/s, "
          f"min = {np.min(v_outer):.0f}, max = {np.max(v_outer):.0f}")
    print(f"  Median %% points within 1σ (per galaxy): {np.median(f1) * 100:.0f}%")
    print()

    # Tully-Fisher-like correlation: log(v_0) vs log(v_outer)
    if np.all(v0_fitted > 0) and np.all(v_outer > 0):
        log_v0 = np.log10(v0_fitted)
        log_vo = np.log10(v_outer)
        # robust linear fit
        slope, intercept = np.polyfit(log_vo, log_v0, 1)
        r_pearson = float(np.corrcoef(log_v0, log_vo)[0, 1])
        print("  log(v_0_fitted) vs log(v_obs_outer) linear fit:")
        print(f"    slope = {slope:.2f}  (Tully-Fisher M-σ relation predicts ~1)")
        print(f"    intercept = {intercept:.2f}")
        print(f"    Pearson r = {r_pearson:.3f}")

    out_dir = Path(_ROOT, "evidence", "sparc_175")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "sparc_per_galaxy_v0.json", "w") as f:
        json.dump({
            "r0_kpc": R0_KPC_CANONICAL,
            "n_galaxies_fitted": len(results),
            "n_skipped": len(skipped),
            "skipped_names": skipped,
            "summary": {
                "chi2_per_dof_median": float(np.median(chi2_per_dof)),
                "chi2_per_dof_mean": float(np.mean(chi2_per_dof)),
                "frac_under_1": float(np.mean(chi2_per_dof < 1)),
                "frac_under_5": float(np.mean(chi2_per_dof < 5)),
                "frac_under_10": float(np.mean(chi2_per_dof < 10)),
                "frac_under_50": float(np.mean(chi2_per_dof < 50)),
                "v0_fitted_median": float(np.median(v0_fitted)),
                "tully_fisher_slope": float(slope) if len(v0_fitted) > 2 else None,
                "tully_fisher_pearson_r": r_pearson if len(v0_fitted) > 2 else None,
            },
            "per_galaxy": [asdict(r) for r in results],
        }, f, indent=2)
    print(f"Wrote {out_dir / 'sparc_per_galaxy_v0.json'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
