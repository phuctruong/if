#!/usr/bin/env python3
"""
sparc_175_validation.py — IF Theory predictions vs the SPARC database
of 175 disk galaxies with high-quality rotation curves (Lelli, McGaugh,
Schombert 2016, AJ 152, 157).

For each galaxy at each radius the SPARC rotmod.dat file gives:

    Rad, Vobs, errV, Vgas, Vdisk, Vbul, SBdisk, SBbul

The baryonic rotation contribution is the quadrature sum of the gas,
stellar disk, and bulge components:

    v_baryon(R) = √(V_gas² + V_disk² + V_bul²)

The prime field contributes (zero-parameter, universal v_0):

    v_prime(R)  = v_0 · √(R · |dΦ/dR|),
        Φ(r)   = 1/log(r/r_0 + 1)

The IF Theory total prediction adds in quadrature (claim #80):

    v_total(R) = √(v_baryon(R)² + v_prime(R)²)

We compare to the observed rotation velocities and report per-galaxy
χ²/dof along with population statistics. The test passes the IF Theory
if a substantial fraction of galaxies have χ²/dof < 5 with the universal
prime-field parameters and zero free parameters per galaxy.

Reference parameters (single source of truth, from prime_field_theory):
    r_0 = 0.6594900863537677 kpc           (Mersenne Tower, claim #1)
    v_0 = 397.27 km/s                      (virial scale, ±30%)

Run with:
    python3 predictions/sparc_175_validation.py
"""

from __future__ import annotations

import json
import logging
import math
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from prime_field_theory import PrimeFieldTheory  # noqa: E402

SPARC_DIR = Path("/home/phuc/Downloads/if/data/sparc/Rotmod_LTG")


@dataclass
class GalaxyResult:
    name: str
    n_points: int
    chi2: float
    dof: int
    chi2_per_dof: float
    mean_residual_kms: float
    rms_residual_kms: float
    rms_v_obs_kms: float
    fraction_within_1sigma: float
    fraction_within_2sigma: float


def load_rotmod(path: Path) -> dict:
    """Read a SPARC _rotmod.dat file. Returns dict of arrays."""
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
        R=np.asarray(R),
        Vobs=np.asarray(Vobs),
        errV=np.asarray(errV),
        Vgas=np.asarray(Vgas),
        Vdisk=np.asarray(Vdisk),
        Vbul=np.asarray(Vbul),
    )


def predict_v_total(R_kpc: np.ndarray, Vbar_kms: np.ndarray, pft: PrimeFieldTheory) -> np.ndarray:
    """v_total = √(v_baryon² + v_prime²) at each radius."""
    R_mpc = R_kpc * 1e-3
    v_prime = np.asarray([float(pft.orbital_velocity(r)) for r in R_mpc])
    return np.sqrt(Vbar_kms ** 2 + v_prime ** 2)


def evaluate_galaxy(name: str, path: Path, pft: PrimeFieldTheory,
                    min_floor_err: float = 1.0) -> Optional[GalaxyResult]:
    """Compute per-galaxy fit statistics. None if data unusable."""
    data = load_rotmod(path)
    R = data["R"]
    if len(R) == 0:
        return None
    # Use only positive radii (a few SPARC files have R = 0 entries)
    keep = R > 0
    R = R[keep]
    Vobs = data["Vobs"][keep]
    errV = np.maximum(data["errV"][keep], min_floor_err)  # avoid /0
    Vgas = data["Vgas"][keep]
    Vdisk = data["Vdisk"][keep]
    Vbul = data["Vbul"][keep]
    if len(R) < 3:
        return None

    Vbar = np.sqrt(Vgas ** 2 + Vdisk ** 2 + Vbul ** 2)
    Vtot = predict_v_total(R, Vbar, pft)

    residuals = Vobs - Vtot
    chi2 = float(np.sum((residuals / errV) ** 2))
    dof = max(len(R) - 0, 1)  # zero free parameters in IF part
    return GalaxyResult(
        name=name,
        n_points=len(R),
        chi2=chi2,
        dof=dof,
        chi2_per_dof=chi2 / dof,
        mean_residual_kms=float(np.mean(residuals)),
        rms_residual_kms=float(np.sqrt(np.mean(residuals ** 2))),
        rms_v_obs_kms=float(np.sqrt(np.mean(Vobs ** 2))),
        fraction_within_1sigma=float(np.mean(np.abs(residuals / errV) < 1.0)),
        fraction_within_2sigma=float(np.mean(np.abs(residuals / errV) < 2.0)),
    )


def main() -> int:
    pft = PrimeFieldTheory(use_mersenne_tower=True)
    print(f"r0_kpc = {pft.r0_kpc}, v0_kms = {pft.v0_kms}")

    files = sorted(SPARC_DIR.glob("*_rotmod.dat"))
    print(f"Found {len(files)} SPARC galaxies in {SPARC_DIR}\n")

    results: List[GalaxyResult] = []
    skipped: List[str] = []
    for fp in files:
        name = fp.stem.replace("_rotmod", "")
        try:
            r = evaluate_galaxy(name, fp, pft)
            if r is None:
                skipped.append(name)
            else:
                results.append(r)
        except Exception as e:
            skipped.append(f"{name} (error: {e})")

    if not results:
        print("No galaxies evaluable.")
        return 1

    # Population statistics
    chi2_per_dof = np.array([r.chi2_per_dof for r in results])
    rms_resid = np.array([r.rms_residual_kms for r in results])
    f1 = np.array([r.fraction_within_1sigma for r in results])
    f2 = np.array([r.fraction_within_2sigma for r in results])

    print("=" * 78)
    print(f"SPARC 175 — IF Theory zero-parameter prediction (v_0 = {pft.v0_kms:.1f} km/s,")
    print(f"            r_0 = {pft.r0_kpc:.4f} kpc, baryon = √(V_gas² + V_disk² + V_bul²))")
    print("=" * 78)
    print(f"  Galaxies evaluated      : {len(results)} / {len(files)}")
    print(f"  Skipped                 : {len(skipped)}")
    print()
    print(f"  χ²/dof   median         : {np.median(chi2_per_dof):.2f}")
    print(f"           mean           : {np.mean(chi2_per_dof):.2f}")
    print(f"           min            : {np.min(chi2_per_dof):.2f}  ({results[int(np.argmin(chi2_per_dof))].name})")
    print(f"           max            : {np.max(chi2_per_dof):.2f}  ({results[int(np.argmax(chi2_per_dof))].name})")
    print(f"  RMS residual (km/s) median: {np.median(rms_resid):.1f}")
    print()
    print(f"  Fraction of galaxies with χ²/dof < 5  : {np.mean(chi2_per_dof < 5):.1%}")
    print(f"  Fraction of galaxies with χ²/dof < 10 : {np.mean(chi2_per_dof < 10):.1%}")
    print(f"  Fraction of galaxies with χ²/dof < 50 : {np.mean(chi2_per_dof < 50):.1%}")
    print()
    print(f"  Median %% points within 1σ (per galaxy): {np.median(f1) * 100:.0f}%")
    print(f"  Median %% points within 2σ (per galaxy): {np.median(f2) * 100:.0f}%")
    print()

    # Save
    out_dir = Path(_ROOT, "evidence", "sparc_175")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "sparc_results.json", "w") as f:
        json.dump({
            "r0_kpc": pft.r0_kpc,
            "v0_kms": pft.v0_kms,
            "n_galaxies_evaluated": len(results),
            "n_skipped": len(skipped),
            "skipped_names": skipped,
            "summary": {
                "chi2_per_dof_median": float(np.median(chi2_per_dof)),
                "chi2_per_dof_mean": float(np.mean(chi2_per_dof)),
                "frac_under_5": float(np.mean(chi2_per_dof < 5)),
                "frac_under_10": float(np.mean(chi2_per_dof < 10)),
                "frac_under_50": float(np.mean(chi2_per_dof < 50)),
                "median_pct_within_1sigma": float(np.median(f1)),
                "median_pct_within_2sigma": float(np.median(f2)),
            },
            "per_galaxy": [asdict(r) for r in results],
        }, f, indent=2)
    print(f"Wrote {out_dir / 'sparc_results.json'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
