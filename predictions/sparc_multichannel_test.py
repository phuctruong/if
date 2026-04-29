#!/usr/bin/env python3
"""
sparc_multichannel_test.py — IF Theory rotation-curve test using the
multi-prime substrate (gai 18-channel framework from geo's substrate
engine), against the SPARC 175-galaxy database.

The single-prime test (sparc_175_validation.py) failed structurally:
v_prime ∝ 1/log(R/r_0) decreases too fast for flat rotation curves.
This script tests the geo Stage 5-21 hypothesis: the prime field is
actually a SUM over multiple prime channels, each with its own
characteristic scale r_p, weighted by 1/p:

    Φ_total(r) = Σ_p (1/p) · 1/log(r/r_p + 1)

with r_p = r_0 · p (so p = 11 has r_p = 7.3 kpc, p = 71 has r_p = 47 kpc,
covering the SPARC galactic-scale range).

The motivation: different primes peak at different physical scales.
Larger primes contribute at galaxy outskirts; smaller primes at galaxy
cores. The combined sum may give a flatter v_prime(R) than any single
prime alone, reproducing the observed flat rotation curves.

This is the cleanest first-principles attempt to fix the SPARC failure
without abandoning the "no dark matter" axiom.
"""

from __future__ import annotations

import json
import logging
import math
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

logging.basicConfig(level=logging.WARNING)

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from prime_field_util import R0_KPC_CANONICAL  # noqa: E402

SPARC_DIR = Path("/home/phuc/Downloads/if/data/sparc/Rotmod_LTG")
OUT_DIR = Path(_ROOT, "evidence", "sparc_multichannel")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# gai 18-channel primes from CLAUDE.md / geo onion stages 5-21
GAI_PRIMES = [11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]
R0_KPC = R0_KPC_CANONICAL


def Phi_p_kpc(r_kpc: np.ndarray, r_p_kpc: float) -> np.ndarray:
    """Single-prime Phi_p(r) = 1/log(r/r_p + 1)."""
    return 1.0 / np.log(r_kpc / r_p_kpc + 1.0)


def dPhi_p_dr_kpc(r_kpc: np.ndarray, r_p_kpc: float) -> np.ndarray:
    """|dΦ_p/dr| = 1 / [r_p · (r/r_p + 1) · log²(r/r_p + 1)] in 1/kpc."""
    u = r_kpc / r_p_kpc + 1.0
    return 1.0 / (r_p_kpc * u * np.log(u) ** 2)


def v_prime_multichannel_kms(
    R_kpc: np.ndarray,
    v0_kms: float,
    primes: List[int] = GAI_PRIMES,
    r_0_base_kpc: float = R0_KPC,
    r_p_scaling: str = "linear_p",
) -> np.ndarray:
    """v_prime(R) using the multi-channel sum.

    r_p_scaling options:
      "linear_p":   r_p = r_0 · p     (covers ~7-47 kpc range)
      "log_p":      r_p = r_0 · log(p)·c  (slow scaling)
      "p_squared":  r_p = r_0 · p²    (covers ~80-3300 kpc; for very large r)
    """
    R = np.asarray(R_kpc, dtype=float)
    grad_total = np.zeros_like(R)
    for p in primes:
        if r_p_scaling == "linear_p":
            r_p = r_0_base_kpc * p
        elif r_p_scaling == "log_p":
            r_p = r_0_base_kpc * (math.log(p) * 5)
        elif r_p_scaling == "p_squared":
            r_p = r_0_base_kpc * p * p
        else:
            raise ValueError(r_p_scaling)
        grad_total += (1.0 / p) * dPhi_p_dr_kpc(R, r_p)
    return v0_kms * np.sqrt(R * grad_total)


def load_rotmod(path: Path) -> dict:
    R, Vobs, errV, Vgas, Vdisk, Vbul = [], [], [], [], [], []
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            R.append(float(parts[0])); Vobs.append(float(parts[1]))
            errV.append(float(parts[2])); Vgas.append(float(parts[3]))
            Vdisk.append(float(parts[4])); Vbul.append(float(parts[5]))
    return dict(R=np.asarray(R), Vobs=np.asarray(Vobs), errV=np.asarray(errV),
                Vgas=np.asarray(Vgas), Vdisk=np.asarray(Vdisk), Vbul=np.asarray(Vbul))


def evaluate_galaxy(
    name: str,
    path: Path,
    v0_kms: float,
    r_p_scaling: str,
    min_floor_err: float = 1.0,
) -> dict:
    d = load_rotmod(path)
    R = d["R"]
    keep = R > 0
    R = R[keep]; Vobs = d["Vobs"][keep]; errV = np.maximum(d["errV"][keep], min_floor_err)
    Vbar = np.sqrt(d["Vgas"][keep] ** 2 + d["Vdisk"][keep] ** 2 + d["Vbul"][keep] ** 2)
    if len(R) < 3:
        return {"name": name, "skipped": True}

    v_prime = v_prime_multichannel_kms(R, v0_kms, r_p_scaling=r_p_scaling)
    v_total = np.sqrt(Vbar ** 2 + v_prime ** 2)
    residuals = Vobs - v_total
    chi2 = float(np.sum((residuals / errV) ** 2))
    dof = max(len(R) - 0, 1)
    return {
        "name": name,
        "n_points": len(R),
        "chi2": chi2,
        "dof": dof,
        "chi2_per_dof": chi2 / dof,
        "rms_residual_kms": float(np.sqrt(np.mean(residuals ** 2))),
        "v_prime_at_outer_kms": float(v_prime[-1]),
        "v_obs_outer_kms": float(Vobs[-1]),
    }


def run_population(v0_kms: float, scaling: str) -> dict:
    files = sorted(SPARC_DIR.glob("*_rotmod.dat"))
    results, skipped = [], []
    for fp in files:
        name = fp.stem.replace("_rotmod", "")
        try:
            r = evaluate_galaxy(name, fp, v0_kms=v0_kms, r_p_scaling=scaling)
            if r.get("skipped"):
                skipped.append(name)
            else:
                results.append(r)
        except Exception as e:
            skipped.append(f"{name} ({e})")
    chi2_per_dof = np.array([r["chi2_per_dof"] for r in results])
    return {
        "v0_kms": v0_kms,
        "r_p_scaling": scaling,
        "n_evaluated": len(results),
        "n_skipped": len(skipped),
        "chi2_per_dof_median": float(np.median(chi2_per_dof)),
        "chi2_per_dof_mean": float(np.mean(chi2_per_dof)),
        "frac_under_1": float(np.mean(chi2_per_dof < 1)),
        "frac_under_5": float(np.mean(chi2_per_dof < 5)),
        "frac_under_10": float(np.mean(chi2_per_dof < 10)),
        "frac_under_50": float(np.mean(chi2_per_dof < 50)),
        "per_galaxy": results,
        "skipped_names": skipped,
    }


def main() -> int:
    print(f"Multi-channel SPARC test using gai 18-channel primes: {GAI_PRIMES}")
    print(f"r_0 base (kpc) = {R0_KPC}")
    print()

    cases = [
        (397.0, "linear_p"),    # universal v_0, r_p = r_0·p (book scaling)
        (200.0, "linear_p"),    # half v_0
        (100.0, "linear_p"),    # quarter v_0
        (397.0, "log_p"),       # log scaling
        (397.0, "p_squared"),   # quadratic scaling
    ]
    summary = {"cases": []}
    print(f"{'v_0':>6} {'scaling':<12} {'n_eval':>7} {'med χ²/dof':>13} "
          f"{'mean χ²/dof':>13} {'< 5':>8} {'< 10':>8} {'< 50':>8}")
    print("-" * 78)
    for v0, scaling in cases:
        r = run_population(v0, scaling)
        print(f"{v0:>6.0f} {scaling:<12} {r['n_evaluated']:>7} "
              f"{r['chi2_per_dof_median']:>13.2f} {r['chi2_per_dof_mean']:>13.2f} "
              f"{r['frac_under_5']:>8.1%} {r['frac_under_10']:>8.1%} {r['frac_under_50']:>8.1%}")
        # store summary only (not full per_galaxy, save space)
        summary["cases"].append({k: v for k, v in r.items() if k not in ("per_galaxy",)})

    out_file = OUT_DIR / "sparc_multichannel_results.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
