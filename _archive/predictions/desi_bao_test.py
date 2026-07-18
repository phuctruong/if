#!/usr/bin/env python3
"""
desi_bao_test.py — IF Theory dark-energy prediction vs DESI DR1 BAO.

The IF Theory "Bubble Universe" predicts w(z) ≈ -0.999995 across all z
(claim #11), i.e., essentially indistinguishable from a cosmological
constant (w₀ = -1, w_a = 0). DESI 2024 (DR1) released BAO measurements
at 7 redshift bins and reported headline tension with ΛCDM in the
w₀-w_a parameter space.

This test asks: is ΛCDM-equivalent IF Theory consistent with DESI DR1
BAO data alone (no Pantheon, no Planck)?

Procedure:
  1. Load DESI BAO measurements: 12 (z, value, quantity) tuples for
     DV/rs, DM/rs, DH/rs.
  2. Load block-diagonal 12×12 covariance matrix.
  3. Compute ΛCDM theoretical predictions for the same 12 (z, quantity)
     using Planck 2018 cosmology (Ωm = 0.315, h = 0.674, rs = 147.0 Mpc).
  4. Compute χ² = (data - theory)^T · C^-1 · (data - theory)
  5. Report χ²/dof. Match within 2σ ≈ χ² ~ 12 (12 dof) is acceptable.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Tuple

import astropy.units as u
import numpy as np
from astropy.cosmology import FlatLambdaCDM

DESI_DIR = Path("/home/phuc/Downloads/if/data/desi_dr1/bao_likelihoods")
OUT_DIR = Path(__file__).resolve().parent.parent / "evidence" / "desi_bao"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Planck 2018 best-fit (TT,TE,EE+lowE+lensing) — used as the IF Theory
# "ΛCDM-equivalent" cosmology since IF Theory predicts w ≈ -1.
PLANCK_OMEGA_M = 0.315
PLANCK_H = 0.674
RS_MPC = 147.0  # sound horizon at drag epoch (Planck 2018)

CLIGHT_KMS = 299792.458


def load_desi_means(path: Path) -> List[Tuple[float, float, str]]:
    rows = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 3:
                continue
            rows.append((float(parts[0]), float(parts[1]), parts[2]))
    return rows


def load_desi_cov(path: Path) -> np.ndarray:
    rows = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            try:
                rows.append([float(p) for p in parts])
            except ValueError:
                continue
    return np.asarray(rows)


def predict(z: float, kind: str, cosmo: FlatLambdaCDM, rs: float) -> float:
    """Return predicted value of DV/rs, DM/rs, or DH/rs at z."""
    if kind == "DM_over_rs":
        DM = cosmo.comoving_distance(z).to(u.Mpc).value
        return DM / rs
    elif kind == "DH_over_rs":
        Hz_kmps_per_Mpc = cosmo.H(z).to(u.km / u.s / u.Mpc).value
        DH = CLIGHT_KMS / Hz_kmps_per_Mpc
        return DH / rs
    elif kind == "DV_over_rs":
        DM = cosmo.comoving_distance(z).to(u.Mpc).value
        Hz = cosmo.H(z).to(u.km / u.s / u.Mpc).value
        DH = CLIGHT_KMS / Hz
        DV = (z * DM ** 2 * DH) ** (1.0 / 3.0)
        return DV / rs
    else:
        raise ValueError(f"unknown kind: {kind}")


def verdict_for_p_value(p_value: float) -> str:
    if p_value > 0.05:
        return f"CONSISTENT (p = {p_value:.3f} > 0.05)"
    if p_value > 0.003:  # ~3σ
        return f"TENSION-2σ (p = {p_value:.3f})"
    return f"FAILED (p = {p_value:.4f}, > 3σ)"


def exit_code_for_p_value(p_value: float) -> int:
    """Return nonzero only for publication-blocking DESI disagreement."""
    return 0 if p_value > 0.003 else 1


def main() -> int:
    means_path = DESI_DIR / "desi_2024_gaussian_bao_ALL_GCcomb_mean.txt"
    cov_path = DESI_DIR / "desi_2024_gaussian_bao_ALL_GCcomb_cov.txt"
    rows = load_desi_means(means_path)
    cov = load_desi_cov(cov_path)
    print(f"Loaded {len(rows)} DESI measurements, covariance shape {cov.shape}")
    assert cov.shape == (len(rows), len(rows))

    cosmo = FlatLambdaCDM(H0=PLANCK_H * 100, Om0=PLANCK_OMEGA_M)
    print("ΛCDM (IF Theory bubble: w ≈ -1):")
    print(f"  Ω_m = {PLANCK_OMEGA_M}, h = {PLANCK_H}, r_s = {RS_MPC:.1f} Mpc")
    print()

    z_arr, data_arr, theory_arr, kind_arr = [], [], [], []
    print(f"{'z':>6} {'kind':<14} {'data':>10} {'theory':>10} {'(d-t)/σ':>10}")
    sigmas = np.sqrt(np.diag(cov))
    for i, (z, val, kind) in enumerate(rows):
        pred = predict(z, kind, cosmo, RS_MPC)
        diff_sigma = (val - pred) / sigmas[i]
        print(f"{z:>6.3f} {kind:<14} {val:>10.4f} {pred:>10.4f} {diff_sigma:>+10.2f}")
        z_arr.append(z)
        data_arr.append(val)
        theory_arr.append(pred)
        kind_arr.append(kind)

    data = np.asarray(data_arr)
    theory = np.asarray(theory_arr)
    diff = data - theory

    # χ² with full covariance
    cov_inv = np.linalg.inv(cov)
    chi2 = float(diff @ cov_inv @ diff)
    dof = len(rows)

    print()
    print("=" * 60)
    print(f"χ² (full covariance) = {chi2:.2f}")
    print(f"dof                  = {dof}")
    print(f"χ²/dof               = {chi2 / dof:.2f}")
    print(f"P(χ² > {chi2:.1f}) for {dof} dof: see scipy.stats.chi2.sf")
    from scipy.stats import chi2 as chi2dist
    p_value = float(chi2dist.sf(chi2, dof))
    print(f"P-value              = {p_value:.4f}")
    verdict = verdict_for_p_value(p_value)
    print(f"VERDICT              = {verdict}")
    print("=" * 60)

    out = {
        "cosmology": "Planck 2018 ΛCDM (IF Theory bubble: w₀ = -0.999995 ≈ -1)",
        "Omega_m": PLANCK_OMEGA_M,
        "h": PLANCK_H,
        "rs_mpc": RS_MPC,
        "n_measurements": len(rows),
        "chi2": chi2,
        "dof": dof,
        "chi2_per_dof": chi2 / dof,
        "p_value": p_value,
        "verdict": verdict,
        "per_point": [
            {"z": z, "kind": k, "data": d, "theory": t, "sigma": float(s)}
            for z, k, d, t, s in zip(z_arr, kind_arr, data_arr, theory_arr, sigmas, strict=False)
        ],
    }
    with open(OUT_DIR / "desi_bao_lcdm_test.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT_DIR / 'desi_bao_lcdm_test.json'}")

    return exit_code_for_p_value(p_value)


if __name__ == "__main__":
    sys.exit(main())
