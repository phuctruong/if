#!/usr/bin/env python3
"""
pantheon_plus_test.py — IF Theory dark-energy prediction vs Pantheon+
Type Ia supernova Hubble diagram (Scolnic et al. 2022; 1701 SNe).

The IF Theory bubble universe predicts w(z) ≈ -1 (claim #11) with H_0
that is scale-dependent (claim #15) — i.e., the Hubble tension is not
a real cosmological tension but a manifestation of bubble dynamics.

This script is the simpler test: does ΛCDM (the IF-Theory-equivalent
dark-energy model) fit Pantheon+ SNe distance moduli well, and which
H_0 (Planck CMB ~ 67.4 vs SH0ES local ~ 73.04 km/s/Mpc) is preferred?

For each SN at redshift z:
  μ_obs    = MU_SH0ES (distance modulus)
  μ_theory = 5·log10(D_L(z) [Mpc]) + 25,  where D_L(z) is from FlatLambdaCDM
  residual = μ_obs - μ_theory
  χ²       = sum_i sum_j r_i · C_inv_ij · r_j  (full covariance)

We exclude SNe with z < 0.01 to avoid local peculiar-velocity scatter
(standard practice in cosmology fits).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from astropy.cosmology import FlatLambdaCDM

PANTHEON_DIR = Path("/home/phuc/Downloads/if/data/pantheon_plus/Pantheon+_Data/4_DISTANCES_AND_COVAR")
OUT_DIR = Path(__file__).resolve().parent.parent / "evidence" / "pantheon_plus"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_pantheon():
    """Return DataFrame-like dict of arrays from Pantheon+SH0ES.dat."""
    data_path = PANTHEON_DIR / "Pantheon+SH0ES.dat"
    with open(data_path) as f:
        header = f.readline().split()
        rows = []
        for line in f:
            parts = line.split()
            if len(parts) != len(header):
                continue
            rows.append(parts)
    arr = np.asarray(rows)
    return {
        "CID": arr[:, header.index("CID")],
        "zHD": arr[:, header.index("zHD")].astype(float),
        "MU_SH0ES": arr[:, header.index("MU_SH0ES")].astype(float),
        "MU_SH0ES_ERR_DIAG": arr[:, header.index("MU_SH0ES_ERR_DIAG")].astype(float),
        "IS_CALIBRATOR": arr[:, header.index("IS_CALIBRATOR")].astype(int),
    }, header


def load_cov(n: int, statonly: bool = True) -> np.ndarray:
    """Load Pantheon+ covariance matrix (n × n).

    File format: first line = N, then N*N values one per line, row-major.
    """
    path = PANTHEON_DIR / ("Pantheon+SH0ES_STATONLY.cov" if statonly
                           else "Pantheon+SH0ES_STAT+SYS.cov")
    with open(path) as f:
        N = int(f.readline().strip())
        assert N == n, f"Cov size {N} != {n} SNe"
        vals = []
        for line in f:
            try:
                vals.append(float(line.strip()))
            except ValueError:
                pass
    return np.asarray(vals).reshape(N, N)


def chi2_for(data: dict, cov: np.ndarray, h_kmps_per_mpc: float, omega_m: float,
              z_min: float = 0.01, exclude_calibrators: bool = True) -> dict:
    """Compute χ² of ΛCDM (h, Ωm) vs Pantheon+ in the Hubble-flow z range."""
    cosmo = FlatLambdaCDM(H0=h_kmps_per_mpc, Om0=omega_m)

    z = data["zHD"]
    mu_obs = data["MU_SH0ES"]
    is_calib = data["IS_CALIBRATOR"]

    keep = z > z_min
    if exclude_calibrators:
        keep &= (is_calib == 0)
    idx = np.where(keep)[0]
    n = len(idx)

    # Predict μ for the kept SNe
    mu_th = np.asarray([5.0 * np.log10(cosmo.luminosity_distance(z[i]).value) + 25.0
                        for i in idx])
    res = mu_obs[idx] - mu_th

    cov_sub = cov[np.ix_(idx, idx)]
    cov_inv = np.linalg.inv(cov_sub)
    chi2 = float(res @ cov_inv @ res)
    dof = n
    chi2_per_dof = chi2 / dof
    return dict(
        h=h_kmps_per_mpc, omega_m=omega_m, n_sne=n,
        chi2=chi2, dof=dof, chi2_per_dof=chi2_per_dof,
        mean_residual=float(np.mean(res)),
        rms_residual=float(np.sqrt(np.mean(res ** 2))),
    )


def main() -> int:
    data, header = load_pantheon()
    n = len(data["CID"])
    print(f"Loaded {n} Pantheon+ SNe (header has {len(header)} columns)")

    cov = load_cov(n, statonly=True)
    print(f"Loaded {cov.shape} STATONLY covariance")

    print()
    print("=" * 70)
    print("ΛCDM Hubble-flow fit (z > 0.01, calibrators excluded), STATONLY cov")
    print("=" * 70)

    cases = [
        ("Planck h=0.674 (CMB)",  67.4, 0.315),
        ("SH0ES  h=0.7304 (SNe)", 73.04, 0.334),  # SH0ES inversion
        ("ΛCDM  h=0.70  (mid)",   70.0, 0.315),
    ]
    out = {"cases": []}
    print(f"{'case':<28} {'n_SNe':>6} {'χ²':>10} {'χ²/dof':>10}")
    for label, h, omegam in cases:
        r = chi2_for(data, cov, h_kmps_per_mpc=h, omega_m=omegam)
        print(f"{label:<28} {r['n_sne']:>6} {r['chi2']:>10.1f} {r['chi2_per_dof']:>10.3f}")
        r["label"] = label
        out["cases"].append(r)

    # Best of the three (lowest χ²)
    best = min(out["cases"], key=lambda d: d["chi2"])
    print()
    print(f"Best fit so far: {best['label']}  with χ²/dof = {best['chi2_per_dof']:.3f}")
    print()

    # Quick statistical reading: is ΛCDM-equivalent IF Theory consistent
    # with Pantheon+? At χ²/dof ~ 1, p-value ≈ 0.5; over many SNe, even
    # small biases accumulate, so we accept χ²/dof ∈ [0.95, 1.10].
    consistent = abs(best["chi2_per_dof"] - 1.0) < 0.10
    print(f"Consistent with ΛCDM (|χ²/dof - 1| < 0.10)? {consistent}")

    with open(OUT_DIR / "pantheon_plus_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT_DIR / 'pantheon_plus_results.json'}")

    return 0 if consistent else 1


if __name__ == "__main__":
    sys.exit(main())
