#!/usr/bin/env python3
"""
joint_cosmology_bayes.py — joint Bayesian comparison of IF Theory
(ΛCDM-equivalent: w₀ = -1, w_a = 0) vs evolving-dark-energy ΛCDM
(w₀, w_a free) on combined Pantheon+ SNe + DESI DR1 BAO data.

This closes the last "OPEN" item in SCORE.md: a model-evidence
comparison rather than per-test χ². Per Aaronson/Sagan: "let the
data speak across all probes simultaneously."

Models compared:
  H_IF       : w(z) = -0.999995 ≈ -1, w_a = 0  (Bubble Universe / IF Theory)
                — zero free parameters in the dark-energy sector
                — total free parameters (h + Ω_m): 2
  H_w0wa    : w₀, w_a free per DESI 2024 best fit (-0.83, -0.69)
                — 4 free parameters (h, Ω_m, w₀, w_a)

Combined likelihood: -2·ln(L) = χ²_Pantheon+ + χ²_DESI

Information criteria:
  AIC = -2·ln(L) + 2·k         (lower wins)
  BIC = -2·ln(L) + k·ln(N)     (lower wins; stronger param penalty)

The IF Theory wins if its χ² is comparable to (or only slightly worse
than) the 4-parameter best fit — the AIC/BIC penalty for 2 extra
parameters then favors the simpler model.

NB: this is a "Bayes-factor-flavored" comparison without full
posterior sampling. A proper emcee run is the next step (see
SCORE.md).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
from astropy.cosmology import Flatw0waCDM, FlatLambdaCDM
import astropy.units as u

DESI_DIR = Path("/home/phuc/Downloads/if/data/desi_dr1/bao_likelihoods")
PANTHEON_DIR = Path("/home/phuc/Downloads/if/data/pantheon_plus/Pantheon+_Data/4_DISTANCES_AND_COVAR")
OUT_DIR = Path(__file__).resolve().parent.parent / "evidence" / "joint_cosmology_bayes"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLIGHT_KMS = 299792.458
RS_MPC = 147.0


def load_desi_means(path: Path):
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
            try:
                rows.append([float(p) for p in s.split()])
            except ValueError:
                continue
    return np.asarray(rows)


def predict_bao(z: float, kind: str, cosmo) -> float:
    if kind == "DM_over_rs":
        return cosmo.comoving_distance(z).to(u.Mpc).value / RS_MPC
    elif kind == "DH_over_rs":
        Hz = cosmo.H(z).to(u.km / u.s / u.Mpc).value
        return (CLIGHT_KMS / Hz) / RS_MPC
    elif kind == "DV_over_rs":
        DM = cosmo.comoving_distance(z).to(u.Mpc).value
        Hz = cosmo.H(z).to(u.km / u.s / u.Mpc).value
        DH = CLIGHT_KMS / Hz
        return ((z * DM ** 2 * DH) ** (1.0 / 3.0)) / RS_MPC
    raise ValueError(kind)


def chi2_desi(cosmo) -> Tuple[float, int]:
    rows = load_desi_means(DESI_DIR / "desi_2024_gaussian_bao_ALL_GCcomb_mean.txt")
    cov = load_desi_cov(DESI_DIR / "desi_2024_gaussian_bao_ALL_GCcomb_cov.txt")
    data = np.asarray([r[1] for r in rows])
    theory = np.asarray([predict_bao(r[0], r[2], cosmo) for r in rows])
    diff = data - theory
    cov_inv = np.linalg.inv(cov)
    return float(diff @ cov_inv @ diff), len(rows)


def load_pantheon():
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
        "zHD": arr[:, header.index("zHD")].astype(float),
        "MU_SH0ES": arr[:, header.index("MU_SH0ES")].astype(float),
        "IS_CALIBRATOR": arr[:, header.index("IS_CALIBRATOR")].astype(int),
    }


def load_pantheon_cov(n: int) -> np.ndarray:
    path = PANTHEON_DIR / "Pantheon+SH0ES_STATONLY.cov"
    with open(path) as f:
        N = int(f.readline().strip())
        assert N == n
        vals = [float(line.strip()) for line in f if line.strip()]
    return np.asarray(vals).reshape(N, N)


def chi2_pantheon(cosmo) -> Tuple[float, int]:
    data = load_pantheon()
    n = len(data["zHD"])
    cov = load_pantheon_cov(n)
    z = data["zHD"]
    keep = (z > 0.01) & (data["IS_CALIBRATOR"] == 0)
    idx = np.where(keep)[0]
    mu_obs = data["MU_SH0ES"][idx]
    z_keep = z[idx]
    mu_th = np.asarray([5.0 * np.log10(cosmo.luminosity_distance(zi).value) + 25.0
                        for zi in z_keep])
    res = mu_obs - mu_th
    cov_sub = cov[np.ix_(idx, idx)]
    cov_inv = np.linalg.inv(cov_sub)
    return float(res @ cov_inv @ res), len(idx)


def evaluate(name: str, cosmo, n_params: int) -> dict:
    chi2_d, n_d = chi2_desi(cosmo)
    chi2_p, n_p = chi2_pantheon(cosmo)
    chi2_total = chi2_d + chi2_p
    n_total = n_d + n_p
    aic = chi2_total + 2 * n_params
    bic = chi2_total + n_params * math.log(n_total)
    return {
        "model": name,
        "n_params": n_params,
        "chi2_desi": chi2_d, "n_desi": n_d,
        "chi2_pantheon": chi2_p, "n_pantheon": n_p,
        "chi2_total": chi2_total, "n_total": n_total,
        "chi2_per_dof": chi2_total / (n_total - n_params),
        "AIC": aic,
        "BIC": bic,
    }


def main() -> int:
    print("=" * 78)
    print("Joint Pantheon+ + DESI DR1 BAO model comparison")
    print("=" * 78)
    print()

    h_planck = 0.674
    omega_m = 0.315

    # H_IF: ΛCDM (w = -1, w_a = 0) at SH0ES h (since Pantheon+ prefers SH0ES)
    # We use SH0ES h to match Pantheon+ (the Hubble tension is then carried
    # by the bubble mechanism per claim #15, separately validated).
    h_sh0es = 0.7304
    cosmo_lcdm_sh0es = FlatLambdaCDM(H0=h_sh0es * 100, Om0=omega_m)
    cosmo_lcdm_planck = FlatLambdaCDM(H0=h_planck * 100, Om0=omega_m)

    # H_w0wa: DESI 2024 best fit
    cosmo_w0wa_planck = Flatw0waCDM(H0=h_planck * 100, Om0=omega_m, w0=-0.83, wa=-0.69)
    cosmo_w0wa_sh0es = Flatw0waCDM(H0=h_sh0es * 100, Om0=omega_m, w0=-0.83, wa=-0.69)

    cases = [
        ("ΛCDM (IF, h=Planck)",  cosmo_lcdm_planck, 2),
        ("ΛCDM (IF, h=SH0ES)",   cosmo_lcdm_sh0es, 2),
        ("w₀wa  (h=Planck, DESI best fit)", cosmo_w0wa_planck, 4),
        ("w₀wa  (h=SH0ES, DESI best fit)",  cosmo_w0wa_sh0es, 4),
    ]
    results = []
    for name, cosmo, k in cases:
        r = evaluate(name, cosmo, k)
        results.append(r)
        print(f"  {name:<38} k={k}  χ²_total={r['chi2_total']:>9.1f}  "
              f"χ²/dof={r['chi2_per_dof']:>5.2f}  AIC={r['AIC']:>9.1f}  "
              f"BIC={r['BIC']:>9.1f}")

    # Best by AIC and BIC
    best_aic = min(results, key=lambda r: r["AIC"])
    best_bic = min(results, key=lambda r: r["BIC"])
    print()
    print(f"Best by AIC: {best_aic['model']}  (AIC = {best_aic['AIC']:.1f})")
    print(f"Best by BIC: {best_bic['model']}  (BIC = {best_bic['BIC']:.1f})")
    print()
    print("Δ(IF best vs w₀wa best):")
    if_results = [r for r in results if "IF" in r["model"]]
    w0wa_results = [r for r in results if "w₀wa" in r["model"]]
    if_best = min(if_results, key=lambda r: r["AIC"])
    w0wa_best = min(w0wa_results, key=lambda r: r["AIC"])
    delta_aic = if_best["AIC"] - w0wa_best["AIC"]
    delta_bic = if_best["BIC"] - w0wa_best["BIC"]
    print(f"  ΔAIC = {delta_aic:+.1f}  (negative ⇒ IF preferred)")
    print(f"  ΔBIC = {delta_bic:+.1f}  (negative ⇒ IF preferred; BIC penalizes complexity more)")
    print()
    if delta_bic < -2:
        verdict = "IF Theory PREFERRED — fewer parameters and comparable fit"
    elif abs(delta_bic) < 2:
        verdict = "INDISTINGUISHABLE — both models fit at similar BIC"
    else:
        verdict = "IF Theory DISFAVORED — w₀wa fits substantially better"
    print(f"VERDICT: {verdict}")

    out = {
        "test": "Joint Pantheon+ + DESI DR1 BAO Bayesian comparison",
        "results": results,
        "delta_AIC_IF_minus_w0wa": float(delta_aic),
        "delta_BIC_IF_minus_w0wa": float(delta_bic),
        "verdict": verdict,
    }
    with open(OUT_DIR / "joint_bayes_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT_DIR / 'joint_bayes_results.json'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
