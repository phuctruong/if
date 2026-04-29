#!/usr/bin/env python3
"""
boss_published_xi_test.py — IF Theory correlation-function shape test
against the Cuesta et al. 2016 BOSS DR12 published consensus ξ(r)
measurements (LOWZ + CMASS monopole, pre-reconstruction).

For each sample we test:

    H_0 (zero free parameters): ξ(r) = C_XI · [Φ(r)]²
        with Φ(r) = 1/log(r/r_0 + 1), r_0 = 0.6595 kpc, C_XI = 62.

    H_1 (one free parameter):  ξ(r) = A · [Φ(r)]²
        amplitude A fitted; tests if the SHAPE matches even when amplitude
        is allowed to differ from the C_XI = 62 derivation.

    H_2 (two free parameters): ξ(r) = A · [Φ(r/r_eff)]²
        amplitude + effective r_0 both fitted; tests if a different scale
        for the BOSS-galaxy regime (vs the dark-matter-halo regime where
        r_0 = 0.6595 kpc was derived) rescues the fit.

Reports Pearson r between log(ξ_data) and log(ξ_predicted), plus χ²/dof
using the published errors.

Reference:
    Cuesta et al. 2016, MNRAS 457, 1770 (arXiv:1509.06371)
    DR12 LOWZ and CMASS correlation function monopoles.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import pearsonr

logging.basicConfig(level=logging.WARNING)

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from prime_field_util import R0_KPC_CANONICAL, C_XI_CANONICAL  # noqa: E402

BOSS_DIR = Path("/home/phuc/Downloads/if/data/boss_published_xi")
R0_MPC = R0_KPC_CANONICAL / 1000.0


def load_cuesta(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read a Cuesta 2016 corrfunction_x0_prerecon.dat file.
    Returns (r_mpc_h, xi, sigma_xi) with header lines skipped."""
    rows = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 3:
                continue
            try:
                rows.append((float(parts[0]), float(parts[1]), float(parts[2])))
            except ValueError:
                continue
    arr = np.asarray(rows)
    return arr[:, 0], arr[:, 1], arr[:, 2]


def field_phi(r_mpc: np.ndarray, r0_mpc: float = R0_MPC) -> np.ndarray:
    """IF Theory prime field Φ(r) = 1/log(r/r_0 + 1)."""
    r_kpc = r_mpc * 1000.0
    return 1.0 / np.log(r_kpc / (r0_mpc * 1000.0) + 1.0)


def xi_zero_param(r_mpc: np.ndarray) -> np.ndarray:
    """ξ(r) at zero free parameters: C_XI · [Φ(r)]²."""
    return C_XI_CANONICAL * field_phi(r_mpc) ** 2


def xi_one_param(r_mpc: np.ndarray, amplitude: float) -> np.ndarray:
    """ξ(r) with amplitude as free parameter: A · [Φ(r)]²."""
    return amplitude * field_phi(r_mpc) ** 2


def xi_two_param(r_mpc: np.ndarray, amplitude: float, r0_mpc_fit: float) -> np.ndarray:
    """ξ(r) with amplitude + effective r_0 free."""
    return amplitude * field_phi(r_mpc, r0_mpc_fit) ** 2


def evaluate_sample(name: str, path: Path, r_min: float = 8.0, r_max: float = 150.0,
                    out_dir: Path = Path(".")) -> dict:
    """Run all three hypotheses on a sample. Restrict to r in [r_min, r_max] Mpc/h
    where xi is well-resolved (avoids the smallest-r BAO issues and the noisy tail).
    """
    r_full, xi_data_full, sigma_full = load_cuesta(path)
    keep = (r_full >= r_min) & (r_full <= r_max) & (xi_data_full > 0)
    r = r_full[keep]
    xi = xi_data_full[keep]
    sig = sigma_full[keep]
    n = len(r)
    print(f"\n=== {name} ===")
    print(f"  Loaded {len(r_full)} bins; using {n} bins in r ∈ [{r_min}, {r_max}] Mpc/h")
    print(f"  ξ_data range: [{xi.min():.4f}, {xi.max():.4f}]")

    r_mpc = r / 0.6774  # convert Mpc/h to Mpc using Planck15 h ~ 0.6774
    # NOTE: BOSS r is in Mpc/h. The IF model is in Mpc. Convert.

    out = {"r_min_mpch": r_min, "r_max_mpch": r_max, "n_bins": n}

    # H_0: zero free parameters
    xi_h0 = xi_zero_param(r_mpc)
    chi2_h0 = float(np.sum(((xi - xi_h0) / sig) ** 2))
    dof_h0 = n
    log_xi = np.log(xi)
    log_xi_h0 = np.log(xi_h0)
    r_p_h0, _ = pearsonr(log_xi, log_xi_h0)
    out["H0"] = {
        "amplitude": "C_XI = 62 (derived)",
        "r0_mpc": R0_MPC,
        "free_params": 0,
        "pearson_r_log_log": float(r_p_h0),
        "chi2": chi2_h0,
        "dof": dof_h0,
        "chi2_per_dof": chi2_h0 / dof_h0,
    }
    print(f"  H_0 (zero params, C_XI=62, r_0=0.6595 kpc):")
    print(f"      Pearson r(log) = {r_p_h0:+.4f}; χ²/dof = {chi2_h0 / dof_h0:.2f}")
    print(f"      ξ_pred range: [{xi_h0.min():.4e}, {xi_h0.max():.4e}]")
    print(f"      ξ_data/ξ_pred mean: {np.mean(xi / xi_h0):.2e}")

    # H_1: amplitude fitted
    try:
        popt, _ = curve_fit(xi_one_param, r_mpc, xi, sigma=sig, absolute_sigma=True,
                            p0=[60.0], maxfev=10000)
        xi_h1 = xi_one_param(r_mpc, *popt)
        chi2_h1 = float(np.sum(((xi - xi_h1) / sig) ** 2))
        dof_h1 = n - 1
        r_p_h1, _ = pearsonr(log_xi, np.log(xi_h1))
        out["H1"] = {
            "amplitude_fitted": float(popt[0]),
            "r0_mpc": R0_MPC,
            "free_params": 1,
            "pearson_r_log_log": float(r_p_h1),
            "chi2": chi2_h1,
            "dof": dof_h1,
            "chi2_per_dof": chi2_h1 / dof_h1,
        }
        print(f"  H_1 (1 param: amplitude):")
        print(f"      A_fit = {popt[0]:.3e} ; r_0 = 0.6595 kpc fixed")
        print(f"      Pearson r(log) = {r_p_h1:+.4f}; χ²/dof = {chi2_h1 / dof_h1:.2f}")
    except Exception as e:
        print(f"  H_1 fit failed: {e}")
        out["H1"] = {"error": str(e)}

    # H_2: amplitude + r_0 fitted
    try:
        popt, _ = curve_fit(xi_two_param, r_mpc, xi, sigma=sig, absolute_sigma=True,
                            p0=[60.0, R0_MPC], maxfev=20000,
                            bounds=([0.0, 1e-6], [1e8, 100.0]))
        xi_h2 = xi_two_param(r_mpc, *popt)
        chi2_h2 = float(np.sum(((xi - xi_h2) / sig) ** 2))
        dof_h2 = n - 2
        r_p_h2, _ = pearsonr(log_xi, np.log(xi_h2))
        out["H2"] = {
            "amplitude_fitted": float(popt[0]),
            "r0_mpc_fitted": float(popt[1]),
            "r0_kpc_fitted": float(popt[1] * 1000),
            "free_params": 2,
            "pearson_r_log_log": float(r_p_h2),
            "chi2": chi2_h2,
            "dof": dof_h2,
            "chi2_per_dof": chi2_h2 / dof_h2,
        }
        print(f"  H_2 (2 params: amplitude + r_0):")
        print(f"      A_fit = {popt[0]:.3e} ; r_0_fit = {popt[1] * 1000:.3f} kpc")
        print(f"      Pearson r(log) = {r_p_h2:+.4f}; χ²/dof = {chi2_h2 / dof_h2:.2f}")
    except Exception as e:
        print(f"  H_2 fit failed: {e}")
        out["H2"] = {"error": str(e)}

    return out


def main() -> int:
    samples = {
        "LOWZ_DR12": BOSS_DIR / "Cuesta_2016_LOWZDR12_corrfunction_x0_prerecon.dat",
        "CMASS_DR12": BOSS_DIR / "Cuesta_2016_CMASSDR12_corrfunction_x0_prerecon.dat",
    }
    out = {}
    for name, path in samples.items():
        out[name] = evaluate_sample(name, path)

    # Summary
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    for name, d in out.items():
        h0 = d.get("H0", {})
        h1 = d.get("H1", {})
        h2 = d.get("H2", {})
        print(f"  {name:14s}  H0 χ²/dof={h0.get('chi2_per_dof', float('nan')):8.2e}  "
              f"H1 χ²/dof={h1.get('chi2_per_dof', float('nan')):6.2f}  "
              f"H2 χ²/dof={h2.get('chi2_per_dof', float('nan')):6.2f}")
        if h2:
            print(f"                    H2 r_0_fit = {h2.get('r0_kpc_fitted', float('nan')):.2f} kpc"
                  f"   (canonical = {R0_KPC_CANONICAL:.4f} kpc)")

    out_dir = Path(_ROOT, "evidence", "boss_published_xi")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "boss_xi_test.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nWrote {out_dir / 'boss_xi_test.json'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
