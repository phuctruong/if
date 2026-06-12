#!/usr/bin/env python3
"""lowz_clustering_replication.py — independent end-to-end clustering replication.

Referee artifact (Claude Fable 5 review loop, 2026-06-12). Unlike
boss_published_xi_test.py (which uses the published Cuesta 2016
consensus ξ), this measures ξ(r) DIRECTLY from the staged SDSS DR12
LOWZ South galaxy catalog + random0 catalog with a transparent
Landy-Szalay estimator, then asks the discriminating question:

    Does the IF shape [1/log(r/r0+1)]² describe the MEASURED ξ(r)
    better than an untuned power-law null?

Config mirrors the dark_matter_sdss.ipynb 'quick' test (25k galaxies,
10× randoms, 15 bins, fitting range 20–80 Mpc) so the result is
comparable to the notebook's claimed r = 0.984 / χ²/dof = 3.9 table.

Honesty notes:
- FKP/systematic weights are NOT applied (shape-level test only; the
  notebook's sdss_util applies the same simplification at quick tier).
- Errors are Poisson on DD per bin (no jackknife here) — fine for a
  shape comparison, declared so nobody reads the χ² as survey-grade.
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.cosmology import Planck15
from scipy.spatial import cKDTree

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
from prime_field_util import R0_KPC_CANONICAL  # noqa: E402

DATA = Path.home() / "Downloads" / "if" / "data" / "sdss_dr12" / "lowz"
OUT_DIR = _ROOT / "evidence" / "adversarial"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_GAL = 25_000
RANDOM_FACTOR = 10
N_BINS = 15
R_MIN, R_MAX = 1.0, 150.0
FIT_RANGE = (20.0, 80.0)
Z_RANGE = (0.15, 0.43)
SEED = 65537


def radecz_to_xyz(ra, dec, z):
    dc = Planck15.comoving_distance(z).value  # Mpc
    ra_r, dec_r = np.radians(ra), np.radians(dec)
    return np.column_stack([
        dc * np.cos(dec_r) * np.cos(ra_r),
        dc * np.cos(dec_r) * np.sin(ra_r),
        dc * np.sin(dec_r),
    ])


def load_catalog(path: Path, n_max: int, rng) -> np.ndarray:
    with fits.open(path) as hdul:
        d = hdul[1].data
        ra, dec, z = d["RA"], d["DEC"], d["Z"]
    m = (z >= Z_RANGE[0]) & (z <= Z_RANGE[1])
    ra, dec, z = ra[m], dec[m], z[m]
    if len(ra) > n_max:
        idx = rng.choice(len(ra), n_max, replace=False)
        ra, dec, z = ra[idx], dec[idx], z[idx]
    return radecz_to_xyz(ra, dec, z)


def pair_counts(tree_a, tree_b, edges, same: bool) -> np.ndarray:
    cum = tree_a.count_neighbors(tree_b, edges)
    counts = np.diff(cum).astype(float)
    if same:
        counts /= 2.0  # each pair counted twice (self-pairs at r>0 absent)
    return counts


def main() -> int:
    rng = np.random.default_rng(SEED)
    gal_path = DATA / "galaxy_DR12v5_LOWZ_South.fits.gz"
    ran_path = DATA / "random0_DR12v5_LOWZ_South.fits.gz"
    for p in (gal_path, ran_path):
        if not p.exists():
            print(f"MISSING {p}")
            return 1

    print("Loading galaxies…")
    D = load_catalog(gal_path, N_GAL, rng)
    print(f"  {len(D)} galaxies (z {Z_RANGE[0]}–{Z_RANGE[1]})")
    print("Loading randoms…")
    R = load_catalog(ran_path, N_GAL * RANDOM_FACTOR, rng)
    print(f"  {len(R)} randoms")

    edges = np.logspace(np.log10(R_MIN), np.log10(R_MAX), N_BINS + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])

    print("Building trees + counting pairs (DD, DR, RR)…")
    tD, tR = cKDTree(D), cKDTree(R)
    DD = pair_counts(tD, tD, edges, same=True)
    DR = pair_counts(tD, tR, edges, same=False)
    RR = pair_counts(tR, tR, edges, same=True)

    nD, nR = float(len(D)), float(len(R))
    dd = DD / (nD * (nD - 1) / 2)
    dr = DR / (nD * nR)
    rr = RR / (nR * (nR - 1) / 2)
    valid = RR > 0
    xi = np.full(N_BINS, np.nan)
    xi[valid] = (dd[valid] - 2 * dr[valid] + rr[valid]) / rr[valid]
    xi_err = np.full(N_BINS, np.nan)
    xi_err[valid] = (1 + xi[valid]) / np.sqrt(np.maximum(DD[valid], 1.0))

    print("\n   r [Mpc]      xi(r)        err")
    for c, x, e in zip(centers, xi, xi_err):
        print(f"  {c:8.2f}  {x:10.4f}  {e:9.4f}")

    # --- shape comparison on the fitting range, positive xi only
    m = valid & (xi > 0) & (centers >= FIT_RANGE[0]) & (centers <= FIT_RANGE[1])
    r_fit, xi_fit = centers[m], xi[m]
    rk = r_fit * 1000.0
    pred_if = (1.0 / np.log(rk / R0_KPC_CANONICAL + 1.0)) ** 2
    pred_pl = r_fit ** -1.8  # exponent irrelevant for log-log pearson

    def pearson_log(a, b):
        return float(np.corrcoef(np.log(a), np.log(b))[0, 1])

    r_if = pearson_log(xi_fit, pred_if)
    r_pl = pearson_log(xi_fit, pred_pl)

    # amplitude-marginalized chi2 in log space (1 implicit param each):
    def chi2_shape(pred):
        resid = np.log(xi_fit) - np.log(pred)
        resid -= resid.mean()  # marginalize amplitude
        sig = xi_err[m] / xi_fit  # log-space sigma
        return float(np.sum((resid / sig) ** 2) / max(len(xi_fit) - 1, 1))

    chi2_if, chi2_pl = chi2_shape(pred_if), chi2_shape(pred_pl)

    print(f"\nFit range {FIT_RANGE} Mpc, n={m.sum()} bins, IF r0={R0_KPC_CANONICAL} kpc")
    print(f"  Pearson r(log):  IF = {r_if:+.4f}   power-law null = {r_pl:+.4f}")
    print(f"  shape χ²/dof (amplitude-marginalized): IF = {chi2_if:.2f}  null = {chi2_pl:.2f}")

    verdict = ("IF-FAVORED" if (r_if - r_pl) >= 0.01 and chi2_if < chi2_pl
               else "NON-DISCRIMINATING-OR-NULL-FAVORED")
    print(f"VERDICT: {verdict}")

    out = {
        "artifact": "independent LOWZ South Landy-Szalay replication",
        "n_galaxies": int(nD), "n_randoms": int(nR),
        "bins_mpc": centers.tolist(), "xi": xi.tolist(), "xi_err": xi_err.tolist(),
        "fit_range_mpc": FIT_RANGE,
        "pearson_log_IF": r_if, "pearson_log_power_law": r_pl,
        "chi2_shape_IF": chi2_if, "chi2_shape_power_law": chi2_pl,
        "caveats": ["no FKP/systot weights", "Poisson errors, no jackknife",
                    "single subsample seed 65537"],
        "verdict": verdict,
    }
    with open(OUT_DIR / "lowz_clustering_replication.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {OUT_DIR / 'lowz_clustering_replication.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
