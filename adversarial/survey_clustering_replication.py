#!/usr/bin/env python3
"""survey_clustering_replication.py — generalized end-to-end ξ(r) replication v2.

Supersedes lowz_clustering_replication.py (v1 kept; identical estimator)
with: (a) survey configs for BOSS LOWZ and DESI DR1 LRG SGC, (b) proper
completeness×FKP weights, (c) leave-one-out jackknife errors over RA
stripes. Verdict statistic mirrors the v2 locked criterion
(evidence/lss_bao_locked_prediction/lss_bao_locked_prediction_v2.json):
IF must beat the untuned power-law null on Pearson-r(log) margin AND
amplitude-marginalized shape χ².

Usage:
  python3 adversarial/survey_clustering_replication.py lowz
  python3 adversarial/survey_clustering_replication.py desi_lrg_sgc
"""
from __future__ import annotations

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

DATA = Path.home() / "Downloads" / "if" / "data"
OUT_DIR = _ROOT / "evidence" / "adversarial"

N_GAL = 25_000
RANDOM_FACTOR = 10
N_BINS = 15
R_MIN, R_MAX = 1.0, 150.0
FIT_RANGE = (5.0, 120.0)
N_JACKKNIFE = 8
SEED = 65537

SURVEYS = {
    "lowz": {
        "galaxies": DATA / "sdss_dr12/lowz/galaxy_DR12v5_LOWZ_South.fits.gz",
        "randoms": DATA / "sdss_dr12/lowz/random0_DR12v5_LOWZ_South.fits.gz",
        "z_range": (0.15, 0.43),
        "gal_weight": lambda d: (d["WEIGHT_SYSTOT"] * (d["WEIGHT_NOZ"] + d["WEIGHT_CP"] - 1.0)
                                 * d["WEIGHT_FKP"]),
        "ran_weight": lambda d: d["WEIGHT_FKP"],
    },
    "desi_lrg_sgc": {
        "galaxies": DATA / "desi_dr1/lss/LRG_SGC_clustering.dat.fits",
        "randoms": DATA / "desi_dr1/lss/LRG_SGC_0_clustering.ran.fits",
        "z_range": (0.4, 1.1),
        "gal_weight": lambda d: d["WEIGHT"] * d["WEIGHT_FKP"],
        "ran_weight": lambda d: d["WEIGHT"] * d["WEIGHT_FKP"],
    },
}


def load(path: Path, z_range, weight_fn, n_max, rng):
    with fits.open(path) as hdul:
        d = hdul[1].data
        ra, dec, z = np.asarray(d["RA"], float), np.asarray(d["DEC"], float), np.asarray(d["Z"], float)
        try:
            w = np.asarray(weight_fn(d), float)
        except KeyError:
            w = np.ones_like(z)
    m = (z >= z_range[0]) & (z <= z_range[1]) & np.isfinite(w) & (w > 0)
    ra, dec, z, w = ra[m], dec[m], z[m], w[m]
    if len(ra) > n_max:
        idx = rng.choice(len(ra), n_max, replace=False)
        ra, dec, z, w = ra[idx], dec[idx], z[idx], w[idx]
    dc = Planck15.comoving_distance(z).value
    ra_r, dec_r = np.radians(ra), np.radians(dec)
    xyz = np.column_stack([dc * np.cos(dec_r) * np.cos(ra_r),
                           dc * np.cos(dec_r) * np.sin(ra_r),
                           dc * np.sin(dec_r)])
    return xyz, w, ra


def xi_landy_szalay(D, wD, R, wR, edges):
    tD, tR = cKDTree(D), cKDTree(R)
    sD, sR = wD.sum(), wR.sum()
    DD = np.diff(tD.count_neighbors(tD, edges, weights=(wD, wD))) / 2.0
    DR = np.diff(tD.count_neighbors(tR, edges, weights=(wD, wR)))
    RR = np.diff(tR.count_neighbors(tR, edges, weights=(wR, wR))) / 2.0
    dd = DD / (sD * sD / 2.0)
    dr = DR / (sD * sR)
    rr = RR / (sR * sR / 2.0)
    xi = np.full(len(edges) - 1, np.nan)
    v = RR > 0
    xi[v] = (dd[v] - 2 * dr[v] + rr[v]) / rr[v]
    return xi


def main() -> int:
    survey = sys.argv[1] if len(sys.argv) > 1 else "lowz"
    cfg = SURVEYS[survey]
    rng = np.random.default_rng(SEED)
    for k in ("galaxies", "randoms"):
        if not cfg[k].exists():
            print(f"MISSING {cfg[k]}")
            return 1

    D, wD, raD = load(cfg["galaxies"], cfg["z_range"], cfg["gal_weight"], N_GAL, rng)
    R, wR, raR = load(cfg["randoms"], cfg["z_range"], cfg["ran_weight"], N_GAL * RANDOM_FACTOR, rng)
    print(f"{survey}: {len(D)} galaxies, {len(R)} randoms (weighted)")

    edges = np.logspace(np.log10(R_MIN), np.log10(R_MAX), N_BINS + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])
    xi = xi_landy_szalay(D, wD, R, wR, edges)

    # jackknife over RA stripes (leave-one-out)
    q = np.quantile(raR, np.linspace(0, 1, N_JACKKNIFE + 1))
    regD = np.clip(np.searchsorted(q, raD, side="right") - 1, 0, N_JACKKNIFE - 1)
    regR = np.clip(np.searchsorted(q, raR, side="right") - 1, 0, N_JACKKNIFE - 1)
    jk = []
    for j in range(N_JACKKNIFE):
        mD, mR = regD != j, regR != j
        jk.append(xi_landy_szalay(D[mD], wD[mD], R[mR], wR[mR], edges))
        print(f"  jackknife {j+1}/{N_JACKKNIFE} done")
    jk = np.array(jk)
    xi_err = np.sqrt((N_JACKKNIFE - 1) / N_JACKKNIFE *
                     np.nansum((jk - np.nanmean(jk, axis=0)) ** 2, axis=0))

    m = np.isfinite(xi) & (xi > 0) & (centers >= FIT_RANGE[0]) & (centers <= FIT_RANGE[1]) & (xi_err > 0)
    r_fit, xi_fit, err_fit = centers[m], xi[m], xi_err[m]
    rk = r_fit * 1000.0
    pred_if = (1.0 / np.log(rk / R0_KPC_CANONICAL + 1.0)) ** 2
    pred_pl = r_fit ** -1.8

    def pearson_log(p):
        return float(np.corrcoef(np.log(xi_fit), np.log(p))[0, 1])

    def chi2_shape(p):
        resid = np.log(xi_fit) - np.log(p)
        resid -= resid.mean()
        return float(np.sum((resid / (err_fit / xi_fit)) ** 2) / max(len(xi_fit) - 1, 1))

    r_if, r_pl = pearson_log(pred_if), pearson_log(pred_pl)
    x2_if, x2_pl = chi2_shape(pred_if), chi2_shape(pred_pl)
    passes = (r_if - r_pl >= 0.01) and (x2_pl / x2_if >= 2.0)
    verdict = ("IF-FAVORED (v2 lock criterion met)" if passes
               else "NON-DISCRIMINATING-OR-NULL-FAVORED (v2 lock criterion NOT met)")

    print(f"\n{survey} fit {FIT_RANGE} Mpc n={m.sum()} (weighted, jackknife errors):")
    print(f"  r(log): IF={r_if:+.4f}  null={r_pl:+.4f}   shape χ²/dof: IF={x2_if:.1f}  null={x2_pl:.1f}")
    print(f"VERDICT: {verdict}")

    out = {
        "artifact": f"end-to-end weighted+jackknife clustering replication v2 — {survey}",
        "n_galaxies": int(len(D)), "n_randoms": int(len(R)),
        "weights": "completeness x FKP (survey-standard)",
        "jackknife_regions": N_JACKKNIFE,
        "bins_mpc": centers.tolist(), "xi": xi.tolist(), "xi_err_jackknife": xi_err.tolist(),
        "fit_range_mpc": FIT_RANGE,
        "pearson_log_IF": r_if, "pearson_log_power_law": r_pl,
        "chi2_shape_IF": x2_if, "chi2_shape_power_law": x2_pl,
        "v2_lock_criterion_met": bool(passes),
        "verdict": verdict,
    }
    (OUT_DIR / f"survey_replication_{survey}.json").write_text(json.dumps(out, indent=2))
    print(f"Wrote {OUT_DIR / f'survey_replication_{survey}.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
