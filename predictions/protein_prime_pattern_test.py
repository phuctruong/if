#!/usr/bin/env python3
"""
protein_prime_pattern_test.py — IF Theory structural pattern test on
protein structures (PDB experimental + AFDB AlphaFold predictions).

Same pattern as BOSS ξ(r) shape test:
  1. Real public data (PDB + AFDB)
  2. IF Theory prediction for the structural distance distribution
  3. χ² / Pearson r against the prediction

Theory predictions tested:

  P_1  Pairwise CA-CA distance distribution n(r) decays slower than
       random sphere packing (which gives n(r) ∝ r² up to cutoff).
       IF Theory: prime field couples residues via Φ(r) = 1/log(r/r_0+1),
       so the residue-pair density should track this field shape.
       Concretely: n(r) ~ r^α with α < 2 in the asymptotic regime.

  P_2  PDB and AFDB give the same distance distribution shape (no fold-
       class bias) — i.e., the prime pattern is a universal structural
       property, not a fitting artifact.

  P_3  Distance distribution shape matches between proteins of different
       sizes (residue count) when normalized by R_g (radius of gyration).
       This is the protein analog of the Tully-Fisher universality test.

References:
  - Lelli, McGaugh, Schombert 2016 (SPARC) — analog test for galaxies
  - Cuesta et al. 2016 (BOSS ξ(r)) — analog test for LSS

Usage:  python3 predictions/protein_prime_pattern_test.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import pearsonr

PDB_DIR = Path("/home/phuc/Downloads/if/data/pdb")
AFDB_DIR = Path("/home/phuc/Downloads/if/data/afdb")
OUT_DIR = Path(__file__).resolve().parent.parent / "evidence" / "protein_prime"
OUT_DIR.mkdir(parents=True, exist_ok=True)

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from prime_field_util import R0_KPC_CANONICAL  # noqa: E402

# For protein-scale test, the IF Theory's r_0 = 0.6595 kpc is way off-scale.
# We test instead the SHAPE of n(r) vs the IF Theory functional form, with r_0
# fitted as one parameter (this is the protein-scale Resolution Prime; analog
# of how BOSS xi(r) needed regime-dependent r_0).


def load_pdb_ca(path: Path) -> Optional[np.ndarray]:
    coords = []
    with open(path) as f:
        for line in f:
            if not line.startswith("ATOM") and not line.startswith("HETATM"):
                continue
            atom = line[12:16].strip()
            if atom != "CA":
                continue
            try:
                x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
                coords.append((x, y, z))
            except ValueError:
                continue
    if not coords:
        return None
    return np.asarray(coords)


def pairwise_distances(X: np.ndarray) -> np.ndarray:
    diff = X[:, None, :] - X[None, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=-1))
    iu = np.triu_indices(len(X), k=1)
    return d[iu]


def radius_of_gyration(X: np.ndarray) -> float:
    Xc = X - X.mean(axis=0)
    return float(np.sqrt(np.mean(np.sum(Xc * Xc, axis=-1))))


def histogram_by_r_over_rg(structures: Dict[str, np.ndarray],
                            n_bins: int = 30,
                            r_min_norm: float = 0.1,
                            r_max_norm: float = 4.0) -> Tuple[np.ndarray, np.ndarray, int]:
    """Combined histogram of (pairwise_distance / R_g), summing across proteins.
    Returns (bin_centers, counts, n_proteins_used)."""
    bins = np.linspace(r_min_norm, r_max_norm, n_bins + 1)
    counts = np.zeros(n_bins, dtype=float)
    n_used = 0
    for name, X in structures.items():
        if X is None or len(X) < 4:
            continue
        Rg = radius_of_gyration(X)
        if Rg <= 0:
            continue
        d = pairwise_distances(X) / Rg
        h, _ = np.histogram(d, bins=bins)
        # normalize per protein (so big proteins don't dominate)
        if h.sum() > 0:
            counts += h / h.sum()
            n_used += 1
    centers = 0.5 * (bins[:-1] + bins[1:])
    return centers, counts, n_used


def model_random_sphere(r: np.ndarray, A: float, p: float, cutoff: float) -> np.ndarray:
    """Random uniform sphere: n(r) ∝ r² · g(r/cutoff). For comparison."""
    return A * (r ** 2) * np.exp(-(r / cutoff) ** p)


def model_if_theory(r: np.ndarray, A: float, r_0: float, alpha: float) -> np.ndarray:
    """IF Theory shape: n(r) ∝ r^alpha · Φ(r)² with Φ = 1/log(r/r_0+1).

    This is the structural analog of ξ(r) = C_XI · Φ²; the r^alpha prefactor
    accounts for the pair-counting volume factor (which is r² for random
    placement but should differ for structured prime patterns).
    """
    eps = 1e-12
    phi = 1.0 / np.log(np.maximum(r / r_0 + 1.0, 1.0 + eps))
    return A * (r ** alpha) * phi ** 2


def evaluate(structures: Dict[str, np.ndarray], label: str) -> dict:
    centers, counts, n_used = histogram_by_r_over_rg(structures)
    if n_used == 0:
        return {"label": label, "skipped": True}
    # mask zeros to avoid log issues
    mask = counts > 0
    r = centers[mask]
    c = counts[mask]

    out = {"label": label, "n_proteins": n_used, "n_bins": int(mask.sum())}

    # Fit random-sphere model
    try:
        popt, _ = curve_fit(model_random_sphere, r, c,
                             p0=[1.0, 2.0, 1.0], maxfev=10000,
                             bounds=([0, 0.1, 0.01], [np.inf, 10.0, 100.0]))
        c_pred = model_random_sphere(r, *popt)
        chi2 = float(np.sum((c - c_pred) ** 2 / np.maximum(c, 1e-6)))
        r_p, _ = pearsonr(np.log(c), np.log(np.maximum(c_pred, 1e-12)))
        out["random_sphere"] = {
            "A": float(popt[0]), "p_exponent": float(popt[1]),
            "cutoff": float(popt[2]),
            "chi2": chi2, "pearson_r_log": float(r_p),
        }
    except Exception as e:
        out["random_sphere"] = {"error": str(e)}

    # Fit IF Theory shape
    try:
        popt, _ = curve_fit(model_if_theory, r, c,
                             p0=[1.0, 0.1, 2.0], maxfev=10000,
                             bounds=([0, 1e-3, 0.0], [np.inf, 10.0, 10.0]))
        c_pred = model_if_theory(r, *popt)
        chi2 = float(np.sum((c - c_pred) ** 2 / np.maximum(c, 1e-6)))
        r_p, _ = pearsonr(np.log(c), np.log(np.maximum(c_pred, 1e-12)))
        out["if_theory"] = {
            "A": float(popt[0]), "r_0": float(popt[1]), "alpha": float(popt[2]),
            "chi2": chi2, "pearson_r_log": float(r_p),
        }
    except Exception as e:
        out["if_theory"] = {"error": str(e)}

    out["centers"] = centers.tolist()
    out["counts"] = counts.tolist()

    return out


def main() -> int:
    # Load PDB experimental structures
    pdb_files = sorted(PDB_DIR.glob("*.pdb"))
    pdb_struct = {}
    for fp in pdb_files:
        X = load_pdb_ca(fp)
        if X is not None and len(X) >= 10:
            pdb_struct[fp.stem.upper()] = X
    print(f"Loaded {len(pdb_struct)} PDB structures with ≥10 CA atoms")

    # Load AFDB structures (.cif or .pdb format)
    afdb_files = sorted(list(AFDB_DIR.glob("*.cif")) + list(AFDB_DIR.glob("*.pdb")))
    afdb_struct = {}
    for fp in afdb_files:
        X = load_pdb_ca(fp)
        if X is not None and len(X) >= 10:
            afdb_struct[fp.stem.upper()] = X
    print(f"Loaded {len(afdb_struct)} AFDB structures")

    out = {
        "pdb": evaluate(pdb_struct, "PDB experimental"),
        "afdb": evaluate(afdb_struct, "AlphaFold predictions") if afdb_struct else None,
    }

    print("\n" + "=" * 78)
    print("PROTEIN STRUCTURAL PATTERN TEST — IF Theory shape vs random sphere")
    print("=" * 78)
    for key, d in out.items():
        if d is None or d.get("skipped"):
            print(f"\n[{key}] SKIPPED")
            continue
        print(f"\n[{key}] {d.get('label', key)} — {d['n_proteins']} proteins, {d['n_bins']} bins")
        rs = d.get("random_sphere", {})
        ift = d.get("if_theory", {})
        if "chi2" in rs:
            print(f"  random sphere n(r) ∝ r²·g(r):     "
                  f"χ² = {rs['chi2']:.4f}, Pearson r(log) = {rs['pearson_r_log']:+.4f}, "
                  f"exponent = {rs['p_exponent']:.2f}")
        if "chi2" in ift:
            print(f"  IF Theory   n(r) ∝ r^α·Φ(r)²:     "
                  f"χ² = {ift['chi2']:.4f}, Pearson r(log) = {ift['pearson_r_log']:+.4f}, "
                  f"α = {ift['alpha']:.2f}, r_0 = {ift['r_0']:.3f}")
        if "chi2" in rs and "chi2" in ift:
            ratio = rs["chi2"] / ift["chi2"]
            print(f"  random / IF χ² ratio: {ratio:.2f}  "
                  f"({'IF wins' if ratio > 1 else 'random wins'})")

    out_file = OUT_DIR / "protein_prime_pattern_results.json"
    with open(out_file, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nWrote {out_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
