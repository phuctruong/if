#!/usr/bin/env python3
"""
protein_contact_shape_test.py — IF Theory shape test for protein
residue-residue contacts vs sequence separation.

This is the protein analog of the BOSS ξ(r) shape test (which gave
Pearson r = 0.98 in log-log space). Instead of galaxies separated in
3D space at distance r, we look at residues separated in 1D sequence
at separation k = |i - j|.

For each protein structure:
  - For sequence separations k = 1, 2, ..., L−1
  - count contacts: pairs (i, j) with |i−j| = k AND ||CA_i − CA_j|| < r_contact
  - normalize by the number of possible pairs at separation k (= L−k)
  - this gives P(contact | k) for each protein
  - average across all proteins

IF Theory prediction (claim #80 generalized to 1D): the contact density
follows the prime field shape:

    P_predicted(k) = A · Φ(k)²  with  Φ(k) = 1/log(k/k_0 + 1)

Test: fit (A, k_0), compute Pearson r between log(P_data) and
log(P_predicted), and χ²/dof. Compare against:
  - random (uniform) baseline
  - polymer-folded baseline P(k) ∝ k^(-3/2) (Flory ideal chain)
  - 1/k power-law baseline

Same structural pattern as the BOSS test:
  PASS if Pearson r > 0.90 and IF Theory beats the comparison models.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import pearsonr

PDB_DIR = Path("/home/phuc/Downloads/if/data/pdb")
AFDB_DIR = Path("/home/phuc/Downloads/if/data/afdb")
OUT_DIR = Path(__file__).resolve().parent.parent / "evidence" / "protein_contact_shape"
OUT_DIR.mkdir(parents=True, exist_ok=True)

R_CONTACT_ANG = 8.0  # standard CA-CA contact threshold


def load_pdb_ca(path: Path) -> Optional[np.ndarray]:
    coords = []
    with open(path) as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            atom = line[12:16].strip()
            if atom != "CA":
                continue
            try:
                coords.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
            except ValueError:
                continue
    return np.asarray(coords) if coords else None


def contact_probability(structure: np.ndarray, r_contact: float = R_CONTACT_ANG,
                        k_max: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """For sequence separation k = 1..k_max, return (k, P(contact|k))."""
    L = len(structure)
    diff = structure[:, None, :] - structure[None, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=-1))
    contact = d < r_contact
    k_arr = np.arange(1, min(k_max + 1, L))
    P = np.zeros_like(k_arr, dtype=float)
    for ki, k in enumerate(k_arr):
        # All (i, j) with j = i + k
        diag = np.diagonal(contact, offset=k)
        P[ki] = float(np.mean(diag))
    return k_arr, P


def aggregate_contact_probability(structures: Dict[str, np.ndarray],
                                   k_max: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """Average P(k) across structures (each weighted equally)."""
    by_k = {k: [] for k in range(1, k_max + 1)}
    for X in structures.values():
        if X is None or len(X) < 5:
            continue
        k_arr, P = contact_probability(X, k_max=k_max)
        for k, p in zip(k_arr, P):
            by_k[int(k)].append(p)
    k_out = []
    P_out = []
    for k in sorted(by_k):
        if by_k[k]:
            k_out.append(k)
            P_out.append(float(np.mean(by_k[k])))
    return np.asarray(k_out), np.asarray(P_out)


# Models
def model_random(k: np.ndarray, A: float) -> np.ndarray:
    return A * np.ones_like(k, dtype=float)


def model_polymer(k: np.ndarray, A: float) -> np.ndarray:
    """Flory ideal chain: P(k) ∝ k^(-3/2)."""
    return A * k.astype(float) ** (-1.5)


def model_power_law(k: np.ndarray, A: float, alpha: float) -> np.ndarray:
    return A * k.astype(float) ** (-alpha)


def model_if_theory(k: np.ndarray, A: float, k_0: float) -> np.ndarray:
    """IF Theory shape: P(k) ∝ Φ(k)² = (1/log(k/k_0 + 1))²."""
    return A * (1.0 / np.log(k.astype(float) / k_0 + 1.0)) ** 2


def fit_and_score(model, p0, k, P) -> dict:
    P_safe = np.maximum(P, 1e-12)
    try:
        popt, _ = curve_fit(model, k, P, p0=p0, maxfev=10000)
        pred = model(k, *popt)
        pred_safe = np.maximum(pred, 1e-12)
        chi2 = float(np.sum((P - pred) ** 2 / P_safe))
        r_p, _ = pearsonr(np.log(P_safe), np.log(pred_safe))
        return {"params": [float(p) for p in popt], "chi2": chi2,
                "pearson_r_log": float(r_p)}
    except Exception as e:
        return {"error": str(e)}


def evaluate(structures: Dict[str, np.ndarray], label: str, k_min: int = 4,
             k_max: int = 50) -> dict:
    k, P = aggregate_contact_probability(structures, k_max=k_max)
    mask = (k >= k_min) & (P > 0)
    k = k[mask]; P = P[mask]
    if len(k) < 5:
        return {"label": label, "skipped": True}

    out = {"label": label, "n_proteins": len(structures), "n_bins": int(len(k)),
           "k_values": k.tolist(), "P_data": P.tolist()}
    out["random"] = fit_and_score(model_random, [0.05], k, P)
    out["polymer_flory"] = fit_and_score(model_polymer, [1.0], k, P)
    out["power_law"] = fit_and_score(model_power_law, [1.0, 1.0], k, P)
    out["if_theory"] = fit_and_score(model_if_theory, [1.0, 5.0], k, P)
    return out


def main() -> int:
    pdb_files = sorted(PDB_DIR.glob("*.pdb"))
    pdb_struct = {fp.stem.upper(): load_pdb_ca(fp) for fp in pdb_files}
    pdb_struct = {k: v for k, v in pdb_struct.items() if v is not None and len(v) >= 20}

    afdb_files = sorted(list(AFDB_DIR.glob("*.cif")) + list(AFDB_DIR.glob("*.pdb")))
    afdb_struct = {fp.stem.upper(): load_pdb_ca(fp) for fp in afdb_files}
    afdb_struct = {k: v for k, v in afdb_struct.items() if v is not None and len(v) >= 20}

    print(f"Loaded {len(pdb_struct)} PDB and {len(afdb_struct)} AFDB structures (≥20 res)")
    out = {
        "pdb": evaluate(pdb_struct, "PDB experimental"),
        "afdb": evaluate(afdb_struct, "AlphaFold predictions") if afdb_struct else None,
    }

    print("\n" + "=" * 78)
    print("PROTEIN CONTACT-SHAPE TEST (sequence separation k = 4..50)")
    print("=" * 78)
    for key, d in out.items():
        if d is None or d.get("skipped"):
            continue
        print(f"\n[{key}] {d['label']} — {d['n_proteins']} proteins, {d['n_bins']} bins")
        for model_name in ("random", "polymer_flory", "power_law", "if_theory"):
            m = d.get(model_name, {})
            if "error" in m:
                print(f"  {model_name:<18} ERROR: {m['error']}")
                continue
            p = m.get("params", [])
            print(f"  {model_name:<18} χ² = {m['chi2']:8.4e}  "
                  f"Pearson r(log) = {m['pearson_r_log']:+.4f}  "
                  f"params = [{', '.join(f'{x:.3f}' for x in p)}]")

        # Verdict: best model
        scores = {n: d[n].get("pearson_r_log", -2)
                  for n in ("random", "polymer_flory", "power_law", "if_theory")
                  if "pearson_r_log" in d[n]}
        if scores:
            best = max(scores, key=scores.get)
            print(f"  → best model by Pearson r(log): {best} ({scores[best]:+.4f})")

    out_file = OUT_DIR / "protein_contact_shape_results.json"
    with open(out_file, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nWrote {out_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
