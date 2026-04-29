#!/usr/bin/env python3
"""
if_theory_minimal_folding.py — minimal IF Theory protein folding test.

Without the full gai folding model, we test what a SEQUENCE-INDEPENDENT
mean-d(k) prediction can achieve on PDB targets. This sets the floor
for any "universal-shape only" folding theory.

Method:
  1. From the corpus of PDB structures, compute the population-average
     d(k) curve for each sequence separation k.
  2. For a target structure of length L, predict the distance matrix:
        D_pred[i, j] = d_mean(|i - j|)
     for all (i, j).
  3. Apply classical MDS to recover a 3D structure from D_pred.
  4. Procrustes-align with the experimental structure.
  5. Compute RMSD and TM-score-like agreement.

What this test does:
  - Sets the lower bound on what any universal-d(k) folding model
    achieves. A real folding model must do BETTER than this.
  - Shows whether IF Theory's d(k) shape (Φ(k) form from contact_shape_test)
    captures meaningful protein structure information.

What this test does NOT do:
  - Test sequence-specific folding (no AA-type information used)
  - Compete with AlphaFold (which uses MSA + Evoformer to derive
    per-protein-specific contact maps)

Reference value: AlphaFold typical TM ≈ 0.85 on hard CASP targets;
random coil baseline TM ≈ 0.05; native fold TM = 1.00.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PDB_DIR = Path("/home/phuc/Downloads/if/data/pdb")
OUT_DIR = Path(__file__).resolve().parent.parent / "evidence" / "if_minimal_folding"
OUT_DIR.mkdir(parents=True, exist_ok=True)


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


def universal_d_k_curve(structures: Dict[str, np.ndarray], k_max: int = 200) -> np.ndarray:
    """Compute population-average mean d(k) for k = 1..k_max across structures."""
    accum = {k: [] for k in range(1, k_max + 1)}
    for X in structures.values():
        if X is None or len(X) < 5:
            continue
        L = len(X)
        for k in range(1, min(k_max + 1, L)):
            # All (i, j) with j = i + k
            d = np.linalg.norm(X[:L - k] - X[k:], axis=1)
            accum[k].append(d.mean())
    out = np.zeros(k_max + 1)
    for k in range(1, k_max + 1):
        if accum[k]:
            out[k] = float(np.mean(accum[k]))
    return out


def mds_3d(D: np.ndarray) -> np.ndarray:
    """Classical MDS: distance matrix → 3D coords (modulo rigid motion)."""
    N = D.shape[0]
    J = np.eye(N) - np.ones((N, N)) / N
    B = -0.5 * J @ (D ** 2) @ J
    B = 0.5 * (B + B.T)
    eigvals, eigvecs = np.linalg.eigh(B)
    idx = np.argsort(eigvals)[::-1][:3]
    L = np.maximum(eigvals[idx], 0.0)
    V = eigvecs[:, idx]
    return V * np.sqrt(L)


def procrustes_rmsd(X: np.ndarray, Y: np.ndarray) -> float:
    Xc = X - X.mean(axis=0)
    Yc = Y - Y.mean(axis=0)
    H = Xc.T @ Yc
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    Y_rot = Yc @ R
    diff = Xc - Y_rot
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=-1))))


def tm_like_score(X: np.ndarray, Y: np.ndarray) -> float:
    """A simple TM-like score: 1 / (1 + (RMSD / d_0)²) where d_0 = 5.0 Å.
    Not the proper Zhang/Skolnick TM-score (which requires a specific
    sequence alignment), but a rough proxy in [0, 1]."""
    rmsd = procrustes_rmsd(X, Y)
    d0 = 5.0  # Angstroms
    return 1.0 / (1.0 + (rmsd / d0) ** 2)


def main() -> int:
    pdb_files = sorted(PDB_DIR.glob("*.pdb"))
    structures = {fp.stem.upper(): load_pdb_ca(fp) for fp in pdb_files}
    structures = {k: v for k, v in structures.items() if v is not None and len(v) >= 20}
    print(f"Loaded {len(structures)} PDB structures with ≥20 CA atoms")

    # Compute universal d(k)
    d_k = universal_d_k_curve(structures, k_max=200)
    print(f"Computed universal d(k) curve for k = 1..200")
    print(f"  d(1)   = {d_k[1]:.2f} Å  (CA-CA bond length)")
    print(f"  d(5)   = {d_k[5]:.2f} Å  (alpha helix turn)")
    print(f"  d(20)  = {d_k[20]:.2f} Å")
    print(f"  d(50)  = {d_k[50]:.2f} Å")

    # For each target, predict structure using ONLY length + universal d(k)
    print(f"\nLeave-one-out test: predict each target's structure from")
    print(f"the universal d(k) curve computed without that target.")
    print()
    print(f"{'PDB':<8} {'L':>5} {'d̄_obs':>8} {'RMSD (Å)':>10} {'TM-like':>10}")
    print("-" * 60)
    results = []
    for name, X in structures.items():
        L = len(X)
        # Leave-one-out: rebuild d_k without this protein
        loo = {k: v for k, v in structures.items() if k != name}
        d_k_loo = universal_d_k_curve(loo, k_max=L)
        # Predict distance matrix
        D_pred = np.zeros((L, L))
        for i in range(L):
            for j in range(L):
                k = abs(i - j)
                D_pred[i, j] = d_k_loo[k] if k <= L else d_k_loo[-1]
        # MDS
        Y = mds_3d(D_pred)
        # Score
        rmsd = procrustes_rmsd(X, Y)
        tm = tm_like_score(X, Y)
        d_mean_obs = float(np.linalg.norm(X[:-1] - X[1:], axis=1).mean())
        results.append({"pdb": name, "L": L, "rmsd_A": rmsd, "tm_like": tm,
                        "d_mean_obs_A": d_mean_obs})
        print(f"{name:<8} {L:>5} {d_mean_obs:>8.2f} {rmsd:>10.2f} {tm:>10.4f}")

    rmsds = [r["rmsd_A"] for r in results]
    tms = [r["tm_like"] for r in results]
    print()
    print(f"Median RMSD : {np.median(rmsds):.2f} Å")
    print(f"Median TM-like: {np.median(tms):.4f}")
    print(f"  (random coil ≈ 0.05; AlphaFold ≈ 0.85; native = 1.00)")
    print()

    out = {
        "test": "universal-d(k)-only minimal folding (no sequence info)",
        "n_structures": len(structures),
        "median_rmsd_A": float(np.median(rmsds)),
        "median_tm_like": float(np.median(tms)),
        "results": results,
    }
    with open(OUT_DIR / "minimal_folding_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {OUT_DIR / 'minimal_folding_results.json'}")

    # Verdict
    if np.median(tms) > 0.5:
        verdict = "STRONG — universal d(k) alone gives meaningful folds"
    elif np.median(tms) > 0.2:
        verdict = "MODERATE — universal d(k) carries non-trivial structural info"
    else:
        verdict = "WEAK — universal d(k) alone insufficient; sequence info needed"
    print(f"VERDICT: {verdict}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
