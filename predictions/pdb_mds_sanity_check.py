#!/usr/bin/env python3
"""
pdb_mds_sanity_check.py — verify that the gai claim "1D distance matrix
→ exact 3D structure via eigendecomposition" is just classical
Multidimensional Scaling (Young & Householder 1938; Schoenberg 1935).

For any 3D point set X ∈ R^{N×3}, given the pairwise distance matrix D:
    1. B := -1/2 · J · D² · J,  where J = I - 1/N · 1·1^T (centering)
    2. Eigendecompose B = V · Λ · V^T
    3. Take the top 3 eigenvectors weighted by √λ to recover X' (modulo rigid motion)
    4. Compute RMSD between X and X' after Procrustes alignment

For a true 3D point set, RMSD should be at numerical precision (~1e-13).

This is a 90-year-old technique. The claim "1D→3D via eigendecomposition"
proves nothing about protein folding — it's a mathematically trivial
recovery of coordinates from a distance matrix. The actual folding claim
must be tested as: SEQUENCE → STRUCTURE without seeing coordinates.

This script confirms the math is trivial across the 20 staged PDB
structures, refuting any claim that the 1D→3D recovery itself
constitutes folding.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

PDB_DIR = Path("/home/phuc/Downloads/if/data/pdb")
OUT_DIR = Path(__file__).resolve().parent.parent / "evidence" / "pdb_mds"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_pdb_ca_coords(path: Path) -> Optional[np.ndarray]:
    """Read ATOM CA coords from a PDB file. Returns N×3 numpy array."""
    coords = []
    with open(path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            atom = line[12:16].strip()
            if atom != "CA":
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords.append((x, y, z))
            except ValueError:
                continue
    if not coords:
        return None
    return np.asarray(coords)


def pairwise_dist(X: np.ndarray) -> np.ndarray:
    diff = X[:, None, :] - X[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1))


def mds_from_distance(D: np.ndarray, n_dim: int = 3) -> np.ndarray:
    """Classical MDS: D (N×N) → X' (N × n_dim) up to rigid motion."""
    N = D.shape[0]
    J = np.eye(N) - np.ones((N, N)) / N
    B = -0.5 * J @ (D ** 2) @ J
    # Symmetrize to avoid tiny asymmetry from float
    B = 0.5 * (B + B.T)
    eigvals, eigvecs = np.linalg.eigh(B)
    # Take top n_dim eigenvalues (largest)
    idx = np.argsort(eigvals)[::-1][:n_dim]
    L = eigvals[idx]
    V = eigvecs[:, idx]
    # Threshold negative tiny eigenvalues from numerical noise
    L_safe = np.maximum(L, 0.0)
    return V * np.sqrt(L_safe)


def procrustes_rmsd(X: np.ndarray, Y: np.ndarray) -> float:
    """RMSD between X and Y after optimal rigid alignment (no reflection)."""
    Xc = X - X.mean(axis=0)
    Yc = Y - Y.mean(axis=0)
    H = Xc.T @ Yc
    U, _, Vt = np.linalg.svd(H)
    # Allow reflection (Kabsch with reflection avoidance optional)
    R = Vt.T @ U.T
    Y_rot = Yc @ R
    diff = Xc - Y_rot
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=-1))))


def evaluate(pdb_id: str, path: Path) -> dict:
    X = load_pdb_ca_coords(path)
    if X is None or len(X) < 4:
        return {"pdb": pdb_id, "skipped": True, "reason": "no CA atoms"}
    N = len(X)
    D = pairwise_dist(X)
    Y = mds_from_distance(D, n_dim=3)
    rmsd = procrustes_rmsd(X, Y)
    avg_dist = float(D[np.triu_indices(N, k=1)].mean())
    return {
        "pdb": pdb_id,
        "n_residues": N,
        "rmsd_A": rmsd,
        "rmsd_per_avg_dist": rmsd / avg_dist if avg_dist > 0 else None,
        "avg_pairwise_dist_A": avg_dist,
    }


def main() -> int:
    files = sorted(PDB_DIR.glob("*.pdb"))
    print(f"Found {len(files)} PDB files in {PDB_DIR}\n")
    results: List[dict] = []
    for fp in files:
        pdb_id = fp.stem.upper()
        try:
            r = evaluate(pdb_id, fp)
            results.append(r)
        except Exception as e:
            results.append({"pdb": pdb_id, "skipped": True, "reason": str(e)})

    print(f"{'PDB':<8} {'N_res':>6} {'avg_d (Å)':>12} {'RMSD (Å)':>14} {'RMSD/d':>14}")
    print("-" * 60)
    for r in results:
        if r.get("skipped"):
            print(f"{r['pdb']:<8} SKIPPED ({r.get('reason', '')})")
            continue
        print(f"{r['pdb']:<8} {r['n_residues']:>6} "
              f"{r['avg_pairwise_dist_A']:>12.2f} "
              f"{r['rmsd_A']:>14.2e} "
              f"{r['rmsd_per_avg_dist']:>14.2e}")

    rmsds = [r["rmsd_A"] for r in results if not r.get("skipped")]
    if rmsds:
        print()
        print(f"Median RMSD: {np.median(rmsds):.2e} Å")
        print(f"Max RMSD:    {np.max(rmsds):.2e} Å")
        print()
        if np.max(rmsds) < 1e-6:
            print("VERDICT: classical MDS recovers 3D coordinates from distance matrix")
            print("         to numerical precision. The '1D → 3D via eigendecomposition'")
            print("         claim is mathematically TRIVIAL (Young-Householder 1938) and")
            print("         proves nothing about protein folding from sequence alone.")
            print("         The actual folding claim must be tested as a blind")
            print("         CASP15 sequence→structure evaluation.")

    out_file = OUT_DIR / "pdb_mds_results.json"
    with open(out_file, "w") as f:
        json.dump({"results": results}, f, indent=2, default=str)
    print(f"\nWrote {out_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
