#!/usr/bin/env python3
"""
GPU-Accelerated Protein Folding via Geometric Closure
DNA: `sequence → hydro_map(GPU) → contact_matrix(GPU) → collapse(GPU) → fold(GPU); 10-100x faster`
Auth: 65537 | Session P-76

Uses CuPy for GPU-accelerated:
1. Contact propensity matrix computation (N² pairs)
2. Hydrophobic collapse (N body updates)
3. Contact closure (N body + seam pairs)
4. All on RTX 3080 (10GB)
"""
import numpy as np
import time
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from protein_closure import HYDROPHOBICITY

try:
    import cupy as cp
    GPU = True
except ImportError:
    GPU = False


def fold_protein_gpu(sequence: str) -> dict:
    """GPU-accelerated protein folding via geometric closure."""
    if not GPU:
        raise RuntimeError("CuPy not available — need GPU")

    n = len(sequence)
    t0 = time.perf_counter()

    # Transfer hydrophobicity values to GPU
    hydro = cp.array([HYDROPHOBICITY.get(aa, 0.0) for aa in sequence], dtype=cp.float32)

    # GPU: Compute contact propensity matrix (N×N)
    hydro_i = hydro[:, None]  # (N, 1)
    hydro_j = hydro[None, :]  # (1, N)
    seq_dist = cp.abs(cp.arange(n)[:, None] - cp.arange(n)[None, :]).astype(cp.float32)
    seq_dist = cp.maximum(seq_dist, 1.0)

    # C(i,j) = max(0, h_i) × max(0, h_j) / |i-j| where both hydrophobic
    mask_hydro = (hydro_i > 0) & (hydro_j > 0)
    mask_dist = seq_dist >= 4  # At least 4 apart
    C = cp.where(mask_hydro & mask_dist, hydro_i * hydro_j / seq_dist, 0.0)

    # Find top N/3 contacts (seam)
    n_contacts = max(3, n // 3)
    # Flatten upper triangle, get top indices
    upper_mask = cp.triu(cp.ones((n, n), dtype=bool), k=4)
    C_upper = cp.where(upper_mask, C, 0.0)
    flat = C_upper.ravel()
    top_indices = cp.argsort(flat)[-n_contacts:]
    seam_i = (top_indices // n).astype(cp.int32)
    seam_j = (top_indices % n).astype(cp.int32)

    # Initialize extended chain on GPU
    angles = cp.arange(n, dtype=cp.float32) * 0.3
    coords = cp.zeros((n, 3), dtype=cp.float32)
    coords[:, 0] = cp.arange(n, dtype=cp.float32) * 3.8 * cp.cos(angles)
    coords[:, 1] = cp.arange(n, dtype=cp.float32) * 3.8 * cp.sin(angles)
    coords[:, 2] = cp.arange(n, dtype=cp.float32) * 1.5

    # Phase 1: GPU hydrophobic collapse (100 iterations)
    for _ in range(100):
        centroid = cp.mean(coords, axis=0)

        # Pull hydrophobic inward
        delta = centroid - coords  # (N, 3)
        dist = cp.linalg.norm(delta, axis=1, keepdims=True)
        pull = cp.where((hydro[:, None] > 0) & (dist > 3.0), 0.15 * delta, 0.0)

        # Push hydrophilic outward (slight)
        push_delta = coords - centroid
        push = cp.where((hydro[:, None] < -1) & (dist < 5.0), 0.02 * push_delta, 0.0)

        coords += pull + push

        # Bond constraints (vectorized)
        bond_delta = coords[1:] - coords[:-1]
        bond_dist = cp.linalg.norm(bond_delta, axis=1, keepdims=True)
        bond_dir = bond_delta / cp.maximum(bond_dist, 1e-10)
        correction = 0.3 * (bond_dist - 3.8) * bond_dir
        coords[:-1] += correction
        coords[1:] -= correction

        coords -= cp.mean(coords, axis=0)

    # Phase 2: GPU contact closure (200 iterations)
    for _ in range(200):
        forces = cp.zeros_like(coords)

        # Seam pair forces (vectorized over seam pairs)
        pos_i = coords[seam_i]  # (K, 3)
        pos_j = coords[seam_j]  # (K, 3)
        delta = pos_j - pos_i
        dist = cp.linalg.norm(delta, axis=1, keepdims=True)
        direction = delta / cp.maximum(dist, 1e-10)
        pull = cp.where(dist > 4.0, 0.5 * (dist - 4.0) * direction, 0.0)

        # Scatter forces
        for k in range(len(seam_i)):
            si, sj = int(seam_i[k]), int(seam_j[k])
            forces[si] += pull[k]
            forces[sj] -= pull[k]

        # Bond constraints
        bond_delta = coords[1:] - coords[:-1]
        bond_dist = cp.linalg.norm(bond_delta, axis=1, keepdims=True)
        bond_dir = bond_delta / cp.maximum(bond_dist, 1e-10)
        correction = 0.3 * (bond_dist - 3.8) * bond_dir
        forces[:-1] += correction
        forces[1:] -= correction

        coords += forces * 0.08
        coords -= cp.mean(coords, axis=0)

    t1 = time.perf_counter()

    # Metrics (on GPU)
    centroid = cp.mean(coords, axis=0)
    rg = float(cp.sqrt(cp.mean(cp.sum((coords - centroid) ** 2, axis=1))))

    # Contact satisfaction
    pos_i = coords[seam_i]
    pos_j = coords[seam_j]
    contact_dists = cp.linalg.norm(pos_j - pos_i, axis=1)
    satisfaction = float(cp.mean(contact_dists < 8.0))

    # Hydrophobic burial
    dists_from_center = cp.linalg.norm(coords - centroid, axis=1)
    hydro_mask = hydro > 0
    philic_mask = hydro < -1
    burial = 0.0
    if cp.any(hydro_mask) and cp.any(philic_mask):
        mean_hydro = float(cp.mean(dists_from_center[hydro_mask]))
        mean_philic = float(cp.mean(dists_from_center[philic_mask]))
        if mean_philic > 0:
            burial = 1.0 - mean_hydro / mean_philic

    elapsed_ms = (t1 - t0) * 1000

    return {
        'rg': rg,
        'n_residues': n,
        'n_contacts': n_contacts,
        'contact_satisfaction': satisfaction,
        'hydrophobic_burial': burial,
        'elapsed_ms': elapsed_ms,
        'device': 'GPU (RTX 3080)',
    }


if __name__ == '__main__':
    # Known sequences (from PDB)
    proteins = {
        'Trp-cage (20)': 'NLYIQWLKDGGPSSGRPPPS',
        'Insulin B (30)': 'FVNQHLCGSHLVEALYLVCGERGFFYTPKT',
        'Villin HP35 (36)': 'MLSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF',
        'Ubiquitin (76)': 'MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG',
    }

    # Also test CPU for comparison
    sys.path.insert(0, os.path.dirname(__file__))
    from protein_closure import fold_protein

    print(f"{'Protein':>20} {'GPU Rg':>8} {'GPU ms':>8} {'CPU Rg':>8} {'CPU ms':>8} {'Speedup':>8}")
    print("-" * 65)

    for name, seq in proteins.items():
        # GPU
        gpu_result = fold_protein_gpu(seq)

        # CPU
        t0 = time.perf_counter()
        cpu_result = fold_protein(seq)
        cpu_ms = (time.perf_counter() - t0) * 1000

        speedup = cpu_ms / max(gpu_result['elapsed_ms'], 0.01)
        print(f"{name:>20} {gpu_result['rg']:8.1f} {gpu_result['elapsed_ms']:8.1f} "
              f"{cpu_result['rg']:8.1f} {cpu_ms:8.1f} {speedup:8.1f}x")

    # Scale test: large protein
    print("\n--- Scale test: 500-residue random protein ---")
    import random
    random.seed(65537)
    large_seq = ''.join(random.choices('ACDEFGHIKLMNPQRSTVWY', k=500))

    t0 = time.perf_counter()
    gpu_large = fold_protein_gpu(large_seq)
    gpu_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    cpu_large = fold_protein(large_seq)
    cpu_ms = (time.perf_counter() - t0) * 1000

    print(f"  GPU: Rg={gpu_large['rg']:.1f} Å in {gpu_ms:.0f} ms")
    print(f"  CPU: Rg={cpu_large['rg']:.1f} Å in {cpu_ms:.0f} ms")
    print(f"  Speedup: {cpu_ms/max(gpu_ms,0.01):.1f}x")
