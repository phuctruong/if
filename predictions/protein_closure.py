#!/usr/bin/env python3
"""
Protein Folding as Topological Closure — IF Theory Approach
DNA: `fold = find_seam(hydrophobic_boundary) → close(along_seam) → lock(minimize_σ)`
Auth: 65537 | Session P-76

The key insight from the Geometric Big Bang:
  Protein folding is NOT a force field problem (Rg 32 vs 9.5 Å FAILS).
  Protein folding IS a closure operation (like galaxy formation at Rp=3).

The closure algorithm:
  1. Compute contact propensity matrix C(i,j) from sequence
  2. Find the closure seam: boundary between hydrophobic interior and hydrophilic exterior
  3. Fold along the seam: topological operation, not continuous force
  4. Lock the fold: minimize seam mismatch σ

This is exactly how Rp=3 works: 3 nodes form a triangle (first closure).
Protein folding is Rp=71: 71 residues form a closure (3D fold).
"""
import numpy as np
from typing import List, Tuple

# Kyte-Doolittle hydrophobicity scale
HYDROPHOBICITY = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2,
}


def compute_contact_propensity(sequence: str) -> np.ndarray:
    """Compute contact propensity matrix C(i,j).

    C(i,j) = hydrophobicity(i) × hydrophobicity(j) / |i-j|
    Positive C(i,j) = both hydrophobic + distant in sequence = want to be close in 3D.
    """
    n = len(sequence)
    C = np.zeros((n, n))
    for i in range(n):
        hi = HYDROPHOBICITY.get(sequence[i], 0)
        for j in range(i + 2, n):  # Skip neighbors
            hj = HYDROPHOBICITY.get(sequence[j], 0)
            seq_dist = abs(j - i)
            if hi > 0 and hj > 0:  # Both hydrophobic
                C[i, j] = hi * hj / seq_dist
                C[j, i] = C[i, j]
    return C


def find_closure_seam(C: np.ndarray, sequence: str) -> List[Tuple[int, int]]:
    """Find the closure seam: pairs of residues that should be in contact.

    The seam is the boundary between hydrophobic interior and hydrophilic exterior.
    We find the top-k contact pairs that define this boundary.

    v2: Limit contacts to ~2 per residue (proteins average 2-3 long-range contacts).
    This prevents over-compression seen in v1.
    """
    n = len(sequence)
    # Rank all pairs by contact propensity
    pairs = []
    for i in range(n):
        for j in range(i + 4, n):  # Minimum 4 residues apart for real contact
            if C[i, j] > 0:
                pairs.append((C[i, j], i, j))

    # Sort by propensity (strongest contacts first)
    pairs.sort(reverse=True)

    # v2: Limit to ~2 contacts per residue (prevents over-compression)
    contact_count = {}
    max_per_residue = 2
    seam = []
    for _, i, j in pairs:
        ci = contact_count.get(i, 0)
        cj = contact_count.get(j, 0)
        if ci < max_per_residue and cj < max_per_residue:
            seam.append((i, j))
            contact_count[i] = ci + 1
            contact_count[j] = cj + 1

    return seam


def fold_along_seam(sequence: str, seam: List[Tuple[int, int]],
                    initial_coords: np.ndarray = None) -> np.ndarray:
    """Fold the protein along the closure seam.

    This is a TOPOLOGICAL operation: move residues toward their contact
    partners. NOT a continuous force — a discrete closure step.

    Returns 3D coordinates (n, 3).
    """
    n = len(sequence)

    # Start with extended chain
    if initial_coords is None:
        coords = np.zeros((n, 3))
        for i in range(n):
            coords[i] = [i * 3.8 * np.cos(i * 0.3),  # Slight helix
                         i * 3.8 * np.sin(i * 0.3),
                         i * 1.5]  # Extended along z

    else:
        coords = initial_coords.copy()

    # NOTE: Secondary structure integration (Phase 0) tested in v3/v3b.
    # Chou-Fasman over-predicts helix → over-compression. Reverted to v2.
    # Next: better SS predictor (PSIPRED/ESM) or learned local constraints.

    # Phase 1: Global collapse — pull ALL hydrophobic residues toward centroid
    for iteration in range(100):
        centroid = np.mean(coords, axis=0)
        for k in range(n):
            h = HYDROPHOBICITY.get(sequence[k], 0)
            if h > 0:  # Hydrophobic → pull inward
                delta = centroid - coords[k]
                dist = np.linalg.norm(delta)
                if dist > 3.0:
                    coords[k] += 0.15 * delta
            elif h < -1:  # Hydrophilic → push outward (slight)
                delta = coords[k] - centroid
                dist = np.linalg.norm(delta)
                if dist < 5.0:
                    coords[k] += 0.02 * delta

        # Keep chain connectivity (bond length ~3.8 Å)
        for k in range(n - 1):
            delta = coords[k + 1] - coords[k]
            dist = np.linalg.norm(delta)
            if dist != 0:
                direction = delta / dist
                correction = 0.3 * (dist - 3.8) * direction
                coords[k] += correction
                coords[k + 1] -= correction

        coords -= np.mean(coords, axis=0)

    # Phase 2: Contact closure — pull seam pairs together
    for iteration in range(200):
        forces = np.zeros_like(coords)

        for i, j in seam:
            delta = coords[j] - coords[i]
            dist = np.linalg.norm(delta)
            if dist > 4.0:
                direction = delta / max(dist, 1e-10)
                pull = 0.5 * (dist - 4.0) * direction
                forces[i] += pull
                forces[j] -= pull

        for k in range(n - 1):
            delta = coords[k + 1] - coords[k]
            dist = np.linalg.norm(delta)
            if dist != 0:
                direction = delta / dist
                correction = 0.3 * (dist - 3.8) * direction
                forces[k] += correction
                forces[k + 1] -= correction

        coords += forces * 0.08
        coords -= np.mean(coords, axis=0)

    return coords


def measure_fold_quality(coords: np.ndarray, sequence: str,
                         seam: List[Tuple[int, int]]) -> dict:
    """Measure fold quality metrics."""
    n = len(sequence)

    # Radius of gyration
    centroid = np.mean(coords, axis=0)
    rg = np.sqrt(np.mean(np.sum((coords - centroid) ** 2, axis=1)))

    # Contact satisfaction (what fraction of seam contacts are < 8 Å)
    satisfied = 0
    for i, j in seam:
        dist = np.linalg.norm(coords[i] - coords[j])
        if dist < 8.0:
            satisfied += 1
    contact_fraction = satisfied / max(len(seam), 1)

    # Hydrophobic burial (are hydrophobic residues inside?)
    hydro_dists = []
    philic_dists = []
    for k in range(n):
        d = np.linalg.norm(coords[k] - centroid)
        h = HYDROPHOBICITY.get(sequence[k], 0)
        if h > 0:
            hydro_dists.append(d)
        elif h < -1:
            philic_dists.append(d)

    hydro_burial = 0.0
    if hydro_dists and philic_dists:
        mean_hydro = np.mean(hydro_dists)
        mean_philic = np.mean(philic_dists)
        if mean_philic > 0:
            hydro_burial = 1.0 - mean_hydro / mean_philic  # >0 = hydrophobic inside

    return {
        'rg': float(rg),
        'n_residues': n,
        'n_contacts': len(seam),
        'contact_satisfaction': float(contact_fraction),
        'hydrophobic_burial': float(hydro_burial),
    }


def fold_protein(sequence: str) -> dict:
    """Full protein folding pipeline using closure approach."""
    C = compute_contact_propensity(sequence)
    seam = find_closure_seam(C, sequence)
    coords = fold_along_seam(sequence, seam)
    quality = measure_fold_quality(coords, sequence, seam)
    return {**quality, 'coords': coords}


if __name__ == '__main__':
    # Test on three proteins
    proteins = {
        'Insulin B': {
            'sequence': 'FVNQHLCGSHLVEALYLVCGERGFFYTPKT',
            'known_rg': 9.5,  # Å
        },
        'Villin HP35': {
            'sequence': 'LSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF',
            'known_rg': 9.8,
        },
        'Trp-cage': {
            'sequence': 'NLYIQWLKDGGPSSGRPPPS',
            'known_rg': 7.1,
        },
    }

    print(f"{'Protein':>15} {'Rg (Å)':>8} {'Known':>8} {'Error':>8} {'Contacts':>10} {'Burial':>8}")
    print("-" * 65)

    for name, data in proteins.items():
        result = fold_protein(data['sequence'])
        error_pct = abs(result['rg'] - data['known_rg']) / data['known_rg'] * 100
        print(f"{name:>15} {result['rg']:8.1f} {data['known_rg']:8.1f} {error_pct:7.0f}% "
              f"{result['contact_satisfaction']:10.0%} {result['hydrophobic_burial']:8.2f}")

    print("\nClosure approach vs v1 force field:")
    print("  v1: Rg 32 Å (232% error) — force field doesn't collapse")
    print("  v2: See above — closure pulls contacts together")
