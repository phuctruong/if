#!/usr/bin/env python3
"""
Protein Folding Benchmark — Compare IF Theory Closure vs Known PDB Structures
DNA: `benchmark = download_PDB(known) → fold_closure(sequence) → compare(Rg, contacts, burial)`
Auth: 65537 | Session P-76

Downloads real protein structures from RCSB PDB, extracts Cα coordinates,
measures Rg and contacts, then compares with our closure-based folding.

Test proteins chosen to span difficulty levels:
  - Trp-cage (20 residues) — smallest stable protein
  - Villin HP35 (35 residues) — fast folder
  - Insulin B chain (30 residues) — well-characterized
  - Ubiquitin (76 residues) — medium, well-studied
  - Lysozyme (129 residues) — medium-large, classic
  - Myoglobin (153 residues) — large, all-alpha
"""
import os
import json
import numpy as np
from pathlib import Path
from typing import Optional
from io import StringIO

try:
    from Bio.PDB import PDBParser, PDBList
    from Bio.PDB.Polypeptide import is_aa, protein_letters_3to1
    def three_to_one(resname):
        return protein_letters_3to1.get(resname, 'X')
    BIOPYTHON = True
except ImportError:
    BIOPYTHON = False

import sys
sys.path.insert(0, os.path.dirname(__file__))
from protein_closure import fold_protein, HYDROPHOBICITY

BENCHMARK_DIR = Path(__file__).parent.parent / "benchmark"
BENCHMARK_DIR.mkdir(exist_ok=True)
PDB_DIR = BENCHMARK_DIR / "pdb"
PDB_DIR.mkdir(exist_ok=True)
EVIDENCE_DIR = Path(__file__).parent.parent / "evidence"
EVIDENCE_DIR.mkdir(exist_ok=True)

# Benchmark proteins with known Rg values
BENCHMARK_PROTEINS = {
    '1L2Y': {
        'name': 'Trp-cage',
        'chain': 'A',
        'known_rg': 7.1,
        'n_residues': 20,
        'fold_type': 'miniprotein',
    },
    '1VII': {
        'name': 'Villin HP35',
        'chain': 'A',
        'known_rg': 9.8,
        'n_residues': 35,
        'fold_type': 'helical',
    },
    '2HIU': {
        'name': 'Insulin B',
        'chain': 'B',
        'known_rg': 9.5,
        'n_residues': 30,
        'fold_type': 'mixed',
    },
    '1UBQ': {
        'name': 'Ubiquitin',
        'chain': 'A',
        'known_rg': 11.8,
        'n_residues': 76,
        'fold_type': 'mixed',
    },
    '2LZM': {
        'name': 'Lysozyme',
        'chain': 'A',
        'known_rg': 15.0,
        'n_residues': 129,
        'fold_type': 'mixed',
    },
    '1MBN': {
        'name': 'Myoglobin',
        'chain': 'A',
        'known_rg': 15.7,
        'n_residues': 153,
        'fold_type': 'all-alpha',
    },
}


def download_pdb(pdb_id: str) -> Optional[str]:
    """Download PDB file from RCSB."""
    if not BIOPYTHON:
        print("  BioPython not installed")
        return None

    pdb_path = PDB_DIR / f"{pdb_id.lower()}.pdb"
    if pdb_path.exists():
        return str(pdb_path)

    try:
        pdbl = PDBList()
        filename = pdbl.retrieve_pdb_file(pdb_id, pdir=str(PDB_DIR), file_format='pdb')
        if filename and os.path.exists(filename):
            # Rename to simple name
            os.rename(filename, str(pdb_path))
            return str(pdb_path)
    except Exception as e:
        print(f"  Download failed: {e}")

    return None


def extract_sequence_and_coords(pdb_path: str, chain_id: str = 'A') -> tuple:
    """Extract sequence and Cα coordinates from PDB file.

    Returns (sequence_string, ca_coords_array).
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_path)

    sequence = []
    coords = []

    for model in structure:
        for chain in model:
            if chain.id != chain_id:
                continue
            for residue in chain:
                if not is_aa(residue):
                    continue
                try:
                    resname = residue.get_resname()
                    one_letter = three_to_one(resname)
                    ca = residue['CA']
                    sequence.append(one_letter)
                    coords.append(ca.get_vector().get_array())
                except (KeyError, ValueError):
                    continue
        break  # First model only

    if not sequence:
        return '', np.array([])

    return ''.join(sequence), np.array(coords)


def compute_pdb_metrics(coords: np.ndarray, sequence: str) -> dict:
    """Compute metrics for a known PDB structure."""
    centroid = np.mean(coords, axis=0)
    rg = np.sqrt(np.mean(np.sum((coords - centroid) ** 2, axis=1)))

    # Contact map (Cα < 8 Å)
    n = len(coords)
    n_contacts = 0
    for i in range(n):
        for j in range(i + 4, n):
            dist = np.linalg.norm(coords[i] - coords[j])
            if dist < 8.0:
                n_contacts += 1

    # Hydrophobic burial
    hydro_dists = []
    philic_dists = []
    for k in range(len(sequence)):
        if k >= len(coords):
            break
        d = np.linalg.norm(coords[k] - centroid)
        h = HYDROPHOBICITY.get(sequence[k], 0)
        if h > 0:
            hydro_dists.append(d)
        elif h < -1:
            philic_dists.append(d)

    burial = 0.0
    if hydro_dists and philic_dists:
        burial = 1.0 - np.mean(hydro_dists) / max(np.mean(philic_dists), 1e-10)

    return {
        'rg': float(rg),
        'n_contacts': n_contacts,
        'hydrophobic_burial': float(burial),
        'n_residues': len(sequence),
    }


def run_benchmark():
    """Run full benchmark: download PDB → extract → fold → compare."""
    print("=" * 80)
    print("  PROTEIN FOLDING BENCHMARK — IF Theory Closure vs PDB Known Structures")
    print("  Auth: 65537 | DNA: closure(seam) beats force_field(gradient)")
    print("=" * 80)

    results = []

    for pdb_id, info in BENCHMARK_PROTEINS.items():
        print(f"\n--- {info['name']} ({pdb_id}, {info['n_residues']} residues, {info['fold_type']}) ---")

        # Download PDB
        pdb_path = download_pdb(pdb_id)
        if not pdb_path:
            print("  Skipped (download failed)")
            continue

        # Extract sequence and known coordinates
        sequence, known_coords = extract_sequence_and_coords(pdb_path, info['chain'])
        if len(sequence) < 5:
            print(f"  Skipped (sequence too short: {len(sequence)})")
            continue

        print(f"  Sequence ({len(sequence)} res): {sequence[:40]}...")

        # Compute PDB metrics
        pdb_metrics = compute_pdb_metrics(known_coords, sequence)
        print(f"  PDB:     Rg={pdb_metrics['rg']:.1f} Å, {pdb_metrics['n_contacts']} contacts, burial={pdb_metrics['hydrophobic_burial']:.2f}")

        # Fold with IF Theory closure
        fold_result = fold_protein(sequence)
        print(f"  Closure: Rg={fold_result['rg']:.1f} Å, satisfaction={fold_result['contact_satisfaction']:.0%}, burial={fold_result['hydrophobic_burial']:.2f}")

        # Compare
        rg_error = abs(fold_result['rg'] - pdb_metrics['rg']) / pdb_metrics['rg'] * 100
        burial_diff = fold_result['hydrophobic_burial'] - pdb_metrics['hydrophobic_burial']
        print(f"  Error:   Rg {rg_error:.0f}%, burial diff {burial_diff:+.2f}")

        results.append({
            'pdb_id': pdb_id,
            'name': info['name'],
            'n_residues': len(sequence),
            'fold_type': info['fold_type'],
            'pdb_rg': pdb_metrics['rg'],
            'closure_rg': fold_result['rg'],
            'rg_error_pct': rg_error,
            'pdb_burial': pdb_metrics['hydrophobic_burial'],
            'closure_burial': fold_result['hydrophobic_burial'],
            'contact_satisfaction': fold_result['contact_satisfaction'],
        })

    # Summary
    if results:
        print(f"\n{'=' * 80}")
        print(f"  SUMMARY")
        print(f"{'=' * 80}")
        print(f"\n{'Protein':>15} {'N':>4} {'PDB Rg':>8} {'Ours':>8} {'Error':>7} {'Burial':>7}")
        print("-" * 55)
        for r in results:
            marker = "✅" if r['rg_error_pct'] < 30 else "⚠️" if r['rg_error_pct'] < 60 else "❌"
            print(f"{r['name']:>15} {r['n_residues']:4d} {r['pdb_rg']:8.1f} {r['closure_rg']:8.1f} "
                  f"{r['rg_error_pct']:6.0f}% {r['closure_burial']:7.2f} {marker}")

        avg_error = np.mean([r['rg_error_pct'] for r in results])
        print(f"\nAverage Rg error: {avg_error:.0f}%")
        print(f"AlphaFold2 average: <1 Å RMSD (essentially perfect)")
        print(f"Our approach: geometric closure (zero training, runs on CPU)")

        # Save evidence
        evidence_path = EVIDENCE_DIR / "protein_benchmark_results.json"
        with open(evidence_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nEvidence saved: {evidence_path}")

    return results


if __name__ == '__main__':
    run_benchmark()
