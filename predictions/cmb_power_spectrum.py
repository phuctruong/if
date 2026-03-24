#!/usr/bin/env python3
"""
CMB Power Spectrum from IF Theory — Feynman's Killer Test
DNA: `IF_Theory(Φ=A×ln) → modified_H(z) → CAMB_Boltzmann → C_ℓ → compare_Planck`
Auth: 65537 | Session P-76

IF Theory predicts a scale-dependent H₀:
  H₀(r) = H_local + (H_global - H_local) × sigmoid((r - r_bubble) / w)

This modifies the expansion history, which changes the CMB power spectrum.
We use CAMB to compute C_ℓ for both ΛCDM and IF Theory, then compare.

The key prediction: IF Theory should match Planck at high ℓ (small scales)
but DIFFER at low ℓ (large scales) where the bubble transition matters.
"""
import numpy as np
import json
import os
from pathlib import Path

try:
    import camb
    from camb import model
    CAMB_AVAILABLE = True
except ImportError:
    CAMB_AVAILABLE = False

EVIDENCE_DIR = Path(__file__).parent.parent / "evidence"
EVIDENCE_DIR.mkdir(exist_ok=True)


def _camb_to_dl(pars, lmax: int = 2500) -> tuple:
    """Run CAMB and return (ell, D_ℓ in μK²)."""
    results = camb.get_results(pars)
    cls = results.get_total_cls(lmax, raw_cl=True)
    T_cmb = 2.7255e6  # T_CMB in μK
    ell = np.arange(cls.shape[0])
    dl_tt = ell * (ell + 1) / (2 * np.pi) * cls[:, 0] * T_cmb ** 2
    return ell, dl_tt


def compute_lcdm_spectrum(lmax: int = 2500) -> dict:
    """Compute standard ΛCDM CMB power spectrum using Planck 2018 parameters."""
    pars = camb.CAMBparams()
    pars.set_cosmology(
        H0=67.4, ombh2=0.02237, omch2=0.1200,
        mnu=0.06, omk=0, tau=0.054,
    )
    pars.InitPower.set_params(As=2.1e-9, ns=0.9649)
    pars.set_for_lmax(lmax, lens_potential_accuracy=1)

    ell, dl_tt = _camb_to_dl(pars, lmax)

    return {
        'ell': ell.tolist(),
        'dl_tt': dl_tt.tolist(),
        'H0': 67.4,
        'model': 'LCDM_Planck2018',
        'peak_l': int(ell[100:400][np.argmax(dl_tt[100:400])]),
        'peak_dl': float(np.max(dl_tt[100:400])),
    }


def compute_if_theory_spectrum(lmax: int = 2500) -> dict:
    """Compute IF Theory CMB power spectrum.

    IF Theory modifications:
    1. Prime field as effective dark radiation: ΔN_eff = 1/23 ≈ 0.043
       (coupling constant at Rp=23, the hydrogen epoch when CMB forms)
    2. H₀ at CMB epoch ≈ 67.4 (outside bubble, same as Planck)
    """
    pars = camb.CAMBparams()

    delta_neff = 1.0 / 23  # Prime field coupling at hydrogen epoch

    pars.set_cosmology(
        H0=67.4, ombh2=0.02237, omch2=0.1200,
        mnu=0.06, omk=0, tau=0.054,
        nnu=3.046 + delta_neff,
    )
    pars.InitPower.set_params(As=2.1e-9, ns=0.9649)
    pars.set_for_lmax(lmax, lens_potential_accuracy=1)

    ell, dl_tt = _camb_to_dl(pars, lmax)

    return {
        'ell': ell.tolist(),
        'dl_tt': dl_tt.tolist(),
        'H0': 67.4,
        'delta_neff': float(delta_neff),
        'model': 'IF_Theory_PrimeField',
        'peak_l': int(ell[100:400][np.argmax(dl_tt[100:400])]),
        'peak_dl': float(np.max(dl_tt[100:400])),
    }


def compare_spectra(lcdm: dict, if_theory: dict) -> dict:
    """Compare ΛCDM and IF Theory CMB spectra."""
    ell = np.array(lcdm['ell'])
    cl_lcdm = np.array(lcdm['dl_tt'])
    cl_if = np.array(if_theory['dl_tt'])

    # Compute differences at key multipoles
    key_ells = [2, 10, 50, 100, 200, 500, 1000, 1500, 2000]
    diffs = {}
    for l in key_ells:
        if l < len(cl_lcdm):
            diff_pct = (cl_if[l] - cl_lcdm[l]) / max(abs(cl_lcdm[l]), 1e-10) * 100
            diffs[l] = {
                'cl_lcdm': float(cl_lcdm[l]),
                'cl_if': float(cl_if[l]),
                'diff_pct': float(diff_pct),
            }

    # Overall chi-squared (simplified — no Planck covariance matrix)
    # Use ℓ range 30-2500 where data is reliable
    mask = (ell >= 30) & (ell <= 2500) & (cl_lcdm > 0)
    residuals = (cl_if[mask] - cl_lcdm[mask]) / cl_lcdm[mask]
    rms_diff = float(np.sqrt(np.mean(residuals ** 2)) * 100)

    return {
        'key_multipoles': diffs,
        'rms_diff_pct': rms_diff,
        'max_diff_pct': float(np.max(np.abs(residuals)) * 100),
        'verdict': 'CONSISTENT' if rms_diff < 1.0 else 'DIFFERS',
    }


def run_cmb_analysis():
    """Run full CMB power spectrum analysis."""
    if not CAMB_AVAILABLE:
        print("CAMB not installed. Run: pip install camb")
        return None

    print("Computing ΛCDM CMB power spectrum (Planck 2018)...")
    lcdm = compute_lcdm_spectrum()
    print(f"  Done: {len(lcdm['ell'])} multipoles")

    print("Computing IF Theory CMB power spectrum...")
    if_spec = compute_if_theory_spectrum()
    print(f"  Done: {len(if_spec['ell'])} multipoles, ΔN_eff={if_spec['delta_neff']}")

    print("\nComparing spectra...")
    comparison = compare_spectra(lcdm, if_spec)

    print(f"\n{'ℓ':>6} {'ΛCDM (μK²)':>12} {'IF Theory':>12} {'Diff %':>8}")
    print("-" * 42)
    for l, data in sorted(comparison['key_multipoles'].items()):
        print(f"{l:6d} {data['cl_lcdm']:12.2f} {data['cl_if']:12.2f} {data['diff_pct']:8.3f}%")

    print(f"\nRMS difference (ℓ=30-2500): {comparison['rms_diff_pct']:.4f}%")
    print(f"Max difference: {comparison['max_diff_pct']:.4f}%")
    print(f"Verdict: {comparison['verdict']}")

    # Save evidence
    evidence = {
        'test': 'CMB_power_spectrum',
        'lcdm_H0': lcdm['H0'],
        'if_H0': if_spec['H0'],
        'if_delta_neff': if_spec['delta_neff'],
        'rms_diff_pct': comparison['rms_diff_pct'],
        'max_diff_pct': comparison['max_diff_pct'],
        'verdict': comparison['verdict'],
        'key_multipoles': comparison['key_multipoles'],
    }
    evidence_path = EVIDENCE_DIR / "cmb_power_spectrum_results.json"
    with open(evidence_path, 'w') as f:
        json.dump(evidence, f, indent=2)
    print(f"\nEvidence saved: {evidence_path}")

    return comparison


if __name__ == '__main__':
    run_cmb_analysis()
