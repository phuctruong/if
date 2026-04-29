"""Tests for MW v(10 kpc) σ-accounting.

Run with: pytest tests/test_mw_rotation_sigma.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_v_prime_from_zero_param_mode_is_138():
    """Bare prime-field at 10 kpc reproduces the audit's 138.28 km/s figure."""
    import logging
    logging.basicConfig(level=logging.WARNING)
    from prime_field_theory import PrimeFieldTheory
    pft = PrimeFieldTheory(use_mersenne_tower=True)
    v = float(pft.velocity_at_10kpc())
    assert 137.0 < v < 140.0, f"v_prime(10 kpc) = {v}, expected ≈ 138.28"


def test_v_total_quadrature():
    """v_total = √(v_prime² + v_baryon²) ≈ 211.3 km/s with v_p ≈ 138, v_b = 160."""
    from predictions.mw_rotation_sigma_accounting import predict_mw_v10
    pred = predict_mw_v10()
    assert 210.0 < pred.v_total_kms < 213.0, \
        f"v_total = {pred.v_total_kms}, expected ≈ 211.3"


def test_combined_uncertainty_is_in_30_to_40_kms():
    """σ(Δ) = √(σ_total² + σ_obs²) should be 30-40 km/s."""
    from predictions.mw_rotation_sigma_accounting import predict_mw_v10
    pred = predict_mw_v10()
    assert 30.0 < pred.sigma_combined_kms < 40.0, \
        f"σ_combined = {pred.sigma_combined_kms}, expected 30-40"


def test_mw_consistent_within_1sigma():
    """The headline test: MW v(10 kpc) must be consistent with IF Theory + Sofue
    baryons within 1σ when uncertainties are properly propagated.

    This is the σ-accounting fix to the 2026-04-29 audit's claim of '4σ short'.
    """
    from predictions.mw_rotation_sigma_accounting import predict_mw_v10
    pred = predict_mw_v10()
    assert pred.deviation_sigma < 1.0, \
        f"MW deviation = {pred.deviation_sigma:.2f}σ, expected < 1σ"


def test_required_v0_within_30pct_uncertainty():
    """The v_0 needed to make v_total = 220 exactly should be within ±30%
    of the theoretical v_0 ≈ 397 km/s."""
    from predictions.mw_rotation_sigma_accounting import required_v0_for_match
    v0_req = required_v0_for_match()
    v0_default = 397.0
    deviation_pct = abs(v0_req / v0_default - 1)
    assert deviation_pct < 0.30, \
        f"v_0 deviation = {deviation_pct * 100:.1f}%, expected < 30%"
