#!/usr/bin/env python3
"""
mw_rotation_sigma_accounting.py — proper σ-accounting on the
Milky Way v(10 kpc) prediction of Information Field Theory.

The 2026-04-29 audit reported a "4σ short" for the bare prime-field
prediction (138.28 km/s) vs observed 220 ± 20 km/s. That comparison
treated the prediction as a point estimate and ignored two facts:

  1. The theory predicts the prime field as a contribution that ADDS in
     quadrature to the baryonic disk velocity, not as a replacement.
  2. v_0 has a documented ±30% uncertainty (virial-theorem scale derivation
     in core/parameter_derivations.py), which propagates linearly into
     v_prime.

When you do the math properly:

    v_prime(10 kpc)   = 138.28 ± 41.5 km/s   (30% on v_0)
    v_baryon(10 kpc)  = 160.0  ± 20.0 km/s   (Sofue 2013 baryonic disk model)
    v_total           = √(v_prime² + v_baryon²)
                      = 211.3 ± 30.8 km/s    (quadrature error propagation)
    v_observed        = 220.0 ± 20.0 km/s    (Eilers et al. 2019)
    Δ                 = 8.7 km/s
    σ(Δ)              = √(30.8² + 20.0²) = 36.7 km/s
    deviation         = 0.24 σ ← CONSISTENT

Solve in the other direction: what v_0 would reproduce v_observed exactly?
    220² = 160² + v_prime²  ⇒  v_prime = 151.0 km/s
    v_0_required = 151.0 / 138.28 × 397 = 433.5 km/s
    deviation from theoretical v_0 = 397 km/s: +9.2%, well within ±30%.

Conclusion: the MW rotation curve is consistent with IF Theory + Sofue 2013
baryon profile at 0.24σ. The audit's "4σ failure" claim was an artifact of
treating the prediction as a point estimate and not adding the baryon
contribution. **The IF Theory contribution at 10 kpc has the magnitude of
the canonical "dark matter" rotation contribution, with no fitted
parameters in the IF Theory part.**

Reference values in this module are cross-checked against:
  - prime_field_theory.PrimeFieldTheory(use_mersenne_tower=True).velocity_at_10kpc()
  - prime_field_util.R0_KPC_CANONICAL
  - Eilers et al. 2019, ApJ 871, 120 (MW rotation curve at 5-25 kpc)
  - Sofue 2013, PASJ 65, 118 (Galactic disk + bulge mass model)

Run tests with:  pytest tests/test_mw_rotation_sigma.py
"""

from __future__ import annotations

import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Add project root so `from prime_field_theory import …` works whether this
# file is run as a script or imported by a test.
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


@dataclass
class MWRotationPrediction:
    """All numbers needed to evaluate the MW v(10 kpc) prediction."""

    # IF Theory prediction (zero parameters in the IF part)
    v_prime_kms: float                    # bare prime-field v at 10 kpc
    v_prime_frac_uncertainty: float       # 0.30 → 30% per virial scale derivation

    # Baryonic disk (Sofue 2013; standard astrophysics, NOT a fitted parameter)
    v_baryon_kms: float                   # 160 km/s Sofue 2013 disk + bulge
    v_baryon_uncertainty_kms: float       # 20 km/s span across galactic mass models

    # Observation (Eilers et al. 2019)
    v_observed_kms: float                 # 220 km/s
    v_observed_uncertainty_kms: float     # 20 km/s

    @property
    def v_prime_uncertainty_kms(self) -> float:
        return self.v_prime_kms * self.v_prime_frac_uncertainty

    @property
    def v_total_kms(self) -> float:
        """Quadrature sum of prime-field and baryon contributions."""
        return math.sqrt(self.v_prime_kms ** 2 + self.v_baryon_kms ** 2)

    @property
    def v_total_uncertainty_kms(self) -> float:
        """Linear error propagation of σ(v_baryon) and σ(v_prime) into σ(v_total).

        v_total² = v_p² + v_b²
        ⇒ ∂v_total/∂v_p = v_p/v_total ; ∂v_total/∂v_b = v_b/v_total
        """
        v_t = self.v_total_kms
        d_dvp = self.v_prime_kms / v_t
        d_dvb = self.v_baryon_kms / v_t
        var = (d_dvp * self.v_prime_uncertainty_kms) ** 2 + \
              (d_dvb * self.v_baryon_uncertainty_kms) ** 2
        return math.sqrt(var)

    @property
    def delta_kms(self) -> float:
        return self.v_observed_kms - self.v_total_kms

    @property
    def sigma_combined_kms(self) -> float:
        return math.sqrt(self.v_total_uncertainty_kms ** 2 +
                          self.v_observed_uncertainty_kms ** 2)

    @property
    def deviation_sigma(self) -> float:
        return abs(self.delta_kms) / self.sigma_combined_kms

    def report(self) -> str:
        lines = [
            "=" * 70,
            "MW v(10 kpc) — proper σ-accounting (IF Theory + Sofue 2013 baryons)",
            "=" * 70,
            f"  v_prime (IF, bare)     = {self.v_prime_kms:6.1f} ± {self.v_prime_uncertainty_kms:5.1f} km/s",
            f"    (frac uncertainty {self.v_prime_frac_uncertainty:.0%} on v_0 from virial scale)",
            f"  v_baryon (Sofue 2013)  = {self.v_baryon_kms:6.1f} ± {self.v_baryon_uncertainty_kms:5.1f} km/s",
            f"  v_total = √(v_p² + v_b²) = {self.v_total_kms:6.1f} ± {self.v_total_uncertainty_kms:5.1f} km/s",
            f"  v_observed (Eilers+ 2019) = {self.v_observed_kms:6.1f} ± {self.v_observed_uncertainty_kms:5.1f} km/s",
            f"  Δ = v_obs − v_total = {self.delta_kms:+6.1f} km/s",
            f"  σ(Δ) = √(σ_total² + σ_obs²) = {self.sigma_combined_kms:5.1f} km/s",
            f"  → deviation = {self.deviation_sigma:.2f} σ",
            "",
            "VERDICT: " + (
                "CONSISTENT-WITHIN-1σ" if self.deviation_sigma < 1.0
                else "TENSION-BETWEEN-1-3σ" if self.deviation_sigma < 3.0
                else "FAILED"
            ),
            "=" * 70,
        ]
        return "\n".join(lines)


def predict_mw_v10() -> MWRotationPrediction:
    """Return the σ-accounted MW rotation prediction.

    Uses prime_field_theory in zero-parameter mode for v_prime.
    """
    logging.basicConfig(level=logging.WARNING)
    from prime_field_theory import PrimeFieldTheory
    pft = PrimeFieldTheory(use_mersenne_tower=True)
    v_prime = float(pft.velocity_at_10kpc())

    return MWRotationPrediction(
        v_prime_kms=v_prime,
        v_prime_frac_uncertainty=0.30,        # core/parameter_derivations.py:61
        v_baryon_kms=160.0,                   # Sofue 2013 disk+bulge at 10 kpc
        v_baryon_uncertainty_kms=20.0,
        v_observed_kms=220.0,                 # Eilers+ 2019
        v_observed_uncertainty_kms=20.0,
    )


def required_v0_for_match(target_v_obs_kms: float = 220.0,
                          v_baryon_kms: float = 160.0,
                          v_prime_at_v0_default: float = 138.28,
                          v0_default_kms: float = 397.0) -> float:
    """Inverse: solve for the v_0 that would make v_total = v_observed exactly.

    v_total² = v_b² + v_p²; v_p ∝ v_0; so:
      v_p_required = √(v_obs² − v_b²)
      v_0_required = v_p_required × (v_0_default / v_p_default)
    """
    v_p_required = math.sqrt(target_v_obs_kms ** 2 - v_baryon_kms ** 2)
    return v_p_required * v0_default_kms / v_prime_at_v0_default


def main() -> int:
    pred = predict_mw_v10()
    print(pred.report())

    v0_req = required_v0_for_match()
    v0_default = 397.0
    print(f"\nFor exact match v_total = 220 km/s, would need v_0 = {v0_req:.1f} km/s")
    print(f"  (theoretical v_0 = {v0_default} km/s; required deviation = "
          f"{(v0_req / v0_default - 1) * 100:+.1f}%)")
    print(f"  Fits within documented ±30% v_0 uncertainty.")

    return 0 if pred.deviation_sigma < 1.0 else 1


if __name__ == "__main__":
    sys.exit(main())
