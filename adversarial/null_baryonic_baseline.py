#!/usr/bin/env python3
"""
null_baryonic_baseline.py — adversarial test of whether the prime field
is doing real work.

If you turn off the prime field entirely (v_prime ≡ 0, ξ from baryons +
noise only), what does the residual to observation look like? If the
baryon-only null model is *almost* as good as IF Theory, then the
prime field is decorative — most of the explanatory work is being done
by the baryons (well-known physics) and the theory's distinctive claim
("dark matter is the prime field") is overstated.

This is the standard "null hypothesis" check: never trust a positive
result without comparing it to the trivial alternative.

Methodology (no external downloads):

  1. **MW v(10 kpc).** The IF Theory prediction is
        v_total = √(v_baryon² + v_prime²)
        with v_baryon = 160 km/s (Sofue 2013 disk+bulge),
        v_prime  = v₀ √(R/(R+r₀)) ≈ 138.28 km/s at R = 10 kpc.
     The observation is 220 ± 20 km/s (Eilers 2019).
     The null is v_total = v_baryon = 160 km/s. We compute the deviation
     in σ for each.

  2. **One canonical SPARC galaxy: NGC 3198.** This galaxy is the
     canonical flat-rotation-curve case; its baryon model and asymptotic
     V_flat are well-measured. We embed the published values:
        - V_flat (SPARC table) ≈ 150 km/s
        - V_baryon at R = R_d (one disk scale length, where the curve
          first flattens) ≈ 105 km/s [median Lelli 2016]
        - V_obs at R = R_d ≈ 150 km/s (the curve is flat past R_d)
        - σ_V ≈ 8 km/s typical SPARC error.
     IF: v_total = √(V_baryon² + V_flat²·R/(R+r₀)) at R = R_d
     Null: v_total = V_baryon.
     We compute (V_obs - v_total)/σ_V for both.

  3. **Aggregate gap.** Across the 2 cases, what is the mean improvement
     in (data - prediction)/σ when we add the prime field?

If the prime field improves σ-deviation by less than 2σ on average, the
honest framing is: "the prime field is statistically detectable but
small. Calling it 'the explanation for dark matter' is stronger than
the data supports."

Note: this is two data points, not a population. The full statement
requires the SPARC 175 sample (see `predictions/sparc_per_galaxy_ml.py`
and `predictions/sparc_shape_only_test.py` which run the population
test). This adversarial script is the *minimal embedded* version that
can be replicated in seconds without external data.

Outputs:
  - JSON: evidence/adversarial/null_baryonic_baseline.json
  - Verdict: "PRIME-FIELD-NECESSARY" or "GAP-SMALL".
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, List

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from prime_field_util import R0_KPC_CANONICAL  # noqa: E402

OUT_DIR = _ROOT / "evidence" / "adversarial"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Embedded test cases ----------------------------------------------------

V0_KMS = 397.0          # cosmological virial scale
R0_KPC = R0_KPC_CANONICAL

# Case 1: Milky Way at R = 10 kpc (Eilers+ 2019, Sofue 2013)
# IF Theory uses the gradient-based formula v_prime = v₀·√(R·|dΦ/dr|)
# for galactic rotation at the cosmological v₀. This gives 138.28 km/s
# at R = 10 kpc — matches predictions/mw_rotation_sigma_accounting.py.
MW = {
    "name": "Milky Way @ R=10 kpc",
    "formula": "gradient",   # v = v₀·√(R·|dΦ/dr|)
    "R_kpc": 10.0,
    "V_baryon_kms": 160.0,
    "V_obs_kms": 220.0,
    "V_obs_sigma_kms": 20.0,
    "v0_kms": V0_KMS,        # cosmological virial scale, ± 30%
}

# Case 2: NGC 3198 (canonical SPARC flat-rotation example).
# Numbers below are from Karukes & Salucci 2017 / Lelli SPARC table.
# We pick R = 1 disk scale length (R_d ≈ 2.62 kpc) where the baryon
# disk peaks and the flat regime is just beginning. Past R_d the
# baryons fall off and the prime field carries the curve.
# Robust embedded values (no fitting): Lelli, McGaugh & Schombert 2016
# SPARC table reports for NGC 3198:
#   Distance = 13.8 Mpc, R_d = 2.62 kpc, V_flat = 150 km/s
#   At R = 5·R_d ≈ 13.1 kpc, the curve is firmly flat at 150 km/s.
# We use R = 13.1 kpc, V_obs = 150 km/s, σ = 8 km/s.
# Baryon contribution at that radius is small — from SPARC rotmod
# files, V_disk(13 kpc) ≈ 92 km/s, V_gas(13 kpc) ≈ 35 km/s, V_bul = 0
# (no bulge for NGC 3198). Y_disk ≈ 0.5 (standard SPARC value).
NGC3198 = {
    "name": "NGC 3198 @ R=13.1 kpc",
    "formula": "vflat",      # v² = V_flat²·R/(R+r₀)
    "R_kpc": 13.1,
    # v_baryon = √(V_gas² + Y·(V_disk² + V_bul²)) with Y = 0.5
    "V_baryon_kms": math.sqrt(35.0 ** 2 + 0.5 * (92.0 ** 2 + 0.0 ** 2)),
    "V_obs_kms": 150.0,
    "V_obs_sigma_kms": 8.0,
    # For NGC 3198 the SPARC V_flat = 150 km/s anchors the asymptotic
    # prime-field velocity (see `predictions/sparc_shape_only_test.py`).
    "v0_kms": 150.0,  # SPARC V_flat for this galaxy
}


def v_prime_vflat(R_kpc: float, V_flat_kms: float,
                  r0_kpc: float = R0_KPC) -> float:
    """v_prime² = V_flat² · R / (R + r₀).

    Per-galaxy formula used in `predictions/sparc_shape_only_test.py`:
    the SPARC-table V_flat anchors the asymptotic, r₀ controls the shape.
    """
    return V_flat_kms * math.sqrt(R_kpc / (R_kpc + r0_kpc))


def v_prime_gradient(R_kpc: float, v0_kms: float,
                     r0_kpc: float = R0_KPC) -> float:
    """v_prime = v₀ · √(R · |dΦ/dr|).

    The IF Theory orbital-velocity formula at the cosmological v₀
    (see `core/parameter_derivations.py:_v_at` and
    `predictions/mw_rotation_sigma_accounting.py`). At R=10 kpc with
    canonical r₀ this gives ~138 km/s.
    """
    x = R_kpc / r0_kpc + 1.0
    log_x = math.log(x)
    grad = 1.0 / (r0_kpc * x * log_x ** 2)  # |dΦ/dr| in 1/kpc
    return v0_kms * math.sqrt(R_kpc * grad)


def v_prime_for_case(case: Dict) -> float:
    if case["formula"] == "gradient":
        return v_prime_gradient(case["R_kpc"], case["v0_kms"])
    if case["formula"] == "vflat":
        return v_prime_vflat(case["R_kpc"], case["v0_kms"])
    raise ValueError(f"Unknown formula: {case['formula']}")


def v_total_if(case: Dict) -> float:
    vp = v_prime_for_case(case)
    return math.sqrt(case["V_baryon_kms"] ** 2 + vp ** 2)


def v_total_null(case: Dict) -> float:
    """Null model: baryons only, no prime field."""
    return case["V_baryon_kms"]


def evaluate(case: Dict) -> Dict:
    vt_if = v_total_if(case)
    vt_null = v_total_null(case)
    sigma_if = abs(case["V_obs_kms"] - vt_if) / case["V_obs_sigma_kms"]
    sigma_null = abs(case["V_obs_kms"] - vt_null) / case["V_obs_sigma_kms"]
    improvement_sigma = sigma_null - sigma_if
    return {
        "name": case["name"],
        "formula": case["formula"],
        "R_kpc": case["R_kpc"],
        "V_baryon_kms": case["V_baryon_kms"],
        "V_obs_kms": case["V_obs_kms"],
        "V_obs_sigma_kms": case["V_obs_sigma_kms"],
        "v0_kms": case["v0_kms"],
        "v_prime_kms": v_prime_for_case(case),
        "v_total_if_kms": vt_if,
        "v_total_null_kms": vt_null,
        "delta_obs_minus_if_kms": case["V_obs_kms"] - vt_if,
        "delta_obs_minus_null_kms": case["V_obs_kms"] - vt_null,
        "sigma_if": sigma_if,
        "sigma_null": sigma_null,
        "improvement_sigma": improvement_sigma,
    }


def run() -> Dict:
    cases = [MW, NGC3198]
    evaluations: List[Dict] = [evaluate(c) for c in cases]

    mean_sigma_if = sum(e["sigma_if"] for e in evaluations) / len(evaluations)
    mean_sigma_null = sum(e["sigma_null"] for e in evaluations) / len(evaluations)
    mean_improvement = mean_sigma_null - mean_sigma_if

    # Verdict thresholds
    NECESSARY_THRESHOLD = 2.0   # mean improvement ≥ 2σ → prime field necessary
    SMALL_THRESHOLD = 0.5       # mean improvement < 0.5σ → gap small

    if mean_improvement >= NECESSARY_THRESHOLD:
        verdict = "PRIME-FIELD-NECESSARY"
        verdict_detail = (
            f"Adding the prime field improves data-vs-prediction agreement "
            f"by {mean_improvement:.2f}σ on average across the embedded "
            f"cases. The baryon-only null is significantly worse, so the "
            f"prime field is doing real work."
        )
    elif mean_improvement < SMALL_THRESHOLD:
        verdict = "GAP-SMALL"
        verdict_detail = (
            f"Adding the prime field only improves agreement by "
            f"{mean_improvement:.2f}σ on average. The prime field is not "
            f"distinguished from the baryon-only baseline in these tests."
        )
    else:
        verdict = "PRIME-FIELD-MARGINAL"
        verdict_detail = (
            f"Adding the prime field improves agreement by "
            f"{mean_improvement:.2f}σ on average — detectable but modest. "
            f"Calling the prime field 'the explanation for dark matter' "
            f"requires the full SPARC population test, not these 2 cases."
        )

    return {
        "inputs": {
            "r0_kpc_canonical": R0_KPC_CANONICAL,
            "v0_kms_cosmological": V0_KMS,
        },
        "cases": evaluations,
        "summary": {
            "mean_sigma_if": mean_sigma_if,
            "mean_sigma_null": mean_sigma_null,
            "mean_improvement_sigma": mean_improvement,
        },
        "verdict": verdict,
        "verdict_detail": verdict_detail,
        "honest_framing": (
            "If the prime field improves σ-deviation by less than 2σ on "
            "average over the baryon-only null model, the claim that 'the "
            "prime field replaces dark matter' is stronger than the data "
            "in this minimal embedded test supports. (The full claim rests "
            "on the SPARC 175 population test in `predictions/`.)"
        ),
        "caveat": (
            "This is a 2-point embedded test for runnable adversarial "
            "purposes. The population statement requires "
            "predictions/sparc_per_galaxy_ml.py over SPARC 175."
        ),
    }


def main() -> int:
    result = run()

    print("=" * 78)
    print("ADVERSARIAL TEST: baryon-only null baseline vs IF Theory")
    print("=" * 78)
    print(f"  Canonical r₀ = {R0_KPC_CANONICAL:.6f} kpc; v₀_cosmo = {V0_KMS} km/s")
    print()
    print(f"  {'case':>26} {'V_bary':>8} {'v_prime':>9} {'v_IF':>8} "
          f"{'V_obs':>7} {'σ_IF':>7} {'σ_null':>7} {'Δσ':>6}")
    for e in result["cases"]:
        print(f"  {e['name']:>26} "
              f"{e['V_baryon_kms']:>8.2f} "
              f"{e['v_prime_kms']:>9.2f} "
              f"{e['v_total_if_kms']:>8.2f} "
              f"{e['V_obs_kms']:>7.1f} "
              f"{e['sigma_if']:>7.2f} "
              f"{e['sigma_null']:>7.2f} "
              f"{e['improvement_sigma']:>+6.2f}")
    print()
    print(f"  Mean σ_IF   = {result['summary']['mean_sigma_if']:.2f}")
    print(f"  Mean σ_null = {result['summary']['mean_sigma_null']:.2f}")
    print(f"  Mean improvement = {result['summary']['mean_improvement_sigma']:+.2f} σ")
    print()
    print(f"  Honest framing:")
    print(f"    {result['honest_framing']}")
    print()
    print(f"  Caveat:")
    print(f"    {result['caveat']}")
    print()
    print(f"  VERDICT: {result['verdict']}")
    print(f"    {result['verdict_detail']}")
    print("=" * 78)

    out_file = OUT_DIR / "null_baryonic_baseline.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {out_file}")

    return 0 if result["verdict"] == "PRIME-FIELD-NECESSARY" else 1


if __name__ == "__main__":
    sys.exit(main())
