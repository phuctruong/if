#!/usr/bin/env python3
"""
c_xi_uniqueness_test.py — adversarial test of the C_XI = 62 claim.

The Mersenne Tower Theorem (`mersenne_tower_theorem.py`) derives
C_XI = 2 · π(127) = 62 as the exact normalization of the two-point
correlation function:

    ξ(r) = C_XI · [Φ(r)]²,  C_XI = 62.

The selling point is that 62 is **not fitted**. It falls out of number
theory (π(127) = 31 is itself a Mersenne prime, the only known
Mersenne prime that is tower-closed in this sense).

This script asks the adversarial question: **does the BOSS ξ(r) shape
care?** If nearby integers — 60, 61, 63, 64, 65 — fit the data
comparably to 62 within the published errors, then the integer-from-
number-theory derivation is decorative for *this particular dataset*.
The Mersenne Tower Theorem may still be a beautiful mathematical
result, but the empirical claim "BOSS confirms C_XI = 62 exactly"
becomes weaker.

The framing is honest: we DON'T expect the BOSS shape alone to
uniquely pin down 62 vs 61 or 63. The amplitude is dominated by σ₈
and r₀ via ξ(r) = C_XI · [Φ(r)]², and a few-percent integer shift can
be absorbed elsewhere. The point of this test is to **quantify** how
much.

Methodology (no external downloads):

  - Use a single representative BOSS CMASS DR12 ξ(r) bin (r ≈ 25 Mpc/h,
    ξ ≈ 0.0237, σ_ξ ≈ 0.0015), cross-checked against the boss test.
  - For each candidate integer C ∈ {60, 61, 62, 63, 64, 65}, compute
    χ²(C) and Δχ² = χ²(C) - χ²(62).
  - Apply standard Wilks: 1σ ⇔ |Δχ²| ≈ 1; 5σ ⇔ |Δχ²| ≥ 25.

If neighboring integers are within 1σ of 62 by this metric, the
integer-from-Mersenne-tower derivation is not distinguished by this
dataset.

Caveat: this is a SINGLE-BIN test for tractability without downloads.
The full BOSS ξ(r) is 18 bins with covariance — but most of the
amplitude information is in the well-resolved bins like the one
chosen. If even single-bin doesn't pin down 62 ± 1, the multi-bin
test almost certainly doesn't either.

Outputs:
  - JSON: evidence/adversarial/c_xi_uniqueness_test.json
  - Verdict: "DISTINGUISHED" or "INTEGER-RANGE-FITS" (decorative).
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

from prime_field_util import C_XI_CANONICAL, R0_KPC_CANONICAL  # noqa: E402

OUT_DIR = _ROOT / "evidence" / "adversarial"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- BOSS CMASS representative bin (Cuesta 2016 DR12) -----------------------
H_PLANCK15 = 0.6774
BOSS_R_MPCH = 25.0
BOSS_R_MPC = BOSS_R_MPCH / H_PLANCK15
BOSS_XI_DATA = 0.0237
BOSS_XI_SIGMA = 0.0015

CANDIDATE_C_XI = [60, 61, 62, 63, 64, 65]


def field_phi(r_mpc: float, r0_kpc: float = R0_KPC_CANONICAL) -> float:
    """Φ(r) = 1 / log(r_kpc / r0_kpc + 1)."""
    r_kpc = r_mpc * 1000.0
    return 1.0 / math.log(r_kpc / r0_kpc + 1.0)


def xi_pred(r_mpc: float, c_xi: float, r0_kpc: float = R0_KPC_CANONICAL) -> float:
    return c_xi * field_phi(r_mpc, r0_kpc) ** 2


def chi2_single_bin(c_xi: float) -> float:
    """χ² at the representative BOSS bin for a candidate C_XI."""
    return ((xi_pred(BOSS_R_MPC, c_xi) - BOSS_XI_DATA) / BOSS_XI_SIGMA) ** 2


def best_fit_amplitude() -> float:
    """C_XI value that would minimize χ² (continuous, not integer)."""
    # Single-bin χ² is quadratic in C_XI; minimum is at the value that
    # makes ξ_pred = ξ_obs exactly.
    phi_sq = field_phi(BOSS_R_MPC) ** 2
    return BOSS_XI_DATA / phi_sq


def best_fit_r0_for_c_xi(c_xi: float,
                          r0_kpc_min: float = 1e-30,
                          r0_kpc_max: float = 1e30) -> Tuple[float, float]:
    """Find r₀ such that c_xi · Φ(BOSS bin, r₀)² = ξ_obs (single-bin).

    Returns (r0_kpc_best, chi2_at_best). For single-bin, the minimum χ²
    is zero whenever such r₀ exists — i.e., any integer C_XI fits the
    bin with some r₀. The interesting question becomes: how does that
    r₀ differ from the canonical 0.6595 kpc?
    """
    # We need 1/log(r/r0 + 1)² = ξ_obs / c_xi
    target = BOSS_XI_DATA / c_xi
    if target <= 0:
        return float("nan"), float("inf")
    # 1/log(...)² = target ⇒ log(...) = ±1/√target ⇒ r/r0 = exp(±1/√target) - 1
    # Take positive root (r > 0, r₀ > 0):
    inv_log = 1.0 / math.sqrt(target)
    # r_kpc = BOSS_R_MPC * 1000
    r_kpc = BOSS_R_MPC * 1000.0
    ratio = math.exp(inv_log) - 1.0
    if ratio <= 0:
        return float("nan"), float("inf")
    r0 = r_kpc / ratio
    # Clamp to plausible range; if out of range, return NaN.
    if not (r0_kpc_min < r0 < r0_kpc_max):
        return float("nan"), float("inf")
    # χ² at this r₀: zero by construction
    return r0, 0.0


def run() -> Dict:
    chi2_canonical = chi2_single_bin(C_XI_CANONICAL)
    best_c = best_fit_amplitude()

    table: List[Dict] = []
    for c in CANDIDATE_C_XI:
        chi2_c = chi2_single_bin(c)
        delta_chi2 = chi2_c - chi2_canonical
        # |Δχ²| ≈ σ² in single-parameter offset metric.
        sigma_offset = math.sqrt(max(abs(delta_chi2), 0.0))
        # Best-fit r₀ for this integer (a "regime r₀" that absorbs the
        # integer choice — directly tests whether 60-65 are interchangeable
        # once the resolution-prime principle is allowed to slide r₀):
        r0_for_c, _ = best_fit_r0_for_c_xi(c)
        table.append({
            "c_xi": c,
            "xi_predicted": xi_pred(BOSS_R_MPC, c),
            "chi2": chi2_c,
            "delta_chi2_vs_62": delta_chi2,
            "sigma_offset_vs_62": sigma_offset,
            "r0_kpc_best_fit": r0_for_c,
            "r0_ratio_vs_canonical": (
                r0_for_c / R0_KPC_CANONICAL if not math.isnan(r0_for_c)
                else float("nan")
            ),
            "tag": "CANONICAL (Mersenne tower)" if c == C_XI_CANONICAL
                   else f"{c - C_XI_CANONICAL:+d} from canonical",
        })

    # Count integers within 1σ of 62 (i.e. |Δχ²| < 1).
    within_1sigma = [r["c_xi"] for r in table
                     if r["c_xi"] != C_XI_CANONICAL
                     and abs(r["delta_chi2_vs_62"]) < 1.0]
    within_3sigma = [r["c_xi"] for r in table
                     if r["c_xi"] != C_XI_CANONICAL
                     and abs(r["delta_chi2_vs_62"]) < 9.0]

    # Range of integers within 1σ around the best-fit continuous
    # amplitude — a different way of asking "how many integers are
    # statistically equivalent?"
    integer_floor = math.floor(best_c)
    integer_ceil = math.ceil(best_c)
    # Allow ±5 range search.
    integers_consistent_with_data: List[int] = []
    for c in range(max(1, integer_floor - 5), integer_ceil + 6):
        chi2_c = chi2_single_bin(c)
        # 1σ for single bin = χ² < 1 above min.
        # We compare to the continuous best fit (χ² = 0 there).
        if chi2_c < 1.0:
            integers_consistent_with_data.append(c)

    # Crucial honest framing: check whether the canonical C_XI = 62 model
    # *itself* fits the data at all. If χ²(C=62) is catastrophically bad,
    # saying "C=62 is distinguished from C=60-65" is misleading — they are
    # ALL bad in the same way. The Mersenne tower then sits inside a
    # cluster of equally-failing integers, not at a uniquely-fitting point.
    chi2_canonical = chi2_single_bin(C_XI_CANONICAL)
    canonical_sigma_off_from_data = math.sqrt(chi2_canonical)
    canonical_catastrophic = canonical_sigma_off_from_data > 5.0

    if canonical_catastrophic:
        # Both the canonical and its neighbors fail vs the data in the
        # bare zero-parameter mode — this matches the published BOSS test
        # where H_0 (C_XI = 62, zero-param) gave χ²/dof ≈ 67000. The
        # honest verdict is that the integer choice cannot be tested
        # against data in this regime: the model is in an entirely
        # different amplitude regime than the BOSS galaxy data, which the
        # "resolution-prime principle" is invoked to explain.
        verdict = "AMPLITUDE-REGIME-MISMATCH"
        verdict_detail = (
            f"At canonical r₀ = 0.6595 kpc, ALL integers in "
            f"{{60-65}} give χ² > {min(r['chi2'] for r in table):.0e} "
            f"vs the BOSS bin σ — they are all "
            f"{canonical_sigma_off_from_data:.0f}σ off because the BOSS "
            f"galaxy data sits in a different amplitude regime than the "
            f"halo-derived r₀. The Mersenne tower integer 62 is NOT "
            f"distinguished from its neighbors by BOSS, because the "
            f"amplitude scale itself doesn't match. The shape test "
            f"(boss_published_xi_test.py H_1) absorbs this by fitting "
            f"amplitude separately; in that regime, the integer 62 is "
            f"compatible with a range of nearby values."
        )
    elif len(within_1sigma) >= 2:
        verdict = "INTEGER-RANGE-FITS"
        verdict_detail = (
            f"Integers {within_1sigma} fit within Δχ² < 1 of C_XI = 62 at "
            f"the representative BOSS bin. The integer-from-Mersenne-tower "
            f"derivation is NOT empirically distinguished from neighboring "
            f"integers by this dataset."
        )
    elif len(within_1sigma) == 1:
        verdict = "MARGINAL"
        verdict_detail = (
            f"Only one neighboring integer ({within_1sigma[0]}) is within "
            f"1σ of canonical C_XI = 62. The Mersenne tower derivation is "
            f"borderline distinguished."
        )
    else:
        verdict = "DISTINGUISHED"
        verdict_detail = (
            "All neighboring integers in {60, 61, 63, 64, 65} are more "
            "than 1σ from canonical C_XI = 62, AND C_XI = 62 itself is "
            "within a few σ of the data. The integer is empirically "
            "singled out."
        )

    return {
        "inputs": {
            "boss_r_mpch": BOSS_R_MPCH,
            "boss_r_mpc": BOSS_R_MPC,
            "boss_xi": BOSS_XI_DATA,
            "boss_xi_sigma": BOSS_XI_SIGMA,
            "r0_kpc_canonical": R0_KPC_CANONICAL,
            "c_xi_canonical": C_XI_CANONICAL,
            "candidates": CANDIDATE_C_XI,
        },
        "canonical_chi2": chi2_canonical,
        "canonical_sigma_offset_from_data": canonical_sigma_off_from_data,
        "canonical_is_catastrophic_at_canonical_r0": canonical_catastrophic,
        "best_fit_continuous_c_xi": best_c,
        "best_fit_delta_from_canonical": best_c - C_XI_CANONICAL,
        "integers_consistent_within_1sigma_of_data": integers_consistent_with_data,
        "table": table,
        "within_1sigma_of_canonical": within_1sigma,
        "within_3sigma_of_canonical": within_3sigma,
        "verdict": verdict,
        "verdict_detail": verdict_detail,
        "honest_framing": (
            "If 2+ neighboring integers in {60, 61, 63, 64, 65} fit the "
            "BOSS shape within Δχ² < 1 of C_XI = 62, the 'C_XI = 62 from "
            "number theory' derivation is decorative for this dataset — "
            "the Mersenne Tower Theorem remains a real number-theoretic "
            "result, but the empirical falsifier 'BOSS confirms 62 "
            "exactly' is not what it sounds."
        ),
    }


def main() -> int:
    result = run()

    print("=" * 78)
    print("ADVERSARIAL TEST: C_XI uniqueness from BOSS ξ(r)")
    print("=" * 78)
    print(f"  BOSS CMASS DR12 representative bin:")
    print(f"    r = {BOSS_R_MPCH} Mpc/h = {BOSS_R_MPC:.3f} Mpc (h = {H_PLANCK15})")
    print(f"    ξ_obs = {BOSS_XI_DATA} ± {BOSS_XI_SIGMA}")
    print()
    print(f"  Best-fit continuous C_XI = {result['best_fit_continuous_c_xi']:.3f} "
          f"({result['best_fit_delta_from_canonical']:+.3f} from canonical 62)")
    print(f"  Integers fitting the data within 1σ:"
          f" {result['integers_consistent_within_1sigma_of_data']}")
    print()
    print(f"  {'C_XI':>5} {'ξ_pred':>12} {'χ²':>10} {'Δχ² vs 62':>12} "
          f"{'σ vs 62':>9} {'r0_fit(kpc)':>13} {'r0_ratio':>10} tag")
    for r in result["table"]:
        r0_str = (f"{r['r0_kpc_best_fit']:>13.4g}" if not math.isnan(r['r0_kpc_best_fit'])
                  else f"{'NaN':>13}")
        rr_str = (f"{r['r0_ratio_vs_canonical']:>10.4g}"
                  if not math.isnan(r['r0_ratio_vs_canonical'])
                  else f"{'NaN':>10}")
        print(f"  {r['c_xi']:>5d} "
              f"{r['xi_predicted']:>12.4e} "
              f"{r['chi2']:>10.3f} "
              f"{r['delta_chi2_vs_62']:>+12.3f} "
              f"{r['sigma_offset_vs_62']:>9.2f} "
              f"{r0_str} "
              f"{rr_str} "
              f"{r['tag']}")
    print()
    print(f"  Integers within 1σ of canonical C_XI = 62: "
          f"{result['within_1sigma_of_canonical']}")
    print(f"  Integers within 3σ of canonical C_XI = 62: "
          f"{result['within_3sigma_of_canonical']}")
    print()
    print(f"  Honest framing:")
    print(f"    {result['honest_framing']}")
    print()
    print(f"  VERDICT: {result['verdict']}")
    print(f"    {result['verdict_detail']}")
    print("=" * 78)

    out_file = OUT_DIR / "c_xi_uniqueness_test.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {out_file}")

    # Exit 0 only if the integer is empirically singled out by the data
    # cited; exit 1 if any honest finding (range fits, marginal, amplitude
    # mismatch) shows the canonical claim is weaker than stated.
    return 0 if result["verdict"] == "DISTINGUISHED" else 1


if __name__ == "__main__":
    sys.exit(main())
