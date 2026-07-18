#!/usr/bin/env python3
"""delta_max_forward_prediction.py — the de-circularized Hubble-tension test.

Supersedes the INTERPRETATION of delta_max_derivation.py (kept for
history), whose 2026-06-12 referee review showed was circular: it took
the observed SH0ES/Planck ratio as INPUT and reported a "0.3% match"
against a calibration to the same ratio.

This script runs the logic FORWARD, the only direction that predicts:

  INPUT  (independent of any H0 measurement):
    - cosmic void under-density range from void catalogs:
      Pan et al. 2012 (SDSS DR7): typical 30-70% under-dense
      Sutter et al. 2014 (ZOBOV): 20-80% range
    - LTB linear-order matter-dominated enhancement:
      H_local/H_inf = sqrt(1 + delta_void/3)

  OUTPUT (the prediction):
    - a band for H_local/H_inf, hence for H_local given Planck H_inf.

  TEST:
    - does the SH0ES measurement fall inside the predicted band?

This is weaker than the old "0.3% match" — and that is the point: the
0.3% was arithmetic, this band is physics. A band this wide (set by the
width of the void-depth distribution) is honestly reported as a
CONSISTENCY test, not a precision derivation. It becomes sharp if/when
the actual local void depth at the SH0ES volume is measured directly.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / "evidence" / "delta_max_forward"
OUT_DIR.mkdir(parents=True, exist_ok=True)

H_PLANCK = 67.4          # km/s/Mpc (input: CMB, not local)
H_SH0ES = 73.04          # km/s/Mpc (the measurement under test — NOT an input)
H_SH0ES_ERR = 1.04

# Void-depth priors from catalogs (independent of H0):
VOID_RANGES = {
    "Pan_2012_typical": (0.30, 0.70),
    "Sutter_2014_range": (0.20, 0.80),
}


def h_ratio(delta_void: float) -> float:
    """LTB linear-order matter-dominated enhancement."""
    return math.sqrt(1.0 + delta_void / 3.0)


def main() -> int:
    print("=" * 72)
    print("FORWARD prediction: void catalogs + LTB -> H_local band -> compare SH0ES")
    print("=" * 72)
    results = {}
    for name, (lo, hi) in VOID_RANGES.items():
        h_lo, h_hi = H_PLANCK * h_ratio(lo), H_PLANCK * h_ratio(hi)
        inside = h_lo <= H_SH0ES <= h_hi
        # where in the band does SH0ES sit -> implied void depth
        implied_delta = 3.0 * ((H_SH0ES / H_PLANCK) ** 2 - 1.0)
        results[name] = {
            "delta_void_range": [lo, hi],
            "predicted_H_local_band": [round(h_lo, 2), round(h_hi, 2)],
            "sh0es_inside_band": bool(inside),
            "implied_delta_void_at_sh0es": round(implied_delta, 3),
        }
        print(f"\n{name}: δ_void ∈ [{lo}, {hi}]")
        print(f"  predicted H_local ∈ [{h_lo:.2f}, {h_hi:.2f}] km/s/Mpc")
        print(f"  SH0ES = {H_SH0ES} ± {H_SH0ES_ERR} → inside band: {inside}")

    implied = results["Pan_2012_typical"]["implied_delta_void_at_sh0es"]
    verdict = (
        "CONSISTENT — SH0ES falls inside the void-catalog-predicted H_local band; "
        f"implied local void depth {implied:.0%} is observationally typical. "
        "HONEST WEIGHT: the band is wide (~5 km/s/Mpc); this is a consistency "
        "test, not a precision derivation. It sharpens only with a direct "
        "measurement of the local void depth at the SH0ES volume."
        if all(r["sh0es_inside_band"] for r in results.values())
        else "TENSION — SH0ES falls outside the predicted band; mechanism disfavored."
    )
    print(f"\nVERDICT: {verdict}")

    out = {
        "artifact": "delta_max forward prediction (de-circularized)",
        "supersedes_interpretation_of": "predictions/delta_max_derivation.py",
        "inputs": {"H_planck": H_PLANCK, "void_ranges": VOID_RANGES,
                   "ltb": "H_local/H_inf = sqrt(1 + delta_void/3)"},
        "measurement_under_test": {"H_sh0es": H_SH0ES, "err": H_SH0ES_ERR},
        "results": results,
        "verdict": verdict,
    }
    (OUT_DIR / "delta_max_forward.json").write_text(json.dumps(out, indent=2))
    print(f"Wrote {OUT_DIR / 'delta_max_forward.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
