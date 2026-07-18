#!/usr/bin/env python3
"""Pre-registered IF Theory H0 transition curve.

This script writes a specific scale-dependent H0(L) curve before adding any
new intermediate-scale H0 data. It is a prediction artifact, not a fit report.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / "evidence" / "hubble_transition_curve"
OUT_DIR.mkdir(parents=True, exist_ok=True)

H0_GLOBAL = 67.4
V0_KMS = 397.0
DELTA_VOID = 0.50
L_SH0ES_REFERENCE_MPC = 5.0


def r_bubble_mpc(v0_kms: float = V0_KMS, h0_global: float = H0_GLOBAL) -> float:
    """Bubble radius r_b = v0/H0 * sqrt(3), in Mpc."""
    return (v0_kms / h0_global) * math.sqrt(3.0)


def delta_h_from_void(delta_void: float = DELTA_VOID) -> float:
    """LTB linear-order local Hubble enhancement from a void under-density."""
    return math.sqrt(1.0 + delta_void / 3.0) - 1.0


def delta_max() -> float:
    """Maximum bubble enhancement mapped from the SH0ES reference scale."""
    return delta_h_from_void() * math.exp(L_SH0ES_REFERENCE_MPC / r_bubble_mpc())


def h0_at_scale(distance_mpc: float) -> float:
    """Locked IF transition curve H0(L) = H_inf * (1 + delta_max exp(-L/r_b))."""
    return H0_GLOBAL * (1.0 + delta_max() * math.exp(-distance_mpc / r_bubble_mpc()))


def transition_table() -> list[dict[str, float]]:
    distances = [
        0.0, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 50.0,
        75.0, 100.0, 150.0, 300.0, 500.0, 1000.0, 14000.0,
    ]
    return [
        {
            "distance_mpc": distance,
            "h0_km_s_mpc": h0_at_scale(distance),
        }
        for distance in distances
    ]


def main() -> int:
    rows = transition_table()
    rb = r_bubble_mpc()
    dmax = delta_max()
    out = {
        "artifact": "IF Theory pre-registered H0 transition curve",
        "model": "H0(L)=H0_global*(1+delta_max*exp(-L/r_bubble))",
        "status": "PREDICTION_LOCKED_NOT_INDEPENDENTLY_CONFIRMED",
        "inputs": {
            "H0_global_km_s_mpc": H0_GLOBAL,
            "v0_km_s": V0_KMS,
            "delta_void": DELTA_VOID,
            "L_SH0ES_reference_mpc": L_SH0ES_REFERENCE_MPC,
        },
        "derived": {
            "r_bubble_mpc": rb,
            "delta_h_at_SH0ES_reference": delta_h_from_void(),
            "delta_max": dmax,
        },
        "confirmation_rule": {
            "pass": "Independent distance-binned H0 data must follow the locked monotone curve with chi2/dof <= 2 without refitting r_bubble or delta_max.",
            "fail": "A precise monotone-flat H0 across 1-300 Mpc, or a transition radius outside 5-25 Mpc, falsifies this version.",
        },
        "curve": rows,
    }
    json_path = OUT_DIR / "hubble_transition_curve.json"
    csv_path = OUT_DIR / "hubble_transition_curve.csv"
    json_path.write_text(json.dumps(out, indent=2))
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["distance_mpc", "h0_km_s_mpc"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"r_bubble = {rb:.3f} Mpc")
    print(f"delta_max = {dmax:.6f}")
    print(f"H0(5 Mpc) = {h0_at_scale(5.0):.3f} km/s/Mpc")
    print(f"H0(100 Mpc) = {h0_at_scale(100.0):.3f} km/s/Mpc")
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
