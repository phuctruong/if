#!/usr/bin/env python3
"""power_law_null_test.py — does the BOSS 'Pearson r = +0.98' discriminate?

Referee check added 2026-06-12 (Claude Fable 5 external review pass).

SCORE.md headlines 'BOSS ξ(r) shape — Pearson r = +0.98 in log-log' as a
PASS for ξ(r) = C_XI·[Φ(r)]². This test asks the only question that
matters for that statistic: would a generic no-theory null do as well?

Null model: a pure power law ξ ∝ r^(-γ). Key fact: Pearson r computed in
log-log space is INVARIANT to γ (log of a power law is affine in log r,
and Pearson r is affine-invariant), so the null needs no tuning at all —
every power law scores identically.

PASS (for IF) requires the IF shape to beat the power-law null by a
margin (Δr ≥ 0.01 on both samples). If the null matches or beats IF,
the r = 0.98 headline has no discriminating power and must not be
counted as evidence for the [1/log]² form specifically.

Exit 0 either way (this is a measurement, not a gate); verdict in JSON.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

from prime_field_util import R0_KPC_CANONICAL  # noqa: E402

BOSS_DIR = Path.home() / "Downloads" / "if" / "data" / "boss_published_xi"
OUT_DIR = _ROOT / "evidence" / "adversarial"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FIT_RANGE_MPC = (10.0, 180.0)


def load_cuesta(path: Path) -> np.ndarray:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        rows.append((float(parts[0]), float(parts[1])))
    return np.array(rows)


def pearson_log(xi: np.ndarray, pred: np.ndarray) -> float:
    return float(np.corrcoef(np.log(xi), np.log(pred))[0, 1])


def main() -> int:
    samples = {
        "LOWZ_DR12": BOSS_DIR / "Cuesta_2016_LOWZDR12_corrfunction_x0_prerecon.dat",
        "CMASS_DR12": BOSS_DIR / "Cuesta_2016_CMASSDR12_corrfunction_x0_prerecon.dat",
    }
    results = {}
    for name, path in samples.items():
        if not path.exists():
            print(f"SKIP {name}: {path} not staged (see REPLICATION.md step 4)")
            return 1
        d = load_cuesta(path)
        r_mpc, xi = d[:, 0], d[:, 1]
        m = (xi > 0) & (r_mpc >= FIT_RANGE_MPC[0]) & (r_mpc <= FIT_RANGE_MPC[1])
        r, x = r_mpc[m], xi[m]
        rk = r * 1000.0  # kpc

        r_if = pearson_log(x, (1.0 / np.log(rk / R0_KPC_CANONICAL + 1.0)) ** 2)
        r_pl = pearson_log(x, r ** -1.8)  # any exponent gives the same r

        results[name] = {
            "n_bins": int(m.sum()),
            "pearson_log_IF_shape": round(r_if, 4),
            "pearson_log_power_law_null": round(r_pl, 4),
            "delta": round(r_if - r_pl, 4),
        }
        print(f"{name}: IF r={r_if:+.4f}  power-law null r={r_pl:+.4f}  "
              f"Δ={r_if - r_pl:+.4f}  (n={m.sum()})")

    if_beats_null = all(v["delta"] >= 0.01 for v in results.values())
    verdict = (
        "DISCRIMINATING — IF shape beats the power-law null by ≥0.01 on all samples"
        if if_beats_null else
        "NON-DISCRIMINATING — a generic power law matches or beats the IF shape; "
        "the 'Pearson r = +0.98' BOSS headline is not evidence for [1/log]² "
        "specifically and should be reported as a consistency check only"
    )
    print(f"\nVERDICT: {verdict}")

    out = {
        "test": "power_law_null_test",
        "fit_range_mpc": FIT_RANGE_MPC,
        "r0_kpc": R0_KPC_CANONICAL,
        "note": ("Pearson r in log-log space is affine-invariant, hence identical "
                 "for every power-law exponent — the null requires zero tuning."),
        "results": results,
        "verdict": verdict,
    }
    with open(OUT_DIR / "power_law_null_test.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {OUT_DIR / 'power_law_null_test.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
