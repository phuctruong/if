#!/usr/bin/env python3
"""dwarf_regime_split.py — WHERE does IF lose to MOND? The regime question.

Referee artifact (Claude Fable 5 review loop, 2026-06-12). IF's galactic
law v² = v₀²·R/(R+r₀) saturates for R ≫ r₀ ≈ 0.66 kpc — i.e. it predicts
near-flat curves essentially everywhere beyond ~2 kpc. MOND instead keeps
shaping curves through the low-acceleration regime. Dwarf/LSB galaxies
(slowly rising curves over many kpc) are therefore the regime where the
two theories diverge MAXIMALLY — the discriminating subsample.

This reads the sealed full-run evidence
(evidence/sparc_fair_benchmark/sparc_fair_benchmark_full.json, all 175
galaxies, identical fairness rules: 1 fitted M/L each for IF and MOND)
plus the SPARC master table, splits by V_flat class, and reports
per-class medians, head-to-head win rates, and the verdict.

Outcomes (pre-stated):
- If IF ≈ MOND on spirals but loses badly on dwarfs → the saturating
  form is the failure mode; the law is incomplete at low accelerations.
- If IF loses uniformly → the deficit is not regime-specific.
- If IF WINS any class → report it: that class is IF's evidence base.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from predictions.sparc_corrected_log_potential import SPARC_TABLE, parse_sparc_table  # noqa: E402

FULL_JSON = _ROOT / "evidence" / "sparc_fair_benchmark" / "sparc_fair_benchmark_full.json"
OUT_DIR = _ROOT / "evidence" / "adversarial"

DWARF_VFLAT = 80.0    # km/s
MASSIVE_VFLAT = 150.0  # km/s


def classify(vflat: float) -> str:
    if vflat <= 0 or np.isnan(vflat):
        return "unknown"
    if vflat < DWARF_VFLAT:
        return "dwarf"
    if vflat < MASSIVE_VFLAT:
        return "intermediate"
    return "massive"


def main() -> int:
    if not FULL_JSON.exists():
        print(f"MISSING {FULL_JSON} — run predictions/sparc_fair_benchmark.py --max-galaxies 0 first")
        return 1
    bench = json.loads(FULL_JSON.read_text())
    table = parse_sparc_table(SPARC_TABLE)

    groups: dict[str, list[dict]] = {}
    for row in bench["per_galaxy"]:
        name = row["name"]
        meta = table.get(name)
        vflat = float(meta.get("Vflat_kms", 0.0)) if meta else 0.0
        cls = classify(vflat)
        groups.setdefault(cls, []).append({
            "name": name, "vflat": vflat,
            "if_x2": row["IF"]["chi2_per_dof"],
            "mond_x2": row["MOND"]["chi2_per_dof"],
            "nfw_x2": row["NFW"]["chi2_per_dof"],
        })

    print(f"{'class':14s} {'n':>4s} {'IF med':>9s} {'MOND med':>9s} {'NFW med':>8s} "
          f"{'IF/MOND':>8s} {'IF wins':>8s}")
    summary = {}
    for cls in ["dwarf", "intermediate", "massive", "unknown"]:
        rows = groups.get(cls, [])
        if not rows:
            continue
        if_x = np.array([r["if_x2"] for r in rows])
        mo_x = np.array([r["mond_x2"] for r in rows])
        nf_x = np.array([r["nfw_x2"] for r in rows])
        wins = float(np.mean(if_x < mo_x))
        ratio = float(np.median(if_x) / np.median(mo_x))
        print(f"{cls:14s} {len(rows):4d} {np.median(if_x):9.2f} {np.median(mo_x):9.2f} "
              f"{np.median(nf_x):8.2f} {ratio:8.2f} {wins:8.1%}")
        summary[cls] = {
            "n": len(rows),
            "median_chi2_IF": float(np.median(if_x)),
            "median_chi2_MOND": float(np.median(mo_x)),
            "median_chi2_NFW": float(np.median(nf_x)),
            "IF_over_MOND_median_ratio": ratio,
            "IF_beats_MOND_fraction": wins,
        }

    d, m = summary.get("dwarf"), summary.get("massive")
    if d and m:
        regime_specific = d["IF_over_MOND_median_ratio"] > 1.5 * m["IF_over_MOND_median_ratio"]
        if regime_specific:
            verdict = ("REGIME-SPECIFIC-DEFICIT: IF's loss to MOND concentrates in dwarfs "
                       f"(ratio {d['IF_over_MOND_median_ratio']:.2f} vs {m['IF_over_MOND_median_ratio']:.2f} "
                       "in massive) — the saturating R/(R+r0) form is the failure mode at "
                       "low accelerations; the galactic law is incomplete there.")
        elif d["IF_over_MOND_median_ratio"] < 1.0:
            verdict = "IF-FAVORED-IN-DWARFS: IF beats MOND in the maximally discriminating regime."
        else:
            verdict = ("UNIFORM-DEFICIT: IF trails MOND across classes by similar factors; "
                       "the deficit is not regime-specific.")
    else:
        verdict = "INSUFFICIENT-CLASSES"
    print(f"\nVERDICT: {verdict}")

    out = {
        "artifact": "dwarf regime split — IF vs MOND discriminating subsample",
        "source": str(FULL_JSON.relative_to(_ROOT)),
        "class_definition": {"dwarf": f"Vflat < {DWARF_VFLAT}",
                             "intermediate": f"{DWARF_VFLAT} <= Vflat < {MASSIVE_VFLAT}",
                             "massive": f"Vflat >= {MASSIVE_VFLAT}"},
        "summary": summary,
        "verdict": verdict,
    }
    (OUT_DIR / "dwarf_regime_split.json").write_text(json.dumps(out, indent=2))
    print(f"Wrote {OUT_DIR / 'dwarf_regime_split.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
