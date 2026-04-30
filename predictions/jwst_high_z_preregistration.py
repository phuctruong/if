#!/usr/bin/env python3
"""Pre-register JWST high-redshift confirmation and falsification thresholds."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from predictions.jwst_early_galaxies import JWSTEarlyGalaxyPredictions  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent.parent / "evidence" / "jwst_preregistration"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def prediction_grid() -> list[dict[str, float]]:
    predictor = JWSTEarlyGalaxyPredictions()
    masses = [1e8, 5e8, 1e9, 5e9, 1e10]
    rows = []
    for mass in masses:
        result = predictor.predict_maximum_formation_redshift(mass)
        t_if = float(result["t_formation_if_myr"])
        t_lcdm = float(result["t_formation_lcdm_myr"])
        speedup = t_lcdm / t_if if t_if > 0 else float("inf")
        rows.append(
            {
                "stellar_or_halo_mass_msun": mass,
                "earliest_z_if": float(result["z_max_if"]),
                "earliest_z_lcdm": float(result["z_max_lcdm"]),
                "t_formation_if_myr": t_if,
                "t_formation_lcdm_myr": t_lcdm,
                "speedup_factor": speedup,
            }
        )
    return rows


def main() -> int:
    out = {
        "artifact": "IF Theory JWST high-z preregistration",
        "status_as_of_2026_04_30": "SUPPORTIVE_BELOW_TRIGGER",
        "current_context": {
            "highest_confirmed_used_for_context": "MoM-z14, z_spec=14.44, below the z>=20 trigger",
            "do_not_count_as_confirmation": "Any post-hoc z<20 agreement or photometric-only z>=20 candidate",
        },
        "locked_confirmation_thresholds": {
            "confirm": {
                "condition": "At least one spectroscopically confirmed mature galaxy at z >= 20",
                "maturity_cut": "stellar_mass_msun >= 1e8 OR M_UV <= -19.5 with non-AGN-dominated spectrum",
                "deadline": "2030-01-01",
            },
            "strong_confirm": {
                "condition": "At least one spectroscopically confirmed mature galaxy at z >= 25",
                "maturity_cut": "stellar_mass_msun >= 1e8 OR M_UV <= -19.5 with non-AGN-dominated spectrum",
                "deadline": "2030-01-01",
            },
            "fail_closed": {
                "condition": "Sufficient JWST/Roman spectroscopy covers the expected survey volume and finds no mature galaxies at z >= 18",
                "note": "Survey volume threshold must be specified before applying this fail condition to a named survey.",
            },
        },
        "if_theory_prediction_grid": prediction_grid(),
    }
    path = OUT_DIR / "jwst_high_z_preregistration.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {path}")
    print("Trigger: spectroscopic mature galaxy z >= 20; strong trigger z >= 25.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
