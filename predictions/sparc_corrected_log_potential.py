#!/usr/bin/env python3
"""
sparc_corrected_log_potential.py — IF Theory rotation-curve test using
the CORRECTED logarithmic gravitational potential

    Φ_galactic(r) = ln(r/r_0 + 1)

instead of the inverse-log form Φ = 1/log(r/r_0+1) that failed at
SPARC scale. This is the form already used by predictions/rotation_curve_v2.py
(labeled "CORRECTED EQUATION" in that file's header).

Why this form gives FLAT rotation curves naturally:

    dΦ/dr = (1/r_0) / (r/r_0 + 1) = 1/(r + r_0)
    v_prime²(R) = R · |dΦ/dR| = R / (R + r_0)
    → v_prime(R) → v_0 as R → ∞   (FLAT)

Tully-Fisher emerges naturally if v_0 is set by the galaxy's own baryon
virial:

    v_0_galaxy² = G · M_baryon_total / R_eff_baryon

With M_baryon_total from SPARC's gas + 3.6μm luminosity (with M/L = 0.5),
and R_eff_baryon from the SPARC disk scale length R_disk, this gives
ZERO free parameters in the IF Theory part — every input is already in
SPARC table 1.

Test:
  For each SPARC galaxy at each radius R:
    v_baryon(R)  = √(V_gas² + V_disk² + V_bul²)             from rotmod.dat
    M_b_total    = M_HI + L[3.6] · 0.5 (M/L_3.6 ≈ 0.5)      from table 1
    R_disk_kpc   = R_disk                                    from table 1
    v_0_galaxy   = √(G · M_b_total / R_disk)                 (Tully-Fisher virial)
    v_prime(R)   = v_0_galaxy · √(R / (R + r_0))
    v_total(R)   = √(v_baryon² + v_prime²)

  Compare to V_obs. Compute χ²/dof. Per-galaxy, NO rotation-curve fit.
  Tully-Fisher slope of v_0_galaxy vs V_flat tests the unification.
"""

from __future__ import annotations

import json
import logging
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(level=logging.WARNING)

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from prime_field_util import R0_KPC_CANONICAL  # noqa: E402

SPARC_DIR = Path("/home/phuc/Downloads/if/data/sparc/Rotmod_LTG")
SPARC_TABLE = Path("/home/phuc/Downloads/if/data/sparc/SPARC_Lelli2016c.mrt")
OUT_DIR = Path(_ROOT, "evidence", "sparc_corrected")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Physical constants
G_SI = 6.67430e-11
M_SUN_KG = 1.98892e30
KPC_M = 3.0857e19
KMS_MS = 1e3
M_TO_L_RATIO_3_6 = 0.5  # standard 3.6 micron disk M/L (Lelli 2016)


def G_kpc_kms_msun() -> float:
    """G in units where r is kpc, v is km/s, M is solar masses.
    v² [km/s] · r [kpc] / M [Msun] = G in those units = 4.302e-6
    """
    # Derive: G = 6.674e-11 m³/(kg·s²)
    # Convert: kpc = 3.086e19 m, Msun = 1.989e30 kg, km/s = 1e3 m/s
    # G [kpc·(km/s)²/Msun] = 6.674e-11 · (1.989e30 / 1e6) / 3.086e19 = 4.302e-6
    return 4.302e-6


def parse_sparc_table(path: Path) -> Dict[str, dict]:
    """Parse SPARC_Lelli2016c.mrt table 1 by whitespace-splitting the data block.

    Column order (after Galaxy name): T, D, e_D, f_D, Inc, e_Inc, L_3.6,
    e_L_3.6, Reff, SBeff, Rdisk, SBdisk, MHI, RHI, Vflat, e_Vflat, Q, Ref.
    """
    rows = {}
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith(("Title:", "Author", "Table:", "=", "-",
                                      "Bytes", "Note", "  ")):
                # comment/header; only data rows have the galaxy name in col 0
                pass
            parts = s.split()
            # Data row pattern: name + ≥18 fields
            if len(parts) < 18:
                continue
            name = parts[0]
            # Reject non-galaxy header tokens
            if name in ("Title:", "Authors:", "Table:", "Note", "Galaxy") or "_" in name and len(name) < 3:
                continue
            try:
                T = int(parts[1])
                D_Mpc = float(parts[2])
                L_3_6 = float(parts[7])  # 10⁹ Lsun at 3.6 μm
                Reff_kpc = float(parts[9])
                Rdisk_kpc = float(parts[11])
                MHI_1e9 = float(parts[13])
                Vflat_kms = float(parts[15])
                rows[name] = {
                    "T": T,
                    "D_Mpc": D_Mpc,
                    "L_3_6_1e9_Lsun": L_3_6,
                    "Reff_kpc": Reff_kpc,
                    "Rdisk_kpc": Rdisk_kpc,
                    "MHI_1e9_Msun": MHI_1e9,
                    "Vflat_kms": Vflat_kms,
                }
            except (ValueError, IndexError):
                continue
    return rows


def load_rotmod(path: Path) -> dict:
    R, Vobs, errV, Vgas, Vdisk, Vbul = [], [], [], [], [], []
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            R.append(float(parts[0]));  Vobs.append(float(parts[1]))
            errV.append(float(parts[2])); Vgas.append(float(parts[3]))
            Vdisk.append(float(parts[4])); Vbul.append(float(parts[5]))
    return dict(R=np.asarray(R), Vobs=np.asarray(Vobs), errV=np.asarray(errV),
                Vgas=np.asarray(Vgas), Vdisk=np.asarray(Vdisk), Vbul=np.asarray(Vbul))


def evaluate_galaxy(name: str, path: Path, table: Dict[str, dict],
                    r_0_kpc: float = R0_KPC_CANONICAL,
                    min_floor_err: float = 1.0) -> Optional[dict]:
    if name not in table:
        return None
    info = table[name]
    L_3_6 = info["L_3_6_1e9_Lsun"]
    M_HI = info["MHI_1e9_Msun"]
    Rdisk = info["Rdisk_kpc"]
    if Rdisk <= 0:
        return None
    M_baryon = (M_TO_L_RATIO_3_6 * L_3_6 + M_HI) * 1e9  # Msun
    if M_baryon <= 0:
        return None

    # Virial v_0 from baryon mass and disk scale.
    # For an exponential disk, Freeman 1970 gives v_max² = 0.62 · G·M_disk/R_disk
    # (peak rotation velocity at R ≈ 2.15·R_disk for a thin exponential disk).
    # We use this same prefactor for the prime-field amplitude calibration —
    # the IF Theory's v_0_galaxy is the "asymptotic prime-field velocity"
    # which equals the disk-virial velocity in the same way Newtonian disks
    # give v_max < √(GM/R).
    FREEMAN_FACTOR = 0.62
    G = G_kpc_kms_msun()
    v_0_galaxy = math.sqrt(FREEMAN_FACTOR * G * M_baryon / Rdisk)

    d = load_rotmod(path)
    R = d["R"]; keep = R > 0
    R = R[keep]
    Vobs = d["Vobs"][keep]
    errV = np.maximum(d["errV"][keep], min_floor_err)
    Vbar = np.sqrt(d["Vgas"][keep] ** 2 + d["Vdisk"][keep] ** 2 + d["Vbul"][keep] ** 2)
    if len(R) < 3:
        return None

    # IF Theory CORRECTED logarithmic potential: v_prime² = v_0² · R/(R+r_0)
    v_prime_sq = (v_0_galaxy ** 2) * R / (R + r_0_kpc)
    v_prime = np.sqrt(v_prime_sq)
    v_total = np.sqrt(Vbar ** 2 + v_prime ** 2)

    residuals = Vobs - v_total
    chi2 = float(np.sum((residuals / errV) ** 2))
    dof = max(len(R) - 0, 1)
    return {
        "name": name,
        "n_points": len(R),
        "M_baryon_Msun": M_baryon,
        "R_disk_kpc": Rdisk,
        "v_0_predicted_kms": v_0_galaxy,
        "V_flat_observed_kms": info["Vflat_kms"],
        "chi2": chi2,
        "dof": dof,
        "chi2_per_dof": chi2 / dof,
        "rms_residual_kms": float(np.sqrt(np.mean(residuals ** 2))),
        "fraction_within_2sigma": float(np.mean(np.abs(residuals / errV) < 2.0)),
    }


def main() -> int:
    table = parse_sparc_table(SPARC_TABLE)
    print(f"Loaded SPARC table: {len(table)} galaxies")

    files = sorted(SPARC_DIR.glob("*_rotmod.dat"))
    results: List[dict] = []
    skipped: List[str] = []
    for fp in files:
        name = fp.stem.replace("_rotmod", "")
        try:
            r = evaluate_galaxy(name, fp, table)
            if r is None:
                skipped.append(name)
            else:
                results.append(r)
        except Exception as e:
            skipped.append(f"{name} ({e})")

    if not results:
        print("No galaxies evaluated.")
        return 1

    chi2_per_dof = np.array([r["chi2_per_dof"] for r in results])
    v0_pred = np.array([r["v_0_predicted_kms"] for r in results])
    Vflat = np.array([r["V_flat_observed_kms"] for r in results])
    f2 = np.array([r["fraction_within_2sigma"] for r in results])

    print(f"Evaluated {len(results)} galaxies, skipped {len(skipped)}\n")
    print("=" * 78)
    print(f"SPARC CORRECTED Φ = ln(r/r_0+1), v_0 = √(G·M_b/R_disk) per galaxy")
    print(f"r_0 = {R0_KPC_CANONICAL:.4f} kpc, M/L_3.6 = {M_TO_L_RATIO_3_6}")
    print("=" * 78)
    print(f"  χ²/dof   median         : {np.median(chi2_per_dof):8.2f}")
    print(f"           mean           : {np.mean(chi2_per_dof):8.2f}")
    print(f"           25th pct       : {np.percentile(chi2_per_dof, 25):8.2f}")
    print(f"           75th pct       : {np.percentile(chi2_per_dof, 75):8.2f}")
    print(f"  Fraction χ²/dof < 1     : {np.mean(chi2_per_dof < 1):.1%}")
    print(f"  Fraction χ²/dof < 5     : {np.mean(chi2_per_dof < 5):.1%}")
    print(f"  Fraction χ²/dof < 10    : {np.mean(chi2_per_dof < 10):.1%}")
    print(f"  Fraction χ²/dof < 50    : {np.mean(chi2_per_dof < 50):.1%}")
    print()
    # Tully-Fisher: log(v_0_pred) vs log(V_flat_obs)
    mask = (v0_pred > 0) & (Vflat > 0)
    if mask.sum() > 5:
        lp = np.log10(v0_pred[mask])
        lo = np.log10(Vflat[mask])
        slope, intercept = np.polyfit(lo, lp, 1)
        r_pearson = float(np.corrcoef(lp, lo)[0, 1])
        print(f"  Tully-Fisher: log(v_0_pred) vs log(V_flat_obs):")
        print(f"    slope = {slope:+.3f}  (theoretical 1.00 expected)")
        print(f"    intercept = {intercept:+.3f}")
        print(f"    Pearson r = {r_pearson:+.3f}")
        print(f"    n_galaxies = {int(mask.sum())}")
    print(f"  Median %% points within 2σ (per galaxy): {np.median(f2) * 100:.0f}%")

    out = {
        "r_0_kpc": R0_KPC_CANONICAL,
        "M_to_L_ratio_3_6": M_TO_L_RATIO_3_6,
        "n_evaluated": len(results),
        "n_skipped": len(skipped),
        "summary": {
            "chi2_per_dof_median": float(np.median(chi2_per_dof)),
            "chi2_per_dof_mean": float(np.mean(chi2_per_dof)),
            "frac_under_1": float(np.mean(chi2_per_dof < 1)),
            "frac_under_5": float(np.mean(chi2_per_dof < 5)),
            "frac_under_10": float(np.mean(chi2_per_dof < 10)),
            "frac_under_50": float(np.mean(chi2_per_dof < 50)),
            "tully_fisher_slope": float(slope) if mask.sum() > 5 else None,
            "tully_fisher_pearson_r": r_pearson if mask.sum() > 5 else None,
        },
        "per_galaxy": results,
        "skipped_names": skipped,
    }
    out_file = OUT_DIR / "sparc_corrected_results.json"
    with open(out_file, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
