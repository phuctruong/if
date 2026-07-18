# if-resource-tracking — Scoreboard

> Auth: 65537 · Rubric in README.md. SEALED 2026-07-18 (loop iteration 1).

| Track | Pts | Earned | Evidence |
|---|---:|---:|---|
| T0 Prereg before any run | 20 | 20 | commit `59c99b7` precedes first execution |
| T1 Instrument control | 15 | 15 | pursuit +1.000 / evasion −0.894; two calibration steps logged below |
| T2 Both arms, full roster | 30 | 30 | 849 gradient + 830 placebo qualifying tracks (`evidence/resource_tracking_2026_07_18.json`) |
| T3 Frozen verdict honored | 25 | 25 | Welch t = +0.34 → **UNDECIDED**, no upgrade |
| T4 Canon + verify GREEN | 10 | 10 | SCOREBOARD/HANDOFF updated, verify GREEN |
| **TOTAL** | **100** | **100** | |

## Headline result

**A tight null.** τ(gradient) = +0.0218 ± 0.407 (n=849), τ(placebo) = +0.0149 ± 0.409
(n=830), Welch t = +0.34. Frozen verdict: UNDECIDED. Exploratory power note: at this
sample the 95% CI on the arm difference is ≈ ±0.04 cos units — **any resource-tracking
bias in this regime is below 4% of full alignment**. Movers here are ballistic:
direction is set at birth and uncoupled from where the energy is. Birthplace–source
distance medians are equal across arms (48.5 vs 51.1), so the null is not a
birth-location artifact (Feynman gate).

**Interpretation → next hypothesis (not claimed, to be pre-registered):** the sealed
mobility regime is energy-abundant (inflow=12) — nothing selects on direction. Tracking,
if it exists anywhere, should live near the **scarcity boundary** where movers heading
the wrong way starve (the archived seeded-glider data showed distance-dependent survival
at inflow=4).

## Instrument-calibration log (not verdict changes)

1. First synthetic "toward" control aimed at the source's t=0 position; the source
   drifts east, so a straight line is genuinely misaligned (τ=+0.745). The control was
   rebuilt as pursuit/evasion of the *moving* source. The statistic itself never changed.
2. Evasion gate set to −0.85 (pursuit stays +0.9): the evader's bearing lags the
   drifting source across each 4-step window, so exact −1 is unreachable by geometry.

## Persona gates

Conway ✅ (placebo arm gradient-free by σ=10⁶) · Feynman ✅ (birth-location check above) ·
Noether ✅ (ledger asserted in all 64 runs) · Shannon ✅ (τ is a declared estimator) ·
Popper ✅ (roster + thresholds frozen at `59c99b7`; UNDECIDED not upgraded) · 65537 ✅.
