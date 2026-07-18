# IF Resource Tracking — do emergent movers move *toward* energy?

> Auth: 65537 · Master Equation: Purpose × Evidence × Love
> Started 2026-07-18 after `if-mobility-search` sealed. That hackathon found the mobility
> regime and left the mover audit UNDECIDED, with the diagnosis that raw regional harvest
> is confounded by scramble-ignited growth. This hackathon asks a cleaner question that
> needs **no scramble fork at all**.

## The decisive question

In the sealed mobility regime, is the movement of emergent mobile structures **biased
toward the resource source**, relative to an otherwise-identical universe with no
resource gradient? A yes means the universe grows structures whose *motion is coupled to
where the energy is* — resource-tracking without any designed sensor, the property the
original handoff asked for. Mechanism (selection vs sensing) is explicitly NOT
adjudicated here; that is the next question if τ > 0.

## Pre-committed protocol (FROZEN before any run; amendments logged, never silent)

- **Universe**: the sealed mobility regime `B3/S23, e_birth=0.25, e_maint=0.01,
  inflow=12, ρ=0.15`, 600 steps, movers per D2/D3 of `if-mobility-search`.
- **Arms**: GRADIENT σ=40 vs PLACEBO σ=10⁶ (≈uniform inflow; same drifting source
  point exists formally, so the identical statistic is computable). Same seed roster
  both arms: **seeds 1–32, declared here, verdict on the full roster.**
- **Statistic** (per mobile track): τ = mean over 4-step windows (glider period) of
  cos(angle between window displacement and the minimal-image direction from current
  COM to the analytic source position). Windows with |displacement| = 0 or source
  distance < 3 cells are skipped. A track needs ≥ 10 valid windows to count.
- **Primary verdict**: Welch t between the gradient-arm and placebo-arm τ populations.
  t > +2 → movers track resource · t < −2 → movers anti-track · |t| ≤ 2 → undecided.
  Minimum 20 qualifying tracks per arm, else VOID (extend roster by declaration only).
- **Instrument control** (must pass before interpretation): synthetic straight-line
  paths toward / away from the source must score τ ≈ +1 / −1.

## Rubric (100)

| | Pts | |
|---|---:|---|
| T0 Prereg committed before any run | 20 | git history is the proof |
| T1 Instrument control passes | 15 | synthetic ±1 check |
| T2 Both arms run on the full declared roster | 30 | evidence JSON |
| T3 Frozen verdict applied, no upgrade/downgrade | 25 | UNDECIDED stays undecided |
| T4 Canon + HANDOFF + verify GREEN | 10 | |

Persona gates: Conway (no agency in rules; placebo arm truly gradient-free),
Feynman (τ not an artifact of birth-location bias — check: report birthplace–source
distance distributions per arm as exploratory context), Noether (ledger holds),
Shannon (statistic is a declared estimator), Popper (roster + thresholds frozen),
Phuc/65537. Both outcomes seal.
