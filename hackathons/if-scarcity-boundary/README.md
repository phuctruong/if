# IF Scarcity Boundary — does tracking emerge where direction decides survival?

> Auth: 65537 · Master Equation: Purpose × Evidence × Love
> Started 2026-07-18 (loop iteration 2), after `if-resource-tracking` returned a tight
> null: under abundance (inflow=12) emergent movers are ballistic — nothing selects on
> direction. This hackathon tests the successor hypothesis: **resource-tracking can only
> be selected near the scarcity boundary**, where movers heading away from energy starve.

## The decisive question

At the lowest inflow that still grows movers (the scarcity edge), is mover motion biased
toward the resource source relative to a gradient-free placebo? A positive here — where
abundance gave a null — would be the program's first emergent selection-produces-agency
result: same rules, direction coupling appearing exactly where the energy economics make
direction matter.

## Pre-committed protocol (FROZEN before any run; two stages, separately frozen)

### Stage S1 — find the boundary (selection rule frozen; results exploratory)

- Sweep inflow ∈ **{0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0}** at the sealed
  regime's other parameters (`B3/S23, e_birth=0.25, e_maint=0.01, σ=40, ρ=0.15`),
  seeds 7–14 declared (8 per inflow), 600 steps.
- **Frozen selection rule**: inflow\* = the smallest swept inflow whose mean emergent
  mobile tracks per run ≥ 0.5 (the D4 criterion of `if-mobility-search`).
- If inflow\* = 12 (only the known regime passes), S2 is declared MOOT and the negative
  is sealed; no re-sweep without a logged amendment.

### Stage S2 — the confirmatory τ test at inflow\* (primary, frozen)

- Identical statistic and machinery as `if-resource-tracking` (τ = 4-step-window cosine
  to the analytic source; ≥10 valid windows per track): GRADIENT σ=40 vs PLACEBO σ=10⁶,
  both at inflow\*. **Fresh seed roster: 33–96 (64 per arm), declared here.**
- **Primary verdict**: Welch t between arm τ populations. t > +2 → tracking at scarcity
  (with the abundance null as the built-in contrast) · t < −2 → anti-tracking ·
  |t| ≤ 2 → undecided. Minimum 20 qualifying tracks per arm else VOID (extension only
  by logged declaration).
- **Secondary (declared, exploratory label)**: within the gradient arm, Pearson r
  between per-track τ and track lifetime (do source-pointing movers live longer?),
  reported with the placebo arm's r as contrast. No verdict rides on it.

## Rubric (100)

| | Pts | |
|---|---:|---|
| S0 Prereg committed before any run | 20 | git history is the proof |
| S1 Boundary sweep on declared grid + frozen rule applied | 20 | evidence JSON |
| S2 τ test at inflow\*, full fresh roster | 30 | evidence JSON |
| S3 Frozen verdict honored; MOOT/VOID honored if hit | 20 | |
| S4 Canon + HANDOFF + verify GREEN | 10 | |

Persona gates: Conway (rules untouched; scarcity is an energy knob, not an agency term) ·
Feynman (if τ>0 appears, check it is not pure survivorship geometry — the placebo
contrast and birth-distance context are his gate) · Noether (ledger holds at every
inflow) · Shannon (same declared estimator as iteration 1, unchanged) · Popper (selection
rule for inflow\* frozen before the sweep; fresh seeds for S2; no threshold moves) ·
Phuc/65537. Both outcomes seal.
