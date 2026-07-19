# if-recharge-threshold — Scoreboard

> Auth: 65537 · SEALED 2026-07-19. Prereg `555a392`; amendment #1 (instrument only)
> logged and committed before the interpreted run. Layer: SCIENCE. Every number below
> is a property of a toy model with invented constants — see §Scope.

| Track | Pts | Earned | |
|---|---:|---:|---|
| Prereg before any run | 20 | 20 | `555a392` |
| Ledger + PERPETUAL_RECHARGE enforced | 20 | 20 | F monotonically decreasing in every run; ledger exact to 1e-6; **no run showed recharge** |
| Run 1 VOID honored, defects fixed not tuned | 20 | 20 | zero-variance seeds + saturated capture caught before verdicts taken |
| Q1–Q4 verdicts applied as frozen | 25 | 25 | incl. reporting that Q2's "THRESHOLD" is a seeding artifact, not a density transition |
| Canon + verify | 15 | 15 | |
| **TOTAL** | **100** | **100** | |

## The frozen verdicts, and what they honestly mean

**Q1 — the single conscious being.** ΔΦ = +0.131 (20σ). But the mechanism is not
"one agent rescues the system": with 2 reflective agents in 100, reflection **sweeps to
fixation** (refl_life 0.92). Reflection is heritable and teachable here, so the
population converts. *My pre-registered prediction (a lone reflective agent sits in the
parasite band) was WRONG at these parameters — logged.*

**Q2 — threshold: technically fired, honestly a seeding effect.** The largest jump is
between ρ₀ = 0 and ρ₀ = 0.02 — i.e. between *none* and *any*. From 0.02 → 1.0 the curve
is smooth and saturating (0.719 → 0.801). **The real threshold is existence (N ≥ 1),
not a critical density.** Claiming a phase transition in density would be an overclaim
and is refused.

**Q3 — teaching lowers the bar.** ρ₉₀ falls 0.10 → 0.05 (halved). At ρ₀ = 0.02,
teaching raises lifetime-average reflective fraction 0.52 → 0.92. Transmission does not
change the ceiling; it changes how fast and from how small a seed the ceiling is reached.

**Q4 — no parasite band at the frozen parameters.** See the exploratory sweep below for
where it *does* appear.

## The two results that matter most

**1. Reflection buys intensity, not duration.** Φ rises 0.588 → 0.801 (+36%) while
t_end *falls* 368 → 345 (−6%). The ordered phase is not extended — more of the discharge
is routed through living structure before it ends. **Burns brighter, slightly shorter.**

**2. Selection does not protect the system** (exploratory cost sweep, ρ₀ = 0.2, labeled
non-pre-registered, no verdict weight): reflection pays until its upkeep premium reaches
≈ 6–7× the naive metabolic cost; at 8× and 11× it is frankly parasitic (Φ 0.52, 0.43 vs
0.588 baseline; t_end collapses 410 → 157). **And it still sweeps to fixation there**
(refl_life 0.92–0.95). Reflection wins the *individual* competition for flux while
making the *collective* outcome worse — it is individually selected and collectively
destructive above the crossover, and nothing in the dynamics can see the difference.

## Scope (binding — violating this is a layer leak)

Invented constants, mean-field toy, no space, no death by anything but starvation.
Whether reflection pays *at all* is a parameter choice (CAP_R/M_R), **not a finding**.
What the model can support is the *shape* of the answer: fixation from a single seed;
transmission lowering the seed requirement; intensity-not-duration; and the existence of
a cost crossover beyond which a trait spreads while degrading its own substrate. Nothing
here says anything about the actual universe. The meaning reading lives in
`canon/30-meaning/`, never in this file.
