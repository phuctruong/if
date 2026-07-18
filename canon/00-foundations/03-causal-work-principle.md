# The IF Causal-Work Principle

> Layer: SCIENCE. The flagship claim. Distilled from Paper 2 (`../extracted/paper-02-extracted.md`); consensus-strongest claim per BOTH ChatGPT arc and Gemini panel round 1.

## Statement

Internally maintained information is **constitutive of agency** if and only if
interventionally preserving it yields more net useful work (or future
viability) than the complete cost of maintaining it:

```
W_C = W_intact − W_scrambled − C_model          (Turing's ablation form)

Π_A = (W_intact − W_scrambled) / (C_sensing + C_memory + C_prediction + C_control)

Agency ⟺ Π_A > 1, robustly across environments
```

where `W_scrambled` is measured under matched interventions that **erase,
scramble, temporally displace, or falsify** selected internal information
while preserving relevant physical and statistical properties (marginals
preserved — scrambling must not smuggle in a thermodynamic difference).

## Why intervention, not correlation

Correlation ≠ agency: information may be stored but causally idle, predictive
but unused, or useful but costlier than its benefit ("a net-negative
thermodynamic parasite" — Gemini). Only the ablation delta, with ALL costs on
the ledger, separates:

```
passive dissipative structure → reactive controller → predictive agent → reflective agent
```

## Positioning vs prior art (the exact daylight)

| Theory | What it says | Where IF differs |
|---|---|---|
| Kolchinsky–Wolpert semantic info | correlations are meaningful if removing them hurts viability | IF adds the FULL cost side: info can be KW-meaningful yet IF-negative (parasite). **Divergence cases are the key experiment.** |
| Friston FEP | systems minimize surprise | IF doesn't assume the objective; it audits whether model-keeping pays. If FEP-optimal agents are always Π_A-positive, IF reduces to FEP — that's a falsifier to test, not to fear. |
| Hoel causal emergence | macro can out-cause micro | IF is about cost-complete work delta, not scale comparison; complementary. |
| England dissipative adaptation | structure absorbs/dissipates | England's structures are blind; IF marks the line where they stop being blind. |

## The phase-transition hypothesis (IF-H1)

Claim: Π_A crossing 1 is not just a bookkeeping line — under broad conditions
it behaves like a phase boundary: discontinuity in adaptive performance,
critical slowing, hysteresis, universality across rule families.

**Kill condition (Gemini round 1, adopted):** if net viability scales smoothly
and linearly with information capacity across all tested rule families, the
phase-transition claim is FALSE — agency is a gradient, not a state of matter.
Record in kill log; the principle survives as a measurement tool even if the
transition dies.

## Experimental protocol (notebook `04_if_causal_work_threshold.ipynb`)

1. Environments: resource gradients, varying predictability/volatility.
2. Agents: emergent bounded structures, NOT predeclared (Conway gate).
3. Interventions: erase / scramble-preserving-marginals / time-shift / falsify.
4. Sweep: memory cost, sensor cost, model depth, perturbation rate.
5. Measure: Π_A landscape; look for discontinuity, hysteresis, universality.
6. Ledger integrity: property-based conservation tests auto-fail leaky runs (Noether gate).

Companion divergence notebooks: KW-vs-IF disagreement cases; FEP cost audit;
finite-memory Maxwell demon (Landauer ablation).
