# The Informational Battery

> Layer: SCIENCE. Distilled from Paper 1 (`../extracted/paper-01-extracted.md`) + Gemini panel round 1 (Landauer-coupling critique).

## Definition

The **informational battery** of a bounded system is:

```
B(system) = physically accessible nonequilibrium capacity
          + the structured correlations that determine HOW that capacity can be used
```

It is NOT: a new substance, energy created by information, Vopson-style
mass-energy-information equivalence, or a metaphor. It is an accounting object.

## The three ledgers

| Ledger | Quantity | Units | Estimator obligations |
|---|---|---|---|
| Energy | free energy / extractable work | joules | conservation to ε under property tests |
| Thermodynamic entropy | physical macrostate entropy | J/K | defined macrostate partition, stated coarse-graining |
| Information | structured correlations (mutual information between subsystem states) | bits | declared random variables, measure, null distribution |

**Coupled, not walled off (panel round 1 fix).** "Never add bits to joules"
means no ledger *substitutes* for another — it does not mean they don't
interact. Landauer's principle (E ≥ kT ln 2 per erased bit) is precisely a
**coupling term between ledgers**: logically irreversible operations on the
information ledger force minimum debits on the energy/entropy ledgers. The
battery formalism must carry these coupling terms explicitly; a model whose
ledgers never touch is thermodynamically unsound, and a model that free-mixes
them is numerology. The narrow bridge between those two failure modes IS the
theory.

## What "charge" and "recharge" mean (de-mystified)

- **Charge** = capacity + correlations available for future useful work.
- **Discharge** = irreversible use of capacity (entropy exported, work extracted).
- **Recharge** = local battery increase, ALWAYS paid for: ΔS_agent < 0 requires
  ΔS_environment ≥ −ΔS_agent plus explicit energy import. Anything else is
  `PERPETUAL_RECHARGE` (forbidden).

## Why this object is worth defining (the novelty claim, kept honest)

Prior art each holds a piece: Prigogine (nonequilibrium capacity sustains
structure), Landauer (information ops cost energy), Kolchinsky–Wolpert
(viability-relevant correlations), England (dissipative adaptation). The
battery's contribution is **the unified audit**: one object whose three
coupled ledgers let you ask, quantitatively, *whether a given correlation is
worth its keep* — the question the Causal-Work Principle
(`03-causal-work-principle.md`) answers by intervention.

## Falsifiers

1. Maxwell-demon simulation with finite memory: if the three-ledger accounting
   can be made to violate the second law under deterministic ablation
   (erasure without Landauer debit goes undetected), the formalism is broken.
2. If battery values depend on arbitrary coarse-graining choices with no
   stable window, the object is not well-defined.
3. If no simulation can distinguish battery-rich from battery-poor states by
   *future* behavior (capacity is epiphenomenal), the object is useless.
