# The IF Break-Even Theorem (candidate) — IF's kT ln 2

> Layer: SCIENCE. Constructed 2026-07-18 by frontier panel round 2 (Claude Fable 5 High
> primary, Gemini convergent — full texts `../extracted/frontier-panel/round2-*.md`),
> adopted by the Founding Panel. Status: CANDIDATE — proof program open.

## Setup

Discrete-time agent in a partially observable environment: environment state X_t,
sensor Y_t, persistent internal state M_t, action A_t, temperature T, β = 1/kT.
Per cycle at steady state over horizon τ:

```
I_mem   = I(M_t ; X_t)      total stored information about the world
I_pred  = I(M_t ; X_{t+τ})  information the memory carries about the horizon the action couples to
I_react = I(Y_t ; X_{t+τ})  predictive information the memoryless twin gets FREE from the current sensor
I_use   = transfer entropy M → work-coordinate (action-relevant information actually driving extraction)
```

## The inequality (Sagawa–Ueda ceiling × Still floor × twin difference)

```
ΔW_net ≤ kT·[I_pred − I_react]  −  kT·(I_mem − I_pred)  −  β⁻¹-scaled C_overhead
                                    └── nostalgia floor ──┘
```

**Break-even (candidate theorem):** internally maintained information is
net-work-productive iff

```
I_pred − I_react  >  (I_mem − I_pred)  +  β·C_overhead
```

*The predictive surplus over the reactive twin must exceed the nostalgia
(non-predictive stored bits) plus the dimensionless non-memory overhead.*
Equality — the **IF kT ln 2 line** — holds when every sub-bound saturates:
zero nostalgia (store only what predicts), reversible sensing, quasistatic control.

**Nostalgia** (I_mem − I_pred) is the self-deception term of the battery ledger:
stored bits with no predictive power, each one a pure thermodynamic liability
(Still et al. floor). A reflective agent's first duty is deleting its own nostalgia.

## The parasite band is a theorem, not a bug

The ablation criterion Π_A charges only C_model and ignores both I_react and
C_overhead; the competitive criterion charges everything and credits the twin's
free harvest. Therefore competitive break-even occurs at strictly higher
predictability than ablation break-even whenever the reactive twin extracts
anything:

```
band width = β·C_overhead + [I_react − nostalgia(p*₁)] > 0
```

Notebook 04 v0.1 observed exactly this (p*₁ = 0.64, p*₂ = 0.995) BEFORE the
derivation existed. Simulation surprised; math then showed it was necessary.
**This is the program's first original result.**

## The apparatus-boundary normalization (kills "the measure is a knob")

Never draw an absolute boundary. Define everything as differences against the
**canonical twin A₀** — the work-maximizing MEMORYLESS policy on the identical
environment and sensor (the POMDP collapsed to the MDP on current observation,
unique up to ties):

```
ΔC ≡ C[A] − C[A₀]        ΔW ≡ W[A] − W[A₀]
```

Everything shared (outer wall, reservoirs, actuators) cancels in the difference;
what remains is exactly "the persistent state + the compute that reads/writes it."
The guardrail: A₀ must be *optimal* — a crippled reference is a computable,
detectable violation. "Where is the boundary?" (unanswerable) becomes
"solve this MDP" (a definite computation).

## The dimensionless invariant Θ* (rung-274177 seal condition)

Two ratios, two thresholds:

```
Π_A = kT[I_pred − I_scr] / C_model     (internal causal efficiency)   = 1 at p*₁
Π_C = kT[I_pred − I_react] / ΔC_full   (architectural efficiency)     = 1 at p*₂
```

Raw p is a bad coordinate (predictive information diverges as p → 1). The
universality candidate is:

```
Θ* ≡ Π_A evaluated where Π_C = 1
```

— the fraction of full apparatus cost concentrated in belief-maintenance, at
competitive break-even. Dimensionless, boundary-free.

**STATUS UPDATE (2026-07-18, notebook 04f + cost control): raw Θ* is FALSIFIED
as the universal constant** — it scales ~1/C_MEMORY and the ring/Kalman families
are statistically distinguishable (3.5–9.6σ) at every cost level once crossing
statistics tighten. What survives, and points to the refined candidate: the two
families track in lockstep across a 3× cost range (5–15% apart), and the clean
work-per-bit ratio (04g) is stable across environments. **IF-H1 (restated v3):
some cost-rescaled invariant — leading candidates Θ*·C_model and the clean
work-per-bit coefficient — takes the same value across ≥3 unrelated rule
families.** The falsifier killed the naive form and pointed at the better one;
that is the mechanism working.

## The proof program (what exists, what's missing)

| Piece | Source | Status |
|---|---|---|
| Work ceiling ΔW ≤ kT·ΔI_use | Sagawa–Ueda feedback fluctuation theorems | off the shelf |
| Maintenance floor C_mem ≥ kT·(I_mem − I_pred) | Still–Sivak–Crooks–Bialek thermodynamics of prediction | off the shelf |
| Steady-state rates, İ_use over horizon | Barato–Seifert transducer/information-flow bounds | off the shelf |
| Interventional semantics (scramble/erase) | Kolchinsky–Wolpert viability | off the shelf |
| **DPI-for-interventions lemma: R ≤ 0** — W_intact − W_scr = kT[ΔI_use] + R with R (off-shell work) provably non-positive: *scrambling never creates usable work* | — | **MISSING — the theorem to prove** |
| **Signed usable-information functional** J = kT·Cov(belief-driven action, true work gradient) — can go NEGATIVE for falsified belief (ordinary MI cannot); prove J squeezed between SU ceiling and Still floor | — | **MISSING — the one genuinely new mathematical object in the program** |

## Refutation notebooks (attack the load-bearing parts first)

1. **Θ* scatter test** (`04f_kalman_theta_star.ipynb`): LQG/Kalman family with
   per-bit metabolic debit; same protocol; compare Θ*_Kalman vs Θ*_ring against
   bootstrap error. Scatter beyond error → IF-H1 dead, T1 world-specific.
   (Agreement obliges a third family before any universality claim.)
2. **Ratchet R-test** (`04g_scramble_ratchet.ipynb`): engineer a world where the
   scramble's dumped heat drives a ratchet doing useful work; measure
   R = (W_intact − W_scr) − kT·ΔI_use across seeds. Robust R > 0 → the lemma is
   false, Π_A over-counts, and "ablated work = information content" collapses.
   *This is the notebook to WANT to run — it attacks the spine, not the decorations.*

## Standing flag (Claude R2, adopted)

This entire construction is agency-thermodynamics. Nothing in it supports the
cosmology branch, and the two must not share a public corpus near publication —
a refuted ratchet result must never be usable against the cosmology preregistration,
or vice versa.
