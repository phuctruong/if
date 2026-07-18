# The Thermodynamic Parasite Band

## Two Break-Even Thresholds Separate Information That Is Causally Load-Bearing From Information That Pays For Itself

**Author:** Phuc Vinh Truong
**Series:** IF Theory Canonical Papers
**Paper:** 16 (positive result — new)
**Layer:** SCIENCE
**Status:** Primary positive result, 2026-07-18. Observed before it was derived.
**Evidence:** `notebooks/04_if_causal_work_threshold.ipynb`, `notebooks/04e_update_law_ablation.ipynb`, `notebooks/04h_eta_star_three_families.ipynb`

---

## Abstract

An agent's internal memory can be **causally load-bearing and simultaneously not worth
having**. We report a robust dissociation between two break-even criteria that the
agency literature routinely conflates:

- the **ablation criterion** Π_A — scrambling the memory measurably reduces extracted
  work relative to keeping it;
- the **competitive criterion** Π_C — the memorying agent outperforms the *optimal
  memoryless agent* in the same environment.

These cross at different points. Between them lies the **thermodynamic parasite band**:
a regime where destroying an agent's information demonstrably costs it work (so the
information is doing causal work) while the agent would nonetheless be better off never
having had the memory at all. In our reference environment the bands are wide — ablation
break-even at environment predictability p\*₁ ≈ 0.64, competitive break-even at
p\*₂ ≈ 0.995.

The result was **observed first and derived afterward**: the band's existence follows
necessarily from the break-even inequality, because Π_A charges only the model cost while
Π_C additionally charges full apparatus overhead and credits the memoryless twin's free
harvest of instantaneous sensory information. It replicated in three independent contexts,
including one where a *component* of a single agent was a parasite while another component
of the same agent was not.

The practical consequence: **an ablation experiment alone cannot establish that a
mechanism is adaptive.** Ablation establishes causal participation. Only comparison
against an optimal simpler competitor establishes that the mechanism earns its keep.

---

## 1. The distinction

Let W_intact be the work extracted by an agent with an internal model, W_scr the work
extracted when that model is interventionally scrambled (marginal-preserving), W_react the
work extracted by the optimal memoryless policy on the identical environment and sensor,
C_model the declared cost of maintaining the model, and ΔC_full the full apparatus cost
difference against the memoryless twin. Then

\[
\Pi_A=\frac{W_{\mathrm{intact}}-W_{\mathrm{scr}}}{C_{\mathrm{model}}},
\qquad
\Pi_C=\frac{W_{\mathrm{intact}}-W_{\mathrm{react}}}{\Delta C_{\mathrm{full}}}.
\]

Π_A > 1 says: *this information is doing causal work in excess of its own maintenance.*
Π_C > 1 says: *having this apparatus at all beats not having it.* The literature treats
these as the same question. They are not.

## 2. Why the band must exist (derivation)

Π_A ignores two quantities that Π_C charges:

1. **The reactive free harvest.** A memoryless agent still extracts predictive value from
   its instantaneous sensor: I_react = I(Y_t ; X_{t+τ}) ≥ 0. Π_A never subtracts it, so
   Π_A credits the memory with work the twin would have obtained for free.
2. **Non-memory overhead.** Sensing and control costs incurred *because* the agent carries
   a model, beyond the model's own storage cost.

Hence

\[
\text{band width}=\beta\,C_{\mathrm{overhead}}+\bigl[I_{\mathrm{react}}-\text{nostalgia}(p^*_1)\bigr]>0
\]

whenever the memoryless twin extracts anything at all and non-memory overhead is nonzero.
**The band is not an artifact of any particular environment; it is a structural consequence
of charging the full apparatus rather than the memory alone.**

## 3. Evidence

### 3.1 Primary observation (drift-gradient ring world)

A deterministic ring world with a drifting resource peak, seeded, with explicit per-cycle
debits for sensing, memory, and movement. Sweeping environment predictability p:

| p | W_intact | W_scrambled | W_reactive | Π_A | memory advantage |
|---|---|---|---|---|---|
| 0.635 | 3373.5 | 3270.7 | 3440.0 | **1.28** | **−66.5** |
| 0.770 | 3386.0 | 3186.7 | 3440.0 | 2.49 | −54.0 |
| 0.905 | 3370.2 | 2999.5 | 3440.0 | 4.63 | −69.8 |
| 0.995 | 3595.2 | 2869.0 | 3440.0 | 9.08 | **+155.2** |

Across the whole shaded region the agent's memory is **causally load-bearing by a wide
margin** (scrambling costs it hundreds of work units, Π_A up to 4.8) while the agent is
**simultaneously losing to a memoryless competitor**. Only at p ≈ 0.995 do the two
criteria finally agree.

Crucially, the derivation in §2 did not exist when this sweep was run. The simulation
produced the dissociation; the inequality then showed it was obligatory.

### 3.2 Replication within a single agent (switching-law world)

In an agent carrying two distinct memories — a state estimate and a model of the
environment's update law — the two components received *opposite* audit verdicts under the
same protocol: ablating the rule-model selectively destroyed post-switch recovery (6σ),
while the state-smoother component was outperformed by using the raw observation directly.
One agent, two memories, opposite verdicts. Component-wise causal-work auditing is
precisely the instrument that separates them.

*(Declared: the smoother's parasitism in that experiment is confounded with a fixed-gain
configuration that is provably suboptimal in a switching world; the claim is held pending
a per-regime gain optimization. The dissociation between components is not affected.)*

### 3.3 Third context (three-family study)

The two-threshold structure appeared in all three substrate families examined in the
companion study — a lattice forager, a linear-Gaussian controller, and a run-and-tumble
chemotactic swimmer — despite those families sharing no mechanism. **The parasite band
survived the same experiment that falsified the program's universality claim.** It does
not depend on any invariant.

## 4. What this does and does not license

**Licensed.** Ablation establishes *causal participation*. That is a real and useful
finding about a mechanism. It is the correct conclusion to draw from a knockout.

**Not licensed.** Ablation does *not* establish that a mechanism is adaptive, efficient,
or worth its cost. A mechanism can be knocked out with large measurable effect and still
be a net thermodynamic liability to its bearer. Any claim of the form "we ablated X and
performance dropped, therefore X is adaptive" requires the second comparison against an
optimal simpler competitor.

**Scope.** These are simulation results with declared cost parameters. They demonstrate
that the band exists and is structurally obligatory under the stated accounting; they do
not measure the band's width in any biological system.

## 5. Falsifiers

1. A family in which Π_A and Π_C cross at the same point across a full predictability
   sweep, with nonzero reactive harvest and nonzero non-memory overhead, would contradict
   §2's derivation.
2. A demonstration that the reactive twin used here is not the optimal memoryless policy
   (the twin-optimality requirement) would invalidate the competitive threshold's location
   — though not the existence of two distinct criteria.
3. Evidence that the band closes as cost parameters approach any physically motivated
   limit would restrict the result's relevance to artificial cost regimes.

## 6. Relationship to the program's other results

This paper is deliberately narrow. It does **not** claim a universal constant — that claim
was tested and falsified (`P15-falsification-of-universality.md`). It does not require the
information denominator that P15 showed to be non-portable, since both criteria here use
declared cost denominators. It inherits one caveat from the clean-channel experiment
(notebook 04g v2): Π_A also absorbs memory-state-dependent energetics where a substrate has
them, so the band's *width* is substrate-contaminable even though its *existence* is not.

The band is what survived when the larger claims did not, and it survived because it was
the most modest thing we measured.

---

**Cross-references:** `canon/00-foundations/03-causal-work-principle.md` ·
`canon/00-foundations/04-break-even-theorem.md` (derivation) ·
`canon/papers/P05-agency-threshold.md` · `canon/papers/P15-falsification-of-universality.md`
