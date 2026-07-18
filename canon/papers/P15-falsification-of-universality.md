# The Falsification of Causal-Work Universality

## Why No Cost-Free Dimensionless Invariant Survives Three Substrates, and What Remains of the Agency Threshold

**Author:** Phuc Vinh Truong
**Series:** IF Theory Canonical Papers
**Paper:** 15 (negative result — new, not a revision of an extracted draft)
**Layer:** SCIENCE
**Status:** Primary result, 2026-07-18. Pre-registered stop rule honored.
**Evidence:** `notebooks/04h_eta_star_three_families.ipynb`, `scripts/{eta_star,upsilon}.py`

---

## Abstract

IF Theory's central near-term claim (IF-H1) was that a dimensionless quantity built
from the causal-work ratios — the fraction of apparatus cost concentrated in
belief-maintenance at competitive break-even — takes a common critical value across
structurally unrelated agent substrates. This paper reports that **the claim is false
in the regime we can measure**, and documents the two independent ways it failed.

First, the information-denominated form (η\* = ΔW_ablation / kT·ΔI_use) proved
**not measurable in a family-portable way**: three successive declared estimators of
I_use each failed by a distinct and instructive pathology, culminating in the finding
that any estimator keyed to environmental prediction structurally privileges
predictor-shaped agents and cannot, even in principle, test substrate-independence.
Second, the cost-denominated form with dimensional rescaling (Υ_IF = Θ\*·C_model/ν_active)
**scattered across families at 3.8–182σ** at every cost level tested.

Per a pre-registered stop rule adopted before the experiment, two failed principled
rescalings terminate the universality claim. We therefore report IF-H1 as falsified
rather than attempting a third rescaling. We delimit precisely what dies (a universal
constant; the "physical law of agency" framing) and what survives intact (Π_A and Π_C
as per-family measurement instruments; the parasite band; the rule/state dissociation).

---

## 1. What was claimed

The causal-work principle defines agency operationally: internally maintained
information is constitutive of agency when interventionally preserving it yields more
net useful work than the full cost of the apparatus that maintains it. Two ratios
formalize this (see `canon/00-foundations/04-break-even-theorem.md`):

\[
\Pi_A = \frac{kT[I_{\mathrm{pred}} - I_{\mathrm{scr}}]}{C_{\mathrm{model}}},
\qquad
\Pi_C = \frac{kT[I_{\mathrm{pred}} - I_{\mathrm{react}}]}{\Delta C_{\mathrm{full}}}.
\]

IF-H1 held that some dimensionless combination evaluated at competitive break-even
(Π_C = 1) is **substrate-independent** — the same number for a lattice forager, a
linear-Gaussian controller, and a chemotactic swimmer. A confirmed invariant would
have been the program's field-defining result: a candidate physical constant of agency.

The hypothesis passed through three successively weaker forms, each killed:

| Version | Quantity | Fate |
|---|---|---|
| IF-H1 v1 | same threshold p\* across families | dead on arrival (thresholds are coordinate-dependent) |
| IF-H1 v2 | raw Θ\* = Π_A\|_{Π_C=1} | killed by cost-invariance control: Θ\* ∝ 1/C_memory, families separate 3.5–9.6σ |
| IF-H1 v3 | η\* = ΔW/kT·ΔI_use (cost-free by construction) | **not measurable** (§3) |
| IF-H1 v4 | Υ_IF = Θ\*·C_model/ν_active | **scatter 3.8–182σ** (§4) |

## 2. Method

Three families, deliberately chosen so that no two share a mechanism:

- **Ring** — discrete lattice, drifting resource hill, 1-bit sign belief, per-step move cost.
- **Kalman** — continuous line, noisy observations (σ = 0.6), smoothed position+velocity
  belief, actuation-proportional control cost.
- **Chemotaxis** — run-and-tumble. **Scalar concentration sensing only**: no position,
  no direction, no target anywhere in the agent. Memory is the previous concentration
  (the temporal-comparison register of real bacteria); the action is a tumble probability.

The chemotaxis family exists specifically to satisfy the Conway gate: it is not a
tracker in disguise, and its inclusion is what made the falsification possible.

Protocol: 8 seeds per family, 20 000 steps, τ = 1 enforced, C_MEMORY ∈ {0.010, 0.020},
competitive break-even located per seed by linear interpolation of the intact-minus-
reactive work advantage, all interventions marginal-preserving.

## 3. Result 1 — the information denominator is not family-portable

η\* was proposed precisely to escape the cost-dependence that killed Θ\*: a ratio of
work to *bits* carries no cost units at all. Constructing it requires a declared
estimator for I_use. Three were built and tested:

**v1 — I(agent error ; next drift).** *Inverted.* A well-performing agent's tracking
error is small and carries little information about the world; a scrambled agent errs
systematically *in the direction of* the drift. Scrambling therefore **raised** measured
MI (Kalman ΔI_use < 0 throughout). The error is not the channel.

**v2 — I(control decision ; next drift).** *Structurally biased.* This is the natural
reading of "information driving the work-extracting degree of freedom," and it works
for both tracker families. It fails completely for chemotaxis: ΔI_use ≈ 0 while the
work gap between intact and scrambled agents was more than threefold (10867 vs 3598).
A chemotactic swimmer never represents the drift; it climbs a local gradient. **An
estimator keyed to environmental prediction can only detect agents shaped like
predictors — it cannot test substrate-independence, because it presupposes the
substrate.** This is the paper's most transferable methodological finding.

**v3 — I(control decision ; work increment).** Family-portable by construction: every
agent has a work increment. Kalman then yields **persistently negative** ΔI_use
(−1435 to −3232 bits/run): scrambling the anticipation term makes the decision–work
coupling *more* statistically informative, because a systematically wrong lead produces
a tight deterministic relationship between decision and (poor) outcome. Informativeness
is not usefulness — the very confusion the signed functional *J* was conjectured to
repair, here appearing as a concrete measurement failure rather than a thought experiment.

**Corroboration from the corpus.** The source working papers (P02 §Order parameters,
P05 §1) define the primary observable with a **cost** denominator, Π_A = ΔW_enabled /
C_model, and never an information denominator. That choice, undocumented as to
motivation, is vindicated here: the information denominator is not robustly estimable
across substrates with any of the three natural definitions.

## 4. Result 2 — the rescaled cost form scatters

Υ_IF = Θ\*·C_model/ν_active (ν_active = number of dynamically updated belief variables:
ring 1, Kalman 2, chemotaxis 1):

| C_MEMORY | ring | Kalman | chemotaxis | pairwise |
|---|---|---|---|---|
| 0.010 | 2621 ± 22 | 1053 ± 7 | 635 ± 6 | 80–182σ **scatter** |
| 0.020 | 2843 ± 36 | 1222 ± 9 | 679 ± 245 | 3.8–77σ **scatter** |

No pair is consistent at any cost level. The separations are not marginal.

## 5. Verdict, and the stop rule

The pre-registered rule stated: *if η\* fails across families, and one further
principled rescaling also fails, IF-H1 is dead — no third fishing expedition.*
Both conditions are met. **IF-H1 universality is falsified.** We do not attempt a
third rescaling, and this paper is the public record of that commitment being honored.

**What is dead:**
- The claim that a universal dimensionless constant of agency exists in this framework.
- The "physical law of agency" framing for Θ\*, η\*, Υ_IF, and relatives.
- Rung 274177 as originally worded (*"the same threshold across ≥3 rule families"*).

**What survives, stated conservatively:**
- Π_A and Π_C as **per-family measurement instruments**. Nothing here impugns their
  internal validity; they measure what they measure within a substrate.
- **The parasite band** — the dissociation of ablation-positive from
  competitive-positive information — which is a *structural* consequence of the
  break-even inequality and was independently replicated in a third context
  (notebook 04e). It does not require universality.
- **The rule/state dissociation** (ablating a model of the update law selectively
  destroys post-switch recovery, 6σ) — a within-family causal claim.
- The break-even inequality itself as an accounting identity; its two missing lemmas
  (R ≤ 0 and the signed functional J) remain open, and §3 arguably *strengthens* the
  case that J is necessary, since informativeness demonstrably diverged from usefulness.

**Logged but explicitly not claimed:** the ordering ring > Kalman > chemotaxis is
stable across both cost levels with roughly preserved ratios (2.49 and 2.33 for
ring/Kalman). Under the stop rule we record this and decline to build on it. A
regularity noticed *after* a falsification is precisely the pattern a pre-commitment
exists to protect against.

## 6. Why a negative result was worth the cost

Three considerations, none of them consolation:

1. **It was the program's own decisive test.** The adversarial panel identified the
   third-family experiment as the single most important thing to run, ahead of the
   deeper lemma work, precisely because everything downstream depended on the answer.
   It was run first, and it answered.
2. **The estimator finding is transferable beyond IF Theory.** Any framework proposing
   to measure "how much an agent's information is worth" across substrates faces §3's
   trilemma: error-based estimators invert, prediction-based estimators presuppose
   predictors, and outcome-based estimators confuse informativeness with usefulness.
   This constrains the semantic-information and empowerment literatures generally.
3. **A theory that can die is worth more than one that cannot.** The falsification was
   produced by the same discipline that produced the parasite band — frozen contracts,
   marginal-preserving interventions, cost controls, and a stop rule written before the
   data existed. Reporting the kill is the evidence that the discipline is real.

## 7. Falsifiers of *this* paper

- A declared I_use estimator that is (a) portable across the three families, (b) yields
  ΔI_use ≥ 0 for all of them, and (c) is not tuned per substrate, would reopen η\*.
- A demonstration that the chemotaxis family is not genuinely alien (e.g., that it
  implicitly computes a drift estimate) would weaken the substrate-independence test.
- Evidence that the scatter is an artifact of the crossing-location method rather than
  of the quantity itself (test: locate Π_C = 1 by an independent method and re-measure).

Any of these would be a valid challenge, and each is cheaper to run than the original
experiment. We would welcome the reopening; we will not perform it as an act of hope.

---

**Cross-references:** `canon/00-foundations/04-break-even-theorem.md` (the inequality and
the stop rule) · `canon/papers/P05-agency-threshold.md` (IF-H1 as previously stated) ·
`SCOREBOARD.md` §Kill log · `hackathons/if-agency-lab-274177/` (the hackathon that ran it).
