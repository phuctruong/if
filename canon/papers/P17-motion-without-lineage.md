# Motion Without Lineage

## A Pre-Registered Negative Arc: Energy-Gated Cellular Automata Grow Movers but No Heredity, Hence No Selectable Agency

**Author:** Phuc Vinh Truong
**Series:** IF Theory Canonical Papers
**Paper:** 17 (negative arc — five pre-registered experiments, one day)
**Layer:** SCIENCE
**Status:** Sealed 2026-07-18. Every verdict below was frozen before its experiment ran.
**Evidence:** `evidence/mobility_{controls,sweep_stageA,confirm_stageB}_2026_07_18.json`, `evidence/mover_audit_2026_07_18.json`, `evidence/resource_tracking_2026_07_18.json`, `evidence/scarcity_boundary_2026_07_18.json`, `evidence/lineage_2026_07_18.json`, `evidence/rule_search_2026_07_18.json`, `evidence/power_audit_2026_07_18.json`
**Hackathons (prereg commits in each):** `if-mobility-search`, `if-resource-tracking`, `if-scarcity-boundary`, `if-lineage`, `if-rule-search`, `if-causal-power`
**Films:** `videos/mobility-regime.mp4`, `videos/starved-vs-abundant.mp4`, `videos/scramble-ignition.mp4`

---

## Abstract

We asked whether an artificial universe satisfying the Conway gate — local rules
containing no agency terms, plus an exactly conserved energy ledger — can grow agents,
in the specific sense the IF program requires: emergent structures whose organization
does thermodynamic work and whose behavior can be selected toward resources. The answer,
established by five pre-registered experiments in one session, is a structured **no**
with an exact mechanism:

1. **Movers emerge, controlled by energy economics, not rules.** The prior "still lifes
   only" state of the lab was energy starvation: dropping construction cost from
   E_BIRTH = 1.0 to 0.25 turns the same B3/S23 rules from a crystal desert into a
   substrate producing ~24 emergent mobile structures per 600-step run.
2. **Movers are ballistic.** Two independent, separately pre-registered tracking tests —
   under abundance (1,679 tracks) and at the scarcity floor (293 tracks) — found no
   coupling between mover motion and resource direction (bias < 4% of full alignment
   in the powered test).
3. **The mechanism is a missing heredity unit.** A birth-attribution census found
   one-shot fission of movers to be common (~14.5/run) but repeat producers —
   structures emitting ≥ 2 eventually-mobile children, the minimal unit on which
   directional selection could accumulate — entirely absent (0 in 24 runs).
4. **The absence generalizes.** A declared six-family rule grid (HighLife, LowDeath,
   Pedestrian Life, 34 Life, 2×2, Day & Night) × 4 energy configurations was barren of
   producers, firing the pre-registered stop rule that closes the branch.
5. **Organization's causal work is indistinguishable from zero under the drawdown
   observable.** Across 74 audited emergent movers (intact vs count-preserving scramble
   forks), W_C has median +0.4, 51–62% positive, but a heavy negative tail — scrambled
   debris occasionally ignites growth that out-harvests the organized mover — leaving
   t ≈ −0.9; the pre-registered rest-clause ended same-design sampling.

The arc's one-sentence conclusion: **in energy-gated Life-like CA at laboratory scale,
motion is cheap and lineage is not — and without lineage there is nothing for selection
to make into an agent.**

## 1. Why this arc had to run

P15 falsified IF-H1 for hand-designed agents; the kill log then recorded that the
Conway gate had never actually been satisfied — no experiment had tested agents *a
universe produced*. The five experiments here are the systematic discharge of that
debt. Each ran under the session disciplines: protocol frozen and committed before
execution; painful verdict tiers pre-declared and honored (VOID once, MOOT-avoidance
once, ABSENT once, BARREN once, UNDECIDED-then-rest twice); no threshold moved after
seeing data; every kill published the same session.

## 2. The mobility phase boundary is in construction cost (positive result)

Declared 216-config sweep (3 rules × E_BIRTH {0.25, 1} × E_MAINT {0, 0.01} ×
inflow {1, 4, 12} × σ {14, 40, 10⁶} × ρ {0.08, 0.15}): every top mobility config has
E_BIRTH = 0.25. Confirmation at 8 seeds × 600 steps: mean 24.0 emergent mobile tracks
per run (pre-frozen threshold: 0.5). A follow-up inflow sweep found **no threshold in
energy income** — movers persist smoothly from inflow 0.5 to 12 (2.25 → 28.1 per run,
a 24× range). Construction cost, not income, is the order parameter of the
motion phase. (Incidental instrument result: the original still-life detector used
min_size = 6 and was constitutionally blind to 5-cell gliders; re-audit at min_size = 5
confirmed the still-life verdict anyway.)

## 3. Ballistic movers: two pre-registered tracking nulls

Statistic τ: mean windowed cosine between mover displacement and bearing-to-source,
gradient arm vs gradient-free placebo arm (σ = 10⁶) computed with the identical
estimator. Abundance (inflow 12): τ = +0.022 vs +0.015, Welch t = +0.34, n = 849/830.
Scarcity floor (inflow 0.5, fresh seeds): τ = +0.078 vs +0.052, Welch t = +0.54,
n = 152/141. Secondary r(τ, lifetime) ≈ 0 in both arms: no survival advantage to
pointing at the resource. Direction is set at birth and never revised — the movers
are projectiles, not agents.

## 4. The heredity gap (the arc's mechanism)

Selection cannot act on a trait that dies with its bearer. Birth-attribution census
(newborn track's cells, dilated r = 2, overlapping a live track's previous cells →
production event; ≥ 2 eventually-mobile children → producer): production events
common (14.5 mover-productions/run — chaotic blobs shed a glider once), producers
**absent** (0/24 gradient runs; 1/24 placebo, below the frozen bar). One-shot shedding
gives motion; only repeat emission gives lineage; this substrate has the first and not
the second. The declared six-family rule search then returned zero flagged configs out
of 24 — the stop rule closed the branch. Scope honestly stated: our search is ~10²
soups against soup-search folklore's ~10⁹; BARREN means *this program's affordable
budget found none*, and reopening requires a cheaper detection idea, not more soup.

## 5. The scramble-ignition confound (methodological result)

The causal-work audit forks a universe and scrambles the mover's cells in place
(count-preserving). In an energy-rich medium the scrambled debris sometimes explodes
into growth that out-harvests the organized structure it replaced (W_C to −93). Median
mover W_C is slightly positive; the mean is not. Two consequences: (a) organization
in this substrate is energy-*frugal*, not energy-greedy — raw harvest is the wrong
observable for organization's value; (b) scramble-ablation is not a clean
"remove the organization" intervention in open dynamical media — it *adds* a live
perturbation. This is the same lesson as P16's ablation-vs-competition dissociation,
arrived at from an independent direction.

## 6. What survives for the IF program

- The Conway-gate substrate itself: real, cheap, ledger-exact, with a controlled
  mobility phase. It remains the honest venue for any future causal-work observable
  (harvest-per-mass-step is the drafted candidate; needs its own prereg).
- The construction-cost phase boundary — a clean, reproducible, positive result.
- The constraint delivered to theory: **any IF claim that life-like tracking arises
  wherever energy flows is now false at this scale.** Tracking requires heredity;
  heredity did not come free. A future ℒ_IF or agency model must either supply a
  replication mechanism or drop emergent tracking from its predictions.

## 7. Falsifiability of this paper's own claims

Every negative here is scoped to declared grids and budgets recorded in the evidence
JSONs. Any of them dies to: a producer family found inside the declared grids at the
declared scale (re-run our exact scripts); a tracking signal at |τ diff| > 0.1 under
either frozen design; a causal-work verdict at |t| > 2 under the frozen drawdown design
at any n. The scripts are deterministic given seeds; every number in this paper
regenerates from `scripts/` in minutes to hours.
