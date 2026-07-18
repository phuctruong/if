# ℒ_IF Design Constraints — what any candidate must survive BEFORE it may be frozen

> Auth: 65537 · Layer: SCIENCE · Written 2026-07-18 (loop iteration 6).
> **This document evaluates nothing.** It runs no fits, touches no SPARC data, and
> spends no pre-registration legitimacy. It exists so that when an ℒ_IF is committed
> (the single blocker for Phase 3 C4/C5 and all of Phase 4), the commitment is a real
> bet and not a hand-chosen number. Sagan phrasing remains in force for the whole
> branch: *specified, not yet implemented, not yet tested.*

## The situation

The unified-geometry hypothesis (P07, `01-unified-geometry-hypothesis.md`) has no
implementable galaxy-scale law. The archived log-potential law is dead (kill log
2026-07-18: median χ²/dof 7.13 vs MOND 3.71 vs NFW 1.14; BIC 85.9 / 50.9 / 19.8 on 175
galaxies, fair rules). The branch therefore starts at a **deficit**. The admission bar
is already frozen and lives with the Phase-3 record: median χ²/dof ≤ 3.71 to be
*interesting*, ≤ 1.14 with BIC ≤ 19.8 to *win*, 30% held-out, no per-galaxy IF
parameters. Locally reproduced baselines to beat: MOND 3.298, NFW 0.938.

## Hard constraints (violate any one → the candidate is dead on arrival)

- **C1 Dimensional closure.** Every term carries dimensions; every new constant has a
  declared unit and a single global value. No "information" entering an equation of
  motion without a declared estimator mapping it to observables (Shannon gate — the
  P15 lesson: prediction-keyed estimators presuppose predictor-shaped systems).
- **C2 Newtonian recovery.** ℒ_IF → GR/Newton in the high-acceleration, low-information
  limit, quantitatively: solar-system PPN bounds are not negotiable.
- **C3 Zero per-galaxy freedom.** At most one fitted M/L per galaxy (the fair-rules
  convention every competitor gets). Any galaxy-indexed IF parameter = automatic kill.
  This is what killed nothing yet but disqualified freezing P11 as "hand-chosen numbers."
- **C4 The frozen admission bar** above, unchanged. A candidate that cannot plausibly
  clear "interesting" (≤ 3.71) has no business being frozen — but plausibility
  arguments happen HERE, in prose, not by running SPARC.
- **C5 BTFR compatibility.** The baryonic Tully–Fisher relation (slope ~4, tight
  scatter) must come out, not be put in. MOND survives because it predicts this; any
  IF law that cannot say *why* v⁴ ∝ M_b is disfavored before fitting.
- **C6 The P17 constraint (new, from the agency branch).** Any ℒ_IF narrative in which
  "information accumulates wherever energy flows" is now empirically false at
  laboratory scale: energy-gated substrates produced motion but no lineage and no
  tracking. If ℒ_IF couples geometry to *information structure*, it must specify what
  counts as structure without assuming life-like accumulation — or restrict its domain
  to scales where accumulation is independently evidenced.
- **C7 Cross-scale consistency.** Whatever functional replaces the galaxy law must be
  the SAME functional the cosmology-scale claims use (P07's fixed cross-scale relation),
  or explicitly declare the scale-bridging map. No per-scale re-tuning.
- **C8 Regeneration.** The candidate, once frozen, must be evaluable by a deterministic
  script from `data/sparc/` (checksummed) with the held-out split fixed by seed in the
  prereg commit.

## Candidate families (sketches for future panel review — NOT proposals)

| Family | One-line form | Why it might clear C5 | Known risk |
|---|---|---|---|
| A. Information-density modified inertia | m → m·f(I(r)/I₀), f→1 at high a | If I(r) tracks baryon surface density, MOND-like phenomenology follows | "I(r)" is a knob unless C1's estimator is real — the exact P15 failure mode |
| B. Entropic-gradient potential | Φ_IF = Φ_N + λ∇S_config | Prior art exists (entropic gravity); S_config computable from photometry | Prior-art self-audit mandatory (P16 discipline); Verlinde-adjacent → novelty burden high |
| C. Memory-kernel gravity | g(r,t) = g_N * K(τ_dyn/τ_IF) | A single global timescale τ_IF is maximally rigid (C3-safe) | Must not reduce to a MOND interpolation function in disguise — if it does, say so and stop |

The next legitimate step is NOT to fit any of these. It is: pick one family, derive its
BTFR behavior on paper (C5), check C2 on paper, run the P16-style prior-art self-audit,
and only then — if it survives prose — freeze the exact functional + constants-fitting
procedure in a prereg commit and touch SPARC once.

## Forbidden states for this branch

`FIT_BEFORE_FREEZE` (any SPARC contact before the prereg commit) ·
`PER_GALAXY_KNOB` (C3 violation) · `INTERPOLATION_SMUGGLING` (a MOND μ-function renamed
without acknowledgment) · `ESTIMATOR_HANDWAVE` (an information term with no declared
measurement procedure).
