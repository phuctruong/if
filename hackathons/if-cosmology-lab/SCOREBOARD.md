# IF Cosmology Lab — SCOREBOARD

> Boss #6: ΛCDM. Entered at a declared deficit. Honesty gate: Sagan (extraordinary claims).

| Track | Score | Evidence |
|---|---:|---|
| C1 Data restored | 10/10 | ✅ SPARC re-fetched 2026-07-18 (175 rotmod files + Lelli2016c table), sha256 pinned in `data/sparc/CHECKSUMS.txt` |
| C2 Baseline reproduction | 25/25 | ✅ **REPRODUCED** with an independently written pipeline: MOND 3.298 vs 3.707 (11%), NFW 0.938 vs 1.144 (18%); BIC 50.59 vs 50.87, 20.42 vs 19.80. NFW < MOND ordering confirmed. `notebooks/01_sparc_baseline_reproduction.ipynb` |
| C3 **Does the hypothesis have a galaxy law?** | **20/20** | **AUDIT COMPLETE — verdict: NO.** P07's action is self-described as "a broad target class, not a unique theory"; ℒ_IF is a free function, so μ_IF/a_IF are symbols awaiting a theory. Nothing to fit. Written up: `canon/20-cosmology/03-testability-audit-2026-07-18.md` |
| C4 Central test (expansion–growth) | 0/25 | **Blocked on the same free function** — not a scheduling problem, a specification problem |
| C5 Distinctive prediction (EFE hysteresis) | 0/10 | Requires an implemented law |
| C6 Preregistration prep | 7/10 | **P11 status resolved (see below).** Preregistered galaxy-scale admission criteria FROZEN (χ²/dof ≤ 3.71 to be interesting, ≤ 1.14 + BIC ≤ 19.8 to win, 30% held-out, no per-galaxy IF params) — from the archived fair benchmark, so the bar is legitimate and cannot move. Euclid forecast still pending |
| **TOTAL** | **62/100** | 2026-07-18 (C1+C2 complete) |

## The finding that matters more than the score

The cosmology branch is **specified but not implemented**. That is a different thing from
"untested" and a very different thing from "promising." Until a specific ℒ_IF is committed
in a timestamped commit *before* evaluation, every downstream test (C2, C4, C5) is blocked
not by effort but by the absence of a theory to test.

The correct public phrasing, binding on all abstracts, talks, and book chapters
(Sagan gate): **"specified, not yet implemented, not yet tested."**

## Why the score is low and should stay low

Scoring 25/100 while the tracks are blocked is the honest state. The alternative —
choosing an ℒ_IF now and fitting SPARC with it — would score points and be
`RETROFIT_FORECAST`: a post-hoc choice of free function dressed as a prediction. The
archived generation already ran that experiment for real and lost to both MOND and NFW.

## P11 (Euclid preregistration): resolved — it CANNOT be executed yet, and that is stated

P11 is a **template with `freeze_datetime_utc: null`**. The honest finding, now recorded
rather than left ambiguous: **a preregistration cannot be executed until an ℒ_IF exists.**
Freezing predicted values for μ, η, w_IF, a_IF while ℒ_IF remains a free function would
freeze *numbers chosen by hand*, not predictions *derived by a theory* — the form of
`RETROFIT_FORECAST` that a preregistration is specifically supposed to prevent. Freezing
the wrong thing is worse than not freezing, because it purchases credibility the theory
has not earned.

**Therefore rung 65537 is BLOCKED, not pending** — blocked on a theory, not on a deadline.
If Euclid DR1 (~Oct 2026) arrives before an ℒ_IF is committed, the correct action is to
make no IF claim about it at all.

What *can* be frozen now, and has been: the **galaxy-scale admission criteria** (above),
because those are performance thresholds derived from an already-run adversarial benchmark
rather than predictions of an unwritten theory.

## Next moves (in order)

1. ✅ ~~Re-fetch SPARC~~ → done, checksummed
2. ✅ ~~Reproduce baselines, no IF cell~~ → done, within tolerance
3. **The blocking step**: propose a specific ℒ_IF with derived μ_IF and a_IF, committed
   *before* any evaluation. Until this exists, C4/C5/C6 cannot proceed and no amount of
   effort substitutes. → unblocks everything
4. Only after 3: expansion–growth consistency, EFE hysteresis, Euclid freeze.
