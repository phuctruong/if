# IF Cosmology Lab — SCOREBOARD

> Boss #6: ΛCDM. Entered at a declared deficit. Honesty gate: Sagan (extraordinary claims).

| Track | Score | Evidence |
|---|---:|---|
| C1 Data restored | 0/10 | SPARC absent locally (`/home/phuc/Downloads/if/data/sparc/` empty); re-fetch pending |
| C2 Baseline reproduction | 0/25 | Blocked on C1 |
| C3 **Does the hypothesis have a galaxy law?** | **20/20** | **AUDIT COMPLETE — verdict: NO.** P07's action is self-described as "a broad target class, not a unique theory"; ℒ_IF is a free function, so μ_IF/a_IF are symbols awaiting a theory. Nothing to fit. Written up: `canon/20-cosmology/03-testability-audit-2026-07-18.md` |
| C4 Central test (expansion–growth) | 0/25 | **Blocked on the same free function** — not a scheduling problem, a specification problem |
| C5 Distinctive prediction (EFE hysteresis) | 0/10 | Requires an implemented law |
| C6 Preregistration prep | 5/10 | **Preregistered galaxy-scale admission criteria FROZEN** (χ²/dof ≤ 3.71 to be interesting, ≤ 1.14 + BIC ≤ 19.8 to win, 30% held-out, no per-galaxy IF params) — from the archived fair benchmark, so the bar is legitimate and cannot move. Euclid forecast still pending |
| **TOTAL** | **25/100** | 2026-07-18 |

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

## Next moves (in order)

1. Re-fetch SPARC (Lelli 2016c table + 175 Rotmod_LTG files); checksum and pin. → C1
2. Reproduce the archived benchmark end-to-end for MOND and NFW only (no IF cell). → C2
3. **Only then**, and only in a commit that precedes any evaluation: propose a specific
   ℒ_IF with a derived μ_IF and a_IF. → unblocks C4/C5
4. Euclid forecast freeze before DR1 (~Oct 2026). → C6, rung 65537
