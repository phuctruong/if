# IF Cosmology Lab — Phase 3

> Auth: 65537 · Started 2026-07-18 after Phase 2 sealed **by falsification**.
> 🎮 Boss #6: **ΛCDM**. Entering at a deficit, on purpose, with eyes open.

## The honest starting position

Two facts inherited before a single line of new code:

1. **The prior-generation IF galaxy law was fairly tested at full scale and lost.**
   Archived 175-galaxy SPARC benchmark: median χ²/dof — IF 7.13, MOND 3.71, NFW 1.14;
   median BIC — IF 85.9, MOND 50.9, NFW 19.8. Losing on BIC too means it wasn't a
   parameter-count story. (`_archive/evidence/sparc_fair_benchmark/`)
2. **Phase 2 killed the flagship agency claim.** The program has no universal constant.

Phase 3 therefore begins by *assuming the cosmology branch is probably wrong* and asking
what evidence could change that — not by looking for a fit. The panel's standing verdict
(3/10 plausibility, 10/10 stakes) is the operating assumption.

## Rubric (100 points)

| Track | Pts | What earns them |
|---|---:|---|
| **C1 Data restored** | 10 | SPARC (Lelli 2016c table + 175 rotmod files) re-fetched, checksummed, pinned; DESI/Planck products located |
| **C2 Baseline reproduction (Feynman gate)** | 25 | Pipeline reproduces published RAR + BTFR + NFW/MOND fits to archived tolerances **before any IF cell exists**; ΛCDM/DESI posterior reproduction |
| **C3 Does the current hypothesis even have a galaxy law?** | 20 | Honest audit: the unified-geometry hypothesis (P07/P08) has no implemented g_obs(ρ_baryon). Either derive one from the b(z) state **or state plainly that the branch is not yet testable at galaxy scale** — the second is a valid, scoring outcome |
| **C4 The central test** | 25 | Expansion–growth consistency: fit H(z), D_A, D_L → predict fσ₈(z); then reverse. Kill condition: b_expansion(z) ≠ b_growth(z) → unification dead |
| **C5 Distinctive prediction** | 10 | EFE environmental hysteresis specified precisely enough to be wrong (the one observable no competitor predicts) |
| **C6 Preregistration prep** | 10 | Euclid forecast frozen in a timestamped commit before DR1 (~Oct 2026) → rung 65537 |

## Standing rules (harder here than anywhere)

- **No IF plot before the baseline plots agree** (Feynman gate). Non-negotiable.
- **One state, no sector-split fits** (Noether gate). Separate parameters for
  DM-like and DE-like effects = `SECTOR_SPLIT_FIT`, the unification is dead, log it.
- **Mocks before catalogs** (Rubin gate). Every statistic validated on mocks with
  known truth before touching real data.
- **`_archive/` code may be mined but never trusted** — re-verify, state provenance,
  never `SILENT_ARCHIVE_RESURRECTION`.
- A clean "not yet testable" beats a fitted curve. **C3 pays 20 points for admitting it.**

## Seal condition

Either a preregistered, falsifiable IF cosmology prediction is on the record before
Euclid DR1 (rung 65537), **or** the canon states plainly that the cosmology branch
could not be brought to testable form and says why. Both seal the phase.
