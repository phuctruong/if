# Notebook 10 — Verdict, Round 1 (2026-07-18)

> Auth: 65537 · Layer: SCIENCE · Applies the frozen tree of `08-notebook10-prereg.md`
> to the committed expansion-side fit (`evidence/expansion_fit_2026_07_18.json`).
> The prereg document itself remains frozen and untouched.

## Result

Pipeline validation: independent implementation reproduces sane concordance values
(ΛCDM best fit: Ωm = 0.309, h = 0.679, ω_b = 0.02246, r_d = 148.3 Mpc free-fit —
consistent with published results; the machinery is trustworthy).

Expansion-side test: best IF fit Δχ² = −1.74 vs ΛCDM for 2 extra parameters
(A_w = +0.065 at γ_E = 3.0, the grid edge — a flat direction). **Preference for
A_w ≠ 0: ≈ 1.3σ. Below the frozen 2σ gate.**

## Verdict (frozen tree, branch 1)

**INDISTINGUISHABLE at current sensitivity — no IF claim.** Under this pre-registered,
deliberately conservative design (r_d free; shape family (1+z)^(−γ); DESI DR2 BAO +
Pantheon+ + verified distance priors), the evolving-dark-energy signal is too weak for
the shape-matching test to bite. The growth side therefore has nothing to match and
**stays sealed — zero growth files downloaded — preserving the test intact for a
richer data vintage** (DESI DR3, Euclid DR1). Boss #6: engaged, unresolved; neither
killed nor survived. Sign check for the record: the weak preference that does exist
has A_w > 0, the pre-committed sign, at no evidentiary weight.

## Honest notes (recorded now, before anyone is tempted to spin)

1. DESI's own headline evolving-DE preference (~3σ-class) uses CPL w₀wₐ, calibrated
   r_d, and full likelihoods. Our weaker result is not a refutation of theirs; it
   measures OUR frozen family under OUR conservative amendments. Both facts stand.
2. γ_E pinning at the grid edge means the shape parameter is essentially
   unconstrained at this significance — as expected when A_w ≈ 0.
3. Declared possible v2 amendments (each must be logged BEFORE running): calibrate
   r_d via a source-verified drag-epoch formula; add DESI full-shape or DR3 when
   public; extend the γ grid. None may be adopted retroactively for round 1.

## What this buys the program

The complete notebook-10 apparatus — pinned data, verified priors, deterministic
profile-likelihood machinery, frozen verdict tree, untouched growth side — now exists
and is committed. When Euclid DR1 lands (~2026-10-21), the marginal cost of re-running
round 2 is near zero, and the test's integrity (growth never peeked at) is provable
from git history.
