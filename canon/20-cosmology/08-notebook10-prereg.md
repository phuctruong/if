# Notebook 10 Pre-Registration — Expansion–Growth Consistency (Boss #6)

> Auth: 65537 · Layer: SCIENCE · Frozen 2026-07-18 (loop iteration 10), BEFORE any
> chain, likelihood, or growth compilation is opened in this repo generation.
> `FIT_BEFORE_FREEZE` applies chain-by-chain from this commit forward.

## The bet, stated honestly

IF unification claims ONE substrate history b(z) drives both the expansion deviation
and the growth deviation. **Disclosure:** an expansion-side deviation from ΛCDM
(evolving dark energy, w₀ > −1 with wₐ < 0) has been publicly reported by DESI since
2024–2025 at ~3σ-class significance. That signal is therefore *retrodiction* here and
earns IF nothing. **The IF bet is the growth side:** if the expansion deviation is
real, the growth deviation must exist and must share the same redshift shape. ΛCDM
predicts neither; generic quintessence predicts the first without constraining the
second; IF ties them together. That tie is the falsifiable content.

## Frozen parameterization

    s(z; γ)  = (1+z)^(−γ)                (one shape parameter)
    w(z)     = −1 + A_w · s(z; γ_E)      (expansion side)
    μ(z)     = 1 + A_μ · s(z; γ_G)       (growth side; μ = G_eff/G on linear scales)

Two amplitudes (A_w, A_μ) and two shape parameters (γ_E, γ_G) fitted on their own
sides, each alongside standard cosmological parameters. The IF unification statement
under test: **γ_E = γ_G** (shared shape). No per-dataset knobs. The b(z)-level
interpretation (b ∝ 1 − s, discharge history) is narrative until this test passes;
the test itself is purely at the (w, μ) level.

## Frozen datasets (public; exact versions pinned in the notebook's CONTRACT cell at implementation)

Expansion: DESI DR2 BAO likelihood products · Pantheon+ SNe · Planck-2018 distance
priors. Growth: a published fσ₈(z) compilation (pinned at implementation, chosen by
citation count not by result) · Planck lensing amplitude as cross-check. The
`_archive/` BAO/DESI pipelines may be mined for machinery, never for numbers.

## Frozen verdict tree

1. **Both sides consistent with A = 0 (≤ 2σ):** verdict "INDISTINGUISHABLE — no IF
   claim at current sensitivity." (No unification credit; the DESI hint failed to
   survive whatever data-vintage we pin.)
2. **Expansion prefers A_w ≠ 0 (≥ 2σ), growth side then fit:**
   - |γ_E − γ_G| ≤ 2σ_combined **and** A_μ ≠ 0 in the direction the joint fit
     requires → **UNIFICATION SURVIVES ITS FIRST TEST** (survives ≠ proven; this
     feeds the Euclid prereg, notebook 14).
   - |γ_E − γ_G| > 2σ_combined, or growth demands A_μ consistent with 0 while
     expansion demands A_w ≠ 0 at ≥ 3σ → **UNIFICATION DEAD.** Kill published same
     session; boss #6 resolves against us; the P07 cross-scale claim falls with it.
3. **Growth-side data internally inconsistent** (compilation vs Planck lensing at
   > 3σ under our model): verdict "DATA NOT READY — parked," reported as-is.

Primary metric = the shape comparison (i). Declared secondary (reported, no verdict
weight): evidence ratio of joint-b(z) fit vs independent-sides fit.

## Sign derivation obligation

Before the growth chains are opened, a short derivation note must state whether the
IF narrative (discharging battery, P01/P07) fixes the SIGN of A_μ given sign(A_w) —
and if it cannot, say so explicitly. Deriving the sign after seeing growth data is
forbidden (`POSTDICTED_SIGN`).

## Execution order (each step commits before the next begins)

1. This prereg (this commit).
2. Sign-derivation note from P01/P07 narrative.
3. Notebook 10 skeleton with CONTRACT cell (datasets pinned, no data loaded).
4. Expansion-side fit → commit results.
5. Growth-side fit → commit results.
6. Verdict application, SCOREBOARD/kill-log update, same session as (5).
