# Lattice Hardening — what survives the RAR closure, and what to build before DR1

> Auth: 65537 · Layer: SCIENCE · Written 2026-07-18 (loop iteration 9).
> Applies the `06-sigma-degeneracy-check.md` result to every entry of
> `02-prediction-lattice.md`. No fits run. Output: a ranked, honest build queue for the
> Euclid window (~Oct 2026) and the minimal theory object it actually requires.

## 1. Entry-by-entry status under the RAR closure

| Entry | Status after iterations 6–8 |
|---|---|
| 01 SPARC baseline | ✅ ALIVE as validation only (already reproduced: MOND 3.298, NFW 0.938) |
| 02 rotation-curve holdout | ⛔ **CLOSED BY PROOF** — any g_IF[g_b, ∇g_b] either lives inside RAR residuals (undetectable) or is falsified by them. Kept in the lattice as a tombstone with this annotation; do not build. |
| 03 EFE environment | ⚠️ survives ONLY through the **hysteresis clause** — an environment-memory lag is IF-distinctive (MOND's EFE is instantaneous); the static-EFE part is a MOND test, not ours |
| 04b a_IF(z) evolution | ⚠️ alive but blocked on the scalar history b(z); needs no galaxy functional — reduced commitment (see §2) |
| 05 wide binaries | ⬇️ demoted — a MOND-vs-Newton discriminator (currently contested in the literature), not IF-distinctive; useful as external constraint input only |
| 06 dynamics vs lensing η | alive, needs field-level theory; post-DR1 horizon |
| 07 voids | alive in principle; needs μ_IF[b]; medium horizon |
| 08 cluster-merger memory | ✅ **MOST IF-DISTINCTIVE SURVIVOR** — a universal relaxation law Δx_lens(t) = Δx₀e^(−t/τ_IF) is predicted by neither collisionless ΛCDM nor MOND; needs merger catalogs (lensing–X-ray offsets + collision ages), not a galaxy law |
| 09 ΛCDM reproduction | ✅ alive, validation-only, prerequisite for 10/11 |
| 10 expansion–growth consistency | ✅ **THE CENTERPIECE (boss #6)** — needs only the scalar b(z) + response maps; public chains suffice; the falsifier (two different b(z) histories required → UNIFICATION DEAD) is exactly the honest bet |
| 11 H₀/growth tensions | alive, downstream of 10 |
| 12 cosmic information history | ✅ alive — the KL estimator on CAMELS is buildable now; C1-grade estimator freeze required first |
| 13 web topology | alive, independent of ℒ_IF; medium effort |
| 14 Euclid prereg | ✅ the target; fed by 10 (+12 if ready) |
| 21/22 GW entries | alive, post-DR1 horizon |

## 2. The reduced theory object (the real yield of this hardening)

The dead galaxy-law program demanded a *functional on galactic fields*. Everything the
pre-DR1 lattice actually needs is smaller:

    b(z)        one scalar history (parameterized, e.g. 2–3 numbers)
    μ[b], w[b]  two declared response maps (growth strength; expansion pressure)
    τ_IF        one relaxation constant (entry 08)

No per-galaxy anything; C1–C8 apply verbatim; the P17 constraint (C6) applies to any
narrative about *why* b(z) evolves. Freezing THIS object is a commitment measured in
~5 numbers — small enough to be honest, big enough to be killed.

## 3. Ranked build queue for the Euclid window

1. **Notebook 10 — expansion–growth consistency (boss #6).** Fit b(z) to expansion-side
   data (BAO/SNe/H(z) public chains), predict growth side (fσ₈, P(k) ratio); reverse;
   demand b_expansion ≡ b_growth. All data public TODAY. Falsifier already frozen in the
   lattice. This is the single highest-value pre-DR1 circuit and needs its own prereg
   (parameterization + datasets + consistency metric frozen before any chain is read).
2. **Entry 08 — cluster-merger memory survey (prose first).** Inventory published
   merging-cluster offset measurements + collision-age estimates; if ≥ ~10 systems with
   both exist, a τ_IF prereg is possible; if not, record the data gap and park.
3. **Entry 12 — I_NL estimator freeze.** Specify KL[nonlinear‖linear] to the pixel level
   on CAMELS snapshots (which fields, which coarse-graining, which prior); stability
   across sims/feedback IS the entry's own falsifier.
4. **Entry 03-hysteresis — design sketch only** (needs environment-change samples; likely
   post-DR1).

Then notebook 14 freezes whatever 10 (and possibly 12) produce, timestamped before
DR1 (~2026-10-21). If b(z) fails its own consistency test first, the Euclid prereg
records the *failure* — per the standing discipline, that too is a keepable prediction.

## 4. What this document forbids

Building entry 02 (`CLOSED_BY_PROOF`); freezing any b(z) *after* looking at growth-side
data (`FIT_BEFORE_FREEZE` applies chain-by-chain); presenting entry-05 results as
IF evidence (`NOT_OURS_TO_CLAIM`).
