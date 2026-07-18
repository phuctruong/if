# Horizon-Response Gravity — a disciplined no-dark-sector research program

> **Refactored 2026-06-13.** This repository pivoted from the *Prime Field*
> (now falsified as discriminating physics — see
> `archive/2026-06-13-prime-field-deprecated/MANIFEST.md`) to its one
> surviving, real result: **the dark-matter acceleration scale is the cosmic
> horizon, a₀ ≈ cH₀/(2π).** The goal — *no dark matter particle, no dark
> energy fluid* — survives; the prime *mechanism* did not.

Author: Phuc Vinh Truong · phuc@phuc.net · MIT.
Review record: `audits/PEER_REVIEW_FABLE5_2026-06-12.md` + the two-LLM and
three-LLM (`/leak4`) consensus in `~/Dropbox/solace/hackathon-if-100/`.

## The claim (honest, narrow, real)

The apparent dark sector is the **response of geometry/inertia to baryons
under a de Sitter horizon constraint** — not invisible substance. Core family:

```
g_obs = ν(g_bar / a₀) · g_bar ,   a₀ = α·cH₀ ,   α ≈ 1/(2π)
  galaxy limit       → QUMOND / RAR
  relativistic limit → AeST-class scalar/vector (Skordis–Złośnik)
  cosmology          → horizon entropy / effective Λ / possible w(z)
```

## What is established (gate 1 — galaxies) ✅

On staged SPARC (3,391 points, `falsification_harness.py` in the hackathon dir):

| Model | RAR scatter |
|---|---|
| **Horizon a₀ = cH₀/2π (ZERO parameters)** | **0.204 dex** |
| MOND (a₀ free-fit) | 0.203 dex |
| Newton (baryons only, no dark sector) | 0.293 dex |

The **zero-parameter horizon prediction ties free-fit MOND** and clearly beats
baryons-only. The dark-matter-like acceleration in galaxies is set by the
cosmic horizon scale, with nothing tuned. This is an independent rediscovery
of Milgrom/McGaugh + the a₀–cH₀ coincidence. *(Caveat: "no room for particle
DM" — intrinsic RAR scatter → 0 — needs the full error budget; not yet proven
here. See harness.)*

## What is falsified (and archived)

The Prime Field as fundamental physics. On real BOSS ξ(r) the form
[1/log(r/r₀+1)]² **ranks last** of 7 smooth shapes (ΛCDM-linear χ²/dof ≈ 0.8–1.4
vs IF ≈ 75–295); C_XI=62 is not distinguished from neighbors; "primeness" never
enters the computation. The old high-σ was correlation-vs-zero (any declining
curve passes). Full record: `archive/2026-06-13-prime-field-deprecated/`.

## The four gates (in order of difficulty) and where we stand

1. **Galaxies (RAR/BTFR)** — ✅ horizon a₀ passes at zero parameters.
2. **Clusters** — ⏳ MOND's residual ~2×; may force "minimal" not "zero" dark sector.
3. **CMB acoustic peaks** — ⚠️ **make-or-break.** Three-LLM `/leak4` panel verdict
   (2026-06-13): AeST fits the peaks *only* via a field with w≈0, c_s²≈0, that
   clusters with abundance ≈ Ω_cdm — i.e. it avoids the dark-matter *particle*,
   not the dark-matter *role*. Verlinde has no early-universe sector. So the CMB
   is *not yet cleanly won* by any no-dark-sector theory.
4. **Expansion vs growth (w(z), fσ₈)** — ⏳ DESI DR2 / Euclid discriminate.

## The locked next test (panel-selected, highest value)

**Deep-MOND weak-lensing RAR:** stacked galaxy-galaxy lensing around isolated
galaxies (KiDS/DES) at g ~ 10⁻¹¹–10⁻¹² m/s². A single-a₀ theory predicts a
tight, parameter-free continuation of the RAR; ΛCDM predicts halo-mass-set
scatter/upturn. Chosen because it tests the program where it **cannot absorb a
tuned field** — no early-universe dust loophole, a₀ fixed with zero freedom.

## Honest probabilities (two-/three-LLM consensus)

- a₀ ≈ cH₀ is a real clue: **50–70%**
- no particle dark matter via modified gravity: **5–15%**
- fully replace ΛCDM (no DM *and* no DE): **1–5%** — a real long-shot at a
  Nobel-scale prize, pursued opportunistically as data (DESI DR2, JWST z≥25,
  deep lensing) arrives.

## Repo discipline
Every claim is gated **model-vs-data or model-vs-model — never
correlation-vs-zero.** Surviving evidence: `predictions/sparc_per_galaxy_ml.py`,
`predictions/sparc_corrected_log_potential.py`, `adversarial/dwarf_regime_split.py`,
`adversarial/power_law_null_test.py`, `adversarial/survey_clustering_replication.py`,
`predictions/mw_rotation_sigma_accounting.py`. See `THE-FINAL-LAW.md`,
`FALSIFIABILITY.md`, `REPLICATION.md`.
