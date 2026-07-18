# Archived 2026-06-13 — Prime-Field-as-fundamental-physics (deprecated)

## Why
A full adversarial review (Claude + ChatGPT, independent convergence; sealed in
`~/Dropbox/solace/hackathon-if-100/CONSENSUS-two-LLM.md`) falsified the prime
field as *discriminating physics*:

- The LSS form ξ ∝ [1/log(r/r₀+1)]² **ranks last** of 7 smooth shapes on real
  BOSS ξ(r): ΛCDM-linear χ²/dof ≈ 0.8–1.4, power law ≈ 5, IF ≈ 75–295.
- C_XI = 62 is **not distinguished** from neighboring integers by the data.
- "Primeness" **never enters the computation** — the field evaluates the smooth
  PNT asymptotic 1/log(x) at continuous coordinates; no integer-prime structure
  is ever used. The "prime field" is mathematically the "1/log field."
- The old "high σ" was **correlation-vs-zero** (rejecting "uncorrelated with a
  declining curve"), which any declining curve passes — not evidence for the form.
- Devil's-advocate rescues (free-r₀ fit; prime redshift-quantization on 375k
  SDSS galaxies) also failed.

This is falsification as strong as science allows for a discriminating claim.
The GOAL ("no dark matter/energy") survives; the prime MECHANISM does not.

## What's archived here (false-physics front matter + falsified demos)
- `THEORY.md` — prime field as fundamental law (front-matter claim, falsified)
- `VALIDATION.md` — inflated per-survey σ tables + the retracted
  "χ²-variation-validates-zero-parameters" argument (variation evidences
  non-tuning, never correctness)
- `INDEPENDENT_VALIDATION.md` — "Solace AGI replication" never externally verified
- `SIMPLE.md` — popular prime-field framing
- `mersenne_tower_theorem_paper.md` — C_XI = 2·π(127) = 62 derivation (decorative)
- `prime_field_demo.ipynb`, `dark_matter_{sdss,desi,euclid}.ipynb`,
  `dark_energy_{bao_proof,demo}.ipynb` — prime-LSS / prime-dark-energy notebooks.
  (Their data runs were real and reproduce; the *claims* are falsified.)

## Deprecated CODE retained in-tree for import-safety (NOT moved)
The surviving modified-gravity evidence imports `R0_KPC_CANONICAL` from
`prime_field_util.py`, so the prime code is left in place to avoid breaking the
pipeline. These files are **deprecated as physics** but kept as utilities:
- `prime_field_theory.py`, `prime_field_util.py` (only the r₀ constant + helpers
  are still used; the 1/log "field" is dead)
- `mersenne_tower_*.py` + `tests/test_mersenne_tower.py` (number theory; decorative)
- prime-LSS prediction scripts: `predictions/boss_published_xi_test.py`,
  `predictions/lss_bao_locked_prediction.py`, `predictions/sparc_175_validation.py`,
  `predictions/sparc_175_per_galaxy_v0.py`, `predictions/sparc_multichannel_test.py`,
  `adversarial/c_xi_uniqueness_test.py`

## What SURVIVED (the new direction — see ../../README.md, ../../THE-FINAL-LAW.md)
Horizon-Response Gravity: no dark matter particle, no dark energy fluid; the dark
sector is the response of geometry/inertia to baryons under a de Sitter horizon
constraint, with **a₀ ≈ cH₀/(2π)**. Gate 1 (SPARC RAR) passes at ZERO parameters.
Surviving evidence: `predictions/sparc_per_galaxy_ml.py`,
`predictions/sparc_corrected_log_potential.py`, `adversarial/dwarf_regime_split.py`,
`adversarial/power_law_null_test.py`, `adversarial/survey_clustering_replication.py`,
`predictions/mw_rotation_sigma_accounting.py`, `audits/PEER_REVIEW_FABLE5_2026-06-12.md`.

Everything here is fully recoverable via git history. Nothing was deleted.
EOF
