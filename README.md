# Information Field Theory — A Maximally-Rigid Cosmology, Honestly Scored

**A logarithmic potential with one stellar M/L per galaxy (MOND's own
convention) and zero shape freedom — which beats MOND head-to-head in
massive spirals, loses to it in dwarfs, and stakes its cosmological
claims on pre-registered bets it can publicly lose.**

Author: Phuc Vinh Truong · phuc@phuc.net · MIT License · Version 2.0.0
Repository: https://github.com/phuctruong/if
External referee review: [`audits/PEER_REVIEW_FABLE5_2026-06-12.md`](audits/PEER_REVIEW_FABLE5_2026-06-12.md) · Honest composite: **73/100** (SCORE.md §"Scoreboard v2")

---

## TL;DR

One logarithm, two regime faces (deriving the transition is THE open
problem), one scale, one M/L per galaxy.

```
Galactic (validated):  Φ_gal(r) = ln(r/r₀ + 1)  →  v² = v₀²·R/(R+r₀)
LSS (shape-only):      Φ(r) = 1/log(r/r₀ + 1)   →  ξ = C_XI·[Φ]²
                       r₀ = 0.659 kpc, derived from σ₈
```

**EARNED (executed, discriminating):**
- **Massive spirals: beats MOND head-to-head** — median χ²/dof 4.18 vs
  5.86, n=54, identical 1-param fairness (`adversarial/dwarf_regime_split.py`)

**Consistent but not discriminating (truthful labels):**
- MW rotation 0.23σ · SPARC TF slope +1.024 · Pantheon+ 0.932 at SH0ES h
  (IF ≡ Λ to 5 decimals — Λ's successes are inherited, not won)
- LSS ξ(r) shape r ≥ 0.93 on every survey — **an untuned power law
  scores the same or better** (`adversarial/power_law_null_test.py`)
- Hubble tension: void catalogs forward-predict H_local ∈ [70.7, 74.9];
  SH0ES (73.04) lands inside (`predictions/delta_max_forward_prediction.py`)

**At stake (pre-registered, can publicly lose):**
- JWST mature galaxy at z ≥ 25 by 2030 (`BETS.md` #1)
- LSS v2 lock: IF must BEAT the power-law null on DESI DR2+
  (`evidence/lss_bao_locked_prediction/lss_bao_locked_prediction_v2.json`)
- Dwarf regime: the law loses there 2.2× today; any fix must not break
  the massive-spiral win (F-DWARF, `FALSIFIABILITY.md`)

Known failure modes are listed before reviewers find them. That is the
point of this repo.

[Quick Start ↓](#quick-start) · [What's confirmed ↓](#whats-confirmed-vs-hypothesis) · [Math ↓](#mathematical-foundation)

---

## Quick Start

```bash
git clone https://github.com/phuctruong/if
cd if
pip install -r requirements.txt

# Run all tests
python3 -m pytest tests audits -v

# Reproduce the headline result (SPARC Tully-Fisher)
python3 predictions/sparc_per_galaxy_ml.py
```

Each prediction writes JSON to `evidence/<test_name>/` so you can diff against the committed results.

---

## What it explains (plain language)

**Galaxy rotation curves stay flat instead of falling.** A logarithmic field around any mass keeps orbital velocities constant past the visible disk. No invisible halo needed.

**Galaxies cluster the way they do.** The two-point correlation ξ(r) takes the shape [Φ(r)]² with normalization C_XI = 62 — an exact integer from the Mersenne Tower Theorem (π(127) = 31).

**The universe accelerates.** Once galaxies' gravitational "bubbles" reach r = 10.20 Mpc (derived from v₀/H₀·√3), they decouple from cosmic expansion. That decoupling looks like dark energy, but with w(z) ≈ -1 and no Λ to tune.

**The Hubble "tension" isn't a tension.** H₀ varies smoothly between 67.4 (cosmic) and 69.5 (local) because measurement scale interacts with bubble structure.

---

## Scope

This repository is the **public, peer-reviewable cosmology layer** of a broader framework. Protein folding, materials science, and other applications of the same prime substrate live in a separate private repository. Cite this repo for cosmology; the rest is forthcoming.

---

## What's confirmed vs hypothesis

Following Preskill's NISQ-era discipline: separate confirmed, partial, and conjectural claims explicitly. See `SCORE.md` for the per-claim table with σ values and evidence pointers.

### Confirmed (12 PASS)

| Claim | Test | Result |
|---|---|---|
| Mersenne tower C_XI = 62 | `tests/test_mersenne_tower.py` | machine-verified, π(127)=31 from Eratosthenes |
| r₀ canonical | `tests/test_canonical_constants.py` | single source of truth |
| MW v(10 kpc) | `predictions/mw_rotation_sigma_accounting.py` | 0.23σ |
| SPARC Tully-Fisher | `predictions/sparc_per_galaxy_ml.py` | slope +1.024, r +0.950, χ²/dof 7.13 |
| SPARC shape | `predictions/sparc_shape_only_test.py` | median χ²/dof 5.03 (MOND-class) |
| BOSS DR12 ξ(r) | `predictions/boss_published_xi_test.py` | Pearson r +0.98 vs Cuesta 2016 |
| Pantheon+ | `predictions/pantheon_plus_test.py` | χ²/dof 0.932 at SH0ES h |
| Hubble bubble | `predictions/hubble_tension_bubble_test.py` | r_bubble 10.20 Mpc derived |
| δ_max | `predictions/delta_max_derivation.py` | matches calibration to 0.3% |
| JWST early galaxies | independent web search | consistent with JADES-GS-z14-0 |
| Casimir consistency | `predictions/casimir_consistency_test.py` | signal 8 dex below sensitivity |

### Tension within ΛCDM-class bounds (1)

| Claim | Test | Result |
|---|---|---|
| DESI DR1 BAO with w(z) ≈ -1 | `predictions/desi_bao_test.py` | χ²/dof = 1.79, p = 0.044 — same ~2σ tension as standard ΛCDM |

### Open / hypothesis (1)

| Claim | Status |
|---|---|
| Better-than-ΛCDM Bayesian evidence on combined data | OPEN — needs `emcee`/`dynesty` joint fit over BOSS + Pantheon+ + DESI + Planck. Est. 5-10σ Bayes-factor preference per `SCORE.md`. |

---

## Mathematical foundation

The **Mersenne Tower Theorem** (`mersenne_tower_theorem.py`) is the only load-bearing algebraic claim:

> Among the 52 known Mersenne primes M_p = 2^p − 1, **M₇ = 127 is the unique tower-closed Mersenne prime** — the only one for which π(M_p) is itself a Mersenne prime. Specifically: π(127) = 31 = M₅.

Therefore the two-point correlation normalization

```
C_XI = 2 · π(M₇) = 2 · 31 = 62
```

is exact number theory under three axioms:

- **A1** Information Primacy — PNT amplitude = 1
- **A2** Closure Constraint — all constants from prime-counting structure
- **A3** Two-Point Observability — ξ(r) = C_XI · [Φ(r)]²

The axioms are physical postulates, not theorems. See `FALSIFIABILITY.md` for what would falsify each.

### The full system (galactic scale)

```
Φ(r)        = ln(r/r₀ + 1)                         prime field potential
v₀²         = 0.62 · G · M_baryon / R_disk         Freeman 1970 disk virial
v_total²(R) = v_baryon²(R) + v₀² · R / (R + r₀)    rotation curve
```

For cosmological scales, the same Φ(r) shape applies with a regime-dependent r₀ (Resolution Prime: ~0.66 kpc galactic, ~100 Mpc-class LSS).

---

## Reproducing every prediction

```bash
# Galactic rotation
python3 predictions/mw_rotation_sigma_accounting.py
python3 predictions/sparc_per_galaxy_ml.py
python3 predictions/sparc_shape_only_test.py

# Cosmological structure
python3 predictions/boss_published_xi_test.py
python3 predictions/desi_bao_test.py
python3 predictions/pantheon_plus_test.py

# Hubble tension
python3 predictions/hubble_tension_bubble_test.py
python3 predictions/delta_max_derivation.py

# Consistency check
python3 predictions/casimir_consistency_test.py
```

For public-survey data downloads:

```bash
python3 download_survey_data.py --dry-run --surveys sdss desi euclid --products minimal
python3 download_survey_data.py --surveys sdss desi euclid --products minimal
python3 download_survey_data.py --surveys euclid --products euclid-q1 --max-euclid-tiles 3 --max-euclid-attempts 12
```

Files stage to `~/Downloads/if/data/` and are recorded in `DATA_MANIFEST.json` with SHA-256. Euclid tile discovery is fail-closed and bounded by `--max-euclid-attempts` because IRSA catalog listings can be slow or incomplete.

---

## Repository structure

```
if/
├── README.md, SCORE.md, FALSIFIABILITY.md, REPLICATION.md, SIMPLE.md
├── THEORY.md, VALIDATION.md, TECHNICAL.md, FAQ.md, CHANGELOG.md
├── CITATION.cff, LICENSE, requirements.txt
│
├── mersenne_tower_theorem.py     # Algebraic foundation
├── prime_field_theory.py         # Main field implementation
├── prime_field_util.py           # Common utilities
├── dark_energy_util.py           # Bubble Universe model
│
├── predictions/                  # Runnable prediction scripts (one per claim)
├── tests/                        # 13 pytest tests
├── audits/                       # Audit reports + cross-checks
├── evidence/                     # JSON output from every prediction
│
├── papers/everyday/              # 8 accessible papers
├── papers/physics/               # 11 technical research papers
│
├── *.ipynb                       # Interactive notebooks (SPARC, BAO, demo)
├── sdss_util.py, desi_util.py, euclid_util.py   # Survey data loaders
└── download_survey_data.py       # Public-data fetcher
```

---

## Independent validation

- **2025-12** — Solace AGI (Claude Opus 4.5) claimed independent reimplementation, 5/5 tests PASS. *Provenance not externally verified.*
- **2026-04-29** — Claude Opus 4.7 (1M context) audit: baseline 50/100, identified 9 BLOCKERs and 19 SERIOUS findings (`audits/full_gap_report.md`), resolved the 5 most critical via direct code edits + 14 new prediction scripts + 13 pytest tests against 8 staged public datasets. Final composite ~97/100. All commits public.
- **2026-Q3 (planned)** — independent replication on a fresh clone by an external collaborator; CASP15 blind protocol for folding once the gai model code is integrated.

---

## Citation

```bibtex
@software{Truong_2026,
  author    = {Truong, Phuc Vinh},
  title     = {{Information Field Theory: A zero-parameter framework
                for galactic and cosmological observations}},
  year      = 2026,
  month     = apr,
  publisher = {Zenodo},
  version   = {1.0.0},
  doi       = {pending},
  url       = {https://github.com/phuctruong/if}
}
```

Machine-readable: `CITATION.cff`.

---

## Reporting issues / submitting peer review

Open a GitHub issue with:

1. **What claim** are you challenging? (cite the `SCORE.md` row)
2. **What test** did you run, on what data, on what platform?
3. **What result** did you get? (paste the JSON)
4. **What's your conclusion?**

Anything goes — the goal is honest validation.

---

## License

MIT (see `LICENSE`). Maximally open: run the tests, reproduce the validation, fork, extend, contradict.

---

## See also

- **`SIMPLE.md`** — Feynman-style one-page summary
- **`SCORE.md`** — per-claim PASS / TENSION / FAIL / OPEN with σ values
- **`FALSIFIABILITY.md`** — Aaronson-style falsification criteria per claim
- **`BETS.md`** — dated public predictions with sharp pass/fail criteria
- **`REPLICATION.md`** — Curie-style independent-replication protocol
- **`adversarial/`** — runnable scripts that attempt to break the theory
- **`METRICS.md`** — distribution + reach tracking
- **`THEORY.md`, `VALIDATION.md`, `TECHNICAL.md`, `FAQ.md`** — deeper docs
- **`CHANGELOG.md`** — version history
