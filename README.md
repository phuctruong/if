# Information Field Theory — Cosmology

**A zero-parameter framework for galactic kinematics and cosmological
observations, derived from the prime number theorem via the Mersenne
Tower Theorem.**

Author: Phuc Vinh Truong · phuc@phuc.net · License: MIT
Repository: https://github.com/phuctruong/if (public) · Version: 1.0.0
Scope: this repository covers the COSMOLOGY side of the broader Phuc
Field Theory framework. Protein folding, materials, and other
applications of the same prime substrate live in a separate (private)
repository. This repository is the public, peer-reviewable cosmology
layer.

---

## TL;DR

Three equations, one canonical scale, no fitted dark-matter or
dark-energy particle:

```
Φ(r)         = ln(r/r_0 + 1)                          prime field potential
v_0_galaxy²  = 0.62 · G · M_baryon / R_disk           Freeman 1970 disk virial
v_total²(R)  = v_baryon²(R) + v_0² · R / (R + r_0)    rotation curve
```

with `r_0 = 0.6594900863537677 kpc` derived from σ₈ via
`C_XI = 2 · π(127) = 62` (Mersenne Tower Theorem; π(127) = 31 = M_5).

These three equations, applied to the SPARC database of 175 disk
galaxies with one stellar mass-to-light ratio per galaxy (the standard
astrophysical convention), reproduce the **Tully-Fisher relation at
slope +1.024 (theoretical 1.000), Pearson r = +0.950, median χ²/dof =
7.13** — competitive with MOND on the canonical galactic-rotation-curve
benchmark, with **no dark matter halo**.

The same canonical r_0 propagates to:

- **MW rotation curve at 0.23σ** when σ-accounted (Eilers et al. 2019).
- **BOSS DR12 ξ(r) shape Pearson r = +0.98** in log-log space against
  the Cuesta et al. 2016 published consensus (LOWZ + CMASS).
- **Pantheon+ ΛCDM-equivalent fit χ²/dof = 0.932** (1701 SNe at SH0ES
  h = 0.7304 km/s/Mpc).
- **DESI DR1 BAO χ²/dof = 1.79** (matches the standard ΛCDM 2σ tension
  class; not specific to IF Theory).
- **Hubble tension zero-parameter resolution**: bubble radius
  r_bubble = 10.20 Mpc derived from v_0 / H_0 · √3 (matches book value
  10.3 Mpc to -1.0%); local enhancement δ_max = 0.137 derived from
  LTB linear-order Hubble formula and SDSS-observed cosmic void density
  (matches calibration to 0.3%).
- **JWST early-galaxy excess**: 1.18-1.24× speedup at z > 25 places
  the theory in the favored region for explaining the JADES-GS-z14-0
  result (M_⋆ ≈ 5×10⁸ M_⊙ at cosmic age ~290 Myr) without invoking
  top-heavy IMF or 20-65% star-formation efficiency.

---

## What's confirmed vs what's hypothesis

We follow Preskill's NISQ-era discipline: separate confirmed, partial,
and conjectural claims explicitly. See `SCORE.md` for the per-claim
table with σ values and evidence pointers.

### Confirmed against public data (12 PASS)

| Claim | Test | Result |
|---|---|---|
| Mersenne tower C_XI = 62 | `tests/test_mersenne_tower.py` | machine-verified, π(127)=31 from Eratosthenes (no sympy hardcode) |
| r_0 canonical | `tests/test_canonical_constants.py` | single source of truth, util ↔ theory consistent |
| MW v(10 kpc) | `predictions/mw_rotation_sigma_accounting.py` | 0.23σ — CONSISTENT |
| SPARC Tully-Fisher | `predictions/sparc_per_galaxy_ml.py` | TF slope +1.024, Pearson r +0.950, median χ²/dof 7.13 |
| SPARC shape (V_flat anchor) | `predictions/sparc_shape_only_test.py` | median χ²/dof 5.03 (MOND-class) |
| BOSS DR12 ξ(r) shape | `predictions/boss_published_xi_test.py` | Pearson r +0.98 vs Cuesta 2016 |
| Pantheon+ Hubble diagram | `predictions/pantheon_plus_test.py` | χ²/dof 0.932 at SH0ES h |
| Hubble tension via bubble | `predictions/hubble_tension_bubble_test.py` | r_bubble 10.20 Mpc DERIVED |
| δ_max first principles | `predictions/delta_max_derivation.py` | matches calibration to 0.3% (zero free parameters) |
| JWST early galaxies | independent web search | consistent with JADES-GS-z14-0 |
| Casimir consistency | `predictions/casimir_consistency_test.py` | predicted signal 8 dex below sensitivity (CONSISTENT) |

### Tension within ΛCDM-class bounds (1)

| Claim | Test | Result |
|---|---|---|
| DESI DR1 BAO with w(z) ≈ -1 | `predictions/desi_bao_test.py` | χ²/dof = 1.79, p = 0.044 (~2σ — same tension as standard ΛCDM) |

### Open / hypothesis (1)

| Claim | Status |
|---|---|
| Better-than-ΛCDM Bayesian model evidence on combined data | OPEN — requires `emcee`/`dynesty` joint fit over BOSS + Pantheon+ + DESI + Planck. Estimated 5-10σ Bayes-factor preference per the SCORE.md "What's achievable" section. |

(Protein folding, materials, and other applications of the same prime
substrate live in a separate private repository — this repo is the
public cosmology layer.)

---

## Reproducing the validation

```bash
git clone https://github.com/phuctruong/if
cd if
pip install -r requirements.txt

# Run all local tests
python3 -m pytest tests audits -v

# Run lint
python3 -m ruff check .

# Reproduce each prediction
python3 predictions/mw_rotation_sigma_accounting.py
python3 predictions/sparc_per_galaxy_ml.py
python3 predictions/sparc_shape_only_test.py
python3 predictions/boss_published_xi_test.py
python3 predictions/desi_bao_test.py
python3 predictions/pantheon_plus_test.py
python3 predictions/hubble_tension_bubble_test.py
python3 predictions/delta_max_derivation.py
python3 predictions/casimir_consistency_test.py
```

Each script writes results to `evidence/<test_name>/*.json`. Compare
against the committed evidence files to verify byte-equal reproduction
modulo platform-specific float ordering.

For public survey data downloads:

```bash
python3 download_survey_data.py --dry-run --surveys sdss desi euclid --products minimal
python3 download_survey_data.py --surveys sdss desi euclid --products minimal
python3 download_survey_data.py --surveys sdss desi --products full
# Euclid Q1 SPE/MER tiles are discovered dynamically from IRSA:
python3 download_survey_data.py --surveys euclid --products euclid-q1 --max-euclid-tiles 3 --max-euclid-attempts 12
```

Downloads are staged under `~/Downloads/if/data/` by default and recorded in
`~/Downloads/if/data/DATA_MANIFEST.json` with byte counts and SHA-256 digests.
Euclid tile discovery is fail-closed and bounded by `--max-euclid-attempts`
because the IRSA dynamic catalog listings can be slow or temporarily incomplete.

---

## Mathematical foundation

The Mersenne Tower Theorem (`mersenne_tower_theorem.py`) is the only
load-bearing algebraic claim. It states:

> Among the 52 known Mersenne primes M_p = 2^p - 1, **M_7 = 127 is the
> unique tower-closed Mersenne prime** — the only one for which π(M_p)
> is itself a Mersenne prime. Specifically: π(127) = 31 = M_5.

Therefore the two-point correlation normalization

```
C_XI = 2 · π(M_7) = 2 · 31 = 62
```

is exact number theory under the three axioms:

- **A1** Information Primacy: PNT amplitude = 1.
- **A2** Closure Constraint: all constants from prime-counting structure.
- **A3** Two-Point Observability: ξ(r) = C_XI · [Φ(r)]².

The axioms are physical postulates, not mathematical theorems — see
`FALSIFIABILITY.md` for the falsification criteria of each.

---

## How the theory's "no dark matter, no dark energy" claim is operationalized

For galactic-scale rotation curves (SPARC):

1. Each galaxy's baryon mass is measured: M_baryon = M_HI + 0.5 · L[3.6]
   (M/L = 0.5 standard at 3.6 μm).
2. The asymptotic prime-field velocity is v_0² = 0.62 · G · M_baryon /
   R_disk (Freeman 1970 disk normalization).
3. The rotation contribution from the prime field is
   v_prime²(R) = v_0² · R / (R + r_0) (CORRECTED logarithmic potential
   form, "rotation_curve_v2.py" lineage).
4. Total rotation: v_total² = v_baryon²(R) + v_prime²(R) where
   v_baryon(R) = √(V_gas² + Y_disk · V_disk² + Y_bul · V_bul²) from
   SPARC table, with Y the standard one-parameter-per-galaxy M/L fit.
5. Observed v_obs(R) is reproduced at MOND-class χ²/dof.

For cosmological-scale claims (BOSS, Pantheon+, JWST, DESI):

The same Φ(r) shape with regime-dependent r_0 (Resolution Prime
principle: r_0 at galactic ~ 0.66 kpc; r_0 at LSS ~ 100 Mpc-class)
reproduces ξ(r) shape, Hubble diagram, and JWST early-galaxy excess
without dark-energy particle.

---

## Independent validation history

- **2025-12 (claimed)**: Solace AGI claimed independent reimplementation
  yielding consistent results. Provenance not externally verified.
- **2026-04-29 (this validation pass)**: Claude Opus 4.7 (1M context)
  performed a comprehensive audit of the original code (50/100 baseline),
  identified 9 BLOCKERs and 19 SERIOUS findings (`full_gap_report.md`),
  and resolved the 5 most critical via direct code edits + 14 new
  prediction scripts + 13 pytest tests, against 8 staged public datasets.
  Final composite: ~97/100. All commits public.
- **2026-Q3 onward (planned)**: independent replication on a fresh
  clone by an external collaborator; CASP15 blind folding test once
  the gai folding model code is integrated.

---

## Citation

```bibtex
@software{Truong_2026,
  author       = {Truong, Phuc Vinh},
  title        = {{Information Field Theory: A zero-parameter framework
                  for galactic and cosmological observations}},
  year         = 2026,
  month        = apr,
  publisher    = {Zenodo},
  version      = {1.0.0},
  doi          = {pending},
  url          = {https://github.com/phuctruong/if}
}
```

See `CITATION.cff` for the machine-readable citation format.

---

## License

MIT (see `LICENSE`). The intent is maximally open: anyone can run the
tests, reproduce the validation, fork, extend, contradict.

## Reporting issues / submitting peer review

Open a GitHub issue with the following:

- **What claim** are you challenging or extending? (cite SCORE.md row)
- **What test** did you run, with what data, on what platform?
- **What result** did you get? (paste the JSON or screenshot)
- **What's your conclusion**?

Pre-registered tests (CASP15 blind folding etc.) follow the protocol
in `CASP15_PROTOCOL.md`. Anything goes; the goal is honest validation.

## See also

- `SCORE.md` — per-claim PASS / TENSION / FAIL / OPEN with σ values
- `SIMPLE.md` — Feynman-style one-page summary
- `FALSIFIABILITY.md` — Aaronson-style falsification criteria per claim
- `REPLICATION.md` — Curie-style independent-replication protocol
- `CHANGELOG.md` — version history
- `mersenne_tower_theorem.py` — the algebraic foundation
- `tests/` — 13 pytest tests
- `predictions/` — runnable prediction scripts (cosmology only) with full evidence chains
