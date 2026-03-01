# IF Theory -- Information as First Force

**Prime Field Theory: a zero-parameter model explaining dark matter and dark energy -- 95% of the universe -- from one equation derived from the prime number theorem.**

> **Version:** 3.0 | **Belt:** White | **Rung Target:** 274177 | **Updated:** 2026-03-01

```
+==============================================================+
|  IF THEORY -- Information as First Force                      |
|  Belt: [][]@@@@@@  (White -- first recipe / theory installs)  |
|  Rung: @@@@@@@@@@@@............  274177 (deterministic proofs) |
|  Papers: 19  |  Predictions: 8  |  Tests: 19  |  Notebooks: 7|
|  Galaxies validated: 3,500,000+  |  Parameters: ZERO          |
|  Pipeline: papers->diagrams->styleguides->tests->code->seal   |
+==============================================================+
```

### 10 Uplift Principles (Paper 17)

| # | Principle | What It Means |
|---|-----------|--------------|
| P1 | Gamification | Belt ladder, rung system, GLOW scores on proofs |
| P2 | Magic Words | DNA equations, /distill, prime channels [2][3][5][7][11][13] |
| P3 | Famous Personas | 47 experts (STORY-47 prime) on call -- load by domain, not always-on |
| P4 | Skills | prime-safety + prime-coder + prime-math (exact arithmetic engine) |
| P5 | Recipes | Proof recipes, simulation scripts, validation notebooks |
| P6 | Access Tools | Python simulations, Jupyter notebooks, exact arithmetic |
| P7 | Memory | 19 papers, validation reports, NORTHSTAR.md |
| P8 | Care | Honest about uncertainty, celebrate real proofs |
| P9 | Knowledge | 19 papers, 9 modules, 7 notebooks, 8 prediction scripts |
| P10 | God | 65537 = divine prime, physics = truth, evidence is sacred |

## The Core Equation

The entire framework follows from one equation based on prime number distribution:

```
Phi(r) = 1/log(r/r0 + 1)

Where:
  Amplitude = 1 (exactly, from the prime number theorem pi(x) ~ x/log(x))
  r0 = 0.65 kpc (uniquely derived from the observed sigma_8 = 0.8111)

Zero adjustable parameters. Everything derived from first principles.
```

## What It Explains

### Dark Matter: Emergent from the Logarithmic Field

At galactic scales (r < 10 Mpc), the logarithmic potential creates dark matter effects:
- Galaxy rotation curves remain flat (Milky Way: 226 +/- 68 predicted vs 220 +/- 20 observed)
- Gravitational lensing stronger than visible matter predicts
- Structure formation in the early universe
- Correlation r > 0.93 across 3.5+ million galaxies (SDSS, DESI, Euclid)

### Dark Energy: The Bubble Universe Mechanism

At larger scales (r > 14 Mpc), gravitational bubbles drive cosmic acceleration:
- Bubble scale: r_bubble = 10.3 Mpc (derived, not fitted)
- Dark energy EoS: w(z) = -1 + 5x10^-6/(1+z)
- BAO fit: chi^2/dof = 1.72, BIC beats LCDM with zero parameters
- DESI DR1 validated across 13 BAO measurements, 7 tracers

### Independent Validation (December 2025)

| Test | Result | Key Finding |
|------|--------|-------------|
| Milky Way Rotation | PASS | 226 km/s predicted vs 220+/-20 observed |
| Correlation Shape | PASS | Pearson r = 0.9975 (12.7 sigma) |
| Bubble Universe | PASS | w0 = -0.999995, <1% BAO shift |
| chi^2/dof Variation | PASS | 20,531x variation proves zero params |
| Information Criteria | PASS | Bayes Factor K = 12.7 favors model |

## New Predictions (December 2025)

### JWST "Impossible" Early Galaxies
JWST discovered massive galaxies at z~15 that appear impossible under LCDM. IF Theory explains naturally:
- Formation speedup: 1.18-1.24x faster than LCDM
- **Testable:** If JWST finds mature galaxies at z > 25, this CONFIRMS IF Theory

### Hubble Tension Resolution
The H0 tension (67 vs 73 km/s/Mpc) resolved by scale-dependent H0:
- Local (10 Mpc): 69.5 km/s/Mpc | CMB: 67.4 km/s/Mpc
- **Testable:** H0 should vary smoothly with distance scale

## Why Zero Parameters Matters

Most theories have adjustable parameters that can be tuned to match observations. This allows them to fit almost anything, reducing their predictive power.

IF Theory has **ZERO** adjustable parameters:
- Cannot be tuned to match data
- Makes absolute predictions
- Maximally falsifiable
- Still matches observations across 3.5+ million galaxies

The 13,700x variation in chi^2/dof proves we are not adjusting anything.

## Quick Start

```bash
# Clone and install
git clone https://github.com/phuctruong/if.git
cd if
pip install -r requirements.txt

# Run main demonstration
python prime_field_theory.py

# Independent validation (5/5 PASS)
python validate_from_first_principles.py

# Predictions
python predictions/jwst_early_galaxies.py
python predictions/hubble_tension.py

# Verification tests
python -m pytest audits/ -v

# Notebooks
jupyter notebook
```

## Evidence (3.5+ Million Galaxies)

### Dark Matter Tests

| Survey | Galaxies | Redshift | Correlation | Significance |
|--------|----------|----------|-------------|--------------|
| SDSS DR12 LOWZ | 361,762 | 0.15-0.43 | 0.988 | 6.3 sigma |
| SDSS DR12 CMASS | 777,202 | 0.43-0.70 | 0.983 | 6.0 sigma |
| DESI DR1 ELG | 129,724 | 0.8-1.6 | 0.978 | 8.2 sigma |
| Euclid DR1 | 490,000 | 0.5-2.5 | 0.940 | 7.1 sigma |

### Dark Energy Tests (DESI DR1 BAO)

| Metric | Bubble Universe | LCDM (6 params) | Winner |
|--------|----------------|-----------------|---------|
| chi^2 | 22.3 | ~12 | LCDM (can fit) |
| AIC | **22.3** | 24.0 | Bubble Universe |
| BIC | **22.3** | 27.4 | Bubble Universe |
| Parameters | **0** | 6 | Bubble Universe |

Information criteria prefer IF Theory despite higher chi^2 because zero parameters.

## Project Structure

```
if/
+-- README.md                             <- You are here
+-- CLAUDE.md                             # v4.0 (10 Uplift Principles)
+-- AGENTS.md                             # v3.0 (coding rules + persona loading)
+-- TODO.md                               # Codex-ready task backlog
+-- NORTHSTAR.md                          # Strategic vision + proof metric
+-- ROADMAP.md                            # Publication timeline
|
+-- Core Implementation/
|   +-- prime_field_theory.py             # Main theory (core equation)
|   +-- dark_energy_util.py               # Bubble Universe model
|   +-- prime_field_util.py               # Common utilities
|   +-- mersenne_tower_theorem.py         # Formal proof
|   +-- mersenne_tower_conjecture.py      # Original conjecture
|
+-- predictions/
|   +-- cosmological.py                   # Cosmological predictions
|   +-- jwst_early_galaxies.py            # JWST early galaxy prediction
|   +-- hubble_tension.py                 # Hubble tension resolution
|   +-- cmb_cold_spot.py                  # CMB cold spot prediction
|   +-- s8_tension.py                     # S8 tension prediction
|   +-- orbital_dynamics.py              # Orbital dynamics validation
|   +-- bounded_math.py                   # Bounded exact arithmetic
|   +-- observational.py                  # Observational predictions
|
+-- audits/
|   +-- test_verification_ladder.py       # Rung verification tests
|   +-- test_synthetic_validation.py      # Synthetic validation
|   +-- test_cross_validation.py          # Cross-validation
|
+-- papers/
|   +-- everyday/ (8 accessible papers)
|   +-- physics/ (11 technical papers)
|
+-- Documentation/
|   +-- THEORY.md                         # Complete theoretical framework
|   +-- VALIDATION.md                     # Detailed test results
|   +-- TECHNICAL.md                      # Implementation guide
|   +-- FAQ.md                            # Common questions
|
+-- Notebooks/
|   +-- Dark Matter Validation/           # SDSS, DESI, Euclid analysis
|   +-- Dark Energy Validation/           # BAO proof notebooks
|
+-- Data Utilities/
    +-- sdss_util.py, desi_util.py, euclid_util.py
```

## Ecosystem Integration

IF Theory is the mathematical substrate powering the Solace ecosystem:

```
IF Theory (this project)
    |
    +-- pvideo: Mersenne Tower + prime field equations = physics engine
    +-- pzip: Information-theoretic foundations = compression algorithm
    +-- phucnet: Papers + books = publication channel
    +-- solaceagi.com: Avatar physics = IF Theory field equations
```

## Verification Tower

| Rung | What It Means | Status |
|------|---------------|--------|
| **641** | Single validation works, evidence captured | LOCKED |
| **274177** | Seed sweep, replay stability, null edge (target) | IN PROGRESS |
| **65537** | Adversarial sweep for claims feeding pvideo | FUTURE |

## Key Invariants

1. **Zero parameters** -- nothing adjustable, everything derived from first principles
2. **Exact arithmetic** -- Fraction/Decimal in verification, never float
3. **Deterministic** -- same seed = byte-identical across platforms
4. **Evidence by default** -- every claim backed by executable verification
5. **Maximally falsifiable** -- cannot be tuned; either right or wrong
6. **Convergence certified** -- iterative methods prove halting with R_p tolerance

## Historical Context

Great unifications in physics:
- **Newton**: Terrestrial and celestial gravity (1687)
- **Maxwell**: Electricity and magnetism (1865)
- **Einstein**: Space and time, matter and energy (1905/1915)
- **Standard Model**: Three fundamental forces (1970s)
- **Prime Field Theory**: Dark matter and dark energy (2025)

---

## Knowledge Network

```
PAPERS (19)                        PREDICTIONS (8)
+-- everyday/ (8 papers)           +-- jwst_early_galaxies.py
+-- physics/ (11 papers)           +-- hubble_tension.py
                                   +-- cmb_cold_spot.py
DOCUMENTATION (4)                  +-- s8_tension.py
+-- THEORY.md                      +-- orbital_dynamics.py
+-- VALIDATION.md                  +-- cosmological.py
+-- TECHNICAL.md                   +-- observational.py
+-- FAQ.md                         +-- bounded_math.py

NOTEBOOKS (7)                      AUDITS (19 tests)
+-- Dark Matter/ (3 notebooks)     +-- test_verification_ladder.py
+-- Dark Energy/ (2 notebooks)     +-- test_synthetic_validation.py
+-- prime_field_demo.ipynb         +-- test_cross_validation.py
```

## Belt Progression

| Belt | Criteria | Status |
|------|----------|--------|
| **White** | **First recipe / theory installs** | **CURRENT** |
| Yellow | All 19 papers in Prime Paper Format + diagrams | NEXT |
| Orange | Stillwater Store skill submitted (IF Theory validator) | -- |
| Green | ApJ paper accepted + Mersenne Tower formalized | -- |
| Blue | pvideo integration live (IF Theory invariant checker) | -- |
| Black | Physics = truth. Information = first force. Primes = law. | -- |

---

**Status:** Paper format migration + 10 Uplift Principles (2026-03-01)
**Rung Target:** 274177 (Deterministic proofs, seed sweep, replay stable)
**DNA:** `F = I x G; information IS reality; compression = understanding; prime_frequencies -> all_physics`
**Next:** Format the papers. Build the diagrams. Publish the proofs. Trust the evidence. Love the physics.

The universe is built on prime foundations. We are here to prove it.

---

**Author:** Phuc Vinh Truong | phuc@phuc.net
**License:** MIT
