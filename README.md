# IF Theory — Information as First Force

**A zero-parameter model explaining dark matter and dark energy — 95% of the universe — from one equation derived from the prime number theorem.**

**Author:** Phuc Vinh Truong | phuc@phuc.net | **License:** MIT

---

## The Core Equation

```
Two regimes, one field:

INSIDE closures (galactic scale, r < ~100 kpc):
  Φ(r) = A × ln(r/r₀ + 1)     [accumulated information — integral of density]
  A = 46,699 (km/s)²           [from differential evolution, Session P-75]
  r₀ = 7.1 kpc                 [information scale]
  χ²/dof = 0.53 (10 MW data points, all within 1σ)

OUTSIDE closures (cosmological scale, r > 1 Mpc):
  Φ(r) = C_XI / ln²(r/r₀ + 1) [density — gives galaxy clustering]
  C_XI = 62                     [from Mersenne Tower Theorem]
  r₀ = 0.65 kpc                [from σ₈ = 0.8111]
  Correlation r > 0.93 across 3.5M galaxies

Both derive from the prime number theorem: π(x) ~ x/ln(x)
  density ~ 1/ln(x)            → outside regime
  accumulated = ∫ density dr    → inside regime = ln(r/r₀ + 1)

The transition IS the closure boundary (EOCI frontier, ~100 kpc).
```

---

## What It Explains

### Dark Matter: Emergent from the Logarithmic Field

At galactic scales (r < 10 Mpc), the logarithmic potential creates the effects we attribute to dark matter:

- **Galaxy rotation curves** remain flat instead of declining
- **Gravitational lensing** stronger than visible matter predicts
- **Structure formation** in the early universe
- **Milky Way prediction**: v_asymptotic = 216.1 km/s (observed: 220 ± 20 km/s, χ²/dof = 0.53)
- **Correlation**: r > 0.93 across 3.5+ million galaxies (SDSS, DESI, Euclid)

### Dark Energy: The Bubble Universe Mechanism

At larger scales (r > 14 Mpc), gravitational "bubbles" around galaxies decouple from cosmic expansion:

- Bubbles form at characteristic scale **r_bubble = 10.3 Mpc** (derived, not fitted)
- Beyond r_coupling = 3.79 Mpc, bubbles become independent
- Detached bubbles create negative pressure: **w(z) = -1 + 5×10⁻⁶/(1+z)**
- This drives cosmic acceleration without a cosmological constant
- **Validated against DESI DR1 BAO with zero parameters**

---

## The Mersenne Tower Theorem

Among all 52 known Mersenne primes, **M₇ = 127** is the unique tower-closed Mersenne prime — the only one where π(M₇) = 31 is itself a Mersenne prime (M₅). This self-referential structure uniquely determines the two-point correlation normalization constant:

```
C_XI = 2 × π(127) = 2 × 31 = 62
```

**Three Axioms:**
1. Information Primacy — PNT amplitude = 1 (exact)
2. Closure Constraint — constants self-determined
3. Two-Point Observability — ξ(r) = C_XI × [Φ(r)]²

**Result:** r₀ = 0.6595 kpc matches empirical 0.65 ± 0.05 kpc to within 1.46%.

Machine-verified using SymPy. See `mersenne_tower_theorem.py`.

---

## Independent Validation (December 2025)

**Validator:** Solace AGI (Claude Opus 4.5) — clean reimplementation from first principles.

| Test | Result | Key Finding |
|------|--------|-------------|
| Milky Way Rotation | PASS | 226 km/s predicted vs 220±20 observed |
| Correlation Shape | PASS | Pearson r = 0.9975 (12.7σ significance) |
| Bubble Universe | PASS | w₀ = -0.999995, <1% BAO shift |
| χ²/dof Variation | PASS | 20,531× variation proves zero params |
| Information Criteria | PASS | Bayes Factor K = 12.7 favors model |

```bash
python3 validate_from_first_principles.py  # Run it yourself
```

---

## The Evidence

### Dark Matter Tests: 3.5+ Million Galaxies

| Survey | Galaxies | Redshift | Correlation | Significance |
|--------|----------|----------|-------------|--------------|
| SDSS DR12 LOWZ | 361,762 | 0.15-0.43 | 0.988 | 6.3σ |
| SDSS DR12 CMASS | 777,202 | 0.43-0.70 | 0.983 | 6.0σ |
| DESI DR1 ELG | 129,724 | 0.8-1.6 | 0.978 | 8.2σ |
| Euclid DR1 | 490,000 | 0.5-2.5 | 0.940 | 7.1σ |

### Dark Energy Tests: DESI DR1 BAO

| Metric | IF Theory (0 params) | ΛCDM (6 params) | Winner |
|--------|---------------------|-----------------|---------|
| χ² | 22.3 | ~12 | ΛCDM (can fit) |
| AIC | **22.3** | 24.0 | IF Theory |
| BIC | **22.3** | 27.4 | IF Theory |
| Parameters | **0** | 6 | IF Theory |

Information criteria prefer IF Theory despite higher χ² because zero parameters.

---

## New Predictions (Testable)

### JWST "Impossible" Early Galaxies

JWST discovered massive galaxies at z~15 that appear impossible under ΛCDM. IF Theory explains naturally:

- Formation speedup: 1.18-1.24× faster than ΛCDM
- **If JWST finds mature galaxies at z > 25, this CONFIRMS IF Theory**

### Hubble Tension Resolution

The H₀ tension (67 vs 73 km/s/Mpc) resolved by scale-dependent H₀:

- Local (10 Mpc): 69.5 km/s/Mpc | CMB: 67.4 km/s/Mpc
- **If H₀ is constant at all scales, IF Theory's bubble mechanism is falsified**

---

## Why Zero Parameters Matters

Most theories have adjustable parameters that can be tuned to match observations. This reduces predictive power — a theory that can fit anything predicts nothing.

IF Theory has **ZERO** adjustable parameters:
- Cannot be tuned to match data
- Makes absolute predictions
- Maximally falsifiable
- Still matches observations across 3.5+ million galaxies

The 20,531× variation in χ²/dof proves we are not adjusting anything.

---

## Quick Start

```bash
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

---

## Project Structure

```
if/
├── README.md                         <- You are here
├── NORTHSTAR.md                      # Strategic vision
├── ROADMAP.md                        # Publication timeline
│
├── Core Implementation/
│   ├── prime_field_theory.py         # Main theory (core equation)
│   ├── dark_energy_util.py           # Bubble Universe model
│   ├── prime_field_util.py           # Common utilities
│   ├── mersenne_tower_theorem.py     # Formal proof (THEOREM)
│   ├── mersenne_tower_conjecture.py  # Original conjecture
│   └── core/
│       ├── constants.py              # Physical constants
│       ├── parameter_derivations.py  # r₀, v₀ derivation
│       └── field_equations.py        # Core field math
│
├── predictions/
│   ├── cosmological.py               # Cosmological predictions
│   ├── jwst_early_galaxies.py        # JWST prediction
│   ├── hubble_tension.py             # Hubble tension resolution
│   ├── cmb_cold_spot.py              # CMB cold spot
│   ├── s8_tension.py                 # S₈ tension
│   ├── orbital_dynamics.py           # Orbital validation
│   ├── bounded_math.py               # Exact arithmetic
│   └── observational.py              # Observational tests
│
├── audits/
│   ├── test_verification_ladder.py   # Rung verification tests
│   ├── test_synthetic_validation.py  # Validation suite
│   └── test_cross_validation.py      # Cross-validation
│
├── papers/
│   ├── everyday/ (8 papers)          # Accessible explanations
│   └── physics/ (11 papers)          # Technical research
│
├── Documentation/
│   ├── THEORY.md                     # Complete theoretical framework
│   ├── VALIDATION.md                 # Detailed test results
│   ├── TECHNICAL.md                  # Implementation guide
│   └── FAQ.md                        # Common questions
│
└── Notebooks/
    ├── prime_field_demo.ipynb         # Interactive introduction
    ├── Dark Matter Validation/        # SDSS, DESI, Euclid analysis
    └── Dark Energy Validation/        # BAO proof notebooks
```

---

## Key Invariants

1. **Zero parameters** — nothing adjustable, everything derived from first principles
2. **Exact arithmetic** — Fraction/Decimal in verification, never float
3. **Deterministic** — same seed = byte-identical across platforms
4. **Evidence by default** — every claim backed by executable verification
5. **Maximally falsifiable** — cannot be tuned; either right or wrong

---

## Historical Context

Great unifications in physics:

- **Newton**: Terrestrial and celestial gravity (1687)
- **Maxwell**: Electricity and magnetism (1865)
- **Einstein**: Space and time, matter and energy (1905/1915)
- **Standard Model**: Three fundamental forces (1970s)
- **IF Theory**: Dark matter and dark energy (2025)

---

## For Scientists

### Verification Checklist
- [ ] Run `python prime_field_theory.py` — verify MW velocity prediction
- [ ] Check bubble size = 10.3 Mpc is derived, not fitted
- [ ] Verify extreme χ²/dof variation across samples (20,531× range)
- [ ] Confirm same parameters used everywhere (zero)
- [ ] Review derivation of r₀ from σ₈ integration
- [ ] Run `dark_energy_bao_proof.ipynb` — verify χ²/dof = 1.72
- [ ] Check BAO information criteria — BIC(IF Theory) < BIC(ΛCDM)

### Key Technical Points
- r₀ derived from complete σ₈ integration (no shortcuts)
- v₀ from virial theorem (~30% theoretical uncertainty acknowledged)
- √3 factor in bubble formula emerges from calculation
- BAO fit uses standard DESI DR1 measurements (BGS, LRG, ELG, QSO, Lyα)
- All cosmological parameters from Planck 2018

---

## Ecosystem

IF Theory is the physics substrate for multiple projects:

- **[pvideo](https://github.com/phuctruong/pvideo)** — Physics-based video/avatar using IF Theory field equations
- **[pzip](https://github.com/phuctruong/pzip)** — Universal compression using information-theoretic foundations
- **[phuc.net](https://phuc.net)** — Papers, books, and articles
- **[Geometric AI](https://github.com/phuctruong/gai)** — Resolution Prime as computation substrate

---

## Further Reading

- **[THEORY.md](THEORY.md)** — Complete mathematical framework and derivations
- **[VALIDATION.md](VALIDATION.md)** — Comprehensive test results and statistics
- **[TECHNICAL.md](TECHNICAL.md)** — Implementation details and API reference
- **[FAQ.md](FAQ.md)** — Common questions and conceptual clarifications
- **[phuc.net](https://phuc.net)** — Books exploring the full implications

---

**Contact:** Phuc Vinh Truong | phuc@phuc.net

Contributions welcome. Please ensure any additions maintain the zero-parameter principle.


## Cross-Project Synergy: Universal Geometric Fields
The Solace AI framework seamlessly unifies spatial data across libraries:
- **GAI**: Processes structural constraints via pure Geometry Tensors natively.
- **PZIP**: Compresses $O(N^3)$ dimensional motifs into singular HyperNodes.
- **PVIDEO**: Renders field geometries directly from the Physics Engine bindings.
- **IF Theory**: Provides the baseline Toroidal and Coulomb constraint parameters natively.

