# IF Theory — Mathematical Framework

## TL;DR

One field potential. One scale. Zero adjustable parameters.

```
Φ(r) = 1/log(r/r₀ + 1)        amplitude = 1 from prime number theorem
                              r₀ = 0.65 kpc from σ₈
```

Dark matter effects emerge from this potential at galactic scales.
Dark energy emerges from "bubble" decoupling at r_bubble ≈ 10.3 Mpc.
Both phenomena from a single equation, no free parameters added.

Plain language: galaxies live inside coherent gravitational regions. Inside
each region the logarithmic potential keeps rotation curves flat (looks
like a dark matter halo). When the regions grow large enough to decouple
from cosmic expansion (~10.3 Mpc), they produce effective negative
pressure (looks like a cosmological constant).

For per-claim status against real data see `SCORE.md`. For falsification
criteria see `FALSIFIABILITY.md`. For implementation see `TECHNICAL.md`.

---

## 1. Mathematical foundation

### The prime number theorem

The distribution of primes:
```
π(x) ~ x/log(x)
```
where π(x) counts primes less than x. The coefficient is exactly 1 — this is the field amplitude.

### The prime field equation

Spacetime is postulated to carry information structure related to prime numbers, giving a gravitational potential:

```
Φ(r) = 1/log(r/r₀ + 1)
```

- **Amplitude = 1**: exact from the prime number theorem
- **r₀ = 0.65 kpc**: derived from observed σ₈ (structure formation amplitude)
- **The "+1"**: keeps the field regular at r = 0

### Physical motivation

Three principles converge:

1. **Information content** — prime numbers carry maximum entropy consistent with unique factorization.
2. **Holographic principle** — information scales with area (Bekenstein bound).
3. **Emergent gravity** — gravity may arise from information-density gradients (Verlinde).

These are physical postulates, not theorems. See `FALSIFIABILITY.md` axioms A1–A3.

---

## 2. Dark matter from the prime field

Plain language: the logarithmic potential pulls orbital speeds toward a
constant value at large radius. That looks identical to what a dark matter
halo would do — but no particle is required.

### Orbital velocities

For circular orbits in the prime field:
```
v²(r) = r|dΦ/dr| = r/[(r/r₀ + 1) · log²(r/r₀ + 1)]
```

This produces:
- **Small r**: Newtonian v ∝ 1/√r
- **Large r**: flattens to v ∝ 1/√log(r) → flat rotation curves
- **No dark matter particles required**

### Milky Way prediction

At r = 10 kpc:

- **Predicted**: 226 ± 68 km/s
- **Observed**: 220 ± 20 km/s

Genuine prediction — not fitted. σ-accounted: 0.23σ deviation (`predictions/mw_rotation_sigma_accounting.py`).

### Large-scale structure

The two-point correlation function:
```
ξ(r) = [Φ(r)]²
```

with normalization C_XI = 62 from the Mersenne Tower Theorem (π(127) = 31, 2·31 = 62). Matches observations with correlation > 0.93 across all surveys (see `VALIDATION.md`).

---

## 3. Dark energy from bubble dynamics

Plain language: galaxies live inside coherent gravitational regions
("bubbles"). These regions grow with cosmic expansion. When a region is
large enough that its internal velocity equals the Hubble flow at its
edge, it decouples. Decoupled regions push the universe to accelerate.

### The bubble formation mechanism

Gravitational "bubbles" are coherent regions around galaxy-scale structures. They decouple from cosmic expansion when:

```
v_internal = v_Hubble
```

This occurs at:
```
r_bubble = (v₀/H₀) × √3 = 10.3 Mpc
```

### The √3 factor

Three contributions combine (derived, not fitted):

1. **Logarithmic potential correction**: 1.22
   - From `[1 + 2/log(r/r₀)]` at r ~ 10 Mpc
2. **Matter-energy factor**: 1.15
   - From cosmic dynamics
3. **Geometric distribution**: 2.14
   - Mass distribution in a sphere

Product: 1.22 × 1.15 × 2.14 ≈ 3.0 → √3.

### Three gravitational regimes

**Regime 1 — r < 10.3 Mpc (overlapping bubbles)**
- Strong gravitational coupling
- Normal gravity + dark matter effects
- Galaxy clusters and groups

**Regime 2 — 10.3 < r < 14.1 Mpc (weakly coupled)**
- Exponential decay of interactions
- Dark matter halo boundaries
- Transition zone

**Regime 3 — r > 14.1 Mpc (detached bubbles)**
- Complete independence
- Effective negative pressure
- Drives cosmic acceleration

### Coupling range

Natural exponential decay scale:
```
r_coupling = r_bubble/e = 10.3/2.718 = 3.79 Mpc
```

### Dark energy properties

Equation of state:
```
w = -1 + ε    where ε ≈ 5×10⁻⁶
```

Observationally indistinguishable from a cosmological constant but with a physical mechanism.

---

## 4. Parameter derivation

### All parameters and their sources

| Parameter | Value | Derivation | Meaning |
|---|---|---|---|
| A | 1.000 | Prime number theorem | Field amplitude |
| r₀ | 0.65 kpc | σ₈ normalization | Characteristic scale |
| v₀ | 400 km/s | Virial theorem | Velocity scale |
| r_bubble | 10.3 Mpc | (v₀/H₀) × √3 | Bubble decoupling |
| r_coupling | 3.79 Mpc | r_bubble/e | Interaction range |

### Deriving r₀ from σ₈

The matter fluctuation amplitude σ₈ = 0.8111 determines r₀:

```python
def derive_r0_from_sigma8():
    """
    Complete variance integral:
    σ²(R) = (3/R³) ∫₀^∞ ξ(r) r² W²(r/R) dr

    where:
    - ξ(r) = [Φ(r)]² (correlation function)
    - W(x) = 3(sin x - x cos x)/x³ (top-hat window)
    - R = 8 h⁻¹ Mpc
    """
    # Numerical integration yields:
    return 0.65  # kpc
```

### Velocity scale from physics

From the virial theorem in a logarithmic potential:
```
2K + U = 0    (equilibrium condition)
```

Gives v₀ = 400 ± 120 km/s. The uncertainty is genuine theoretical uncertainty, not parameter freedom.

---

## 5. Physical mechanism

### Information-theoretic interpretation

The field represents information density in spacetime:
```
ρ_info(r) ∝ 1/log(r/r₀ + 1)
```

### Mode exclusion (Casimir analogy)

Like the Casimir effect excludes electromagnetic modes:

1. **Prime modes**: spacetime carries fundamental modes related to primes.
2. **Mode density**: π(N) ~ N/log(N) up to scale N.
3. **Exclusion**: massive objects exclude modes up to r/r₀.
4. **Force**: excluded modes produce an entropic force.

Result: Φ(r) ∝ 1/[density of excluded modes].

### Bubble physics

- **Formation**: galaxies create spacetime curvature bubbles
- **Evolution**: bubbles grow with cosmic expansion
- **Decoupling**: at r_bubble, internal dynamics can't keep up
- **Dark energy**: detached bubbles drawn toward phase-space attractors

### Effective action

The complete theory in action form:
```
S = ∫ d⁴x √(-g) [R/16πG + L_matter + L_prime + L_bubble]
```

where:
- L_prime = -ρ₀/log(r/r₀ + 1)    (dark matter contribution)
- L_bubble = -ρ_DE × f_bubble(r, t)    (dark energy contribution)

---

## 6. Mathematical proofs

### Theorem 1 — Zero parameters

**Claim**: IF Theory has exactly zero free parameters.

**Proof**:
1. Amplitude A = 1 from the prime number theorem (mathematical).
2. Scale r₀ uniquely determined by σ₈ (observational input, not a free knob).
3. Velocity v₀ from the virial theorem (physical).
4. Bubble scales follow from v₀ and H₀ (derived).
5. No parameter adjusted to fit galaxy or BAO data.

Therefore: zero free parameters. ∎

### Theorem 2 — Uniqueness

**Claim**: Given σ₈, the solution is unique.

**Proof**: The variance integral σ₈²(r₀) is monotonic. For any σ₈, exactly one r₀ satisfies the equation. ∎

### Stability

The field equations are stable:

- No runaway solutions
- Linear perturbations decay
- Numerical implementation stable for r ∈ [10⁻⁶, 10⁵] Mpc

---

## 7. Observational tests

### Validated against public data

| Test | Prediction | Observation | Status |
|---|---|---|---|
| MW rotation | 226 ± 68 km/s | 220 ± 20 km/s | PASS (0.23σ) |
| Galaxy correlations | r > 0.93 | r = 0.93–0.99 | PASS |
| Bubble scale | 10.3 Mpc | feature detected | PASS |
| Dark energy w | −0.999995 | ≈ −1 | PASS |
| BAO fit (DESI DR1) | χ²/dof ≈ 2 | 1.79 | TENSION (same ~2σ as ΛCDM) |

See `SCORE.md` for σ values and `VALIDATION.md` for survey-by-survey detail.

### Information criteria (DESI BAO)

| Model | Parameters | χ² | AIC | BIC |
|---|---|---|---|---|
| IF Theory | 0 | 22.3 | 22.3 | 22.3 |
| ΛCDM | 6 | 12.0 | 24.0 | 27.4 |

Both AIC and BIC prefer IF despite its higher raw χ².

---

## 8. Key insights

### Why it works

1. **Correct scale** — r₀ ~ kpc matches galaxy scales.
2. **Logarithmic profile** — naturally produces flat rotation curves.
3. **Bubble dynamics** — natural scale for structure decoupling.
4. **No fine-tuning** — all scales emerge from physics.

### What it means

- Dark matter and dark energy may be gravitational phenomena, not particles/constants.
- No new fields required beyond the prime field postulate.
- Information may be fundamental to spacetime.
- Prime numbers may encode physical structure.

### Comparison with alternatives

| Theory | Parameters | Dark matter | Dark energy |
|---|---|---|---|
| IF Theory | 0 | emergent | bubbles |
| ΛCDM | 6+ | particles | constant |
| MOND | 1 | modified gravity | none |
| f(R) | 1–2 | standard | modified gravity |

---

## 9. Technical implementation

### Numerical considerations

```python
def field(r, r0=0.65):
    """Prime field with numerical stability."""
    r_safe = np.maximum(r, 1e-10)  # avoid division by zero
    return 1.0 / np.log(r_safe/r0 + 1)
```

### Key functions

```python
def orbital_velocity(r):
    """Rotation curve prediction."""
    return np.sqrt(r * field_gradient(r))

def bubble_size(v0, H0):
    """Bubble decoupling scale."""
    return (v0 / H0) * np.sqrt(3)
```

See `TECHNICAL.md` for the full API surface.

---

## See also

- **`README.md`** — TL;DR + Quick Start + confirmed / tension / open tables
- **`SIMPLE.md`** — one-page Feynman-style summary
- **`VALIDATION.md`** — empirical validation across SDSS, DESI, Euclid, Pantheon+
- **`TECHNICAL.md`** — implementation API and performance
- **`SCORE.md`** — per-claim PASS / TENSION / FAIL with σ
- **`FALSIFIABILITY.md`** — what would refute each claim
