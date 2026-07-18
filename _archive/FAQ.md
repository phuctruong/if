# IF Theory — Frequently Asked Questions

Short answers to common questions. For per-claim PASS/TENSION/FAIL with
σ values, see `SCORE.md`. For the load-bearing equations and verification
commands, see `README.md`.

---

## Theoretical foundation

### Q: Why connect prime numbers to gravity?

Three lines of evidence converge:

1. **Information may be fundamental.** The holographic principle has information scaling with area, not volume — suggesting information underlies spacetime geometry.
2. **Primes are a natural basis.** Prime numbers carry maximum entropy consistent with unique factorization — the "most random" distribution available.
3. **Emergent gravity.** Following Verlinde, gravity may emerge from information gradients. The logarithmic form Φ ∝ ln(r/r₀+1) appears naturally from prime counting via π(x) ~ x/log(x).

The field equation represents information density in spacetime. This is a physical postulate (axiom A1 in `FALSIFIABILITY.md`), not a theorem.

### Q: How do "informational modes" produce the log form?

Through mode counting, analogous to the Casimir effect:

```
Force ∝ -(excluded modes) × (energy per mode)
```

For the prime field:

1. **Mode density**: Number of prime modes up to scale N is π(N) ~ N/log(N).
2. **Mode exclusion**: A mass of size r excludes modes up to r/r₀.
3. **Result**: Φ(r) ∝ 1/[density of excluded modes] = 1/log(r/r₀ + 1).

### Q: What's the action principle?

```
S = ∫ d⁴x √(-g) [R/16πG + L_matter + L_prime + L_bubble]
```

- `L_prime = -ρ₀/log(r/r₀ + 1)` produces the dark-matter-like contribution
- `L_bubble = -ρ_DE × f_bubble(r, t)` produces the dark-energy-like contribution

Both enter through the stress-energy tensor — General Relativity remains the framework.

---

## The Bubble Universe

### Q: What are the "bubbles"?

Coherent gravitational regions around galaxies:

- Zones where a galaxy's own curvature dominates
- Scales where internal dynamics matter more than cosmic expansion

Not literal bubbles — gravitational domains that can decouple from the Hubble flow.

### Q: How do bubbles produce dark energy without new physics?

A phase transition in gravitational coupling:

1. Galaxies create curved spacetime regions (bubbles).
2. Bubbles grow with cosmic expansion.
3. At ~10.3 Mpc, internal velocity equals Hubble flow → decoupling.
4. Beyond ~14.1 Mpc, bubbles become independent.
5. Independent bubbles produce effective negative pressure → dark energy.

No new fields. Emergent behavior from gravity at specific scales.

### Q: Why √3 in the bubble formula?

Derived, not assumed. Three factors combine:

- Logarithmic correction at 10 Mpc: 1.22
- Matter-energy dynamics: 1.15
- Geometric mass distribution: 2.14

Product: 1.22 × 1.15 × 2.14 = 3.00 → √3.

### Q: How does this address the cosmological constant problem?

| | ΛCDM | Bubble Universe |
|---|---|---|
| Λ value | requires fine-tuning to 10⁻¹²⁰ | not used |
| Why now? | unexplained coincidence | natural — happens when universe is old enough for bubbles |
| Mechanism | none | r_bubble = v₀/H₀ × √3 |

---

## Zero parameters

### Q: Is this *really* zero parameters?

Zero **adjustable** parameters. Nothing fitted to galaxy or BAO data.

The distinction:

- We use cosmological inputs (H₀, Ω_m, σ₈) like all theories
- We add ZERO additional parameters
- All scales derive from first principles
- Cannot improve fit by adjusting anything

The ~30% uncertainty in v₀ is theoretical uncertainty, not parameter freedom.

### Q: What about cosmological parameters?

Like the speed of light — external measurements describing the universe we're in, not knobs we tune.

### Q: How are all the bubble scales derived?

From the decoupling condition v_internal = v_Hubble:

- r_bubble = (v₀/H₀) × √3
- r_coupling = r_bubble/e (natural exponential decay scale)
- r_detachment = r_bubble + r_coupling

Change v₀ or H₀ and the scales change predictably. No independent freedom.

---

## Statistical interpretation

### Q: How should we read high χ²/dof values?

Differently than for parametric models. For zero-parameter models:

1. **Cannot minimize χ²** — no parameters to tune.
2. **High values are expected** — measures absolute agreement, not best-fit agreement.
3. **Correlation matters** — shape agreement is the primary metric.
4. **Variation is signal** — wide χ²/dof variation proves no tuning is happening.

### Q: Why does χ²/dof vary from 2.4 to 32,849?

The variation *itself* is evidence of zero parameters:

| Parameters | Typical χ²/dof variation |
|---|---|
| 2+ | ~2× between samples |
| 1 | ~4× |
| 0 | 13,700× (this work) |

A model with free parameters would always tune to χ²/dof ≈ 1.

### Q: Is χ²/dof = 1.72 good for BAO?

For zero parameters against DESI DR1, yes:

- ΛCDM: χ²/dof ~ 1.0 with 6 fitted parameters
- IF Theory: χ²/dof = 1.79 with 0 parameters (p = 0.044, ~2σ — same tension as standard ΛCDM)

Information criteria account for the complexity penalty:

- AIC: 22.3 (IF) vs 24.0 (ΛCDM)
- BIC: 22.3 (IF) vs 27.4 (ΛCDM)

---

## Implementation

### Q: How can I verify no hidden parameters?

```python
# 1. Amplitude from prime number theorem
assert amplitude == 1.0  # mathematical, not fitted

# 2. Scale from σ₈
r0 = derive_r0_from_sigma8()  # full integration shown

# 3. Velocity from physics
v0 = virial_theorem_velocity()  # no galaxy data used

# 4. MW prediction is NOT 220
mw_velocity = theory.predict(10.0)
assert abs(mw_velocity - 220) > 1  # not calibrated
```

### Q: Why convergence warnings?

The σ₈ integration spans many orders of magnitude. Warnings are normal. The code has documented fallback values; everything is transparent.

### Q: Should I install Numba?

Yes — 10-20× speedup on pair counting:

```bash
pip install numba
```

The code auto-detects and uses it.

---

## Physical interpretation

### Q: How does this differ from MOND?

| Aspect | IF Theory | MOND | ΛCDM |
|---|---|---|---|
| Free parameters | 0 | 1 (a₀) | 6+ |
| Dark matter | emergent | modified gravity | particle |
| Dark energy | bubbles | none | Λ |
| Basis | prime numbers | empirical | phenomenology |

### Q: Compatible with General Relativity?

Yes. We don't modify Einstein's equations. We add:

- Information density (dark-matter-like contribution)
- Bubble dynamics (dark-energy-like contribution)
- Both through the stress-energy tensor
- GR remains the framework

### Q: What are "prime attractors"?

Hypothesised preferred configurations in phase space:

- Detached bubbles evolve toward specific states
- States relate to prime number patterns
- Produces effective negative pressure

This is the most speculative aspect — listed as OPEN in `FALSIFIABILITY.md`.

---

## Validation

### Q: What predictions are validated against real public data?

See `SCORE.md` for the full table with σ values. Headline:

| Claim | Test | Result |
|---|---|---|
| MW v(10 kpc) | `predictions/mw_rotation_sigma_accounting.py` | 0.23σ |
| SPARC Tully-Fisher | `predictions/sparc_per_galaxy_ml.py` | slope +1.024, r +0.950 |
| BOSS DR12 ξ(r) | `predictions/boss_published_xi_test.py` | Pearson r +0.98 vs Cuesta 2016 |
| Pantheon+ | `predictions/pantheon_plus_test.py` | χ²/dof 0.932 at SH0ES h |
| Hubble bubble | `predictions/hubble_tension_bubble_test.py` | r_bubble 10.20 Mpc derived |
| JWST early galaxies | independent literature search | consistent with JADES-GS-z14-0 |
| DESI DR1 BAO | `predictions/desi_bao_test.py` | χ²/dof = 1.79 (same ~2σ tension as ΛCDM) |

### Q: What would falsify the theory?

Sharp criteria per claim are in `FALSIFIABILITY.md`. Examples:

1. MW v(10 kpc) outside 200–240 km/s (would invalidate the σ-accounted PASS).
2. ξ(r) on a clean galaxy sample with log-log Pearson r < 0.90.
3. Cosmic void catalogue showing typical voids < 10% under-dense.
4. JWST z > 16 galaxy at much lower mass than the prediction.
5. w(z) at z ≈ 0.5 differing from −1 by more than 5σ (combined Pantheon+ / DESI / Planck).

### Q: Why prefer this over ΛCDM?

| Criterion | IF | ΛCDM |
|---|---|---|
| Free parameters | 0 | 6+ |
| Fine-tuning | none required | Λ to 10⁻¹²⁰ |
| Mechanism for dark energy | bubbles (geometric) | unexplained constant |
| Falsifiability | maximum | high |
| Information criteria (DESI DR1) | AIC 22.3, BIC 22.3 | AIC 24.0, BIC 27.4 |

These advantages are tentative until independent replication (see `REPLICATION.md` for the protocol).

---

## See also

- **`README.md`** — overview with TL;DR, Quick Start, and confirmed/tension/open tables
- **`SIMPLE.md`** — one-page Feynman-style summary
- **`THEORY.md`** — full mathematical framework
- **`VALIDATION.md`** — detailed empirical results across surveys
- **`SCORE.md`** — per-claim PASS / TENSION / FAIL with σ
- **`FALSIFIABILITY.md`** — explicit falsification criteria
- **`REPLICATION.md`** — independent-replication protocol
