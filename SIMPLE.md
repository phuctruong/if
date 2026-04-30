# IF Theory on a Postcard

> Per Richard Feynman: "What I cannot create, I do not understand."
> If you can't fit it on one page, you don't yet understand it.

## The whole theory in three lines

```
Φ(r) = ln(r/r_0 + 1)                                  the prime field potential
v_0² = 0.62 · G · M_baryon / R_disk                    Freeman 1970 disk virial
v_total²(R) = v_baryon²(R) + v_0² · R / (R + r_0)      observed rotation curve
```

That's it. With `r_0 = 0.6595 kpc` (derived from σ₈ + Mersenne tower
`C_XI = 2 · π(127) = 62`), the three equations above predict 175 galaxy
rotation curves at Tully-Fisher slope = +1.024 (theoretical 1.000),
Pearson r = +0.950, and median χ²/dof = 7.13 across the SPARC database
with **one parameter per galaxy** (M/L, standard astrophysics).

## What follows from those three lines

| Phenomenon | What you see in the equations |
|---|---|
| Flat rotation curves | `v² → v_0²` as `R → ∞` (FLAT by construction) |
| Tully-Fisher relation | `v_0 ∝ √(M_b)` from baryon virial |
| "Dark matter" | The `v_0²·R/(R+r_0)` term — no particle needed |
| BAO peak | Different `r_0` at LSS scale (Resolution Prime) |
| Hubble tension | Bubble of radius `v_0/H_0·√3 ≈ 10.2 Mpc`; LTB enhancement δ_void ≈ 50% (cosmic-void typical) gives 8.4% local Hubble enhancement |
| JWST early massive galaxies | π(N)/N is dense at small N → structure forms fast early |
| `C_XI = 62` (Mersenne Tower) | Number theory: π(127) = 31 = M_5; 2 × 31 = 62 |

## What it predicts vs. what it doesn't

**It predicts:** rotation curves, galaxy clustering shape, dark-energy
behavior, Hubble tension magnitude, JWST early-galaxy excess sign,
no-dark-matter-particle, no-dark-energy-particle.

**It doesn't predict (yet):** specific protein folds, specific Casimir
asymmetry signal at lab scale, specific particle physics at Standard
Model scale. These are deferred to higher-prime-channel theories.

## What would falsify it

1. A galaxy with measured `v_flat` not matching `v_0 = √(0.62 · G · M_b / R_disk)` to within 30%.
2. A cosmic void measured at < 30% under-density that still produces a Hubble
   tension consistent with naive H_0 expectations.
3. A laboratory measurement at any scale showing prime-channel residuals
   that contradict the Φ = ln(r/r_0+1) functional form.
4. JWST measuring z > 16 galaxies at less massive than ΛCDM Press-Schechter
   prediction (the "old slow universe" scenario).

## What it does NOT yet claim

- **Protein folding**: gai's TM=1.00 is conjecture pending CASP15 blind test.
- **Specific new physics at p > 71**: hypothesis only; needs Casimir
  asymmetry detector at sub-femtometer precision.
- **Consciousness as time radiator** (Tier 5): philosophically grounded,
  empirically untested.

## How to verify (10 minutes)

```bash
git clone https://github.com/phuctruong/if
cd if
pip install -r requirements.txt
python3 -m pytest tests audits -v
python3 predictions/sparc_per_galaxy_ml.py   # → TF slope 1.024, r 0.95
python3 predictions/mw_rotation_sigma_accounting.py   # → 0.23σ
python3 predictions/boss_published_xi_test.py   # → r 0.98
```

That's the whole theory and its whole evidence. Anything more is
elaboration.
