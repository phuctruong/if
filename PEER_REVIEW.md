# IF Theory — Internal Peer Review

**Reviewer:** Anonymous, conducted at the author's request
**Date:** 2026-05-19
**Scope:** all .py files in the repo root, `core/`, `predictions/`, and
`audits/`; all .md files in the repo root; the four headline notebooks.

This is the unredacted internal peer-review that the [README](README.md)
references. It is included publicly because hiding it would defeat the
point of doing it.

---

## 1. What the codebase actually proves

Most of what runs in this repo is honest support code: a logarithmic
field, its gradient, its Laplacian, careful input validation, a few
plotting helpers, and a number of "prediction" modules that compute
closed-form expressions of primes and constants. The load-bearing
result of the entire project is contained in a single function inside
`core/parameter_derivations.py`. That function performs the Peebles
§36 real-space pair-counting integral, runs scipy.optimize on it with
tight tolerances, and inverts σ₈ → r₀ given the Mersenne-tower
normalization C_XI = 62. The output of that inversion is **r₀ = 0.6595
kpc**, and the empirical galaxy-fit value is **r₀ ≈ 0.65 kpc**.
Agreement to ~1.5%. That single coincidence is the entire reason
this project is worth reviewing carefully. Without it, the rest of
the repo is interesting numerology; with it, the rest is context.

`core/field_equations.py` has unusually careful input validation —
None / zero / negative / NaN / Inf each get a specific raised
exception. That kind of care is rare and welcome.

`mersenne_tower_theorem.py` exactly verifies, via `sympy.primepi`, that
π(127) = 31 and that 31 = 2⁵ − 1, and walks the small-p Mersenne
exponents to confirm M₇ is the unique tower-closed Mersenne prime in
the verified set. The number theory is correct; the headline
*uniqueness across all 52 known Mersenne primes* is open (see below).

`audits/test_verification_ladder.py` cross-checks Φ in float versus
Decimal to ~1e-8 agreement. That is a legitimate engineering test of
numerical stability; it is not a test of any physical claim.

The bubble decoupling scale `r_bubble = (v₀/H₀)√3 ≈ 10.3 Mpc` is
genuinely derived from theoretical inputs, not fit.

## 2. What the codebase does NOT prove

The Dec-2025 verdict line "the theory is validated" is overclaim. The
specific overclaims and their honest fixes:

### 2.1 The "5/5 PASS" first-principles validator

Before the 2026-05-19 honesty pass, `audits/validate_from_first_principles.py`
included two tests (χ²/dof variation and information-criteria
comparison) whose values were **hardcoded Python literals**:

```python
chi2_dof_values = {
    "SDSS LOWZ (best)": 1.6,
    "SDSS LOWZ (worst)": 20188,
    # ...
}
```

This block was labeled "Simulated χ²/dof values from actual validation
runs" and the script printed "✓ MINIMAL PARAMETERS CONFIRMED." The
values themselves are reportedly from real notebook runs, but a script
that loads constants from a Python dict and reports PASS is not doing
validation — it is summarizing. The 2026-05-19 pass moved those values
to `audits/reported_fits.json` with documented provenance, renamed the
functions from `validate_*` to `report_*`, and updated the script's
output to explicitly state "loaded from JSON, NOT re-fit." The
honesty issue is resolved; the underlying numbers still need to be
re-derived by a live fit before they can be cited as validation.

### 2.2 Postdiction as prediction

Three "prediction" files use the **observed amplitude of a tension as
a model input**:

- `predictions/hubble_tension.py`: `delta_h0 = H0_LOCAL - H0_CMB`
- `predictions/s8_tension.py`: `delta_s8 = S8_CMB - S8_LENSING`
- `predictions/cmb_cold_spot.py`: scale factor `T_CMB_FLUCTUATIONS * 8`
  tuned to reproduce the observed ~150 μK deficit.

Calling these "predictions" of the observed tensions is wrong; the
observed amplitudes are by construction reproduced. The *shape* of
H₀(r) (monotonically decreasing on ~10 Mpc) and the *shape* of S8(z)
(logarithmic smoothing) ARE falsifiable predictions, and remain
worthwhile. The 2026-05-19 pass added a `POSTDICTION_NOTICE` constant
to each module and rewrote each docstring to distinguish shape (real
prediction) from amplitude (fit by construction).

### 2.3 The Mersenne Tower "theorem"

`mersenne_tower_theorem.py` calls itself a theorem. The number-theoretic
content is real, but the uniqueness lemma L3 is only exactly verified
for Mersenne exponents p ≤ 19. For larger p the file uses the
prime-counting *approximation* π(x) ~ x/ln(x), and the prose argument
for p > 127 ("Mersenne primes are exponentially sparse") is not a
proof — it is a research note. Status downgraded to "partial theorem
+ open uniqueness lemma" with the open-problem framing now in the
module docstring. Two routes to closing the gap are documented:
exact π via Meissel–Mertens–Lehmer/Deléglise–Rivat, or an analytic
argument using Schoenfeld/Dusart bounds.

### 2.4 Silent fallbacks against CLAUDE.md's own ban

The project's CLAUDE.md mandates a fallback ban. Two of the most
critical functions in the σ₈ → r₀ inversion silently substituted
hardcoded values when the optimization failed:

- `core/parameter_derivations.py:212`: `return 0.00065` (the empirical
  r₀) when optimization residual exceeded tolerance.
- `core/parameter_derivations.py:417`: `return np.pi * np.sqrt(3)` when
  the σ₈ pair-counting integral returned non-positive.

This meant the codebase could claim "r₀ derived from σ₈" while
actually returning the empirical value. The 2026-05-19 pass replaced
both with `raise RuntimeError(...)`, honoring the project's own
constitution.

### 2.5 Divergent cosmological constants

Six prediction files contained `except ImportError: SIGMA_8 = 0.8159`
blocks, where the canonical value (in `core/constants.py`) is 0.8111.
On any failed import the cosmology would silently shift. The
2026-05-19 pass stripped all such blocks across `predictions/*.py`;
every module now imports σ₈ from `core.constants` or fails honestly.

### 2.6 Premature historical positioning

The previous README placed IF Theory in a line of "Great unifications"
alongside Newton (1687), Maxwell (1865), Einstein (1905/1915), and the
Standard Model (1970s), with IF Theory listed at "2025." That is a
self-elevation no unrefereed theory has earned. Removed.

### 2.7 The "Cross-Project Synergy" block

The previous README ended with a block describing "Geometry Tensors,"
"HyperNodes," "Toroidal and Coulomb constraint parameters." None of
those terms exist anywhere else in this codebase. Removed as
buzzword-soup violating the project's own Anti-Clippy rule.

## 3. What a hostile referee would still attack

After the 2026-05-19 honesty pass, the remaining attack surface for a
Nature- or ApJ-grade referee is:

1. **BAO global fit is χ²/dof = 1.72 (p = 0.034, 2.1σ tension).**
   Information criteria favor IF Theory because ΛCDM uses 6 free
   parameters, but the raw fit is under tension. Every public statement
   about BAO performance must travel with the 2.1σ disclosure.

2. **v₀ ±30% theoretical uncertainty.** A theory whose key observable
   (rotation velocity) carries ±30% theoretical uncertainty is not
   strongly constraining. The 1σ overlap with observed MW v_asymptotic
   is comfortable but rests on the wide error bar.

3. **Galaxy bias parameters are sample-specific.** `core/constants.py`
   defines per-sample `nu_0` and `nu_z` for LOWZ/CMASS/ELG/LRG/QSO.
   Strict zero-parameter-ism would require these to be derived or
   shared; in practice they are inputs from the observational
   literature. Acceptable, but the "zero free parameters" headline
   must qualify with "zero NEW parameters beyond Planck 2018 plus
   standard galaxy bias modeling."

4. **The σ₈ → r₀ inversion is one number against one number.** It is
   the load-bearing result, but it is a single coincidence. The
   correlation-shape agreement across 3.5M+ galaxies (r > 0.93) is
   what elevates it from coincidence to pattern; but a full
   correlation-function fit (not just a Pearson r against a power-law
   reference) must be done at the catalog level before the pattern is
   considered established.

5. **The Mersenne tower observation is, separately, a beautiful piece
   of number theory.** Its physics interpretation requires the factor
   of 2 in `ξ = C_XI × Φ²` to be a derived rather than definitional
   choice. Currently it is definitional (Axiom A3). This is the
   weakest link in the physics derivation.

## 4. What this review does NOT cover

- The two-regime story in the previous README (inside vs. outside
  closure, with A = 46,699 and r₀ = 7.1 kpc for the inside regime) is
  not reflected in `THEORY.md`, `VALIDATION.md`, or any code module.
  Either propagate it consistently or remove it. The new README
  references the single-regime story to match the codebase.
- The recent Phase-18 commits (CASP15 protein evidence, "Solace
  Software 5.0 master release," quantum superconductor DFT bypass) are
  outside the scope of this physics review. They appear to come from a
  different research thread sharing the same repo.

## 5. Verdict

**The σ₈ → r₀ inversion at 1.5% agreement is a real result.**

It is small enough that any competent reviewer can verify it in an
afternoon, and big enough that — *if* it survives independent
re-derivation with refined σ₈ values and a closed Mersenne uniqueness
lemma — it is a serious contribution. The cross-survey correlation
shape (Pearson r > 0.93 across 3.5M+ galaxies) is real empirical
support for the shape claim, though the χ²/dof values reported in
older drafts need to be reproduced by a live audit script.

**The headline "zero parameters explain 95% of the universe" is not
yet earned by what runs in this repo today.** It can be earned by
closing five specific items (see README §"What 'Nobel-worthy' actually
means here"). Those five items are all tractable; the program is on a
real path.

In the meantime, the most useful thing a serious reader can do is run
`python core/parameter_derivations.py` and see r₀ = 0.6595 kpc fall
out of an integral. That number is the only thing that matters in the
end.

---

*This document is itself a work in progress. If you read it and find
another issue, please open an issue or email phuc@phuc.net.*
