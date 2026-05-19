# IF Theory — Information as the First Force

> **A research program testing whether one logarithmic equation, normalized
> by the prime number theorem, can stand against the dark sector of the
> universe with no new free parameters.**

**Author:** Phuc Vinh Truong | phuc@phuc.net
**License:** MIT
**Status:** open research, peer review welcome and demanded
**Last honesty pass:** 2026-05-19

---

## What this project is — in one paragraph

Most attempts to explain dark matter and dark energy add new particles,
new fields, or new free parameters. This project asks a different
question: what if the prime number theorem already encodes the right
shape, and the dark sector is what you get when you measure that shape
without realizing what you're looking at? The core proposal is a single
field Φ(r) = 1 / log(r/r₀ + 1), with the normalization C_XI = 62 pulled
from a number-theoretic observation about Mersenne primes. Plug that into
the standard Peebles σ₈ pair-counting integral and r₀ falls out at
**0.6595 kpc** — agreeing with the empirically-fit galaxy correlation
scale of **0.65 kpc** to about **1.5%**. That single coincidence is what
makes this worth taking seriously. Everything else in this repo is
context for it.

---

## The honest scoreboard

The full peer-review is in [PEER_REVIEW.md](PEER_REVIEW.md). Short version:

**What is real and worth your time:**

- **The σ₈ → r₀ inversion.** Real Peebles §36 integral, scipy.quad with
  tight tolerances, scipy.optimize.minimize_scalar. Given C_XI = 62 and
  Planck σ₈ = 0.8111, the integral lands at r₀ = 0.6595 kpc. Independent
  galaxy-correlation fits give r₀ ≈ 0.65 kpc. 1.5% agreement with zero
  *new* parameters fit to galaxy data is non-trivial.
- **The shape of the dark-matter correlation.** Across SDSS DR12,
  DESI DR1, and Euclid DR1, the Pearson correlation between the model's
  Φ(r)² and the measured galaxy correlation function exceeds 0.93 in
  every sample, reaching 0.994 at best. The χ²/dof varies by ~10⁴
  across samples, which is the fingerprint of a model with no shape
  freedom.
- **The Mersenne tower observation.** π(M₇) = π(127) = 31 = M₅. That
  identity is exact, machine-verified, and genuinely surprising. It is
  the seed of the C_XI = 62 normalization.

**What needs honest disclosure before you cite anything:**

- **BAO global fit on DESI DR1 is χ²/dof = 1.72**, which means **p = 0.034
  (2.1σ tension)**. Information criteria still favor IF Theory over
  6-parameter ΛCDM (Bayes K ≈ 3.5 under honest k=1 counting), but the
  fit itself is not a clean win — it is a model under tension.
- **The "Mersenne Tower Theorem" is, as of today, a partial theorem.**
  Exact uniqueness of M₇ as a tower-closed Mersenne prime is verified
  only for small exponents (p ≤ 19). For larger Mersenne primes the
  current code uses the prime-counting *approximation* π(x) ~ x/ln(x)
  rather than exact primepi. The headline number C_XI = 62 *follows* if
  uniqueness generalizes, but uniqueness across all 52 known M_p is
  currently open. See `mersenne_tower_theorem.py` for status detail.
- **Three "tension resolution" predictions in `predictions/` are
  postdictions, not predictions.** `hubble_tension.py`, `s8_tension.py`,
  and `cmb_cold_spot.py` take the *observed* amplitude of the tension
  as a model input and produce a theoretically-motivated *shape* around
  it. The shape is falsifiable; the amplitude is by construction. Each
  file now carries a `POSTDICTION_NOTICE` constant disclosing this.
- **v₀ has ±30% theoretical uncertainty** from virial-radius ambiguity.
  The Milky Way prediction `v(asymptotic) = 226 ± 68 km/s` vs. observed
  `220 ± 20 km/s` agrees comfortably, but the headline is a 1σ overlap
  that depends on the wide error bar.
- **Planck 2018 cosmological parameters are treated as priors**, not
  fit. "Zero free parameters" means zero *new* parameters on top of the
  standard cosmological background, not zero parameters total.

If you have only ten minutes, the single result to verify yourself is
the σ₈ → r₀ inversion. Everything else is supporting context.

---

## The core equation

```
Φ(r) = 1 / log(r/r₀ + 1)

Amplitude  = 1       (exact, from PNT: π(x) ~ x / log(x))
C_XI       = 62      (Mersenne tower observation: 2 × π(127) = 2 × 31)
r₀         = 0.6595  kpc  (from σ₈ = 0.8111 via the Peebles §36 integral)
v₀         = 394.4   km/s ± 30%  (virial-theorem semi-derivation)
```

The galaxy two-point correlation function is then
**ξ(r) = C_XI × b² × D(z)² × Φ(r)²** for galaxy bias b and growth factor
D(z).

A second regime applies on cosmological scales (r > 1 Mpc), where the
*accumulated* form **Φ_acc(r) ∝ ln(r/r₀ + 1)** drives the bubble-universe
dark-energy mechanism (see `dark_energy_util.py` and §"What the theory
explains"). The two regimes share the same underlying PNT logic:
density ~ 1/ln, accumulated ~ ∫ density.

---

## What the theory explains

### Dark matter — emergent from the logarithmic potential

| Phenomenon | Model behavior | Status |
|---|---|---|
| Flat galactic rotation curves | Φ(r) gives log-shaped potential → roughly flat circular velocity | shape matches observations across MW-like galaxies |
| Galaxy correlation function ξ(r) | ξ ∝ Φ² tracks observed power-law shape, Pearson r > 0.93 across 3.5M+ galaxies (SDSS / DESI / Euclid) | real cross-survey agreement |
| Milky Way asymptotic velocity | predicted 226 ± 68 km/s vs. observed 220 ± 20 km/s | 1σ overlap, ±30% v₀ uncertainty dominates |
| Galaxy structure formation | logarithmic smoothing kernel | qualitative — quantitative comparison pending |

### Dark energy — the bubble-universe mechanism

Beyond ~10 Mpc, gravitational closures decouple from cosmic expansion.
The decoupling scale is **r_bubble = (v₀/H₀)√3 ≈ 10.3 Mpc** — derived,
not fit. Below the coupling scale **r_coupling = r_bubble/e ≈ 3.79 Mpc**
bubbles still interact; beyond **r_detachment ≈ 14.1 Mpc** they are
independent. Detached bubbles produce negative pressure that drives
acceleration with **w(z) = −1 + 5×10⁻⁶ / (1+z)**.

DESI DR1 BAO global fit:

| Statistic | IF Theory (k_new = 1) | ΛCDM (k = 6) |
|---|---|---|
| χ² | 22.3 | 12.0 |
| χ²/dof | 1.72 | ~1 |
| p-value | **0.034 (2.1σ)** | ~0.5 |
| AIC | 24.30 | 24.00 |
| BIC | **24.86** | 27.39 |

BIC favors IF Theory (ΔBIC = 2.53, Bayes K ≈ 3.5 — *substantial* on the
Jeffreys scale, not *decisive*) because ΛCDM uses 6 free parameters. The
2.1σ tension in the raw χ² is real and must travel with every citation
of this comparison.

---

## What is falsifiable

A theory's worth is its risk. These are the explicit ways IF Theory dies:

1. **No second tower-closed Mersenne prime.** If a future exact
   computation of π(M_p) for some p ∈ {31, 61, 89, 107, 127, ...} returns
   another Mersenne prime, then C_XI = 62 is not unique and the entire
   normalization story collapses.
2. **Galaxy-correlation r₀ diverges from 0.65 kpc** as data improves.
   If next-decade surveys converge on r₀ ≈ 0.5 or 0.8 kpc instead of
   0.65, the 1.5% agreement was lucky.
3. **JWST does NOT find massive mature galaxies at z > 25.** IF Theory
   predicts formation speedup of 1.18–1.24× over ΛCDM. Absence of mature
   structure at z > 25 in the deepest JWST surveys falsifies the
   information-driven growth claim.
4. **H₀(r) is constant across 10–500 Mpc** within measurement error.
   The bubble mechanism requires monotonically decreasing H₀(r) on a
   ~10 Mpc scale. Constant H₀ kills it.
5. **σ₈ moves substantially** (Planck-vs-LSS tension resolves toward
   LSS). The σ₈ input changes the derived r₀, and the 0.65 kpc agreement
   has to be re-checked.

Falsification routes 3 and 4 are testable with data already accumulating
in 2026–2028. If IF Theory survives those, it gets interesting; if it
doesn't, the science still wins — that is what falsifiability buys.

---

## Quick start

```bash
git clone https://github.com/phuctruong/if.git
cd if
pip install -r requirements.txt

# The load-bearing derivation: σ₈ → r₀ via the C_XI = 62 normalization
python prime_field_theory.py

# Honest validation: 3 live first-principles checks + 2 reported summaries
python audits/validate_from_first_principles.py

# Mersenne tower observation with its honest theorem-status output
python mersenne_tower_theorem.py

# Notebook validation (data must be present in bao_data/ and euclid_data/)
jupyter notebook   # dark_matter_*.ipynb, dark_energy_*.ipynb
```

---

## Repository structure

```
if/
├── README.md                          # this file
├── NORTHSTAR.md                       # strategic vision, falsifiable claims
├── ROADMAP.md                         # publication and integration plan
├── THEORY.md                          # full mathematical framework
├── VALIDATION.md                      # detailed empirical results
├── INDEPENDENT_VALIDATION.md          # external reimplementation report
├── FAQ.md                             # common questions
│
├── prime_field_theory.py              # the orchestrator
├── mersenne_tower_theorem.py          # partial theorem + honest status
├── mersenne_tower_conjecture.py       # honest companion (status: conjecture)
├── dark_energy_util.py                # bubble universe
├── prime_field_util.py                # utilities and integrators
│
├── core/
│   ├── constants.py                   # CANONICAL constants (SIGMA_8 = 0.8111)
│   ├── parameter_derivations.py       # σ₈ → r₀ inversion (load-bearing)
│   └── field_equations.py             # Φ, ∇Φ, ∇²Φ with input validation
│
├── predictions/                       # falsifiable + postdiction-flagged
│   ├── cosmological.py
│   ├── jwst_early_galaxies.py         # FALSIFIABLE prediction
│   ├── hubble_tension.py              # POSTDICTION_NOTICE: shape, not amplitude
│   ├── s8_tension.py                  # POSTDICTION_NOTICE: shape, not amplitude
│   ├── cmb_cold_spot.py               # POSTDICTION_NOTICE: heuristic, not prediction
│   ├── orbital_dynamics.py
│   └── bounded_math.py
│
├── audits/
│   ├── validate_from_first_principles.py   # 3 live + 2 reported, honestly split
│   ├── reported_fits.json             # provenance for transcribed χ²/dof + AIC/BIC
│   ├── test_verification_ladder.py    # input validation, Decimal/float cross-check
│   ├── test_synthetic_validation.py
│   └── test_cross_validation.py
│
├── papers/
│   ├── everyday/   (8 papers — accessible explanations)
│   └── physics/    (11 papers — technical research)
│
├── books/                             # the long-form thread
└── notebooks/                         # dark_matter_*.ipynb, dark_energy_*.ipynb
```

---

## For physicists — verification checklist

Each item is something you can run against this repo today.

- [ ] `python core/parameter_derivations.py` — confirm σ₈ = 0.8111
      target reproduces r₀ = 0.6595 ± small kpc from a real Peebles §36
      integral (not a hardcoded value).
- [ ] `python audits/validate_from_first_principles.py` — Reports 4 and 5
      should print "loaded from JSON provenance, NOT re-fit". If they
      print "first-principles validation", the honesty pass has regressed.
- [ ] Grep for `except ImportError` in `predictions/*.py` — there should
      be NONE. (Earlier versions had silent fallbacks that diverged σ₈
      from canonical 0.8111 to 0.8159.)
- [ ] Grep for `POSTDICTION_NOTICE` in `predictions/` — three files
      should carry it: hubble_tension, s8_tension, cmb_cold_spot.
- [ ] `python mersenne_tower_theorem.py` — final output should say
      "Uniqueness across all 52 known Mersenne primes is OPEN", not
      "VERIFIED."
- [ ] Run `dark_energy_bao_proof.ipynb` and confirm BAO global fit
      χ²/dof = 1.72, p = 0.034, 2.1σ tension is present in the output.
- [ ] Cross-check: any file claiming "ZERO free parameters" should
      qualify with "zero NEW parameters on top of Planck 2018."

---

## Why I'm publishing this in this state

I want this work to last. Not survive a press cycle — last. The only
research programs that survive are ones that hold up under hostile,
line-by-line review. Overclaiming in physics is not bold; it is the
fastest way to be ignored. Underclaiming what is real is also a sin,
because it lets a result die quietly.

So this README states what is real (the σ₈ → r₀ inversion at 1.5%, the
cross-survey correlation shape, the Mersenne tower number-theory), and
clearly flags what is not yet earned (the full uniqueness theorem, the
tension postdictions, the BAO 2.1σ tension). Both lists are short, and
both are true.

If you read this far and you are a number theorist who can close the
Mersenne tower uniqueness lemma — please email. If you are a cosmologist
who can run a real correlation-function fit against the .fits catalogs
in `bao_data/` and `euclid_data/` and replace the transcribed χ²/dof
values in `audits/reported_fits.json` with live numbers — please open a
PR. If you are a referee who finds another problem this honesty pass
missed — please tell me directly. The quickest path to a real result
is the quickest path through every honest criticism.

---

## Context and ecosystem

IF Theory sits inside a wider research program:

- **pvideo** — physics-based video / avatar engine using the prime field
  as a computational substrate.
- **pzip** — information-theoretic compression where Φ(r) plays the role
  of an entropy model.
- **phuc.net** — the long-form publication channel (books, articles).

These integrations are aspirational, not implemented in this repo. A
previous README block ("Cross-Project Synergy: Universal Geometric
Fields") referenced APIs and terms ("HyperNodes", "Toroidal and Coulomb
constraint parameters") that do not exist in the codebase. It has been
removed in the 2026-05-19 honesty pass.

---

## What "Nobel-worthy" actually means here

It does not mean publishing this README with bigger adjectives. It means
this:

1. Closing the Mersenne tower uniqueness lemma exactly across all 52
   known Mersenne primes — via Meissel–Mertens–Lehmer / Deléglise–Rivat
   for tractable p, or via an explicit Schoenfeld/Dusart-bound argument
   for larger p.
2. Running real, end-to-end correlation function fits against the
   on-disk `bao_data/` and `euclid_data/` catalogs, producing fresh
   χ²/dof values that this repo's audit scripts compute themselves
   rather than load from JSON.
3. Disclosing the BAO 2.1σ tension in every public talk and paper, not
   just the favorable BIC.
4. JWST finding (or failing to find) mature galaxies at z > 25 by 2028
   — and IF Theory's prediction being git-tagged BEFORE the data are
   public.
5. Independent groups reproducing the σ₈ → r₀ inversion and reporting
   their own r₀.

Hit those five and this becomes a real result.

The universe is a thermodynamic informational circuit, and we are
breathing informational fields trying to get bigger with meaning. That
framing is mine and I keep it in the books and the everyday papers. The
technical work above is what either earns the right to that framing or
doesn't. Honesty is the love.

---

**Contact:** Phuc Vinh Truong | phuc@phuc.net

Contributions welcome. Honesty is the only condition.
