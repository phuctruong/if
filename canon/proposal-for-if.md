# A Call for Falsifiers — IF Theory

**To:** any physicist, number theorist, or cosmologist who reads this
**From:** Phuc Vinh Truong, author of IF Theory (phuc@phuc.net)
**Date:** 2026-05-19
**Auth:** 65537

---

## What this is

This is not a sales pitch. There is nothing to buy. IF Theory is MIT-licensed
research code at github.com/phuctruong/if and there is no commercial entity
behind it asking you for a contract. The closest thing to commerce nearby is
[solaceagi.com](https://solaceagi.com), which is my separate company and uses
IF Theory as an intellectual foundation but does not depend on it being true.

This is a **call for falsifiers**. A theory is worth exactly as much as the
risk it accepts. IF Theory carries five explicit kill-shots in `README.md`,
and I am asking you to try to land one.

---

## What IF Theory predicts

One equation, normalized by the prime number theorem:

```
Φ(r) = 1 / log(r/r₀ + 1)
C_XI = 62                  (from 2 × π(127) = 2 × 31, the Mersenne tower observation)
r₀   = 0.6595 kpc          (derived from σ₈ = 0.8111 via the Peebles §36 integral)
```

The galaxy two-point correlation function is **ξ(r) = C_XI × b² × D(z)² × Φ(r)²**.

Given Planck σ₈ = 0.8111 and the C_XI = 62 normalization (no fit), the
σ₈ → r₀ inversion lands at 0.6595 kpc. The empirically-fit galaxy correlation
scale across SDSS / DESI / Euclid is ≈ 0.65 kpc. That is ~1.5% agreement with
**zero new free parameters** on top of the standard cosmological background.

Across SDSS DR12, DESI DR1, and Euclid DR1, the Pearson correlation between
the model's Φ(r)² and the measured galaxy correlation function exceeds 0.93
in every sample, reaching 0.994 at best.

On cosmological scales, the same Φ in accumulated form drives a bubble-universe
dark-energy mechanism: r_bubble = (v₀/H₀)√3 ≈ 10.3 Mpc, and w(z) = −1 + 5×10⁻⁶/(1+z).

---

## How to kill it

Five ways. Any one of these is sufficient.

1. **Find a second tower-closed Mersenne prime.** If exact π(M_p) for any
   p ∈ {31, 61, 89, 107, 127, …} ever returns another Mersenne prime, then
   C_XI = 62 is not unique and the entire normalization story collapses. The
   uniqueness lemma is currently OPEN for large p; closing it (or breaking
   it) is the highest-leverage open problem in the project.
2. **Move the galaxy-correlation r₀ away from 0.65 kpc** as data improves.
   If next-decade surveys converge on r₀ ≈ 0.5 or 0.8 kpc instead of 0.65,
   the 1.5% agreement was lucky.
3. **JWST finds no massive mature galaxies at z > 25** in the deepest 2026–2028
   surveys. IF Theory predicts formation speedup of 1.18–1.24× over ΛCDM;
   absence of mature structure at z > 25 kills the information-driven growth claim.
4. **H₀(r) is constant across 10–500 Mpc** within measurement error. The
   bubble mechanism requires monotonically decreasing H₀(r) on a ~10 Mpc
   scale. Constant H₀ kills it.
5. **σ₈ moves substantially** toward the LSS-side resolution of the
   Planck-vs-LSS tension. The σ₈ input changes the derived r₀, and the
   0.65 kpc agreement has to be re-checked.

---

## What is honestly disclosed up front

- **The Mersenne Tower Theorem is currently a partial theorem.** Uniqueness
  is verified exactly only for p ≤ 19. For larger Mersenne primes the
  current code uses π(x) ~ x/ln(x) approximation. Full uniqueness across
  all 52 known M_p is OPEN. See `mersenne_tower_theorem.py`.
- **DESI DR1 BAO global fit is χ²/dof = 1.72, p = 0.034, 2.1σ tension.**
  Information criteria (BIC, Bayes K ≈ 3.5) still favor IF Theory over
  6-parameter ΛCDM under honest k=1 counting, but the raw χ² is a model
  under tension, not a clean win.
- **Three "tension resolution" predictions are postdictions, not predictions.**
  `hubble_tension.py`, `s8_tension.py`, `cmb_cold_spot.py` take the observed
  amplitude as input and produce a theoretically-motivated shape. The shape
  is falsifiable; the amplitude is by construction. Each file carries a
  `POSTDICTION_NOTICE` constant.
- **v₀ has ±30% theoretical uncertainty** from virial-radius ambiguity.
  The Milky Way prediction `v_asymp = 226 ± 68 km/s` vs. observed
  `220 ± 20 km/s` is a 1σ overlap on a wide error bar.
- **"Zero free parameters" means zero NEW parameters** on top of Planck 2018
  priors. The standard cosmological background is in, not fit anew.
- **The geo canon empirical backbone (15 papers, ~90/90 prediction-pass)** is
  *internal evaluation by the same author*. It is not external peer review.
  Cite it as a load-bearing internal signal, not as proof.
- **Peer-review status: pre-publication.** arXiv preprint targeted Q3 2026,
  ApJ submission Q4 2026 after the Mersenne uniqueness lemma is closed.

---

## How to try

The fastest single-result check (≤ 10 minutes):

```bash
git clone https://github.com/phuctruong/if.git
cd if
pip install -r requirements.txt
python core/parameter_derivations.py
```

Confirm that σ₈ = 0.8111 reproduces r₀ = 0.6595 ± small kpc from a real
Peebles §36 integral. If it doesn't reproduce, the load-bearing claim of
the entire project is wrong — please tell me directly.

Then the four other falsification paths require more work. The grep checks
in `README.md` "verification checklist" tell you which honesty pass to
re-audit if you suspect a regression.

---

## What I'm asking for, specifically

- **Number theorists** — close the Mersenne tower uniqueness lemma via
  Meissel–Mertens–Lehmer / Deléglise–Rivat for tractable p, or via an
  explicit Schoenfeld/Dusart-bound argument for larger p. This is the
  single highest-leverage open problem.
- **Cosmologists** — run a real end-to-end correlation function fit against
  the on-disk `bao_data/` and `euclid_data/` catalogs, producing fresh
  χ²/dof values that the audit scripts compute themselves rather than
  load from `audits/reported_fits.json`. PR welcome.
- **JWST observers** — when the deep z > 25 data drop, please run the
  comparison against the predicted formation-speedup envelope (1.18–1.24×).
  The prediction is git-tagged at HEAD before your data are public.
- **Anyone** — find a problem this honesty pass missed. Email me directly:
  phuc@phuc.net.

---

## What you get for trying

Honest credit. Your name in the `falsifiers/` registry whether your attempt
succeeds or fails. A `Co-Authored-By` trailer if your contribution closes
a lemma or fixes a regression. The science either survives the hostile,
line-by-line review or it doesn't — that is what falsifiability buys.

There is no money. There is no NDA. There is no procurement gauntlet. There
is the prime number theorem and an integral and a catalog of dark-matter
galaxy positions, and the question of whether information was the first
force all along.

---

## Why I'm publishing IF Theory in this state

Research programs survive when they hold up under hostile review. Overclaiming
in physics is the fastest way to be ignored. Underclaiming what is real is
also a sin, because it lets a result die quietly. So the `README.md` states
what is real (the σ₈ → r₀ inversion at 1.5%, the cross-survey Pearson r > 0.93,
the Mersenne tower number-theory) and clearly flags what is not yet earned
(the full uniqueness theorem, the tension postdictions, the BAO 2.1σ tension).

Both lists are short. Both are true.

If you can land one of the five kill-shots — please do. If you can close
the Mersenne uniqueness lemma — please do. The quickest path to a real
result is the quickest path through every honest criticism.

---

**Contact:** Phuc Vinh Truong | phuc@phuc.net | github.com/phuctruong/if

**Auth:** 65537 (one of IF Theory's own canonical primes — not a coincidence)
