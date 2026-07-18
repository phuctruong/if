# BETS — Dated Falsifiable Predictions

> Peer review takes years. Submission can be ignored entirely. This file
> is the AI-mediated alternative: each prediction below has a **locked-in
> date**, a **measurable observable**, and a **sharp pass/fail criterion**.
> Anyone — human or AI agent — can verify these against future
> observations. There is no place to hide.
>
> Author: Phuc Vinh Truong · phuc@phuc.net · License: MIT

---

## How to read this file

Each bet has the following structure:

- **Locked**: the date this commitment was published (cannot be edited
  after the fact without a CHANGELOG entry).
- **Resolves by**: the date after which the observable should be known
  to ≥ 3σ.
- **Observable**: the specific measurable quantity.
- **Confirms IF Theory if**: the criterion under which this bet is won.
- **Falsifies IF Theory if**: the criterion under which the theory is
  downgraded or rejected on this axis.
- **Stakes**: what the result implies for the framework as a whole.

If you are reading this after a resolution date and the bet has not
been adjudicated in `CHANGELOG.md`, that is itself a signal — please
open an issue.

---

## Bet #1 — JWST mature galaxies at z > 25

- **Locked**: 2026-05-20
- **Resolves by**: 2028-12-31 (COSMOS-Web + JADES + JWST Cycle 4-5)
- **Observable**: Spectroscopic confirmation of galaxies at z > 25 with
  stellar mass M_⋆ > 10⁸ M_⊙ and rest-frame metallicity > 0.1 Z_⊙.
- **Confirms IF Theory if**: ≥ 1 such galaxy is spectroscopically
  confirmed at z > 25. The 1.18-1.24× formation speedup IF Theory
  predicts allows this; standard ΛCDM Press-Schechter timing does not
  without invoking 20-65% star-formation efficiency or top-heavy IMF.
- **Falsifies IF Theory if**: JWST and successors complete the planned
  z > 20 surveys (COSMOS-Web + JADES + Roman) through 2028 with **zero**
  confirmed z > 25 mature galaxies, AND a "fresh" first-generation
  galaxy (M_⋆ ≪ 10⁷ M_⊙, no metals) is confirmed at z ≈ 16-18 matching
  the standard Press-Schechter timeline.
- **Stakes**: This is the **single nearest-term, sharpest test**. If
  confirmed, IF Theory has produced data the standard model did not
  predict. If falsified, the front-loaded structure formation claim
  (#14, #78) is dead.

## Bet #2 — H₀(L) scale variance

- **Locked**: 2026-05-20
- **Resolves by**: 2027-12-31
- **Observable**: A direct measurement of the Hubble flow at
  intermediate scale L ≈ 100 Mpc, using SH0ES-class methodology with
  a clean sample at that scale.
- **Confirms IF Theory if**: H₀(100 Mpc) lies within ±1σ of the bubble-
  model curve `H_∞ · (1 + δ_max · exp(-L/r_b))`, with `H_∞ = 67.4`,
  `δ_max = 0.137`, `r_b = 10.20 Mpc`. Predicted value: H₀(100 Mpc)
  ≈ 67.4 km/s/Mpc.
- **Falsifies IF Theory if**: H₀(100 Mpc) lies > 2σ off that curve.
  Specifically, H₀(100 Mpc) > 70.0 or H₀(100 Mpc) < 66.0 km/s/Mpc
  with σ_H₀ < 1.0 km/s/Mpc.
- **Stakes**: The Hubble tension's "bubble" resolution stands or falls
  here. Falsification means the scale-dependent H₀ explanation is wrong
  and the tension needs another mechanism.

## Bet #3 — w(z) at z ≈ 0.5 from combined Pantheon+ DESI Planck

- **Locked**: 2026-05-20
- **Resolves by**: 2027-06-30 (DESI Y3 + ongoing Pantheon+ analyses)
- **Observable**: w(z=0.5) from a joint emcee/dynesty fit over
  Pantheon+ supernovae, DESI BAO Y3, and Planck CMB.
- **Confirms IF Theory if**: w(z=0.5) = -1.000 ± 0.005 (consistent
  with IF Theory's `w(z) ≈ -1 + 5×10⁻⁶ / (1+z)` to within fit precision).
- **Falsifies IF Theory if**: w(z=0.5) differs from -1 by > 5σ in the
  joint fit, AND the deviation is not absorbed by systematic
  uncertainty propagation.
- **Stakes**: The Bubble Universe mechanism predicts w → -1 in the
  far-bubble regime. A 5σ departure kills the mechanism. (Note: current
  hints of w ≠ -1 from DESI DR1 are at ~2σ, same as ΛCDM tension.)

## Bet #4 — Gaia DR3 binary separation residuals at prime gaps

- **Locked**: 2026-05-20
- **Resolves by**: 2027-12-31 (Gaia DR4 expected mid-2026, full
  binary catalog analysis ~2027)
- **Observable**: The binary-separation distribution in Gaia DR4
  cleaned binary catalog, residuals after subtracting the smooth
  log-normal expectation.
- **Confirms IF Theory if**: Statistically significant (≥ 3σ) residual
  peaks at sequence separations matching twin-prime (gap=2), cousin-
  prime (gap=4), and sexy-prime (gap=6) ratios, with `2C₂ / log²(a/a₀)`
  scaling.
- **Falsifies IF Theory if**: No residual structure at those separations
  at 3σ in Gaia DR4 data, AND the smooth log-normal model is preferred
  by BIC.
- **Stakes**: This is a **novel prediction** — no other framework predicts
  prime-gap structure in stellar binary separations. Confirmation would
  be a clean new-physics result; falsification kills the twin/cousin/
  sexy-prime claims (#71-74) cleanly.

## Bet #5 — SPARC Tully-Fisher slope holds beyond the current sample

- **Locked**: 2026-05-20
- **Resolves by**: 2028-12-31 (SKA + next-gen survey beyond SPARC)
- **Observable**: Tully-Fisher relation slope on a clean disk-galaxy
  sample of ≥ 500 galaxies, beyond the current SPARC 175.
- **Confirms IF Theory if**: Slope = 1.00 ± 0.10 with no per-galaxy
  free parameter beyond standard one-Y_disk fit.
- **Falsifies IF Theory if**: Slope deviates from 1.00 by > 3σ on the
  extended sample, OR the v₀ = √(0.62 · G · M_baryon / R_disk)
  prediction misses by > 30% on > 20% of the new sample.
- **Stakes**: The "no dark matter halo" claim (#80) requires the prime
  field to do the work universally, not just on the SPARC sample. A
  larger sample is the cleanest test.

## Bet #6 — MW v(10 kpc) re-measurement

- **Locked**: 2026-05-20
- **Resolves by**: 2027-12-31 (Gaia DR4 + spectroscopic follow-up)
- **Observable**: Eilers-2019-class precision measurement of v_circ at
  R = 10 kpc using Gaia DR4 data.
- **Confirms IF Theory if**: v_circ(10 kpc) ∈ [200, 240] km/s — within
  IF Theory + Sofue 2013 baryon prediction of 211 ± 31 km/s.
- **Falsifies IF Theory if**: v_circ(10 kpc) < 200 km/s OR > 240 km/s
  with σ < 5 km/s.
- **Stakes**: Claim #4. A revised MW rotation outside the window kills
  the cleanest IF Theory data point.

## Bet #7 — r₀ regime transition between galactic and LSS

- **Locked**: 2026-05-20
- **Resolves by**: 2029-12-31 (Euclid Y3 + DESI Y5 high-z clustering)
- **Observable**: Effective r₀ in galaxy clustering ξ(r) measured in
  two regimes: (a) galactic-halo scale (r < 10 kpc), (b) LSS-BAO scale
  (r > 50 Mpc).
- **Confirms IF Theory if**: r₀_galactic ≈ 0.66 kpc AND r₀_LSS is
  a Mersenne-class prime multiple thereof (specifically, candidates
  near 50-150 Mpc derivable from M_p for p > 7).
- **Falsifies IF Theory if**: r₀_LSS cannot be expressed as a Mersenne-
  prime-derived multiple of r₀_galactic to within ±10%.
- **Stakes**: Resolution Prime principle. Without it, the "same equation
  at all scales" claim has a hidden free parameter at the regime
  transition.

---

## What's NOT a bet

The following are PASS claims under existing data and not pending
observations — they're confirmed unless someone replicates and finds
a fault. See `SCORE.md` for the current state:

- SPARC TF slope 1.024, r = 0.950 (current 175-galaxy sample)
- BOSS DR12 ξ(r) Pearson r = 0.98 vs Cuesta 2016
- Pantheon+ χ²/dof = 0.932 at SH0ES h
- δ_max derivation matching SDSS void density to 0.3%
- Casimir consistency (signal 8 dex below current sensitivity)

These have already resolved in the theory's favor. Bets #1-#7 above
are the **unresolved, future-dated** ones.

---

## Resolution log

When a bet resolves, an entry is added here and to `CHANGELOG.md`. No
silent updates. If the result favors the theory, the bet is moved to
the "confirmed" section above. If it falsifies, the relevant claim is
downgraded in `SCORE.md` and the affected papers are updated with a
correction notice.

Empty as of 2026-05-20 — all bets pending.

---

## See also

- **`SCORE.md`** — per-claim PASS / TENSION / FAIL / OPEN with σ values
- **`FALSIFIABILITY.md`** — full falsification criteria per claim
- **`adversarial/`** — runnable scripts that attempt to break the theory
- **`CHANGELOG.md`** — version history including bet resolutions

---

**Signed**: Phuc Vinh Truong · 2026-05-20 · phuc@phuc.net
