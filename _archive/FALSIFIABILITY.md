# Falsifiability Criteria

> Per Scott Aaronson and Carl Sagan: "If a theory cannot be disproven,
> it is not scientific." Every load-bearing claim in this repository
> must have a sharp, quantitative falsification criterion. This file
> lists them. If you find evidence that meets any criterion below, the
> corresponding claim is falsified — please open an issue.

## TL;DR

Three physical postulates (A1, A2, A3) underlie the Mersenne Tower
derivation of `C_XI = 62`. Each is falsifiable, with sharp σ thresholds.
Per-claim falsifiers follow. Open / unfalsifiable claims are listed at
the end with the work needed to sharpen them.

For per-claim PASS / TENSION / FAIL status see `SCORE.md`.

## The three axioms (physical postulates, not theorems)

The Mersenne Tower Theorem derives `C_XI = 62` from three axioms. Each
axiom is a physical postulate that can be falsified:

### A1 — Information Primacy

**Postulate.** The dimensionless prefactor of the prime field potential
is exactly 1: Φ(r) = 1 / log(r/r₀ + 1) (for the cosmological-density
regime) or Φ(r) = ln(r/r₀ + 1) (for the integrated-potential regime),
with no additional fitted amplitude.

**Falsification.** A measurement at any scale showing that the prefactor
is statistically inconsistent with 1 to 5σ. Specifically:

- A SPARC galaxy where v_0_observed differs from
  √(0.62 · G · M_baryon / R_disk) by more than 30% with no plausible
  per-galaxy systematic explanation.
- A galaxy cluster where the IF Theory prediction misses the observed
  velocity dispersion by > 5σ after accounting for known baryonic and
  gas contributions.
- A laboratory measurement at any distance scale showing a residual
  modulation of Φ that cannot be absorbed into r₀ or M_baryon.

### A2 — Closure Constraint

**Postulate.** All dimensionless constants in the theory derive from
prime-counting structure (PNT and the Mersenne tower). No external
calibration is permitted.

**Falsification.** A claim that requires a fitted constant outside the
{r₀, v₀_galaxy, Y_disk, Y_bul} set, where r₀ is set by σ₈ + Mersenne
tower and Y values are standard astrophysical M/L ratios already in
SPARC. Specifically:

- Any successful fit of an observation requiring a 5th or 6th
  fitted parameter not in the above list.
- Any deviation in Hubble tension's δ_max derivation that requires
  more than the LTB linear-order formula plus SDSS-observed void
  density.

### A3 — Two-Point Observability

**Postulate.** Galaxy correlation function ξ(r) = C_XI · [Φ(r)]² with
C_XI = 62 derived from π(127) = 31.

**Falsification.** A measurement of ξ(r) on a clean galaxy sample at
some redshift range showing the shape is *not* proportional to
[1 / log(r/r₀ + 1)]² to within 5σ in the fitting range, where r₀ is
the regime-appropriate scale (Resolution Prime principle: r₀ ≈ 0.66
kpc galactic, ≈ 100 Mpc-class for LSS).

## Empirical claims with explicit falsification

### Claim #4: MW v(10 kpc) consistent at 0.23σ

**Falsification.** A new Eilers-2019-class precision measurement at
10 kpc showing v_circ < 200 km/s or > 240 km/s, since the IF Theory +
Sofue 2013 baryon prediction is 211 ± 31 km/s.

### Claim #5: BOSS galaxy correlation r > 0.98 shape

**Falsification.** A ξ(r) measurement on a clean galaxy sample (any
of LOWZ, CMASS, DESI, Euclid) showing log-log Pearson r between data
and the IF Theory shape < 0.90 in the standard fitting range
(8 Mpc/h ≤ r ≤ 150 Mpc/h).

### Claim #11: Bubble universe w(z) ≈ -0.999995

**Falsification.** A combined Pantheon+ + DESI + Planck Bayesian fit
showing w(z) at z ≈ 0.5 differs from -1 by more than 5σ.

### Claim #13: r_bubble = 10.3 Mpc derived from v₀ + H₀

**Falsification.** A measurement of the local-vs-distant Hubble flow
transition scale that disagrees with 10.3 Mpc by more than ~25%, OR
a future precise determination of v₀ that produces a derived
r_bubble outside [8.0, 12.5] Mpc.

### Claim #14: JWST early-galaxy 1.18-1.24× speedup at z > 25

**Falsification.** Spectroscopic confirmation of a z > 16 galaxy with
M_⋆ ≪ 10⁷ M_⊙ AND no metal enrichment — i.e., a "fresh" first-
generation galaxy on the standard ΛCDM Press-Schechter timeline,
ruling out front-loaded structure formation. This requires JWST or
its successors; window open until ~2028-2030.

### Claim #15: Hubble tension via bubble mechanism (zero-parameter)

**Falsification.**
- The SDSS void catalog refutes a typical local under-density of
  30-70% (i.e., a clean re-measurement showing typical voids are
  < 10% under-dense).
- A direct H_0(L) measurement at intermediate scale L ≈ 100 Mpc
  showing a value that lies > 2σ off the bubble-model prediction
  curve `H_∞ · (1 + δ_max · exp(-L/r_b))`.

### Claim #36: Casimir asymmetry from prime channels p > 13

**Falsification.** A precision Casimir measurement at sub-femtometer
gap (achievable via collider scattering or high-energy spectroscopy)
showing the residual modulation amplitude *exceeds* the predicted
ε ≈ L_p / d. The current tabletop experiments (Decca 2007, Sushkov
2011) cannot falsify; they are simply insensitive to the predicted
signal.

### Claim #80: No dark matter; prime field IS the rotation contribution

**Falsification.**
- A galaxy population analysis (e.g., a clean sample beyond SPARC)
  showing Tully-Fisher slope significantly different from 1.0 at the
  per-galaxy v_0 = √(0.62 · G · M_baryon / R_disk) prediction.
- A direct dark-matter particle detection (DAMA, XENON, LUX-ZEPLIN
  successor) showing a clean signal incompatible with the mundane
  baryon-only universe + prime-field cosmological contribution.
- A cluster with a "missing dark matter" signature that cannot be
  reproduced by the prime field at the cluster's R_disk-equivalent
  scale.

## Open / unfalsifiable claims (downgrade or sharpen)

These claims need quantitative falsification criteria added before
they can be considered scientific in the strictest sense:

| Claim | Issue |
|---|---|
| #65 Resolution Prime threshold for protein folding | needs a specific p_res value to falsify against |
| #66 Logic-mapped AI claim | meta-claim; not directly falsifiable |
| #71-75 Twin-prime / cousin-prime ↔ multiplicity | needs a specific quantitative prediction (e.g., binary fraction at separation a falls off as `2C₂ / log²(a/a₀)`) |
| #81 Beyond-four-forces channels p > 23 | needs specific experimental signatures per channel |

## What would *strengthen* the theory beyond the current evidence

A claim that PASSES is good. A claim that PASSES *and* makes a *novel
prediction* that is later confirmed is much stronger. Currently
unconfirmed novel predictions:

1. Gaia DR3 binary-separation distribution should have residual
   structure at sequence separations matching twin-prime, cousin-prime,
   and sexy-prime gaps (claims #71-74).
2. JWST z = 12-16 stellar-mass function should follow π(M)/M
   allocation, not Press-Schechter (claim #78). Distinguishable in
   2027-2030 with COSMOS-Web + JADES + Roman.
3. Future high-precision galaxy clustering at z ~ 1-2 should reveal
   a *transition* in effective r₀ between galactic-halo regime and
   LSS-BAO regime, mediated by a Mersenne-class prime (claim #77).

If any of these confirm at > 3σ, the theory is no longer "matches
existing data"; it has produced data the standard model didn't predict.

## Process commitment

Every claim in `SCORE.md` either:

1. PASSES against public data with the test in `predictions/` —
   in which case it's confirmed at the σ level reported.
2. FAILS — in which case the claim is downgraded or removed.
3. OPEN — in which case the falsification criterion is listed here
   so future tests can adjudicate.

If you can show a claim should be FAILED but is currently marked
PASSED, please open a GitHub issue. The goal is honest validation,
not preserved priors.

---

## See also

- **`README.md`** — overview with TL;DR and Quick Start
- **`SCORE.md`** — per-claim PASS / TENSION / FAIL with σ values
- **`VALIDATION.md`** — survey-by-survey empirical detail
- **`REPLICATION.md`** — independent-replication protocol
- **`THEORY.md`** — full mathematical framework

---

## Dwarf-regime falsifier + massive-spiral protection clause (added 2026-06-12)

Source: executed regime split, `adversarial/dwarf_regime_split.py`
(full SPARC 175, sealed JSON). Current state: IF beats MOND in massive
spirals (median χ²/dof ratio 0.71, n=54) and loses 2.2–3.8× in
dwarfs/intermediates.

- **Falsifier F-DWARF:** if any future extension of the galactic law
  (new term, regime interpolation, r₀ mapping) brings dwarf median
  χ²/dof below MOND's *but degrades the massive-spiral median by more
  than 20%*, the extension is rejected — the massive-spiral win is the
  theory's hardest-won empirical asset and may not be traded away
  silently (never-worse, LAI-27 analog).
- **Falsifier F-MASSIVE:** if an independent re-analysis (different
  M/L priors, different error floors) erases the massive-spiral win
  (ratio ≥ 1.0), the theory's strongest galactic claim falls back to
  "within 2× of MOND" and SCORE.md must be downgraded accordingly.
- **Pre-registered expectation:** the un-extended law will keep losing
  in dwarfs. A future claimed dwarf win without a published extension
  mechanism should be treated as a fitting artifact.
