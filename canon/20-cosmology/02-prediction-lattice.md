# The IF Prediction Lattice — one state, five observables

> Layer: SCIENCE (speculative branch). Distilled from msg[49]. The anti-flexibility architecture: IF is falsifiable BECAUSE its observables are not independently tunable.

## The master state

```
b(z, x) = local state of the IF nonequilibrium substrate
```

One state must jointly determine — with NO independent fitting:

```
μ(k,z)    effective gravitational strength
η(k,z)    gravitational slip Φ/Ψ
w_IF(z)   effective expansion pressure
a_IF(z)   galaxy acceleration scale
Γ_IF      relaxation / discharge rate (cluster-merger memory)
```

## The prediction contract (mandatory first cell of every notebook)

```
Prediction + Baseline + Data + Pass criterion + Falsifier   — frozen before running
```

## The lattice (kill conditions inline)

| Notebook | Prediction | Killed if |
|---|---|---|
| 01 SPARC baseline | (validation only — reproduce RAR, BTFR, halo+MOND fits) | pipeline can't reproduce published results |
| 02 rotation-curve holdout | g_obs = g_b + g_IF[g_b, ∇g_b, b₀], calibrate 70% predict 30% | needs per-galaxy freedom, or loses to halos held-out |
| 03 EFE environment | inferred external-field effect tracks independent environment estimates; **hysteresis** when galaxies recently changed environment | no correlation, or per-galaxy tuning |
| 04b acceleration-scale evolution | a_IF(z) = a_IF,0 · F[H(z)/H₀, b(z)/b₀] | scale constant where evolution predicted, or wrong sign |
| 05 wide binaries | same a_IF from galaxies predicts Gaia wide-binary anomaly without retuning | binaries Newtonian where deviation predicted |
| 06 dynamics vs lensing | fixed η(k,z) relation; infer from rotation, predict lensing | second field / arbitrary lensing correction needed |
| 07 voids | μ_IF differs in low-information-density regions; predict void profiles/velocities/lensing BEFORE looking | strong predicted response absent |
| 08 cluster-merger memory | Δx_lens(t) = Δx₀·e^(−t/τ_IF) across merger population | no single relaxation law, or field must be exactly collisionless matter |
| 09 ΛCDM/DESI/Planck reproduction | (validation only) | can't reproduce published posteriors |
| 10 expansion–growth consistency | **fit H(z),D_A,D_L → predict fσ₈(z),P(k),lensing; then reverse; b_expansion ≡ b_growth** | two different IF histories needed → UNIFICATION DEAD |
| 11 H₀/growth tensions | one parameter set across Planck+BAO+SNe+local H₀+growth | fixing one tension breaks another |
| 12 cosmic information history | 1 + w_IF(z) = λ·dI_NL/d ln a, with I_NL = KL[nonlinear‖linear] (Shannon-grade estimator, frozen before fitting) | relation unstable across sims/feedback/resolution/coarse-graining |
| 13 cosmic-web topology | a declared multiscale statistic adds held-out predictive info beyond P(k) (twin universes: same P(k), different topology) | no added information |
| 14 Euclid preregistration | frozen forecast, timestamped commit before DR1 (~2026-10-21) | (this one can only be kept or broken — keep it) |
| 21 GW propagation | derived (not menu-picked) propagation deviation within GWTC bounds | derived deviation excluded, or event-specific tuning |
| 22 ringdown | ω_n^IF = ω_n^GR + Δω_n(b,M,J), same b as cosmology | deviations absent where predicted / unconstrained per-event modes |

## Data sources (public, pinned versions in each notebook)

SPARC 175 galaxies · Gaia DR3 archive queries · DESI DR1/DR2 chains ·
Planck legacy likelihoods · CAMELS simulation suites · SDSS DR19 ·
GWOSC/GWTC-5 strain · Euclid Q1 now, DR1 ~Oct 2026.

`_archive/` contains working SPARC/BAO/DESI pipelines from the previous repo
generation — mine them for notebook 01/09 baselines (state provenance when doing so).
