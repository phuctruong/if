<!-- extracted from ChatGPT (driver) on 2026-07-18 -->

# IF Cosmology  
## A Joint Expansion–Growth Consistency Test Without Independent Dark Matter or Dark Energy

**Author:** Phuc Vinh Truong  
**Series:** IF Theory Working Papers  
**Paper:** 9  
**Date:** July 18, 2026  
**Status:** Cosmological inference and falsification protocol; no empirical success claimed

---

## Abstract

The IF Unified Geometry Hypothesis proposes that dark-matter-like attraction and dark-energy-like expansion arise from different regimes of one dynamical geometric sector rather than from an independent collisionless dark-matter particle fluid plus a separate cosmological-constant or dark-energy component. The decisive test of this proposal is not whether an IF model can fit the cosmic expansion history and structure growth simultaneously after unrestricted adjustment. It is whether the IF state inferred from one sector predicts the other.

This paper defines the **IF Expansion–Growth Consistency Test**. Let \(b(a)\) denote the homogeneous state of the IF sector, and let one covariant action determine its background density, effective pressure, clustering response, gravitational slip, sound speed, and stability. The same state must generate:

\[
\left\{
H(a),\,
D_A(a),\,
D_L(a),\,
D(a),\,
f\sigma_8(a),\,
P(k,a),\,
\Phi(k,a)+\Psi(k,a)
\right\}.
\]

The central consistency requirement is:

\[
\boxed{
b_{\mathrm{expansion}}(a)
=
b_{\mathrm{growth}}(a)
=
b_{\mathrm{lensing}}(a)
}
\]

within the joint uncertainty implied by one shared parameter vector.

Three deliberately separated inference routes are proposed:

1. **Expansion-to-growth:** fit baryon acoustic oscillations, supernova distances, and background CMB information; then predict growth and lensing without new IF freedom.
2. **Growth-to-expansion:** fit redshift-space distortions, full-shape clustering, cosmic shear, and CMB lensing; then predict distances and expansion.
3. **Early-to-late:** fit the primordial and recombination-era observables; then predict both late expansion and low-redshift structure.

The standard cosmological model is the required baseline. Planck’s final full-mission analysis found that six-parameter spatially flat \(\Lambda\)CDM provides an excellent description of the CMB data. DESI Data Release 2 measured baryon acoustic oscillations using more than fourteen million galaxy and quasar tracers and released posterior chains and best-fit cosmology products publicly in October 2025. DESI’s combined analyses reported that flat \(\Lambda\)CDM remains a good description of the BAO measurements while combinations with CMB and supernova datasets can prefer a time-varying dark-energy parameterization, with the significance dependent on the supernova compilation. citeturn201242search15turn201242search3turn201242search1

The IF analysis will initially use public compressed likelihoods and chains rather than raw images or spectra. Expansion inputs include DESI DR2 BAO, Pantheon+ supernovae, and Planck products. Growth inputs include DESI clustering and redshift-space information, weak-lensing measurements such as DES Year 3, and Planck lensing. Pantheon+ contains 1,701 light curves from 1,550 distinct Type Ia supernovae spanning \(0.001<z<2.26\). The DES Year 3 cosmic-shear analysis used more than one hundred million source galaxies and demonstrated both the statistical power of weak lensing and its sensitivity to intrinsic-alignment, baryonic-feedback, and nonlinear-modeling assumptions. citeturn201242search5turn638735academia34

The principal falsifier is parameter splitting. Let \(\theta_E\) be the IF parameters inferred from expansion and \(\theta_G\) those inferred from growth. The unified hypothesis requires:

\[
\boxed{
\theta_E=\theta_G.
}
\]

If independent parameter vectors, functions, initial conditions, or transition histories are needed, then IF Theory has not unified the dark sector. A model that reproduces the data only by behaving exactly like cold dark matter at early times and independently like arbitrary dark energy at late times may remain phenomenologically viable, but its proposed informational unification has failed unless the connection between those regimes is derived and predictive.

---

## Keywords

Cosmology; cosmic expansion; structure growth; modified gravity; dark matter; dark energy; DESI; Planck; supernovae; weak lensing; redshift-space distortions; consistency tests; IF Theory.

---

# 1. Introduction

Cosmological expansion and cosmic structure growth are governed by related but observationally distinguishable physics.

The expansion history determines:

- how the scale factor evolves;
- the distance–redshift relation;
- the age of the universe;
- the volume associated with a redshift interval.

Structure growth determines:

- how primordial density perturbations become galaxies, filaments, voids, and clusters;
- how rapidly matter falls into gravitational potentials;
- how much gravitational lensing those potentials produce;
- how clustering changes with time and scale.

In general relativity with specified matter components, expansion and growth are linked. Once:

\[
H(a)
\]

and the gravitating contents are known, the linear growth history is highly constrained.

Modified-gravity and unified-dark-sector models change this relationship. Two theories can generate nearly identical distances while producing different:

\[
f\sigma_8(z),
\qquad
P(k,z),
\qquad
\Phi+\Psi.
\]

This is why fitting the expansion history alone cannot establish a theory of the dark sector.

The IF proposal is unusually restrictive. It seeks to replace both independent dark components with one geometric state.

The appropriate question is therefore:

\[
\boxed{
\text{Does one IF state inferred from cosmic distances correctly predict}
\atop
\text{the formation and lensing of structure?}
}
\]

The reverse question is equally important:

\[
\boxed{
\text{Does the IF state inferred from structure correctly predict}
\atop
\text{the expansion history?}
}
\]

If the two answers disagree, the central unification claim fails.

---

# 2. Scientific Scope

Paper 9 tests:

- homogeneous expansion;
- linear cosmological perturbations;
- mildly nonlinear clustering through validated compressed products;
- weak gravitational lensing;
- CMB background and lensing constraints;
- cross-sector parameter consistency.

It does not yet provide:

- a complete nonlinear galaxy-formation simulation;
- a definitive cluster-merger test;
- a quantum origin for the IF field;
- a derivation of primordial perturbations from first principles;
- a solution to the vacuum-energy problem;
- a final Euclid analysis.

Paper 9 requires that the selected IF model already possess:

- a covariant action or a clearly labeled effective phenomenological closure;
- a general-relativistic limit;
- stable scalar and tensor perturbations;
- a defined early-universe clustering mode;
- no independent cold-dark-matter fluid in the minimal test;
- no independent cosmological constant in the minimal test.

A phenomenological model may be used to determine whether the unification is observationally plausible before the full action is completed. Such a model must not be mistaken for a fundamental derivation.

---

# 3. Observational Foundation

## 3.1 Planck

The final Planck release provides temperature, polarization, lensing, likelihood, map, and cosmological-parameter products from the full mission. The collaboration found that the six-parameter flat \(\Lambda\)CDM model gives an excellent description of the CMB and tightly constrains departures from it. The Planck Legacy Archive makes the public mission products available for reproducible analysis. citeturn201242search9turn201242search15turn201242search24

Planck constrains:

- the baryon density;
- the effective early clustering density;
- primordial fluctuation amplitude and tilt;
- the angular acoustic scale;
- reionization optical depth;
- CMB lensing;
- the integrated expansion history to recombination.

A no-particle-dark-matter IF theory must reproduce the physical effects conventionally attributed to the Planck-inferred cold-dark-matter density.

---

## 3.2 DESI DR2 BAO

DESI DR2 reported BAO measurements from more than fourteen million galaxies and quasars over three years of observations. The BAO measurements constrain transverse and radial distance combinations across redshift. DESI publicly released its DR2 cosmological posterior chains and posterior-maximizing parameter products in October 2025. citeturn201242search3turn201242search1

DESI DR2 found that the BAO data are well described by flat \(\Lambda\)CDM. When combined with CMB and supernova data, an evolving \(w_0w_a\) parameterization can improve the fit, with the reported preference depending materially on which supernova compilation is included. citeturn201242search3

The IF program must therefore compare against both:

\[
\Lambda\mathrm{CDM}
\]

and:

\[
w_0w_a\mathrm{CDM}.
\]

---

## 3.3 Type Ia supernovae

Pantheon+ provides 1,701 light curves from 1,550 distinct Type Ia supernovae extending from:

\[
z=0.001
\]

to:

\[
z=2.26.
\]

The compilation includes a covariance treatment for calibration and other systematic uncertainties and improves substantially upon the original Pantheon sample. citeturn201242search5

Supernovae constrain relative luminosity distances:

\[
D_L(z).
\]

They do not alone provide an absolute distance scale without calibration or an external anchor.

---

## 3.4 Full-shape clustering and redshift-space distortions

BAO primarily constrains geometric distances. Full-shape galaxy clustering contains additional information about:

- the broadband power-spectrum shape;
- the Alcock–Paczyński effect;
- redshift-space distortions;
- the fluctuation amplitude;
- growth.

Recent analyses combine DESI DR1 full-shape information with DESI DR2 BAO while treating covariance between the releases. Such analyses provide constraints on quantities including:

\[
\Omega_m,\quad
H_0,\quad
\sigma_8
\]

and extensions beyond \(\Lambda\)CDM. citeturn638735academia33

The primary IF analysis will use validated public likelihoods or compressed measurements rather than reconstructing the entire full-shape pipeline initially.

---

## 3.5 Weak gravitational lensing

Cosmic shear measures coherent distortions of background-galaxy images caused by intervening gravitational potentials.

The DES Year 3 cosmic-shear analysis used over one hundred million source galaxies and measured the low-redshift clustering-amplitude combination:

\[
S_8
=
\sigma_8
\sqrt{
\frac{\Omega_m}{0.3}
}.
\]

Its analysis also demonstrated that intrinsic alignments, nonlinear matter modeling, and baryonic physics must be propagated carefully. citeturn638735academia34

Weak lensing is indispensable because it tests:

\[
\Phi+\Psi,
\]

whereas nonrelativistic galaxy motion responds predominantly to:

\[
\Psi.
\]

---

## 3.6 Euclid

ESA’s current timeline lists a **DR1-Foundation** release for November 2026. The relevant core cosmological products therefore remain future or staged inputs to this paper’s final preregistered test. citeturn638735search3

Paper 9 will construct and freeze an IF forecast before using the relevant future Euclid cosmology products.

---

# 4. Standard Cosmological Baseline

The baseline is spatially flat \(\Lambda\)CDM.

Its background expansion is:

\[
\boxed{
H^2(a)
=
H_0^2
\left[
\Omega_r a^{-4}
+
\Omega_m a^{-3}
+
\Omega_\Lambda
\right].
}
\]

The matter density is:

\[
\Omega_m
=
\Omega_b+\Omega_c+\Omega_\nu,
\]

where:

- \(\Omega_b\) is baryonic matter;
- \(\Omega_c\) is cold dark matter;
- \(\Omega_\nu\) is the massive-neutrino contribution.

Linear subhorizon matter growth approximately obeys:

\[
\boxed{
D''+
\left(
2+\frac{H'}{H}
\right)D'
-
\frac32
\Omega_m(a)D
=
0,
}
\]

where primes denote derivatives with respect to:

\[
\ln a.
\]

The logarithmic growth rate is:

\[
\boxed{
f(a)
=
\frac{d\ln D}{d\ln a}.
}
\]

Redshift-space measurements often constrain:

\[
\boxed{
f\sigma_8(z).
}
\]

The IF model must reproduce or improve upon the baseline’s joint predictive performance.

---

# 5. The IF Cosmological State

Let:

\[
b(a)
\]

be the homogeneous IF state.

It may represent:

- an order parameter;
- a covariant field background;
- a normalized accessible-capacity state;
- a collective geometric configuration.

Its physical meaning must eventually be derived from Paper 7’s action.

For phenomenological testing, define:

\[
\boxed{
\frac{db}{d\ln a}
=
-\Gamma
\left(
b,a;\theta_b
\right).
}
\]

The initial condition is:

\[
b(a_i)=b_i.
\]

The same:

\[
b(a)
\]

must determine:

- IF density;
- IF pressure;
- effective gravitational strength;
- gravitational slip;
- sound speed;
- anisotropic stress;
- galactic acceleration scale.

---

# 6. Background Expansion

Define IF energy density and pressure:

\[
\rho_{\mathrm{IF}}(b,a),
\qquad
p_{\mathrm{IF}}(b,a).
\]

The background equations are:

\[
\boxed{
3M_{\mathrm{Pl}}^2H^2
=
\rho_r+\rho_b+\rho_{\mathrm{IF}},
}
\]

\[
\boxed{
-2M_{\mathrm{Pl}}^2\dot H
=
\frac43\rho_r
+
\rho_b
+
\rho_{\mathrm{IF}}
+
p_{\mathrm{IF}}.
}
\]

No independent cold-dark-matter density appears in the minimal model.

No independent cosmological-constant density appears in the minimal model.

The effective IF equation of state is:

\[
\boxed{
w_{\mathrm{IF}}(a)
=
\frac{
p_{\mathrm{IF}}(a)
}{
\rho_{\mathrm{IF}}(a)
}.
}
\]

The IF sector must behave differently across epochs.

A viable approximate sequence is:

\[
\boxed{
\begin{aligned}
w_{\mathrm{IF}}(a\ll1)&\approx0,\\
c_{s,\mathrm{IF}}^2(a\ll1)&\ll1,\\
w_{\mathrm{IF}}(a\sim1)&<-\frac13.
\end{aligned}
}
\]

The first regime provides dark-matter-like clustering.

The second provides accelerated expansion.

The transition must arise from one dynamical law.

---

# 7. Linear Perturbations

Use Newtonian gauge:

\[
ds^2
=
-(1+2\Psi)dt^2
+
a^2(t)(1-2\Phi)d\mathbf x^2.
\]

The modified Poisson equation is parameterized as:

\[
\boxed{
-k^2\Psi
=
4\pi Ga^2
\mu(k,a)
\rho_b\Delta_b.
}
\]

The gravitational-slip parameter is:

\[
\boxed{
\eta(k,a)
=
\frac{\Phi}{\Psi}.
}
\]

The lensing-response parameter is:

\[
\boxed{
\Sigma(k,a)
=
\frac{
\mu(k,a)
\left[
1+\eta(k,a)
\right]
}{
2
}.
}
\]

In the final IF model:

\[
\mu,\quad
\eta,\quad
\Sigma
\]

must be derived from:

\[
b(a)
\]

and the common action.

They may not be reconstructed independently.

---

# 8. Baryonic Growth

Because the minimal model contains no particle cold dark matter, baryonic perturbations respond to the IF-modified potential.

A schematic baryonic growth equation is:

\[
\boxed{
D_b''+
\left(
2+\frac{H'}{H}
\right)D_b'
-
\frac32
\Omega_b(a)
\mu_{\mathrm{eff}}(k,a)
D_b
=
S_{\mathrm{IF}}(k,a),
}
\]

where:

- \(\mu_{\mathrm{eff}}\) modifies the gravitational response;
- \(S_{\mathrm{IF}}\) represents independently evolving IF perturbations.

A purely algebraic enhancement of baryonic gravity after recombination is unlikely to reproduce the early gravitational potentials required by the CMB.

Therefore, the IF perturbation:

\[
\delta_{\mathrm{IF}}
\]

must possess its own stable evolution.

A schematic equation is:

\[
\boxed{
\delta_{\mathrm{IF}}''
+
A(k,a)\delta_{\mathrm{IF}}'
+
B(k,a)\delta_{\mathrm{IF}}
=
C(k,a)\delta_b.
}
\]

The coefficients must be derived from the covariant theory.

---

# 9. Early Dark-Matter-Like Regime

The IF sector must provide an early clustering mode satisfying approximately:

\[
w_{\mathrm{IF}}\approx0,
\]

\[
c_{s,\mathrm{IF}}^2\ll1,
\]

\[
\left|
\pi_{\mathrm{IF}}
\right|
\text{ sufficiently constrained},
\]

where:

\[
\pi_{\mathrm{IF}}
\]

is anisotropic stress.

The mode must:

- begin from consistent primordial initial conditions;
- survive the radiation era;
- source metric potentials;
- influence baryon acoustic oscillations;
- produce the correct acoustic-peak structure;
- generate the later matter distribution.

If this mode is observationally indistinguishable from cold dark matter, then IF Theory has changed the ontology but not the phenomenology.

That may still be scientifically legitimate if the field predicts different behavior elsewhere.

It is not sufficient evidence for a fundamentally new dark-sector law.

---

# 10. Late Accelerating Regime

Late acceleration requires:

\[
\rho_{\mathrm{IF}}
+
3p_{\mathrm{IF}}
<0.
\]

The same field that clustered earlier must transition into or reveal a negative-pressure homogeneous regime.

Possible mechanisms include:

- kinetic-state transition;
- condensation;
- potential domination;
- derivative self-interaction;
- background–perturbation decoupling;
- scale-dependent phase behavior.

The mechanism must not destroy the earlier successful perturbations.

It must also avoid:

- ghosts;
- negative sound-speed squared;
- singular crossings;
- strong coupling;
- excessive late integrated Sachs–Wolfe effects.

---

# 11. The IF Consistency Lock

Define the shared IF parameter vector:

\[
\theta
=
\left[
\theta_b,
\theta_{\mathrm{kin}},
\theta_{\mathrm{trans}},
\theta_{\mathrm{pert}},
b_i
\right].
\]

The defining map is:

\[
\boxed{
\mathcal O_{\mathrm{cosmo}}
=
\mathfrak F
\left[
\theta
\right],
}
\]

where:

\[
\mathcal O_{\mathrm{cosmo}}
=
\left\{
H,
D_A,
D_L,
r_d,
D_b,
f\sigma_8,
P,
\mu,
\eta,
\Sigma,
C_\ell^{TT},
C_\ell^{TE},
C_\ell^{EE},
C_\ell^{\phi\phi}
\right\}.
\]

No subset receives an independently fitted IF function.

---

# 12. Expansion-Inferred State

Let expansion data be:

\[
\mathcal D_E
=
\left\{
\text{BAO},
\text{SNe},
\text{background CMB}
\right\}.
\]

Infer:

\[
P
\left(
\theta_E
\mid
\mathcal D_E
\right).
\]

For every posterior sample:

\[
\theta_E^{(s)},
\]

calculate predicted growth:

\[
\mathcal O_G^{(s)}
=
\mathfrak F_G
\left[
\theta_E^{(s)}
\right].
\]

The expansion-to-growth posterior predictive distribution is:

\[
\boxed{
P
\left(
\mathcal O_G
\mid
\mathcal D_E
\right)
=
\int
P
\left(
\mathcal O_G
\mid\theta
\right)
P
\left(
\theta\mid\mathcal D_E
\right)
d\theta.
}
\]

No growth data may be used to select the background posterior or adjust the perturbation response after this distribution is generated.

---

# 13. Growth-Inferred State

Let growth data be:

\[
\mathcal D_G
=
\left\{
\text{RSD},
\text{full-shape clustering},
\text{cosmic shear},
\text{CMB lensing}
\right\}.
\]

Infer:

\[
P
\left(
\theta_G
\mid
\mathcal D_G
\right).
\]

Predict expansion:

\[
\boxed{
P
\left(
\mathcal O_E
\mid
\mathcal D_G
\right)
=
\int
P
\left(
\mathcal O_E
\mid\theta
\right)
P
\left(
\theta\mid\mathcal D_G
\right)
d\theta.
}
\]

Again, expansion data are withheld during model selection.

---

# 14. Early-Inferred State

Let early-universe data be:

\[
\mathcal D_{\mathrm{early}}
=
\left\{
\text{CMB TT},
\text{CMB TE},
\text{CMB EE},
\text{CMB lensing},
\text{BBN prior}
\right\}.
\]

Infer:

\[
P
\left(
\theta_{\mathrm{early}}
\mid
\mathcal D_{\mathrm{early}}
\right).
\]

Predict:

- DESI BAO;
- supernova distances;
- low-redshift growth;
- weak lensing;
- the local galactic acceleration scale through Paper 8.

This is the strongest route because it spans the greatest temporal baseline.

---

# 15. State-Reconstruction Consistency

For each inference route, reconstruct the IF background:

\[
b_E(a),
\qquad
b_G(a),
\qquad
b_{\mathrm{early}}(a).
\]

Define differences:

\[
\Delta b_{EG}(a)
=
b_E(a)-b_G(a),
\]

\[
\Delta b_{E\mathrm{early}}(a)
=
b_E(a)-b_{\mathrm{early}}(a).
\]

With covariance:

\[
C_b(a,a'),
\]

define a functional inconsistency statistic:

\[
\boxed{
Q_b
=
\int
d\ln a
\int
d\ln a'
\,
\Delta b(a)
C_b^{-1}(a,a')
\Delta b(a').
}
\]

The implementation may use a discretized redshift basis.

A large:

\[
Q_b
\]

indicates that different observables require incompatible IF histories.

The threshold is calibrated using synthetic universes generated from the IF model.

---

# 16. Parameter-Splitting Test

Fit a split model with:

\[
\theta_E
\]

for expansion and:

\[
\theta_G
\]

for growth.

Compare against the unified model:

\[
\theta_E=\theta_G.
\]

The split model likelihood is:

\[
\mathcal L_{\mathrm{split}}
=
\mathcal L_E(\theta_E)
\mathcal L_G(\theta_G).
\]

The unified likelihood is:

\[
\mathcal L_{\mathrm{unified}}
=
\mathcal L_E(\theta)
\mathcal L_G(\theta).
\]

Compare through:

- Bayes factors;
- predictive information criteria;
- posterior parameter differences;
- simulation-calibrated likelihood-ratio statistics.

The central requirement is not that the split model never fits better. It has more freedom.

The question is whether the data demand that extra freedom strongly enough to reject unification.

---

# 17. Cross-Predictive Scores

## 17.1 Expansion-to-growth score

\[
\boxed{
S_{E\rightarrow G}
=
\ln
P
\left(
\mathcal D_G
\mid
\mathcal D_E,
\mathrm{IF}
\right).
}
\]

## 17.2 Growth-to-expansion score

\[
\boxed{
S_{G\rightarrow E}
=
\ln
P
\left(
\mathcal D_E
\mid
\mathcal D_G,
\mathrm{IF}
\right).
}
\]

## 17.3 Early-to-late score

\[
\boxed{
S_{\mathrm{early}\rightarrow\mathrm{late}}
=
\ln
P
\left(
\mathcal D_{\mathrm{late}}
\mid
\mathcal D_{\mathrm{early}},
\mathrm{IF}
\right).
}
\]

The scores are compared with:

- \(\Lambda\)CDM;
- \(w_0w_a\)CDM;
- a phenomenological modified-gravity model;
- a split IF model.

---

# 18. Null-Test Function

In general relativity with a known matter density, expansion predicts growth.

Construct an IF-specific null function:

\[
\boxed{
\mathcal N_{\mathrm{IF}}(k,a)
=
\mu_{\mathrm{growth}}(k,a)
-
\mu_{\mathrm{expansion}}(k,a;\theta_E).
}
\]

Under IF consistency:

\[
\mathcal N_{\mathrm{IF}}(k,a)=0
\]

within uncertainty.

Similarly, define a lensing null:

\[
\boxed{
\mathcal N_{\mathrm{lens}}(k,a)
=
\Sigma_{\mathrm{observed}}(k,a)
-
\Sigma_{\mathrm{IF}}
\left[
b_E(a),k
\right].
}
\]

A statistically significant scale- or redshift-dependent departure rejects the selected IF closure.

---

# 19. IF-C0 Minimal Phenomenological Closure

Before a final covariant action is implemented, use a deliberately restricted model.

## 19.1 State evolution

\[
\boxed{
\frac{db}{d\ln a}
=
-\gamma
b^n
\left(
1-b
\right)^m.
}
\]

Require:

\[
0\leq b\leq1.
\]

---

## 19.2 Density split

Define the IF density as:

\[
\rho_{\mathrm{IF}}(a)
=
\rho_{\mathrm{IF},0}
F_\rho
\left[
b(a)
\right].
\]

The same function must generate early matter-like scaling and late acceleration.

One possible restricted form is:

\[
\boxed{
\rho_{\mathrm{IF}}
=
\rho_*
\left[
b\,a^{-3}
+
\lambda
\left(
1-b
\right)
\right],
}
\]

but the parameter:

\[
\lambda
\]

cannot simply reproduce a hidden cosmological constant without dynamical justification.

This form is a diagnostic, not a preferred final theory.

---

## 19.3 Effective gravity

Let:

\[
\boxed{
\mu(k,a)
=
1+
\alpha_\mu
b(a)
\frac{
1
}{
1+\left[k/k_c(a)\right]^2
}.
}
\]

---

## 19.4 Slip

Let:

\[
\boxed{
\eta(k,a)
=
1+
\alpha_\eta
\left[
1-b(a)
\right]
\frac{
1
}{
1+\left[k/k_\eta(a)\right]^2
}.
}
\]

The parameters:

\[
\alpha_\mu,\quad
\alpha_\eta,\quad
k_c,\quad
k_\eta
\]

must eventually be related by the action.

The phenomenological test allows only a very small frozen family.

---

## 19.5 Galaxian link

Paper 8 requires:

\[
\boxed{
a_{\mathrm{IF}}(a)
=
a_{\mathrm{IF},0}
F_a[b(a)].
}
\]

The minimal Hubble lock is:

\[
F_a[b(a)]
=
\frac{
H(a)
}{
H_0
}.
\]

The cosmological fit therefore predicts the galaxy-scale acceleration evolution.

---

# 20. Baseline Models

## 20.1 Flat \(\Lambda\)CDM

Required baseline with standard neutrino treatment.

## 20.2 \(w_0w_a\)CDM

\[
w(a)
=
w_0+w_a(1-a).
\]

This allows flexible late expansion while retaining ordinary dark matter and general relativity.

## 20.3 Phenomenological modified gravity

Allow independent:

\[
\mu(k,a),
\qquad
\eta(k,a)
\]

within a low-dimensional parameterization.

This tests whether the IF consistency lock is too restrictive.

## 20.4 Unified dark fluid

Compare with a fluid that transitions from matter-like to dark-energy-like behavior.

## 20.5 Relativistic MOND-like or scalar–aether comparator

Compare against at least one existing relativistic model capable of producing nonstandard galaxy gravity and cosmological clustering.

## 20.6 Split IF

Expansion and growth receive separate IF parameters.

This is the direct alternative to unification.

---

# 21. Data Combination Ladder

The analysis proceeds incrementally.

## Combination A — BAO only

Tests geometric expansion with minimal external assumptions.

## Combination B — BAO plus supernovae

Constrains relative late-time expansion.

## Combination C — BAO plus background CMB

Adds the sound-horizon and recombination distance relation.

## Combination D — BAO plus CMB plus supernovae

Primary expansion fit.

## Combination E — RSD and full shape

Primary clustering-growth fit.

## Combination F — Cosmic shear and CMB lensing

Primary lensing fit.

## Combination G — Growth combination

\[
\mathrm{RSD}
+
\mathrm{full\ shape}
+
\mathrm{shear}
+
\mathrm{CMB\ lensing}.
\]

## Combination H — Full joint fit

Used only after the cross-prediction tests are frozen and reported.

---

# 22. Sound-Horizon Discipline

BAO observations constrain distances relative to the sound horizon:

\[
r_d.
\]

A model that changes early cosmology changes:

\[
r_d.
\]

Therefore, the IF analysis must not use a standard-\(\Lambda\)CDM sound horizon while claiming nonstandard pre-recombination physics.

Two valid routes exist.

## Route 1 — Full early calculation

Compute:

\[
r_d
\]

from the IF background and perturbations.

## Route 2 — Free ruler diagnostic

Treat:

\[
r_d
\]

as a nuisance parameter for a late-time-only test.

Route 2 cannot support claims that the model explains the CMB or early dark matter.

---

# 23. Primordial Initial Conditions

The IF field introduces possible additional primordial modes.

The minimal model begins with one adiabatic mode.

Any IF isocurvature mode must be:

- derived;
- parameterized;
- constrained;
- included in the complexity count.

The theory may not add a free primordial spectrum solely to repair CMB residuals.

The primordial parameters include:

\[
A_s,\qquad
n_s,\qquad
\tau_{\mathrm{reio}}.
\]

If the IF mechanism generates them, that derivation belongs to a later paper.

---

# 24. Neutrinos

Massive neutrinos affect:

- expansion;
- CMB lensing;
- matter-power suppression;
- growth.

The IF analysis must include a consistent neutrino model across all baselines.

Neutrino mass may not be adjusted differently in the expansion and growth fits.

Otherwise, apparent IF consistency could be created through nuisance-sector splitting.

---

# 25. Nonlinear Scales

IF gravity may change nonlinear evolution.

Standard nonlinear fitting formulae calibrated on \(\Lambda\)CDM cannot automatically be applied.

The analysis will therefore use a scale ladder.

## Conservative scale cut

Use only scales where linear or validated perturbative IF calculations are reliable.

## Perturbative extension

Use effective-field-theory or emulator calculations validated against IF simulations.

## Nonlinear extension

Use dedicated IF \(N\)-body or hydrodynamical simulations.

The primary inference uses conservative scales.

Small-scale information is added only after validation.

---

# 26. Galaxy Bias

Observed galaxies trace the underlying matter or effective gravitational field imperfectly.

Bias parameters must be included consistently.

The IF model may alter:

- halo formation;
- tracer bias;
- redshift-space mapping;
- velocity fields.

A standard bias model may be used initially as a diagnostic, but final claims require testing its validity in IF simulations.

---

# 27. Weak-Lensing Systematics

The weak-lensing analysis must propagate:

- shear calibration;
- photometric-redshift uncertainty;
- intrinsic alignments;
- baryonic feedback;
- nonlinear modeling;
- survey masks;
- source selection.

The DES Year 3 analysis illustrates that these effects are central components of the cosmological likelihood rather than minor corrections. citeturn638735academia34

The IF model cannot interpret every lensing residual as modified gravity.

---

# 28. Stability-First Prior

For every proposed parameter point, calculate:

\[
Q_s,\quad
c_s^2,\quad
Q_T,\quad
c_T^2.
\]

Reject if:

\[
Q_s\leq0,
\]

\[
c_s^2<0,
\]

\[
Q_T\leq0,
\]

or if the tensor speed violates the required limit.

The stability prior is applied before observational likelihood evaluation.

An unstable best fit is not a viable cosmology.

---

# 29. Parameter Identifiability

The IF transition may be poorly constrained if multiple parameter combinations produce similar:

\[
H(a).
\]

Growth and lensing may break those degeneracies.

Before real-data fitting:

1. compute Fisher or sensitivity matrices;
2. run synthetic posterior recovery;
3. identify unconstrained combinations;
4. remove or fix nonidentifiable parameters;
5. report prior-dominated directions.

A theory with ten parameters but only two measured combinations does not possess ten empirically established physical quantities.

---

# 30. Synthetic-Universe Program

Generate mock datasets from:

- \(\Lambda\)CDM;
- \(w_0w_a\)CDM;
- IF-C0;
- split IF;
- modified gravity with no IF unification;
- unified dark fluid.

For each mock:

- fit every model;
- perform cross-prediction;
- calculate parameter-splitting evidence;
- measure false rejection;
- measure false confirmation;
- test uncertainty coverage.

The analysis must demonstrate that it can reject false IF unification before applying it to the real universe.

---

# 31. Blinding

Where practical, apply blinding to:

- growth-amplitude calibration;
- selected covariance elements;
- posterior parameter displays;
- model-label identities.

The analysis decisions are frozen before unblinding.

Blinding is especially important because IF Theory has a strong desired conclusion.

---

# 32. Preregistered Decision Rules

The precise numerical thresholds will be calibrated using synthetic experiments.

The qualitative rules are fixed here.

## Rule 1 — Background adequacy

The model must provide acceptable posterior predictive behavior for:

- BAO;
- supernovae;
- CMB background observables.

## Rule 2 — Growth adequacy

Expansion-calibrated IF predictions must provide acceptable behavior for:

- RSD;
- full shape;
- weak lensing;
- CMB lensing.

## Rule 3 — Reverse prediction

Growth-calibrated IF predictions must provide acceptable behavior for expansion.

## Rule 4 — Parameter equality

The split model must not be decisively required.

## Rule 5 — Stability

The viable posterior must remain within the theoretically stable region.

## Rule 6 — Complexity

IF must not obtain comparable fit solely through much greater effective freedom.

## Rule 7 — Early-universe success

The full no-particle-dark-matter claim is withheld unless the CMB spectra and matter transfer function are reproduced.

---

# 33. Core Hypotheses

## CG-H1 — Background-fit hypothesis

One IF background state reproduces BAO, supernova, and CMB-distance constraints without an independent dark-energy term.

### Falsifier

A separate late-time component or arbitrary expansion function is required.

---

## CG-H2 — Early-clustering hypothesis

The IF perturbation sector replaces the gravitational role of cold dark matter before recombination.

### Falsifier

The CMB acoustic structure or matter transfer function cannot be reproduced.

---

## CG-H3 — Expansion-to-growth hypothesis

Expansion-inferred IF parameters predict observed low-redshift growth.

### Falsifier

Predicted:

\[
f\sigma_8,\quad
P(k),\quad
\text{or lensing}
\]

is incompatible with held-out measurements.

---

## CG-H4 — Growth-to-expansion hypothesis

Growth-inferred IF parameters predict BAO and supernova distances.

### Falsifier

The predicted expansion history is incompatible with held-out distance data.

---

## CG-H5 — State-consistency hypothesis

\[
b_E(a)
=
b_G(a)
=
b_{\mathrm{early}}(a)
\]

within simulation-calibrated uncertainty.

### Falsifier

The reconstructed IF states differ significantly.

---

## CG-H6 — Parameter-lock hypothesis

One parameter vector fits every sector.

### Falsifier

The data decisively require:

\[
\theta_E\neq\theta_G.
\]

---

## CG-H7 — Lensing-slip hypothesis

The action-derived:

\[
\eta(k,a)
\]

predicts weak and CMB lensing.

### Falsifier

An independent slip function is required.

---

## CG-H8 — Stable-posterior hypothesis

The observationally supported region is free of ghost and gradient instabilities.

### Falsifier

Only unstable parameter values fit the data.

---

## CG-H9 — Galaxy–cosmology lock

The cosmological IF history predicts the acceleration-scale evolution tested in Paper 8.

### Falsifier

The galaxy and cosmology sectors require incompatible:

\[
a_{\mathrm{IF}}(z).
\]

---

## CG-H10 — Lower-complexity hypothesis

The IF model explains the combined data with effective complexity competitive with the components it replaces.

### Falsifier

It needs more independent functions than cold dark matter plus dark energy.

---

## CG-H11 — Prospective-survey hypothesis

A frozen IF prediction succeeds on future Euclid or later survey products.

### Falsifier

The preregistered prediction fails.

---

# 34. Failure Modes

## 34.1 Joint-fit illusion

The model fits expansion and growth simultaneously only because both sectors are adjusted together.

## 34.2 Hidden parameter split

The same symbol has different effective values in background and perturbation code.

## 34.3 Sound-horizon borrowing

The model uses the \(\Lambda\)CDM sound horizon despite changing early cosmology.

## 34.4 Dark matter by another name

The IF perturbation is exactly a freely normalized cold collisionless component with no additional prediction.

## 34.5 Dark energy by another name

A constant term inside the IF Lagrangian supplies all late acceleration.

## 34.6 Arbitrary transition

A switching function is chosen after inspecting the desired redshift behavior.

## 34.7 Instability masking

Unstable points are included because they improve the likelihood.

## 34.8 Nonlinear misuse

\(\Lambda\)CDM nonlinear corrections are applied outside their validated domain.

## 34.9 Growth-data double counting

Correlated DESI, lensing, or CMB products are treated as independent.

## 34.10 Supernova cherry-picking

Only the supernova compilation giving the strongest desired result is reported.

## 34.11 Prior-volume victory

Bayesian preference arises mainly from restrictive priors rather than predictive performance.

## 34.12 Parameter nonidentifiability

Broad prior-dominated directions are described as physical measurements.

## 34.13 Posterior predictive omission

Only parameter contours are reported, not failures in observable space.

## 34.14 Tension exploitation

Existing differences among datasets are presented as evidence for IF without demonstrating that IF resolves them consistently.

## 34.15 Euclid hindsight

The model is modified after future Euclid results are opened.

---

# 35. Deterministic Jupyter-Notebook Program

## Notebook 09A — \(\Lambda\)CDM Background Reproduction

Reproduce:

\[
H(z),\quad
D_A(z),\quad
D_L(z),\quad
r_d.
\]

Compare against a trusted cosmology solver.

---

## Notebook 09B — \(\Lambda\)CDM Growth Reproduction

Reproduce:

\[
D(z),\quad
f(z),\quad
f\sigma_8(z).
\]

Validate limiting cases and numerical convergence.

---

## Notebook 09C — DESI DR2 BAO Manifest

Download:

- compressed measurements;
- covariance;
- posterior chains;
- best-fit products.

Save checksums and licenses.

---

## Notebook 09D — Pantheon+ Reproduction

Reproduce the public supernova likelihood and covariance handling.

---

## Notebook 09E — Planck Distance and Full-Likelihood Baselines

Begin with compressed background information.

Then reproduce selected official chains or likelihood results before adding IF.

---

## Notebook 09F — Weak-Lensing Baseline

Reproduce a public DES Year 3 or equivalent conservative-scale likelihood.

Validate:

- intrinsic-alignment treatment;
- shear calibration;
- redshift nuisance parameters;
- baryonic scale cuts.

---

## Notebook 09G — Full-Shape and RSD Baseline

Implement validated compressed DESI clustering products.

Track covariance with BAO products.

---

## Notebook 09H — IF-C0 Background Solver

Integrate:

\[
b(a),
\quad
\rho_{\mathrm{IF}}(a),
\quad
p_{\mathrm{IF}}(a),
\quad
H(a).
\]

---

## Notebook 09I — IF Perturbation Solver

Integrate:

\[
\delta_b,
\quad
\delta_{\mathrm{IF}},
\quad
\Phi,
\quad
\Psi.
\]

Calculate:

\[
\mu,\quad
\eta,\quad
\Sigma.
\]

---

## Notebook 09J — Stability Map

Evaluate:

\[
Q_s,\quad
c_s^2,\quad
Q_T,\quad
c_T^2.
\]

Produce the allowed prior domain.

---

## Notebook 09K — Sound-Horizon Consistency

Calculate:

\[
r_d
\]

from the IF early background.

Compare with free-ruler diagnostics.

---

## Notebook 09L — CLASS or CAMB Regression

Reproduce standard spectra before modifying the solver.

Require agreement for:

- TT;
- TE;
- EE;
- lensing;
- matter power.

---

## Notebook 09M — IF Boltzmann Implementation

Add the selected IF equations.

Document every code change and equation mapping.

---

## Notebook 09N — Expansion-Only Inference

Fit:

\[
\mathcal D_E.
\]

Save posterior samples before evaluating growth.

---

## Notebook 09O — Expansion-to-Growth Prediction

Generate:

\[
P(\mathcal O_G\mid\mathcal D_E).
\]

No growth refitting.

---

## Notebook 09P — Growth-Only Inference

Fit:

\[
\mathcal D_G.
\]

Save posterior samples before evaluating expansion.

---

## Notebook 09Q — Growth-to-Expansion Prediction

Generate:

\[
P(\mathcal O_E\mid\mathcal D_G).
\]

---

## Notebook 09R — Early-to-Late Prediction

Fit early data and forecast:

- BAO;
- supernovae;
- low-redshift growth;
- lensing;
- Paper 8 acceleration evolution.

---

## Notebook 09S — State-Reconstruction Test

Calculate:

\[
b_E(a),\quad
b_G(a),\quad
b_{\mathrm{early}}(a),\quad
Q_b.
\]

---

## Notebook 09T — Parameter-Splitting Test

Compare:

\[
\theta_E=\theta_G
\]

against:

\[
\theta_E\neq\theta_G.
\]

---

## Notebook 09U — Cross-Predictive Model Comparison

Compare IF against:

- \(\Lambda\)CDM;
- \(w_0w_a\)CDM;
- modified gravity;
- unified dark fluid;
- split IF.

---

## Notebook 09V — Synthetic Recovery

Generate and recover all benchmark universes.

Measure false-positive and false-negative rates.

---

## Notebook 09W — Nonlinear Scale Audit

Determine which scales are safe under the current IF calculation.

Fail closed by excluding unvalidated scales.

---

## Notebook 09X — Galaxy–Cosmology Lock

Send cosmological posterior samples to Paper 8’s:

\[
a_{\mathrm{IF}}(z).
\]

Test consistency with galaxy data.

---

## Notebook 09Y — Euclid Frozen Forecast

Save:

- predicted bins;
- covariance assumptions;
- summary statistics;
- exclusion thresholds;
- git commit;
- environment hash.

---

## Notebook 09Z — Adversarial Audit

A separate agent attempts to prove that any IF success results from:

- parameter splitting;
- hidden \(\Lambda\);
- dark matter relabeling;
- prior choices;
- sound-horizon borrowing;
- scale-cut tuning;
- dataset selection;
- covariance errors;
- unstable dynamics;
- post hoc transition design.

---

# 36. Computational Architecture

```text
if_cosmology/
├── data/
│   ├── manifests/
│   ├── desi_bao/
│   ├── pantheon_plus/
│   ├── planck/
│   ├── growth/
│   └── lensing/
├── background/
│   ├── lcdm.py
│   ├── if_state.py
│   ├── distances.py
│   └── sound_horizon.py
├── perturbations/
│   ├── initial_conditions.py
│   ├── baryons.py
│   ├── if_field.py
│   ├── metric.py
│   └── stability.py
├── boltzmann/
│   ├── reference/
│   ├── regression/
│   └── if_patch/
├── likelihoods/
│   ├── bao.py
│   ├── supernovae.py
│   ├── cmb.py
│   ├── rsd.py
│   ├── full_shape.py
│   └── lensing.py
├── inference/
│   ├── expansion_only.py
│   ├── growth_only.py
│   ├── early_only.py
│   ├── joint.py
│   └── split_parameters.py
├── prediction/
│   ├── expansion_to_growth.py
│   ├── growth_to_expansion.py
│   ├── early_to_late.py
│   └── euclid_forecast.py
├── validation/
│   ├── synthetic_recovery.py
│   ├── coverage.py
│   ├── covariance.py
│   └── null_tests.py
└── tests/
```

---

# 37. Reproducibility Record

Every inference run emits:

```yaml
experiment_id: if-cosmology-09
paper_version: null
git_commit: null
environment_hash: null
data_manifest_hash: null
boltzmann_code_version: null
if_patch_hash: null

model_name: null
model_parameters: {}
parameter_split: false
bare_cosmological_constant: 0
particle_cold_dark_matter: false

datasets:
  expansion: []
  growth: []
  early: []
  lensing: []

scale_cuts: {}
covariance_hashes: {}
likelihood_versions: {}

stability_prior_version: null
minimum_scalar_kinetic: null
minimum_scalar_sound_speed_squared: null
minimum_tensor_kinetic: null
maximum_tensor_speed_deviation: null

expansion_posterior_hash: null
growth_posterior_hash: null
early_posterior_hash: null

expansion_to_growth_score: null
growth_to_expansion_score: null
early_to_late_score: null
state_consistency_statistic: null
parameter_split_evidence: null

posterior_predictive_failures: []
stability_failures: []
numerical_failures: []
result_hash: null
```

---

# 38. Validation Requirements

## 38.1 Dual implementation

Implement the background and linear perturbation equations independently in:

1. a readable Python reference;
2. the modified Boltzmann solver.

Compare at randomly selected parameter points.

---

## 38.2 Constraint residuals

Track:

- Friedmann constraint;
- energy conservation;
- Einstein constraints;
- gauge consistency;
- initial-condition residuals.

---

## 38.3 Solver convergence

Vary:

- integration tolerances;
- time sampling;
- wavenumber sampling;
- hierarchy truncation;
- interpolation settings.

---

## 38.4 Standard-limit recovery

As IF modifications vanish, recover:

- ordinary general relativity;
- standard radiation and baryon evolution;
- the declared reference dark-sector limit where applicable.

Because the minimal model contains no cold-dark-matter fluid or cosmological constant, the exact route to \(\Lambda\)CDM may require a limiting IF state that becomes phenomenologically equivalent.

That equivalence must be explicit.

---

# 39. Criteria for Success

## Level 1 — Background feasibility

A stable IF state reproduces late expansion.

## Level 2 — Early clustering

The same field reproduces the CMB and linear matter transfer function without particle cold dark matter.

## Level 3 — Expansion-to-growth prediction

Expansion-calibrated parameters predict growth and lensing.

## Level 4 — Reverse prediction

Growth-calibrated parameters predict expansion.

## Level 5 — State consistency

All inference routes reconstruct the same IF history.

## Level 6 — Galaxy–cosmology closure

The cosmological state predicts Paper 8’s galactic acceleration evolution.

## Level 7 — Competitive model evidence

The unified model competes successfully with \(\Lambda\)CDM after complexity penalties and cross-prediction.

## Level 8 — Prospective confirmation

A frozen Euclid or later-survey forecast is independently confirmed.

---

# 40. What Would Count as a Major Discovery?

A technically important result would be:

\[
\boxed{
\text{A stable single-field IF model reproduces both a matter-like}
\atop
\text{early regime and an accelerating late regime.}
}
\]

A stronger observational result would be:

\[
\boxed{
\text{IF parameters inferred from expansion predict structure growth}
\atop
\text{and lensing without additional functions or recalibration.}
}
\]

A field-changing result would be:

\[
\boxed{
b_{\mathrm{expansion}}(a)
=
b_{\mathrm{growth}}(a)
=
b_{\mathrm{lensing}}(a)
=
b_{\mathrm{galaxies}}(a)
}
\]

across the measured cosmic history.

A Nobel-class result would additionally require:

- a distinctive quantitative prediction;
- publication before the measurement;
- independent observational confirmation;
- successful reproduction by other groups;
- no comparably simple conventional explanation.

---

# 41. Relationship to Dark Matter

A successful Paper 9 model would eliminate the need for an independent particle cold-dark-matter fluid within the tested cosmological model.

It would not prove that no dark-matter particle exists anywhere.

An IF field that clusters like dark matter may coexist with other unseen particles.

The scientifically precise conclusion would be:

> The tested observations do not require a separate cold-dark-matter particle component once the IF geometric sector is included.

The stronger ontological statement would require direct-particle searches and additional astrophysical evidence.

---

# 42. Relationship to Dark Energy

A successful model would generate acceleration without an independent cosmological constant or arbitrary dark-energy fluid.

It would not automatically solve the vacuum-energy problem.

Quantum vacuum contributions and radiative stability remain separate theoretical challenges.

The precise conclusion would be:

> Late acceleration emerges dynamically from the same IF sector responsible for early clustering and modified gravity.

---

# 43. Relationship to the Informational Battery

Paper 1 defined an informational battery through physically accessible nonequilibrium capacity.

Paper 9 has not yet derived:

\[
b(a)
\]

from that microscopic battery framework.

The cosmological connection remains provisional until the theory shows:

1. what is out of equilibrium;
2. what constitutes the accessible capacity;
3. how the state is measured covariantly;
4. what physical process changes \(b(a)\);
5. why its stress-energy has the required signs and perturbations.

Without that bridge, the phrase **informational** remains interpretive.

---

# 44. Relationship to Cosmic Structure Information

Paper 10 will test whether measurable information and topology in the cosmic web add predictive content beyond conventional two-point statistics.

Paper 9 must not define its IF state using a statistic measured from late structure and then claim to predict that same structure.

Any structure-information coupling must be:

- independently defined;
- temporally causal;
- validated in simulations;
- tested against null universes.

---

# 45. Criteria for Rejection or Major Revision

The Paper 9 unification should be rejected or substantially revised if:

1. the IF field cannot reproduce the CMB without an independent dark-matter-like component;
2. late acceleration requires an independent cosmological constant;
3. expansion-inferred parameters fail growth;
4. growth-inferred parameters fail expansion;
5. reconstructed IF states disagree;
6. parameter splitting is decisively favored;
7. lensing requires an independent slip function;
8. stable parameter space does not overlap good-fit parameter space;
9. the model borrows a \(\Lambda\)CDM sound horizon inconsistently;
10. nonlinear predictions rely on unvalidated standard corrections;
11. effective complexity exceeds the components being replaced;
12. galaxy and cosmology acceleration histories disagree;
13. success depends on one preferred supernova sample without explanation;
14. a future preregistered prediction fails;
15. every failure is addressed by adding a new independent IF function.

---

# 46. Conclusion

The IF cosmological hypothesis does not succeed by fitting more datasets with more freedom.

It succeeds only through enforced dependence.

The same state must explain:

\[
\boxed{
\text{early gravitational clustering}
+
\text{galaxy-scale missing gravity}
+
\text{late accelerated expansion}
+
\text{lensing}.
}
\]

The central consistency condition is:

\[
\boxed{
b_{\mathrm{expansion}}(a)
=
b_{\mathrm{growth}}(a)
=
b_{\mathrm{lensing}}(a).
}
\]

The central parameter condition is:

\[
\boxed{
\theta_E=\theta_G.
}
\]

The decisive experimental sequence is:

\[
\boxed{
\text{fit expansion}
\rightarrow
\text{predict growth}
}
\]

followed by:

\[
\boxed{
\text{fit growth}
\rightarrow
\text{predict expansion}.
}
\]

The stronger temporal test is:

\[
\boxed{
\text{fit the early universe}
\rightarrow
\text{predict the late universe}.
}
\]

If those routes require different IF histories, then the proposed unified geometry has failed.

If they converge on one state and that state also predicts galaxies and future survey measurements, IF Theory will have moved from a conceptual unification to a serious alternative cosmological framework.

The next paper asks whether the cosmic web contains a measurable multiscale informational signature that adds predictive information beyond ordinary clustering statistics:

\[
\boxed{
\textit{Information and Topology in the Cosmic Web:}
\atop
\textit{Testing IF Statistics Beyond the Two-Point Function.}
}
\]

---

# References

1. Planck Collaboration. “Planck 2018 Results VI: Cosmological Parameters.” The final full-mission analysis found strong agreement between the CMB observations and six-parameter flat \(\Lambda\)CDM. citeturn201242search15turn201242search9

2. DESI Collaboration. “DESI DR2 Results II: Measurements of Baryon Acoustic Oscillations and Cosmological Constraints.” The analysis uses more than fourteen million galaxies and quasars and reports BAO and cosmological constraints from three years of DESI observations. citeturn201242search3

3. DESI Collaboration. “DESI DR2 Cosmology Chains and Data Products Released.” The official release includes MCMC chains and posterior-maximizing cosmological products. citeturn201242search1

4. Brout, D. et al. “The Pantheon+ Analysis: Cosmological Constraints.” Pantheon+ contains 1,701 light curves from 1,550 distinct supernovae over \(0.001<z<2.26\). citeturn201242search5

5. Secco, L. F. et al. “Dark Energy Survey Year 3 Results: Cosmology from Cosmic Shear and Robustness to Modeling Uncertainty.” The analysis uses more than one hundred million source galaxies and quantifies major weak-lensing modeling uncertainties. citeturn638735academia34

6. Forero-Sánchez, D. et al. “Cosmological Constraints from a Joint DESI DR1 Full-Shape and DR2 BAO Analysis.” The work combines correlated DESI full-shape and BAO measurements through a compressed joint treatment. citeturn638735academia33

7. ESA. “Planck Publications and Planck Legacy Archive.” Official public Planck products and final-release papers. citeturn201242search9turn201242search24

8. ESA. “Euclid Timeline.” The current schedule lists DR1-Foundation for November 2026. citeturn638735search3
