<!-- extracted from ChatGPT (driver) on 2026-07-18 -->

# A Preregistered IF Forecast for Euclid  
## Equations, Observables, Blinding Rules, and Conditions for Prospective Falsification

**Author:** Phuc Vinh Truong  
**Series:** IF Theory Working Papers  
**Paper:** 11  
**Date:** July 18, 2026  
**Status:** Structural preregistration protocol; numerical forecast package must be frozen before access to the designated Euclid confirmatory products

---

## Abstract

IF Theory proposes that one dynamical geometric state may account for phenomena conventionally separated into dark-matter-like attraction, cosmic structure growth, gravitational lensing, and dark-energy-like expansion. Papers 7–10 defined the theoretical and observational program. The present paper converts that program into a prospective test whose equations, parameter relationships, observables, statistical endpoints, scale cuts, nuisance models, and failure conditions must be frozen before decisive Euclid cosmology products are examined.

Euclid is designed to map the large-scale structure of the Universe across cosmic time, primarily through weak gravitational lensing and galaxy clustering. ESA describes the mission as observing billions of galaxies across more than one-third of the sky to study both the expansion history and the growth of structure. citeturn755666view2turn571713search3

As of July 18, 2026, ESA’s updated release schedule distinguishes a **DR1-Foundation release in November 2026** from the broader **full DR1 release in mid-2027**. The updated DR1 timeline states that the foundation release is expected to include raw data, calibrated images, catalogues, and spectra over approximately \(1900\,\mathrm{deg}^2\). Older Euclid timeline material may still display an earlier October 2026 DR1 date; this paper follows the newer staged schedule. citeturn364680search2turn364680search4

The primary IF forecast is not that Euclid will merely detect a departure from \(\Lambda\)CDM. It is that the same IF state and parameter vector inferred independently from cosmic expansion must predict:

1. redshift-space structure growth;
2. weak-lensing amplitudes and scale dependence;
3. gravitational slip;
4. galaxy–galaxy lensing;
5. environment-sensitive cosmic-web statistics;
6. the redshift evolution of the galactic acceleration scale.

The preregistered consistency condition is:

\[
\boxed{
b_E(z)
=
b_G(z)
=
b_L(z)
=
b_W(z)
=
b_{\mathrm{gal}}(z),
}
\]

where the subscripts denote states inferred from expansion, growth, lensing, cosmic-web organization, and galaxy dynamics.

The corresponding parameter lock is:

\[
\boxed{
\theta_E
=
\theta_G
=
\theta_L
=
\theta_W
=
\theta_{\mathrm{gal}}
=
\theta_{\mathrm{IF}}.
}
\]

The primary Euclid null observables are:

\[
\mathcal N_G(k,z)
=
\mu_{\mathrm{obs}}(k,z)
-
\mu_{\mathrm{IF}}(k,z),
\]

\[
\mathcal N_L(k,z)
=
\Sigma_{\mathrm{obs}}(k,z)
-
\Sigma_{\mathrm{IF}}(k,z),
\]

\[
\mathcal N_b(z)
=
b_{\mathrm{Euclid}}(z)
-
b_{\mathrm{pre-Euclid}}(z),
\]

and:

\[
\mathcal N_{EG}(k,z)
=
E_G^{\mathrm{obs}}(k,z)
-
E_G^{\mathrm{IF}}(k,z).
\]

The theory is prospectively falsified if the designated Euclid observations require independent growth and lensing functions, if parameter splitting is decisively favored, if the stable IF posterior fails the preregistered cross-predictive intervals, if the predicted scale or redshift dependence has the wrong sign, or if the model is altered after the confirmatory data are opened.

This paper establishes a two-stage freeze. **Stage A**, completed by the publication of this protocol, freezes the hypotheses, equations, outcome hierarchy, and anti-hindsight rules. **Stage B** must freeze the numerical forecast vectors, covariance matrices, code hashes, data-independent scale cuts, and pass–fail thresholds after synthetic validation but before access to the designated Euclid confirmatory products. Stage A alone is not sufficient to claim a successful prospective prediction.

---

## Keywords

Euclid; preregistration; modified gravity; dark matter; dark energy; weak gravitational lensing; galaxy clustering; redshift-space distortions; gravitational slip; cosmic web; prospective falsification; IF Theory.

---

# 1. Purpose

A theory becomes easier to preserve after every failed test if it is allowed to change:

- its equations;
- its parameters;
- its redshift dependence;
- its scale dependence;
- its preferred statistic;
- its data selection;
- its interpretation of success.

A genuinely prospective test removes those freedoms before the result is known.

The purpose of Paper 11 is therefore not to forecast that Euclid will favor IF Theory.

It is to establish:

\[
\boxed{
\text{what IF Theory predicts before Euclid decides whether it is true.}
}
\]

The protocol must answer six questions in advance:

1. Which IF model is being tested?
2. Which Euclid observables are confirmatory?
3. Which quantities are calibrated using pre-Euclid data?
4. Which parameters are forbidden from changing?
5. What result counts as compatibility, support, or falsification?
6. What procedures prevent information leakage and hindsight?

---

# 2. Status of This Preregistration

This paper is a **structural preregistration**.

It freezes:

- the hypothesis hierarchy;
- the observable definitions;
- the theoretical consistency locks;
- the model-comparison sequence;
- the primary and secondary endpoints;
- the parameter-splitting tests;
- the blinding strategy;
- the rules governing amendments;
- the interpretation categories.

It does not yet contain the final numerical Euclid forecast vector because the complete IF covariant model, Boltzmann implementation, nonlinear emulator, and synthetic-recovery pipeline have not yet passed the required validation.

Therefore:

\[
\boxed{
\text{Paper 11 is necessary but not sufficient for prospective status.}
}
\]

A valid prospective claim additionally requires a timestamped Stage B package containing:

- numerical predictions;
- uncertainties;
- covariance assumptions;
- exact redshift and scale bins;
- model version;
- code commit;
- environment lock;
- data-manifest hashes;
- cryptographic digest;
- public archival timestamp.

---

# 3. Euclid Release Boundary

## 3.1 Data already available

Euclid’s Quick Data Release 1 was published on March 19, 2025 and included imaging and catalog products from the three Euclid deep fields and an Orion field. Those quick-release products support pipeline development and broad astrophysical analyses but are not equivalent to the mission’s principal wide-survey cosmological data products. citeturn979100search0turn979100search10

Q1 may be used for:

- archive-access testing;
- image-processing validation;
- photometric-catalog familiarity;
- mask and geometry development;
- nonconfirmatory morphology experiments;
- software integration.

Q1 may not be represented as an independent prospective test if it was examined before the Stage B forecast was frozen.

---

## 3.2 DR1-Foundation

ESA’s updated schedule states that the DR1-Foundation release is planned for November 2026 and is expected to cover approximately:

\[
1900\,\mathrm{deg}^2.
\]

The release is expected to include foundational data products such as raw data, calibrated images, catalogues, and spectra. citeturn364680search2turn364680search4

The Stage B manifest must identify which DR1-Foundation tables or derived products are:

- pipeline-development data;
- nuisance-calibration data;
- exploratory data;
- confirmatory data.

The same product may not serve simultaneously as unrestricted development data and a blinded confirmatory test.

---

## 3.3 Full DR1

ESA’s updated Euclid timeline lists the broader full DR1 for mid-2027. citeturn364680search4

Full DR1 is the preferred main confirmatory boundary for any statistic requiring:

- validated cosmological shear products;
- broad tomographic weak-lensing measurements;
- galaxy-clustering likelihoods;
- cross-correlations;
- survey-level covariance;
- cosmological parameter products.

If appropriate cosmological products are not included in the foundation release, the preregistration remains sealed until full DR1.

---

## 3.4 Release contamination rule

A product is considered contaminated for a prospective test when any member of the analysis team has inspected:

- the unblinded statistic;
- parameter constraints;
- residual structure;
- model-ranking result;
- IF-versus-\(\Lambda\)CDM comparison;
- any transformation strongly predictive of the confirmatory endpoint.

Public availability alone does not automatically invalidate a forecast.

Inspection and use do.

The access log must identify:

- person;
- date;
- product;
- purpose;
- whether the product was blinded;
- whether the analysis specification changed afterward.

---

# 4. Euclid’s Relevant Probes

Euclid’s principal cosmological probes are weak gravitational lensing and galaxy clustering, which jointly measure the formation of structure and the geometry of the Universe. citeturn571713search3turn979100search7

The IF forecast uses five probe classes.

## Probe E1 — Spectroscopic galaxy clustering

Measures:

- baryon acoustic oscillations;
- Alcock–Paczyński distortion;
- redshift-space distortions;
- broadband clustering;
- growth rate.

## Probe E2 — Photometric galaxy clustering

Measures:

- projected tracer clustering;
- tomographic density evolution;
- cross-bin correlations;
- magnification effects.

## Probe E3 — Weak gravitational lensing

Measures:

- cosmic shear;
- projected Weyl-potential structure;
- tomographic lensing evolution;
- non-Gaussian convergence statistics.

## Probe E4 — Galaxy–galaxy lensing

Measures:

- the cross-correlation between foreground tracers and background shear;
- the relation between matter dynamics and lensing;
- gravity around different environments and galaxy populations.

## Probe E5 — Cosmic-web and environmental statistics

Measures:

- density-split clustering;
- marked correlations;
- void and filament observables;
- topology;
- multiscale information summaries.

E5 is secondary unless the corresponding estimator has passed Paper 10’s simulation and survey-systematics tests before unblinding.

---

# 5. Theory Being Tested

The preregistration tests a specific model version, not the general sentence:

> Information affects gravity.

The confirmatory model must be a member of Paper 7’s covariant target class:

\[
\boxed{
S
=
\int d^4x\sqrt{-g}
\left[
\frac{M_{\mathrm{Pl}}^2}{2}R
+
M^4
\mathcal L_{\mathrm{IF}}
\left(
X,\mathcal A,\mathcal K_1,\mathcal K_2
\right)
\right]
+
S_m[g_{\mu\nu},\Psi_m].
}
\]

The scalar-defined causal frame is:

\[
X
=
-
\frac{
g^{\mu\nu}\nabla_\mu\Theta\nabla_\nu\Theta
}{
2M^4
},
\]

\[
u_\mu
=
-
\frac{
\nabla_\mu\Theta
}{
M^2\sqrt{2X}
}.
\]

The complete Stage B package must identify the exact:

\[
\mathcal L_{\mathrm{IF}}.
\]

No free function may remain unspecified in the confirmatory model.

---

# 6. Stable-Model Entry Requirements

The IF model may enter the Euclid preregistration only after it satisfies all of the following.

## 6.1 Covariant closure

The background, scalar perturbations, tensor perturbations, lensing, and galaxy limits derive from one action.

## 6.2 General-relativistic local limit

The model possesses a derived limit compatible with local gravity tests.

## 6.3 Luminal tensor propagation

The tested late-time branch satisfies:

\[
c_T^2=1
\]

to the required observational accuracy.

## 6.4 No ghost

\[
Q_s>0,
\qquad
Q_T>0.
\]

## 6.5 No gradient instability

\[
c_s^2\geq0,
\qquad
c_T^2\geq0.
\]

## 6.6 Adequate cutoff

The effective-theory cutoff exceeds the cosmological wavenumbers included in the analysis.

## 6.7 Numerical regression

The modified cosmological solver reproduces its reference limits and satisfies constraint residual tolerances.

A model that fails one entry requirement may not be rescued by a favorable Euclid likelihood.

---

# 7. Frozen IF State

Let the homogeneous state be:

\[
b(a).
\]

It obeys:

\[
\boxed{
\frac{db}{d\ln a}
=
-\Gamma_{\mathrm{IF}}
\left(
b,a;\theta_b
\right).
}
\]

The Stage B model freezes:

- the function \(\Gamma_{\mathrm{IF}}\);
- its parameter vector;
- the initial condition;
- any branch-selection rule;
- the range of physical solutions.

The same state determines:

\[
\boxed{
\left\{
H,\rho_{\mathrm{IF}},p_{\mathrm{IF}},
\mu,\eta,\Sigma,
a_{\mathrm{IF}}
\right\}.
}
\]

No Euclid-specific state variable is added.

---

# 8. Background Equations

The background equations are:

\[
\boxed{
3M_{\mathrm{Pl}}^2H^2
=
\rho_r+\rho_b+\rho_\nu+\rho_{\mathrm{IF}},
}
\]

\[
\boxed{
-2M_{\mathrm{Pl}}^2\dot H
=
\frac43\rho_r+\rho_b+\rho_\nu+p_\nu
+
\rho_{\mathrm{IF}}+p_{\mathrm{IF}}.
}
\]

The minimal IF model includes:

- radiation;
- baryons;
- neutrinos;
- the IF sector.

It excludes:

- an independent cold-dark-matter fluid;
- an independent cosmological-constant term;
- an independently fitted dark-energy equation of state.

If either excluded component is added after the Euclid result is known, the minimal hypothesis has been falsified.

A later hybrid model may be studied under a new preregistration.

---

# 9. Linear Gravity Functions

The scalar perturbations are represented by:

\[
-k^2\Psi
=
4\pi Ga^2
\mu_{\mathrm{IF}}(k,a)
\rho_{\mathrm{cl}}\Delta_{\mathrm{cl}},
\]

\[
\eta_{\mathrm{IF}}(k,a)
=
\frac{\Phi}{\Psi},
\]

and:

\[
\boxed{
\Sigma_{\mathrm{IF}}(k,a)
=
\frac{
\mu_{\mathrm{IF}}(k,a)
\left[
1+\eta_{\mathrm{IF}}(k,a)
\right]
}{
2
}.
}
\]

Here \(\rho_{\mathrm{cl}}\Delta_{\mathrm{cl}}\) denotes the complete clustering source derived from baryons, neutrinos, and IF perturbations.

The functions:

\[
\mu_{\mathrm{IF}},
\qquad
\eta_{\mathrm{IF}},
\qquad
\Sigma_{\mathrm{IF}}
\]

must be calculated by the model.

They may not be independent spline functions.

---

# 10. Growth Equations

Define the logarithmic growth rate:

\[
f(k,a)
=
\frac{
d\ln D(k,a)
}{
d\ln a
}.
\]

For the relevant clustering field, the predicted growth observable is:

\[
\boxed{
f\sigma_8(k,z).
}
\]

A schematic subhorizon equation is:

\[
D''+
\left(
2+\frac{H'}{H}
\right)D'
-
\frac32
\Omega_{\mathrm{cl}}(a)
\mu_{\mathrm{IF}}(k,a)
D
=
S_{\mathrm{rel}}(k,a),
\]

where \(S_{\mathrm{rel}}\) contains relativistic and additional-field contributions.

The actual forecast uses the full perturbation solver, not this approximation.

---

# 11. Weak-Lensing Observables

For source redshift bins \(i,j\), the tomographic shear spectrum is:

\[
\boxed{
C_\ell^{\gamma_i\gamma_j}
=
\int
\frac{d\chi}{\chi^2}
W_i^\gamma(\chi)
W_j^\gamma(\chi)
P_{\Phi+\Psi}
\left(
k=\frac{\ell+1/2}{\chi},
z(\chi)
\right).
}
\]

The Weyl-potential spectrum depends on:

\[
\Sigma_{\mathrm{IF}}(k,z).
\]

The Stage B forecast freezes:

- redshift-bin definitions;
- multipole range;
- Limber or non-Limber treatment;
- nonlinear prescription;
- intrinsic-alignment model;
- photometric-redshift nuisance model;
- shear-calibration nuisance model;
- baryonic-feedback treatment.

---

# 12. Galaxy-Clustering Observables

For spectroscopic tracers, the redshift-space galaxy power spectrum may be represented schematically by:

\[
\boxed{
P_g^s(k,\mu,z)
=
\left[
b_g(k,z)
+
f(k,z)\mu^2
\right]^2
P_m(k,z)
D_{\mathrm{FoG}}
+
P_{\mathrm{shot}}
+
P_{\mathrm{corr}}.
}
\]

The primary compressed observables include:

\[
P_0(k,z),
\qquad
P_2(k,z),
\qquad
P_4(k,z),
\]

together with BAO and Alcock–Paczyński information.

The Stage B package freezes:

- tracer classes;
- redshift bins;
- \(k\)-range;
- galaxy-bias expansion;
- counterterms;
- velocity-dispersion model;
- shot-noise model;
- window-convolution procedure.

---

# 13. Galaxy–Galaxy Lensing

For foreground bin \(i\) and source bin \(j\):

\[
\boxed{
C_\ell^{g_i\gamma_j}
=
\int
\frac{d\chi}{\chi^2}
W_i^g(\chi)
W_j^\gamma(\chi)
P_{g,\Phi+\Psi}
\left(
\frac{\ell+1/2}{\chi},z
\right).
}
\]

Galaxy–galaxy lensing is especially valuable because it connects:

- tracer clustering;
- gravitational dynamics;
- lensing;
- galaxy bias.

The same nuisance model for foreground tracers must be used in:

\[
C_\ell^{gg}
\]

and:

\[
C_\ell^{g\gamma}.
\]

A separate lensing mass normalization is prohibited.

---

# 14. The \(E_G\) Consistency Observable

A gravity consistency statistic may be defined schematically as:

\[
\boxed{
E_G(k,z)
=
\frac{
k^2\left[
\Phi(k,z)+\Psi(k,z)
\right]
}{
3H_0^2a^{-1}
f(k,z)\delta(k,z)
}.
}
\]

In the IF model:

\[
\boxed{
E_G^{\mathrm{IF}}(k,z)
=
\mathcal E
\left[
\mu_{\mathrm{IF}}(k,z),
\eta_{\mathrm{IF}}(k,z),
f(k,z),
\Omega_{\mathrm{cl}}(z)
\right].
}
\]

Euclid can constrain \(E_G\)-like combinations through galaxy clustering, redshift-space distortions, and galaxy–galaxy lensing.

The primary IF prediction concerns both:

- amplitude;
- scale dependence.

An arbitrary constant rescaling after unblinding is forbidden.

---

# 15. Galactic Acceleration Lock

Paper 8 proposed the restricted relation:

\[
\boxed{
a_{\mathrm{IF}}(z)
=
a_{\mathrm{IF},0}
\frac{H(z)}{H_0}.
}
\]

The general covariant model may derive a different fixed function:

\[
a_{\mathrm{IF}}(z)
=
a_{\mathrm{IF},0}
F_a[b(z)].
\]

The Stage B package must choose exactly one.

The choice is frozen before the Euclid confirmatory data are opened.

Euclid does not directly measure resolved rotation curves for the full survey sample, but its expansion, lensing, clustering, environment, and galaxy-population information constrain the state that predicts:

\[
a_{\mathrm{IF}}(z).
\]

The cosmology-to-galaxy forecast is:

\[
\boxed{
P
\left[
a_{\mathrm{IF}}(z)
\mid
\mathcal D_{\mathrm{Euclid}}
\right].
}
\]

This prediction is compared with independent galaxy-kinematic measurements.

---

# 16. Cosmic-Web State Lock

Paper 10 defined:

\[
\hat b_W(z)
=
\mathcal E_W
\left[
\mathbf I_{\mathrm{IF}}(z)
\right].
\]

The cosmic-web vector may include:

\[
\mathbf I_{\mathrm{IF}}
=
\left[
\mathcal I_{\mathrm{top}},
\mathcal I_{\mathrm{tidal}},
\mathcal I_{\mathrm{marked}},
\mathcal I_{\mathrm{memory}},
\ldots
\right].
\]

For Euclid, the primary web statistic is selected only after simulation validation.

Once frozen:

- its smoothing scales;
- topology filtration;
- mark;
- redshift bins;
- estimator architecture;
- summary dimension

may not be changed in response to the real result.

The required state identity is:

\[
\boxed{
\hat b_W(z)
=
b_E(z).
}
\]

---

# 17. Hypothesis Hierarchy

The analysis distinguishes five nested hypotheses.

## H0 — Flat \(\Lambda\)CDM

General relativity with conventional cold dark matter and a cosmological constant.

## H1 — Flexible late-time dark energy

Conventional dark matter with a flexible late expansion history.

## H2 — Flexible modified gravity

Independent low-dimensional:

\[
\mu(k,z)
\]

and:

\[
\eta(k,z)
\]

functions.

## H3 — Split IF

IF language and functional forms are retained, but expansion, growth, and lensing receive separate parameter vectors.

## H4 — Unified IF

One stable covariant IF action and one parameter vector generate every sector.

The critical comparison is:

\[
\boxed{
H4\ \text{versus}\ H3.
}
\]

If H3 is strongly required, the proposed unification fails even when some modified-gravity effect is present.

---

# 18. Calibration Data

The unified IF model may be calibrated before Euclid using a declared pre-Euclid dataset:

\[
\mathcal D_{\mathrm{pre}}
=
\left\{
\mathcal D_{\mathrm{CMB}},
\mathcal D_{\mathrm{BAO}},
\mathcal D_{\mathrm{SNe}},
\mathcal D_{\mathrm{growth}},
\mathcal D_{\mathrm{gal}}
\right\}.
\]

The exact dataset versions must be frozen.

For every parameter sample:

\[
\theta^{(s)}
\sim
P
\left(
\theta
\mid
\mathcal D_{\mathrm{pre}}
\right),
\]

the pipeline produces Euclid observables:

\[
\mathbf y_{\mathrm{Euclid}}^{(s)}
=
\mathfrak F_{\mathrm{Euclid}}
\left[
\theta^{(s)}
\right].
\]

The resulting pre-Euclid predictive distribution is:

\[
\boxed{
P
\left(
\mathbf y_{\mathrm{Euclid}}
\mid
\mathcal D_{\mathrm{pre}},
H4
\right).
}
\]

This distribution is frozen.

Euclid data may update the posterior but may not redefine what H4 predicted.

---

# 19. Primary Confirmatory Endpoints

## Endpoint P1 — Expansion-to-growth prediction

Use pre-Euclid expansion-calibrated parameters to predict Euclid clustering growth.

\[
S_{E\rightarrow G}
=
\ln
P
\left(
\mathcal D_G^{\mathrm{Euclid}}
\mid
\mathcal D_E^{\mathrm{pre}},H4
\right).
\]

---

## Endpoint P2 — Expansion-to-lensing prediction

Use the same expansion posterior to predict Euclid weak lensing.

\[
S_{E\rightarrow L}
=
\ln
P
\left(
\mathcal D_L^{\mathrm{Euclid}}
\mid
\mathcal D_E^{\mathrm{pre}},H4
\right).
\]

---

## Endpoint P3 — Growth–lensing relation

Test the frozen IF relation between:

\[
\mu
\]

and:

\[
\Sigma.
\]

The endpoint is the Euclid likelihood of the joint:

\[
E_G(k,z)
\]

or equivalent growth–lensing statistic.

---

## Endpoint P4 — Parameter splitting

Compare:

\[
\theta_E=\theta_G=\theta_L
\]

against separate:

\[
\theta_E,\theta_G,\theta_L.
\]

---

## Endpoint P5 — IF state reconstruction

Compare:

\[
b_E(z),
\quad
b_G(z),
\quad
b_L(z).
\]

---

# 20. Secondary Endpoints

## S1 — Web-state consistency

\[
b_W(z)=b_E(z).
\]

## S2 — Galaxy-state consistency

\[
a_{\mathrm{IF}}(z)
=
a_{\mathrm{IF},0}F_a[b_E(z)].
\]

## S3 — Scale-dependent growth

Test the predicted:

\[
k
\]

dependence of:

\[
f\sigma_8(k,z).
\]

## S4 — Scale-dependent slip

Test the predicted scale dependence of:

\[
\eta(k,z).
\]

## S5 — Environmental marked statistic

Test a simulation-frozen low-density or tidal mark.

## S6 — Non-Gaussian weak-lensing statistic

Test a frozen convergence topology or map statistic.

Secondary endpoints cannot rescue a failure of the primary endpoints.

---

# 21. Exact Null Vectors

Define a Euclid data vector:

\[
\mathbf d
=
\left[
\mathbf d_G,
\mathbf d_L,
\mathbf d_{GL},
\mathbf d_W
\right].
\]

Define the pre-Euclid IF prediction:

\[
\mathbf m_{\mathrm{IF}}.
\]

The residual is:

\[
\boxed{
\mathbf r_{\mathrm{IF}}
=
\mathbf d-\mathbf m_{\mathrm{IF}}.
}
\]

With preregistered covariance \(C\):

\[
\boxed{
Q_{\mathrm{IF}}
=
\mathbf r_{\mathrm{IF}}^\top
C^{-1}
\mathbf r_{\mathrm{IF}}.
}
\]

The distribution of \(Q_{\mathrm{IF}}\) is calibrated using synthetic surveys, not assumed to be exactly chi-squared when:

- covariance depends on parameters;
- likelihoods are non-Gaussian;
- nuisance parameters are marginalized;
- nonlinear summaries are used.

---

# 22. State-Consistency Statistic

Let:

\[
\mathbf b_E,
\quad
\mathbf b_G,
\quad
\mathbf b_L,
\quad
\mathbf b_W
\]

be state vectors evaluated on common redshift nodes.

Define:

\[
\bar{\mathbf b}
=
C_{\mathrm{tot}}
\sum_X
C_X^{-1}\mathbf b_X,
\]

where:

\[
C_{\mathrm{tot}}
=
\left(
\sum_X C_X^{-1}
\right)^{-1}.
\]

Define:

\[
\boxed{
Q_b
=
\sum_X
\left(
\mathbf b_X-\bar{\mathbf b}
\right)^\top
C_X^{-1}
\left(
\mathbf b_X-\bar{\mathbf b}
\right).
}
\]

The state-consistency \(p\)-value is calibrated on unified IF mocks.

A large value means that different probes require different IF histories.

---

# 23. Parameter-Split Statistic

Let the unified parameter vector be:

\[
\theta.
\]

Let the split vector be:

\[
\theta_{\mathrm{split}}
=
\left[
\theta_E,
\theta_G,
\theta_L
\right].
\]

Compute:

- evidence difference;
- held-out predictive difference;
- likelihood-ratio statistic;
- posterior parameter tension.

Define:

\[
\Delta\ln Z_{\mathrm{split}}
=
\ln Z_{\mathrm{split}}
-
\ln Z_{\mathrm{unified}}.
\]

The provisional interpretation is:

\[
\Delta\ln Z_{\mathrm{split}}<1:
\quad
\text{no meaningful demand for splitting},
\]

\[
1\leq\Delta\ln Z_{\mathrm{split}}<3:
\quad
\text{weak demand},
\]

\[
3\leq\Delta\ln Z_{\mathrm{split}}<5:
\quad
\text{strong concern},
\]

\[
\boxed{
\Delta\ln Z_{\mathrm{split}}\geq5:
\quad
\text{unification rejected}
}
\]

provided the evidence calculation has passed synthetic calibration.

A simulation-calibrated frequentist equivalent may be used as a co-primary check.

---

# 24. Prospective Falsification Rules

Unified IF is rejected under any one of the following hard conditions.

## F1 — Stability failure

The Euclid-supported parameter region has no meaningful overlap with the theoretically stable region.

## F2 — Cross-prediction failure

At least two independent primary probe classes fall outside the frozen 99% posterior predictive region in a coherent direction.

## F3 — Parameter-split failure

The split IF model is decisively favored:

\[
\Delta\ln Z_{\mathrm{split}}\geq5,
\]

with confirmed calibration and no identified data-processing failure.

## F4 — State inconsistency

The state-consistency statistic exceeds the preregistered 99.9th percentile of unified-IF mock universes.

## F5 — Wrong-sign prediction

A distinctive IF deviation is detected with the opposite sign from the frozen prediction at at least:

\[
5\sigma.
\]

## F6 — Wrong scale dependence

The observed scale dependence cannot be produced anywhere in the stable frozen IF parameter space.

## F7 — Independent slip required

Lensing requires an independent normalization or slip function not generated by the action.

## F8 — Independent growth source required

Structure formation requires an added clustering component outside the frozen IF action.

## F9 — Independent acceleration required

Expansion requires an added cosmological constant or dark-energy term.

## F10 — Model amendment after unblinding

A substantive equation or parameter relation is changed after confirmatory results are inspected.

F10 does not prove nature rejected all possible IF theories.

It invalidates the preregistered test.

---

# 25. Compatibility Is Not Confirmation

If Euclid observables lie within the IF predictive distribution, the result is:

\[
\textbf{compatible},
\]

not necessarily:

\[
\textbf{confirmed}.
\]

Compatibility may occur because IF closely reproduces \(\Lambda\)CDM.

For positive evidence, the model must make a distinctive prediction.

A valid confirmation requires all of:

1. the prediction differs materially from baseline models;
2. the sign and amplitude were frozen;
3. Euclid detects the predicted effect;
4. the effect appears in at least two related probes;
5. split or flexible alternatives do not explain it substantially better;
6. independent analysts reproduce the result.

---

# 26. Evidence Categories

## Outcome E0 — Invalid test

The analysis suffered:

- data leakage;
- code failure;
- covariance failure;
- untracked amendments;
- inadequate mock calibration.

No scientific conclusion is assigned.

---

## Outcome E1 — IF falsified

One or more hard falsification rules are met.

---

## Outcome E2 — IF compatible but unnecessary

Unified IF fits Euclid, but:

- \(\Lambda\)CDM performs comparably or better;
- no distinctive IF deviation is detected;
- the IF model is effectively a more complicated reparameterization.

---

## Outcome E3 — New-physics anomaly, not uniquely IF

Euclid detects a departure from \(\Lambda\)CDM, but:

- flexible modified gravity;
- evolving dark energy;
- systematics;
- neutrinos;
- baryonic effects

explain it as well as or better than unified IF.

---

## Outcome E4 — IF supported

Unified IF:

- passes every primary cross-prediction;
- avoids parameter splitting;
- outperforms simpler alternatives predictively;
- correctly predicts at least one distinctive deviation.

---

## Outcome E5 — Prospective IF confirmation

E4 is achieved and the result is:

- independently reproduced;
- observed in a later release or independent survey;
- quantitatively consistent without model revision.

---

# 27. Baseline Comparison Rules

The IF model is compared with:

1. flat \(\Lambda\)CDM;
2. \(w_0w_a\)CDM;
3. low-dimensional phenomenological modified gravity;
4. unified dark fluid;
5. split IF;
6. one existing covariant modified-gravity theory;
7. a systematics-expanded \(\Lambda\)CDM model.

The primary comparison metric is held-out predictive performance, supplemented by evidence and effective complexity.

No model is intentionally weakened through:

- narrow nuisance priors;
- inferior nonlinear treatment;
- omitted covariance;
- obsolete calibration.

---

# 28. Effective Complexity

Report:

- number of physical parameters;
- number of nuisance parameters;
- number of unconstrained functional degrees of freedom;
- posterior effective dimensionality;
- prior information;
- computational complexity.

The claim:

> IF has fewer components than dark matter plus dark energy

is not meaningful if one IF free function contains dozens of effective degrees of freedom.

The relevant comparison is predictive compression:

\[
\boxed{
\mathcal C_{\mathrm{pred}}
=
\frac{
\text{held-out information explained}
}{
\text{effective model complexity}
}.
}
\]

---

# 29. Blinding Plan

## 29.1 Parameter blinding

Apply secret offsets or linear transformations to:

- IF transition parameters;
- growth-amplitude parameters;
- state-consistency plots;
- model-comparison statistics.

## 29.2 Observable blinding

Where possible, hide the absolute amplitude of:

- shear spectra;
- \(E_G\);
- parameter splitting;
- web-state reconstruction.

## 29.3 Label blinding

During pipeline validation, analysts may receive simulations labeled only:

- Model A;
- Model B;
- Model C.

## 29.4 Unblinding checklist

Unblinding occurs only after:

- all code tests pass;
- scale cuts are frozen;
- nuisance models are frozen;
- covariance is validated;
- synthetic coverage passes;
- the preregistration digest is archived;
- the red-team report is signed.

---

# 30. Analysis Firewalls

Use distinct roles where practical.

## Forecast team

Produces the Stage B IF prediction.

## Pipeline team

Implements Euclid data handling without knowing the forecast direction where feasible.

## Systematics team

Attempts to generate false IF signals.

## Red-team group

Attempts to reject the analysis before unblinding.

## Unblinding custodian

Maintains blinding keys and the amendment log.

A solo-founder implementation cannot create perfect organizational independence.

In that case, separation is simulated through:

- isolated repositories;
- encrypted prediction files;
- automated scoring;
- AI agents with restricted context;
- timestamped logs;
- external replication requests.

---

# 31. Nuisance Parameters

The full nuisance vector may include:

\[
\phi
=
\left[
\delta z_i,
m_i,
A_{\mathrm{IA}},
\eta_{\mathrm{IA}},
b_{g,i},
b_{2,i},
b_{s^2,i},
P_{\mathrm{shot},i},
A_{\mathrm{bary}},
\Sigma m_\nu,
\ldots
\right].
\]

The same nuisance treatment must be applied consistently to:

- IF;
- \(\Lambda\)CDM;
- modified-gravity baselines.

Nuisance priors are fixed using:

- external calibration;
- Euclid validation products permitted by the preregistration;
- simulation-based bounds.

They may not be tightened selectively to harm a comparator.

---

# 32. Conservative Scale Cuts

The Stage B package defines:

\[
k_{\max}(z)
\]

and:

\[
\ell_{\max}^{ij}
\]

before unblinding.

The initial confirmatory analysis uses scales where:

- perturbation theory is validated;
- IF nonlinear corrections are tested;
- baryonic uncertainty is controlled;
- bias expansion remains adequate;
- covariance is stable.

A later small-scale analysis may improve power but is secondary.

The primary result must survive conservative scales.

---

# 33. Nonlinear IF Modeling

Euclid precision requires nonlinear modeling.

The IF program must use one of:

1. dedicated IF \(N\)-body simulations;
2. a validated perturbative prescription;
3. an emulator trained on dedicated IF simulations;
4. conservative exclusion of unvalidated nonlinear scales.

Using a \(\Lambda\)CDM nonlinear correction without validation is prohibited.

The nonlinear pipeline must predict both:

- matter clustering;
- Weyl-potential clustering.

---

# 34. Survey-Systematics Injection

Before unblinding, unified IF mocks and baseline mocks are passed through:

- survey mask;
- depth variation;
- photometric-redshift errors;
- shear calibration;
- blending;
- intrinsic alignments;
- redshift failures;
- selection functions;
- galaxy bias;
- fiber or slitless-spectroscopy effects;
- magnification;
- observational noise.

The analysis must correctly recover the input model under these conditions.

---

# 35. Synthetic Recovery Requirements

Stage B cannot be frozen until the pipeline demonstrates:

## 35.1 Parameter recovery

Posterior means are acceptably unbiased.

## 35.2 Coverage

Nominal intervals achieve empirical coverage.

## 35.3 False-positive control

\(\Lambda\)CDM mocks are not systematically identified as IF.

## 35.4 False-negative control

Distinctive IF mocks are detected at the expected rate.

## 35.5 Split-model calibration

The parameter-split statistic has known behavior under both unified and split universes.

## 35.6 Systematics diagnosis

Injected systematic errors are not routinely mistaken for the frozen IF signal.

---

# 36. Numerical Forecast Package

The Stage B forecast contains the following frozen arrays.

```yaml
redshift_bins:
  spectroscopic: []
  photometric_lens: []
  photometric_source: []

scale_bins:
  k_edges: []
  ell_edges: []

if_predictions:
  H_over_H0: []
  angular_diameter_distance: []
  f_sigma8: []
  mu: []
  eta: []
  sigma_lensing: []
  E_G: []
  C_ell_gg: []
  C_ell_gammagamma: []
  C_ell_ggamma: []
  web_state: []
  galaxy_acceleration_scale: []

prediction_covariance:
  file_hash: null

primary_null_vectors:
  growth: []
  lensing: []
  growth_lensing: []
  state_consistency: []

pass_fail_thresholds:
  cross_prediction: null
  parameter_split: null
  state_consistency: null
  wrong_sign: null
```

No value may remain marked `null` when the forecast is sealed.

---

# 37. Cryptographic Freeze

The Stage B release must include:

- Git commit;
- container or environment digest;
- source archive checksum;
- forecast-array checksum;
- data-manifest checksum;
- notebook-output checksum;
- PDF or manuscript checksum;
- public timestamp.

A suitable manifest is:

```yaml
preregistration_id: IF-EUCLID-11-B
paper_version: 11.0
freeze_datetime_utc: null

git_commit: null
repository_archive_sha256: null
container_digest: null
lockfile_sha256: null

theory_model_id: null
lagrangian_sha256: null
equation_manifest_sha256: null
parameter_prior_sha256: null

forecast_file_sha256: null
covariance_file_sha256: null
mock_validation_sha256: null

euclid_products_reserved_for_confirmation: []
euclid_products_allowed_for_development: []

unblinding_key_custodian: null
public_timestamp_location: null
```

---

# 38. Amendment Rules

## 38.1 Before Stage B freeze

The model may change freely, but no later claim may describe Stage A alone as a numerical prediction.

## 38.2 After Stage B but before confirmatory access

A correction is allowed only for:

- demonstrated coding error;
- impossible data-product assumption;
- confirmed unit error;
- documented mission-product change.

The old forecast remains archived.

The amendment receives a new version.

## 38.3 After confirmatory access

No substantive amendment preserves prospective status.

A revised model is considered:

\[
\text{IF-Euclid Postdiction 1},
\]

not the original forecast.

---

# 39. Positive Prediction Candidates

The final Stage B package must choose one primary distinctive prediction.

Candidates include:

## Candidate D1 — Scale-dependent \(E_G\)

\[
E_G(k,z)
\]

departs from the \(\Lambda\)CDM expectation with a frozen sign and transition scale.

## Candidate D2 — Growth–lensing mismatch relative to GR

\[
\Sigma_{\mathrm{IF}}(k,z)
\neq
\mu_{\mathrm{IF}}(k,z)
\]

in a fixed relationship.

## Candidate D3 — Redshift-localized state transition

A feature in:

\[
f\sigma_8(z),
\quad
C_\ell^{\gamma\gamma},
\quad
E_G(z)
\]

occurs near a state-transition redshift fixed by pre-Euclid expansion data.

## Candidate D4 — Environment-enhanced modified gravity

A preregistered low-density mark amplifies the same IF signal predicted by the covariant model.

## Candidate D5 — Web–lensing state agreement

A topology-derived state and a weak-lensing-derived state independently reconstruct the same:

\[
b(z).
\]

Only one candidate is primary.

The others remain secondary.

---

# 40. Hardest IF Prediction

The most scientifically valuable forecast is not the largest expected statistical deviation.

It is the most overconstrained one.

The proposed primary target is:

\[
\boxed{
E_G^{\mathrm{IF}}(k,z)
}
\]

because it combines:

- galaxy velocities;
- clustering;
- lensing;
- growth;
- gravitational slip.

Once the pre-Euclid IF action is frozen:

\[
E_G^{\mathrm{IF}}(k,z)
\]

has no independent Euclid normalization.

The theory predicts:

- its redshift trend;
- its scale trend;
- its amplitude;
- its covariance with growth and lensing.

---

# 41. Cross-Release Replication

The ideal sequence is:

## Test 1 — DR1-Foundation or eligible initial product

Apply the frozen pipeline to the earliest suitable confirmatory product.

## Test 2 — Full DR1

Repeat without revising the physical model.

Nuisance updates are permitted only if specified prospectively.

## Test 3 — Later Euclid release

Test the same physical parameter relations over a larger volume.

## Test 4 — Independent survey

Repeat with:

- DESI;
- Rubin/LSST;
- Roman;
- CMB lensing;
- independent galaxy dynamics.

A result appearing in only one release remains provisional.

---

# 42. Core Hypotheses

## EF-H1 — Unified-state hypothesis

Euclid expansion, growth, and lensing reconstruct one IF state.

### Falsifier

\[
b_E(z),
\quad
b_G(z),
\quad
b_L(z)
\]

are incompatible.

---

## EF-H2 — Parameter-lock hypothesis

One parameter vector predicts every primary Euclid probe.

### Falsifier

The split model is decisively favored.

---

## EF-H3 — Growth forecast hypothesis

The pre-Euclid expansion posterior predicts Euclid growth.

### Falsifier

Euclid growth lies outside the frozen predictive distribution.

---

## EF-H4 — Lensing forecast hypothesis

The pre-Euclid expansion and growth model predicts Euclid weak lensing.

### Falsifier

An independent lensing normalization or slip function is required.

---

## EF-H5 — \(E_G\) hypothesis

The frozen IF \(E_G(k,z)\) relation matches Euclid.

### Falsifier

The observed amplitude, sign, or scale dependence is incompatible.

---

## EF-H6 — Stable-posterior hypothesis

Euclid-supported parameters remain theoretically stable.

### Falsifier

Only unstable parameter space fits.

---

## EF-H7 — Galaxy–cosmology hypothesis

The Euclid-inferred state predicts the independently measured galactic acceleration-scale evolution.

### Falsifier

Galaxy dynamics require another state or scaling law.

---

## EF-H8 — Web-state hypothesis

The frozen cosmic-web estimator recovers the same state as lensing and clustering.

### Falsifier

The web state requires an independent redshift calibration.

---

## EF-H9 — Predictive-compression hypothesis

Unified IF explains the combined data without greater effective freedom than the components it replaces.

### Falsifier

Its fit depends on excessive functional flexibility.

---

## EF-H10 — Prospective-repeatability hypothesis

The full DR1 or later release reproduces the initial IF result without physical-model revision.

### Falsifier

The signal fails to recur.

---

# 43. Notebook Program

## Notebook 11A — Preregistration Manifest Builder

Creates and validates:

- model identifiers;
- data boundaries;
- hypotheses;
- endpoints;
- hashes;
- amendment policy.

---

## Notebook 11B — Covariant Equation Freeze

Exports the exact action and derived equations into a machine-readable symbolic manifest.

---

## Notebook 11C — Stability Prior

Calculates:

\[
Q_s,
\quad
c_s^2,
\quad
Q_T,
\quad
c_T^2.
\]

Produces the frozen allowed domain.

---

## Notebook 11D — Pre-Euclid Posterior

Fits only the declared calibration data.

Saves immutable posterior samples.

---

## Notebook 11E — Euclid Observable Generator

Maps each pre-Euclid posterior sample to:

- clustering;
- shear;
- galaxy–galaxy lensing;
- \(E_G\);
- web-state predictions.

---

## Notebook 11F — Euclid Bin Optimizer

Selects bins using simulations only.

It must not inspect real confirmatory endpoints.

---

## Notebook 11G — Scale-Cut Validation

Determines:

\[
k_{\max}(z)
\]

and:

\[
\ell_{\max}^{ij}.
\]

---

## Notebook 11H — Nonlinear Emulator Validation

Tests accuracy throughout the frozen prior domain.

---

## Notebook 11I — Survey-Systematics Injector

Creates realistic Euclid-like mock observations.

---

## Notebook 11J — Unified IF Mock Suite

Generates synthetic universes from the frozen IF model.

---

## Notebook 11K — \(\Lambda\)CDM Mock Suite

Measures false-positive IF detection.

---

## Notebook 11L — Split-IF Mock Suite

Calibrates parameter-split detection power.

---

## Notebook 11M — Coverage and Calibration

Tests posterior and predictive interval coverage.

---

## Notebook 11N — \(E_G\) Forecast

Produces the frozen:

\[
E_G(k,z)
\]

matrix and covariance.

---

## Notebook 11O — State-Reconstruction Forecast

Produces expected uncertainties on:

\[
b_E,
\quad
b_G,
\quad
b_L.
\]

---

## Notebook 11P — Cosmic-Web Forecast

Freezes the Paper 10 state estimator and topology statistic.

---

## Notebook 11Q — Galaxy-Acceleration Forecast

Produces:

\[
a_{\mathrm{IF}}(z)
\]

from the pre-Euclid posterior.

---

## Notebook 11R — Blinding Generator

Creates parameter and observable transformations.

---

## Notebook 11S — Automated Scorer

Accepts a blinded Euclid data vector and returns only:

- pass;
- warning;
- hard failure;
- invalid test.

---

## Notebook 11T — Parameter-Split Test

Computes:

\[
\Delta\ln Z_{\mathrm{split}}.
\]

---

## Notebook 11U — State-Consistency Test

Computes:

\[
Q_b.
\]

---

## Notebook 11V — Baseline Model Comparison

Runs every comparator through identical nuisance and scale treatments.

---

## Notebook 11W — Amendment Auditor

Detects changes in:

- equations;
- priors;
- binning;
- scale cuts;
- statistics;
- nuisance models.

---

## Notebook 11X — Forecast Sealer

Creates:

- immutable archives;
- SHA-256 digests;
- release notes;
- public timestamp package.

---

## Notebook 11Y — Confirmatory Euclid Run

Runs only after authorization to unseal the reserved products.

---

## Notebook 11Z — Independent Reproduction

A separate team or AI implementation reconstructs the result from the sealed package.

---

# 44. Reproducibility Record

Each Euclid analysis emits:

```yaml
experiment_id: if-euclid-forecast-11
preregistration_stage: B
preregistration_version: null
freeze_datetime_utc: null

paper_hash: null
git_commit: null
repository_hash: null
container_digest: null
environment_lock_hash: null

euclid_release: null
euclid_product_ids: []
euclid_product_hashes: []
access_log_hash: null

theory_model_id: null
action_hash: null
equation_hash: null
parameter_prior_hash: null
stability_prior_hash: null

redshift_bins_hash: null
scale_cuts_hash: null
nuisance_model_hash: null
covariance_hash: null

pre_euclid_posterior_hash: null
forecast_vector_hash: null
forecast_covariance_hash: null

growth_cross_predictive_score: null
lensing_cross_predictive_score: null
E_G_score: null
parameter_split_log_evidence: null
state_consistency_statistic: null
web_state_consistency: null
galaxy_state_consistency: null

baseline_model_scores: {}
effective_complexities: {}

blinding_version: null
unblinding_datetime_utc: null
amendment_count_after_freeze: null

decision:
  test_validity: null
  compatibility: null
  support_level: null
  falsification_rule_triggered: null

result_hash: null
```

---

# 45. Failure Modes

## 45.1 Structural preregistration presented as numerical prediction

Paper 11 is published, but no numerical Stage B package is sealed before Euclid access.

## 45.2 Data leakage

Analysts inspect confirmatory results before freezing equations or cuts.

## 45.3 Moving primary endpoint

The primary statistic changes after the first result is disappointing.

## 45.4 Parameter resurrection

A parameter fixed before Euclid is allowed to vary afterward.

## 45.5 Hidden model split

Background and perturbation code use nominally identical parameters with different effective meanings.

## 45.6 Unstable fit

The best observational solution contains a ghost or gradient instability.

## 45.7 Scale-cut tuning

Small-scale bins are included or removed according to their effect on IF significance.

## 45.8 Nuisance asymmetry

IF and \(\Lambda\)CDM receive different nuisance freedom.

## 45.9 Covariance hindsight

The covariance model is changed because the original one weakens IF evidence.

## 45.10 Web-statistic shopping

Many topology and information statistics are tested, and only the most favorable is reported.

## 45.11 Wrong-sign reinterpretation

A predicted positive deviation is observed negative and described as a different IF regime.

## 45.12 Foundation/full-release confusion

An exploratory foundation-release analysis is later described as a blinded full-DR1 prediction.

## 45.13 Hybrid rescue

Cold dark matter or a cosmological constant is added after failure without declaring the minimal model rejected.

## 45.14 Baseline underfitting

IF is compared against an inferior implementation of \(\Lambda\)CDM or modified gravity.

## 45.15 Euclid anomaly ownership

Any deviation from \(\Lambda\)CDM is described as evidence for IF without testing alternatives.

---

# 46. Criteria for a Valid Prospective Test

The test is valid only if:

1. the exact IF action is frozen;
2. all physical parameters and priors are frozen;
3. the stable parameter domain is frozen;
4. numerical forecasts are generated before confirmatory access;
5. redshift and scale bins are frozen;
6. the covariance strategy is frozen;
7. nuisance models are frozen;
8. primary and secondary endpoints are frozen;
9. falsification thresholds are frozen;
10. access and amendment logs are complete;
11. the cryptographic forecast digest is publicly timestamped;
12. the scorer reproduces its decisions on synthetic mocks.

Failure of any condition produces Outcome E0 rather than scientific support.

---

# 47. What Would Count as Success?

## Level 1 — Valid preregistration

The forecast is sealed before data access.

## Level 2 — Cross-predictive compatibility

Expansion-calibrated IF predicts Euclid growth and lensing.

## Level 3 — No parameter splitting

Euclid does not require separate IF sectors.

## Level 4 — State closure

\[
b_E=b_G=b_L.
\]

## Level 5 — Distinctive prediction

A frozen IF deviation is detected with the correct sign, shape, and amplitude.

## Level 6 — Web and galaxy closure

\[
b_W=b_E
\]

and the state predicts:

\[
a_{\mathrm{IF}}(z).
\]

## Level 7 — Independent replication

A separate implementation reproduces the result.

## Level 8 — Cross-release confirmation

A later Euclid release confirms the same physical relationship.

---

# 48. What Would Count as a Major Discovery?

A successful preregistration alone is not a discovery.

A major result would be:

\[
\boxed{
\text{An IF model calibrated without Euclid correctly predicts}
\atop
\text{Euclid’s growth, weak-lensing, and gravity-consistency}
\atop
\text{observables without parameter splitting.}
}
\]

A stronger result would be:

\[
\boxed{
b_E(z)
=
b_G(z)
=
b_L(z)
=
b_W(z)
=
b_{\mathrm{gal}}(z).
}
\]

A potentially field-changing result would require Euclid to detect a deviation from \(\Lambda\)CDM whose:

- sign;
- redshift;
- scale;
- amplitude;
- cross-probe covariance

were all frozen by IF Theory before measurement.

A Nobel-class result would further require:

- independent survey confirmation;
- theoretical stability;
- successful local and early-universe tests;
- reproduction by multiple groups;
- no comparably simple conventional explanation.

---

# 49. Relationship to Falsifiability

A theory is not made scientific merely by listing possible observations.

Its survival conditions must be narrower than its reinterpretive freedom.

Paper 11 therefore imposes:

\[
\boxed{
\text{one action}
+
\text{one parameter vector}
+
\text{one forecast}
+
\text{one scoring rule}.
}
\]

A failed result may motivate a new theory.

It may not retroactively change the old prediction.

---

# 50. Relationship to IF Theory’s Larger Program

## 50.1 Papers 1–6

These papers study:

- informational capacity;
- causal work;
- emergent structures;
- domain growth;
- agency;
- memory and repair.

They do not directly determine the Euclid forecast.

## 50.2 Paper 7

Supplies the covariant IF action.

Without a completed Paper 7 model, Stage B cannot be sealed.

## 50.3 Paper 8

Supplies the galaxy acceleration and lensing tests.

## 50.4 Paper 9

Supplies the expansion–growth consistency equations and pre-Euclid posterior.

## 50.5 Paper 10

Supplies the cosmic-web state estimator.

Paper 11 is the lock joining them.

---

# 51. Honest Interpretation of Failure

A failed Euclid forecast may imply:

1. the specific IF action is wrong;
2. the selected parameter relation is wrong;
3. the claimed unification is wrong;
4. IF language is too broad to produce a useful theory;
5. an implementation or systematic assumption failed.

It does not logically prove:

- information has no physical role;
- every modified-gravity theory is false;
- dark matter has been directly detected as a particle;
- dark energy must be a cosmological constant.

The conclusion must match the tested hypothesis.

---

# 52. Honest Interpretation of Success

A successful forecast would show that:

- one IF model predicted a new dataset;
- its internal consistency survived;
- its equations were not adjusted after measurement.

It would not immediately prove:

- the microscopic IF interpretation;
- that dark-matter particles do not exist;
- that the vacuum-energy problem is solved;
- that agency or consciousness shapes cosmology;
- that the universe is a simulation.

Those claims require separate evidence.

---

# 53. Criteria for Rejection or Major Revision

The Paper 11 program should be rejected or substantially revised if:

1. no exact stable IF action is ready before Euclid;
2. the numerical forecast cannot be frozen prospectively;
3. confirmatory products are inspected before sealing;
4. growth and lensing require separate IF parameters;
5. the state-reconstruction test fails;
6. the \(E_G\) prediction has the wrong sign or scale dependence;
7. the stable IF region cannot fit Euclid;
8. the model requires added dark matter or dark energy;
9. the IF result disappears under conservative scales;
10. nuisance flexibility explains the apparent deviation;
11. a generic modified-gravity model predicts Euclid better with fewer assumptions;
12. the cosmic-web state does not match the expansion state;
13. the galaxy acceleration relation is incompatible with the Euclid-inferred state;
14. the later full release fails to reproduce the initial result;
15. amendments are made without forfeiting prospective status.

---

# 54. Conclusion

Euclid is not valuable to IF Theory because it is large.

It is valuable because it can overconstrain the theory.

The IF proposal claims that one state controls:

\[
\boxed{
\text{expansion}
+
\text{growth}
+
\text{lensing}
+
\text{cosmic-web organization}
+
\text{galactic gravity}.
}
\]

The Euclid preregistration freezes that claim as:

\[
\boxed{
b_E(z)
=
b_G(z)
=
b_L(z)
=
b_W(z)
=
b_{\mathrm{gal}}(z).
}
\]

Its parameter version is:

\[
\boxed{
\theta_E
=
\theta_G
=
\theta_L
=
\theta_W
=
\theta_{\mathrm{gal}}.
}
\]

Its central observable test is:

\[
\boxed{
E_G^{\mathrm{obs}}(k,z)
=
E_G^{\mathrm{IF}}(k,z).
}
\]

Its methodological requirement is:

\[
\boxed{
\text{freeze first, observe second, interpret third.}
}
\]

The structural rules are frozen in this paper.

The numerical forecasts must next be generated, validated on synthetic universes, cryptographically sealed, and publicly timestamped before the designated Euclid confirmatory products are accessed.

If IF Theory changes after seeing Euclid, it may begin another scientific model.

It cannot claim to have predicted Euclid.

If the frozen model survives growth, lensing, state consistency, parameter locking, independent replication, and a later release, IF Theory will have passed its first genuinely prospective cosmological test.

---

# References

1. Euclid Collaboration, Mellier, Y. et al. “Euclid. I. Overview of the Euclid Mission.” Euclid’s main cosmological probes include weak gravitational lensing and galaxy clustering. citeturn571713search3

2. European Space Agency. “Euclid.” ESA describes the mission as mapping the large-scale structure of the Universe across time by observing billions of galaxies over more than one-third of the sky. citeturn755666view2

3. European Space Agency. “DR1 Timeline.” The updated schedule describes a DR1-Foundation release in November 2026, including foundational products over approximately \(1900\,\mathrm{deg}^2\). citeturn364680search2

4. European Space Agency. “Euclid Timeline.” The updated timeline distinguishes DR1-Foundation in November 2026 and full DR1 in mid-2027. citeturn364680search4

5. European Space Agency. “Euclid Q1 Contents.” The Q1 release contains VIS and NIR data, mosaics, catalogues, photometric-redshift products, and spectroscopic products from selected fields. citeturn979100search0

6. Euclid Collaboration. “Q1 Papers.” The Q1 scientific and data-processing publications document the first quick-release products and their limitations. citeturn979100search10
