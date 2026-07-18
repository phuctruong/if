<!-- extracted from ChatGPT (driver) on 2026-07-18 -->

# Information and Topology in the Cosmic Web  
## Testing IF Statistics Beyond the Two-Point Function

**Author:** Phuc Vinh Truong  
**Series:** IF Theory Working Papers  
**Paper:** 10  
**Date:** July 18, 2026  
**Status:** Simulation, information-theoretic, and observational test protocol; no distinctive IF signal claimed

---

## Abstract

The cosmic web contains clusters, filaments, walls, tunnels, and voids generated through the nonlinear gravitational evolution of primordial fluctuations. Conventional cosmological analyses describe much of this structure through two-point statistics, especially the correlation function and power spectrum. For a Gaussian random field, the two-point function contains the complete statistical information. The late-time density field, however, is strongly non-Gaussian and contains phase correlations, morphology, topology, environmental dependence, and multiscale organization that are not generally captured by the power spectrum alone.

This paper asks whether IF Theory identifies a measurable cosmic-structure statistic that adds predictive information beyond established clustering summaries. The challenge is deliberately severe. Persistent homology, Betti curves, Minkowski functionals, density-split clustering, marked correlations, wavelet statistics, field-level inference, and neural summary statistics already probe information beyond conventional two-point measurements. Persistent topology has been used to characterize the hierarchical evolution of clusters, filaments, tunnels, and voids, while recent analyses have used it for cosmological-parameter and neutrino-mass inference. Marked statistics have been developed specifically to enhance environmentally dependent modified-gravity signals. Consequently, IF Theory cannot claim novelty merely for applying information theory or topology to the cosmic web. citeturn252655search7turn252655search16turn974973academia40turn252655search34

The paper proposes an **IF Cosmic Information Vector** rather than a single entropy scalar:

\[
\boxed{
\mathbf I_{\mathrm{IF}}
=
\left[
\mathcal I_{\mathrm{NG}},
\mathcal I_{\mathrm{top}},
\mathcal I_{\mathrm{tidal}},
\mathcal I_{\mathrm{memory}},
\mathcal I_{\mathrm{marked}},
\mathcal I_{\mathrm{cross}}
\right].
}
\]

Its components quantify:

1. non-Gaussian information relative to a covariance-matched Gaussian field;
2. persistent topological organization;
3. information in tidal geometry beyond density;
4. cross-epoch memory retained by the evolving density field;
5. environment-weighted clustering;
6. information shared across scales, tracers, and epochs.

The principal IF hypothesis is not that these quantities are nonzero. Standard nonlinear gravitational evolution already predicts that. The proposed distinctive claim is:

\[
\boxed{
\text{One low-dimensional IF state inferred from the evolution of}
\atop
\text{multiscale cosmic organization predicts expansion, growth,}
\atop
\text{lensing, and galactic acceleration more accurately than}
\atop
\text{conventional statistics with matched information capacity.}
}
\]

A statistic counts as an IF contribution only if it adds held-out predictive value after conditioning on the power spectrum, bispectrum, one-point density distribution, survey mask, tracer abundance, and baryonic nuisance parameters. The primary success measure is therefore not an aesthetically compelling topology plot but the cross-validated information increment:

\[
\boxed{
\Delta\mathcal P_{\mathrm{IF}}
=
\operatorname{ELPD}
\left[
\mathbf S_{\mathrm{base}}
+
\mathbf I_{\mathrm{IF}}
\right]
-
\operatorname{ELPD}
\left[
\mathbf S_{\mathrm{base}}
\right],
}
\]

where ELPD is expected held-out log predictive density and \(\mathbf S_{\mathrm{base}}\) contains standard cosmological summaries.

Development proceeds through analytically controlled random fields, Quijote simulations, CAMELS hydrodynamical simulations, survey mocks, SDSS/BOSS and DESI galaxy data, and future preregistered Euclid measurements. Quijote contains 44,100 full \(N\)-body simulations spanning more than 7,000 cosmological models. As of June 2026, CAMELS documents more than two petabytes from 16,960 \(N\)-body and hydrodynamical simulations, making it suitable for separating cosmological signals from uncertain galaxy-formation physics. citeturn974973academia38turn605920search2turn605920search3

The IF claim is falsified if its proposed statistics contain no stable held-out information beyond established summaries, if apparent gains disappear under survey forward modeling, if they primarily encode galaxy bias or baryonic feedback, if the result depends on arbitrary smoothing and filtration choices, or if a generic field-level model extracts the same information without the IF interpretation.

---

## Keywords

Cosmic web; information theory; persistent homology; Betti numbers; Minkowski functionals; nonlinear structure; marked correlations; density-split clustering; field-level inference; modified gravity; cosmological simulations; IF Theory.

---

# 1. Introduction

The large-scale distribution of matter is not a random collection of isolated objects.

Gravitational evolution organizes matter into a web containing:

- high-density nodes and clusters;
- elongated filaments;
- sheet-like walls;
- underdense voids;
- tunnels and connected cavities;
- nested structures across many scales.

The power spectrum describes the variance of Fourier modes:

\[
\boxed{
\left\langle
\delta(\mathbf k)
\delta^*(\mathbf k')
\right\rangle
=
(2\pi)^3
\delta_D
\left(
\mathbf k-\mathbf k'
\right)
P(k).
}
\]

For a statistically homogeneous Gaussian field, the mean and two-point covariance fully determine the distribution.

Nonlinear evolution breaks that simplicity.

Fourier phases become correlated. The density distribution becomes non-Gaussian. Filaments connect clusters. Voids merge. Halo formation becomes environmentally dependent. Fields with nearly identical power spectra can display different morphology and topology.

This motivates statistics beyond:

\[
P(k).
\]

Established approaches include:

- the bispectrum and higher-order correlation functions;
- one-point density distributions;
- counts in cells;
- peaks and minima;
- void statistics;
- density-split clustering;
- marked correlation functions;
- Minkowski functionals;
- Betti numbers;
- persistent homology;
- wavelet scattering;
- graph representations;
- learned neural summaries;
- field-level inference.

Density-split clustering has already been applied to the BOSS CMASS sample with simulation-calibrated forward models and combined with ordinary two-point clustering to constrain cosmological parameters. Persistent homology has progressed from conceptual cosmic-web descriptions to simulation-based parameter inference and applications to real weak-lensing maps. citeturn731924academia40turn731924search13turn974973search20

IF Theory therefore faces a sharp test:

\[
\boxed{
\text{What does IF add that is not already contained in the}
\atop
\text{established program of nonlinear cosmological statistics?}
}
\]

The answer cannot be:

> The cosmic web contains information.

Every measured statistic contains information.

The answer cannot be:

> Topology matters.

That is established.

The answer cannot be:

> Nonlinear structure contains more than the power spectrum.

That also is established.

The IF proposal must identify:

1. a clearly defined quantity;
2. a physical mechanism connecting it to the IF state;
3. a prediction not used to design the statistic;
4. a held-out test against strong alternatives;
5. a result robust to observation, bias, baryons, and numerical method.

---

# 2. Scientific Scope

Paper 10 studies the measurable organization of cosmic structure.

It tests whether an IF-motivated information vector:

- summarizes nonlinear organization;
- predicts cosmological parameters;
- distinguishes IF gravity from conventional gravity;
- links structure at different epochs;
- adds information beyond standard statistics;
- connects with the expansion–growth state defined in Paper 9.

It does not assume that:

- the cosmic web is alive;
- cosmic structure possesses agency;
- galaxies process information intentionally;
- information creates cosmic energy;
- entropy production causes expansion;
- topology by itself proves modified gravity;
- one scalar statistic can summarize the entire universe.

The word **information** is used in its explicit statistical senses:

- Shannon entropy;
- relative entropy;
- mutual information;
- predictive information;
- Fisher information;
- expected predictive information;
- topological persistence.

These quantities must not be treated as interchangeable.

---

# 3. The Novelty Boundary

## 3.1 Betti numbers and persistent homology

For a three-dimensional excursion set or point-cloud filtration:

- \(\beta_0\) counts connected components;
- \(\beta_1\) counts independent loops or tunnels;
- \(\beta_2\) counts enclosed cavities or void-like regions.

Persistent homology tracks the birth and death of these features as a density threshold, distance scale, or filtration parameter changes.

Previous work has developed persistent Betti-number descriptions of the cosmic web and connected persistent-diagram features to hierarchical gravitational structure formation. It has shown that clusters, filaments, tunnels, and voids leave recognizable signatures in persistence diagrams and that those diagrams retain more detailed multiscale information than a single Euler characteristic. citeturn252655search7turn252655search16turn252655academia78

IF Theory cannot claim novelty for computing:

\[
\beta_0,\quad
\beta_1,\quad
\beta_2.
\]

---

## 3.2 Minkowski functionals

In three spatial dimensions, four Minkowski functionals describe the geometry of excursion sets:

\[
V_0,\quad
V_1,\quad
V_2,\quad
V_3,
\]

corresponding broadly to:

- volume;
- surface area;
- integrated mean curvature;
- Euler characteristic.

Minkowski functionals can capture non-Gaussian morphology, but their incremental cosmological information depends on the field, modeling assumptions, included baseline statistics, noise, and nonlinear regime. A simulation study combining clustering and weak-lensing Minkowski functionals found no additional \(\Omega_m\)-\(\sigma_8\) information beyond a simplified \(3\times2\)-point analysis in its tested setup, demonstrating that “beyond two point” does not automatically mean “more useful.” Other simulation-based weak-lensing studies have found meaningful improvements from non-Gaussian summaries, including Minkowski functionals, under different setups. citeturn974973search0turn974973search8

This mixed evidence is exactly the standard Paper 10 must adopt.

---

## 3.3 Marked statistics

A marked statistic assigns each tracer a weight:

\[
m_i
=
m
\left(
\text{local environment}_i
\right).
\]

A marked correlation function is schematically:

\[
\boxed{
\mathcal M(r)
=
\frac{
1+W(r)
}{
1+\xi(r)
},
}
\]

where \(W(r)\) is the weighted pair statistic and \(\xi(r)\) is the ordinary correlation function.

Marks can amplify low-density or environmentally screened regions. Simulation studies have demonstrated sensitivity of marked correlations to modified-gravity models whose effects depend on local environment. citeturn974973academia40turn974973academia41

IF Theory cannot claim novelty merely for weighting voids or low-density galaxies more heavily.

---

## 3.4 Density-split clustering

Density-split methods divide locations or tracers by their surrounding density and measure:

- clustering within density classes;
- cross-correlations around underdense and overdense regions;
- sometimes lensing around the same classes.

These statistics capture environment-dependent information beyond the ordinary galaxy two-point function. The BOSS CMASS analysis demonstrated that such measurements can be forward modeled and used for cosmological inference while accounting for redshift-space effects, galaxy–halo connection, and assembly bias. citeturn731924academia40

---

## 3.5 Tidal geometry

The scalar density contrast:

\[
\delta
\]

does not fully specify local anisotropic deformation.

Let the tidal or deformation tensor be:

\[
T_{ij}
=
\partial_i\partial_j\Phi.
\]

Its eigenvalues:

\[
\lambda_1,\lambda_2,\lambda_3
\]

describe collapse or expansion along principal axes.

Recent information-theoretic work has proposed using the joint distribution of tidal eigenvalues, shear invariants, morphological classes, and multifractal dimensions to quantify cosmic-web information beyond density alone. This is closely related to the IF objective and sharply reduces the novelty of any generic “tidal information entropy” claim. citeturn731924academia41turn731924search0

---

## 3.6 Cross-epoch information

The evolving matter field contains statistical memory of earlier states. Recent work has explicitly studied mutual information between density fields at different redshifts as a measure of retained cosmic coherence. citeturn731924search24

IF Theory therefore cannot claim novelty for asking whether the cosmic density field remembers its past.

Its stronger task is to show that a specific cross-epoch information measure predicts an IF observable not already predictable from conventional growth statistics.

---

## 3.7 Field-level inference

Field-level methods attempt to infer cosmology directly from the spatial field rather than compressing it into a small set of summary statistics. Such methods can, in principle, retain Fourier phases and higher-order information discarded by the power spectrum. Recent work has applied field-level approaches to modified-gravity inference and to perturbative large-scale-structure modeling. citeturn731924search2turn731924search9turn731924search27

This creates the hardest comparator.

If a generic field-level model performs as well as or better than the IF statistic, then the IF summary may still be convenient but is not uniquely fundamental.

---

# 4. The Core IF Question

The core question is not whether cosmic structure contains nonlinear information.

It is:

\[
\boxed{
\text{Is there a low-dimensional, physically interpretable}
\atop
\text{information state whose evolution jointly predicts cosmic}
\atop
\text{topology, expansion, growth, lensing, and galactic gravity?}
}
\]

Let the Paper 9 state be:

\[
b(a).
\]

Paper 10 proposes that measurable nonlinear organization:

\[
\mathbf I_{\mathrm{IF}}(a)
\]

may act as an observational estimator of that state:

\[
\boxed{
\hat b_{\mathrm{web}}(a)
=
\mathcal E_{\mathrm{IF}}
\left[
\mathbf I_{\mathrm{IF}}(a)
\right].
}
\]

The unification claim requires:

\[
\boxed{
\hat b_{\mathrm{web}}(a)
=
b_{\mathrm{expansion}}(a)
=
b_{\mathrm{growth}}(a)
=
b_{\mathrm{lensing}}(a).
}
\]

If the web estimator requires an unrelated state or flexible redshift-dependent recalibration, it does not support IF unification.

---

# 5. Density Fields and Tracers

## 5.1 Matter density

Define the matter-density contrast:

\[
\boxed{
\delta_m(\mathbf x,z)
=
\frac{
\rho_m(\mathbf x,z)-\bar\rho_m(z)
}{
\bar\rho_m(z)
}.
}
\]

The true matter field is not directly observed at all scales.

---

## 5.2 Galaxy density

Observed galaxies define:

\[
\delta_g(\mathbf x,z).
\]

A bias expansion may be written schematically:

\[
\boxed{
\delta_g
=
b_1\delta_m
+
\frac{b_2}{2}\delta_m^2
+
b_{s^2}s^2
+
\epsilon
+
\cdots
}
\]

where:

- \(b_1,b_2,b_{s^2}\) are bias parameters;
- \(s^2\) is a tidal operator;
- \(\epsilon\) is stochasticity.

Topological statistics measured from galaxies are therefore not automatically the topology of total matter.

---

## 5.3 Lensing field

Weak lensing provides a projected estimate of gravitational-potential structure.

The convergence field is approximately:

\[
\boxed{
\kappa(\boldsymbol\theta)
=
\int_0^{\chi_s}
d\chi\,
W_\kappa(\chi)
\delta_{\mathrm{lens}}
\left(
\chi\boldsymbol\theta,\chi
\right).
}
\]

In modified gravity:

\[
\delta_{\mathrm{lens}}
\]

depends on the Weyl potential:

\[
\Phi+\Psi.
\]

Galaxy topology and lensing topology may differ if the IF gravitational slip differs from general relativity.

This difference is a potential discriminating signal.

---

# 6. Smoothing and Filtration

Every information or topology statistic depends on how the field is represented.

Let the smoothed field be:

\[
\boxed{
\delta_R(\mathbf x)
=
\int
d^3y\,
W_R
\left(
|\mathbf x-\mathbf y|
\right)
\delta(\mathbf y).
}
\]

The analysis must report:

- smoothing kernel;
- scale \(R\);
- grid resolution;
- tracer density;
- boundary correction;
- mask treatment;
- interpolation method.

A claim existing only at one smoothing scale is weak.

The primary analysis uses a scale family:

\[
R\in
\left[
R_{\min},R_{\max}
\right].
\]

---

## 6.1 Density excursion filtration

For threshold \(\nu\), define the superlevel set:

\[
\boxed{
X_\nu
=
\left\{
\mathbf x:
\delta_R(\mathbf x)\geq\nu
\right\}.
}
\]

As \(\nu\) decreases, isolated high-density components merge, loops form, and cavities close.

---

## 6.2 Point-cloud filtration

For discrete tracers, use an alpha, Čech, Vietoris–Rips, or distance-to-measure filtration.

The primary filtration must be chosen before the confirmatory analysis.

Alternative filtrations are robustness checks, not opportunities to select the most favorable result.

---

# 7. The IF Cosmic Information Vector

Define:

\[
\boxed{
\mathbf I_{\mathrm{IF}}
=
\left[
\mathcal I_{\mathrm{NG}},
\mathcal I_{\mathrm{top}},
\mathcal I_{\mathrm{tidal}},
\mathcal I_{\mathrm{memory}},
\mathcal I_{\mathrm{marked}},
\mathcal I_{\mathrm{cross}}
\right].
}
\]

No scalar combination is initially privileged.

Each component answers a different question.

---

# 8. Non-Gaussian Information

## 8.1 Gaussian reference

Construct a Gaussian field:

\[
\delta_G
\]

with the same:

- mean;
- power spectrum;
- survey window;
- sampling;
- noise;

as the measured or simulated field.

Let:

\[
p_R(\delta,z)
\]

be the one-point distribution of the smoothed nonlinear field and:

\[
p_{G,R}(\delta,z)
\]

the covariance-matched Gaussian reference.

Define:

\[
\boxed{
\mathcal I_{\mathrm{NG}}^{(1)}
(R,z)
=
D_{\mathrm{KL}}
\left[
p_R(\delta,z)
\parallel
p_{G,R}(\delta,z)
\right].
}
\]

This is a non-Gaussianity or negentropy-like measure.

It is not the total information content of the universe.

---

## 8.2 Field-level relative information

For a tractable field representation \(\mathbf d\), define:

\[
\boxed{
\mathcal I_{\mathrm{NG}}^{(\mathrm{field})}
=
D_{\mathrm{KL}}
\left[
p_{\mathrm{NL}}(\mathbf d)
\parallel
p_G(\mathbf d\mid C)
\right].
}
\]

Direct high-dimensional KL estimation is difficult.

The analysis will use:

- likelihood-ratio estimation;
- classifier-based density-ratio methods;
- normalizing flows validated on analytic fields;
- lower-dimensional sufficient summaries.

Every estimator must be tested for finite-sample bias.

---

## 8.3 Conditional non-Gaussian information

To isolate information not captured by baseline summaries:

\[
\boxed{
\mathcal I_{\mathrm{NG}\mid\mathrm{base}}
=
I
\left(
\Theta;
\mathbf S_{\mathrm{NG}}
\mid
P,B,\mathrm{PDF}
\right),
}
\]

where:

- \(\Theta\) is the cosmological or gravity model;
- \(P\) is the power spectrum;
- \(B\) is the bispectrum;
- PDF is the one-point density distribution;
- \(\mathbf S_{\mathrm{NG}}\) is the candidate IF statistic.

This conditional information is more relevant than the raw entropy value.

---

# 9. Persistent Topological Information

## 9.1 Persistence diagrams

For homology dimension \(q\), define a persistence diagram:

\[
\boxed{
\mathcal D_q
=
\left\{
(b_i,d_i)
\right\},
}
\]

where:

- \(b_i\) is the filtration value at feature birth;
- \(d_i\) is the value at feature death;
- persistence is:

\[
\pi_i
=
|d_i-b_i|.
\]

---

## 9.2 Betti curves

At filtration parameter \(\nu\):

\[
\boxed{
\beta_q(\nu)
=
\#\left\{
i:
b_i\leq\nu<d_i
\right\}.
}
\]

In three dimensions:

- \(\beta_0\): connected components;
- \(\beta_1\): loops and tunnels;
- \(\beta_2\): cavities.

---

## 9.3 Persistence entropy

A normalized persistence weight is:

\[
p_i
=
\frac{
\pi_i
}{
\sum_j\pi_j
}.
\]

Define:

\[
\boxed{
S_{\mathrm{pers},q}
=
-\sum_i p_i\ln p_i.
}
\]

Persistence entropy compresses the diagram but loses location information.

It must be reported alongside the full diagram or richer diagram embeddings.

---

## 9.4 Total persistence

\[
\boxed{
\operatorname{TP}_{q,p}
=
\sum_i
\pi_i^p.
}
\]

Different values of \(p\) emphasize short- or long-lived features differently.

No value of \(p\) may be selected after inspecting model separation.

---

## 9.5 IF topological vector

Define:

\[
\boxed{
\mathcal I_{\mathrm{top}}
=
\left[
\beta_q(\nu),
S_{\mathrm{pers},q},
\operatorname{TP}_{q,p},
\text{diagram embedding}
\right]_{q=0,1,2}.
}
\]

The IF state estimator may use this vector, but the analysis must determine whether the full topology adds information beyond:

- power spectrum;
- bispectrum;
- one-point PDF;
- ordinary void catalog;
- Minkowski functionals.

---

# 10. Minkowski Geometry

For excursion set \(X_\nu\), define in three dimensions:

\[
\boxed{
\mathbf V(\nu)
=
\left[
V_0(\nu),
V_1(\nu),
V_2(\nu),
V_3(\nu)
\right].
}
\]

Interpretations are approximately:

\[
V_0:
\text{volume fraction},
\]

\[
V_1:
\text{surface area},
\]

\[
V_2:
\text{integrated mean curvature},
\]

\[
V_3:
\text{Euler characteristic}.
\]

The Euler characteristic relates to Betti numbers:

\[
\boxed{
\chi
=
\beta_0-\beta_1+\beta_2.
}
\]

Because many different Betti triplets can share the same \(\chi\), persistent homology can retain topology discarded by the Euler characteristic alone.

---

# 11. Tidal Information

Let the tidal tensor eigenvalues be ordered:

\[
\lambda_1\geq\lambda_2\geq\lambda_3.
\]

Define invariants:

\[
I_1
=
\lambda_1+\lambda_2+\lambda_3,
\]

\[
I_2
=
\lambda_1\lambda_2+
\lambda_1\lambda_3+
\lambda_2\lambda_3,
\]

\[
I_3
=
\lambda_1\lambda_2\lambda_3.
\]

Because:

\[
I_1\propto\delta,
\]

the remaining invariants contain anisotropic information not determined by density alone.

Define:

\[
\boxed{
\mathcal I_{\mathrm{tidal}}
=
I
\left[
(I_2,I_3);
\Theta
\mid
I_1,P(k)
\right].
}
\]

This asks how much cosmological information tidal anisotropy adds after density and two-point clustering are known.

---

## 11.1 Morphological classes

For threshold \(\lambda_{\mathrm{th}}\), classify a location by the number of eigenvalues above threshold.

Possible classes are:

- void;
- wall;
- filament;
- node.

The exact sign convention and threshold are not universal.

Every result must report:

\[
\lambda_{\mathrm{th}}.
\]

The primary analysis uses continuous eigenvalue distributions rather than relying only on discrete labels.

---

# 12. Cross-Epoch Memory

## 12.1 Lagrangian correspondence

When simulation particles or reconstructed initial conditions provide correspondence, define:

\[
\delta_R(\mathbf q,z_1),
\qquad
\delta_R(\mathbf q,z_2)
\]

for common Lagrangian location \(\mathbf q\).

Define:

\[
\boxed{
\mathcal I_{\mathrm{memory}}
(R;z_1,z_2)
=
I
\left[
\delta_R(\mathbf q,z_1);
\delta_R(\mathbf q,z_2)
\right].
}
\]

---

## 12.2 Random baseline subtraction

Finite estimators can produce positive mutual information even for independent fields.

Define:

\[
\boxed{
\mathcal I_{\mathrm{memory}}^{\mathrm{corr}}
=
\hat I_{\mathrm{true}}
-
\mathbb E
\left[
\hat I_{\mathrm{shuffled}}
\right].
}
\]

Uncertainty is estimated through:

- repeated shuffles;
- simulation realizations;
- spatial block resampling.

---

## 12.3 Conditional memory

To isolate memory beyond linear growth:

\[
\boxed{
\mathcal I_{\mathrm{NL\ memory}}
=
I
\left[
\delta(z_1);
\delta(z_2)
\mid
\delta_{\mathrm{linear}}(z_1),
D(z_2)/D(z_1)
\right].
}
\]

This is a difficult estimator and begins in simulations only.

---

## 12.4 IF hypothesis

The IF proposal predicts that:

\[
\mathcal I_{\mathrm{memory}}
\]

should follow the same state history:

\[
b(a)
\]

that controls Paper 9 expansion and growth.

The restricted form is:

\[
\boxed{
\frac{
d\mathcal I_{\mathrm{memory}}
}{
d\ln a
}
=
F_M
\left[
b(a);\theta
\right].
}
\]

The function must be derived or frozen using simulations before observational application.

---

# 13. Cross-Scale Information

For smoothed fields at scales:

\[
R_1<R_2,
\]

define:

\[
\boxed{
\mathcal I_{\mathrm{cross}}
(R_1,R_2;z)
=
I
\left[
\delta_{R_1};
\delta_{R_2}
\right].
}
\]

Nonlinear gravitational collapse transfers and couples structure across scales.

The conditional cross-scale information is:

\[
\boxed{
\mathcal I_{\mathrm{cross}\mid P}
=
I
\left[
\delta_{R_1};
\delta_{R_2}
\mid
P(k)
\right].
}
\]

This notation means conditioning within the simulation ensemble on matched or near-matched power-spectrum summaries. It does not mean that one single universe has a fixed known \(P(k)\) at every realization.

---

# 14. Marked Environmental Information

Define a smoothed local density:

\[
\delta_R(\mathbf x_i).
\]

A bounded mark may be:

\[
\boxed{
m_i
=
\left[
\frac{
1+\delta_*
}{
1+\delta_*+\delta_R(\mathbf x_i)
}
\right]^p.
}
\]

Positive \(p\) upweights underdense environments.

Alternative marks may use:

- tidal anisotropy;
- filament distance;
- void membership;
- local potential;
- environmental history;
- predicted screening state.

The primary mark family is frozen in simulations.

---

## 14.1 IF-mark hypothesis

If IF gravity differs most in low-density, weakly screened, or rapidly relaxing environments, then an IF-derived mark should increase model separation.

The primary question is:

\[
\boxed{
\Delta\mathcal P_{\mathrm{mark}}
=
\operatorname{ELPD}
\left[
P(k)+\mathcal M_{\mathrm{IF}}
\right]
-
\operatorname{ELPD}
\left[
P(k)
\right].
}
\]

The mark succeeds only if it improves held-out inference after shot-noise correction and galaxy-bias marginalization.

---

# 15. The IF Web-State Estimator

Let:

\[
\mathbf s(z)
\]

be a finite summary vector containing selected components of:

\[
\mathbf I_{\mathrm{IF}}.
\]

Define an estimator:

\[
\boxed{
\hat b_{\mathrm{web}}(z)
=
\mathcal E_\omega
\left[
\mathbf s(z)
\right].
}
\]

The estimator may be:

- linear;
- Gaussian-process based;
- neural;
- likelihood-ratio based;
- simulation-based Bayesian inference.

The simplest model is preferred.

---

## 15.1 Training target

The estimator is trained in simulations where:

\[
b(z)
\]

is known from the input IF cosmology.

For conventional cosmologies without an IF field, define an effective comparison state only if it is mathematically explicit.

No arbitrary labels such as “high information” and “low information” are permitted.

---

## 15.2 State consistency

After training:

\[
\boxed{
\hat b_{\mathrm{web}}(z)
\stackrel{?}{=}
b_{\mathrm{expansion}}(z).
}
\]

Define:

\[
\Delta b(z)
=
\hat b_{\mathrm{web}}(z)
-
b_{\mathrm{expansion}}(z).
\]

A covariance-weighted statistic is:

\[
\boxed{
Q_{\mathrm{web}}
=
\Delta\mathbf b^\top
C_b^{-1}
\Delta\mathbf b.
}
\]

The threshold is calibrated on synthetic universes.

---

# 16. Primary Baseline Summary Set

The base summary vector is:

\[
\boxed{
\mathbf S_{\mathrm{base}}
=
\left[
P_0(k),
P_2(k),
P_4(k),
B(k_1,k_2,k_3),
p(\delta_R),
n_g
\right],
}
\]

where:

- \(P_\ell\) are redshift-space power-spectrum multipoles;
- \(B\) is a bispectrum summary;
- \(p(\delta_R)\) is the one-point density PDF;
- \(n_g\) is tracer abundance.

A reduced baseline may omit the bispectrum during early development.

The final claim must compare against a strong baseline containing at least:

- two-point clustering;
- a leading higher-order statistic;
- tracer density;
- survey nuisance information.

---

# 17. Primary Scientific Endpoint

The primary endpoint is held-out predictive gain:

\[
\boxed{
\Delta\mathcal P_{\mathrm{IF}}
=
\operatorname{ELPD}
\left[
\Theta
\mid
\mathbf S_{\mathrm{base}}
+
\mathbf I_{\mathrm{IF}}
\right]
-
\operatorname{ELPD}
\left[
\Theta
\mid
\mathbf S_{\mathrm{base}}
\right].
}
\]

Possible prediction targets \(\Theta\) include:

- \(\Omega_m\);
- \(\sigma_8\);
- neutrino mass;
- modified-gravity strength;
- IF transition parameters;
- \(b(z)\);
- Paper 9 expansion–growth residual;
- model identity.

The primary target is fixed before the held-out test.

---

## 17.1 Practical significance

Statistical significance alone is insufficient.

A successful statistic must produce at least one of:

- materially tighter held-out parameter inference;
- reduced parameter degeneracy;
- improved model discrimination;
- improved calibration;
- an accurate cross-epoch prediction;
- a lower-dimensional sufficient representation.

The numerical threshold for material improvement is calibrated before observational unblinding.

---

# 18. Null Universes

## 18.1 Gaussian random field

Generate a Gaussian field with the same:

\[
P(k)
\]

as the target field.

Expected result:

- no genuinely nonlinear topology beyond Gaussian expectation;
- estimator biases quantified.

---

## 18.2 Phase-randomized field

Preserve Fourier amplitudes and randomize phases.

The power spectrum remains fixed.

Topology and morphology change.

This is the most direct baseline for information beyond the two-point function.

---

## 18.3 Lognormal field

Generate a lognormal field with matched:

- power spectrum;
- approximate one-point PDF.

This tests whether IF statistics detect more than simple non-Gaussian density skewness.

---

## 18.4 Bispectrum-matched field

Where feasible, construct fields approximately matching:

- power spectrum;
- bispectrum;
- one-point PDF.

Any remaining discriminating information lies at higher order or in spatial organization not captured by those summaries.

---

## 18.5 Shuffled environment

Preserve tracer positions but permute marks or environmental labels.

This establishes the null marked-correlation distribution.

---

## 18.6 Wrong-mask universe

Apply deliberately incorrect masks and selection functions.

The analysis must identify the resulting bias rather than interpret it as topology.

---

## 18.7 Standard-gravity universe

Use \(\Lambda\)CDM simulations as the null physical model.

IF statistics must not produce a false IF state.

---

# 19. Simulation Program

## 19.1 Analytic random fields

Begin with:

- Gaussian random fields;
- lognormal fields;
- controlled non-Gaussian fields;
- Voronoi-web models;
- fractal and multifractal point processes.

These provide known or controllable morphology.

---

## 19.2 Quijote

Quijote contains 44,100 \(N\)-body simulations spanning more than 7,000 cosmological models and was designed to quantify the information content of cosmological observables and support machine-learning applications. It includes snapshots, halo and void catalogs, and numerous summary statistics. citeturn974973academia38turn605920search32

Quijote will be used for:

- cosmological parameter sensitivity;
- covariance estimation;
- phase-matched controls;
- neutrino tests;
- Fisher comparisons;
- summary-statistic selection.

---

## 19.3 CAMELS

CAMELS combines \(N\)-body and hydrodynamical simulations across cosmological and astrophysical parameters. Its documentation reports 16,960 simulations and more than two petabytes of data as of June 2026, including thousands of matched \(N\)-body and hydrodynamic runs and multifield maps and grids. citeturn605920search2turn605920search3turn605920search20

CAMELS will be used to test whether an apparent IF statistic is actually driven by:

- stellar feedback;
- active-galactic-nucleus feedback;
- gas physics;
- star formation;
- tracer selection.

---

## 19.4 IF simulations

Paper 7 and Paper 9 must eventually provide dedicated IF structure-formation simulations.

Without them, Paper 10 can test generic information statistics but cannot validate the distinctive IF prediction.

The IF simulation suite must vary:

- shared IF parameters;
- initial conditions;
- baryonic physics;
- numerical resolution;
- box size.

---

# 20. Observational Data Ladder

## 20.1 BOSS and legacy SDSS

BOSS galaxy catalogs provide an established testbed with mature masks, mocks, covariance products, and clustering analyses.

Density-split clustering and other beyond-two-point methods have already been demonstrated on BOSS data. citeturn731924academia40

---

## 20.2 SDSS DR19

SDSS DR19 is the current public SDSS release and was released in July 2025. It is the first major public release containing SDSS-V spectroscopic products, although not every DR19 component is optimized for the same large-scale cosmology analysis as BOSS. citeturn252655search0turn252655search2turn252655search29

DR19 is useful for:

- pipeline development;
- tracer studies;
- cross-checks;
- future SDSS-V large-scale-structure products.

---

## 20.3 DESI DR1

DESI DR1 was released in March 2025 and contains 18.7 million new main-survey spectra from its first year. DESI’s published first-year full-shape analysis used millions of galaxy and quasar redshifts across multiple tracer and redshift bins. citeturn252655search12turn731924search30

DESI DR1 provides the principal current spectroscopic target for a large-volume observed cosmic-web analysis.

---

## 20.4 DESI DR2 products

Public DR2 products currently include cosmological chains and best-fit products associated with the three-year BAO analysis. Those products are valuable for Paper 9 but do not by themselves provide the complete raw three-year galaxy field required for every Paper 10 topology analysis. citeturn252655search3turn252655search25

The analysis must not claim access to unreleased object-level data.

---

## 20.5 Euclid

Euclid released Quick Data Release 2 on June 24, 2026. The next broader DR1 release is scheduled for late 2026. The relevant wide-area cosmological products remain a prospective test for this paper. citeturn605920search0turn605920search1turn605920search12

The IF topology prediction must be frozen before using those decisive products.

---

# 21. Survey Forward Modeling

A measured cosmic-web statistic depends on more than cosmology.

The mock pipeline must include:

- survey geometry;
- angular mask;
- radial selection;
- redshift failure;
- fiber assignment;
- completeness;
- redshift uncertainty;
- stellar contamination;
- tracer bias;
- halo occupation;
- assembly bias;
- peculiar velocities;
- Alcock–Paczyński distortion;
- reconstruction choices;
- shot noise.

The primary comparison occurs in observed space.

A statistic measured on a periodic simulation cube cannot be compared directly with a masked survey catalog.

---

# 22. Boundary and Mask Treatment

Topology is particularly sensitive to boundaries.

A mask can create:

- artificial disconnected components;
- false tunnels;
- false cavities;
- truncated filaments;
- altered persistence.

Required controls include:

1. periodic-box benchmark;
2. survey-mask injection;
3. random-catalog correction;
4. buffer-zone analysis;
5. relative homology where appropriate;
6. masked-field simulations;
7. null tests under rotated or displaced masks.

The mask pipeline is frozen before examining model differences.

---

# 23. Shot Noise and Sparse Sampling

A discrete tracer sample introduces Poisson and non-Poisson sampling effects.

The observed point set is:

\[
\left\{
\mathbf x_i
\right\}_{i=1}^{N_g}.
\]

Topology can change substantially as:

\[
n_g
\]

changes.

The analysis must compare catalogs at matched:

- number density;
- redshift distribution;
- bias;
- survey geometry.

Downsampling tests determine whether model differences survive equal tracer density.

Marked-correlation work has shown that discreteness corrections can be essential for unbiased environmental statistics. citeturn974973academia41

---

# 24. Redshift-Space Distortions

Observed radial positions include peculiar velocities:

\[
\boxed{
\mathbf s
=
\mathbf x
+
\frac{
\mathbf v\cdot\hat{\mathbf n}
}{
aH
}
\hat{\mathbf n}.
}
\]

Redshift-space distortions:

- compress structures on large scales;
- elongate virialized clusters;
- alter filament connectivity;
- distort void shapes;
- change persistence diagrams.

The model must predict topology in redshift space rather than correcting the data to an assumed real-space cosmology and then testing that cosmology.

---

# 25. Galaxy Bias and Assembly Bias

A gravity or IF signal can be imitated by changes in how galaxies occupy halos and environments.

The mock program will vary:

- halo occupation;
- stellar-to-halo relations;
- satellite fraction;
- velocity bias;
- concentration dependence;
- environmental assembly bias.

A statistic is useful only if it separates:

\[
\text{gravity or cosmology}
\]

from:

\[
\text{galaxy formation and selection}.
\]

Density-split analyses of BOSS have found evidence that environment-dependent assembly bias can matter in the galaxy–halo model, reinforcing this requirement. citeturn731924academia40

---

# 26. Baryonic Physics

Hydrodynamical feedback changes the small-scale density and lensing fields.

For each statistic, define a baryonic sensitivity:

\[
\boxed{
\Delta_B\mathbf I
=
\mathbf I_{\mathrm{hydro}}
-
\mathbf I_{N\text{-body}}.
}
\]

Define IF or gravity sensitivity:

\[
\boxed{
\Delta_{\mathrm{IF}}\mathbf I
=
\mathbf I_{\mathrm{IF}}
-
\mathbf I_{\mathrm{GR}}.
}
\]

A useful scale requires:

\[
\left|
\Delta_{\mathrm{IF}}\mathbf I
\right|
\gtrsim
\left|
\Delta_B\mathbf I
\right|
\]

or sufficiently accurate baryonic marginalization.

The primary analysis uses conservative scales where baryonic uncertainty is controlled.

---

# 27. Information Sufficiency and Complementarity

A new summary statistic can be:

- sufficient;
- complementary;
- redundant;
- unstable.

## 27.1 Redundancy

If:

\[
I
\left(
\Theta;
\mathbf I_{\mathrm{IF}}
\mid
\mathbf S_{\mathrm{base}}
\right)
\approx0,
\]

then the IF statistic adds no parameter information.

---

## 27.2 Complementarity

If:

\[
I
\left(
\Theta;
\mathbf I_{\mathrm{IF}}
\mid
\mathbf S_{\mathrm{base}}
\right)
>0,
\]

it adds information beyond the baseline.

---

## 27.3 Sufficiency

If:

\[
P
\left(
\Theta\mid
\mathbf I_{\mathrm{IF}}
\right)
\approx
P
\left(
\Theta\mid
\text{full field}
\right),
\]

then the statistic is approximately sufficient for \(\Theta\) within the tested model class.

The full field-level posterior is the benchmark where computationally feasible.

---

# 28. Model Comparison

Compare at least:

1. power spectrum only;
2. power spectrum plus bispectrum;
3. power spectrum plus density PDF;
4. power spectrum plus Minkowski functionals;
5. power spectrum plus persistent homology;
6. power spectrum plus marked statistics;
7. combined IF vector;
8. learned neural summary;
9. field-level inference.

The IF vector is successful only if it provides a favorable balance of:

- predictive information;
- interpretability;
- robustness;
- computational cost;
- transfer across surveys.

---

# 29. Core Hypotheses

## CT-H1 — Beyond-two-point hypothesis

At fixed power spectrum, IF topology and information summaries distinguish nonlinear structure histories.

### Falsifier

Phase-randomized or alternative fields with matched two-point statistics are indistinguishable under the IF vector.

---

## CT-H2 — Beyond-bispectrum hypothesis

At fixed power spectrum, bispectrum, and one-point PDF, at least one IF component retains held-out cosmological or gravity information.

### Falsifier

All IF information is captured by those established summaries.

---

## CT-H3 — Tidal-information hypothesis

Tidal invariants provide predictive information beyond density and power-spectrum information.

### Falsifier

After conditioning on density and standard clustering, tidal information adds no stable prediction.

---

## CT-H4 — Cross-epoch-memory hypothesis

The cosmic field retains measurable nonlinear cross-epoch information not reducible to the linear growth factor.

### Falsifier

Cross-epoch mutual information is fully explained by linear evolution and estimator bias.

---

## CT-H5 — IF-state hypothesis

One web-derived state:

\[
\hat b_{\mathrm{web}}(z)
\]

matches the state inferred independently from expansion and growth.

### Falsifier

Web, expansion, and growth require incompatible state histories.

---

## CT-H6 — Modified-gravity sensitivity hypothesis

Environment- and topology-sensitive IF statistics distinguish IF gravity from general relativity after matching the power spectrum and tracer population.

### Falsifier

Differences vanish after matching bias, abundance, and two-point clustering.

---

## CT-H7 — Baryonic robustness hypothesis

The IF signal survives hydrodynamical feedback marginalization on preregistered scales.

### Falsifier

Baryonic uncertainty is larger than or degenerate with the proposed signal.

---

## CT-H8 — Survey robustness hypothesis

The information gain survives realistic masks, selection, redshift-space distortions, and shot noise.

### Falsifier

The simulation signal disappears in survey-space mocks.

---

## CT-H9 — Cross-survey hypothesis

A statistic calibrated on one survey or tracer predicts another without full retraining.

### Falsifier

Every sample requires an unrelated empirical calibration.

---

## CT-H10 — Predictive-compression hypothesis

A low-dimensional IF vector approaches the predictive performance of field-level inference for selected IF parameters.

### Falsifier

The vector loses most relevant information or requires dimension comparable to the field itself.

---

## CT-H11 — Prospective Euclid hypothesis

A frozen IF topology or information forecast predicts future Euclid cosmology products.

### Falsifier

The preregistered forecast fails.

---

# 30. Prediction Hierarchy

## Prediction T1 — Matched-power separation

Fields generated under IF and standard gravity with matched:

\[
P(k)
\]

have different persistent-diagram and tidal-information distributions.

---

## Prediction T2 — Environment-enhanced separation

The largest IF–GR separation occurs in underdense or weakly screened regions selected before measurement.

---

## Prediction T3 — Cross-epoch coherence

The IF model predicts a distinct redshift evolution of:

\[
\mathcal I_{\mathrm{memory}}
\]

at fixed linear growth.

---

## Prediction T4 — Web–expansion consistency

The topology-inferred:

\[
\hat b_{\mathrm{web}}(z)
\]

matches Paper 9’s:

\[
b_{\mathrm{expansion}}(z).
\]

---

## Prediction T5 — Web–lensing consistency

Galaxy topology and lensing topology are linked through Paper 7’s gravitational-slip prediction.

---

## Prediction T6 — Galaxy-scale link

The cosmic-web state predicts Paper 8’s:

\[
a_{\mathrm{IF}}(z).
\]

Thus:

\[
\boxed{
a_{\mathrm{IF}}(z)
=
a_{\mathrm{IF},0}
F_a
\left[
\hat b_{\mathrm{web}}(z)
\right].
}
\]

---

# 31. Preregistered Analysis Sequence

## Stage 0 — Mathematical validation

Validate every statistic on fields with known topology and information.

## Stage 1 — Gaussian and lognormal nulls

Measure estimator bias and false detections.

## Stage 2 — Standard \(\Lambda\)CDM simulations

Reproduce established topology evolution.

## Stage 3 — Standard beyond-two-point benchmarks

Reproduce Minkowski, marked, density-split, and persistence baselines.

## Stage 4 — Hydrodynamical robustness

Use CAMELS to quantify baryonic sensitivity.

## Stage 5 — IF simulation forecast

Freeze the IF–GR difference before observational analysis.

## Stage 6 — Survey-mock validation

Inject masks, selection, bias, and redshift-space effects.

## Stage 7 — BOSS/SDSS validation

Test mature lower-volume catalogs.

## Stage 8 — DESI DR1 confirmatory test

Apply the frozen pipeline to the larger current public spectroscopic sample.

## Stage 9 — Cross-probe test

Compare galaxy and weak-lensing topology.

## Stage 10 — Euclid preregistration

Freeze the future forecast before DR1 cosmology products are used.

---

# 32. Statistical Standards

## 32.1 Simulation independence

Subvolumes from one simulation are not fully independent universes.

Covariance estimation must account for:

- shared initial phases;
- shared long-wavelength modes;
- repeated subvolume extraction;
- simulation-family dependence.

---

## 32.2 Paired-phase simulations

Paired initial conditions are valuable for reducing cosmic variance when comparing gravity models.

The analysis must distinguish variance reduction from independent evidence.

---

## 32.3 Hyperparameter separation

Smoothing scale, persistence threshold, mark parameters, and diagram embeddings are hyperparameters.

They are selected on training simulations only.

---

## 32.4 Covariance uncertainty

Covariance matrices estimated from finite simulations introduce uncertainty.

Use:

- analytic corrections where valid;
- shrinkage;
- simulation-based likelihoods;
- covariance marginalization;
- held-out calibration.

---

## 32.5 Coverage

Posterior credible intervals must achieve correct empirical coverage in synthetic universes.

A tight but miscalibrated posterior is a failure.

---

## 32.6 Multiple statistics

Testing many topological summaries creates a large look-elsewhere effect.

The primary statistic and endpoint are preregistered.

All explored summaries are reported.

---

# 33. Deterministic Jupyter-Notebook Program

## Notebook 10A — Gaussian Random-Field Laboratory

Generate Gaussian fields with known:

\[
P(k).
\]

Validate:

- density PDF;
- covariance;
- Minkowski functionals;
- Betti curves;
- estimator bias.

---

## Notebook 10B — Phase-Randomized Controls

Preserve Fourier amplitudes and randomize phases.

Quantify which statistics detect the lost spatial organization.

---

## Notebook 10C — Lognormal and Controlled Non-Gaussian Fields

Construct fields with matched power spectra and adjustable skewness, kurtosis, and phase coupling.

---

## Notebook 10D — Persistence Pipeline Validation

Implement:

- cubical complexes;
- alpha complexes;
- persistence diagrams;
- Betti curves;
- persistence landscapes;
- persistence images.

Validate on synthetic objects with known topology.

---

## Notebook 10E — Minkowski Functional Validation

Compute:

\[
V_0,V_1,V_2,V_3.
\]

Check analytic Gaussian-field expectations where applicable.

---

## Notebook 10F — Tidal Tensor and Web Classes

Calculate:

\[
T_{ij},
\quad
\lambda_i,
\quad
I_1,I_2,I_3.
\]

Validate rotational invariance and numerical derivatives.

---

## Notebook 10G — Mutual-Information Estimator Audit

Compare:

- histogram estimators;
- kernel estimators;
- nearest-neighbor estimators;
- neural estimators;
- density-ratio methods.

Test bias on known distributions.

---

## Notebook 10H — Cross-Epoch Memory

Measure:

\[
\mathcal I_{\mathrm{memory}}(R;z_1,z_2)
\]

in \(N\)-body trajectories.

Apply shuffled and linear-growth controls.

---

## Notebook 10I — Marked Statistics

Implement density, tidal, filament, and IF-predicted marks.

Audit shot-noise sensitivity.

---

## Notebook 10J — Density-Split Baseline

Reproduce a standard density-split clustering pipeline before introducing IF summaries.

---

## Notebook 10K — Quijote Data Manifest

Download or register:

- snapshots;
- halo catalogs;
- void catalogs;
- parameter tables;
- initial seeds.

Create checksums and data lineage.

---

## Notebook 10L — Quijote Parameter Sensitivity

Estimate derivatives of each statistic with respect to:

\[
\Omega_m,
\Omega_b,
h,
n_s,
\sigma_8,
M_\nu,
w.
\]

---

## Notebook 10M — Fisher Information Benchmark

Compare the information matrices of:

- power spectrum;
- bispectrum;
- topology;
- combined summaries.

Use stability-corrected numerical derivatives.

---

## Notebook 10N — CAMELS Hydrodynamic Audit

Measure statistic variation across:

- cosmology;
- supernova feedback;
- active-galactic-nucleus feedback;
- simulation code.

---

## Notebook 10O — Baryonic-Safe Scale Selection

Freeze the scales retained for observed analysis.

---

## Notebook 10P — IF versus GR Simulations

Measure the preregistered:

\[
\Delta_{\mathrm{IF}}\mathbf I.
\]

Match:

- initial phases;
- tracer density;
- power spectrum where possible;
- halo occupation.

---

## Notebook 10Q — Survey Mask Laboratory

Inject SDSS, BOSS, DESI, and Euclid-like windows into simulation mocks.

---

## Notebook 10R — Redshift-Space Topology

Compare real- and redshift-space statistics.

Build the forward model.

---

## Notebook 10S — Galaxy-Bias Stress Test

Vary halo occupation and assembly bias.

Determine which summaries remain gravity-sensitive.

---

## Notebook 10T — BOSS Pipeline Reproduction

Reproduce published clustering and density-split baseline results.

---

## Notebook 10U — DESI DR1 Manifest

Register public catalogs, randoms, masks, completeness products, and mock resources.

---

## Notebook 10V — DESI DR1 Cosmic-Web Measurement

Measure the frozen IF vector in redshift bins and tracer classes.

---

## Notebook 10W — Weak-Lensing Topology

Measure corresponding summaries in convergence or mass maps.

---

## Notebook 10X — Galaxy–Lensing Cross-Topology

Test whether galaxy and Weyl-potential structures satisfy the IF gravitational-slip relation.

---

## Notebook 10Y — IF Web-State Estimator

Infer:

\[
\hat b_{\mathrm{web}}(z).
\]

Validate calibration and coverage.

---

## Notebook 10Z — Web–Expansion–Growth Consistency

Compare:

\[
\hat b_{\mathrm{web}},
\quad
b_{\mathrm{expansion}},
\quad
b_{\mathrm{growth}},
\quad
b_{\mathrm{lensing}}.
\]

---

## Notebook 10AA — Summary Sufficiency

Compare IF summaries against field-level inference.

---

## Notebook 10AB — Held-Out Predictive Gain

Calculate:

\[
\Delta\mathcal P_{\mathrm{IF}}.
\]

---

## Notebook 10AC — Euclid Frozen Forecast

Save:

- redshift bins;
- smoothing scales;
- topology statistic;
- covariance forecast;
- nuisance model;
- pass threshold;
- failure threshold;
- code and environment hashes.

---

## Notebook 10AD — Adversarial Audit

A separate agent attempts to explain every positive result through:

- power-spectrum leakage;
- bispectrum leakage;
- tracer abundance;
- smoothing choice;
- mask topology;
- shot noise;
- galaxy bias;
- baryonic feedback;
- covariance error;
- simulation overfitting;
- neural-estimator bias;
- post hoc statistic choice.

---

# 34. Computational Architecture

```text
if_cosmic_web/
├── fields/
│   ├── grids.py
│   ├── smoothing.py
│   ├── fft.py
│   ├── tidal.py
│   └── redshift_space.py
├── topology/
│   ├── cubical.py
│   ├── alpha.py
│   ├── betti.py
│   ├── persistence.py
│   └── embeddings.py
├── geometry/
│   ├── minkowski.py
│   ├── curvature.py
│   ├── skeletons.py
│   └── voids.py
├── information/
│   ├── entropy.py
│   ├── relative_entropy.py
│   ├── mutual_information.py
│   ├── cross_epoch.py
│   └── estimator_validation.py
├── clustering/
│   ├── power.py
│   ├── bispectrum.py
│   ├── marked.py
│   └── density_split.py
├── simulations/
│   ├── gaussian/
│   ├── quijote/
│   ├── camels/
│   ├── if_gravity/
│   └── mocks/
├── surveys/
│   ├── boss/
│   ├── sdss/
│   ├── desi/
│   ├── lensing/
│   └── euclid/
├── inference/
│   ├── baseline.py
│   ├── if_vector.py
│   ├── field_level.py
│   ├── state_estimator.py
│   └── cross_prediction.py
├── validation/
│   ├── coverage.py
│   ├── covariance.py
│   ├── masking.py
│   ├── bias.py
│   └── baryons.py
└── tests/
```

---

# 35. Reproducibility Record

Each result emits:

```yaml
experiment_id: if-cosmic-web-10
paper_version: null
git_commit: null
environment_hash: null
data_manifest_hash: null

simulation_suite: null
simulation_id: null
cosmology_parameters: {}
gravity_model: null
astrophysical_parameters: {}
initial_seed: null

tracer_type: null
redshift: null
number_density: null
survey_mask_hash: null
selection_function_hash: null
redshift_space: true

grid_resolution: null
smoothing_kernel: null
smoothing_scale: null
filtration_type: null
filtration_parameters: {}
topology_library_version: null

power_spectrum_hash: null
bispectrum_hash: null
density_pdf_hash: null
minkowski_hash: null
betti_curve_hash: null
persistence_diagram_hash: null
tidal_information_hash: null
cross_epoch_information_hash: null
marked_statistic_hash: null

baseline_predictive_score: null
if_predictive_score: null
incremental_predictive_gain: null
field_level_predictive_score: null

web_state_history_hash: null
expansion_state_history_hash: null
growth_state_history_hash: null
state_consistency_statistic: null

estimator_bias: null
coverage_result: null
covariance_condition_number: null
baryonic_sensitivity: null
bias_sensitivity: null
mask_sensitivity: null

invariant_failures: []
result_hash: null
```

---

# 36. Failure Modes

## 36.1 Entropy without an ensemble

A Shannon or KL quantity is calculated from one field without defining the probability distribution or sampling ensemble.

## 36.2 Bits treated as energy

An information statistic is inserted into a gravitational equation without a physical conversion law or dimensional coefficient.

## 36.3 Power-spectrum leakage

A neural or topological statistic succeeds because the fields were not actually matched in:

\[
P(k).
\]

## 36.4 Tracer-abundance leakage

Model classes have different galaxy counts, making classification trivial.

## 36.5 Bias leakage

The statistic measures the galaxy–halo prescription rather than gravity.

## 36.6 Baryonic leakage

Hydrodynamic feedback generates the apparent IF signature.

## 36.7 Mask topology

Survey holes are counted as cosmic voids or tunnels.

## 36.8 Smoothing hindsight

The smoothing scale is selected after examining model separation.

## 36.9 Filtration hindsight

The topology method is selected because it produces the desired result.

## 36.10 Phase-randomization failure

The statistic does not distinguish the real field from a phase-randomized field with the same power spectrum.

## 36.11 Mutual-information estimator bias

A high-dimensional neural estimator reports false nonzero information.

## 36.12 Simulation memorization

The classifier recognizes simulation code or numerical artifacts.

## 36.13 Covariance underestimation

Subvolumes of one simulation are treated as independent universes.

## 36.14 Baseline straw man

The IF statistic is compared only with the power spectrum, ignoring the bispectrum, density split, marked statistics, or field-level inference.

## 36.15 State-label circularity

The web statistic is trained to reproduce a state defined from the same statistic.

## 36.16 Cosmological causation inflation

A correlation between information growth and expansion is described as evidence that information causes expansion.

## 36.17 Life-language inflation

Filaments are described as information-processing organisms without agency tests.

---

# 37. Criteria for Success

## Level 1 — Valid measurement

The information and topology estimators reproduce analytic and synthetic benchmarks.

## Level 2 — Beyond-power information

The IF vector distinguishes phase-structured fields with matched power spectra.

## Level 3 — Beyond-leading summaries

The vector adds held-out information beyond the power spectrum, bispectrum, and density PDF.

## Level 4 — Physical-model sensitivity

The vector distinguishes IF gravity from general relativity after matching tracer abundance and ordinary clustering.

## Level 5 — Baryonic and survey robustness

The signal survives realistic astrophysical and observational nuisance modeling.

## Level 6 — State reconstruction

The web statistic recovers the known IF state in held-out simulations.

## Level 7 — Cross-sector consistency

The web-derived state matches expansion-, growth-, and lensing-derived states.

## Level 8 — Observational detection

The frozen statistic detects a signal in BOSS, DESI, or lensing data consistent with IF simulations.

## Level 9 — Prospective confirmation

A preregistered Euclid or later-survey prediction succeeds.

---

# 38. What Would Count as a Major Discovery?

A useful cosmological-method result would be:

\[
\boxed{
\text{Persistent topology or another interpretable IF statistic adds}
\atop
\text{stable held-out cosmological information beyond conventional}
\atop
\text{two- and three-point summaries.}
}
\]

A stronger modified-gravity result would be:

\[
\boxed{
\text{An environment-sensitive IF statistic distinguishes IF geometry}
\atop
\text{from general relativity after power spectrum, galaxy bias,}
\atop
\text{abundance, baryons, and survey effects are matched.}
}
\]

A field-changing IF result would be:

\[
\boxed{
\hat b_{\mathrm{web}}(z)
=
b_{\mathrm{expansion}}(z)
=
b_{\mathrm{growth}}(z)
=
b_{\mathrm{lensing}}(z)
}
\]

with no cross-sector recalibration.

The strongest result would prospectively connect cosmic topology to galaxy gravity:

\[
\boxed{
a_{\mathrm{IF}}(z)
=
a_{\mathrm{IF},0}
F_a
\left[
\hat b_{\mathrm{web}}(z)
\right]
}
\]

and have that relation confirmed by independent galaxy and survey measurements.

---

# 39. Relationship to Paper 9

Paper 9 asks whether one IF state explains expansion and growth.

Paper 10 adds an independent estimator:

\[
\hat b_{\mathrm{web}}(z).
\]

The state-consistency test becomes:

\[
\boxed{
b_E(z)
=
b_G(z)
=
b_L(z)
=
b_W(z),
}
\]

where:

- \(E\): expansion;
- \(G\): growth;
- \(L\): lensing;
- \(W\): web information.

This overconstrains the theory.

It is intentionally difficult.

---

# 40. Relationship to Paper 8

Paper 8 tests the galactic acceleration scale:

\[
a_{\mathrm{IF}}(z).
\]

Paper 10 tests whether the large-scale cosmic state predicts that scale:

\[
\boxed{
a_{\mathrm{IF}}(z)
=
a_{\mathrm{IF},0}
F_a
\left[
b_W(z)
\right].
}
\]

A failure would show that the galaxy and cosmic-web sectors do not share one state.

---

# 41. Relationship to the Informational Battery

Paper 1 defined accessible nonequilibrium capacity.

Paper 10 measures statistical organization.

These are not automatically the same quantity.

A future physical bridge must derive:

\[
\boxed{
B_{\mathrm{cosmic}}
=
\mathcal B
\left[
b,
T_{\mu\nu},
g_{\mu\nu}
\right]
}
\]

and explain how the observed information vector relates to that physical capacity.

Until such a bridge exists:

\[
\mathbf I_{\mathrm{IF}}
\]

is an observational information summary, not a cosmic energy reservoir.

---

# 42. Relationship to Entropic Cosmology

Previous cosmological work has proposed relative entropy, configuration entropy, or information-production measures as descriptions of structure growth and possible inputs to cosmological backreaction.

Paper 10 does not assume that increasing information causes accelerated expansion.

The correct sequence is:

1. define the statistic;
2. measure it in simulations;
3. derive its relation to the IF action;
4. freeze the expansion prediction;
5. test observations.

A fitted correlation:

\[
w(z)
\sim
\frac{
d\mathcal I
}{
d\ln a
}
\]

is not a derivation.

---

# 43. Criteria for Rejection or Major Revision

Paper 10’s IF claim should be rejected or substantially revised if:

1. the IF vector adds no information beyond strong conventional summaries;
2. apparent information gain is caused by power-spectrum mismatch;
3. the result depends on one smoothing or filtration choice;
4. galaxy bias or baryonic feedback is degenerate with the IF signal;
5. survey masks create the observed topology;
6. the web-state estimator fails held-out simulations;
7. web-, expansion-, and growth-derived IF histories disagree;
8. a generic learned summary performs better with equal robustness;
9. field-level inference captures the same result without an IF-specific state;
10. the signal fails to transfer from simulations to mocks;
11. the signal fails to transfer from BOSS to DESI;
12. the preregistered Euclid prediction fails;
13. the theory repeatedly introduces new information statistics after each failure;
14. no physical derivation connects the statistic to Paper 7’s geometry.

---

# 44. Conclusion

The cosmic web undeniably contains structure beyond the power spectrum.

That fact does not validate IF Theory.

The scientific problem is to determine whether IF identifies a specific, measurable, transferable, and physically meaningful component of that additional structure.

The proposed measurement vector is:

\[
\boxed{
\mathbf I_{\mathrm{IF}}
=
\left[
\mathcal I_{\mathrm{NG}},
\mathcal I_{\mathrm{top}},
\mathcal I_{\mathrm{tidal}},
\mathcal I_{\mathrm{memory}},
\mathcal I_{\mathrm{marked}},
\mathcal I_{\mathrm{cross}}
\right].
}
\]

The primary test is not whether these quantities vary with cosmology.

It is whether they add held-out prediction beyond:

\[
\boxed{
P(k)
+
B(k_1,k_2,k_3)
+
p(\delta)
+
\text{bias and survey controls}.
}
\]

The central IF consistency requirement is:

\[
\boxed{
\hat b_{\mathrm{web}}(z)
=
b_{\mathrm{expansion}}(z)
=
b_{\mathrm{growth}}(z)
=
b_{\mathrm{lensing}}(z).
}
\]

If topology merely repackages known clustering information, the IF statistic fails.

If it detects simulation code, baryonic feedback, galaxy bias, or survey holes, it fails.

If it produces an independent state unrelated to Paper 9, the unification fails.

If, however, one compact information vector predicts held-out gravity and cosmology across simulations, galaxy maps, lensing maps, and future surveys, it would provide the first measurable bridge between IF Theory’s informational language and the observed structure of the universe.

The next paper will freeze the theory before decisive future data arrive:

\[
\boxed{
\textit{A Preregistered IF Forecast for Euclid:}
\atop
\textit{Equations, Observables, and Conditions for Prospective Falsification.}
}
\]

---

# References

1. Pranav, P. et al. “The Topology of the Cosmic Web in Terms of Persistent Betti Numbers.” The work develops a multiscale topological description of the cosmic web using Betti numbers and persistence diagrams. citeturn252655search7

2. Wilding, G. et al. “Persistent Homology of the Cosmic Web I: Hierarchical Topology in \(\Lambda\)CDM Cosmologies.” The study links persistence-diagram evolution to hierarchical gravitational structure formation. citeturn252655search16

3. van de Weygaert, R. et al. “Alpha, Betti and the Megaparsec Universe.” The study applies alpha shapes and scale-dependent Betti numbers to discrete cosmic structures. citeturn252655academia78

4. Grewal, N. et al. “Minkowski Functionals in Joint Galaxy Clustering and Weak Lensing Analyses.” The tested simplified setup found no additional \(\Omega_m\)-\(\sigma_8\) information beyond its \(3\times2\)-point baseline. citeturn974973search0

5. Armijo, J. et al. “Testing Modified Gravity Using a Marked Correlation Function.” The study shows that environment-based marks can expose modified-gravity differences in simulations. citeturn974973academia40

6. Kärcher, M., Bel, J. and de la Torre, S. “Towards an Optimal Marked Correlation Function Analysis for the Detection of Modified Gravity.” The study evaluates environment-sensitive marks and shot-noise corrections. citeturn974973academia41

7. Paillas, E. et al. “Cosmological Constraints from Density-Split Clustering in the BOSS CMASS Galaxy Sample.” The analysis combines density-split and ordinary clustering with simulation-based forward models. citeturn731924academia40

8. Villaescusa-Navarro, F. et al. “The Quijote Simulations.” Quijote consists of 44,100 \(N\)-body simulations across more than 7,000 cosmological models. citeturn974973academia38

9. CAMELS Collaboration. “CAMELS Documentation.” As of June 2026, the project documents more than two petabytes from 16,960 cosmological simulations. citeturn605920search2turn605920search3

10. Garcia-Bellido, J. “Information Content of the Cosmic Web.” The paper develops an information-theoretic treatment based on tidal eigenvalues, morphology, and multifractal information. citeturn731924academia41

11. Wang, J. et al. “Revealing the Neutrino Mass Through Persistent Homology of the Cosmic Web.” The work applies persistent topology to neutrino signatures. citeturn252655search40

12. Yu, H. et al. “Signatures of Massive Neutrinos in the Cosmic Web via Persistent Homology.” The study reports neutrino-dependent signatures in persistence diagrams and Betti curves. citeturn252655search34

13. DESI Collaboration. “DESI Data Release 1.” The public release contains 18.7 million new main-survey spectra. citeturn252655search12

14. Sloan Digital Sky Survey. “Data Release 19.” DR19 is the current public SDSS release and was released in July 2025. citeturn252655search0turn252655search2

15. Euclid Consortium. “Euclid Quick Data Release 2 and Timeline.” Q2 was released June 24, 2026, with the broader DR1 planned for late 2026. citeturn605920search0turn605920search1turn605920search12
