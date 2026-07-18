# Information and Topology in the Cosmic Web
## Testing IF Statistics Beyond the Two-Point Function

**Author:** Phuc Vinh Truong
**Series:** IF Theory Canonical Papers
**Paper:** 10
**Layer:** SCIENCE
**Status:** canonical revision of extracted draft (2026-07-18)
**Supersedes:** canon/extracted/paper-10-extracted.md

---

> ## Status after 2026-07-18
>
> **The IF-H1 falsification does not touch this paper.** IF-H1 died in the agency
> laboratory — lattice forager, linear-Gaussian (Kalman) controller, run-and-tumble
> chemotactic swimmer — where the claim was that a dimensionless combination
> evaluated at competitive break-even is substrate-independent. Different branch,
> different observables, different substrates, different estimator stack. Nothing
> below inherits \(\eta^*\), \(\Upsilon_{\mathrm{IF}}\), \(\Pi_A\), \(\Pi_C\), or
> any agency invariant. No statistic here was validated by IF-H1 and none is
> invalidated by its death. Stated plainly so the record cannot be misread in
> either direction: **the cosmology branch is unaffected.**
>
> **What this branch is, honestly.** It runs *parallel under discipline*; the
> programme's public identity is agency thermodynamics and cosmology is a
> firewalled side channel. The Founding Panel scored it **3/10 plausibility against
> 10/10 stakes**, and it remains **unpreregistered and untested**. This paper is a
> protocol, not a result.
>
> **`RETROFIT_FORECAST` is this paper's dominant risk.** The instrument set — Betti
> curves \(\beta_q(\nu)\), persistence diagrams and their entropy/total-persistence
> compressions, the four Minkowski functionals, tidal-invariant distributions,
> marked correlation functions, cross-epoch mutual information — is exceptionally
> susceptible to post-hoc tuning. Each carries free choices (smoothing kernel and
> scale, filtration type, threshold \(\nu\), persistence exponent \(p\), mark
> exponent, diagram embedding, scale cuts, redshift binning), and each moves the
> answer. Therefore: **every statistic must be frozen — estimator, scale cuts,
> filtration, hyperparameters, covariance model, numerical pass criterion — in a
> timestamped commit BEFORE the data are examined.** No such timestamp exists for
> any statistic in this paper. Until one does, every number this pipeline produces
> is exploratory by construction and must be labelled so.
>
> **The Feynman gate is unmet.** Before any IF curve is plotted alongside a
> measurement, the same pipeline must reproduce the standard \(\Lambda\)CDM
> predictions for these same statistics on the same mocks — genus/Euler-characteristic
> evolution, Minkowski-functional amplitudes, persistence-diagram morphology,
> density-split and marked-correlation baselines. Mock validation first.
>
> **The Noether gate binds this paper to Paper 9.** Any topology-level IF signature
> must descend from the *same single* \(b(z)\) governing Paper 9's background and
> growth. \(\hat b_{\mathrm{web}}(z)\) is a *readout* of that one state, not a new
> sector. If topology requires its own free shape parameters, its own
> redshift-dependent recalibration, or its own amplitude, that is a
> `SECTOR_SPLIT_FIT`, not a discovery.
>
> **Ledger discipline (`ENTROPY_CONFLATION` sweep).** *Entropy* is never used bare
> here. **Persistence entropy** \(S_{\mathrm{pers},q}\) is a
> *structural/configurational* disorder measure over normalized persistence
> lifetimes — dimensionless, no joules per kelvin, not thermodynamic. **Shannon
> entropy, relative entropy (KL divergence), and mutual information** are
> information-ledger quantities over declared ensembles. **Thermodynamic entropy**
> appears in no estimator here and must not be introduced by analogy. Bits are
> never added to joules (§36.2, §41).

---

## Abstract

The cosmic web contains clusters, filaments, walls, tunnels, and voids generated through the nonlinear gravitational evolution of primordial fluctuations. Conventional cosmological analyses describe much of this structure through two-point statistics, especially the correlation function and power spectrum. For a Gaussian random field, the two-point function contains the complete statistical information. The late-time density field, however, is strongly non-Gaussian and carries phase correlations, morphology, topology, environmental dependence, and multiscale organization not generally captured by the power spectrum alone.

This paper asks whether IF Theory identifies a measurable cosmic-structure statistic that adds predictive information beyond established clustering summaries. The challenge is deliberately severe. Persistent homology, Betti curves, Minkowski functionals, density-split clustering, marked correlations, wavelet statistics, field-level inference, and neural summary statistics already probe information beyond conventional two-point measurements. Persistent topology has been used to characterize the hierarchical evolution of clusters, filaments, tunnels, and voids, and recent analyses have applied it to cosmological-parameter and neutrino-mass inference. Marked statistics were developed specifically to enhance environmentally dependent modified-gravity signals. IF Theory therefore cannot claim novelty merely for applying information theory or topology to the cosmic web; doing so would be a `NOVELTY_INFLATION` violation.

The paper proposes an **IF Cosmic Information Vector** rather than a single scalar:

\[
\boxed{\;
\mathbf I_{\mathrm{IF}}
=
\left[
\mathcal I_{\mathrm{NG}},\;
\mathcal I_{\mathrm{top}},\;
\mathcal I_{\mathrm{tidal}},\;
\mathcal I_{\mathrm{memory}},\;
\mathcal I_{\mathrm{marked}},\;
\mathcal I_{\mathrm{cross}}
\right].\;}
\]

Its components quantify (1) non-Gaussian information relative to a covariance-matched Gaussian field; (2) persistent topological organization; (3) information in tidal geometry beyond density; (4) cross-epoch memory retained by the evolving density field; (5) environment-weighted clustering; and (6) information shared across scales, tracers, and epochs.

The principal IF hypothesis is not that these quantities are nonzero — standard nonlinear gravitational evolution already predicts that. The distinctive claim is that **one low-dimensional IF state inferred from the evolution of multiscale cosmic organization predicts expansion, growth, lensing, and galactic acceleration more accurately than conventional statistics with matched information capacity.**

A statistic counts as an IF contribution only if it adds held-out predictive value after conditioning on the power spectrum, bispectrum, one-point density distribution, survey mask, tracer abundance, and baryonic nuisance parameters. The primary success measure is therefore not an aesthetically compelling topology plot but the cross-validated information increment

\[
\boxed{\;
\Delta\mathcal P_{\mathrm{IF}}
=
\operatorname{ELPD}\!\left[\mathbf S_{\mathrm{base}}+\mathbf I_{\mathrm{IF}}\right]
-
\operatorname{ELPD}\!\left[\mathbf S_{\mathrm{base}}\right],\;}
\]

where ELPD is expected held-out log predictive density and \(\mathbf S_{\mathrm{base}}\) contains standard cosmological summaries.

Development proceeds through analytically controlled random fields, the Quijote simulation suite, the CAMELS hydrodynamical suite, survey mocks, SDSS/BOSS and DESI galaxy data, and future preregistered Euclid measurements. Quijote comprises 44,100 full \(N\)-body simulations spanning more than 7,000 cosmological models. As of June 2026, CAMELS documents more than two petabytes from 16,960 \(N\)-body and hydrodynamical simulations, making it suitable for separating cosmological signals from uncertain galaxy-formation physics.

The IF claim is falsified if its proposed statistics contain no stable held-out information beyond established summaries, if apparent gains disappear under survey forward modeling, if they primarily encode galaxy bias or baryonic feedback, if the result depends on arbitrary smoothing and filtration choices, or if a generic field-level model extracts the same information without the IF interpretation.

---

## Keywords

Cosmic web; information theory; persistent homology; Betti numbers; Minkowski functionals; nonlinear structure; marked correlations; density-split clustering; field-level inference; modified gravity; cosmological simulations; preregistration; IF Theory.

---

# 1. Introduction

The large-scale distribution of matter is not a random collection of isolated objects. Gravitational evolution organizes matter into a web containing high-density nodes and clusters, elongated filaments, sheet-like walls, underdense voids, tunnels and connected cavities, and nested structures across many scales.

The power spectrum describes the variance of Fourier modes:

\[
\boxed{\;
\left\langle \delta(\mathbf k)\,\delta^*(\mathbf k')\right\rangle
=
(2\pi)^3\,\delta_D\!\left(\mathbf k-\mathbf k'\right)P(k).\;}
\]

For a statistically homogeneous Gaussian field, the mean and two-point covariance fully determine the distribution. Nonlinear evolution breaks that simplicity. Fourier phases become correlated. The one-point density distribution becomes non-Gaussian. Filaments connect clusters. Voids merge. Halo formation becomes environmentally dependent. Fields with nearly identical power spectra can display entirely different morphology and topology.

This motivates statistics beyond \(P(k)\). Established approaches include the bispectrum and higher-order correlation functions; one-point density distributions; counts in cells; peaks and minima; void statistics; density-split clustering; marked correlation functions; Minkowski functionals; Betti numbers; persistent homology; wavelet scattering transforms; graph representations; learned neural summaries; and field-level inference.

Density-split clustering has already been applied to the BOSS CMASS sample with simulation-calibrated forward models and combined with ordinary two-point clustering to constrain cosmological parameters. Persistent homology has progressed from conceptual cosmic-web description to simulation-based parameter inference and to applications on real weak-lensing maps.

IF Theory therefore faces a sharp test:

\[
\boxed{\;\text{What does IF add that is not already contained in the established}\;}
\]
\[
\boxed{\;\text{programme of nonlinear cosmological statistics?}\;}
\]

The answer cannot be *"the cosmic web contains information"* (every measured statistic contains information), nor *"topology matters"*, nor *"nonlinear structure contains more than the power spectrum"* — all three are established.

The IF proposal must instead identify:

1. a clearly defined quantity, with an estimator and dimensions;
2. a physical mechanism connecting it to the IF state \(b(z)\);
3. a prediction not used to design the statistic;
4. a held-out test against strong alternatives;
5. a result robust to observation, tracer bias, baryons, and numerical method.

Item 3 is where this paper is currently weakest.

---

# 2. Scientific Scope

Paper 10 studies the measurable organization of cosmic structure. It tests whether an IF-motivated information vector summarizes nonlinear organization; predicts cosmological parameters; distinguishes IF gravity from conventional gravity; links structure at different epochs; adds information beyond standard statistics; and connects with the expansion–growth state defined in Paper 9.

It does **not** assume that the cosmic web is alive; that cosmic structure possesses agency; that galaxies process information intentionally; that information creates cosmic energy; that thermodynamic-entropy production causes expansion; that topology by itself proves modified gravity; or that one scalar statistic can summarize the entire universe. Interpretive readings of any of these belong to `canon/30-meaning/` and never to this document (layer firewall, CLAUDE.md §6). Attributing agency or purpose to filaments inside a physics document would be a `TELEOLOGY_INJECTION`.

The word **information** is used only in explicit statistical senses, each of which must be named at every use:

- Shannon entropy (information ledger, over a declared ensemble);
- relative entropy / Kullback–Leibler divergence;
- mutual information;
- predictive information;
- Fisher information;
- expected held-out predictive density;
- topological persistence (a geometric lifetime, not an entropy at all).

These quantities are not interchangeable, and none of them is a thermodynamic entropy.

---

# 3. The Novelty Boundary

This section exists to prevent `NOVELTY_INFLATION`. Each subsection states what is already established and therefore *cannot* be claimed.

## 3.1 Betti numbers and persistent homology

For a three-dimensional excursion set or point-cloud filtration:

- \(\beta_0\) counts connected components;
- \(\beta_1\) counts independent loops or tunnels;
- \(\beta_2\) counts enclosed cavities or void-like regions.

Persistent homology tracks the birth and death of these features as a density threshold, distance scale, or filtration parameter changes.

Prior work has developed persistent Betti-number descriptions of the cosmic web and connected persistence-diagram features to hierarchical gravitational structure formation, showing that clusters, filaments, tunnels, and voids leave recognizable signatures in persistence diagrams, and that those diagrams retain more detailed multiscale information than a single Euler characteristic.

IF Theory cannot claim novelty for computing \(\beta_0\), \(\beta_1\), \(\beta_2\).

## 3.2 Minkowski functionals

In three spatial dimensions, four Minkowski functionals describe the geometry of excursion sets:

\[
V_0,\quad V_1,\quad V_2,\quad V_3,
\]

corresponding broadly to volume, surface area, integrated mean curvature, and Euler characteristic.

Minkowski functionals can capture non-Gaussian morphology, but their *incremental* cosmological information depends on the field, modeling assumptions, included baseline statistics, noise, and nonlinear regime. A simulation study combining clustering and weak-lensing Minkowski functionals found no additional \(\Omega_m\)–\(\sigma_8\) information beyond a simplified \(3\times2\)-point analysis in its tested setup — demonstrating that "beyond two point" does not automatically mean "more useful." Other simulation-based weak-lensing studies have found meaningful improvements from non-Gaussian summaries, including Minkowski functionals, under different setups.

This mixed evidence is the standard Paper 10 must adopt: the prior expectation for an incremental gain is not favourable.

## 3.3 Marked statistics

A marked statistic assigns each tracer a weight \(m_i = m(\text{local environment}_i)\). A marked correlation function is schematically

\[
\boxed{\;
\mathcal M(r)=\frac{1+W(r)}{1+\xi(r)},\;}
\]

where \(W(r)\) is the weighted pair statistic and \(\xi(r)\) is the ordinary correlation function.

Marks can amplify low-density or environmentally screened regions. Simulation studies have demonstrated sensitivity of marked correlations to modified-gravity models whose effects depend on local environment, and have shown that discreteness corrections are essential for unbiased environmental statistics.

IF Theory cannot claim novelty merely for weighting voids or low-density galaxies more heavily.

## 3.4 Density-split clustering

Density-split methods divide locations or tracers by their surrounding density and measure clustering within density classes, cross-correlations around underdense and overdense regions, and sometimes lensing around the same classes. These statistics capture environment-dependent information beyond the ordinary galaxy two-point function. The BOSS CMASS analysis demonstrated that such measurements can be forward modeled and used for cosmological inference while accounting for redshift-space effects, the galaxy–halo connection, and assembly bias.

## 3.5 Tidal geometry

The scalar density contrast \(\delta\) does not fully specify local anisotropic deformation. Let the tidal (deformation) tensor be

\[
T_{ij}=\partial_i\partial_j\Phi ,
\]

with eigenvalues \(\lambda_1,\lambda_2,\lambda_3\) describing collapse or expansion along principal axes.

Information-theoretic work has already proposed using the joint distribution of tidal eigenvalues, shear invariants, morphological classes, and multifractal dimensions to quantify cosmic-web information beyond density alone. This is closely related to the IF objective and sharply reduces the novelty of any generic "tidal information" claim.

## 3.6 Cross-epoch information

The evolving matter field contains statistical memory of earlier states. Published work has explicitly studied mutual information between density fields at different redshifts as a measure of retained cosmic coherence.

IF Theory therefore cannot claim novelty for asking whether the cosmic density field remembers its past. Its stronger and only defensible task is to show that a *specific* cross-epoch information measure predicts an IF observable not already predictable from conventional growth statistics.

## 3.7 Field-level inference

Field-level methods infer cosmology directly from the spatial field rather than compressing it into a small set of summary statistics, and can in principle retain Fourier phases and higher-order information discarded by the power spectrum. Field-level approaches have been applied to modified-gravity inference and to perturbative large-scale-structure modeling.

This creates the hardest comparator. **If a generic field-level model performs as well as or better than the IF statistic, the IF summary may still be convenient but is not uniquely fundamental.**

---

# 4. The Core IF Question

The core question is not whether cosmic structure contains nonlinear information. It is:

\[
\boxed{\;\text{Is there a low-dimensional, physically interpretable information state}\;}
\]
\[
\boxed{\;\text{whose evolution jointly predicts cosmic topology, expansion, growth,}\;}
\]
\[
\boxed{\;\text{lensing, and galactic gravity?}\;}
\]

Let the Paper 9 state be \(b(a)\). Paper 10 proposes that measurable nonlinear organization \(\mathbf I_{\mathrm{IF}}(a)\) may act as an observational estimator of that state:

\[
\boxed{\;
\hat b_{\mathrm{web}}(a)=\mathcal E_{\mathrm{IF}}\!\left[\mathbf I_{\mathrm{IF}}(a)\right].\;}
\]

The unification claim requires

\[
\boxed{\;
\hat b_{\mathrm{web}}(a)=b_{\mathrm{expansion}}(a)=b_{\mathrm{growth}}(a)=b_{\mathrm{lensing}}(a).\;}
\]

If the web estimator requires an unrelated state or flexible redshift-dependent recalibration, it does not support IF unification — it falsifies it. This is the Noether gate in its observational form: one state, no independent sector fits.

---

# 5. Density Fields and Tracers

## 5.1 Matter density

Define the matter-density contrast

\[
\boxed{\;
\delta_m(\mathbf x,z)=\frac{\rho_m(\mathbf x,z)-\bar\rho_m(z)}{\bar\rho_m(z)}.\;}
\]

The true matter field is not directly observed at all scales.

## 5.2 Galaxy density

Observed galaxies define \(\delta_g(\mathbf x,z)\). A bias expansion may be written schematically:

\[
\boxed{\;
\delta_g=b_1\delta_m+\frac{b_2}{2}\delta_m^2+b_{s^2}s^2+\epsilon+\cdots\;}
\]

where \(b_1,b_2,b_{s^2}\) are bias parameters, \(s^2\) is a tidal operator, and \(\epsilon\) is stochasticity.

Topological statistics measured from galaxies are therefore **not** automatically the topology of total matter.

## 5.3 Lensing field

Weak lensing provides a projected estimate of gravitational-potential structure. The convergence field is approximately

\[
\boxed{\;
\kappa(\boldsymbol\theta)=\int_0^{\chi_s}d\chi\,W_\kappa(\chi)\,
\delta_{\mathrm{lens}}\!\left(\chi\boldsymbol\theta,\chi\right).\;}
\]

In modified gravity, \(\delta_{\mathrm{lens}}\) depends on the Weyl potential \(\Phi+\Psi\). Galaxy topology and lensing topology may therefore differ if the IF gravitational slip differs from general relativity. That difference is a potential discriminating signal, and one of the few here that is hard to fake with tracer nuisance parameters.

---

# 6. Smoothing and Filtration

Every information or topology statistic depends on how the field is represented. Let the smoothed field be

\[
\boxed{\;
\delta_R(\mathbf x)=\int d^3y\;W_R\!\left(|\mathbf x-\mathbf y|\right)\delta(\mathbf y).\;}
\]

The analysis must report the smoothing kernel; scale \(R\); grid resolution; tracer density; boundary correction; mask treatment; and interpolation method.

A claim existing only at one smoothing scale is weak. The primary analysis uses a scale family \(R\in[R_{\min},R_{\max}]\) whose bounds are frozen before the data are inspected. Selecting \(R\) after seeing model separation is §36.8, a direct `RETROFIT_FORECAST`.

## 6.1 Density excursion filtration

For threshold \(\nu\), define the superlevel set

\[
\boxed{\;
X_\nu=\left\{\mathbf x:\delta_R(\mathbf x)\geq\nu\right\}.\;}
\]

As \(\nu\) decreases, isolated high-density components merge, loops form, and cavities close.

## 6.2 Point-cloud filtration

For discrete tracers, use an alpha, Čech, Vietoris–Rips, or distance-to-measure filtration. **The primary filtration must be chosen before the confirmatory analysis.** Alternative filtrations are robustness checks, not opportunities to select the most favourable result.

---

# 7. The IF Cosmic Information Vector

Define

\[
\boxed{\;
\mathbf I_{\mathrm{IF}}
=
\left[
\mathcal I_{\mathrm{NG}},\;
\mathcal I_{\mathrm{top}},\;
\mathcal I_{\mathrm{tidal}},\;
\mathcal I_{\mathrm{memory}},\;
\mathcal I_{\mathrm{marked}},\;
\mathcal I_{\mathrm{cross}}
\right].\;}
\]

No scalar combination is initially privileged. Each component answers a different question; collapsing them into one number before the estimator is frozen both discards information and opens a tuning channel.

---

# 8. Non-Gaussian Information

## 8.1 Gaussian reference

Construct a Gaussian field \(\delta_G\) with the same mean, power spectrum, survey window, sampling, and noise as the measured or simulated field.

Let \(p_R(\delta,z)\) be the one-point distribution of the smoothed nonlinear field and \(p_{G,R}(\delta,z)\) the covariance-matched Gaussian reference. Define

\[
\boxed{\;
\mathcal I_{\mathrm{NG}}^{(1)}(R,z)
=
D_{\mathrm{KL}}\!\left[p_R(\delta,z)\,\|\,p_{G,R}(\delta,z)\right].\;}
\]

This is a **Shannon relative entropy** (KL divergence) between two declared one-point distributions — a non-Gaussianity measure on the information ledger. It is not a thermodynamic entropy, not a negentropy in any energetic sense, and not the total information content of the universe.

## 8.2 Field-level relative information

For a tractable field representation \(\mathbf d\), define

\[
\boxed{\;
\mathcal I_{\mathrm{NG}}^{(\mathrm{field})}
=
D_{\mathrm{KL}}\!\left[p_{\mathrm{NL}}(\mathbf d)\,\|\,p_G(\mathbf d\mid C)\right].\;}
\]

Direct high-dimensional KL estimation is difficult. The analysis will use likelihood-ratio estimation; classifier-based density-ratio methods; normalizing flows validated on analytic fields; and lower-dimensional sufficient summaries. **Every estimator must be tested for finite-sample bias** on distributions with known divergence (Notebook 10G).

## 8.3 Conditional non-Gaussian information

To isolate information not captured by baseline summaries:

\[
\boxed{\;
\mathcal I_{\mathrm{NG}\mid\mathrm{base}}
=
I\!\left(\Theta;\mathbf S_{\mathrm{NG}}\mid P,B,\mathrm{PDF}\right),\;}
\]

where \(\Theta\) is the cosmological or gravity model, \(P\) the power spectrum, \(B\) the bispectrum, PDF the one-point density distribution, and \(\mathbf S_{\mathrm{NG}}\) the candidate IF statistic.

This conditional mutual information — not any raw Shannon-entropy value — is the relevant quantity.

---

# 9. Persistent Topological Information

## 9.1 Persistence diagrams

For homology dimension \(q\), define a persistence diagram

\[
\boxed{\;
\mathcal D_q=\left\{(b_i,d_i)\right\},\;}
\]

where \(b_i\) is the filtration value at feature birth, \(d_i\) the value at death, and persistence is \(\pi_i=|d_i-b_i|\).

## 9.2 Betti curves

At filtration parameter \(\nu\):

\[
\boxed{\;
\beta_q(\nu)=\#\left\{i:\;b_i\leq\nu<d_i\right\}.\;}
\]

In three dimensions, \(\beta_0\) counts connected components, \(\beta_1\) loops and tunnels, \(\beta_2\) cavities.

## 9.3 Persistence entropy — a configurational measure

A normalized persistence weight is

\[
p_i=\frac{\pi_i}{\sum_j\pi_j},
\]

and

\[
\boxed{\;
S_{\mathrm{pers},q}=-\sum_i p_i\ln p_i .\;}
\]

**Ledger statement (required).** \(S_{\mathrm{pers},q}\) is a **structural/configurational disorder measure** over the lifetime distribution of topological features — a Shannon functional applied to a *geometric* weight vector, not to a physical microstate ensemble. It is dimensionless, has no temperature conjugate, obeys no second law, and must never be summed with, converted into, or compared numerically against a thermodynamic entropy. Conflating the two is `ENTROPY_CONFLATION`.

Persistence entropy compresses the diagram but loses location information. It must be reported alongside the full diagram or richer diagram embeddings, never as a standalone summary.

## 9.4 Total persistence

\[
\boxed{\;
\operatorname{TP}_{q,p}=\sum_i \pi_i^{\,p}.\;}
\]

Different values of \(p\) emphasize short- or long-lived features differently. **No value of \(p\) may be selected after inspecting model separation.** \(p\) is a preregistered hyperparameter.

## 9.5 IF topological vector

\[
\boxed{\;
\mathcal I_{\mathrm{top}}
=
\left[
\beta_q(\nu),\;
S_{\mathrm{pers},q},\;
\operatorname{TP}_{q,p},\;
\text{diagram embedding}
\right]_{q=0,1,2}.\;}
\]

The IF state estimator may use this vector, but the analysis must determine whether the full topology adds information beyond the power spectrum, the bispectrum, the one-point PDF, an ordinary void catalog, and the Minkowski functionals.

---

# 10. Minkowski Geometry

For excursion set \(X_\nu\), define in three dimensions

\[
\boxed{\;
\mathbf V(\nu)=\left[V_0(\nu),V_1(\nu),V_2(\nu),V_3(\nu)\right],\;}
\]

with approximate interpretations: \(V_0\) volume fraction, \(V_1\) surface area, \(V_2\) integrated mean curvature, \(V_3\) Euler characteristic.

The Euler characteristic relates to the Betti numbers:

\[
\boxed{\;\chi=\beta_0-\beta_1+\beta_2.\;}
\]

Because many different Betti triplets share the same \(\chi\), persistent homology can retain topology discarded by the Euler characteristic — and hence by the classical genus curve — alone. That persistence is not merely a repackaging of the genus statistic must be *demonstrated numerically on the mocks*, not asserted.

---

# 11. Tidal Information

Let the tidal-tensor eigenvalues be ordered \(\lambda_1\geq\lambda_2\geq\lambda_3\), and define the invariants

\[
I_1=\lambda_1+\lambda_2+\lambda_3,
\]
\[
I_2=\lambda_1\lambda_2+\lambda_1\lambda_3+\lambda_2\lambda_3,
\]
\[
I_3=\lambda_1\lambda_2\lambda_3 .
\]

Because \(I_1\propto\delta\), the remaining invariants contain anisotropic information not determined by density alone. Define

\[
\boxed{\;
\mathcal I_{\mathrm{tidal}}
=
I\!\left[(I_2,I_3);\Theta \mid I_1,P(k)\right].\;}
\]

This asks how much cosmological information tidal anisotropy adds *after* density and two-point clustering are known.

## 11.1 Morphological classes

For threshold \(\lambda_{\mathrm{th}}\), classify a location by the number of eigenvalues above threshold, giving the classes void, wall, filament, node. The sign convention and threshold are not universal across the literature, so every result must report \(\lambda_{\mathrm{th}}\) explicitly.

The primary analysis uses continuous eigenvalue distributions rather than relying only on discrete labels; discrete classification discards information and adds a tunable knob.

---

# 12. Cross-Epoch Memory

## 12.1 Lagrangian correspondence

When simulation particles or reconstructed initial conditions provide correspondence, define \(\delta_R(\mathbf q,z_1)\) and \(\delta_R(\mathbf q,z_2)\) for a common Lagrangian location \(\mathbf q\), and

\[
\boxed{\;
\mathcal I_{\mathrm{memory}}(R;z_1,z_2)
=
I\!\left[\delta_R(\mathbf q,z_1);\,\delta_R(\mathbf q,z_2)\right].\;}
\]

## 12.2 Random baseline subtraction

Finite estimators produce positive mutual information even for independent fields. Define the bias-corrected quantity

\[
\boxed{\;
\mathcal I_{\mathrm{memory}}^{\mathrm{corr}}
=
\hat I_{\mathrm{true}}-\mathbb E\!\left[\hat I_{\mathrm{shuffled}}\right].\;}
\]

Uncertainty is estimated through repeated shuffles, simulation realizations, and spatial block resampling.

## 12.3 Conditional memory

To isolate memory beyond linear growth:

\[
\boxed{\;
\mathcal I_{\mathrm{NL\ memory}}
=
I\!\left[\delta(z_1);\delta(z_2)\;\middle|\;\delta_{\mathrm{linear}}(z_1),\,D(z_2)/D(z_1)\right].\;}
\]

This is a difficult estimator and begins in simulations only.

## 12.4 IF hypothesis

The IF proposal predicts that \(\mathcal I_{\mathrm{memory}}\) follows the same state history \(b(a)\) that controls Paper 9 expansion and growth. The restricted form is

\[
\boxed{\;
\frac{d\mathcal I_{\mathrm{memory}}}{d\ln a}
=
F_M\!\left[b(a);\theta\right].\;}
\]

**Noether-gate constraint.** \(F_M\) must be *derived* from the IF action, or at minimum frozen from simulations, before observational application, and must introduce no free shape parameters letting the memory sector drift from expansion and growth. An \(F_M\) fitted independently per redshift bin is a `SECTOR_SPLIT_FIT`.

---

# 13. Cross-Scale Information

For smoothed fields at scales \(R_1<R_2\), define

\[
\boxed{\;
\mathcal I_{\mathrm{cross}}(R_1,R_2;z)=I\!\left[\delta_{R_1};\delta_{R_2}\right].\;}
\]

Nonlinear gravitational collapse transfers and couples structure across scales. The conditional cross-scale information is

\[
\boxed{\;
\mathcal I_{\mathrm{cross}\mid P}=I\!\left[\delta_{R_1};\delta_{R_2}\;\middle|\;P(k)\right].\;}
\]

This notation means conditioning *within the simulation ensemble* on matched or near-matched power-spectrum summaries. It does not mean that one single universe has a fixed known \(P(k)\) at every realization.

---

# 14. Marked Environmental Information

Define a smoothed local density \(\delta_R(\mathbf x_i)\). A bounded mark may be

\[
\boxed{\;
m_i=\left[\frac{1+\delta_*}{1+\delta_*+\delta_R(\mathbf x_i)}\right]^{p}.\;}
\]

Positive \(p\) upweights underdense environments. Alternative marks may use tidal anisotropy, filament distance, void membership, local potential, environmental history, or predicted screening state. **The primary mark family — including \(\delta_*\) and \(p\) — is frozen in simulations** before observational use.

## 14.1 IF-mark hypothesis

If IF gravity differs most in low-density, weakly screened, or rapidly relaxing environments, then an IF-derived mark should increase model separation. The primary question is

\[
\boxed{\;
\Delta\mathcal P_{\mathrm{mark}}
=
\operatorname{ELPD}\!\left[P(k)+\mathcal M_{\mathrm{IF}}\right]
-
\operatorname{ELPD}\!\left[P(k)\right].\;}
\]

The mark succeeds only if it improves held-out inference *after* shot-noise correction and galaxy-bias marginalization.

---

# 15. The IF Web-State Estimator

Let \(\mathbf s(z)\) be a finite summary vector containing selected components of \(\mathbf I_{\mathrm{IF}}\). Define an estimator

\[
\boxed{\;
\hat b_{\mathrm{web}}(z)=\mathcal E_\omega\!\left[\mathbf s(z)\right].\;}
\]

The estimator may be linear, Gaussian-process based, neural, likelihood-ratio based, or a full simulation-based Bayesian inference. **The simplest adequate model is preferred** — estimator flexibility is itself a retrofit channel.

## 15.1 Training target

The estimator is trained in simulations where \(b(z)\) is known from the input IF cosmology. For conventional cosmologies without an IF field, an effective comparison state is defined only if it is mathematically explicit. No arbitrary labels such as "high information" and "low information" are permitted — a label without an estimator is `METAPHOR_MATH`.

## 15.2 State consistency

After training, the test is

\[
\boxed{\;
\hat b_{\mathrm{web}}(z)\;\stackrel{?}{=}\;b_{\mathrm{expansion}}(z).\;}
\]

Define \(\Delta b(z)=\hat b_{\mathrm{web}}(z)-b_{\mathrm{expansion}}(z)\) and the covariance-weighted statistic

\[
\boxed{\;
Q_{\mathrm{web}}=\Delta\mathbf b^\top C_b^{-1}\Delta\mathbf b .\;}
\]

The rejection threshold on \(Q_{\mathrm{web}}\) is calibrated on synthetic universes **and recorded in the preregistration commit**, not chosen after the observed value is known.

---

# 16. Primary Baseline Summary Set

The base summary vector is

\[
\boxed{\;
\mathbf S_{\mathrm{base}}
=
\left[
P_0(k),\,P_2(k),\,P_4(k),\,B(k_1,k_2,k_3),\,p(\delta_R),\,n_g
\right],\;}
\]

where \(P_\ell\) are redshift-space power-spectrum multipoles, \(B\) a bispectrum summary, \(p(\delta_R)\) the one-point density PDF, and \(n_g\) the tracer abundance.

A reduced baseline may omit the bispectrum during early development. The **final** claim must compare against a strong baseline containing at least two-point clustering, a leading higher-order statistic, tracer density, and survey nuisance information. Comparing only against \(P(k)\) is the baseline-straw-man failure (§36.14).

---

# 17. Primary Scientific Endpoint

The primary endpoint is held-out predictive gain:

\[
\boxed{\;
\Delta\mathcal P_{\mathrm{IF}}
=
\operatorname{ELPD}\!\left[\Theta\mid \mathbf S_{\mathrm{base}}+\mathbf I_{\mathrm{IF}}\right]
-
\operatorname{ELPD}\!\left[\Theta\mid \mathbf S_{\mathrm{base}}\right].\;}
\]

Possible prediction targets \(\Theta\) include \(\Omega_m\); \(\sigma_8\); neutrino mass; modified-gravity strength; IF transition parameters; \(b(z)\); the Paper 9 expansion–growth residual; and model identity. **The primary target is fixed before the held-out test.**

## 17.1 Practical significance

Statistical significance alone is insufficient. A successful statistic must produce at least one of: materially tighter held-out parameter inference; reduced parameter degeneracy; improved model discrimination; improved calibration; an accurate cross-epoch prediction; or a lower-dimensional sufficient representation.

The numerical threshold for "material improvement" is calibrated and committed before observational unblinding.

---

# 18. Null Universes

| Null | Construction | What it tests |
|---|---|---|
| 18.1 Gaussian random field | Same \(P(k)\) as target | No genuinely nonlinear topology beyond Gaussian expectation; quantifies estimator bias |
| 18.2 Phase-randomized field | Fourier amplitudes preserved, phases randomized; \(P(k)\) fixed | The most direct baseline for information beyond the two-point function |
| 18.3 Lognormal field | Matched \(P(k)\) and approximate one-point PDF | Whether IF statistics detect more than simple non-Gaussian density skewness |
| 18.4 Bispectrum-matched field | Matched \(P(k)\), \(B\), and one-point PDF where feasible | Isolates residual discriminating information at higher order or in spatial organization |
| 18.5 Shuffled environment | Tracer positions preserved, marks/environmental labels permuted | Establishes the null marked-correlation distribution |
| 18.6 Wrong-mask universe | Deliberately incorrect masks and selection functions | The analysis must identify the resulting bias rather than interpret it as topology |
| 18.7 Standard-gravity universe | \(\Lambda\)CDM as the null physical model | IF statistics must not produce a false IF state |

Null 18.7 is the Feynman gate in operational form: the pipeline must reproduce \(\Lambda\)CDM topology correctly before it may report a deviation from it.

---

# 19. Simulation Program

## 19.1 Analytic random fields

Begin with Gaussian random fields; lognormal fields; controlled non-Gaussian fields; Voronoi-web models; and fractal and multifractal point processes. These provide known or controllable morphology, making estimator correctness decidable rather than assumed.

## 19.2 Quijote

Quijote contains 44,100 \(N\)-body simulations spanning more than 7,000 cosmological models and was designed to quantify the information content of cosmological observables and support machine-learning applications. It includes snapshots, halo and void catalogs, and numerous summary statistics.

Quijote will be used for cosmological-parameter sensitivity; covariance estimation; phase-matched controls; neutrino tests; Fisher comparisons; and summary-statistic selection (the last performed on training splits only).

## 19.3 CAMELS

CAMELS combines \(N\)-body and hydrodynamical simulations across cosmological and astrophysical parameters. Its documentation reports 16,960 simulations and more than two petabytes of data as of June 2026, including thousands of matched \(N\)-body and hydrodynamic runs plus multifield maps and grids.

CAMELS will be used to test whether an apparent IF statistic is actually driven by stellar feedback; active-galactic-nucleus feedback; gas physics; star formation; or tracer selection.

## 19.4 IF simulations

Paper 7 and Paper 9 must eventually provide dedicated IF structure-formation simulations. **Without them, Paper 10 can test generic information statistics but cannot validate the distinctive IF prediction at all** — the branch's rate-limiting dependency, and one reason the plausibility score stands at 3/10.

The IF simulation suite must vary shared IF parameters; initial conditions; baryonic physics; numerical resolution; and box size.

---

# 20. Observational Data Ladder

## 20.1 BOSS and legacy SDSS

BOSS galaxy catalogs provide an established testbed with mature masks, mocks, covariance products, and clustering analyses. Density-split clustering and other beyond-two-point methods have already been demonstrated on BOSS data.

## 20.2 SDSS DR19

SDSS DR19 is the current public SDSS release, published in July 2025, and is the first major public release containing SDSS-V spectroscopic products — although not every DR19 component is optimized for the same large-scale-structure cosmology analysis as BOSS. DR19 is useful for pipeline development; tracer studies; cross-checks; and future SDSS-V large-scale-structure products.

## 20.3 DESI DR1

DESI DR1 was released in March 2025 and contains 18.7 million new main-survey spectra from its first year. DESI's published first-year full-shape analysis used millions of galaxy and quasar redshifts across multiple tracer and redshift bins. DR1 provides the principal current spectroscopic target for a large-volume observed cosmic-web analysis.

## 20.4 DESI DR2 products

Public DR2 products currently include cosmological chains and best-fit products associated with the three-year BAO analysis. Those products are valuable for Paper 9 but do not by themselves provide the complete raw three-year galaxy field required for every Paper 10 topology analysis. **The analysis must not claim access to unreleased object-level data.**

## 20.5 Euclid

Euclid released Quick Data Release 2 on 24 June 2026, with a broader DR1 release scheduled for late 2026. The relevant wide-area cosmological products remain a prospective test for this paper. **The IF topology prediction must be frozen — in a timestamped commit — before those decisive products are used.** This is the single highest-value preregistration opportunity available to the branch.

---

# 21. Survey Forward Modeling

A measured cosmic-web statistic depends on more than cosmology. The mock pipeline must include survey geometry; angular mask; radial selection; redshift failure; fiber assignment; completeness; redshift uncertainty; stellar contamination; tracer bias; halo occupation; assembly bias; peculiar velocities; Alcock–Paczyński distortion; reconstruction choices; and shot noise.

The primary comparison occurs in **observed space**. A statistic measured on a periodic simulation cube cannot be compared directly with a masked survey catalog.

---

# 22. Boundary and Mask Treatment

Topology is uniquely sensitive to boundaries. A mask can create artificial disconnected components; false tunnels; false cavities; truncated filaments; and altered persistence.

Required controls:

1. periodic-box benchmark;
2. survey-mask injection;
3. random-catalog correction;
4. buffer-zone analysis;
5. relative homology where appropriate;
6. masked-field simulations;
7. null tests under rotated or displaced masks.

**The mask pipeline is frozen before examining model differences.** Counting survey holes as cosmic voids (§36.7) is the easiest failure mode here to commit.

---

# 23. Shot Noise and Sparse Sampling

A discrete tracer sample introduces Poisson and non-Poisson sampling effects. The observed point set is \(\{\mathbf x_i\}_{i=1}^{N_g}\), and topology can change substantially as \(n_g\) changes.

The analysis must compare catalogs at matched number density; redshift distribution; bias; and survey geometry. Downsampling tests determine whether model differences survive equal tracer density. Marked-correlation work has shown that discreteness corrections can be essential for unbiased environmental statistics.

---

# 24. Redshift-Space Distortions

Observed radial positions include peculiar velocities:

\[
\boxed{\;
\mathbf s=\mathbf x+\frac{\mathbf v\cdot\hat{\mathbf n}}{aH}\,\hat{\mathbf n}.\;}
\]

Redshift-space distortions compress structures on large scales; elongate virialized clusters; alter filament connectivity; distort void shapes; and change persistence diagrams.

The model must predict topology **in redshift space**, rather than correcting the data to an assumed real-space cosmology and then testing that cosmology — which is circular.

---

# 25. Galaxy Bias and Assembly Bias

A gravity or IF signal can be imitated by changes in how galaxies occupy halos and environments. The mock program will vary halo occupation; stellar-to-halo relations; satellite fraction; velocity bias; concentration dependence; and environmental assembly bias.

A statistic is useful only if it separates *gravity or cosmology* from *galaxy formation and selection*. Density-split analyses of BOSS have found evidence that environment-dependent assembly bias matters in the galaxy–halo model, reinforcing this requirement.

---

# 26. Baryonic Physics

Hydrodynamical feedback changes the small-scale density and lensing fields. For each statistic, define a baryonic sensitivity

\[
\boxed{\;
\Delta_B\mathbf I=\mathbf I_{\mathrm{hydro}}-\mathbf I_{N\text{-body}}\;}
\]

and an IF/gravity sensitivity

\[
\boxed{\;
\Delta_{\mathrm{IF}}\mathbf I=\mathbf I_{\mathrm{IF}}-\mathbf I_{\mathrm{GR}} .\;}
\]

A useful scale requires

\[
\left|\Delta_{\mathrm{IF}}\mathbf I\right|\;\gtrsim\;\left|\Delta_B\mathbf I\right|
\]

or sufficiently accurate baryonic marginalization. The primary analysis uses conservative scales where baryonic uncertainty is controlled, and those scales are frozen in Notebook 10O before observed data are touched.

---

# 27. Information Sufficiency and Complementarity

A new summary statistic can be sufficient, complementary, redundant, or unstable.

## 27.1 Redundancy

If

\[
I\!\left(\Theta;\mathbf I_{\mathrm{IF}}\mid \mathbf S_{\mathrm{base}}\right)\approx 0,
\]

the IF statistic adds no parameter information.

## 27.2 Complementarity

If

\[
I\!\left(\Theta;\mathbf I_{\mathrm{IF}}\mid \mathbf S_{\mathrm{base}}\right)>0,
\]

it adds information beyond the baseline.

## 27.3 Sufficiency

If

\[
P\!\left(\Theta\mid \mathbf I_{\mathrm{IF}}\right)\approx P\!\left(\Theta\mid \text{full field}\right),
\]

the statistic is approximately sufficient for \(\Theta\) within the tested model class. The full field-level posterior is the benchmark wherever computationally feasible.

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

The IF vector is successful only if it provides a favorable balance of predictive information; interpretability; robustness; computational cost; and transfer across surveys. Winning on interpretability alone, after losing on predictive information, is not a result.

---

# 29. Core Hypotheses

Each hypothesis carries an explicit falsifier. A hypothesis without one is not admitted.

## CT-H1 — Beyond-two-point hypothesis

At fixed power spectrum, IF topology and information summaries distinguish nonlinear structure histories.

**Falsifier.** Phase-randomized or alternative fields with matched two-point statistics are indistinguishable under the IF vector.

## CT-H2 — Beyond-bispectrum hypothesis

At fixed power spectrum, bispectrum, and one-point PDF, at least one IF component retains held-out cosmological or gravity information.

**Falsifier.** All IF information is captured by those established summaries.

## CT-H3 — Tidal-information hypothesis

Tidal invariants provide predictive information beyond density and power-spectrum information.

**Falsifier.** After conditioning on density and standard clustering, tidal information adds no stable prediction.

## CT-H4 — Cross-epoch-memory hypothesis

The cosmic field retains measurable nonlinear cross-epoch information not reducible to the linear growth factor.

**Falsifier.** Cross-epoch mutual information is fully explained by linear evolution and estimator bias.

## CT-H5 — IF-state hypothesis

One web-derived state \(\hat b_{\mathrm{web}}(z)\) matches the state inferred independently from expansion and growth.

**Falsifier.** Web, expansion, and growth require incompatible state histories.

## CT-H6 — Modified-gravity sensitivity hypothesis

Environment- and topology-sensitive IF statistics distinguish IF gravity from general relativity after matching the power spectrum and tracer population.

**Falsifier.** Differences vanish after matching bias, abundance, and two-point clustering.

## CT-H7 — Baryonic robustness hypothesis

The IF signal survives hydrodynamical feedback marginalization on preregistered scales.

**Falsifier.** Baryonic uncertainty is larger than, or degenerate with, the proposed signal.

## CT-H8 — Survey robustness hypothesis

The information gain survives realistic masks, selection, redshift-space distortions, and shot noise.

**Falsifier.** The simulation signal disappears in survey-space mocks.

## CT-H9 — Cross-survey hypothesis

A statistic calibrated on one survey or tracer predicts another without full retraining.

**Falsifier.** Every sample requires an unrelated empirical calibration.

## CT-H10 — Predictive-compression hypothesis

A low-dimensional IF vector approaches the predictive performance of field-level inference for selected IF parameters.

**Falsifier.** The vector loses most relevant information, or requires a dimension comparable to the field itself.

## CT-H11 — Prospective Euclid hypothesis

A frozen IF topology or information forecast predicts future Euclid cosmology products.

**Falsifier.** The preregistered forecast fails.

CT-H11 is the only hypothesis in this list that can currently deliver a genuine prediction rather than a retrodiction, because it is the only one whose data do not yet exist. It is therefore the branch's highest-value target — and it is worthless without the timestamped freeze.

---

# 30. Prediction Hierarchy

**Prediction T1 — Matched-power separation.** Fields generated under IF and standard gravity with matched \(P(k)\) have different persistent-diagram and tidal-information distributions.

**Prediction T2 — Environment-enhanced separation.** The largest IF–GR separation occurs in underdense or weakly screened regions, selected *before* measurement.

**Prediction T3 — Cross-epoch coherence.** The IF model predicts a distinct redshift evolution of \(\mathcal I_{\mathrm{memory}}\) at fixed linear growth.

**Prediction T4 — Web–expansion consistency.** The topology-inferred \(\hat b_{\mathrm{web}}(z)\) matches Paper 9's \(b_{\mathrm{expansion}}(z)\).

**Prediction T5 — Web–lensing consistency.** Galaxy topology and lensing topology are linked through Paper 7's gravitational-slip prediction.

**Prediction T6 — Galaxy-scale link.** The cosmic-web state predicts Paper 8's \(a_{\mathrm{IF}}(z)\):

\[
\boxed{\;
a_{\mathrm{IF}}(z)=a_{\mathrm{IF},0}\,F_a\!\left[\hat b_{\mathrm{web}}(z)\right].\;}
\]

Prediction T6 inherits the archived SPARC deficit recorded in the kill log: the prior-generation IF galaxy law lost to both MOND and NFW on a fair-rules 175-galaxy benchmark. Any galaxy-scale link proposed here starts from a deficit, not a blank slate.

---

# 31. Preregistered Analysis Sequence

| Stage | Content |
|---|---|
| 0 | **Mathematical validation** — every statistic validated on fields with known topology and information |
| 1 | **Gaussian and lognormal nulls** — estimator bias and false-detection rates measured |
| 2 | **Standard \(\Lambda\)CDM simulations** — reproduce established topology evolution (Feynman gate) |
| 3 | **Standard beyond-two-point benchmarks** — reproduce Minkowski, marked, density-split, and persistence baselines |
| 4 | **Hydrodynamical robustness** — CAMELS quantifies baryonic sensitivity |
| 5 | **IF simulation forecast** — freeze the IF–GR difference before observational analysis |
| 6 | **Survey-mock validation** — inject masks, selection, bias, redshift-space effects |
| 7 | **BOSS/SDSS validation** — test mature lower-volume catalogs |
| 8 | **DESI DR1 confirmatory test** — apply the frozen pipeline to the larger current public spectroscopic sample |
| 9 | **Cross-probe test** — compare galaxy and weak-lensing topology |
| 10 | **Euclid preregistration** — freeze the future forecast before DR1 cosmology products are used |

Stages 0–3 are gates, not milestones. Failure at Stage 2 or 3 halts the programme rather than being routed around.

---

# 32. Statistical Standards

## 32.1 Simulation independence

Subvolumes from one simulation are not fully independent universes. Covariance estimation must account for shared initial phases; shared long-wavelength modes; repeated subvolume extraction; and simulation-family dependence.

## 32.2 Paired-phase simulations

Paired initial conditions are valuable for reducing cosmic variance when comparing gravity models. The analysis must distinguish variance reduction from independent evidence — a paired-phase difference is not \(N\) independent detections.

## 32.3 Hyperparameter separation

Smoothing scale, persistence threshold, mark parameters, and diagram embeddings are hyperparameters. **They are selected on training simulations only**, and their values are written into the preregistration commit.

## 32.4 Covariance uncertainty

Covariance matrices estimated from finite simulations introduce uncertainty. Use analytic corrections where valid; shrinkage; simulation-based likelihoods; covariance marginalization; and held-out calibration.

## 32.5 Coverage

Posterior credible intervals must achieve correct empirical coverage in synthetic universes. **A tight but miscalibrated posterior is a failure**, not a discovery.

## 32.6 Multiple statistics

Testing many topological summaries creates a large look-elsewhere effect — the single largest quantitative reason this paper's `RETROFIT_FORECAST` risk is elevated. The primary statistic and endpoint are preregistered, and **all** explored summaries are reported, including those that failed.

---

# 33. Deterministic Notebook Program

Each notebook declares Prediction · Baseline · Data · Pass criterion · Falsifier before it runs.

| Notebook | Content |
|---|---|
| 10A | **Gaussian random-field laboratory** — fields with known \(P(k)\); validate density PDF, covariance, Minkowski functionals, Betti curves, estimator bias |
| 10B | **Phase-randomized controls** — preserve amplitudes, randomize phases; quantify which statistics detect the lost spatial organization |
| 10C | **Lognormal and controlled non-Gaussian fields** — matched power spectra with adjustable skewness, kurtosis, phase coupling |
| 10D | **Persistence pipeline validation** — cubical complexes, alpha complexes, persistence diagrams, Betti curves, persistence landscapes, persistence images; validated on synthetic objects of known topology |
| 10E | **Minkowski functional validation** — compute \(V_0,V_1,V_2,V_3\); check analytic Gaussian-field expectations where applicable |
| 10F | **Tidal tensor and web classes** — compute \(T_{ij}\), \(\lambda_i\), \(I_1,I_2,I_3\); validate rotational invariance and numerical derivatives |
| 10G | **Mutual-information estimator audit** — compare histogram, kernel, nearest-neighbor, neural, and density-ratio estimators; test bias on known distributions |
| 10H | **Cross-epoch memory** — measure \(\mathcal I_{\mathrm{memory}}(R;z_1,z_2)\) in \(N\)-body trajectories with shuffled and linear-growth controls |
| 10I | **Marked statistics** — density, tidal, filament, and IF-predicted marks; shot-noise sensitivity audit |
| 10J | **Density-split baseline** — reproduce a standard density-split pipeline *before* introducing IF summaries |
| 10K | **Quijote data manifest** — register snapshots, halo catalogs, void catalogs, parameter tables, initial seeds; checksums and data lineage |
| 10L | **Quijote parameter sensitivity** — derivatives of each statistic w.r.t. \(\Omega_m,\Omega_b,h,n_s,\sigma_8,M_\nu,w\) |
| 10M | **Fisher information benchmark** — compare information matrices of power spectrum, bispectrum, topology, and combined summaries; stability-corrected numerical derivatives |
| 10N | **CAMELS hydrodynamic audit** — statistic variation across cosmology, supernova feedback, AGN feedback, simulation code |
| 10O | **Baryonic-safe scale selection** — freeze the scales retained for observed analysis |
| 10P | **IF versus GR simulations** — measure the preregistered \(\Delta_{\mathrm{IF}}\mathbf I\); match initial phases, tracer density, power spectrum where possible, halo occupation |
| 10Q | **Survey mask laboratory** — inject SDSS, BOSS, DESI, and Euclid-like windows into simulation mocks |
| 10R | **Redshift-space topology** — compare real- and redshift-space statistics; build the forward model |
| 10S | **Galaxy-bias stress test** — vary halo occupation and assembly bias; determine which summaries remain gravity-sensitive |
| 10T | **BOSS pipeline reproduction** — reproduce published clustering and density-split baseline results (Feynman gate) |
| 10U | **DESI DR1 manifest** — register public catalogs, randoms, masks, completeness products, mock resources |
| 10V | **DESI DR1 cosmic-web measurement** — measure the frozen IF vector in redshift bins and tracer classes |
| 10W | **Weak-lensing topology** — corresponding summaries in convergence or mass maps |
| 10X | **Galaxy–lensing cross-topology** — test whether galaxy and Weyl-potential structures satisfy the IF gravitational-slip relation |
| 10Y | **IF web-state estimator** — infer \(\hat b_{\mathrm{web}}(z)\); validate calibration and coverage |
| 10Z | **Web–expansion–growth consistency** — compare \(\hat b_{\mathrm{web}}\), \(b_{\mathrm{expansion}}\), \(b_{\mathrm{growth}}\), \(b_{\mathrm{lensing}}\) |
| 10AA | **Summary sufficiency** — compare IF summaries against field-level inference |
| 10AB | **Held-out predictive gain** — calculate \(\Delta\mathcal P_{\mathrm{IF}}\) |
| 10AC | **Euclid frozen forecast** — save redshift bins, smoothing scales, topology statistic, covariance forecast, nuisance model, pass threshold, failure threshold, code and environment hashes |
| 10AD | **Adversarial audit** — a separate agent attempts to explain every positive result through power-spectrum leakage, bispectrum leakage, tracer abundance, smoothing choice, mask topology, shot noise, galaxy bias, baryonic feedback, covariance error, simulation overfitting, neural-estimator bias, and post hoc statistic choice |

Notebook 10AC is the preregistration artifact. Its commit timestamp is the paper's scientific currency; nothing before it counts as a prediction.

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
│   ├── shannon_entropy.py
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

The module formerly named `entropy.py` is named `shannon_entropy.py`: a bare `entropy` symbol in a namespace that also touches gravitational fields is exactly how ledgers get merged.

---

# 35. Reproducibility Record

Each result emits:

```yaml
experiment_id: if-cosmic-web-10
paper_version: null
git_commit: null
preregistration_commit: null      # REQUIRED before any confirmatory claim
preregistration_timestamp: null   # currently absent for every statistic
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

lcdm_baseline_reproduced: null    # Feynman gate; must be true before IF plotting
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

**36.1 Shannon/KL entropy without an ensemble.** A Shannon or KL quantity is calculated from one field without defining the probability distribution or sampling ensemble. A Shannon entropy of a single realization is not defined.

**36.2 Bits treated as energy.** An information statistic is inserted into a gravitational equation without a physical conversion law or dimensional coefficient. This merges the information and energy ledgers and is forbidden outright.

**36.3 Power-spectrum leakage.** A neural or topological statistic succeeds because the fields were not actually matched in \(P(k)\).

**36.4 Tracer-abundance leakage.** Model classes have different galaxy counts, making classification trivial.

**36.5 Bias leakage.** The statistic measures the galaxy–halo prescription rather than gravity.

**36.6 Baryonic leakage.** Hydrodynamic feedback generates the apparent IF signature.

**36.7 Mask topology.** Survey holes are counted as cosmic voids or tunnels.

**36.8 Smoothing hindsight.** The smoothing scale is selected after examining model separation. (`RETROFIT_FORECAST`.)

**36.9 Filtration hindsight.** The topology method is selected because it produces the desired result. (`RETROFIT_FORECAST`.)

**36.10 Phase-randomization failure.** The statistic does not distinguish the real field from a phase-randomized field with the same power spectrum.

**36.11 Mutual-information estimator bias.** A high-dimensional neural estimator reports false nonzero information.

**36.12 Simulation memorization.** The classifier recognizes simulation code or numerical artifacts rather than physics.

**36.13 Covariance underestimation.** Subvolumes of one simulation are treated as independent universes.

**36.14 Baseline straw man.** The IF statistic is compared only with the power spectrum, ignoring the bispectrum, density split, marked statistics, or field-level inference.

**36.15 State-label circularity.** The web statistic is trained to reproduce a state defined from the same statistic.

**36.16 Cosmological causation inflation.** A correlation between information growth and expansion is described as evidence that information *causes* expansion.

**36.17 Life-language inflation.** Filaments are described as information-processing organisms without agency tests. (`TELEOLOGY_INJECTION`; also a layer-firewall breach.)

---

# 37. Criteria for Success

| Level | Criterion |
|---|---|
| 1 | **Valid measurement** — information and topology estimators reproduce analytic and synthetic benchmarks |
| 2 | **Beyond-power information** — the IF vector distinguishes phase-structured fields with matched power spectra |
| 3 | **Beyond-leading summaries** — the vector adds held-out information beyond \(P(k)\), \(B\), and the density PDF |
| 4 | **Physical-model sensitivity** — the vector distinguishes IF gravity from GR after matching tracer abundance and ordinary clustering |
| 5 | **Baryonic and survey robustness** — the signal survives realistic astrophysical and observational nuisance modeling |
| 6 | **State reconstruction** — the web statistic recovers the known IF state in held-out simulations |
| 7 | **Cross-sector consistency** — the web-derived state matches expansion-, growth-, and lensing-derived states |
| 8 | **Observational detection** — the frozen statistic detects a signal in BOSS, DESI, or lensing data consistent with IF simulations |
| 9 | **Prospective confirmation** — a preregistered Euclid or later-survey prediction succeeds |

The branch currently sits **below Level 1**: no estimator in this paper has been validated against analytic benchmarks in this repository.

---

# 38. What Would Count as a Major Discovery?

A useful **cosmological-method** result would be: persistent topology or another interpretable IF statistic adds stable held-out cosmological information beyond conventional two- and three-point summaries.

A stronger **modified-gravity** result would be: an environment-sensitive IF statistic distinguishes IF geometry from general relativity after power spectrum, galaxy bias, abundance, baryons, and survey effects are matched.

A **field-changing IF** result would be

\[
\boxed{\;
\hat b_{\mathrm{web}}(z)=b_{\mathrm{expansion}}(z)=b_{\mathrm{growth}}(z)=b_{\mathrm{lensing}}(z)\;}
\]

with no cross-sector recalibration.

The strongest result would prospectively connect cosmic topology to galaxy gravity,

\[
\boxed{\;
a_{\mathrm{IF}}(z)=a_{\mathrm{IF},0}\,F_a\!\left[\hat b_{\mathrm{web}}(z)\right],\;}
\]

with that relation confirmed by independent galaxy and survey measurements.

Note the asymmetry of stakes: the first result would be a modest methods contribution that the existing literature could plausibly reach without IF; the last would be decisive. The panel's 10/10 stakes rating refers to the last; the 3/10 plausibility rating refers to reaching it.

---

# 39. Relationship to Paper 9

Paper 9 asks whether one IF state explains expansion and growth. Paper 10 adds an independent estimator \(\hat b_{\mathrm{web}}(z)\). The state-consistency test becomes

\[
\boxed{\;
b_E(z)=b_G(z)=b_L(z)=b_W(z),\;}
\]

where \(E\) is expansion, \(G\) growth, \(L\) lensing, and \(W\) web information.

This overconstrains the theory. It is intentionally difficult, and it is the Noether gate made observable: four readouts, one state, zero per-sector freedom.

---

# 40. Relationship to Paper 8

Paper 8 tests the galactic acceleration scale \(a_{\mathrm{IF}}(z)\). Paper 10 tests whether the large-scale cosmic state predicts that scale:

\[
\boxed{\;
a_{\mathrm{IF}}(z)=a_{\mathrm{IF},0}\,F_a\!\left[b_W(z)\right].\;}
\]

A failure would show that the galaxy and cosmic-web sectors do not share one state.

---

# 41. Relationship to the Informational Battery

Paper 1 defined accessible nonequilibrium capacity. Paper 10 measures statistical organization. **These are not automatically the same quantity**, and treating them as one would merge the energy and information ledgers.

A future physical bridge must derive

\[
\boxed{\;
B_{\mathrm{cosmic}}=\mathcal B\!\left[b,\,T_{\mu\nu},\,g_{\mu\nu}\right]\;}
\]

and explain how the observed information vector relates to that physical capacity. Until such a bridge exists, \(\mathbf I_{\mathrm{IF}}\) is an observational information summary, **not a cosmic energy reservoir**.

---

# 42. Relationship to Entropic Cosmology

Prior cosmological work has proposed relative entropy, configurational entropy, or information-production measures as descriptions of structure growth and as possible inputs to cosmological backreaction. Those proposals — including the 2026 entropic-backreaction preprint literature — are prior art. Restating them in IF vocabulary would be `NOVELTY_INFLATION`.

Paper 10 does **not** assume that increasing information causes accelerated expansion. The correct sequence is:

1. define the statistic, with its estimator and ensemble;
2. measure it in simulations;
3. derive its relation to the IF action;
4. freeze the expansion prediction in a timestamped commit;
5. test observations.

A fitted correlation of the form

\[
w(z)\sim\frac{d\mathcal I}{d\ln a}
\]

is not a derivation. It is a curve passed through points, and calling it a prediction after the fact is the exact failure this paper's status note names as its dominant risk.

---

# 43. Criteria for Rejection or Major Revision

Paper 10's IF claim should be rejected or substantially revised if:

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
14. no physical derivation connects the statistic to Paper 7's geometry.

Criterion 13 is a stop rule, and it is binding in the same way the IF-H1 stop rule was binding: two principled attempts, then the claim is dead. The agency branch honored its stop rule on 2026-07-18. This branch will honor its own.

---

# 44. Conclusion

The cosmic web undeniably contains structure beyond the power spectrum. That fact does not validate IF Theory.

The scientific problem is to determine whether IF identifies a specific, measurable, transferable, and physically meaningful component of that additional structure. The proposed measurement vector is

\[
\boxed{\;
\mathbf I_{\mathrm{IF}}
=
\left[
\mathcal I_{\mathrm{NG}},\;
\mathcal I_{\mathrm{top}},\;
\mathcal I_{\mathrm{tidal}},\;
\mathcal I_{\mathrm{memory}},\;
\mathcal I_{\mathrm{marked}},\;
\mathcal I_{\mathrm{cross}}
\right].\;}
\]

The primary test is not whether these quantities vary with cosmology. It is whether they add held-out prediction beyond

\[
\boxed{\;
P(k)+B(k_1,k_2,k_3)+p(\delta)+\text{bias and survey controls}.\;}
\]

The central IF consistency requirement is

\[
\boxed{\;
\hat b_{\mathrm{web}}(z)=b_{\mathrm{expansion}}(z)=b_{\mathrm{growth}}(z)=b_{\mathrm{lensing}}(z).\;}
\]

If topology merely repackages known clustering information, the IF statistic fails. If it detects simulation code, baryonic feedback, galaxy bias, or survey holes, it fails. If it produces an independent state unrelated to Paper 9, the unification fails.

If, however, one compact information vector predicts held-out gravity and cosmology across simulations, galaxy maps, lensing maps, and future surveys, it would provide the first measurable bridge between IF Theory's informational language and the observed structure of the universe.

That outcome is possible. It is not, on current evidence, likely — 3/10 by the panel's own reckoning — and this paper does not pretend otherwise. What it does insist on is that the branch be *decidable*: frozen statistics, a reproduced \(\Lambda\)CDM baseline, one state across four sectors, and a timestamp that predates the data.

The next paper freezes the theory before decisive future data arrive: *A Preregistered IF Forecast for Euclid: Equations, Observables, and Conditions for Prospective Falsification.*

---

## Attribution note

The prior work referenced above is described in plain prose rather than by formal citation because the extracted draft's citation tokens were transcript residue and could not be resolved to verifiable bibliographic records. The substantive attributions retained are: persistent-Betti-number and persistent-homology descriptions of the cosmic web, including hierarchical \(\Lambda\)CDM persistence studies and alpha-shape treatments of discrete cosmic structure; Minkowski-functional analyses in joint galaxy-clustering and weak-lensing settings; marked-correlation studies of modified gravity and of shot-noise-corrected environmental marks; density-split clustering constraints from the BOSS CMASS sample; the Quijote simulation suite; the CAMELS simulation suite and its documentation; information-theoretic treatments of the cosmic web based on tidal eigenvalues, morphology, and multifractal information; persistent-homology studies of massive-neutrino signatures; DESI Data Release 1; SDSS Data Release 19; and the Euclid Quick Data Release 2 and release timeline. **No title, year, journal, volume, or identifier has been reconstructed from memory.** Full bibliographic records must be attached before external circulation.

---

**Cross-references.** Paper 7 (IF geometry and gravitational slip) · Paper 8 (galactic acceleration scale; see the archived SPARC kill in `SCOREBOARD.md`) · Paper 9 (expansion and growth from one \(b(z)\)) · Paper 11 (preregistered Euclid forecast) · Paper 15 (`canon/papers/P15-falsification-of-universality.md`, the IF-H1 falsification — agency branch, disjoint from this one) · `canon/00-foundations/` (three-ledger discipline) · `canon/30-meaning/` (interpretive layer; nothing in this paper depends on it) · `SCOREBOARD.md` (kill log).
