# IF Cosmology
## A Joint Expansion–Growth Consistency Test Without Independent Dark Matter or Dark Energy

**Author:** Phuc Vinh Truong
**Series:** IF Theory Canonical Papers
**Paper:** 9
**Layer:** SCIENCE
**Status:** canonical revision of extracted draft (2026-07-18)
**Supersedes:** canon/extracted/paper-09-extracted.md

---

> ## Status after 2026-07-18
>
> **The IF-H1 falsification does not touch this paper.** IF-H1 died in the agency
> lab — the claim that a dimensionless combination evaluated at competitive
> break-even is substrate-independent across a lattice forager, a linear-Gaussian
> controller, and a run-and-tumble swimmer (`canon/papers/P15-falsification-of-universality.md`).
> Different branch, different observables, different falsifiers. Nothing in
> \(\eta^*\), \(\Upsilon_{\mathrm{IF}}\), or \(\Theta^*\) enters the cosmological
> state \(b(a)\), the consistency lock, or any hypothesis CG-H1 through CG-H11.
> Stated plainly and without defensiveness: this branch is unaffected, and it is
> also not thereby vindicated.
>
> **What the Founding Panel said about this branch stands.** Cosmology runs
> **parallel-under-discipline**; the program's *public identity* is agency
> thermodynamics, and the cosmology branch is firewalled so that its failure
> cannot be read back onto the agency results. Panel scoring: **plausibility
> 3/10, stakes 10/10**. As of this date the branch is **unpreregistered and
> untested** — not one notebook in §35 has been run against data. Preregistration
> is the only currency this branch has; it has not yet spent any.
>
> **This paper's sharpest blade is the Noether gate.** \(w_{\mathrm{IF}}(z)\),
> \(a_{\mathrm{IF}}(z)\), \(\mu(k,z)\), and \(\eta(k,z)\) must all descend from
> ONE \(b(z)\). Any fit that lets the background sector and the perturbation
> sector carry separate parameters is the forbidden state `SECTOR_SPLIT_FIT` and
> falsifies the unification outright — not "weakens", falsifies. The panel ranked
> "expansion–growth consistency: one \(b(z)\) or death" as the single cleanest
> kill-test the unification possesses. §16, §33 (CG-H6), and §34.2 are the
> operational form of that gate.
>
> **The Feynman gate is unmet.** No pipeline in this program has yet reproduced
> Planck or DESI posteriors. Until Notebooks 09A–09G pass, **no IF-model
> cosmology plot is licensed** — not for a paper, not for a talk, not for a
> figure in a repository README. Reproduce before invent.
>
> **`NOVELTY_INFLATION` guard.** The claim "structure entropy causes cosmic
> expansion" is **already retired** in the SCOREBOARD kill log (2026-07),
> defeated by prior art including 2026 entropic-backreaction work. It was
> replaced by the narrower and still-live claim of a *fixed cross-scale
> relation* — one state producing galactic, growth, lensing, and expansion
> observables together. This paper does **not** re-assert the retired novelty,
> and no section of it should be read as doing so. Where coarse-grained
> structural disorder appears below, it appears as a *diagnostic quantity*, never
> as a proposed cause of expansion.
>
> **`COMMANDED_EXPANSION` guard.** No board-growth or capacity-growth picture
> (\(I_N \to I_{N+k}\)) appears in this paper, and none may be imported into it.
> Such a picture is a toy unless derived from the covariant action of Paper 7; if
> a future draft uses one, it must be explicitly labeled a toy and excluded from
> every cosmological inference.
>
> **Ledger discipline.** Three ledgers — energy, thermodynamic entropy, Shannon
> information — are never merged. A cosmology paper is the highest-risk venue for
> a bare "entropy", so every occurrence below is named. Bits are never added to
> joules.

---

## Abstract

The IF Unified Geometry Hypothesis proposes that dark-matter-like attraction and dark-energy-like expansion arise from different regimes of one dynamical geometric sector, rather than from an independent collisionless dark-matter particle fluid plus a separate cosmological constant or dark-energy component. The decisive test is not whether an IF model can fit the cosmic expansion history and structure growth simultaneously after unrestricted adjustment. It is whether the IF state inferred from **one** sector **predicts** the other.

This paper defines the **IF Expansion–Growth Consistency Test**. Let \(b(a)\) denote the homogeneous state of the IF sector, and let one covariant action determine its background density, effective pressure, clustering response, gravitational slip, sound speed, and stability. The same state must generate

\[
\left\{ H(a),\, D_A(a),\, D_L(a),\, D(a),\, f\sigma_8(a),\, P(k,a),\, \Phi(k,a)+\Psi(k,a) \right\}.
\]

The central consistency requirement is

\[
\boxed{\; b_{\mathrm{expansion}}(a) = b_{\mathrm{growth}}(a) = b_{\mathrm{lensing}}(a) \;}
\]

within the joint uncertainty implied by one shared parameter vector.

Three deliberately separated inference routes are proposed. **Expansion-to-growth:** fit baryon acoustic oscillations, supernova distances, and background CMB information; then predict growth and lensing with no new IF freedom. **Growth-to-expansion:** fit redshift-space distortions, full-shape clustering, cosmic shear, and CMB lensing; then predict distances and expansion. **Early-to-late:** fit the primordial and recombination-era observables; then predict both late expansion and low-redshift structure.

The standard cosmological model is the required baseline. The Planck collaboration's final full-mission analysis found that six-parameter spatially flat \(\Lambda\)CDM provides an excellent description of the CMB data. DESI Data Release 2 measured baryon acoustic oscillations using more than fourteen million galaxy and quasar tracers and released posterior chains and best-fit cosmology products publicly in October 2025. DESI's combined analyses reported that flat \(\Lambda\)CDM remains a good description of the BAO measurements, while combinations with CMB and supernova datasets can prefer a time-varying dark-energy parameterization, with the significance depending materially on the supernova compilation used.

The IF analysis will initially use public compressed likelihoods and chains rather than raw images or spectra. Expansion inputs include DESI DR2 BAO, the Pantheon+ supernova compilation, and Planck products. Growth inputs include DESI clustering and redshift-space information, weak-lensing measurements such as the Dark Energy Survey Year 3 cosmic-shear analysis, and Planck lensing. Pantheon+ contains 1,701 light curves from 1,550 distinct Type Ia supernovae spanning \(0.001<z<2.26\). The DES Year 3 cosmic-shear analysis used more than one hundred million source galaxies and demonstrated both the statistical power of weak lensing and its sensitivity to intrinsic-alignment, baryonic-feedback, and nonlinear-modeling assumptions.

The principal falsifier is **parameter splitting**. Let \(\theta_E\) be the IF parameters inferred from expansion and \(\theta_G\) those inferred from growth. The unified hypothesis requires

\[
\boxed{\; \theta_E = \theta_G. \;}
\]

If independent parameter vectors, functions, initial conditions, or transition histories are needed, IF Theory has not unified the dark sector. A model that reproduces the data only by behaving exactly like cold dark matter at early times and independently like arbitrary dark energy at late times may remain phenomenologically viable, but its proposed informational unification has failed unless the connection between those regimes is derived and predictive.

---

## Keywords

Cosmology; cosmic expansion; structure growth; modified gravity; dark matter; dark energy; DESI; Planck; supernovae; weak lensing; redshift-space distortions; consistency tests; IF Theory.

---

# 1. Introduction

Cosmological expansion and cosmic structure growth are governed by related but observationally distinguishable physics.

The expansion history determines how the scale factor evolves, the distance–redshift relation, the age of the universe, and the volume associated with a redshift interval. Structure growth determines how primordial density perturbations become galaxies, filaments, voids, and clusters; how rapidly matter falls into gravitational potentials; how much gravitational lensing those potentials produce; and how clustering changes with time and scale.

In general relativity with specified matter components, expansion and growth are linked: once \(H(a)\) and the gravitating contents are known, the linear growth history is highly constrained. Modified-gravity and unified-dark-sector models break that link. Two theories can generate nearly identical distances while producing different \(f\sigma_8(z)\), \(P(k,z)\), and \(\Phi+\Psi\). This is precisely why fitting the expansion history alone cannot establish a theory of the dark sector.

The IF proposal is unusually restrictive: it seeks to replace both independent dark components with one geometric state. The appropriate question is therefore

\[
\boxed{ \text{Does one IF state inferred from cosmic distances correctly predict the formation and lensing of structure?} }
\]

and, equally important, the reverse

\[
\boxed{ \text{Does the IF state inferred from structure correctly predict the expansion history?} }
\]

If the two answers disagree, the central unification claim fails. That symmetry is the entire content of the Noether gate as it applies at cosmological scale.

---

# 2. Scientific Scope

Paper 9 tests homogeneous expansion; linear cosmological perturbations; mildly nonlinear clustering through validated compressed products; weak gravitational lensing; CMB background and lensing constraints; and cross-sector parameter consistency.

It does **not** yet provide a complete nonlinear galaxy-formation simulation, a definitive cluster-merger test, a quantum origin for the IF field, a derivation of primordial perturbations from first principles, a solution to the vacuum-energy problem, or a final Euclid analysis.

Paper 9 requires that the selected IF model already possess: a covariant action, or a clearly labeled effective phenomenological closure; a general-relativistic limit; stable scalar and tensor perturbations; a defined early-universe clustering mode; no independent cold-dark-matter fluid in the minimal test; and no independent cosmological constant in the minimal test.

A phenomenological model may be used to determine whether the unification is observationally plausible before the full action is completed. Such a model must never be mistaken for a fundamental derivation, and any paper or figure that blurs that distinction is in violation of the layer discipline that governs this repository.

---

# 3. Observational Foundation

## 3.1 Planck

The final Planck release provides temperature, polarization, lensing, likelihood, map, and cosmological-parameter products from the full mission. The collaboration found that the six-parameter flat \(\Lambda\)CDM model gives an excellent description of the CMB and tightly constrains departures from it. The Planck Legacy Archive makes the public mission products available for reproducible analysis.

Planck constrains the baryon density; the effective early clustering density; primordial fluctuation amplitude and tilt; the angular acoustic scale; the reionization optical depth; CMB lensing; and the integrated expansion history to recombination.

A no-particle-dark-matter IF theory must reproduce the physical effects conventionally attributed to the Planck-inferred cold-dark-matter density. This is not negotiable and not partially satisfiable.

## 3.2 DESI DR2 BAO

DESI DR2 reported BAO measurements from more than fourteen million galaxies and quasars over three years of observations, constraining transverse and radial distance combinations across redshift. DESI publicly released its DR2 cosmological posterior chains and posterior-maximizing parameter products in October 2025.

DESI DR2 found the BAO data well described by flat \(\Lambda\)CDM. When combined with CMB and supernova data, an evolving \(w_0w_a\) parameterization can improve the fit, with the reported preference depending materially on which supernova compilation is included. The IF program must therefore compare against **both** \(\Lambda\)CDM and \(w_0w_a\)CDM, and must report against both regardless of which is less flattering.

## 3.3 Type Ia supernovae

The Pantheon+ compilation provides 1,701 light curves from 1,550 distinct Type Ia supernovae extending from \(z=0.001\) to \(z=2.26\), with a covariance treatment for calibration and other systematic uncertainties that improves substantially upon the original Pantheon sample.

Supernovae constrain relative luminosity distances \(D_L(z)\). They do not alone provide an absolute distance scale without calibration or an external anchor.

## 3.4 Full-shape clustering and redshift-space distortions

BAO primarily constrains geometric distances. Full-shape galaxy clustering contains additional information about the broadband power-spectrum shape, the Alcock–Paczyński effect, redshift-space distortions, the fluctuation amplitude, and growth.

Recent analyses combine DESI DR1 full-shape information with DESI DR2 BAO while treating covariance between the releases, providing constraints on \(\Omega_m\), \(H_0\), \(\sigma_8\), and extensions beyond \(\Lambda\)CDM. The primary IF analysis will use validated public likelihoods or compressed measurements rather than reconstructing the entire full-shape pipeline initially.

## 3.5 Weak gravitational lensing

Cosmic shear measures coherent distortions of background-galaxy images caused by intervening gravitational potentials. The DES Year 3 cosmic-shear analysis used over one hundred million source galaxies and measured the low-redshift clustering-amplitude combination

\[
S_8 = \sigma_8 \sqrt{\frac{\Omega_m}{0.3}}.
\]

That analysis also demonstrated that intrinsic alignments, nonlinear matter modeling, and baryonic physics must be propagated carefully rather than absorbed into a single nuisance term.

Weak lensing is indispensable because it tests \(\Phi+\Psi\), whereas nonrelativistic galaxy motion responds predominantly to \(\Psi\). A unified sector that gets one right and the other wrong is detected here and nowhere else.

## 3.6 Euclid

ESA's current timeline lists a **DR1-Foundation** release for November 2026. The relevant core cosmological products therefore remain future or staged inputs to this paper's final preregistered test. Paper 9 will construct and **freeze** an IF forecast before using those products. A forecast written after the products open is a `RETROFIT_FORECAST`, not a prediction.

---

# 4. Standard Cosmological Baseline

The baseline is spatially flat \(\Lambda\)CDM, with background expansion

\[
\boxed{ H^2(a) = H_0^2 \left[ \Omega_r a^{-4} + \Omega_m a^{-3} + \Omega_\Lambda \right]. }
\]

The matter density is \(\Omega_m = \Omega_b + \Omega_c + \Omega_\nu\), where \(\Omega_b\) is baryonic matter, \(\Omega_c\) is cold dark matter, and \(\Omega_\nu\) is the massive-neutrino contribution.

Linear subhorizon matter growth approximately obeys

\[
\boxed{ D'' + \left( 2 + \frac{H'}{H} \right) D' - \frac{3}{2}\,\Omega_m(a)\,D = 0, }
\]

where primes denote derivatives with respect to \(\ln a\). The logarithmic growth rate is

\[
\boxed{ f(a) = \frac{d\ln D}{d\ln a}, }
\]

and redshift-space measurements often constrain the combination

\[
\boxed{ f\sigma_8(z). }
\]

The IF model must reproduce or improve upon the baseline's **joint** predictive performance — not its performance in a favorably chosen sector.

---

# 5. The IF Cosmological State

Let \(b(a)\) be the homogeneous IF state. It may represent an order parameter, a covariant field background, a normalized accessible-capacity state, or a collective geometric configuration. Its physical meaning must eventually be derived from Paper 7's action; until then the word "informational" attached to it is interpretive, not established (see §43).

For phenomenological testing, define

\[
\boxed{ \frac{db}{d\ln a} = -\Gamma\!\left( b, a; \theta_b \right), }
\]

with initial condition \(b(a_i) = b_i\).

The **same** \(b(a)\) must determine the IF density; the IF pressure; the effective gravitational strength; the gravitational slip; the sound speed; the anisotropic stress; and the galactic acceleration scale. There is no permitted second state. A second state is `SECTOR_SPLIT_FIT` under another name.

---

# 6. Background Expansion

Define IF energy density and pressure \(\rho_{\mathrm{IF}}(b,a)\) and \(p_{\mathrm{IF}}(b,a)\). The background equations are

\[
\boxed{ 3M_{\mathrm{Pl}}^2 H^2 = \rho_r + \rho_b + \rho_{\mathrm{IF}}, }
\]

\[
\boxed{ -2M_{\mathrm{Pl}}^2 \dot H = \frac{4}{3}\rho_r + \rho_b + \rho_{\mathrm{IF}} + p_{\mathrm{IF}}. }
\]

No independent cold-dark-matter density appears in the minimal model. No independent cosmological-constant density appears in the minimal model.

The effective IF equation of state is

\[
\boxed{ w_{\mathrm{IF}}(a) = \frac{p_{\mathrm{IF}}(a)}{\rho_{\mathrm{IF}}(a)}. }
\]

The IF sector must behave differently across epochs. A viable approximate sequence is

\[
\boxed{
\begin{aligned}
w_{\mathrm{IF}}(a\ll1) &\approx 0,\\
c_{s,\mathrm{IF}}^2(a\ll1) &\ll 1,\\
w_{\mathrm{IF}}(a\sim1) &< -\tfrac{1}{3}.
\end{aligned}
}
\]

The first regime provides dark-matter-like clustering; the third provides accelerated expansion. **The transition must arise from one dynamical law**, not from a switching function chosen after inspecting the data (§34.6).

---

# 7. Linear Perturbations

Work in Newtonian gauge:

\[
ds^2 = -(1+2\Psi)\,dt^2 + a^2(t)\,(1-2\Phi)\,d\mathbf{x}^2.
\]

The modified Poisson equation is parameterized as

\[
\boxed{ -k^2 \Psi = 4\pi G a^2\, \mu(k,a)\, \rho_b \Delta_b. }
\]

The gravitational-slip parameter is

\[
\boxed{ \eta(k,a) = \frac{\Phi}{\Psi}, }
\]

and the lensing-response parameter is

\[
\boxed{ \Sigma(k,a) = \frac{\mu(k,a)\left[1 + \eta(k,a)\right]}{2}. }
\]

In the final IF model, \(\mu\), \(\eta\), and \(\Sigma\) must be **derived** from \(b(a)\) and the common action. They may not be reconstructed independently. Independent reconstruction is the operational signature of the forbidden `SECTOR_SPLIT_FIT`.

---

# 8. Baryonic Growth

Because the minimal model contains no particle cold dark matter, baryonic perturbations respond to the IF-modified potential. A schematic baryonic growth equation is

\[
\boxed{ D_b'' + \left( 2 + \frac{H'}{H} \right) D_b' - \frac{3}{2}\,\Omega_b(a)\,\mu_{\mathrm{eff}}(k,a)\, D_b = S_{\mathrm{IF}}(k,a), }
\]

where \(\mu_{\mathrm{eff}}\) modifies the gravitational response and \(S_{\mathrm{IF}}\) represents independently evolving IF perturbations.

A purely algebraic enhancement of baryonic gravity after recombination is unlikely to reproduce the early gravitational potentials required by the CMB. The IF perturbation \(\delta_{\mathrm{IF}}\) must therefore possess its own stable evolution, schematically

\[
\boxed{ \delta_{\mathrm{IF}}'' + A(k,a)\,\delta_{\mathrm{IF}}' + B(k,a)\,\delta_{\mathrm{IF}} = C(k,a)\,\delta_b. }
\]

The coefficients \(A\), \(B\), \(C\) must be derived from the covariant theory, not fitted per-epoch.

---

# 9. Early Dark-Matter-Like Regime

The IF sector must provide an early clustering mode satisfying approximately \(w_{\mathrm{IF}} \approx 0\), \(c_{s,\mathrm{IF}}^2 \ll 1\), and \(|\pi_{\mathrm{IF}}|\) sufficiently constrained, where \(\pi_{\mathrm{IF}}\) is the anisotropic stress.

The mode must begin from consistent primordial initial conditions; survive the radiation era; source metric potentials; influence baryon acoustic oscillations; produce the correct acoustic-peak structure; and generate the later matter distribution.

If this mode is observationally indistinguishable from cold dark matter, then IF Theory has changed the ontology but not the phenomenology. That may still be scientifically legitimate **if** the field predicts different behavior elsewhere — the galaxy-scale acceleration link of §19.5 is the designated elsewhere. It is not, by itself, sufficient evidence for a fundamentally new dark-sector law.

---

# 10. Late Accelerating Regime

Late acceleration requires

\[
\rho_{\mathrm{IF}} + 3 p_{\mathrm{IF}} < 0.
\]

The same field that clustered earlier must transition into, or reveal, a negative-pressure homogeneous regime. Candidate mechanisms include a kinetic-state transition, condensation, potential domination, derivative self-interaction, background–perturbation decoupling, and scale-dependent phase behavior.

The mechanism must not destroy the earlier successful perturbations. It must also avoid ghosts, negative sound-speed squared, singular crossings, strong coupling, and excessive late integrated Sachs–Wolfe contributions.

---

# 11. The IF Consistency Lock

Define the shared IF parameter vector

\[
\theta = \left[ \theta_b,\, \theta_{\mathrm{kin}},\, \theta_{\mathrm{trans}},\, \theta_{\mathrm{pert}},\, b_i \right].
\]

The defining map is

\[
\boxed{ \mathcal{O}_{\mathrm{cosmo}} = \mathfrak{F}\left[\theta\right], }
\]

where

\[
\mathcal{O}_{\mathrm{cosmo}} = \left\{ H,\, D_A,\, D_L,\, r_d,\, D_b,\, f\sigma_8,\, P,\, \mu,\, \eta,\, \Sigma,\, C_\ell^{TT},\, C_\ell^{TE},\, C_\ell^{EE},\, C_\ell^{\phi\phi} \right\}.
\]

**No subset receives an independently fitted IF function.** This single sentence is the Noether gate at cosmological scale; §16 measures compliance and §33 (CG-H6) states its falsifier.

---

# 12. Expansion-Inferred State

Let the expansion data be \(\mathcal{D}_E = \{\text{BAO},\, \text{SNe},\, \text{background CMB}\}\), and infer \(P(\theta_E \mid \mathcal{D}_E)\).

For every posterior sample \(\theta_E^{(s)}\), calculate predicted growth \(\mathcal{O}_G^{(s)} = \mathfrak{F}_G[\theta_E^{(s)}]\). The expansion-to-growth posterior predictive distribution is

\[
\boxed{ P\left( \mathcal{O}_G \mid \mathcal{D}_E \right) = \int P\left( \mathcal{O}_G \mid \theta \right) P\left( \theta \mid \mathcal{D}_E \right) d\theta. }
\]

No growth data may be used to select the background posterior or to adjust the perturbation response after this distribution is generated.

---

# 13. Growth-Inferred State

Let the growth data be \(\mathcal{D}_G = \{\text{RSD},\, \text{full-shape clustering},\, \text{cosmic shear},\, \text{CMB lensing}\}\), and infer \(P(\theta_G \mid \mathcal{D}_G)\). Predict expansion:

\[
\boxed{ P\left( \mathcal{O}_E \mid \mathcal{D}_G \right) = \int P\left( \mathcal{O}_E \mid \theta \right) P\left( \theta \mid \mathcal{D}_G \right) d\theta. }
\]

Again, expansion data are withheld during model selection.

---

# 14. Early-Inferred State

Let the early-universe data be

\[
\mathcal{D}_{\mathrm{early}} = \left\{ \text{CMB TT},\, \text{CMB TE},\, \text{CMB EE},\, \text{CMB lensing},\, \text{BBN prior} \right\},
\]

and infer \(P(\theta_{\mathrm{early}} \mid \mathcal{D}_{\mathrm{early}})\). Then predict DESI BAO; supernova distances; low-redshift growth; weak lensing; and the local galactic acceleration scale through Paper 8.

This is the strongest route because it spans the greatest temporal baseline, and correspondingly the least forgiving.

---

# 15. State-Reconstruction Consistency

For each inference route, reconstruct the IF background \(b_E(a)\), \(b_G(a)\), \(b_{\mathrm{early}}(a)\), and define the differences

\[
\Delta b_{EG}(a) = b_E(a) - b_G(a), \qquad \Delta b_{E\mathrm{early}}(a) = b_E(a) - b_{\mathrm{early}}(a).
\]

With covariance \(C_b(a,a')\), define a functional inconsistency statistic

\[
\boxed{ Q_b = \int d\ln a \int d\ln a' \; \Delta b(a)\, C_b^{-1}(a,a')\, \Delta b(a'). }
\]

The implementation may use a discretized redshift basis. A large \(Q_b\) indicates that different observables require incompatible IF histories. The threshold is calibrated on synthetic universes generated from the IF model itself (§30), never chosen after seeing the real value.

---

# 16. Parameter-Splitting Test

Fit a split model with \(\theta_E\) for expansion and \(\theta_G\) for growth, and compare against the unified model \(\theta_E = \theta_G\). The split likelihood is

\[
\mathcal{L}_{\mathrm{split}} = \mathcal{L}_E(\theta_E)\, \mathcal{L}_G(\theta_G),
\]

and the unified likelihood is

\[
\mathcal{L}_{\mathrm{unified}} = \mathcal{L}_E(\theta)\, \mathcal{L}_G(\theta).
\]

Compare through Bayes factors; predictive information criteria; posterior parameter differences; and simulation-calibrated likelihood-ratio statistics.

The central requirement is **not** that the split model never fits better — it has more freedom, so it usually will. The question is whether the data *demand* that extra freedom strongly enough to reject unification. A decisive demand is the `SECTOR_SPLIT_FIT` verdict, and it kills the hypothesis rather than downgrading it.

---

# 17. Cross-Predictive Scores

## 17.1 Expansion-to-growth score

\[
\boxed{ S_{E\rightarrow G} = \ln P\left( \mathcal{D}_G \mid \mathcal{D}_E,\, \mathrm{IF} \right). }
\]

## 17.2 Growth-to-expansion score

\[
\boxed{ S_{G\rightarrow E} = \ln P\left( \mathcal{D}_E \mid \mathcal{D}_G,\, \mathrm{IF} \right). }
\]

## 17.3 Early-to-late score

\[
\boxed{ S_{\mathrm{early}\rightarrow\mathrm{late}} = \ln P\left( \mathcal{D}_{\mathrm{late}} \mid \mathcal{D}_{\mathrm{early}},\, \mathrm{IF} \right). }
\]

Each score is compared with \(\Lambda\)CDM; \(w_0w_a\)CDM; a phenomenological modified-gravity model; and a split IF model.

---

# 18. Null-Test Function

In general relativity with a known matter density, expansion predicts growth. Construct an IF-specific null function

\[
\boxed{ \mathcal{N}_{\mathrm{IF}}(k,a) = \mu_{\mathrm{growth}}(k,a) - \mu_{\mathrm{expansion}}(k,a;\theta_E). }
\]

Under IF consistency, \(\mathcal{N}_{\mathrm{IF}}(k,a) = 0\) within uncertainty. Similarly define a lensing null

\[
\boxed{ \mathcal{N}_{\mathrm{lens}}(k,a) = \Sigma_{\mathrm{observed}}(k,a) - \Sigma_{\mathrm{IF}}\left[ b_E(a), k \right]. }
\]

A statistically significant scale- or redshift-dependent departure rejects the selected IF closure.

---

# 19. IF-C0 Minimal Phenomenological Closure

Before a final covariant action is implemented, use a deliberately restricted model. IF-C0 is a **diagnostic**, explicitly labeled as such; results from it never license a claim about the fundamental theory.

## 19.1 State evolution

\[
\boxed{ \frac{db}{d\ln a} = -\gamma\, b^n \left( 1 - b \right)^m, }
\]

subject to \(0 \leq b \leq 1\).

## 19.2 Density split

Define the IF density as \(\rho_{\mathrm{IF}}(a) = \rho_{\mathrm{IF},0}\, F_\rho[b(a)]\). The same function must generate early matter-like scaling **and** late acceleration. One possible restricted form is

\[
\boxed{ \rho_{\mathrm{IF}} = \rho_* \left[ b\, a^{-3} + \lambda\left( 1 - b \right) \right], }
\]

but the parameter \(\lambda\) cannot simply reproduce a hidden cosmological constant without dynamical justification (§34.5). This form is a diagnostic, not a preferred final theory.

## 19.3 Effective gravity

\[
\boxed{ \mu(k,a) = 1 + \alpha_\mu\, b(a)\, \frac{1}{1 + \left[k/k_c(a)\right]^2}. }
\]

## 19.4 Slip

\[
\boxed{ \eta(k,a) = 1 + \alpha_\eta\left[ 1 - b(a) \right] \frac{1}{1 + \left[k/k_\eta(a)\right]^2}. }
\]

The parameters \(\alpha_\mu\), \(\alpha_\eta\), \(k_c\), \(k_\eta\) must eventually be related by the action. The phenomenological test allows only a very small frozen family; enlarging it mid-analysis is a parameter split in disguise.

## 19.5 Galactic link

Paper 8 requires

\[
\boxed{ a_{\mathrm{IF}}(a) = a_{\mathrm{IF},0}\, F_a[b(a)]. }
\]

The minimal Hubble lock is

\[
F_a[b(a)] = \frac{H(a)}{H_0}.
\]

The cosmological fit therefore **predicts** the galaxy-scale acceleration evolution. This is where the unification is most exposed, and where the branch inherits its known deficit: the archived log-potential galaxy law lost the fair-rules 175-galaxy SPARC benchmark to both MOND and NFW on \(\chi^2\)/dof and on the parameter-penalized criterion. That kill applies to the archived formulation, not to the present unified-geometry hypothesis, which has no galaxy-scale implementation yet — but it is the honest prior any Phase-3 work inherits.

---

# 20. Baseline Models

| Baseline | Content | What it tests |
|---|---|---|
| 20.1 Flat \(\Lambda\)CDM | Required baseline, standard neutrino treatment | Absolute reference |
| 20.2 \(w_0w_a\)CDM | \(w(a) = w_0 + w_a(1-a)\) | Flexible late expansion with ordinary dark matter and GR |
| 20.3 Phenomenological modified gravity | Independent \(\mu(k,a)\), \(\eta(k,a)\) in a low-dimensional parameterization | Whether the IF consistency lock is too restrictive |
| 20.4 Unified dark fluid | Matter-like → dark-energy-like transition | Whether IF's unification adds anything over a generic unified fluid |
| 20.5 Relativistic MOND-like / scalar–aether comparator | At least one existing relativistic model producing nonstandard galaxy gravity and cosmological clustering | Prior art the IF claim must beat |
| 20.6 Split IF | Separate IF parameters for expansion and growth | The direct alternative to unification |

Baseline 20.5 exists specifically as a `NOVELTY_INFLATION` guard: the IF claim must be measured against the best existing relativistic alternative, not against a strawman.

---

# 21. Data Combination Ladder

The analysis proceeds incrementally.

| Combination | Datasets | Role |
|---|---|---|
| A | BAO only | Geometric expansion, minimal external assumptions |
| B | BAO + supernovae | Relative late-time expansion |
| C | BAO + background CMB | Adds sound-horizon and recombination distance relation |
| D | BAO + CMB + supernovae | Primary expansion fit |
| E | RSD + full shape | Primary clustering-growth fit |
| F | Cosmic shear + CMB lensing | Primary lensing fit |
| G | RSD + full shape + shear + CMB lensing | Growth combination |
| H | Full joint fit | Used **only** after the cross-prediction tests are frozen and reported |

---

# 22. Sound-Horizon Discipline

BAO observations constrain distances relative to the sound horizon \(r_d\), and a model that changes early cosmology changes \(r_d\). The IF analysis must therefore not use a standard-\(\Lambda\)CDM sound horizon while claiming nonstandard pre-recombination physics.

**Route 1 — full early calculation.** Compute \(r_d\) from the IF background and perturbations.

**Route 2 — free-ruler diagnostic.** Treat \(r_d\) as a nuisance parameter for a late-time-only test. Route 2 cannot support any claim that the model explains the CMB or early dark matter.

---

# 23. Primordial Initial Conditions

The IF field introduces possible additional primordial modes. The minimal model begins with one adiabatic mode. Any IF isocurvature mode must be derived, parameterized, constrained, and included in the complexity count.

The theory may not add a free primordial spectrum solely to repair CMB residuals. The primordial parameters include \(A_s\), \(n_s\), and \(\tau_{\mathrm{reio}}\). If the IF mechanism generates them, that derivation belongs to a later paper and may not be assumed here.

---

# 24. Neutrinos

Massive neutrinos affect expansion, CMB lensing, matter-power suppression, and growth. The IF analysis must include a consistent neutrino model across all baselines. Neutrino mass may **not** be adjusted differently in the expansion and growth fits; otherwise apparent IF consistency could be manufactured through nuisance-sector splitting — a `SECTOR_SPLIT_FIT` hiding in the nuisance parameters rather than the IF parameters.

---

# 25. Nonlinear Scales

IF gravity may change nonlinear evolution, so standard nonlinear fitting formulae calibrated on \(\Lambda\)CDM cannot automatically be applied. The analysis uses a scale ladder:

- **Conservative scale cut** — only scales where linear or validated perturbative IF calculations are reliable.
- **Perturbative extension** — effective-field-theory or emulator calculations validated against IF simulations.
- **Nonlinear extension** — dedicated IF \(N\)-body or hydrodynamical simulations.

The primary inference uses conservative scales. Small-scale information is added only after validation, and the ladder fails closed (§35, Notebook 09W).

---

# 26. Galaxy Bias

Observed galaxies trace the underlying matter or effective gravitational field imperfectly, so bias parameters must be included consistently. The IF model may alter halo formation, tracer bias, redshift-space mapping, and velocity fields. A standard bias model may be used initially as a diagnostic, but final claims require testing its validity in IF simulations.

---

# 27. Weak-Lensing Systematics

The weak-lensing analysis must propagate shear calibration; photometric-redshift uncertainty; intrinsic alignments; baryonic feedback; nonlinear modeling; survey masks; and source selection. The DES Year 3 analysis illustrates that these effects are central components of the cosmological likelihood rather than minor corrections.

The IF model cannot interpret every lensing residual as modified gravity. Any residual claimed as an IF signature must survive the full systematics propagation first.

---

# 28. Stability-First Prior

For every proposed parameter point, calculate the scalar kinetic term \(Q_s\), the scalar sound speed squared \(c_s^2\), the tensor kinetic term \(Q_T\), and the tensor speed squared \(c_T^2\). Reject the point if \(Q_s \leq 0\), \(c_s^2 < 0\), \(Q_T \leq 0\), or if the tensor speed violates the required observational limit.

The stability prior is applied **before** observational likelihood evaluation. An unstable best fit is not a viable cosmology, and reporting one as a success is the failure mode of §34.7.

---

# 29. Parameter Identifiability

The IF transition may be poorly constrained if multiple parameter combinations produce similar \(H(a)\). Growth and lensing may break those degeneracies. Before real-data fitting:

1. compute Fisher or sensitivity matrices;
2. run synthetic posterior recovery;
3. identify unconstrained combinations;
4. remove or fix nonidentifiable parameters;
5. report prior-dominated directions.

A theory with ten parameters but only two measured combinations does not possess ten empirically established physical quantities.

---

# 30. Synthetic-Universe Program

Generate mock datasets from \(\Lambda\)CDM; \(w_0w_a\)CDM; IF-C0; split IF; modified gravity with no IF unification; and a unified dark fluid.

For each mock: fit every model; perform cross-prediction; calculate parameter-splitting evidence; measure false rejection; measure false confirmation; and test uncertainty coverage.

The analysis must demonstrate that it can **reject false IF unification** before it is applied to the real universe. A pipeline that has never rejected a false IF universe has not been shown capable of rejecting a real one.

---

# 31. Blinding

Where practical, apply blinding to growth-amplitude calibration; selected covariance elements; posterior parameter displays; and model-label identities. Analysis decisions are frozen before unblinding.

Blinding matters more here than in a typical analysis because IF Theory has a strong desired conclusion and the branch was scored 3/10 for plausibility by its own founding panel. The author's preference is a known systematic and is treated as one.

---

# 32. Preregistered Decision Rules

Precise numerical thresholds will be calibrated using synthetic experiments. The qualitative rules are fixed here.

| Rule | Requirement |
|---|---|
| 1 — Background adequacy | Acceptable posterior predictive behavior for BAO, supernovae, CMB background observables |
| 2 — Growth adequacy | Expansion-calibrated IF predictions acceptable for RSD, full shape, weak lensing, CMB lensing |
| 3 — Reverse prediction | Growth-calibrated IF predictions acceptable for expansion |
| 4 — Parameter equality | The split model must not be decisively required |
| 5 — Stability | The viable posterior remains within the theoretically stable region |
| 6 — Complexity | IF must not obtain comparable fit solely through much greater effective freedom |
| 7 — Early-universe success | The full no-particle-dark-matter claim is withheld unless CMB spectra and the matter transfer function are reproduced |

---

# 33. Core Hypotheses

**CG-H1 — Background-fit hypothesis.** One IF background state reproduces BAO, supernova, and CMB-distance constraints without an independent dark-energy term.
*Falsifier:* a separate late-time component or arbitrary expansion function is required.

**CG-H2 — Early-clustering hypothesis.** The IF perturbation sector replaces the gravitational role of cold dark matter before recombination.
*Falsifier:* the CMB acoustic structure or matter transfer function cannot be reproduced.

**CG-H3 — Expansion-to-growth hypothesis.** Expansion-inferred IF parameters predict observed low-redshift growth.
*Falsifier:* predicted \(f\sigma_8\), \(P(k)\), or lensing is incompatible with held-out measurements.

**CG-H4 — Growth-to-expansion hypothesis.** Growth-inferred IF parameters predict BAO and supernova distances.
*Falsifier:* the predicted expansion history is incompatible with held-out distance data.

**CG-H5 — State-consistency hypothesis.** \(b_E(a) = b_G(a) = b_{\mathrm{early}}(a)\) within simulation-calibrated uncertainty.
*Falsifier:* the reconstructed IF states differ significantly (large \(Q_b\)).

**CG-H6 — Parameter-lock hypothesis.** One parameter vector fits every sector.
*Falsifier:* the data decisively require \(\theta_E \neq \theta_G\). **This is the Noether gate and the branch's cleanest kill-test.**

**CG-H7 — Lensing-slip hypothesis.** The action-derived \(\eta(k,a)\) predicts weak and CMB lensing.
*Falsifier:* an independent slip function is required.

**CG-H8 — Stable-posterior hypothesis.** The observationally supported region is free of ghost and gradient instabilities.
*Falsifier:* only unstable parameter values fit the data.

**CG-H9 — Galaxy–cosmology lock.** The cosmological IF history predicts the acceleration-scale evolution tested in Paper 8.
*Falsifier:* the galaxy and cosmology sectors require incompatible \(a_{\mathrm{IF}}(z)\).

**CG-H10 — Lower-complexity hypothesis.** The IF model explains the combined data with effective complexity competitive with the components it replaces.
*Falsifier:* it needs more independent functions than cold dark matter plus dark energy.

**CG-H11 — Prospective-survey hypothesis.** A frozen IF prediction succeeds on future Euclid or later survey products.
*Falsifier:* the preregistered prediction fails.

None of CG-H1 through CG-H11 is preregistered as of 2026-07-18. Each becomes a scientific claim only at the timestamped commit that freezes it before the corresponding data are touched.

---

# 34. Failure Modes

1. **Joint-fit illusion** — the model fits expansion and growth simultaneously only because both sectors are adjusted together.
2. **Hidden parameter split** — the same symbol carries different effective values in the background and perturbation code.
3. **Sound-horizon borrowing** — the model uses the \(\Lambda\)CDM sound horizon despite changing early cosmology.
4. **Dark matter by another name** — the IF perturbation is exactly a freely normalized cold collisionless component with no additional prediction.
5. **Dark energy by another name** — a constant term inside the IF Lagrangian supplies all late acceleration.
6. **Arbitrary transition** — a switching function is chosen after inspecting the desired redshift behavior.
7. **Instability masking** — unstable points are retained because they improve the likelihood.
8. **Nonlinear misuse** — \(\Lambda\)CDM nonlinear corrections are applied outside their validated domain.
9. **Growth-data double counting** — correlated DESI, lensing, or CMB products are treated as independent.
10. **Supernova cherry-picking** — only the supernova compilation giving the strongest desired result is reported.
11. **Prior-volume victory** — Bayesian preference arises mainly from restrictive priors rather than predictive performance.
12. **Parameter nonidentifiability** — broad prior-dominated directions are described as physical measurements.
13. **Posterior predictive omission** — only parameter contours are reported, not failures in observable space.
14. **Tension exploitation** — existing differences among datasets are presented as evidence for IF without demonstrating that IF resolves them consistently.
15. **Euclid hindsight** — the model is modified after future Euclid results are opened.
16. **Novelty inflation** — restating a known entropic-backreaction or unified-dark-fluid result as an IF discovery. The specific retired instance is logged in SCOREBOARD.md; the live claim is the *fixed cross-scale relation*, nothing broader.
17. **Ledger conflation** — describing coarse-grained structural disorder, thermodynamic entropy, and Shannon information with a single word, and thereby appearing to derive an energetic conclusion from an informational premise.

Failure modes 2, 6, and 16 are the three most likely to produce a *false success* that survives casual review. Notebook 09Z exists to hunt them adversarially.

---

# 35. Deterministic Notebook Program

Each notebook declares Prediction · Baseline · Data · Pass criterion · Falsifier before it is run.

| Notebook | Content |
|---|---|
| **09A** — \(\Lambda\)CDM background reproduction | Reproduce \(H(z)\), \(D_A(z)\), \(D_L(z)\), \(r_d\); compare against a trusted cosmology solver |
| **09B** — \(\Lambda\)CDM growth reproduction | Reproduce \(D(z)\), \(f(z)\), \(f\sigma_8(z)\); validate limiting cases and numerical convergence |
| **09C** — DESI DR2 BAO manifest | Download compressed measurements, covariance, posterior chains, best-fit products; save checksums and licenses |
| **09D** — Pantheon+ reproduction | Reproduce the public supernova likelihood and covariance handling |
| **09E** — Planck distance + full-likelihood baselines | Begin with compressed background information, then reproduce selected official chains or likelihood results before adding IF |
| **09F** — Weak-lensing baseline | Reproduce a public DES Year 3 or equivalent conservative-scale likelihood; validate intrinsic-alignment treatment, shear calibration, redshift nuisance parameters, baryonic scale cuts |
| **09G** — Full-shape + RSD baseline | Implement validated compressed DESI clustering products; track covariance with BAO products |
| **09H** — IF-C0 background solver | Integrate \(b(a)\), \(\rho_{\mathrm{IF}}(a)\), \(p_{\mathrm{IF}}(a)\), \(H(a)\) |
| **09I** — IF perturbation solver | Integrate \(\delta_b\), \(\delta_{\mathrm{IF}}\), \(\Phi\), \(\Psi\); calculate \(\mu\), \(\eta\), \(\Sigma\) |
| **09J** — Stability map | Evaluate \(Q_s\), \(c_s^2\), \(Q_T\), \(c_T^2\); produce the allowed prior domain |
| **09K** — Sound-horizon consistency | Calculate \(r_d\) from the IF early background; compare with free-ruler diagnostics |
| **09L** — CLASS/CAMB regression | Reproduce standard TT, TE, EE, lensing, and matter-power spectra before modifying the solver |
| **09M** — IF Boltzmann implementation | Add the selected IF equations; document every code change and equation mapping |
| **09N** — Expansion-only inference | Fit \(\mathcal{D}_E\); save posterior samples before evaluating growth |
| **09O** — Expansion-to-growth prediction | Generate \(P(\mathcal{O}_G \mid \mathcal{D}_E)\); no growth refitting |
| **09P** — Growth-only inference | Fit \(\mathcal{D}_G\); save posterior samples before evaluating expansion |
| **09Q** — Growth-to-expansion prediction | Generate \(P(\mathcal{O}_E \mid \mathcal{D}_G)\) |
| **09R** — Early-to-late prediction | Fit early data; forecast BAO, supernovae, low-redshift growth, lensing, Paper 8 acceleration evolution |
| **09S** — State-reconstruction test | Calculate \(b_E(a)\), \(b_G(a)\), \(b_{\mathrm{early}}(a)\), \(Q_b\) |
| **09T** — Parameter-splitting test | Compare \(\theta_E = \theta_G\) against \(\theta_E \neq \theta_G\) |
| **09U** — Cross-predictive model comparison | Compare IF against \(\Lambda\)CDM, \(w_0w_a\)CDM, modified gravity, unified dark fluid, split IF |
| **09V** — Synthetic recovery | Generate and recover all benchmark universes; measure false-positive and false-negative rates |
| **09W** — Nonlinear scale audit | Determine which scales are safe under the current IF calculation; fail closed by excluding unvalidated scales |
| **09X** — Galaxy–cosmology lock | Send cosmological posterior samples to Paper 8's \(a_{\mathrm{IF}}(z)\); test consistency with galaxy data |
| **09Y** — Euclid frozen forecast | Save predicted bins, covariance assumptions, summary statistics, exclusion thresholds, git commit, environment hash |
| **09Z** — Adversarial audit | A separate agent attempts to prove any IF success results from parameter splitting, hidden \(\Lambda\), dark-matter relabeling, prior choices, sound-horizon borrowing, scale-cut tuning, dataset selection, covariance errors, unstable dynamics, or post-hoc transition design |

Notebooks 09A–09G are the **Feynman gate**. Until every one of them passes, no IF-model cosmology figure exists, licensed or otherwise.

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

The `parameter_split` field is the machine-readable form of the Noether gate: a run that emits `parameter_split: true` and is nonetheless reported as an IF success is a protocol violation, not a result.

---

# 38. Validation Requirements

## 38.1 Dual implementation

Implement the background and linear perturbation equations independently in (1) a readable reference implementation and (2) the modified Boltzmann solver. Compare at randomly selected parameter points.

## 38.2 Constraint residuals

Track the Friedmann constraint; energy conservation; the Einstein constraints; gauge consistency; and initial-condition residuals.

## 38.3 Solver convergence

Vary integration tolerances; time sampling; wavenumber sampling; hierarchy truncation; and interpolation settings.

## 38.4 Standard-limit recovery

As IF modifications vanish, recover ordinary general relativity; standard radiation and baryon evolution; and the declared reference dark-sector limit where applicable. Because the minimal model contains no cold-dark-matter fluid or cosmological constant, the exact route to \(\Lambda\)CDM may require a limiting IF state that becomes phenomenologically equivalent. That equivalence must be made explicit rather than assumed.

---

# 39. Criteria for Success

| Level | Criterion |
|---|---|
| 1 — Background feasibility | A stable IF state reproduces late expansion |
| 2 — Early clustering | The same field reproduces the CMB and linear matter transfer function without particle cold dark matter |
| 3 — Expansion-to-growth prediction | Expansion-calibrated parameters predict growth and lensing |
| 4 — Reverse prediction | Growth-calibrated parameters predict expansion |
| 5 — State consistency | All inference routes reconstruct the same IF history |
| 6 — Galaxy–cosmology closure | The cosmological state predicts Paper 8's galactic acceleration evolution |
| 7 — Competitive model evidence | The unified model competes successfully with \(\Lambda\)CDM after complexity penalties and cross-prediction |
| 8 — Prospective confirmation | A frozen Euclid or later-survey forecast is independently confirmed |

The branch currently stands at **Level 0**: no notebook has been run against data.

---

# 40. What Would Count as a Major Result

A technically important result would be

\[
\boxed{ \text{A stable single-field IF model reproduces both a matter-like early regime and an accelerating late regime.} }
\]

A stronger observational result would be

\[
\boxed{ \text{IF parameters inferred from expansion predict structure growth and lensing without additional functions or recalibration.} }
\]

A field-changing result would be

\[
\boxed{ b_{\mathrm{expansion}}(a) = b_{\mathrm{growth}}(a) = b_{\mathrm{lensing}}(a) = b_{\mathrm{galaxies}}(a) }
\]

across the measured cosmic history.

A result of the highest class would additionally require a distinctive quantitative prediction; publication before the measurement; independent observational confirmation; successful reproduction by other groups; and no comparably simple conventional explanation. The program does not describe its own target in prize terms; the honest shape of the ambition is a durable cross-scale relation, and the honest current state is an unpreregistered branch at 3/10 plausibility.

---

# 41. Relationship to Dark Matter

A successful Paper 9 model would eliminate the need for an independent particle cold-dark-matter fluid **within the tested cosmological model**. It would not prove that no dark-matter particle exists anywhere; an IF field that clusters like dark matter may coexist with other unseen particles.

The scientifically precise conclusion would be:

> The tested observations do not require a separate cold-dark-matter particle component once the IF geometric sector is included.

The stronger ontological statement would require direct-particle searches and additional astrophysical evidence.

---

# 42. Relationship to Dark Energy

A successful model would generate acceleration without an independent cosmological constant or arbitrary dark-energy fluid. It would not automatically solve the vacuum-energy problem; quantum vacuum contributions and radiative stability remain separate theoretical challenges.

The precise conclusion would be:

> Late acceleration emerges dynamically from the same IF sector responsible for early clustering and modified gravity.

---

# 43. Relationship to the Informational Battery

Paper 1 defined an informational battery through physically accessible nonequilibrium capacity, in the energy and thermodynamic-entropy ledgers, with the Shannon-information ledger tracked separately and never summed with them.

Paper 9 has **not** derived \(b(a)\) from that microscopic battery framework. The cosmological connection remains provisional until the theory shows: (1) what is out of equilibrium; (2) what constitutes the accessible capacity, in which ledger; (3) how the state is measured covariantly; (4) what physical process changes \(b(a)\); and (5) why its stress-energy has the required signs and perturbation behavior.

Without that bridge, the word **informational** in "IF cosmology" is an interpretive label, not a physical claim. Naming \(b(a)\) informational while its only defined content is a phenomenological ODE would be `METAPHOR_MATH`. This paper therefore treats \(b(a)\) as a dimensionless dynamical order parameter with a declared estimator route (§15) and a declared falsifier (§33), and defers the informational interpretation to Paper 7's action.

---

# 44. Relationship to Cosmic Structure Information

Paper 10 will test whether measurable information and topology in the cosmic web add predictive content beyond conventional two-point statistics.

Paper 9 must not define its IF state using a statistic measured from late structure and then claim to predict that same structure. Any structure–information coupling must be independently defined; temporally causal; validated in simulations; and tested against null universes.

This is also where `NOVELTY_INFLATION` is most likely to reappear. The retired claim — that coarse-grained structural disorder *causes* cosmic expansion — is not revived by Paper 10 and must not be reintroduced through it. The live claim remains the fixed cross-scale relation between one \(b(a)\) and the observables of §11.

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
8. the stable parameter space does not overlap the good-fit parameter space;
9. the model borrows a \(\Lambda\)CDM sound horizon inconsistently;
10. nonlinear predictions rely on unvalidated standard corrections;
11. effective complexity exceeds the components being replaced;
12. galaxy and cosmology acceleration histories disagree;
13. success depends on one preferred supernova sample without explanation;
14. a future preregistered prediction fails;
15. every failure is addressed by adding a new independent IF function.

Item 15 is the meta-criterion. A theory that answers each falsification with a new free function has stopped being falsifiable, and the correct response is retirement, recorded in the SCOREBOARD kill log.

---

# 46. Conclusion

The IF cosmological hypothesis does not succeed by fitting more datasets with more freedom. It succeeds only through **enforced dependence**. The same state must explain

\[
\boxed{ \text{early gravitational clustering} + \text{galaxy-scale missing gravity} + \text{late accelerated expansion} + \text{lensing}. }
\]

The central consistency condition is

\[
\boxed{ b_{\mathrm{expansion}}(a) = b_{\mathrm{growth}}(a) = b_{\mathrm{lensing}}(a), }
\]

and the central parameter condition is

\[
\boxed{ \theta_E = \theta_G. }
\]

The decisive experimental sequence is

\[
\boxed{ \text{fit expansion} \rightarrow \text{predict growth} }
\]

followed by

\[
\boxed{ \text{fit growth} \rightarrow \text{predict expansion}, }
\]

with the stronger temporal test being

\[
\boxed{ \text{fit the early universe} \rightarrow \text{predict the late universe}. }
\]

If those routes require different IF histories, the proposed unified geometry has failed and is retired. If they converge on one state, and that state also predicts galaxies and future survey measurements, IF Theory will have moved from a conceptual unification to a serious alternative cosmological framework.

Neither outcome is currently in evidence. As of 2026-07-18 this branch has run zero notebooks against data, holds no preregistration, carries a 3/10 plausibility score from its own founding panel, and inherits a known galaxy-scale deficit from the archived SPARC benchmark. It runs parallel-under-discipline and is firewalled from the program's agency-thermodynamics results, whose falsification of IF-H1 neither harms nor helps it.

The next paper asks whether the cosmic web contains a measurable multiscale informational signature that adds predictive content beyond ordinary clustering statistics:

\[
\boxed{ \textit{Information and Topology in the Cosmic Web: Testing IF Statistics Beyond the Two-Point Function.} }
\]

---

# References

Attributions below name authors or collaborations only. Where the source draft's citation was unrecoverable, the result is described generically rather than given a fabricated title, year, or identifier.

1. **Planck Collaboration**, final full-mission cosmological-parameter analysis — six-parameter flat \(\Lambda\)CDM gives an excellent description of the CMB observations.
2. **DESI Collaboration**, DR2 baryon-acoustic-oscillation measurements and cosmological constraints — more than fourteen million galaxies and quasars over three years of observations.
3. **DESI Collaboration**, DR2 cosmology chains and data-product release (October 2025) — MCMC chains and posterior-maximizing cosmological products.
4. **Brout et al.**, the Pantheon+ analysis — 1,701 light curves from 1,550 distinct Type Ia supernovae over \(0.001<z<2.26\).
5. **Secco et al.**, Dark Energy Survey Year 3 cosmic-shear cosmology and robustness to modeling uncertainty — more than one hundred million source galaxies.
6. **Forero-Sánchez et al.**, joint DESI DR1 full-shape and DR2 BAO analysis — correlated full-shape and BAO measurements through a compressed joint treatment.
7. **ESA**, Planck publications and the Planck Legacy Archive — official public mission products.
8. **ESA**, Euclid mission timeline — DR1-Foundation listed for November 2026.

---

**Cross-references:** `canon/papers/P00-scope-definitions-falsification.md` (falsification protocol) · `canon/papers/P01-informational-battery.md` (accessible capacity, §43) · `canon/papers/P04-expansion-complexity-window.md` · `canon/papers/P15-falsification-of-universality.md` (IF-H1 kill; different branch) · `canon/20-cosmology/` · `SCOREBOARD.md` (kill log, including the retired structure-disorder novelty claim) · Paper 7 (covariant action, pending) · Paper 8 (galactic acceleration scale) · Paper 10 (cosmic-web information).
