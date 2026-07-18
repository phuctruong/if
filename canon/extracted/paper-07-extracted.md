<!-- extracted from ChatGPT (driver) on 2026-07-18 -->

# The IF Unified Geometry Hypothesis  
## A Single Informational Origin for Dark-Matter-Like Attraction and Dark-Energy-Like Expansion

**Author:** Phuc Vinh Truong  
**Series:** IF Theory Working Papers  
**Paper:** 7  
**Date:** July 18, 2026  
**Status:** Speculative covariant research proposal; no empirical confirmation claimed

---

## Abstract

The standard cosmological model describes galactic and cosmological observations using two dominant invisible components: nonbaryonic cold dark matter and dark energy, commonly represented by a cosmological constant. This model successfully fits the cosmic microwave background, large-scale structure, gravitational lensing, baryon acoustic oscillations, and the late-time expansion history. Any proposal eliminating both dark components must reproduce those achievements rather than merely explain galaxy rotation curves or cosmic acceleration separately. Planck’s final analysis found strong agreement between its temperature, polarization, and lensing measurements and the six-parameter spatially flat \(\Lambda\)CDM model. DESI Data Release 2 has since measured baryon acoustic oscillations using more than fourteen million galaxies and quasars, substantially tightening constraints on the expansion history. citeturn716898search0turn716898search2

This paper formulates the **IF Unified Geometry Hypothesis**:

\[
\boxed{
\text{Dark-matter-like attraction and dark-energy-like expansion}
\atop
\text{are two dynamical regimes of one nonequilibrium informational}
\atop
\text{geometry rather than two independent material substances.}
}
\]

The proposal does not deny the observations conventionally attributed to dark matter and dark energy. It denies, provisionally, that their correct interpretation must involve a new collisionless particle species plus an independent vacuum-energy component.

A covariant target class is introduced using the spacetime metric and one scalar-defined IF order field. Its timelike background establishes a local causal foliation, while its spatial and temporal derivatives contribute to gravitational dynamics. The same covariant action must generate:

1. a general-relativistic high-acceleration limit;
2. a modified low-acceleration quasistatic limit for galaxies;
3. correct gravitational lensing;
4. an early cosmological regime capable of generating the gravitational potentials normally supplied by cold dark matter;
5. late-time accelerated expansion without an independently inserted cosmological constant;
6. stable scalar, vector-effective, and tensor perturbations;
7. gravitational waves propagating at the observed speed;
8. one constrained relationship among galactic acceleration, expansion, growth, and lensing.

The central methodological requirement is the **IF consistency lock**:

\[
\boxed{
\left\{
H(a),\,
\mu(k,a),\,
\eta(k,a),\,
a_{\mathrm{IF}}(a),\,
c_T(a)
\right\}
=
\mathfrak D
\left[
\mathcal L_{\mathrm{IF}};\theta
\right].
}
\]

The background expansion \(H\), effective gravitational strength \(\mu\), gravitational slip \(\eta\), galactic acceleration scale \(a_{\mathrm{IF}}\), and tensor propagation speed \(c_T\) must all be derived from one IF Lagrangian and one shared parameter set. They may not be selected independently to fit separate datasets.

The paper defines a broad covariant prototype rather than a finished theory. It specifies the weak-field and cosmological limits, identifies severe theoretical obstacles, states discriminating predictions, and presents a deterministic Jupyter-notebook program. The hypothesis is falsified if its galaxy and cosmological regimes require independent functions, if it cannot reproduce the CMB and matter power spectrum without an effectively dark-matter-like clustering degree of freedom, if it fails lensing or cluster-merger tests, or if its perturbations contain ghosts, gradient instabilities, superluminally unacceptable modes, or an excluded gravitational-wave speed.

---

## Keywords

Modified gravity; dark matter; dark energy; MOND; cosmology; informational geometry; scalar-tensor gravity; gravitational lensing; cosmic acceleration; causal structure; unified dark sector.

---

# 1. Introduction

The words **dark matter** and **dark energy** describe observationally inferred phenomena before they describe established microscopic substances.

Dark-matter evidence includes:

- galactic rotation and velocity-dispersion anomalies;
- galaxy and cluster lensing;
- cluster dynamics;
- the growth and distribution of cosmic structure;
- the acoustic structure of the cosmic microwave background;
- CMB lensing;
- gravitational behavior in merging clusters.

Dark-energy evidence is primarily the observed expansion history and late-time acceleration, constrained through supernova distances, baryon acoustic oscillations, the CMB distance scale, and structure growth.

The standard explanation assigns the two phenomena to different components:

\[
\text{cold dark matter}
+
\text{cosmological constant or dark energy}.
\]

The IF hypothesis explores a different possibility:

\[
\boxed{
\text{one underlying geometric state}
\longrightarrow
\begin{cases}
\text{additional effective attraction in inhomogeneous systems},\\
\text{accelerated expansion in the homogeneous universe}.
\end{cases}
}
\]

This proposition is not new in broad form. Modified-gravity theories, relativistic MOND theories, unified-dark-fluid models, scalar-tensor theories, vector-tensor theories, emergent-gravity proposals, superfluid models, and aether-like theories have all attempted to replace or reinterpret portions of the dark sector.

Bekenstein’s TeVeS, for example, introduced metric, scalar, and vector degrees of freedom to provide a relativistic MOND limit and sufficient gravitational lensing. More recently, Skordis and Złośnik constructed a relativistic modified-gravity theory capable of producing MOND phenomenology in quasistatic galaxies while reproducing important linear CMB and matter-power-spectrum behavior. These examples demonstrate that replacing particle dark matter is not logically impossible, but they also define a high novelty threshold: IF Theory cannot merely reproduce MOND and add a cosmological scalar. citeturn452924academia50turn716898search4

The scientific objective is therefore not:

> Invent a function that fits rotation curves and another function that fits expansion.

It is:

\[
\boxed{
\text{Derive both phenomena from the same physical state and show}
\atop
\text{that the resulting cross-scale relationships survive observations.}
}
\]

---

# 2. Scientific Status and Scope

This paper presents:

- a hypothesis;
- a mathematical target class;
- a phenomenological prototype;
- a list of mandatory limits;
- a computational research program;
- explicit failure conditions.

It does not present:

- a completed covariant theory;
- a fitted cosmological model;
- a reproduction of Planck or DESI results;
- a solution to the Bullet Cluster;
- a proof that dark-matter particles do not exist;
- a proof that vacuum energy is absent;
- a derivation from quantum gravity;
- a validated theory of the formation of the universe.

The phrase **no dark matter or dark energy** will be used in the restricted sense:

> No independent nonbaryonic dark-matter particle fluid and no separately inserted cosmological-constant or dark-energy component.

The IF sector may nevertheless possess dynamical degrees of freedom, stress, perturbations, and effective gravitational energy.

If those degrees of freedom cluster and gravitate like dark matter, they are observationally an **effective dark sector**, even when interpreted as geometry rather than particles.

This semantic honesty is mandatory.

---

# 3. Observational Gauntlet

A unified IF theory must explain all of the following with one parameter system.

## 3.1 Solar-System and laboratory gravity

The theory must recover general relativity or an observationally equivalent screened limit in high-acceleration environments.

It must satisfy constraints on:

- post-Newtonian parameters;
- inverse-square behavior;
- equivalence-principle tests;
- planetary ephemerides;
- gravitational redshift;
- local Lorentz violation.

---

## 3.2 Galaxies

The theory must account for:

- resolved rotation curves;
- low-surface-brightness galaxies;
- dwarf galaxies;
- elliptical-galaxy dynamics;
- the radial acceleration relation;
- the baryonic Tully–Fisher relation;
- environmental effects;
- galaxy–galaxy lensing.

The observed radial acceleration relation closely connects the acceleration inferred from baryons with the total acceleration measured from rotation curves across galaxies of widely differing properties. That relationship is an essential target for any IF weak-field limit. citeturn452924academia49

---

## 3.3 Galaxy clusters

The theory must reproduce:

- cluster velocity dispersions;
- hot-gas temperatures;
- hydrostatic masses;
- strong and weak lensing;
- cluster abundance;
- merging-cluster offsets.

The Bullet Cluster is particularly difficult because its dominant lensing peaks are displaced from the collisional hot gas. The original weak-lensing analysis interpreted this separation as evidence for an additional effectively collisionless gravitating component. citeturn716898search5

---

## 3.4 Early universe

Without particle dark matter, the IF sector must still explain:

- gravitational potential wells before recombination;
- matter–radiation equality;
- the relative heights and phases of CMB acoustic peaks;
- CMB lensing;
- primordial perturbation growth;
- compatibility with baryon-density constraints;
- the later matter power spectrum.

Planck independently constrained baryonic and cold-dark-matter-like contributions within \(\Lambda\)CDM and found a strong overall fit to the final temperature, polarization, and lensing data. An IF theory must reproduce these observables through different dynamics, not dismiss them. citeturn716898search0turn716898search9

---

## 3.5 Late-time expansion

The theory must reproduce:

\[
H(z),\qquad
D_A(z),\qquad
D_L(z),
\]

together with:

- BAO distances;
- supernova distances;
- CMB-calibrated distance information;
- the deceleration-to-acceleration transition;
- cosmic-age constraints.

DESI DR2 provides precise BAO measurements across a broad redshift range using over fourteen million galaxy and quasar tracers. citeturn716898search2

---

## 3.6 Growth and lensing

The same model must predict:

\[
f\sigma_8(z),\qquad
P(k,z),\qquad
C_\ell^{\kappa\kappa},\qquad
\text{cosmic shear},
\]

after being fit to expansion—or predict expansion after being fit to growth.

Background expansion alone is not sufficient because many modified-gravity and dark-energy models are nearly degenerate in \(H(z)\) while differing in growth and lensing.

---

## 3.7 Gravitational waves

The tensor propagation speed must satisfy the severe constraints motivated by the near-coincident detection of GW170817 and its electromagnetic counterpart. Modified-gravity models with order-one deviations in the present-day gravitational-wave speed are strongly excluded in the relevant frequency regime. citeturn599772search12turn599772academia39

The IF sector must also predict:

- tensor damping;
- possible extra polarizations;
- gravitational-wave luminosity distance;
- strong-field stability.

---

# 4. The Central IF Interpretation

The informational battery is not introduced here as a literal reservoir of abstract bits.

The cosmological IF state is provisionally interpreted as:

> A nonequilibrium geometric order parameter describing how the causal substrate stores, redistributes, and relaxes physically accessible organization.

Let the IF state be represented by a covariant field:

\[
\Theta(x).
\]

The field is not human knowledge.

It is a physical state variable whose configuration affects geometry.

Its two principal regimes are:

\[
\boxed{
\begin{aligned}
\text{spatially inhomogeneous/quasistatic IF state}
&\rightarrow
\text{additional effective attraction},\\[4pt]
\text{homogeneous/time-evolving IF state}
&\rightarrow
\text{effective cosmological pressure and expansion}.
\end{aligned}
}
\]

The same field must connect these regimes.

---

# 5. Covariant Target Class

## 5.1 Scalar-defined causal frame

Introduce a scalar IF field:

\[
\Theta(x).
\]

Define:

\[
X
=
-\frac{
g^{\mu\nu}
\nabla_\mu\Theta
\nabla_\nu\Theta
}{
2M^4
}.
\]

Where the gradient is timelike:

\[
X>0.
\]

Define a unit timelike vector derived from the scalar:

\[
\boxed{
u_\mu
=
-\frac{
\nabla_\mu\Theta
}{
M^2\sqrt{2X}
}.
}
\]

This does not introduce an independent vector field. It defines a local causal frame from the IF scalar.

Define the spatial projector:

\[
h_{\mu\nu}
=
g_{\mu\nu}
+
u_\mu u_\nu.
\]

Define the congruence acceleration:

\[
a_\mu
=
u^\nu\nabla_\nu u_\mu.
\]

Define the projected expansion tensor:

\[
K_{\mu\nu}
=
h_\mu^{\ \alpha}
h_\nu^{\ \beta}
\nabla_\alpha u_\beta.
\]

Possible dimensionless invariants include:

\[
\mathcal A
=
\frac{
a_\mu a^\mu
}{
a_\star^2
},
\]

\[
\mathcal K_1
=
\frac{
K_{\mu\nu}K^{\mu\nu}
}{
M_K^2
},
\]

\[
\mathcal K_2
=
\frac{
(\nabla_\mu u^\mu)^2
}{
M_K^2
}.
\]

---

## 5.2 Prototype action

The covariant target class is:

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
S_b[g_{\mu\nu},\Psi_b].
}
\]

Here:

- \(R\) is the Ricci scalar;
- \(\mathcal L_{\mathrm{IF}}\) is one shared IF Lagrangian;
- \(\Psi_b\) represents baryonic and radiation fields;
- all ordinary matter couples universally to the same physical metric \(g_{\mu\nu}\).

Universal matter coupling is imposed to reduce equivalence-principle problems and ensure that lensing and matter dynamics arise from the same metric geometry.

The action contains no separately inserted cold-dark-matter fluid and, in the minimal version, no independent cosmological-constant term.

This action is a broad target class, not a unique theory.

---

## 5.3 Field equations

Variation with respect to the metric yields:

\[
\boxed{
G_{\mu\nu}
=
8\pi G
\left(
T_{\mu\nu}^{b}
+
T_{\mu\nu}^{\mathrm{IF}}
\right).
}
\]

Variation with respect to \(\Theta\) yields:

\[
\boxed{
\mathcal E_\Theta=0.
}
\]

Diffeomorphism invariance requires the combined conservation identity:

\[
\nabla_\mu
\left(
T_b^{\mu\nu}
+
T_{\mathrm{IF}}^{\mu\nu}
\right)
=0.
\]

If ordinary matter is minimally and universally coupled:

\[
\nabla_\mu T_b^{\mu\nu}=0
\]

on its equations of motion, leaving the IF field equation consistent with:

\[
\nabla_\mu T_{\mathrm{IF}}^{\mu\nu}=0.
\]

---

# 6. Why This Is Only a Target Class

The action above contains a free function:

\[
\mathcal L_{\mathrm{IF}}.
\]

A sufficiently flexible function could imitate many phenomena and therefore explain nothing.

Scientific success requires replacing the free functional space with:

- a low-dimensional functional family;
- symmetry or stability principles;
- a derivation from the IF substrate;
- fixed asymptotic limits;
- prospective predictions.

The initial notebook program will therefore use a sequence:

\[
\text{phenomenological closure}
\rightarrow
\text{stable covariant model}
\rightarrow
\text{observational inference}.
\]

Observational fitting must not precede theoretical stability and limiting-case validation.

---

# 7. Quasistatic Galactic Limit

## 7.1 Metric

In the weak-field quasistatic limit:

\[
ds^2
=
-\left(
1+\frac{2\Psi}{c^2}
\right)c^2dt^2
+
a^2
\left(
1-\frac{2\Phi}{c^2}
\right)
d\mathbf x^2.
\]

Nonrelativistic motion responds primarily to:

\[
\Psi.
\]

Weak gravitational lensing responds to the Weyl combination:

\[
\Phi+\Psi.
\]

A viable IF model must predict both from one field solution.

---

## 7.2 Effective quasistatic action

A minimal quasistatic prototype is:

\[
S_{\mathrm{QS}}
=
-\frac{1}{8\pi G}
\int d^3x\,
a_{\mathrm{IF}}^2
\,
\mathcal F
\left(
\frac{
|\nabla\Psi|^2
}{
a_{\mathrm{IF}}^2
},
b
\right)
-
\int d^3x\,\rho_b\Psi,
\]

where:

- \(b\) is the local or cosmological IF background state;
- \(a_{\mathrm{IF}}\) is an acceleration scale derived from that state;
- \(\mathcal F\) is not independent of the covariant Lagrangian.

Variation produces:

\[
\boxed{
\nabla\cdot
\left[
\mu_{\mathrm{IF}}
\left(
\frac{
|\nabla\Psi|
}{
a_{\mathrm{IF}}
},
b
\right)
\nabla\Psi
\right]
=
4\pi G\rho_b.
}
\]

This is structurally similar to aquadratic modified-gravity formulations. That mathematical form is established prior art and is not itself an IF innovation.

---

## 7.3 Required asymptotic limits

At high acceleration:

\[
x
=
\frac{
|\nabla\Psi|
}{
a_{\mathrm{IF}}
}
\gg1,
\]

require:

\[
\boxed{
\mu_{\mathrm{IF}}(x,b)\rightarrow1.
}
\]

This restores the ordinary Poisson equation.

At low acceleration:

\[
x\ll1,
\]

a MOND-like limit would require approximately:

\[
\boxed{
\mu_{\mathrm{IF}}(x,b)\rightarrow x.
}
\]

For an isolated spherical baryonic mass:

\[
g_b=\frac{GM_b}{r^2}.
\]

The deep-IF limit gives:

\[
g^2
\approx
a_{\mathrm{IF}}g_b.
\]

Therefore:

\[
g
\approx
\frac{
\sqrt{GM_ba_{\mathrm{IF}}}
}{
r
}.
\]

For circular motion:

\[
\frac{V_f^2}{r}=g.
\]

Thus:

\[
\boxed{
V_f^4
=
GM_ba_{\mathrm{IF}}.
}
\]

This reproduces a baryonic Tully–Fisher-type scaling. Such scaling is established MOND phenomenology and cannot be claimed as a novel IF prediction by itself. citeturn452924academia52

---

# 8. Galactic Novelty Requirement

IF Theory must do more than recover a MOND-like interpolation law.

It must derive:

1. the value of \(a_{\mathrm{IF}}\);
2. its redshift dependence;
3. its environmental dependence;
4. the relativistic lensing response;
5. the transition to cluster and cosmological behavior.

The strongest galactic IF prediction is:

\[
\boxed{
a_{\mathrm{IF}}
=
a_{\mathrm{IF}}
\left[
b(a),H(a)
\right].
}
\]

The acceleration scale must not be an unrelated universal constant inserted only into the galaxy equation.

---

# 9. The Minimal Hubble-Locked Prototype

Before the full action is derived, define a deliberately restrictive phenomenological model:

\[
\boxed{
\frac{
a_{\mathrm{IF}}(z)
}{
a_{\mathrm{IF}}(0)
}
=
\left[
\frac{
H(z)
}{
H_0
}
\right]^p.
}
\]

The minimal lock is:

\[
p=1.
\]

This gives:

\[
\boxed{
a_{\mathrm{IF}}(z)
=
a_{\mathrm{IF},0}
\frac{H(z)}{H_0}.
}
\]

This is not asserted as a fundamental law.

It is a high-value falsifiable prototype connecting the galaxy and expansion sectors.

If local galaxies determine:

\[
a_{\mathrm{IF},0},
\]

then expansion data determine the predicted galactic acceleration scale at higher redshift.

No additional high-redshift acceleration parameter is allowed.

---

# 10. Cosmological Background

For a homogeneous and isotropic universe:

\[
ds^2
=
-dt^2
+
a^2(t)d\mathbf x^2.
\]

Let:

\[
\Theta=\bar\Theta(t).
\]

Then:

\[
X=\bar X(t),
\qquad
a_\mu=0,
\]

while expansion-related invariants depend on:

\[
H=\frac{\dot a}{a}.
\]

The IF sector contributes effective density and pressure:

\[
\rho_{\mathrm{IF}}
=
-\frac{2}{\sqrt{-g}}
\frac{
\delta S_{\mathrm{IF}}
}{
\delta g^{00}
},
\]

\[
p_{\mathrm{IF}}
=
\frac{1}{3a^2}
\frac{
\delta S_{\mathrm{IF}}
}{
\delta g^{ii}
}.
\]

The Friedmann equations are:

\[
\boxed{
3M_{\mathrm{Pl}}^2H^2
=
\rho_b+\rho_r+\rho_{\mathrm{IF}},
}
\]

\[
\boxed{
-2M_{\mathrm{Pl}}^2\dot H
=
\rho_b+\frac43\rho_r
+
\rho_{\mathrm{IF}}
+
p_{\mathrm{IF}}.
}
\]

Late-time acceleration requires:

\[
\boxed{
\rho_{\mathrm{IF}}
+
3p_{\mathrm{IF}}
<0
}
\]

over the relevant epoch.

The pressure must be derived from the action.

It may not be declared negative merely to fit the data.

---

# 11. The Early-Universe Requirement

Eliminating particle dark matter creates an immediate problem.

Before recombination, baryons are tightly coupled to photons. Ordinary baryons alone do not provide the same nearly pressureless gravitational component that cold dark matter supplies in \(\Lambda\)CDM.

The IF sector must therefore possess an early perturbative regime satisfying approximately:

\[
w_{\mathrm{IF}}^{\mathrm{early}}
\approx0,
\]

\[
c_{s,\mathrm{IF}}^2
\ll1,
\]

and sufficiently small anisotropic stress or a precisely predicted nonzero stress.

Its perturbations must grow and provide gravitational potentials even though the background IF interpretation is geometric rather than particulate.

Thus:

\[
\boxed{
\text{No particle dark matter}
\neq
\text{no dark-matter-like clustering mode}.
}
\]

If the IF field cannot cluster effectively before recombination, it will fail the CMB.

---

# 12. Required Cosmic Regime Transition

The same IF sector must behave approximately as:

\[
\boxed{
\begin{aligned}
\text{early universe:}&
\quad
w_{\mathrm{IF}}\approx0,\quad
c_s^2\ll1,\\[2pt]
\text{galactic quasistatic regime:}&
\quad
\mu_{\mathrm{IF}}(x\ll1)\sim x,\\[2pt]
\text{late homogeneous universe:}&
\quad
w_{\mathrm{IF}}<-\frac13.
\end{aligned}
}
\]

This is the central theoretical difficulty.

The transition must occur through one action without:

- ghosts;
- gradient instabilities;
- singular behavior;
- strong coupling;
- arbitrary switching functions chosen after examining data.

---

# 13. Linear Perturbations

In Newtonian gauge:

\[
ds^2
=
-(1+2\Psi)dt^2
+
a^2(t)(1-2\Phi)d\mathbf x^2.
\]

Parameterize modified gravitational response through:

\[
\boxed{
-k^2\Psi
=
4\pi Ga^2
\mu(k,a)
\rho_b\Delta_b,
}
\]

and gravitational slip:

\[
\boxed{
\eta(k,a)
=
\frac{\Phi}{\Psi}.
}
\]

The lensing potential is:

\[
\Phi+\Psi.
\]

Define:

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

Matter motion probes mainly:

\[
\mu.
\]

Lensing probes mainly:

\[
\Sigma.
\]

The IF theory must derive both from:

\[
\mathcal L_{\mathrm{IF}}.
\]

They may not be independently parameterized in the final model.

---

# 14. The IF Consistency Lock

The defining requirement is:

\[
\boxed{
\left\{
H(a),\,
\mu(k,a),\,
\eta(k,a),\,
a_{\mathrm{IF}}(a),\,
c_T(a)
\right\}
=
\mathfrak D
\left[
\mathcal L_{\mathrm{IF}};\theta
\right].
}
\]

This means:

- galaxy dynamics determine combinations of \(\theta\);
- lensing tests the same combinations;
- expansion constrains the same background;
- structure growth tests the same perturbations;
- gravitational waves test the same kinetic sector.

No independent dark-matter parameters are added for the CMB.

No independent dark-energy function is added for expansion.

No separate lensing potential is added after rotation curves are fit.

If independent functions are required, the unified hypothesis has failed.

---

# 15. Gravitational Lensing

A nonrelativistic modified-Poisson law does not uniquely determine lensing.

In a universal metric theory, photons follow null geodesics of:

\[
g_{\mu\nu}.
\]

The same field configuration producing:

\[
\Psi
\]

must also produce:

\[
\Phi.
\]

The theory must predict:

\[
\eta(k,a)
=
\frac{\Phi}{\Psi}.
\]

A successful galaxy model must therefore predict, without additional halo freedom:

- galaxy–galaxy lensing;
- strong lensing;
- cluster lensing;
- cosmic shear;
- CMB lensing.

Relativistic MOND theories historically introduced additional field structure precisely because a nonrelativistic acceleration law alone does not provide a complete lensing theory. citeturn452924academia50

---

# 16. External-Field and Environmental Effects

A nonlinear gravitational equation may violate the strong equivalence principle and produce sensitivity to an external field.

Let:

\[
g_{\mathrm{int}}
\]

be a galaxy’s internal acceleration and:

\[
g_{\mathrm{ext}}
\]

the external gravitational environment.

The IF response may be:

\[
g_{\mathrm{obs}}
=
\mathcal G_{\mathrm{IF}}
\left(
g_b,
g_{\mathrm{ext}},
b,
\tau_{\mathrm{IF}}
\right).
\]

A purely instantaneous model has:

\[
\tau_{\mathrm{IF}}=0.
\]

A dynamical IF geometry may retain environmental memory:

\[
\tau_{\mathrm{IF}}>0.
\]

This creates a distinctive prediction:

\[
\boxed{
\text{galaxies with similar present environments may differ if their}
\atop
\text{recent environmental histories differ.}
}
\]

Such hysteresis would distinguish a dynamical IF field from a purely algebraic interpolation law.

---

# 17. Cluster-Merger Memory

A field algebraically tied to current baryonic density would generally place its strongest effect near the dominant baryonic gas in a collision.

To explain displaced lensing peaks, the IF state must possess autonomous dynamics.

Introduce an effective relaxation equation for a cluster-scale IF configuration \(Q_{\mathrm{IF}}\):

\[
\boxed{
\tau_Q
u^\mu\nabla_\mu Q_{\mathrm{IF}}
+
Q_{\mathrm{IF}}
=
Q_{\mathrm{eq}}
\left[
T_{\mu\nu}^{b}
\right].
}
\]

When:

\[
\tau_Q\rightarrow0,
\]

the IF field follows baryons instantaneously.

When:

\[
\tau_Q>0,
\]

the field may lag or retain a previous configuration during a merger.

A population-level prediction is:

\[
\boxed{
\Delta x_{\mathrm{lens}}(t)
\sim
\Delta x_0
e^{-t/\tau_Q}
}
\]

only in the simplest relaxation model.

The actual offset must be derived through relativistic field evolution.

If the field persists and moves like an effectively collisionless mass concentration indefinitely, its distinction from dark matter becomes interpretive rather than observational.

The theory must predict a measurable difference.

---

# 18. Gravitational-Wave Sector

Tensor perturbations \(h_{ij}\) obey an equation of the schematic form:

\[
\ddot h_{ij}
+
\left(
3+\nu
\right)H\dot h_{ij}
+
c_T^2
\frac{k^2}{a^2}
h_{ij}
+
m_T^2h_{ij}
=
\Pi_{ij}.
\]

The minimal IF requirements are:

\[
\boxed{
c_T^2=1
}
\]

in the relevant observed regime,

\[
m_T^2\approx0
\]

unless a consistent observationally allowed value is derived, and no unstable tensor kinetic term.

The tensor friction modification:

\[
\nu(a)
\]

may alter gravitational-wave luminosity distances:

\[
D_L^{\mathrm{GW}}
\neq
D_L^{\mathrm{EM}}.
\]

If the covariant action predicts a nonzero deviation, it becomes a prospective standard-siren test.

---

# 19. Stability Conditions

Before observational fitting, the theory must pass a perturbative stability audit.

## 19.1 No ghost

The kinetic coefficient of every propagating mode must satisfy:

\[
\boxed{
Q_s>0,\qquad Q_T>0.
}
\]

A negative kinetic coefficient indicates a ghost-like instability.

---

## 19.2 No gradient instability

The squared propagation speeds must satisfy:

\[
\boxed{
c_s^2\geq0,\qquad c_T^2\geq0.
}
\]

A negative value produces exponentially growing short-wavelength modes.

---

## 19.3 Controlled propagation

Superluminal group velocities do not always have one simple interpretation in effective field theory, but unexplained superluminality, loss of hyperbolicity, or causal inconsistency is unacceptable.

---

## 19.4 No catastrophic tachyon

Slow cosmological instabilities may sometimes be phenomenologically manageable, but growth faster than the relevant dynamical timescale is excluded.

---

## 19.5 No strong-coupling collapse

The cutoff of the effective theory must remain above the scales used in:

- galaxies;
- cosmological perturbations;
- gravitational waves.

Stable effective-field-theory parameterizations are specifically designed to impose no-ghost and no-gradient-instability conditions before data analysis. citeturn599772academia40turn599772academia42

---

# 20. Vacuum-Energy Problem

Removing a cosmological constant from the classical action does not solve the vacuum-energy problem.

Quantum fields generically contribute vacuum terms compatible with the symmetries of the theory.

The IF program must eventually explain why:

- vacuum contributions do not gravitate conventionally;
- they are dynamically neutralized;
- they are absorbed into the IF state;
- or the effective cosmological constant is naturally small.

Simply setting:

\[
\Lambda=0
\]

by hand is not an explanation.

This remains one of the largest unsolved problems in the proposal.

---

# 21. Formation of the Universe

The title “unified geometry” does not yet explain the origin of the universe.

A complete IF formation theory would require:

1. a well-defined initial or boundary state;
2. a quantum or pre-geometric description;
3. a mechanism for classical spacetime emergence;
4. primordial perturbation generation;
5. an explanation of the arrow of time;
6. a transition into the hot early universe;
7. successful nucleosynthesis and recombination.

Paper 7 addresses only the postulated effective gravitational sector after a spacetime description exists.

The Big Bang is therefore not yet derived.

---

# 22. Core Hypotheses

## UG-H1 — Single-action hypothesis

Galactic attraction and cosmic acceleration derive from one covariant IF action and one parameter system.

### Falsifier

Separate unrelated functions are needed for the quasistatic and cosmological regimes.

---

## UG-H2 — General-relativistic limit

The IF corrections become negligible or screened in high-acceleration, high-curvature, and locally tested regimes.

### Falsifier

The model violates established Solar-System, pulsar, laboratory, or gravitational-wave constraints.

---

## UG-H3 — Low-acceleration limit

The same theory produces a baryon-linked enhancement in low-acceleration galaxies.

### Falsifier

Galaxy fits require an independently adjustable IF halo for each system.

---

## UG-H4 — Lensing consistency

The metric potentials derived from the IF action predict lensing consistent with the dynamics fit.

### Falsifier

A separate lensing correction or unseen mass distribution is required.

---

## UG-H5 — Early-clustering hypothesis

The IF sector supplies sufficiently cold and stable gravitational perturbations before recombination without a particle dark-matter fluid.

### Falsifier

The theory cannot reproduce CMB peak structure and the linear matter spectrum.

---

## UG-H6 — Late-acceleration hypothesis

The homogeneous IF state yields late acceleration without an independent cosmological constant.

### Falsifier

Acceleration requires adding a separately tuned vacuum-energy or dark-energy term.

---

## UG-H7 — Expansion–growth consistency

The IF state inferred from expansion predicts structure growth and lensing without additional free functions.

### Falsifier

Expansion and growth prefer incompatible IF histories.

---

## UG-H8 — Acceleration-scale evolution

The galactic acceleration scale evolves according to the cosmological IF background.

The minimal prototype is:

\[
a_{\mathrm{IF}}(z)
=
a_{\mathrm{IF},0}
\frac{H(z)}{H_0}.
\]

### Falsifier

High-redshift galaxy dynamics exclude the predicted evolution.

---

## UG-H9 — Merger-memory hypothesis

Cluster-merger lensing offsets follow a finite IF relaxation law distinguishable from stable collisionless particle halos.

### Falsifier

The required field behaves indistinguishably from permanent collisionless dark matter or cannot reproduce offsets.

---

## UG-H10 — Luminal-tensor hypothesis

The physically viable IF model has:

\[
c_T=1
\]

over observed gravitational-wave frequencies and cosmological backgrounds.

### Falsifier

Its derived tensor speed violates multimessenger bounds.

---

## UG-H11 — Stability hypothesis

The parameter region fitting observations also satisfies:

- no ghosts;
- no gradient instability;
- acceptable causal structure;
- adequate effective-theory cutoff.

### Falsifier

Observational success exists only in an unstable region.

---

## UG-H12 — Lower-complexity hypothesis

The IF model provides comparable or superior predictive performance to \(\Lambda\)CDM without greater effective flexibility.

### Falsifier

The model achieves fits only through more unconstrained functions and parameters.

---

# 23. Distinctive Predictions

## Prediction 1 — Redshift-dependent galactic acceleration scale

Under the minimal Hubble lock:

\[
\frac{
a_{\mathrm{IF}}(z)
}{
a_{\mathrm{IF},0}
}
=
\frac{
H(z)
}{
H_0
}.
\]

This predicts a fixed evolution of:

- the radial acceleration relation;
- the baryonic Tully–Fisher normalization;
- outer rotation-curve shapes.

The relationship must be frozen before high-redshift analysis.

---

## Prediction 2 — Fixed gravitational slip

Galaxy dynamics and lensing must satisfy a model-derived:

\[
\eta_{\mathrm{IF}}(k,z).
\]

Once the IF action is fit to one class of observations, it predicts the other.

---

## Prediction 3 — Expansion-to-growth transfer

Fit only:

\[
H(z),\quad D_A(z),\quad D_L(z).
\]

Then predict:

\[
f\sigma_8(z),\quad P(k,z),\quad \text{weak lensing}.
\]

Reverse the procedure as an independent test.

---

## Prediction 4 — Environmental hysteresis

The IF field’s finite relaxation time produces dependence on recent gravitational history, not only present external acceleration.

---

## Prediction 5 — Merger-offset decay

Cluster lensing offsets should evolve according to a field-relaxation timescale and may differ systematically from collisionless-halo predictions.

---

## Prediction 6 — Scale-dependent lensing–growth relation

The same field produces a specific relationship among:

\[
\mu(k,z),\quad
\eta(k,z),\quad
\Sigma(k,z).
\]

No independent functions are permitted.

---

## Prediction 7 — Standard-siren relation

If the IF action modifies tensor friction while preserving:

\[
c_T=1,
\]

it predicts:

\[
\frac{
D_L^{\mathrm{GW}}(z)
}{
D_L^{\mathrm{EM}}(z)
}
\]

from the same background evolution.

---

# 24. The IF-G0 Phenomenological Model

Before implementing the complete covariant action, define a constrained phenomenological model.

## 24.1 Shared state

Let:

\[
b(a)
\]

be one homogeneous IF state.

Its evolution is:

\[
\boxed{
\frac{db}{d\ln a}
=
-\Gamma(b;\theta).
}
\]

---

## 24.2 Expansion

Define:

\[
H^2(a)
=
\frac{8\pi G}{3}
\left[
\rho_b(a)+\rho_r(a)+\rho_{\mathrm{IF}}(b)
\right].
\]

No independent dark-energy equation-of-state function is allowed.

---

## 24.3 Galactic scale

Define:

\[
a_{\mathrm{IF}}(a)
=
a_{\mathrm{IF},0}
F_b
\left[
b(a)
\right].
\]

In the minimal Hubble-locked restriction:

\[
F_b[b(a)]
=
\frac{H(a)}{H_0}.
\]

---

## 24.4 Linear gravity

Define:

\[
\mu(k,a)
=
\mu
\left[
k,b(a);\theta
\right],
\]

\[
\eta(k,a)
=
\eta
\left[
k,b(a);\theta
\right].
\]

These functions must be generated by one low-dimensional parameterization.

They may not be freely reconstructed independently.

---

## 24.5 Purpose

IF-G0 is not the final theory.

It asks whether even a tightly constrained unified closure can approach the data.

If G0 fails decisively, the broad narrative may still survive, but the simplest unification is rejected.

---

# 25. Model-Building Sequence

## Stage 0 — Dimensional closure

Ensure every field, scale, and invariant has consistent dimensions.

## Stage 1 — Quasistatic derivation

Derive:

\[
\mu_{\mathrm{IF}}(x,b)
\]

from the covariant action.

## Stage 2 — Homogeneous background

Derive:

\[
\rho_{\mathrm{IF}},
\qquad
p_{\mathrm{IF}},
\qquad
H(a).
\]

## Stage 3 — Linear perturbations

Derive:

\[
\mu(k,a),
\qquad
\eta(k,a),
\qquad
c_s^2,
\qquad
Q_s.
\]

## Stage 4 — Tensor sector

Derive:

\[
c_T,
\qquad
Q_T,
\qquad
\nu(a).
\]

## Stage 5 — Boltzmann implementation

Modify a public cosmological solver only after the preceding analytical tests pass.

## Stage 6 — Nonlinear dynamics

Develop galaxy, cluster, and cosmological simulations from the same equations.

---

# 26. Deterministic Jupyter-Notebook Program

## Notebook 07A — Covariant-Invariant Audit

Implement symbolic definitions of:

\[
X,\quad
u_\mu,\quad
h_{\mu\nu},\quad
a_\mu,\quad
K_{\mu\nu}.
\]

Verify:

\[
u_\mu u^\mu=-1,
\]

\[
h_{\mu\nu}u^\nu=0.
\]

Check dimensions and symmetries.

---

## Notebook 07B — Background Variation

Derive symbolic background equations for chosen:

\[
\mathcal L_{\mathrm{IF}}.
\]

Verify:

- constraint equations;
- conservation;
- GR limit;
- absence of a hidden cosmological constant.

---

## Notebook 07C — Weak-Field Limit

Expand the action around Minkowski space.

Derive the modified Poisson equation.

Verify:

\[
\mu(x\gg1)\rightarrow1,
\]

\[
\mu(x\ll1)\rightarrow x
\]

for the selected prototype.

---

## Notebook 07D — Spherical Solutions

Solve for:

- point mass;
- exponential disk approximation;
- Plummer sphere;
- Hernquist profile.

Recover asymptotic predictions such as:

\[
V_f^4=GM_ba_{\mathrm{IF}}.
\]

---

## Notebook 07E — Lensing Potentials

Derive:

\[
\Phi,
\qquad
\Psi,
\qquad
\eta,
\qquad
\Sigma.
\]

Verify that light deflection and matter motion arise from one metric.

---

## Notebook 07F — Hubble-Locked Acceleration Forecast

Using public expansion reconstructions, calculate:

\[
a_{\mathrm{IF}}(z).
\]

Produce frozen predictions for high-redshift galaxy dynamics.

No galaxy data are used in this forecasting step beyond the local normalization.

---

## Notebook 07G — Background Phase Search

Search low-dimensional IF Lagrangians for trajectories satisfying:

\[
w_{\mathrm{IF}}^{\mathrm{early}}\approx0,
\]

\[
w_{\mathrm{IF}}^{\mathrm{late}}<-\frac13.
\]

Reject unstable or fine-tuned trajectories.

---

## Notebook 07H — Scalar Stability

Calculate:

\[
Q_s,
\qquad
c_s^2.
\]

Reject:

\[
Q_s\leq0
\]

or:

\[
c_s^2<0.
\]

---

## Notebook 07I — Tensor Stability and Speed

Calculate:

\[
Q_T,
\qquad
c_T^2,
\qquad
\nu(a).
\]

Reject models inconsistent with the required tensor limit.

---

## Notebook 07J — Linear Growth Solver

Implement baryon and IF perturbations.

Calculate:

\[
D(a),
\qquad
f(a),
\qquad
f\sigma_8(a),
\qquad
\Phi+\Psi.
\]

---

## Notebook 07K — Synthetic CMB Feasibility

Before modifying a full Boltzmann code, test whether the IF perturbation mode can remain:

- sufficiently cold;
- stable;
- gravitationally effective;
- compatible with radiation-era evolution.

This is a no-go screening notebook, not a substitute for CMB calculation.

---

## Notebook 07L — Boltzmann-Code Modification

Implement the selected stable model in a code such as CLASS or CAMB.

Reproduce the standard-model limit first.

Then compute:

- TT;
- TE;
- EE;
- lensing;
- matter power spectrum.

---

## Notebook 07M — Expansion–Growth Consistency

Fit background-only synthetic or public compressed data.

Predict growth.

Then fit growth and predict expansion.

---

## Notebook 07N — Cluster-Merger Field Dynamics

Solve a two-component merger toy model containing:

- collisionless galaxies;
- collisional gas;
- dynamical IF field.

Predict lensing offsets and relaxation.

---

## Notebook 07O — Gravitational-Wave Propagation

Calculate:

\[
D_L^{\mathrm{GW}}/D_L^{\mathrm{EM}},
\]

tensor speed, damping, and stability.

---

## Notebook 07P — Solar-System Screening

Solve the high-acceleration limit around a compact baryonic source.

Quantify residual deviations from GR.

---

## Notebook 07Q — Parameter-Count Audit

Compare:

- physical parameters;
- nuisance parameters;
- functional degrees of freedom;
- effective Bayesian complexity

against \(\Lambda\)CDM and competing modified-gravity models.

---

## Notebook 07R — No-Go Search

Search automatically for contradictions among:

- MOND-like galaxies;
- CMB clustering;
- late acceleration;
- luminal tensors;
- scalar stability;
- Solar-System recovery.

A negative result is a central scientific outcome.

---

## Notebook 07S — Independent Reimplementation

A separate coding agent derives and implements the equations without reading the primary implementation.

Compare all limits and numerical solutions.

---

# 27. Computational Architecture

```text
if_geometry/
├── symbolic/
│   ├── tensors.py
│   ├── invariants.py
│   ├── action.py
│   ├── variation.py
│   └── perturbations.py
├── background/
│   ├── equations.py
│   ├── integrator.py
│   └── phase_search.py
├── quasistatic/
│   ├── poisson.py
│   ├── spherical.py
│   ├── disk.py
│   └── lensing.py
├── perturbations/
│   ├── scalar.py
│   ├── tensor.py
│   ├── stability.py
│   └── growth.py
├── boltzmann/
│   ├── class_patch/
│   ├── lcdm_regression/
│   └── if_model/
├── clusters/
│   ├── merger.py
│   └── relaxation.py
├── inference/
│   ├── likelihoods.py
│   ├── synthetic_recovery.py
│   └── model_comparison.py
└── tests/
```

---

# 28. Validation Standard

## 28.1 Symbolic and numerical agreement

Every field equation must be checked through:

- symbolic variation;
- independent hand-derived expression;
- numerical conservation residuals;
- known limits.

---

## 28.2 GR regression

When IF couplings vanish or the screening limit applies, the implementation must recover:

- standard Friedmann evolution;
- standard Poisson gravity;
- standard linear growth;
- standard gravitational waves.

---

## 28.3 Synthetic recovery

Generate mock observations from known IF parameters.

Verify:

- unbiased recovery;
- correct interval coverage;
- identifiability;
- no false preference for IF when mocks are \(\Lambda\)CDM.

---

## 28.4 Stability-first inference

Parameter points failing theoretical stability are excluded before likelihood evaluation.

The inference engine may not report an observationally preferred but physically unstable model as viable.

---

## 28.5 Blind cross-domain tests

Use galaxy data to calibrate the quasistatic limit.

Freeze parameters.

Then test cosmology.

Or use cosmology first and test galaxies.

Do not fit all sectors simultaneously before establishing cross-domain predictivity.

---

# 29. Reproducibility Record

Each model run emits:

```yaml
experiment_id: if-unified-geometry-07
paper_version: null
git_commit: null
environment_hash: null
model_version: null

action_family: null
lagrangian_parameters: {}
matter_coupling: universal_metric
bare_cosmological_constant: 0

background_initial_conditions: {}
background_solution_hash: null
late_acceleration: null
early_equation_of_state: null

scalar_kinetic_coefficient_min: null
scalar_sound_speed_squared_min: null
tensor_kinetic_coefficient_min: null
tensor_speed_deviation_max: null
effective_cutoff_min: null

newtonian_limit_residual: null
deep_if_limit_residual: null
solar_system_residual: null

acceleration_scale_today: null
acceleration_scale_history_hash: null
mu_history_hash: null
eta_history_hash: null
sigma_history_hash: null

cmb_tt_hash: null
cmb_te_hash: null
cmb_ee_hash: null
cmb_lensing_hash: null
matter_power_hash: null
growth_history_hash: null

energy_conservation_residual: null
constraint_residual: null
stability_failures: []
result_hash: null
```

---

# 30. Failure Modes

## 30.1 Renaming dark matter

The IF field clusters, persists, lenses, and moves exactly like collisionless matter but is called geometry solely for branding.

## 30.2 Two theories hidden in one notation

One function fits galaxies and an unrelated function fits expansion.

## 30.3 Free-function overfitting

A flexible Lagrangian is reconstructed directly from every dataset.

## 30.4 CMB avoidance

The theory discusses rotation curves while ignoring acoustic peaks and CMB lensing.

## 30.5 Background-only success

The model fits:

\[
H(z)
\]

but fails growth and lensing.

## 30.6 Lensing patch

An extra coupling is added only after dynamics fail to predict lensing.

## 30.7 Cluster-memory patch

A relaxation term is introduced solely to reproduce the Bullet Cluster without predicting other mergers.

## 30.8 Unstable acceleration

Late acceleration requires:

- ghosts;
- gradient instability;
- strong coupling;
- excluded tensor propagation.

## 30.9 Hidden cosmological constant

A constant term inside the IF Lagrangian performs the entire acceleration job.

## 30.10 Screening by assertion

The theory claims to recover GR locally without deriving the screened solution.

## 30.11 Numerological acceleration scale

The galaxy scale is declared proportional to:

\[
cH_0
\]

without derivation or redshift prediction.

## 30.12 Cosmological hindsight

The model is tuned after seeing every available dataset and then described as predictive.

## 30.13 “Information” without physical definition

The field is called informational, but no measurable informational or nonequilibrium quantity connects it to Papers 1–6.

## 30.14 Big-Bang inflation

A post-recombination modified-gravity model is presented as a complete theory of cosmic formation.

---

# 31. What Would Count as Success?

## Level 1 — Consistent covariant model

A low-dimensional action produces stable equations and a GR limit.

## Level 2 — Galactic success

The model predicts rotation curves and galaxy scaling relations from baryons with minimal system-specific freedom.

## Level 3 — Relativistic success

The same model predicts correct lensing and gravitational-wave propagation.

## Level 4 — Linear-cosmology success

The model reproduces CMB and matter-power-spectrum observables without particle dark matter.

## Level 5 — Acceleration success

The same IF sector yields late-time acceleration without an independent cosmological constant.

## Level 6 — Expansion–growth consistency

A background fit predicts growth and lensing.

## Level 7 — Cluster success

The model predicts cluster and merger lensing without an unseen particle halo.

## Level 8 — Prospective prediction

A frozen IF prediction is confirmed by data unavailable during model construction.

## Level 9 — Microscopic derivation

The covariant action is derived from a deeper IF substrate rather than selected phenomenologically.

---

# 32. What Would Count as a Major Discovery?

A publishable theoretical result could be:

> A stable covariant IF action produces a general-relativistic local limit, a MOND-like galactic limit, and accelerated homogeneous expansion.

That result alone would not establish observational viability.

A field-changing result would be:

\[
\boxed{
\text{One low-parameter IF action reproduces galaxies, lensing, the CMB,}
\atop
\text{large-scale growth, and expansion without particle dark matter}
\atop
\text{or an independent cosmological constant.}
}
\]

A Nobel-class result would require more:

\[
\boxed{
\text{The theory makes a distinctive quantitative prediction before}
\atop
\text{measurement, and the prediction is independently confirmed.}
}
\]

Examples could include:

- redshift evolution of the galactic acceleration scale;
- a fixed lensing–growth relation;
- a cluster-merger relaxation law;
- a gravitational-wave distance anomaly;
- a specific departure in future Euclid or DESI-era observables.

---

# 33. Relationship to Earlier IF Papers

## 33.1 Informational battery

Paper 1 defined the informational battery through:

- gross nonequilibrium capacity;
- operational accessibility;
- latent capacity;
- lawful discharge and recharge.

Paper 7 must eventually identify a cosmological analogue of these quantities.

At present, the connection is conceptual rather than derived.

---

## 33.2 Causal-work principle

Paper 2 concerned systems using information to access work.

The universe as a whole is not assumed to be an agent.

The IF geometry may organize causal accessibility without possessing prediction, intention, or agency.

---

## 33.3 Emergent structures

Paper 3 studies structure formation in artificial universes.

Its results cannot be transferred directly to cosmology without deriving:

- continuum geometry;
- relativistic dynamics;
- observational variables.

---

## 33.4 Expansion–complexity window

Paper 4 asks whether artificial-domain growth produces a complexity window.

That result would not prove cosmic expansion is caused by complexity.

Paper 7 must derive expansion from the covariant action independently.

---

## 33.5 Agency and reflection

Papers 5 and 6 concern predictive agents and self-maintenance.

Galaxies and cosmic fields are not classified as conscious or agentic.

The word **information** must not smuggle purpose into cosmology.

---

# 34. Philosophical Interpretation

If successful, the IF theory could support the interpretation that:

> The phenomena currently separated into missing gravitational mass and vacuum-driven expansion arise from one deeper organization of spacetime.

It would not prove:

- that the universe is a computer;
- that cosmic expansion has a purpose;
- that life causes expansion;
- that consciousness creates gravity;
- that God is a physical field.

Those propositions require separate arguments.

---

# 35. Criteria for Rejection or Major Revision

The IF Unified Geometry Hypothesis should be rejected or substantially revised if:

1. the covariant theory cannot be made stable;
2. the GR limit cannot satisfy local tests;
3. galaxy behavior requires per-galaxy IF halos;
4. lensing requires a separate correction;
5. early IF perturbations cannot reproduce the CMB;
6. late acceleration requires an independent cosmological constant;
7. expansion and growth require incompatible parameter histories;
8. gravitational-wave speed or damping is excluded;
9. cluster mergers cannot be explained;
10. the same field cannot transition consistently among early, galactic, and late-time regimes;
11. the model contains more effective freedom than the two dark components it replaces;
12. no prospective prediction distinguishes it from established alternatives;
13. its “informational” interpretation never acquires a measurable physical definition;
14. the theory becomes indistinguishable from a known scalar–aether or relativistic MOND theory except for terminology.

---

# 36. Conclusion

The IF Unified Geometry Hypothesis does not claim that observations attributed to dark matter and dark energy are false.

It proposes that their division into two independent invisible substances may be incomplete.

The hypothesis is:

\[
\boxed{
\text{One nonequilibrium informational geometry produces}
\atop
\text{additional attractive response when spatially structured and}
\atop
\text{accelerated expansion when evolving homogeneously.}
}
\]

The minimal covariant target class is:

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
S_b[g,\Psi_b].
}
\]

Its defining scientific constraint is:

\[
\boxed{
\left\{
H,\mu,\eta,a_{\mathrm{IF}},c_T
\right\}
=
\mathfrak D
\left[
\mathcal L_{\mathrm{IF}};\theta
\right].
}
\]

One action.

One parameter system.

No independent galaxy law.

No independent dark-energy curve.

No independent lensing patch.

No particle-dark-matter fluid in the minimal model.

No separately inserted cosmological constant.

The price of that conceptual unification is severe.

The IF sector must perform every successful dynamical role normally divided among:

- general relativity;
- cold dark matter;
- dark energy.

The current hypothesis is therefore high risk.

Its present scientific value is not that it already explains the universe.

Its value is that it defines a narrow computational and mathematical path by which the idea can either become a physical theory or fail decisively.

The next paper will begin the first observational branch:

\[
\boxed{
\textit{Galactic Tests of IF Unified Geometry: Rotation Curves,}
\atop
\textit{Environmental Effects, and Gravitational Lensing.}
}
\]

---

# References

1. Planck Collaboration. “Planck 2018 Results VI: Cosmological Parameters.” The final Planck temperature, polarization, and lensing analysis found strong consistency with spatially flat six-parameter \(\Lambda\)CDM. citeturn716898search0

2. DESI Collaboration. “DESI DR2 Results II: Measurements of Baryon Acoustic Oscillations and Cosmological Constraints.” The DR2 BAO analysis uses over fourteen million galaxy and quasar tracers. citeturn716898search2

3. McGaugh, S. S., Lelli, F. and Schombert, J. M. “The Radial Acceleration Relation in Rotationally Supported Galaxies.” citeturn452924academia49

4. McGaugh, S. S. et al. “The Baryonic Tully–Fisher Relation.” citeturn452924academia52

5. Bekenstein, J. D. “Relativistic Gravitation Theory for the MOND Paradigm.” citeturn452924academia50

6. Skordis, C. and Złośnik, T. “A New Relativistic Theory for Modified Newtonian Dynamics.” The theory demonstrates that a relativistic modified-gravity sector can reproduce MOND-like galaxy behavior while matching important linear cosmological observables. citeturn716898search4

7. Skordis, C. and Złośnik, T. “Aether Scalar Tensor Theory: Linear Stability on Minkowski Space.” citeturn716898search19

8. Clowe, D. et al. “A Direct Empirical Proof of the Existence of Dark Matter.” citeturn716898search5

9. Baker, T. et al. “Strong Constraints on Cosmological Gravity from GW170817 and GRB 170817A.” citeturn599772search12

10. Frusciante, N. and Perenon, L. “Effective Field Theory of Dark Energy: A Review.” citeturn599772academia40

11. Lombriser, L. et al. “Inherently Stable Effective Field Theory for Dark Energy and Modified Gravity.” citeturn599772academia42

12. Joyce, A. et al. “Dark Energy versus Modified Gravity.” citeturn599772search27
