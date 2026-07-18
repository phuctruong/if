<!-- extracted from ChatGPT (driver) on 2026-07-18 -->

# Emergent Structure in Resource-Conserving IF Universes  
## Minimal Local Dynamics, Objective Structure Detection, and Tests Against Designed Emergence

**Author:** Phuc Vinh Truong  
**Series:** IF Theory Working Papers  
**Paper:** 3  
**Date:** July 18, 2026  
**Status:** Computational specification awaiting implementation, parameter search, and independent reproduction

---

## Abstract

IF Theory proposes that persistent organization, life-like self-maintenance, and eventually agency might emerge from local physical interactions governed by explicit resource constraints. Demonstrating visually interesting patterns, however, is insufficient. Cellular automata, reaction–diffusion systems, continuous artificial-life substrates, and trained neural cellular automata already produce complex morphologies, locomotion, regeneration, and self-organized behavior. The scientific challenge is to determine whether organization emerges without being inserted through target templates, organism-specific variables, hand-designed seeds, hidden energy sources, or reward functions that directly favor the desired result.

This paper specifies a family of **resource-conserving IF universes** designed to test that question. Primitive elements possess local physical states, conserved material, finite high-grade resource capacity, degraded resource or waste, and globally shared local transition rules. They do not possess primitive variables named life, organism, boundary, fitness, memory, reflection, consciousness, cooperation, or love.

Two reference implementations are proposed. **IF-RC0** is a deterministic accounting model with exact conservation of material and abstract capacity tokens. It is intended for software validation, exhaustive rule search, and reproducible phase mapping. **IF-RC1** is a stochastic-thermodynamic extension whose transition rates satisfy local detailed-balance constraints, permitting explicit calculations of work, heat, and entropy production. RC0 may demonstrate computational resource conservation; only RC1 may support thermodynamic claims.

Structures are identified through a preregistered, target-independent pipeline based on localization, temporal persistence, boundary stability, material and causal continuity, resource throughput, and perturbation response. Self-repair is measured against an undamaged counterfactual twin and against passive-relaxation controls. Replication requires the appearance of dynamically independent descendants that inherit a reproducible organization, not merely temporary fragmentation or spreading activity.

The principal hypotheses are that localized persistent structures occupy reproducible regions of rule space; that some structures maintain themselves through resource throughput rather than static stability alone; that repair can arise without a repair objective; and that detected structures can exhibit stronger causal and predictive closure than arbitrary matched regions. Each hypothesis is accompanied by explicit falsifiers.

The proposed novelty does not lie in generating life-like cellular-automaton patterns. Existing systems such as Lenia and mass-conserving Flow-Lenia already generate spatially localized artificial lifeforms and evolutionary dynamics, while neural cellular automata can be trained to grow and repair prescribed forms. The possible IF contribution is a combined protocol requiring exact resource accounting, target-free discovery, objective post hoc detection, causal interventions, held-out replication, and agreement across substantially different substrates. If those requirements cannot be met, the claim that IF structures emerge autonomously must be rejected.

---

## Keywords

Artificial life; cellular automata; self-organization; resource conservation; morphogenesis; causal emergence; self-repair; replication; nonequilibrium systems; phase diagrams; reproducibility.

---

# 1. Introduction

A small collection of local rules can generate patterns that appear remarkably organized. Turing’s reaction–diffusion framework demonstrated mathematically that diffusion and local chemical interaction can destabilize a uniform state and produce spatial patterning. Cellular automata later supplied discrete computational substrates in which persistent objects, oscillators, moving patterns, and self-reproduction could be studied through local updates. These traditions establish that global pattern need not require global command. citeturn795026search7turn550023search7

Modern artificial-life systems extend this result. Lenia is a continuous cellular-automaton family supporting many localized, resilient, and motile patterns. Flow-Lenia adds mass conservation and localized rule parameters, enabling interacting artificial forms and emergent evolutionary dynamics. Neural cellular automata can be trained to grow a specified target image and regenerate it after damage. Automated search methods can now use diversity search, curiosity-driven exploration, or foundation models to discover previously unseen artificial-life behavior. citeturn775349search2turn795026search1turn795026search3turn550023academia38turn795026academia50

These achievements create a strict novelty boundary for IF Theory.

It is not sufficient to show:

- a beautiful moving pattern;
- a pattern that survives for many updates;
- a trained system that reconstructs a target;
- a mass-conserving cellular automaton;
- a parameter sweep containing organism-like forms;
- an AI agent that labels a simulation “alive.”

The scientific question is narrower and harder:

\[
\boxed{
\text{Can persistent, resource-processing, perturbation-resistant}
\atop
\text{structures arise from minimal local laws without their forms,}
\atop
\text{boundaries, repair objectives, or fitness functions being prescribed?}
}
\]

A second question follows:

\[
\boxed{
\text{Can those structures be detected and tracked by a reproducible}
\atop
\text{procedure that does not depend on human visual preference?}
}
\]

This paper defines the computational system and experimental controls required to answer those questions.

---

# 2. Scientific Scope

Paper 3 concerns the emergence of **structure** and preliminary life-like properties.

It does not claim to establish:

- biological life;
- predictive agency;
- consciousness;
- moral behavior;
- cosmic expansion;
- fundamental particle physics;
- quantum gravity;
- divine purpose.

A persistent structure discovered in this paper may become a candidate system for later agency experiments. It is not an agent merely because it moves, persists, or repairs.

The intended progression is:

\[
\text{local dynamics}
\rightarrow
\text{localized structure}
\rightarrow
\text{self-maintenance}
\rightarrow
\text{repair}
\rightarrow
\text{replication}
\rightarrow
\text{agency tests}.
\]

Each arrow requires a separate operational test.

---

# 3. Prior Art and the Novelty Boundary

## 3.1 Reaction–diffusion pattern formation

Turing’s morphogenesis model showed that a homogeneous chemical state can become unstable under coupled reaction and diffusion, generating organized spatial patterns. This establishes that spontaneous symmetry breaking and pattern formation do not by themselves demonstrate life or agency. citeturn795026search7

IF Theory cannot claim novelty for producing spots, stripes, waves, oscillations, or other reaction–diffusion-like structures.

---

## 3.2 Classical and continuous cellular automata

Lenia generalized cellular-automaton dynamics to continuous states, time, and neighborhoods, producing many spatially localized patterns with movement, resilience, and complex morphology. The later expanded Lenia framework explored multiple channels, kernels, and dimensions. citeturn775349search2turn775349search29

IF Theory therefore cannot classify a simulation as scientifically novel merely because it contains a persistent moving “creature.”

---

## 3.3 Mass-conserving artificial life

Flow-Lenia was explicitly developed as a mass-conserving extension of Lenia and has demonstrated localized patterns, multiple coexisting forms, parameter localization, and emergent evolutionary dynamics. citeturn795026search1turn550023search2

Resource or mass conservation alone is therefore not an IF innovation.

The IF standard is stronger: conservation must be accompanied by a declared capacity ledger, a distinction between high-grade and degraded resources, target-free discovery, and intervention-based tests of maintenance and repair.

---

## 3.4 Trained morphogenesis and regeneration

Growing neural cellular automata can learn a shared local update rule that grows and regenerates a specified target pattern. Such systems compellingly demonstrate decentralized morphogenesis and repair, but the desired form is present in the training loss. citeturn795026search3turn795026search9

IF Theory distinguishes:

\[
\textbf{targeted regeneration}
\]

from:

\[
\textbf{spontaneous self-repair}.
\]

A system trained to reconstruct a particular image demonstrates distributed control, not the spontaneous emergence of the goal of preserving that form.

---

## 3.5 Automated discovery of artificial life

Diversity search has discovered localized structures showing movement, integrity maintenance, obstacle navigation, and generalization in cellular-automaton environments. Foundation-model and curiosity-driven systems have also automated the exploration of large artificial-life spaces and uncovered diverse dynamics. citeturn550023academia39turn550023academia38turn795026academia50

IF Theory may use these methods for exploration, but AI-generated aesthetic or semantic judgments cannot serve as the primary scientific metric. The discovery system must be separated from the confirmatory evaluator.

---

## 3.6 Conservation laws in cellular automata

Conservation laws in cellular automata can be defined and tested mathematically. Lattice-gas and reversible cellular-automaton frameworks also provide mechanisms for exact local transport and microscopic reversibility. citeturn775349academia82turn775349academia83

IF Theory therefore cannot infer physical legitimacy merely from using a cellular automaton. Its specific update rules must satisfy their declared conservation identities.

---

## 3.7 Statistical adaptation

Nonequilibrium statistical physics has investigated how driven systems may become statistically associated with structures adapted to their forcing environment. Such results do not establish agency, but they show that physical drive and dissipation can bias the formation of particular organized states. citeturn775349search8

IF Theory must therefore distinguish adaptation-like selection among physical states from genuine endogenous control.

---

## 3.8 Provisional novelty claim

The prospective contribution of Paper 3 is not any one ingredient.

It is the combined protocol:

\[
\boxed{
\begin{gathered}
\text{minimal globally shared local laws}
+
\text{exact resource accounting}
+
\text{no target morphology}\\
+
\text{automatic structure detection}
+
\text{counterfactual repair testing}
+
\text{held-out replication}
+
\text{cross-substrate confirmation}.
\end{gathered}
}
\]

This is a proposed methodological synthesis. Scientific novelty will exist only if the resulting experiments discover a robust phenomenon not already explained by simpler artificial-life systems.

---

# 4. Design Principles

## 4.1 Nothing life-like is primitive

The primitive state may contain:

- material;
- energetic or capacity state;
- local configuration;
- position or graph connectivity;
- interaction channels.

It may not contain:

- `is_alive`;
- `organism_id`;
- `body`;
- `boundary`;
- `fitness`;
- `repair_target`;
- `memory_score`;
- `intelligence`;
- `reflection`;
- `consciousness`;
- `cooperation_bonus`;
- `love`.

Candidate organisms are interpretations generated after the dynamics are run.

---

## 4.2 Locality

Each update depends only on a declared local causal neighborhood.

For site or node \(i\):

\[
z_i(t+1)
=
T_\theta
\left[
z_i(t),
\{z_j(t):j\in\mathcal N_i(t)\},
\xi_i(t)
\right],
\]

where:

- \(z_i\) is the local state;
- \(\mathcal N_i\) is the causal neighborhood;
- \(\theta\) is the globally shared rule;
- \(\xi_i\) is optional declared noise.

No update may query:

- the complete grid;
- the location of a target shape;
- whether a cell belongs to a detected organism;
- a global score instructing it to repair.

---

## 4.3 Translation and orientation neutrality

The primary reference models should not privilege a particular absolute location.

Where possible, the rules should also be:

- rotation-equivariant;
- reflection-equivariant;
- permutation-invariant over equivalent neighbors.

Any broken symmetry must be declared.

---

## 4.4 Exact resource accounting

Every material and capacity flow must be attributable to:

- local transport;
- local transformation;
- environmental input;
- environmental export.

Numerical clipping must not silently create or destroy resources.

---

## 4.5 Discovery and confirmation must be separated

Search algorithms may identify interesting rules in a discovery set.

Scientific claims must be evaluated using:

- held-out seeds;
- held-out perturbations;
- held-out parameter neighborhoods;
- independently written evaluators;
- frozen classification criteria.

---

## 4.6 No single beauty score

The simulator must not classify structures through one opaque “life-likeness” score.

It should report a vector of independently interpretable measurements:

\[
\mathbf S
=
\left[
L,
P,
B,
T_R,
C,
R,
D,
N
\right],
\]

where the components may represent localization, persistence, boundary strength, throughput, causal closure, repair, descendant formation, and novelty.

---

# 5. General IF Universe

Let the simulated universe at discrete time \(t\) be:

\[
\mathcal U_t
=
\left(
G_t,Z_t,\mathcal R_t,\Lambda
\right),
\]

where:

- \(G_t=(V_t,E_t)\) is a lattice or interaction graph;
- \(Z_t=\{z_i(t)\}_{i\in V_t}\) is the set of local states;
- \(\mathcal R_t\) records global resource ledgers;
- \(\Lambda\) specifies external boundary conditions.

For Paper 3, the primary domain is fixed. Dynamic growth of space is deferred to Paper 4.

Each local state is:

\[
z_i
=
\left(
m_i^1,\ldots,m_i^K,
f_i,
w_i,
s_i
\right),
\]

where:

- \(m_i^a\geq0\) is conserved material of species or channel \(a\);
- \(f_i\geq0\) is high-grade resource or available-capacity material;
- \(w_i\geq0\) is degraded resource or waste;
- \(s_i\in\mathcal S\) is a finite local configuration state.

The word **species** here denotes a simulation channel, not a biological species.

---

# 6. Two Reference Implementations

## 6.1 IF-RC0: deterministic resource accounting

IF-RC0 is designed for:

- exact software testing;
- deterministic replay;
- large rule sweeps;
- phase classification;
- debugging;
- proof of resource conservation.

It uses abstract capacity units.

It does **not**, by itself, establish a physical thermodynamic entropy law.

The RC0 claim is:

\[
\text{the programmed resource ledger closes}.
\]

It is not:

\[
\text{the model is a complete physical thermodynamic universe}.
\]

---

## 6.2 IF-RC1: stochastic-thermodynamic extension

IF-RC1 represents local transitions as stochastic processes with:

- declared state energies;
- heat reservoirs;
- chemical or resource potentials;
- local detailed-balance relations;
- trajectory-level work and heat.

RC1 is intended to support claims about:

- entropy production;
- nonequilibrium maintenance;
- work extraction;
- physical costs.

Any result initially discovered in RC0 must be re-evaluated in RC1 before being interpreted thermodynamically.

---

# 7. IF-RC0 Dynamics

## 7.1 Conserved material transport

Let:

\[
J_{ij}^a(t)
\]

be the net material flux of channel \(a\) from node \(i\) to node \(j\).

Require antisymmetry:

\[
J_{ij}^a
=
-J_{ji}^a.
\]

The update is:

\[
m_i^a(t+1)
=
m_i^a(t)
-
\sum_{j\in\mathcal N_i}
J_{ij}^a(t).
\]

Therefore:

\[
\sum_i m_i^a(t+1)
=
\sum_i m_i^a(t)
\]

under closed boundaries.

For every material channel:

\[
\boxed{
M_a
=
\sum_i m_i^a
=
\text{constant}.
}
\]

---

## 7.2 Pairwise flux rule

A general local flux may be:

\[
J_{ij}^a
=
\operatorname{clip}
\left[
\kappa_a
\left(
\mu_i^a-\mu_j^a
\right),
-J_{\max},
J_{\max}
\right],
\]

where \(\mu_i^a\) is a local potential computed from the neighborhood.

To preserve antisymmetry, each edge flux is calculated once and applied with opposite signs to its endpoints.

The potential may depend on:

\[
\mu_i^a
=
F_\theta^a
\left(
z_i,
\operatorname{Agg}\{z_j:j\in\mathcal N_i\}
\right).
\]

The aggregation must be local and symmetry-compatible.

---

## 7.3 Resource conversion

Let:

\[
c_i(t)\geq0
\]

be high-grade resource consumed by local transitions.

Let:

\[
0\leq\eta_i(t)\leq1
\]

be the fraction credited to declared useful work.

Then:

\[
f_i(t+1)
=
f_i(t)
-
c_i(t)
+
I_i^f(t),
\]

\[
w_i(t+1)
=
w_i(t)
+
\left[
1-\eta_i(t)
\right]c_i(t)
-
O_i^w(t),
\]

and the exported-work ledger changes by:

\[
W_{\mathrm{out}}(t+1)
=
W_{\mathrm{out}}(t)
+
\sum_i\eta_i(t)c_i(t).
\]

Here:

- \(I_i^f\) is external resource input;
- \(O_i^w\) is waste exported through a declared sink.

For a closed RC0 universe:

\[
I_i^f=O_i^w=0.
\]

The abstract capacity balance is:

\[
\boxed{
\sum_i f_i(t)
+
\sum_i w_i(t)
+
W_{\mathrm{out}}(t)
=
C_{\mathrm{total}}.
}
\]

This equation is a simulation conservation law. Physical interpretation requires RC1.

---

## 7.4 Configuration-state transitions

A local configuration transition:

\[
s_i(t)\rightarrow s_i(t+1)
\]

has cost:

\[
c_s
\left[
s_i(t),s_i(t+1)
\right]
\geq0.
\]

The transition may occur only when sufficient local high-grade resource exists:

\[
f_i(t)
\geq
c_s.
\]

The globally shared local rule proposes the transition. It does not know whether the node belongs to a structure.

---

## 7.5 Driven mode

Long-lived self-maintaining structures generally require continuing access to a gradient.

In driven experiments:

- fuel enters through fixed or stochastic source regions;
- waste exits through declared sinks;
- source and sink rules are external boundary conditions;
- all imported and exported capacity is logged.

A structure may reorganize how effectively it intercepts these flows, but it cannot alter the ledger.

---

# 8. IF-RC1 Thermodynamic Dynamics

Let \(x\) and \(y\) be local microstates.

A transition:

\[
x\rightarrow y
\]

occurs at rate:

\[
k_{xy}.
\]

For a transition coupled to a heat bath at inverse temperature \(\beta\), local detailed balance requires a relationship of the form:

\[
\ln
\frac{k_{xy}}{k_{yx}}
=
\beta
\left[
W_{xy}
-
\Delta E_{xy}
\right],
\]

with the exact expression depending on the modeled reservoirs and sign conventions.

For trajectory \(\Gamma\), record:

\[
W[\Gamma],\qquad
Q[\Gamma],\qquad
\Delta S_{\mathrm{sys}}[\Gamma],
\qquad
\Delta S_{\mathrm{env}}[\Gamma].
\]

Mean total entropy production must satisfy:

\[
\boxed{
\left\langle
\Delta S_{\mathrm{tot}}
\right\rangle
=
\left\langle
\Delta S_{\mathrm{sys}}
+
\Delta S_{\mathrm{env}}
\right\rangle
\geq0.
}
\]

Structures may reduce their internal entropy or maintain a narrow state distribution only while exporting entropy or consuming free energy from the environment.

---

# 9. Initial Conditions

The simulation must distinguish three classes of experiment.

## 9.1 Homogeneous null

All sites begin in the same state, with no noise.

This tests whether the deterministic rule preserves exact symmetry.

A deterministic translation-symmetric rule should not spontaneously break perfect symmetry without:

- noise;
- asynchronous updates;
- an instability seeded by numerical asymmetry;
- asymmetric boundaries.

Unexpected structure in this condition may indicate a bug.

---

## 9.2 Perturbed homogeneous state

All sites begin near a common background state:

\[
z_i(0)
=
z_0+\epsilon_i,
\]

where \(\epsilon_i\) is sampled from a declared distribution.

This tests spontaneous amplification of generic fluctuations.

---

## 9.3 Generic sparse seeds

A small number of sites receive random material or configuration perturbations.

The seeds must not be designed to resemble the eventual structure.

The same rule must be tested over many seed realizations.

---

# 10. What Counts as a Structure?

Visual inspection is insufficient.

A candidate structure is a temporally tracked region \(A_t\subseteq V\) satisfying preregistered criteria.

## 10.1 Activity field

Define local deviation from the estimated background:

\[
a_i(t)
=
d_z
\left[
z_i(t),z_{\mathrm{bg}}(t)
\right],
\]

where \(d_z\) is a fixed state-space distance and \(z_{\mathrm{bg}}\) is estimated without using a target pattern.

The detection threshold is determined from null simulations:

\[
a_i>a_{\mathrm{thr}},
\]

where \(a_{\mathrm{thr}}\) may be the \(1-\alpha\) quantile of the null activity distribution.

---

## 10.2 Spatial localization

Define effective occupied size:

\[
N_{\mathrm{eff}}
=
\frac{
\left(
\sum_i a_i
\right)^2
}{
\sum_i a_i^2
}.
\]

A normalized localization measure is:

\[
\boxed{
L
=
1-\frac{N_{\mathrm{eff}}}{|V|}.
}
\]

High \(L\) indicates concentration relative to the whole domain.

Localization alone does not imply structure. A single static spike is localized.

---

## 10.3 Connected candidate regions

Active sites are grouped through:

- connected components;
- density-based clustering;
- persistent-homology features;
- or graph communities.

The primary method must be frozen before confirmation.

Alternative methods serve as robustness checks.

---

## 10.4 Temporal tracking

Candidate regions at times \(t\) and \(t+1\) are matched through a cost incorporating:

- material overlap;
- predicted displacement;
- state-distribution similarity;
- optimal transport;
- shape-independent composition.

A moving structure should not lose identity merely because it changes location.

The matching may use the Hungarian algorithm or a min-cost-flow lineage graph.

---

## 10.5 Persistence

Let \(\tau_A\) be the lifetime of tracked candidate \(A\).

A candidate qualifies as persistent only if:

\[
\tau_A>\tau_{\mathrm{null}},
\]

where \(\tau_{\mathrm{null}}\) is a preregistered high quantile of lifetimes generated by matched null dynamics.

No universal lifetime threshold is assumed.

---

## 10.6 Boundary strength

Let \(\mathcal E_{\mathrm{in}}\) denote edges internal to \(A_t\) and \(\mathcal E_{\mathrm{out}}\) edges crossing its boundary.

Define:

\[
C_{\mathrm{in}}
=
\sum_{(i,j)\in\mathcal E_{\mathrm{in}}}
\left|
J_{ij}
\right|,
\]

\[
C_{\mathrm{cross}}
=
\sum_{(i,j)\in\mathcal E_{\mathrm{out}}}
\left|
J_{ij}
\right|.
\]

A preliminary boundary ratio is:

\[
\boxed{
B_A
=
\frac{
C_{\mathrm{in}}
}{
C_{\mathrm{in}}+C_{\mathrm{cross}}
}.
}
\]

High internal coupling can indicate integration, but an impermeable inert object may also score highly. Boundary strength must be interpreted with throughput and persistence.

---

## 10.7 Resource throughput

For candidate \(A\), define imported fuel:

\[
F_{\mathrm{in}}^A,
\]

exported waste:

\[
W_{\mathrm{out}}^A,
\]

and internal resource conversion:

\[
C_A.
\]

A dynamically self-maintaining structure should exhibit nonzero sustained throughput:

\[
\boxed{
T_A
=
\frac{1}{\tau}
\int_t^{t+\tau}
C_A(t')\,dt'
>0.
}
\]

A static crystal-like pattern may persist with:

\[
T_A\approx0.
\]

That is persistent structure, not metabolic self-maintenance.

---

# 11. Structure Classification Vector

For every candidate, report:

\[
\boxed{
\mathbf S_A
=
\left[
L_A,
\tau_A,
B_A,
T_A,
M_A,
R_A,
D_A,
C_A
\right],
}
\]

where:

- \(L_A\): localization;
- \(\tau_A\): persistence;
- \(B_A\): boundary strength;
- \(T_A\): resource throughput;
- \(M_A\): motility;
- \(R_A\): repair response;
- \(D_A\): descendant or replication evidence;
- \(C_A\): causal-closure evidence.

No universal scalar is initially formed.

This prevents arbitrary weights from converting a weak result into a high “life score.”

---

# 12. Phase Taxonomy

## Phase P0 — Extinction

Activity and nonequilibrium structure decay to the null background.

## Phase P1 — Homogeneous equilibrium

The domain approaches a spatially homogeneous stationary state.

## Phase P2 — Frozen pattern

Persistent spatial organization exists with negligible throughput or adaptation.

## Phase P3 — Distributed turbulence

Activity remains high but does not form stable localized structures.

## Phase P4 — Transient localization

Localized structures form but remain within the null lifetime distribution.

## Phase P5 — Persistent localization

Localized structures survive substantially longer than matched null structures.

## Phase P6 — Throughput-maintained structures

Persistent structures consume resources and export degraded products while maintaining organization.

## Phase P7 — Motile structures

Persistent structures exhibit displacement not explained by diffusion or global drift.

## Phase P8 — Repair-capable structures

After controlled damage, structures return toward their undamaged counterfactual trajectory more strongly than matched passive controls.

## Phase P9 — Replicating structures

Structures generate dynamically independent descendants that inherit reproducible organization.

## Phase P10 — Candidate adaptive structures

Structures modify behavior across environments in a manner later shown to depend causally on internal state.

Paper 3 may identify phases P0–P9.

Predictive agency remains the subject of Papers 2 and 5.

---

# 13. Measuring Motility

For the tracked center of mass:

\[
\mathbf r_A(t)
=
\frac{
\sum_{i\in A_t}a_i(t)\mathbf r_i
}{
\sum_{i\in A_t}a_i(t)
}.
\]

Define mean squared displacement:

\[
\operatorname{MSD}_A(\Delta t)
=
\left\langle
\left|
\mathbf r_A(t+\Delta t)-\mathbf r_A(t)
\right|^2
\right\rangle.
\]

Compare against:

- passive diffusion;
- environmental drift;
- randomized phase controls;
- background material transport.

Directed motility requires displacement beyond these nulls.

Movement alone does not establish agency.

---

# 14. Measuring Self-Repair

## 14.1 Counterfactual twin design

At perturbation time \(t_d\), clone the complete universe state.

Run:

- an undamaged control universe \(U^{(0)}\);
- a damaged universe \(U^{(D)}\).

Apply a localized intervention only to candidate \(A\) in \(U^{(D)}\).

Use identical:

- environmental inputs;
- noise streams where possible;
- boundary conditions;
- update rules.

---

## 14.2 Damage interventions

Damage classes include:

- deletion of a material fraction;
- randomization of local configuration states;
- boundary puncture;
- displacement of components;
- resource deprivation;
- targeted removal of high-flux nodes;
- random removal of matched size.

Damage magnitude is recorded as:

\[
d\in[0,1].
\]

---

## 14.3 Macrostate distance

Let:

\[
D_A
\left(
U^{(D)}_t,U^{(0)}_t
\right)
\]

be a distance between the damaged and control candidate states after optimizing over irrelevant translation, rotation, and labeling symmetries.

Possible components include:

- material-distribution Wasserstein distance;
- state-distribution divergence;
- boundary mismatch;
- throughput mismatch;
- dynamical-mode mismatch.

---

## 14.4 Recovery score

Define:

\[
\boxed{
R_A(\tau)
=
1-
\frac{
D_A
\left(
U^{(D)}_{t_d+\tau},
U^{(0)}_{t_d+\tau}
\right)
}{
D_A
\left(
U^{(D)}_{t_d},
U^{(0)}_{t_d}
\right)
}.
}
\]

Interpretation:

- \(R_A=1\): complete return to the undamaged macrotrajectory;
- \(R_A=0\): no reduction in damage divergence;
- \(R_A<0\): divergence increased.

---

## 14.5 Passive-recovery controls

Self-repair requires outperforming:

- passive diffusion;
- equilibrium relaxation;
- matched nonpersistent patterns;
- randomized local rules;
- undriven material aggregation.

A structure that returns to an attractor because every state in the universe relaxes there demonstrates stability, but not necessarily active repair.

---

## 14.6 No repair optimization

For the strongest claim, the rule-search objective must not include:

- target-image loss;
- post-damage reconstruction;
- structure-specific recovery;
- damage episodes.

Repair is evaluated only after rule selection.

A separate experiment may deliberately evolve repair, but it must be labeled **selected repair**, not spontaneous repair.

---

# 15. Measuring Replication

Replication is easily confused with:

- growth;
- fragmentation;
- diffusion;
- repeated environmental nucleation;
- periodic pattern generation.

A replication claim requires all of the following.

## 15.1 Parent identification

A persistent candidate \(A\) must exist before descendant formation.

## 15.2 Material or causal lineage

There must be measurable transfer from the parent process to the candidate descendants.

## 15.3 Organizational inheritance

Descendants must reproduce a dynamical organization, not merely share material.

Let:

\[
\Sigma_A
\]

be a feature vector describing the parent’s dynamical signature.

Require:

\[
d_\Sigma
\left(
\Sigma_{\mathrm{child}},
\Sigma_{\mathrm{parent}}
\right)
<
\epsilon_\Sigma.
\]

## 15.4 Independent persistence

After separation, parent and child must continue as independently tracked structures for a minimum null-adjusted interval.

## 15.5 Repetition

At least one descendant must itself retain the capacity to produce another descendant under comparable conditions.

Without repeated lineage, the event is reproduction-like fragmentation rather than demonstrated replication.

---

# 16. Causal Individuality

A structure may be spatially localized but causally dominated by its environment.

IF Theory will test whether a candidate is a meaningful macro-unit.

## 16.1 Predictive closure

Let \(A_t\) represent candidate macrostate and \(E_t\) its local environment.

Compare:

\[
I(A_t;A_{t+\tau})
\]

with:

\[
I(E_t;A_{t+\tau}\mid A_t).
\]

A candidate with greater internal predictive continuity is more self-determined, though this remains correlational.

---

## 16.2 Intervention tests

Perform matched interventions on:

- internal candidate states;
- nearby environmental states;
- arbitrary regions of equal size.

Measure changes in:

- candidate persistence;
- future macrostate;
- throughput;
- movement.

A genuine candidate unit should exhibit a reproducible intervention structure distinct from arbitrary partitions.

---

## 16.3 Causal emergence

Microstate and macrostate causal models may be compared using intervention-based effective-information measures. A macrodescription that predicts interventions more selectively than the corresponding noisy or degenerate microdescription may possess causal-emergence evidence. Such evidence does not alone prove life, but it can justify treating the detected structure as a useful causal unit. citeturn795026search8

---

# 17. Rule Complexity

A system with a million-parameter neural rule may generate impressive structures while providing little evidence that minimal laws suffice.

Every rule must report a complexity measure.

Possible measures include:

- number of local state variables;
- neighborhood radius;
- transition-table size;
- parameter count;
- description length;
- compressed source-code length;
- circuit complexity.

Let:

\[
K(\theta)
\]

be a declared rule-complexity estimate.

Scientific comparisons should report both behavior and rule complexity.

The objective is not necessarily the absolutely shortest rule. It is to avoid hiding the organism inside a vast rule table.

---

# 18. Search Strategy

## 18.1 Discovery stage

Search may use:

- random sampling;
- Latin-hypercube sampling;
- Bayesian optimization;
- evolutionary novelty search;
- quality-diversity methods;
- curiosity-driven exploration;
- AI-assisted proposal generation.

Automated artificial-life search has already shown that foundation models and curiosity-driven systems can uncover diverse behaviors across several substrates. IF Theory uses these methods as discovery tools, not as proof. citeturn550023academia38turn795026academia50

---

## 18.2 Discovery objectives

Permitted broad objectives include:

- behavioral diversity;
- temporal nonstationarity;
- localization diversity;
- multiscale entropy;
- compression complexity;
- novelty relative to previous runs.

For the strongest repair claim, discovery may not optimize the repair metric.

For the strongest replication claim, discovery may not directly optimize descendant count.

---

## 18.3 Confirmation stage

Before confirmatory runs:

- freeze the rule;
- freeze the detector;
- freeze the classification criteria;
- freeze parameter ranges;
- reserve unseen seeds and perturbations;
- assign an independent evaluator.

No manual deletion of failed seeds is permitted.

---

## 18.4 Neighborhood robustness

A rule is stronger when nearby parameters also produce related behavior.

Define parameter robustness:

\[
\rho_\theta
=
P_{\theta'\sim\mathcal N(\theta,\Sigma)}
\left[
\text{behavior persists}
\right].
\]

A phenomenon occurring at one isolated floating-point setting may be computationally interesting but physically fragile.

---

# 19. Core Hypotheses

## ES-H1 — Spontaneous-localization hypothesis

Some resource-conserving local-rule families generate persistent localized structures from generic perturbations without target morphology or hand-designed organism seeds.

### Falsifier

All persistent localized structures require:

- specially constructed initial patterns;
- explicit target optimization;
- or isolated parameter values that fail under minimal perturbation.

---

## ES-H2 — Throughput-maintenance hypothesis

Some persistent structures maintain organization through continuing resource throughput rather than passive static stability.

### Prediction

Removing the resource gradient will cause loss of dynamic maintenance after a characteristic depletion time.

### Falsifier

Persistence is unaffected by resource removal, showing that the structure is static or that resource dependence was misidentified.

---

## ES-H3 — Spontaneous-repair hypothesis

Some structures recover from novel damage despite the rule search never evaluating damage or reconstruction.

### Falsifier

Recovery disappears under:

- held-out damage types;
- matched passive-relaxation controls;
- translation-invariant macrostate comparison;
- unseen seeds.

---

## ES-H4 — Causal-individuality hypothesis

Automatically detected persistent structures exhibit stronger internal predictive and intervention-based closure than arbitrary matched regions.

### Falsifier

Candidate boundaries provide no more causal coherence than random or geometrically similar partitions.

---

## ES-H5 — Replication hypothesis

Some resource-conserving rules generate persistent structures capable of producing dynamically independent descendants with inherited organization.

### Falsifier

Apparent replication reduces to:

- fragmentation;
- diffusion;
- repeated external nucleation;
- or nonheritable pattern repetition.

---

## ES-H6 — Robust-phase hypothesis

Persistent and repair-capable structures occupy finite regions of parameter space rather than isolated fine-tuned points.

### Falsifier

The phenomenon disappears under small parameter, numerical, or seed changes.

---

## ES-H7 — Cross-substrate hypothesis

At least one qualitative phase boundary transfers between IF-RC0, IF-RC1, and an independently designed conservative substrate.

### Falsifier

Every result depends on implementation-specific artifacts.

---

## ES-H8 — Thermodynamic-maintenance hypothesis

In RC1, dynamically maintained organization requires positive environmental entropy production or consumption of nonequilibrium free energy.

### Falsifier

The structure persistently restores internal order without an accounted compensating flow.

---

# 20. Primary Experiments

## Experiment 1 — Conservation audit

Run random local states and transitions.

Verify:

\[
\Delta M_a=0
\]

for every conserved material channel and:

\[
\Delta
\left(
F+W+W_{\mathrm{out}}
\right)
=0
\]

for closed RC0 experiments.

This experiment must pass before pattern search begins.

---

## Experiment 2 — Null dynamics

Run:

- homogeneous deterministic states;
- randomized states;
- zero-resource conditions;
- disabled-interaction conditions;
- shuffled-rule controls.

Establish null distributions for:

- localization;
- lifetime;
- apparent repair;
- lineage events;
- complexity.

---

## Experiment 3 — Rule-space phase survey

Sweep rule parameters and classify all outcomes into phases P0–P9.

Generate phase diagrams rather than highlighting isolated attractive runs.

---

## Experiment 4 — Seed independence

For each candidate rule, test thousands of generic seeds.

Report:

- probability of structure formation;
- formation time;
- structure diversity;
- failure modes.

---

## Experiment 5 — Gradient dependence

Vary:

- resource-input rate;
- sink rate;
- spatial source distribution;
- environmental volatility.

Test whether dynamic structures occupy a bounded nonequilibrium region.

---

## Experiment 6 — Perturbation and repair

Apply held-out damage classes.

Compare recovery against:

- undamaged twins;
- passive patterns;
- randomized rule controls;
- nonpersistent structures.

---

## Experiment 7 — Causal boundary

Intervene on:

- candidate interiors;
- candidate boundaries;
- nearby environment;
- random matched regions.

Test whether detected boundaries have distinctive causal significance.

---

## Experiment 8 — Replication and lineage

Construct lineage graphs from automatically tracked candidates.

Require organizational similarity and descendant independence.

---

## Experiment 9 — Rule simplification

Starting from successful rules:

- remove parameters;
- quantize values;
- reduce neighborhood radius;
- prune state channels.

Determine which components are necessary.

---

## Experiment 10 — Cross-implementation reproduction

Reimplement the same conceptual rule family independently.

Compare:

- phase boundaries;
- structure statistics;
- repair behavior;
- conservation.

---

# 21. Deterministic Jupyter-Notebook Program

## Notebook 03A — RC0 State and Ledger

Implement:

- local state representation;
- pairwise antisymmetric transport;
- resource conversion;
- closed and driven boundaries;
- exact ledger assertions.

Primary output:

\[
\max_t|\Delta C_{\mathrm{total}}(t)|.
\]

---

## Notebook 03B — Conservation Property Tests

Use randomized property-based tests for:

- mass conservation;
- capacity conservation;
- nonnegativity;
- translation invariance;
- neighbor relabeling;
- zero-coupling limits.

---

## Notebook 03C — Null Universe Catalog

Generate null distributions for all structure metrics.

The detection thresholds used later must originate here.

---

## Notebook 03D — Structure Detector

Implement:

- activity estimation;
- thresholding;
- connected components;
- optimal-transport tracking;
- lineage construction;
- localization and persistence measures.

Validate using synthetic moving objects with known ground truth.

---

## Notebook 03E — Minimal Rule Sweep

Search low-complexity rule families.

Produce complete phase maps, not selected screenshots.

---

## Notebook 03F — Resource-Gradient Sweep

Map behavior against:

\[
\text{input rate}
\times
\text{degradation rate}
\times
\text{transport rate}.
\]

---

## Notebook 03G — Counterfactual Damage Twins

Implement paired damaged and undamaged universes.

Calculate:

\[
R_A(\tau).
\]

---

## Notebook 03H — Passive-Relaxation Controls

Determine whether apparent repair exceeds ordinary attractor return.

---

## Notebook 03I — Causal Boundary Tests

Estimate internal and environmental intervention effects.

Compare detected candidates against matched random regions.

---

## Notebook 03J — Replication and Lineage

Detect:

- parent formation;
- split events;
- descendant persistence;
- inherited dynamical signatures;
- multigeneration continuation.

---

## Notebook 03K — RC1 Thermodynamic Validation

Implement a small stochastic version with exact state energies and detailed-balance-compatible transitions.

Validate:

- work;
- heat;
- entropy production;
- equilibrium distribution;
- fluctuation statistics.

---

## Notebook 03L — Cross-Substrate Replication

Repeat principal results using:

1. IF-RC0;
2. IF-RC1;
3. an independent mass-conserving continuous or lattice-gas model.

---

## Notebook 03M — Adversarial Audit

A red-team coding agent attempts to show that the reported structures arise from:

- threshold choice;
- numerical clipping;
- periodic boundaries;
- finite precision;
- hand-picked seeds;
- detector leakage;
- implicit targets;
- uncounted resources.

---

# 22. Reproducibility Record

Each simulation emits:

```yaml
experiment_id: if-emergent-structure-03
paper_version: null
git_commit: null
environment_hash: null
implementation: IF-RC0
rule_family: null
rule_parameters: {}
rule_description_length: null

domain_shape: null
boundary_condition: null
initial_condition_class: null
seed: 65537
time_steps: null

initial_material: {}
final_material: {}
material_residual: {}
initial_capacity: null
final_fuel: null
final_waste: null
exported_work: null
capacity_residual: null

structure_detector_version: null
detection_threshold_source: null
candidate_count: null
candidate_metrics: []
lineage_graph_hash: null

damage_protocol: null
recovery_metrics: {}
causal_interventions: {}

invariant_failures: []
result_hash: null
```

Raw trajectories or deterministic replay instructions must be retained for every published figure.

---

# 23. Statistical Standards

## 23.1 Unit of analysis

The unit may be:

- simulation run;
- candidate structure;
- rule;
- parameter neighborhood;
- lineage.

The paper must not treat thousands of time steps from one run as thousands of independent samples.

---

## 23.2 Seed multiplicity

Every confirmatory rule is tested across a preregistered number of independent seeds.

Report:

- success probability;
- confidence interval;
- median formation time;
- failure distribution.

---

## 23.3 Multiple-search correction

If millions of rules are explored, the existence of one extreme pattern is expected.

Confirmatory evidence must come from:

- held-out rule neighborhoods;
- independent reruns;
- preregistered tests;
- replication in another substrate.

---

## 23.4 Detector robustness

Repeat analysis with reasonable alternative:

- thresholds;
- clustering methods;
- tracking costs;
- macrostate distances.

The result is strong only if its qualitative classification is stable.

---

## 23.5 Negative results

Report rule-space regions containing:

- no structure;
- fragile structure;
- false repair;
- detector failures;
- conservation violations.

A phase map containing only successes is not credible.

---

# 24. Failure Modes

## 24.1 Target leakage

A target pattern appears in:

- the loss function;
- rule parameters;
- initial seed;
- detector;
- intervention design.

## 24.2 Free resources

Clipping, normalization, or boundary updates create material or capacity.

## 24.3 Detector hallucination

The tracking algorithm joins unrelated fluctuations into a persistent object.

## 24.4 Periodic-boundary artifact

A moving pattern interacts with its own wake across a toroidal domain.

## 24.5 Numerical attractor

Behavior exists only at one floating-point precision, grid resolution, or update order.

## 24.6 Human cherry-picking

Only attractive trajectories are shown while most seeds fail.

## 24.7 Repair-by-attractor

All damaged states relax toward the same fixed pattern, so apparent repair does not reflect a structure-specific process.

## 24.8 Fragmentation called replication

One object breaks into pieces without inherited self-maintaining organization.

## 24.9 Static order called life

A frozen low-throughput pattern is described as self-maintaining.

## 24.10 Search-objective contamination

The search algorithm directly optimizes the property later claimed to have emerged spontaneously.

## 24.11 Rule-table organism

The local transition rule contains enough complexity to encode the structure implicitly.

## 24.12 Thermodynamic overclaim

Abstract RC0 resource tokens are described as literal energy or entropy without an RC1 derivation.

---

# 25. What Would Count as Success?

## Level 1 — Valid conservative substrate

The implementation obeys its exact ledgers and produces reproducible phase diagrams.

This is necessary but not scientifically novel by itself.

## Level 2 — Target-free persistent localization

Simple rules produce robust localized structures from generic perturbations.

This is an artificial-life result, but related phenomena already exist.

## Level 3 — Spontaneous repair

Held-out perturbation recovery emerges without a repair objective and exceeds passive controls.

This would be more significant.

## Level 4 — Causal individuality

Detected structures possess interventionally meaningful boundaries and macrostate closure.

This could link self-organization to objective individuality.

## Level 5 — Reproducible replication

Structures produce heritable, dynamically independent descendants across multiple generations without a reproduction objective.

This would be a strong origins-of-life result.

## Level 6 — Cross-substrate law

One dimensionless phase boundary predicts structure, maintenance, or repair across independent conservative substrates.

This would be the most important outcome.

---

# 26. Novelty Assessment

## Already established elsewhere

The following are not novel IF claims:

- local rules can form global patterns;
- cellular automata can contain persistent moving structures;
- continuous automata can produce organism-like forms;
- trained cellular automata can regenerate target patterns;
- mass-conserving automata can support evolutionary dynamics;
- automated search can discover artificial-life behavior.

## Potential IF novelty

A result may be genuinely distinctive if it demonstrates:

1. exact resource accounting;
2. no target morphology;
3. no structure-specific reward;
4. automatic detection;
5. held-out spontaneous repair;
6. interventionally meaningful individuality;
7. multigeneration replication;
8. finite parameter robustness;
9. cross-substrate scaling.

The novelty resides in the discovered regularity, not in the project name or vocabulary.

---

# 27. Relationship to the Informational Battery

Paper 1 separated:

\[
B_{\mathrm{gross}}
\]

from:

\[
B_{\mathrm{op}}.
\]

Paper 3 creates systems in which persistent structures may alter access to resource flows.

For a candidate \(A\), later analysis may ask whether its organization increases:

\[
B_{\mathrm{op}}^A
\]

relative to scrambled or destroyed controls.

Paper 3 does not yet call that agency.

It establishes the structures on which Paper 2’s causal-work interventions can be performed.

---

# 28. Relationship to Agency

A persistent self-repairing structure may lack predictive agency.

Examples include:

- a stable reaction–diffusion pattern;
- an autocatalytic cycle;
- an attractor that reconstructs after disturbance;
- a fixed feedback loop.

Agency requires evidence that internal information is used causally to select actions and improve future outcomes.

The progression is:

\[
\boxed{
\text{persistence}
\not\Rightarrow
\text{repair}
\not\Rightarrow
\text{agency}
\not\Rightarrow
\text{consciousness}.
}
\]

Each implication must be tested rather than assumed.

---

# 29. Relationship to Biological Life

A successful IF structure would be an artificial-life candidate, not proof that biological life arose through identical rules.

A biologically relevant extension would require contact with:

- chemistry;
- catalysis;
- compartment formation;
- heredity;
- mutation;
- selection;
- metabolism;
- experimentally measurable reaction networks.

The computational substrate is a laboratory for identifying principles, not a substitute for chemistry.

---

# 30. Relationship to Cosmology

This paper does not simulate the Big Bang.

A lattice or graph with local resource dynamics may illuminate generic relationships among:

- symmetry breaking;
- structure formation;
- gradients;
- persistence;
- expansion in later work.

It does not automatically reproduce:

- spacetime;
- gravity;
- quantum fields;
- cosmic microwave background fluctuations;
- primordial nucleosynthesis;
- cosmic expansion.

Those claims require a separate physical bridge.

---

# 31. Criteria for Rejection or Major Revision

The Paper 3 program should be rejected or substantially revised if:

1. the resource ledger cannot be made exact;
2. all interesting structures require hand-designed seeds;
3. persistent structures occupy only isolated numerical points;
4. detected structures depend strongly on subjective thresholds;
5. repair disappears under counterfactual-twin analysis;
6. apparent replication is fragmentation or repeated nucleation;
7. causal boundaries are no better than arbitrary partitions;
8. rule complexity encodes the observed behavior;
9. RC0 findings disappear in physically disciplined RC1 systems;
10. no result survives independent implementation;
11. simpler existing substrates explain every observation with less machinery.

---

# 32. Conclusion

The existence of a beautiful pattern is not the scientific result.

The result must be a reproducible relationship among:

- local laws;
- resource constraints;
- nonequilibrium drive;
- persistent localization;
- maintenance;
- repair;
- causal individuality;
- replication.

The proposed IF standard is:

\[
\boxed{
\begin{gathered}
\text{A structure is emergent only when its identity, boundary,}
\\
\text{maintenance, and recovery are discovered after the dynamics,}
\\
\text{rather than encoded in the primitive state or objective.}
\end{gathered}
}
\]

The first computational question is:

\[
\boxed{
\text{Do simple resource-conserving local rules possess finite}
\atop
\text{regions of rule space in which persistent, throughput-maintained,}
\atop
\text{perturbation-resistant structures arise from generic fluctuations?}
}
\]

A positive answer would not yet establish life or agency.

It would establish a disciplined artificial universe in which those higher transitions can be tested.

A negative answer would be equally important. It would show that the proposed IF substrate lacks the generative capacity required by the larger theory.

The next paper asks whether allowing the causal domain itself to grow creates a reproducible intermediate regime favorable to organization:

\[
\boxed{
\textit{The Expansion–Complexity Window in Resource-Conserving Causal Networks.}
}
\]

---

# References

1. Turing, A. M. “The Chemical Basis of Morphogenesis.” *Philosophical Transactions of the Royal Society of London B* 237, 37–72 (1952). citeturn795026search7

2. Chan, B. W.-C. “Lenia—Biology of Artificial Life.” *Complex Systems* 28, 251–286 (2019). citeturn775349search2

3. Chan, B. W.-C. “Lenia and Expanded Universe.” Artificial Life Conference Proceedings (2020). citeturn775349search29

4. Mordvintsev, A., Randazzo, E., Niklasson, E. and Levin, M. “Growing Neural Cellular Automata.” *Distill* (2020). citeturn795026search3

5. Plantec, E. et al. “Flow-Lenia: Emergent Evolutionary Dynamics in Mass Conservative Continuous Cellular Automata.” *Artificial Life* 31, 228–248 (2025). citeturn795026search1

6. Hamon, G. et al. “Discovering Sensorimotor Agency in Cellular Automata Using Diversity Search.” (2024). citeturn550023academia39

7. Kumar, A. et al. “Automating the Search for Artificial Life with Foundation Models.” (2024). citeturn550023academia38

8. Michel, T. et al. “Exploring Flow-Lenia Universes with a Curiosity-Driven AI Scientist: Discovering Diverse Ecosystem Dynamics.” (2025). citeturn795026academia50

9. Pivato, M. “Conservation Laws in Cellular Automata.” (2001). citeturn775349academia82

10. Toffoli, T., Capobianco, S. and Mentrasti, P. “When—and How—Can a Cellular Automaton Be Rewritten as a Lattice Gas?” (2007). citeturn775349academia83

11. Perunov, N., Marsland, R. and England, J. “Statistical Physics of Adaptation.” *Physical Review X* 6, 021036 (2016). citeturn775349search8

12. Hoel, E. P., Albantakis, L. and Tononi, G. “Quantifying Causal Emergence Shows That Macro Can Beat Micro.” *Proceedings of the National Academy of Sciences* 110, 19790–19795 (2013). citeturn795026search8
