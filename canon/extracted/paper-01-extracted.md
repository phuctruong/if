<!-- extracted from ChatGPT msg [64] on 2026-07-18 -->

# The Informational Battery  
## Nonequilibrium Capacity, Accessible Organization, and the Physical Value of Information

**Author:** Phuc Vinh Truong  
**Series:** IF Theory Working Papers  
**Paper:** 1  
**Date:** July 17, 2026  
**Status:** Foundational theoretical proposal for simulation and falsification

---

## Abstract

The phrase **informational battery** suggests that organized physical systems may store a capacity that can later be discharged, preserved, restored, or made more useful through information processing. Taken literally and without constraints, however, the metaphor risks confusing information with energy and implying that intelligence can create useful work or reverse entropy without physical cost.

This paper develops a thermodynamically disciplined definition. An informational battery is not a new substance and information is not treated as an independent source of energy. Instead, the battery is defined operationally through three related quantities:

1. **gross nonequilibrium capacity**: the maximum work permitted by the physical state relative to a specified environment;
2. **operationally accessible capacity**: the net work that a specified class of physically embodied controllers can actually obtain;
3. **latent capacity**: the portion of gross capacity that exists physically but is inaccessible to those controllers.

Information matters when it changes the conversion between gross and accessible capacity. A map, memory, sensor, predictive model, or cooperative signal may allow a system to extract work that would otherwise remain inaccessible. The information-processing machinery must nevertheless pay the costs of sensing, storage, computation, error correction, action, communication, and resetting.

The central proposed distinction is therefore:

\[
\boxed{
\text{Information does not create the battery’s energy;}
\atop
\text{it can alter how much of the battery is causally accessible.}
}
\]

The paper defines physical recharge, accessibility recharge, structural recharge, and cooperative recharge. It introduces a computational framework based on Markov processes and resource-conserving artificial environments, specifies falsifiable hypotheses, and outlines a series of deterministic Jupyter notebooks. The strongest prospective result would be a substrate-independent relationship predicting how accessible work changes with predictive information and controller cost. No such universal law is assumed in advance.

---

## Keywords

Nonequilibrium free energy; information thermodynamics; accessible work; exergy; semantic information; predictive information; stochastic thermodynamics; agency; causal work; artificial life.

---

# 1. Introduction

Batteries do not create energy. They store a physically usable difference: chemical potential, electrical potential, pressure, concentration, temperature, height, or another departure from equilibrium. A charged battery can later discharge because its state permits a process that transfers energy while producing useful work.

Living and intelligent systems appear to possess a related but more complex property. They do not merely contain gradients. They contain mechanisms that locate, protect, redirect, and exploit gradients. A seed uses stored chemical organization to grow under suitable conditions. A bacterium uses sensors and internal states to move toward nutrients. An animal uses memory to return to a resource. A human uses a map to reach fuel that would otherwise remain inaccessible. Two systems may occupy the same environment and face the same available energy, yet obtain radically different amounts of useful work because one has more causally relevant information.

Established information thermodynamics already studies how measurement, feedback, memory, and erasure interact with work and entropy. Stochastic thermodynamics provides trajectory-level definitions of work, heat, and entropy production for well-defined nonequilibrium systems. Feedback-controlled engines demonstrate that information can affect the amount of work extracted, while Landauer-type results require the physical implementation of information processing to be included in the complete ledger. citeturn533893search4turn184356academia34turn184356academia36

Experimental work has verified Landauer-scale heat dissipation in controlled memory-erasure systems, including the approach to the expected bound for sufficiently slow erasure protocols. These results support a physical relationship between logically irreversible operations and thermodynamic cost; they do not imply that every abstract bit carries a fixed amount of energy or that possessing information automatically yields extractable work. citeturn719940search6turn719940search16

The thermodynamics of prediction adds an especially relevant insight. A physical system may retain information about past environmental states, but only some of that information predicts the future. Nonpredictive memory can increase dissipation without improving control. Efficient physical models should therefore preserve information that helps anticipate future environmental behavior rather than indiscriminately preserving the past. citeturn533893search3turn533893search7

Similarly, intervention-based semantic-information research distinguishes correlations that merely exist from correlations that contribute to a system’s viability. If selectively destroying a correlation harms the system’s ability to remain viable, that correlation carries information that is meaningful to the system under the chosen viability criterion. citeturn719940search4

The informational-battery proposal builds on these foundations but asks a distinct operational question:

\[
\boxed{
\text{How much physically available capacity exists, how much can a}
\atop
\text{particular system actually access, and how much of the difference}
\atop
\text{is caused by its information-processing organization?}
}
\]

The answer requires more than a single scalar called “information.” It requires an explicit physical boundary, reference environment, time horizon, controller class, work definition, intervention, and complete cost ledger.

---

# 2. The Problem with the Original Battery Metaphor

The intuitive battery metaphor contains several different claims that must be separated.

## 2.1 Physical storage

A system may be away from equilibrium and therefore capable, in principle, of delivering work.

Examples include:

- a charged electrochemical cell;
- a compressed gas;
- a thermal gradient;
- separated chemical reactants;
- concentrated nutrients;
- an elevated mass;
- an organized molecular state.

This is ordinary physical capacity.

---

## 2.2 Accessibility

A capacity may physically exist but remain inaccessible to a particular mechanism.

Fuel inside a locked container has chemical free energy. A system lacking a means to open the container cannot use it. A temperature gradient separated by an insulating barrier may exist without being available to a given engine. A food source may be geographically present but inaccessible to an organism without a sensor, map, or suitable movement policy.

Accessibility is therefore relational:

\[
\text{accessible to whom, using what mechanism, over what time?}
\]

---

## 2.3 Informational guidance

A controller may use information to choose among possible actions.

Examples include:

- selecting the correct gate;
- predicting the location of a future resource;
- distinguishing fuel from poison;
- coordinating with another agent;
- identifying the efficient sequence of operations;
- detecting when a system requires repair.

Information may increase obtainable work without increasing the gross energy physically stored in the environment.

---

## 2.4 Maintenance and computation costs

The controller is physical. It may require:

- sensors;
- memory;
- communication;
- computation;
- movement;
- error correction;
- cooling;
- maintenance;
- resetting.

A larger model may improve predictions while consuming more resources than it saves.

---

## 2.5 Recharge

The word “recharge” may refer to fundamentally different processes:

1. adding new free energy;
2. restoring damaged conversion machinery;
3. learning how to reach an existing resource;
4. reorganizing a resource into a more accessible form;
5. receiving resources or information from another system.

These cases must not be combined into a single unaccounted increase.

---

# 3. Scope and Assumptions

The formal development begins with classical isothermal stochastic systems because they permit clear definitions, exact calculations, and deterministic validation.

Let:

- \(S\) denote the designated battery system;
- \(C\) denote the controller physically coupled to it;
- \(E\) denote the environment and reference reservoirs;
- \(p(x)\) denote a probability distribution over physical states \(x\);
- \(E(x,\lambda)\) denote the energy of state \(x\) under control parameter \(\lambda\);
- \(T\) denote the temperature of a reference heat bath;
- \(\Pi\) denote a specified class of admissible control protocols;
- \(\tau\) denote the time horizon.

Every reported battery value is conditional on:

\[
\boxed{
(S,C,E,T,\Pi,\tau,\text{boundary conditions}).
}
\]

There is no observer-independent claim that a physical state contains one universally accessible quantity regardless of available mechanisms and time.

---

# 4. Gross Nonequilibrium Capacity

## 4.1 Nonequilibrium free energy

For a classical system with probability distribution \(p(x)\), define the nonequilibrium free energy:

\[
F[p,\lambda]
=
\sum_x p(x)E(x,\lambda)
-
T S[p],
\]

where:

\[
S[p]
=
-k_{\mathrm B}
\sum_x p(x)\ln p(x).
\]

Let the corresponding equilibrium distribution be:

\[
p_{\mathrm{eq}}(x\mid\lambda)
=
\frac{
e^{-\beta E(x,\lambda)}
}{
Z(\lambda)
},
\qquad
\beta=\frac{1}{k_{\mathrm B}T}.
\]

Under these assumptions:

\[
F[p,\lambda]-F[p_{\mathrm{eq}},\lambda]
=
k_{\mathrm B}T
D_{\mathrm{KL}}
\left(
p\parallel p_{\mathrm{eq}}
\right).
\]

Relative entropy consequently measures excess free energy in this defined isothermal setting. This relationship has been developed in stochastic and nonequilibrium thermodynamics, but it should not be generalized without modification to arbitrary gravitational, cosmological, quantum, or strongly driven systems. citeturn719940academia75turn719940academia74turn719940search32

---

## 4.2 Definition: gross battery capacity

For the restricted setting above, define the **gross IF capacity**:

\[
\boxed{
B_{\mathrm{gross}}
=
F[p,\lambda]
-
F[p_{\mathrm{eq}},\lambda].
}
\]

Its units are joules.

It measures a theoretical upper resource associated with departure from equilibrium under the stated reference conditions. It does not guarantee that an actual controller can extract all of it.

For more general engineering systems, an exergy-like measure relative to a specified environment may replace the isothermal free-energy difference. The exact measure must be chosen before an experiment and must preserve dimensional consistency.

---

## 4.3 Gross capacity is not organization alone

A visually ordered state may have little usable free energy.

A disordered-looking state may contain a large temperature or chemical gradient.

A long random bit string may have high Shannon entropy yet provide no useful work to a controller.

Therefore:

\[
\boxed{
B_{\mathrm{gross}}
\neq
\text{visual order}
\neq
\text{Shannon information}
\neq
\text{complexity}
\neq
\text{meaning}.
}
\]

---

# 5. Operationally Accessible Capacity

Gross capacity describes what may be thermodynamically available in principle. IF Theory is primarily interested in what a particular physically embodied system can actually convert.

## 5.1 Control protocols

A protocol \(\pi\in\Pi\) may include:

- measurements;
- memory updates;
- feedback;
- changes to control parameters;
- movement;
- chemical transitions;
- communication;
- resource transfer;
- resetting.

For each protocol, define:

\[
W_{\mathrm{out}}^\pi(\tau)
\]

as useful work exported to a designated work reservoir over horizon \(\tau\).

Define:

\[
C_{\mathrm{ctrl}}^\pi(\tau)
\]

as the complete controller cost, including all modeled costs of:

- sensing;
- memory acquisition;
- storage;
- prediction;
- computation;
- communication;
- error correction;
- actuation;
- maintenance;
- reset.

Imported work used directly by the controller must be subtracted rather than hidden as environmental assistance.

---

## 5.2 Definition: operational capacity

Define the **operationally accessible IF capacity**:

\[
\boxed{
B_{\mathrm{op}}
\left(
p,m;\Pi,\tau
\right)
=
\max
\left[
0,
\sup_{\pi\in\Pi}
\mathbb E
\left(
W_{\mathrm{out}}^\pi
-
C_{\mathrm{ctrl}}^\pi
\right)
\right],
}
\]

where \(m\) denotes the controller’s physically instantiated internal information state.

The value depends on:

- the physical state;
- the controller architecture;
- its memory;
- permitted actions;
- time horizon;
- costs;
- environmental boundary.

It is therefore an operational quantity, not a metaphysical property.

---

## 5.3 Accessibility bound

Within the specified isothermal framework and with complete accounting, a physically valid implementation should satisfy:

\[
\boxed{
0\leq
B_{\mathrm{op}}
\leq
B_{\mathrm{gross}}.
}
\]

A computed value above \(B_{\mathrm{gross}}\) indicates at least one of the following:

- an external resource was omitted;
- work was counted twice;
- controller costs were omitted;
- the system boundary was inconsistent;
- the gross-capacity measure was inappropriate;
- numerical or conceptual error occurred.

This inequality is a required consistency test within the defined model class, not a claim already established for every possible nonequilibrium system.

---

# 6. Latent Capacity and the Accessibility Gap

Define the **latent capacity**:

\[
\boxed{
B_{\mathrm{latent}}
=
B_{\mathrm{gross}}
-
B_{\mathrm{op}}.
}
\]

This is physically available capacity that remains operationally inaccessible to the selected controller class over the chosen horizon.

Define the accessibility efficiency:

\[
\boxed{
\eta_{\mathrm{access}}
=
\frac{
B_{\mathrm{op}}
}{
B_{\mathrm{gross}}
},
\qquad
0\leq\eta_{\mathrm{access}}\leq1,
}
\]

when \(B_{\mathrm{gross}}>0\).

Two systems can possess the same gross battery but different accessibility efficiencies.

This is the central informational-battery distinction:

\[
\boxed{
\text{The same physical gradient may support different amounts of}
\atop
\text{net useful work for differently organized systems.}
}
\]

---

# 7. The Role of Information

## 7.1 Information is not counted as joules

Shannon information is measured in bits or nats.

Free energy and work are measured in joules.

They may be related under specified physical processes, but they cannot be directly added:

\[
1\ \text{bit}
+
1\ \text{joule}
\]

is dimensionally meaningless.

Information affects the battery only through an explicitly modeled physical mechanism.

---

## 7.2 Information-enabled accessibility

Let \(m\) be an internal memory or model correlated with the environment.

Let \(\widetilde m\) be a controlled scrambling of that memory designed to destroy a specified predictive relationship while preserving:

- memory size;
- marginal state frequencies;
- approximate energetic cost;
- controller architecture;
- action capacity.

Define the raw accessibility contribution:

\[
\Delta B_m
=
B_{\mathrm{op}}(p,m;\Pi,\tau)
-
B_{\mathrm{op}}(p,\widetilde m;\Pi,\tau).
\]

If:

\[
\Delta B_m>0,
\]

the selected information causally increases operationally accessible capacity under the tested intervention.

If:

\[
\Delta B_m=0,
\]

the information is correlated with the environment but operationally irrelevant under the tested conditions.

If:

\[
\Delta B_m<0,
\]

maintaining or using the information is harmful relative to the scrambled control.

This definition is conceptually related to intervention-based semantic information, which evaluates whether destroying correlations reduces viability, but IF Theory applies the intervention to net accessible work and retains viability as a separate outcome. citeturn719940search4

---

## 7.3 Predictive versus historical information

Let:

\[
I_{\mathrm{past}}
=
I(M_t;E_{t-\tau_p})
\]

and:

\[
I_{\mathrm{pred}}
=
I(M_t;E_{t+\tau_f}\mid E_t).
\]

A memory may retain extensive historical information while providing little predictive value.

The thermodynamics-of-prediction literature argues that nonpredictive retained information is associated with inefficiency and dissipation in systems responding to stochastic environments. IF Theory consequently predicts that accessible-work gain should track interventionally useful predictive information more closely than raw memory capacity or past mutual information. citeturn533893search3turn533893search7

---

# 8. The Informational Battery as a Structured State

The informational battery should not be represented by one mystical scalar.

For a specified system, define:

\[
\boxed{
\mathbb B_{\mathrm{IF}}
=
\left[
B_{\mathrm{gross}},
B_{\mathrm{op}},
B_{\mathrm{latent}},
\eta_{\mathrm{access}},
\dot S_{\mathrm{total}},
\Pi,
\tau
\right].
}
\]

This record states:

- how much nonequilibrium capacity exists;
- how much the controller can access;
- how much remains latent;
- the conversion efficiency;
- the associated entropy-production rate;
- the admissible controller class;
- the evaluation horizon.

Where information interventions are performed, also report:

\[
\Delta B_m,
\qquad
I_{\mathrm{pred}},
\qquad
I_{\mathrm{past}},
\qquad
C_{\mathrm{memory}},
\qquad
C_{\mathrm{computation}}.
\]

This prevents one favorable composite score from hiding poor thermodynamic behavior.

---

# 9. Four Types of Recharge

## 9.1 Physical recharge

A **physical recharge** increases gross nonequilibrium capacity:

\[
\Delta B_{\mathrm{gross}}>0.
\]

Examples include:

- applying electrical work to a battery;
- concentrating chemical fuel;
- restoring a pressure difference;
- creating a thermal gradient;
- importing nutrients;
- receiving radiant energy.

The external source must appear in the energy and entropy ledgers.

---

## 9.2 Accessibility recharge

An **accessibility recharge** increases operational capacity without necessarily increasing gross capacity:

\[
\Delta B_{\mathrm{op}}>0,
\qquad
\Delta B_{\mathrm{gross}}\approx0.
\]

This may occur when a system:

- learns a route;
- discovers a conversion sequence;
- improves a sensor;
- repairs a control pathway;
- forms a useful representation;
- receives a map;
- coordinates with another agent.

The system has not created more physical energy. It has reduced:

\[
B_{\mathrm{latent}}.
\]

Accessibility recharge is the strongest scientifically defensible interpretation of the intuition that intelligence can “recharge” a system.

The process still has costs:

\[
C_{\mathrm{learning}}
+
C_{\mathrm{memory}}
+
C_{\mathrm{reconfiguration}}
>0.
\]

The total thermodynamic ledger must remain valid.

---

## 9.3 Structural recharge

A **structural recharge** restores damaged conversion machinery.

Suppose gross capacity remains available, but controller damage reduces operational access:

\[
B_{\mathrm{op}}^{\mathrm{damaged}}
<
B_{\mathrm{op}}^{\mathrm{intact}}.
\]

Repair increases:

\[
B_{\mathrm{op}}
\]

by restoring the system’s ability to convert existing gradients.

Repair itself consumes capacity and produces entropy.

---

## 9.4 Cooperative recharge

A **cooperative recharge** occurs when another system supplies:

- physical resources;
- information;
- repair;
- protection;
- communication;
- shared infrastructure.

For agent \(i\):

\[
\Delta B_{\mathrm{op},i}>0.
\]

But the total multi-agent ledger must include the donor’s costs:

\[
\Delta B_{\mathrm{total}}
=
\sum_i \Delta B_i
-
C_{\mathrm{transfer}}
-
C_{\mathrm{coordination}}.
\]

Cooperation can produce a genuine net gain when specialization, pooling, error correction, or coordination reduces losses. It cannot create free capacity by ignoring one participant’s expenditure.

---

# 10. Discharge

A battery discharges when gross or operational capacity is converted into work, maintenance, computation, or dissipated energy.

## 10.1 Productive discharge

\[
\Delta B_{\mathrm{gross}}<0,
\qquad
W_{\mathrm{useful}}>0.
\]

Examples:

- movement;
- synthesis;
- construction;
- computation;
- reproduction.

---

## 10.2 Wasteful discharge

\[
\Delta B_{\mathrm{gross}}<0,
\qquad
W_{\mathrm{useful}}\approx0.
\]

Examples:

- uncontrolled leakage;
- friction;
- heat loss;
- failed computation;
- destructive conflict;
- storing irrelevant information.

---

## 10.3 Maintenance discharge

A system may continuously consume capacity merely to remain within a viable state:

\[
\frac{dB_{\mathrm{gross}}}{dt}<0,
\qquad
\frac{dV}{dt}\approx0.
\]

This is not recharge. It is sustained local organization through ongoing discharge and environmental input.

---

# 11. A Minimal Stochastic-Thermodynamic Implementation

To make the theory executable, the first reference implementation will use a finite-state continuous-time Markov process.

Let the system occupy state \(i\) with probability \(p_i(t)\).

Transition rates are:

\[
k_{ij}
\]

from state \(i\) to state \(j\).

The master equation is:

\[
\frac{dp_i}{dt}
=
\sum_j
\left[
p_jk_{ji}
-
p_ik_{ij}
\right].
\]

Define probability current:

\[
J_{ij}
=
p_ik_{ij}
-
p_jk_{ji}.
\]

A standard entropy-production expression is:

\[
\dot S_{\mathrm{tot}}
=
\frac{k_{\mathrm B}}{2}
\sum_{i,j}
J_{ij}
\ln
\frac{
p_ik_{ij}
}{
p_jk_{ji}
}
\geq0.
\]

Stochastic thermodynamics uses such Markov and Langevin descriptions to define work, heat, and entropy production along nonequilibrium trajectories. For open nonequilibrium steady states, housekeeping dissipation must be distinguished from free-energy relaxation. citeturn533893search4turn719940academia74turn719940search17

The implementation will support both:

1. deterministic integration of the master equation;
2. stochastic Gillespie trajectories.

The two methods must agree within preregistered statistical tolerances.

---

# 12. Thought Experiment I: The Locked Fuel Chambers

Consider two identical chambers, left and right.

Exactly one chamber contains fuel capable of delivering work \(W_F\).

Opening a chamber costs:

\[
C_O.
\]

The system can open only one chamber before the fuel expires.

Let:

\[
Y\in\{L,R\}
\]

identify the fuel location.

A memory variable:

\[
M\in\{L,R\}
\]

is correct with probability:

\[
P(M=Y)=q.
\]

A controller opens the chamber indicated by \(M\).

Ignoring learning costs initially:

\[
\mathbb E[W_{\mathrm{net}}]
=
qW_F-C_O.
\]

A random controller obtains:

\[
\mathbb E[W_{\mathrm{random}}]
=
\frac{1}{2}W_F-C_O.
\]

The raw information-enabled gain is:

\[
\Delta B_M
=
\left(
q-\frac12
\right)W_F.
\]

When memory cost \(C_M\) is included:

\[
\Delta B_M^{\mathrm{net}}
=
\left(
q-\frac12
\right)W_F
-
C_M.
\]

The information is beneficial only if:

\[
\boxed{
\left(
q-\frac12
\right)W_F
>
C_M.
}
\]

The fuel energy exists regardless of the memory.

The memory changes the probability that the controller reaches it.

This is the simplest informational battery.

---

# 13. Thought Experiment II: Same Free Energy, Different Accessibility

Construct two environments with equal gross nonequilibrium free energy.

## Environment A

Fuel is concentrated behind one identifiable gate.

## Environment B

The same fuel is distributed among many visually identical gates, most of which contain traps that consume opening work.

The gross capacity is matched:

\[
B_{\mathrm{gross}}^{A}
=
B_{\mathrm{gross}}^{B}.
\]

A controller without a map may obtain:

\[
B_{\mathrm{op}}^{A}
>
B_{\mathrm{op}}^{B}.
\]

A controller with an accurate map may obtain:

\[
B_{\mathrm{op}}^{A}
\approx
B_{\mathrm{op}}^{B}.
\]

This experiment demonstrates that operational capacity depends on both physical organization and controller information.

It also demonstrates why “the information content of the environment” is incomplete. The relevant quantity is the interaction between environmental structure and the controller capable of exploiting it.

---

# 14. Thought Experiment III: Predictive Memory in a Changing Environment

Let a resource switch between two locations according to a Markov process:

\[
P(Y_{t+1}=Y_t)=r.
\]

When:

\[
r=\frac12,
\]

the next location is unpredictable from the present.

When:

\[
r\rightarrow1,
\]

the environment is persistent.

An agent may store the last \(L\) observations.

Memory cost increases with \(L\):

\[
C_M(L)=c_0+c_1L.
\]

Prediction accuracy may improve initially and then saturate:

\[
q(L,r).
\]

The operational battery is:

\[
B_{\mathrm{op}}(L,r)
=
q(L,r)W_F
-
C_O
-
C_M(L)
-
C_{\mathrm{compute}}(L).
\]

The predicted optimum is:

\[
\boxed{
L^*(r)
=
\arg\max_L B_{\mathrm{op}}(L,r).
}
\]

Expected behavior:

- \(L^*\) is small in unpredictable environments;
- \(L^*\) increases where deeper temporal patterns exist;
- excessive memory becomes harmful after predictive gain saturates;
- raw memory capacity is not equivalent to accessible work.

---

# 15. Core Hypotheses

## IB-H1 — Accessibility-gap hypothesis

Systems with equal gross nonequilibrium capacity can have different operational capacity because of differences in controller organization and accessible information.

### Falsifier

After complete cost accounting, controller information and organization add no reproducible predictive value beyond gross free energy and action capacity.

---

## IB-H2 — Accessibility-recharge hypothesis

Learning or reorganization can increase:

\[
B_{\mathrm{op}}
\]

while leaving:

\[
B_{\mathrm{gross}}
\]

approximately unchanged.

### Falsifier

Every apparent accessibility increase is fully explained by uncounted physical energy input or a change in the gross state.

---

## IB-H3 — Predictive-value hypothesis

Operational capacity is more strongly related to interventionally preserved predictive information than to total memory, historical information, or Shannon entropy alone.

### Falsifier

Raw memory capacity or nonpredictive information predicts net work equally well after controller costs are matched.

---

## IB-H4 — Finite-model hypothesis

For environments with bounded predictability and nonzero model cost, operational capacity is maximized at finite model complexity.

\[
0<L^*<\infty.
\]

### Falsifier

More memory and computation always improve net accessible work under realistic nonzero costs.

---

## IB-H5 — Cross-substrate scaling hypothesis

A dimensionless accessibility ratio can predict useful-information transitions across multiple simulated substrate classes.

A candidate is:

\[
\Pi_B
=
\frac{
\Delta B_m
}{
C_{\mathrm{information}}
}.
\]

A positive information advantage requires:

\[
\Pi_B>1.
\]

### Falsifier

Different substrates require unrelated definitions, arbitrary scaling factors, or incompatible thresholds.

---

## IB-H6 — Cooperative-accessibility hypothesis

Cooperation can increase total operational capacity when the combined accessibility gain exceeds communication, coordination, and transfer costs.

### Falsifier

All apparent gains disappear under complete multi-agent accounting.

---

# 16. What Would Be Scientifically New?

Nonequilibrium free energy is established.

Information-dependent work extraction is established.

Landauer costs are established.

Predictive information and thermodynamic efficiency are established.

Intervention-based semantic information is established. citeturn719940search32turn184356academia34turn533893search7turn719940search4

The possible novelty is therefore not the statement:

> “Information can help a system use energy.”

The potentially original contribution would be one or more of the following:

1. A precise decomposition of gross, accessible, and latent nonequilibrium capacity that remains useful across artificial-life systems.
2. A reproducible intervention protocol isolating the work value of internal predictive information.
3. A dimensionless law predicting when information-processing machinery produces net positive operational capacity.
4. A cross-substrate phase boundary that applies to different environments, controllers, and physical implementations.
5. A demonstrated bridge from accessibility gain to the later emergence of agency.

These remain research objectives, not established discoveries.

---

# 17. Notebook Program

## Notebook 01A — Nonequilibrium Free-Energy Validation

**Purpose:** Verify:

\[
F[p]-F[p_{\mathrm{eq}}]
=
k_{\mathrm B}T
D_{\mathrm{KL}}
(p\parallel p_{\mathrm{eq}})
\]

for finite classical systems.

**Tasks:**

- two-state system;
- three-state system;
- randomly generated Hamiltonians;
- exact symbolic comparison;
- numerical tolerance tests;
- relaxation under detailed balance.

**Pass condition:** Agreement to numerical tolerance and monotonic decay during uncontrolled relaxation.

---

## Notebook 01B — Master Equation versus Gillespie

**Purpose:** Validate deterministic and stochastic implementations.

**Tasks:**

- solve the master equation;
- generate stochastic trajectories;
- compare state probabilities;
- compare mean work;
- compare heat;
- compare entropy production;
- verify convergence with trajectory count.

---

## Notebook 01C — Landauer Memory Reset

**Purpose:** Reproduce the qualitative and quantitative reset-cost behavior of a one-bit memory model.

**Tasks:**

- double-well or two-state representation;
- finite-time erasure protocols;
- quasistatic limit;
- heat-distribution calculation;
- protocol-speed dependence.

**Scientific purpose:** Validate the information-cost ledger before using memory in agents.

---

## Notebook 01D — Locked Fuel Chambers

**Purpose:** Measure information-enabled access to fixed physical capacity.

**Sweep:**

- memory accuracy \(q\);
- fuel work \(W_F\);
- opening cost \(C_O\);
- memory cost \(C_M\).

**Prediction:**

\[
\Delta B_M^{\mathrm{net}}
=
\left(q-\frac12\right)W_F-C_M.
\]

The simulation must reproduce this analytical result.

---

## Notebook 01E — Equal Gross Capacity, Unequal Access

**Purpose:** Construct physically matched environments with different structural accessibility.

**Compare:**

- no controller;
- reactive controller;
- mapped controller;
- predictive controller;
- omniscient upper bound.

**Outputs:**

\[
B_{\mathrm{gross}},
B_{\mathrm{op}},
B_{\mathrm{latent}},
\eta_{\mathrm{access}}.
\]

---

## Notebook 01F — Predictive versus Nonpredictive Memory

**Purpose:** Test whether predictive information explains accessible-work gain better than total memory.

**Controls:**

- intact memory;
- shuffled memory;
- time-reversed memory;
- same-size random memory;
- memory of irrelevant variables.

---

## Notebook 01G — Accessibility Recharge

**Purpose:** Demonstrate a case where learning raises operational capacity without raising gross capacity.

**Accounting:**

- energy spent learning;
- entropy produced;
- map accuracy;
- post-learning work extraction;
- break-even time.

**Pass condition:** The accessibility gain survives all learning and memory costs.

---

## Notebook 01H — Cooperative Recharge

**Purpose:** Test whether two agents can jointly access capacity unavailable individually.

**Compare:**

- isolated action;
- communication;
- division of labor;
- deceptive signaling;
- noisy signaling;
- communication cost.

---

# 18. Validation and Reproducibility

Every experiment will save:

```yaml
experiment_id: informational-battery-01
git_commit: null
environment_hash: null
model_class: finite_markov
temperature_kelvin: null
time_horizon: null
controller_class: null
random_seed: 65537
gross_capacity_joules: null
operational_capacity_joules: null
latent_capacity_joules: null
controller_cost_joules: null
entropy_production_joules_per_kelvin: null
information_metrics: {}
invariant_failures: []
result_hash: null
```

Required tests include:

- energy conservation;
- nonnegative total entropy production where applicable;
- dimensional validation;
- zero-information baseline;
- zero-cost limit;
- infinite-cost limit;
- perfect-information limit;
- random-controller limit;
- controller relabeling invariance;
- agreement between independent implementations.

---

# 19. Failure Modes

## 19.1 Hidden external work

An agent may appear to obtain information-powered work because the sensor, controller, or data provider is externally powered.

**Correction:** Expand the physical boundary or log imported work.

---

## 19.2 Free measurement

Some idealized models treat measurement as costless.

Measurement may sometimes approach low energetic cost under particular reversible implementations, but sensing hardware, memory preparation, communication, control, and resetting cannot all be ignored in a full operational system. Information-thermodynamic treatments require the relevant participating elements and reservoirs to be modeled consistently. citeturn184356academia36turn533893academia37

---

## 19.3 Double-counting information value

A controller’s improved work output must not be counted once as output and again as stored informational energy.

Information modifies the protocol. It is not an extra joule reservoir.

---

## 19.4 Inappropriate equilibrium reference

The quantity:

\[
F[p]-F[p_{\mathrm{eq}}]
\]

depends on the chosen Hamiltonian, bath, and equilibrium reference.

A different reference environment can yield a different capacity.

---

## 19.5 Controller-class dependence

A poor controller may leave most capacity latent.

A hypothetical omniscient controller may approach the theoretical upper bound.

Every result must report \(\Pi\).

---

## 19.6 Horizon dependence

A resource may be inaccessible over one second but accessible over one year.

Every result must report \(\tau\).

---

## 19.7 Coarse-graining dependence

Apparent information and entropy can change with the selected state representation.

Robustness across reasonable coarse-grainings is required.

---

## 19.8 Reward leakage

In artificial-life simulations, a controller must not receive a hidden reward for information use.

Its advantage must arise from physical resource conversion and survival under the same rules as its competitors.

---

# 20. Relationship to Life and Agency

The informational battery is not yet a theory of life.

A fire can access chemical gradients without memory.

A heat engine can convert energy without agency.

A feedback device can use information without consciousness.

The battery framework becomes relevant to agency only when a bounded system maintains internal information that causally improves future access to capacity across changing conditions.

That transition is the subject of Paper 2:

\[
\boxed{
\textit{The IF Causal-Work Principle:}
\atop
\textit{When Predictive Information Becomes Physical Agency.}
}
\]

Paper 1 provides the denominator and physical ledger required for that later claim.

---

# 21. Relationship to Cosmology

This paper does not establish that the universe is literally an informational battery.

Applying the framework cosmologically would require defining:

- the physical state space;
- the equilibrium or reference state;
- the relevant energy functional;
- the meaning of accessible work for the universe;
- system boundaries;
- gravitational entropy;
- expansion dynamics;
- quantum degrees of freedom.

A free-energy identity derived for a finite isothermal Markov system cannot simply be applied to the entire universe.

The cosmological phrase “informational battery” should therefore remain a conjectural analogy until a covariant physical model supplies the missing definitions.

---

# 22. Relationship to Meaning and Moral Recharge

Accessibility recharge can describe how information, repair, or cooperation increases a system’s capacity to act.

It does not establish that all increases in agency are morally good.

A coercive system may increase its own operational capacity by reducing that of others.

A later MaxLove analysis must therefore track distributions:

\[
B_{\mathrm{op},1},
B_{\mathrm{op},2},
\dots,
B_{\mathrm{op},N}
\]

and future viable action spaces rather than only aggregate output.

Moral conclusions require explicit normative premises beyond the thermodynamic equations.

---

# 23. Principal Predictions

The informational-battery framework makes six initial predictions.

## Prediction 1

Systems with matched gross nonequilibrium capacity will exhibit different operational capacity when controller information and action mechanisms differ.

## Prediction 2

Scrambling predictive information while preserving memory size and energetic cost will reduce operational capacity more than scrambling nonpredictive information.

## Prediction 3

Increasing memory or model size will produce a finite optimum when information-processing costs are nonzero.

## Prediction 4

Learning can increase operational capacity without increasing gross capacity, but only after a measurable break-even point.

## Prediction 5

Self-repair can increase operational capacity by restoring conversion machinery even when no new gross resource is added.

## Prediction 6

Cooperation can increase total operational capacity only where coordination gains exceed communication and transfer costs.

Each prediction can fail.

---

# 24. Criteria for Success

Paper 1 succeeds at its initial level if:

1. every battery quantity is dimensionally valid;
2. the analytical toy models are reproduced numerically;
3. gross and operational capacity are separated cleanly;
4. information interventions isolate causal accessibility gains;
5. all controller and learning costs are included;
6. no experiment violates the relevant thermodynamic accounting;
7. results survive independent implementations;
8. the definitions remain useful in multiple model classes.

The stronger program succeeds if a common scaling relationship transfers across:

- Markov systems;
- spatial agents;
- reaction networks;
- evolutionary simulations;
- physical laboratory systems.

---

# 25. Criteria for Rejection or Major Revision

The proposed informational-battery framework should be rejected or substantially revised if:

1. operational capacity cannot be defined without arbitrary observer preferences;
2. gross and accessible capacity cannot be separated reproducibly;
3. apparent accessibility recharge always reduces to omitted external energy;
4. predictive information provides no advantage beyond ordinary physical state variables;
5. controller-class dependence makes comparisons scientifically meaningless;
6. no cross-model scaling relation exists;
7. the framework merely renames exergy, semantic information, or control theory without generating new predictions;
8. the battery terminology creates more confusion than explanatory value.

---

# 26. Conclusion

The informational battery is not a container filled with abstract information.

It is not evidence that intelligence creates energy.

It is not a mechanism for reversing the second law.

It is not yet a cosmological field.

The disciplined definition is:

\[
\boxed{
\textbf{An informational battery is a physically embodied}
\atop
\textbf{nonequilibrium capacity whose operational accessibility}
\atop
\textbf{depends on structured, causally effective information.}
}
\]

The framework separates:

\[
\boxed{
\begin{aligned}
B_{\mathrm{gross}}
&=\text{capacity physically available in principle},\\
B_{\mathrm{op}}
&=\text{capacity a defined controller can obtain net of costs},\\
B_{\mathrm{latent}}
&=\text{capacity present but operationally inaccessible}.
\end{aligned}
}
\]

Information enters through the conversion:

\[
B_{\mathrm{gross}}
\longrightarrow
B_{\mathrm{op}}.
\]

Learning, memory, prediction, repair, and cooperation may reduce latent capacity and increase operational access. They do so only through physical machinery that consumes resources and produces entropy.

The most important conceptual result is:

\[
\boxed{
\text{Intelligence need not create new energy to create new capability.}
\atop
\text{It may reveal lawful paths to capacity that was already present}
\atop
\text{but previously inaccessible.}
}
\]

Whether this distinction yields a universal law is unknown.

The next paper will test the stronger proposition that information becomes agency when its causal contribution to accessible work and viability exceeds its physical cost.

---

# References

1. Seifert, U. “Stochastic Thermodynamics, Fluctuation Theorems and Molecular Machines.” *Reports on Progress in Physics* 75, 126001 (2012). citeturn533893search4

2. Parrondo, J. M. R., Horowitz, J. M. and Sagawa, T. “Thermodynamics of Information.” *Nature Physics* 11, 131–139 (2015). citeturn533893search1turn533893search13

3. Parrondo, J. M. R. “Thermodynamics of Information.” Review manuscript (2023). citeturn719940search32

4. Sagawa, T. and Ueda, M. “Information Thermodynamics: Maxwell’s Demon in Nonequilibrium Dynamics.” (2011). citeturn184356academia34

5. Sagawa, T. “Second Law, Entropy Production, and Reversibility in Thermodynamics of Information.” (2017). citeturn184356academia36

6. Landauer, R. “Irreversibility and Heat Generation in the Computing Process.” *IBM Journal of Research and Development* 5, 183–191 (1961).

7. Bérut, A. et al. “Experimental Verification of Landauer’s Principle Linking Information and Thermodynamics.” *Nature* 483, 187–189 (2012). citeturn719940search6

8. Qian, H. “Relative Entropy: Free Energy Associated with Equilibrium Fluctuations and Nonequilibrium Deviations.” (2000). citeturn719940academia75

9. Ge, H. and Qian, H. “The Physical Origins of Entropy Production, Free Energy Dissipation and Their Mathematical Representations.” (2009). citeturn719940academia74

10. Still, S., Sivak, D. A., Bell, A. J. and Crooks, G. E. “Thermodynamics of Prediction.” *Physical Review Letters* 109, 120604 (2012). citeturn533893search3turn533893search23

11. Kolchinsky, A. and Wolpert, D. H. “Semantic Information, Autonomous Agency and Non-equilibrium Statistical Physics.” *Interface Focus* 8, 20180041 (2018). citeturn719940search4

12. Deffner, S. and Jarzynski, C. “Information Processing and the Second Law of Thermodynamics: An Inclusive, Hamiltonian Approach.” (2013). citeturn533893academia37
