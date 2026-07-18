<!-- extracted from ChatGPT msg [69] on 2026-07-18 -->

# The IF Causal-Work Principle  
## When Predictive Information Becomes Physical Agency

**Author:** Phuc Vinh Truong  
**Series:** IF Theory Working Papers  
**Paper:** 2  
**Date:** July 17, 2026  
**Status:** Theoretical proposal and computational protocol awaiting simulation, falsification, and independent review

---

## Abstract

Physical systems frequently contain information about their environments. Crystals preserve traces of their formation, thermostats respond to temperature, organisms retain memories, and intelligent agents construct models of possible futures. Correlation alone, however, does not establish agency. Information may be physically stored yet irrelevant to action, predictive yet unused, or useful but more costly to maintain than the physical benefit it enables.

This paper proposes the **IF Causal-Work Principle**, an intervention-based criterion for identifying when predictive information becomes part of a system’s physical agency. A bounded system is evaluated under an intact condition and under matched interventions that erase, scramble, temporally displace, or falsify selected internal information while preserving relevant physical and statistical properties. The causal-work contribution of that information is the change in net useful work produced by the system after all incremental costs of sensing, memory, computation, communication, control, and action are included.

For internal model \(M\), environment \(E\), and horizon \(\tau\), the central quantity is provisionally defined as:

\[
\mathcal W_C(M;\tau)
=
J_{\mathrm{intact}}(\tau)
-
J_{\mathrm{ablated}}(\tau),
\]

where:

\[
J(\tau)
=
\mathbb E
\left[
W_{\mathrm{useful}}(\tau)
-
C_{\mathrm{total}}(\tau)
\right].
\]

A model carries positive causal-work value when:

\[
\mathcal W_C>0.
\]

Positive causal-work value alone is not sufficient for full agency: simple feedback controllers may satisfy it. IF Theory therefore defines **predictive physical agency** through a conjunction of conditions: bounded persistence, endogenous information maintenance, action-dependent environmental influence, positive causal-work value of predictive internal states, adaptive performance across multiple environments, and the absence of an external controller supplying the relevant decisions.

A dimensionless candidate threshold is introduced:

\[
\Pi_A
=
\frac{
\Delta W_{\mathrm{enabled}}
}{
C_{\mathrm{model}}
},
\]

where \(\Delta W_{\mathrm{enabled}}\) is the additional useful work enabled by the intact model relative to an appropriately matched model-free or scrambled system. The strongest IF hypothesis is that predictive agency becomes selectively sustainable when:

\[
\Pi_A>1.
\]

The paper presents analytical toy models, intervention standards, simulation architectures, phase-transition tests, falsification conditions, and a deterministic Jupyter-notebook program. It also narrows the novelty claim in light of established semantic-information, predictive-information, causal-emergence, empowerment, and physical-intelligence research, including several closely related frameworks published in 2025–2026. The prospective IF contribution is not the broad claim that intelligence converts information into work. It is the discovery—if it exists—of a transferable intervention-based boundary predicting when internally maintained information pays for itself physically and becomes constitutive of autonomous agency.

---

## Keywords

Agency; predictive information; information thermodynamics; causal intervention; work extraction; autonomy; semantic information; viability; artificial life; physical intelligence; phase transition.

---

# 1. Introduction

A physical system may contain information without being an agent.

A rock records information about pressure and temperature. A crystal stores structural regularities. A camera stores images. A thermostat contains information about present temperature and uses that information to activate heating. A bacterium senses chemical gradients. An animal retains memories and anticipates future resource locations.

These systems differ not merely in how much information they contain, but in what that information does.

The central challenge is to distinguish:

\[
\text{information that exists}
\]

from:

\[
\text{information that predicts}
\]

from:

\[
\text{information that causally changes action}
\]

from:

\[
\text{information whose physical benefit exceeds its cost}.
\]

Information thermodynamics has established that information-processing systems must be evaluated as physical systems. Measurement, feedback, memory, work extraction, and erasure participate in thermodynamic ledgers rather than operating outside them. The thermodynamics of prediction further distinguishes predictive information from memory that records the past without helping anticipate the future; nonpredictive memory may contribute to dissipation without improving control. citeturn775554search1turn775554search11

Kolchinsky and Wolpert developed an intervention-based account of semantic information for autonomous nonequilibrium systems. Their framework asks which syntactic correlations matter to a system’s continued viability by intervening on those correlations and observing the resulting viability loss. citeturn775554search0turn775554search12

Causal-emergence research similarly investigates whether macrolevel descriptions can possess greater effective causal organization than microlevel descriptions under specified intervention measures. This provides a possible method for identifying agents or organisms without assuming that the programmer’s preferred partition is necessarily the correct causal scale. citeturn775554search3turn775554search8

Recent work has moved even closer to the present proposal. A 2025–2026 physical-intelligence literature includes measures based on goal-directed work per unit of irreversibly processed information, information or empowerment per joule, rare-valid future amplification, and interaction-level predictive structure. citeturn625612academia33turn625612academia34turn625612academia35turn625612academia36

Consequently, IF Theory cannot claim novelty for any of the following statements:

- intelligent systems are physical;
- information processing has thermodynamic costs;
- predictive information is more useful than indiscriminate memory;
- information may enable work extraction;
- agency requires interaction between action and outcome;
- viable systems can contain information meaningful to their persistence;
- intelligence can be normalized by energy or information-processing cost.

The narrower unresolved question is:

\[
\boxed{
\text{Does a transferable intervention-based physical threshold separate}
\atop
\text{passive correlation, feedback control, and predictive agency?}
}
\]

This paper defines the threshold as a research target. It does not assume that nature contains one universal boundary.

---

# 2. Scope

This paper addresses **functional physical agency**.

It does not claim to explain:

- phenomenal consciousness;
- free will in the metaphysical sense;
- moral responsibility;
- subjective experience;
- human-level intelligence;
- cosmic purpose;
- divine action.

The initial domain consists of:

- finite-state stochastic systems;
- deterministic artificial environments;
- resource-constrained controllers;
- artificial-life simulations;
- adaptive agents;
- evolutionary populations.

The proposed framework may later be tested in:

- chemical reaction networks;
- active matter;
- microbial behavior;
- cellular regulatory systems;
- robots;
- artificial intelligence systems.

No cross-substrate universality is assumed before empirical demonstration.

---

# 3. Prior Art and the Novelty Boundary

## 3.1 Thermodynamics of prediction

Still, Sivak, Bell, and Crooks studied physical systems driven by changing environments and separated stored information about past environmental states from information predictive of future states. Their framework relates thermodynamic inefficiency to nonpredictive information retained by a system. citeturn775554search1turn775554search14

IF Theory therefore cannot claim that predictive information is generally more thermodynamically valuable than irrelevant historical memory as an original insight.

The IF extension is to evaluate the information through controlled ablations and quantify the resulting change in net physical work and autonomous persistence.

---

## 3.2 Semantic information and viability

Kolchinsky and Wolpert proposed that information becomes semantic for a physical system when interventions that remove the information reduce a specified viability measure. Their framework explicitly addresses system–environment decomposition, intervention, timescale, and the possibility of identifying agency through semantic information. citeturn775554search0turn775554search6

IF Theory therefore cannot claim novelty for defining meaningful information through intervention.

Its proposed distinction is to keep two effects separate:

\[
\text{causal-work value}
\]

and:

\[
\text{causal-viability value}.
\]

A system might obtain more work while reducing its long-term persistence. Conversely, information might improve survival without increasing exported work. The two outcomes should be reported independently rather than collapsed into a single favorable score.

---

## 3.3 Causal emergence

Hoel, Albantakis, and Tononi showed that appropriately chosen macroscopic causal models can exhibit greater effective information than corresponding microscopic models when degeneracy and noise are reduced at the macrolevel. citeturn775554search3turn775554search8

IF Theory may use causal emergence to identify candidate agent boundaries or macrostates, but it cannot equate higher effective information with agency automatically.

A macrodescription may be causally informative without:

- self-maintenance;
- prediction;
- endogenous action;
- resource acquisition;
- policy adaptation.

---

## 3.4 Empowerment and control information

Empowerment measures the channel capacity between an agent’s actions and future sensory states. Recent physical-intelligence work has explicitly proposed empowerment per unit energetic cost as one axis of physical intelligence. citeturn625612academia33

IF Theory therefore cannot claim novelty for measuring action influence or control information per joule.

The IF intervention asks a different question:

> How much net useful physical output or persistence disappears when a particular internal predictive representation is selectively destroyed?

Empowerment measures potential influence. Causal-work ablation measures the realized physical contribution of a selected internal model.

---

## 3.5 Goal-directed work per information cost

A recent physical theory of intelligence defines intelligence using goal-directed work produced per unit of irreversibly processed information and develops a framework connecting conservation, encoding, computation, and work extraction. citeturn625612academia35

A separate 2026 proposal defines thermodynamic intelligence as the lawful amplification of rare but valid futures and argues that recursive self-simulation is necessary, under stated assumptions, for high performance on that measure. citeturn625612academia36

These approaches strongly overlap with IF Theory’s general motivation. The IF program must therefore demonstrate a specific advantage rather than presenting another renamed work-efficiency ratio.

The prospective distinction is:

1. a matched causal intervention on an internal model;
2. explicit separation of gross battery capacity from model-enabled accessibility;
3. comparison of intact, erased, scrambled, displaced, and false internal models;
4. a search for a transferable threshold across independently designed substrates;
5. automatic detection of candidate agents rather than assuming the agent boundary.

This distinction remains provisional and must survive a deeper formal literature review.

---

# 4. Conceptual Requirements for Physical Agency

A useful definition should not classify every causal system as an agent.

A falling stone causally changes its environment.

A heat engine extracts work.

A thermostat uses information.

A bacterium navigates.

A planning organism evaluates counterfactual futures.

IF Theory therefore treats agency as graded and requires multiple conditions.

## 4.1 Bounded organization

There must be a candidate system \(A\) distinguishable from an environment \(E\) over a stated time interval.

The boundary may be:

- spatial;
- causal;
- thermodynamic;
- informational;
- functional.

It must be selected through a documented rule rather than chosen because it makes the result favorable.

---

## 4.2 Persistence

The candidate system must maintain some organizational identity across time or component turnover.

Let:

\[
V_t
\]

be a set of variables defining its viable organization.

The system must remain within or repeatedly return to a viability region:

\[
\mathcal V.
\]

---

## 4.3 Action

The system must possess internal transitions that alter its coupling to the environment.

Let:

\[
A_t
\]

denote action variables.

An intervention on \(A_t\) must change the distribution of future environmental or system states:

\[
P(E_{t+\tau},V_{t+\tau}\mid do(A_t=a))
\neq
P(E_{t+\tau},V_{t+\tau}\mid do(A_t=a')).
\]

---

## 4.4 Endogenous information

The relevant internal state must be:

- acquired;
- updated;
- preserved;
- or selected

through processes inside the declared system boundary.

A lookup table externally updated with the correct answer at every step does not establish autonomous prediction by the local system.

---

## 4.5 Predictive content

Let \(M_t\) denote an internal model or memory.

It must contain information about a future-relevant variable beyond the immediately available state:

\[
I(M_t;E_{t+\tau}\mid E_t)>0.
\]

Predictive information alone remains insufficient.

---

## 4.6 Causal use

Changing \(M_t\) while controlling relevant alternatives must change actions and outcomes.

The causal path must include:

\[
M_t
\rightarrow
A_t
\rightarrow
E_{t+\tau}\text{ or }V_{t+\tau}.
\]

A representation correlated with behavior but ignored by the policy does not contribute agency.

---

## 4.7 Net physical benefit

The benefit produced by the model must exceed the physical cost of maintaining and using it under at least some environments.

This is the central IF condition.

---

## 4.8 Generalization or adaptation

A fixed reflex may be useful in one environment. Predictive agency requires performance across a defined environment family or adaptation when environmental statistics change.

The system must not receive a newly hand-coded response for every tested condition.

---

# 5. Formal Setup

Let the total system be:

\[
\Omega
=
(A,E,R),
\]

where:

- \(A\) is the candidate agent;
- \(E\) is the environment;
- \(R\) denotes physical reservoirs, including fuel, heat, matter, or work stores.

The candidate agent contains:

\[
A_t
=
(X_t,M_t,U_t),
\]

where:

- \(X_t\) represents internal physical state;
- \(M_t\) is a selected memory or predictive-model state;
- \(U_t\) is the action or control state.

Let:

\[
Y_t
\]

denote future-relevant environmental variables.

Let:

\[
W_{\mathrm{out}}[0,\tau]
\]

be useful work transferred to a declared work reservoir during the evaluation interval.

Let:

\[
C_{\mathrm{sense}},
C_{\mathrm{memory}},
C_{\mathrm{compute}},
C_{\mathrm{communicate}},
C_{\mathrm{act}},
C_{\mathrm{repair}},
C_{\mathrm{reset}}
\]

be physical costs within the same boundary.

Define total cost:

\[
C_{\mathrm{total}}
=
C_{\mathrm{sense}}
+
C_{\mathrm{memory}}
+
C_{\mathrm{compute}}
+
C_{\mathrm{communicate}}
+
C_{\mathrm{act}}
+
C_{\mathrm{repair}}
+
C_{\mathrm{reset}}.
\]

Define net physical return:

\[
\boxed{
J_W(\tau)
=
\mathbb E
\left[
W_{\mathrm{out}}[0,\tau]
-
C_{\mathrm{total}}[0,\tau]
\right].
}
\]

The system boundary must include externally prepared low-entropy memory, remote computation, sensor power, and actuator power. Recent work on physical-intelligence metrics emphasizes that information-per-joule comparisons become misleading when boundary closure, reset, horizon, and imported low-entropy resources are omitted. citeturn625612academia33

---

# 6. The Intervention Family

No single ablation is sufficient. Destroying memory can also change energy, dynamics, architecture, and action capacity.

IF Theory therefore requires a family of matched interventions.

## 6.1 Erasure intervention

Replace \(M_t\) with a default state:

\[
M_t\rightarrow m_0.
\]

This tests dependence on stored state but may alter state entropy and physical cost.

---

## 6.2 Permutation scrambling

Apply a bijection:

\[
M_t\rightarrow \sigma(M_t)
\]

that preserves marginal frequencies but destroys the intended correspondence between memory and environment.

This is useful when labels are arbitrary and action mappings can be held fixed.

---

## 6.3 Cross-episode scrambling

Assign an internal model from another independent episode:

\[
M_t^{(i)}
\rightarrow
M_t^{(j)},
\qquad
i\neq j.
\]

This approximately preserves model complexity and physical representation while destroying episode-specific predictive content.

---

## 6.4 Temporal displacement

Replace the current model with a delayed or advanced model:

\[
M_t\rightarrow M_{t-\Delta}
\]

or, in simulation controls:

\[
M_t\rightarrow M_{t+\Delta}
\]

where future leakage is used only as a diagnostic upper bound.

This tests whether temporal alignment matters.

---

## 6.5 Predictive-variable scrambling

Preserve information about past states while destroying information specifically predictive of future states.

The objective is to produce:

\[
I(\widetilde M_t;E_{t-\tau_p})
\approx
I(M_t;E_{t-\tau_p})
\]

while reducing:

\[
I(\widetilde M_t;E_{t+\tau_f}\mid E_t).
\]

---

## 6.6 Equal-capacity irrelevant model

Replace the useful model with an equally large and similarly costly representation of an irrelevant variable.

This distinguishes model size from predictive content.

---

## 6.7 False-model intervention

Provide systematically inaccurate predictions with matched confidence and computational cost.

This determines whether the policy uses model content rather than merely model presence.

---

## 6.8 Policy-disconnection intervention

Preserve \(M_t\), but remove the causal edge:

\[
M_t\rightarrow U_t.
\]

This tests whether the information is used by action selection.

---

# 7. Definition of Causal-Work Value

Let:

\[
\mathcal I_0
\]

denote the intact condition and:

\[
\mathcal I_k
\]

an intervention.

For each condition, calculate:

\[
J_W^{(k)}(\tau)
=
\mathbb E
\left[
W_{\mathrm{out}}^{(k)}
-
C_{\mathrm{total}}^{(k)}
\right].
\]

Define intervention-specific causal-work value:

\[
\boxed{
\mathcal W_C^{(k)}(M;\tau)
=
J_W^{(0)}(\tau)
-
J_W^{(k)}(\tau).
}
\]

Because no intervention is perfect, report a vector:

\[
\mathbf W_C
=
\left(
\mathcal W_C^{\mathrm{erase}},
\mathcal W_C^{\mathrm{permute}},
\mathcal W_C^{\mathrm{cross}},
\mathcal W_C^{\mathrm{delay}},
\mathcal W_C^{\mathrm{irrelevant}},
\mathcal W_C^{\mathrm{false}},
\mathcal W_C^{\mathrm{disconnect}}
\right).
\]

A robust effect requires positive values across multiple appropriate interventions.

A single large erasure effect does not establish predictive causal work because erasure may damage the controller nonspecifically.

---

# 8. Gross Benefit, Model Cost, and the Agency Ratio

To study whether maintaining a model pays for itself, separate gross enabled work from incremental model cost.

Let:

\[
W_{\mathrm{use}}^{M}
\]

be useful work with the model.

Let:

\[
W_{\mathrm{use}}^{0}
\]

be useful work under a matched controller lacking that model.

Define gross model-enabled work:

\[
\Delta W_{\mathrm{enabled}}
=
W_{\mathrm{use}}^{M}
-
W_{\mathrm{use}}^{0}.
\]

Define incremental model cost:

\[
C_M
=
C_{\mathrm{total}}^{M}
-
C_{\mathrm{total}}^{0},
\]

excluding work differences already counted in \(\Delta W_{\mathrm{enabled}}\).

Define net model value:

\[
\boxed{
\mathcal W_{\mathrm{net}}
=
\Delta W_{\mathrm{enabled}}
-
C_M.
}
\]

Where:

\[
C_M>0,
\]

define the candidate agency ratio:

\[
\boxed{
\Pi_A
=
\frac{
\Delta W_{\mathrm{enabled}}
}{
C_M
}.
}
\]

Interpretation:

\[
\Pi_A<1
\quad\Rightarrow\quad
\text{the model costs more than the work it enables},
\]

\[
\Pi_A=1
\quad\Rightarrow\quad
\text{physical break-even},
\]

\[
\Pi_A>1
\quad\Rightarrow\quad
\text{the model produces net-positive work value}.
\]

This ratio is not automatically universal. Its meaning depends on the boundary, horizon, model-free comparator, and definition of useful work.

---

# 9. Viability Must Remain Separate

An organism may spend energy to survive without exporting useful mechanical work.

A predator may extract considerable work while increasing its risk of death.

Work and viability therefore require separate measures.

Let:

\[
P_{\mathcal V}(\tau)
=
P(V_t\in\mathcal V\text{ for }0\leq t\leq\tau)
\]

or another preregistered viability function.

Define causal viability:

\[
\boxed{
\mathcal V_C^{(k)}
=
V^{(0)}(\tau)
-
V^{(k)}(\tau).
}
\]

The complete IF result is a vector:

\[
\boxed{
\mathbf A_{\mathrm{IF}}
=
\left[
\mathbf W_C,
\mathbf V_C,
I_{\mathrm{pred}},
C_M,
\Pi_A,
\text{adaptation score}
\right].
}
\]

Bits, joules, and survival probabilities are not combined through arbitrary addition.

---

# 10. Definition of Predictive Physical Agency

A candidate system qualifies as a **predictive physical agent relative to environment family \(\mathcal E\), horizon \(\tau\), and system boundary \(\partial A\)** when all of the following hold.

## Criterion A — Persistence

The system maintains an operational identity or viability distribution over the evaluation interval.

## Criterion B — Endogenous action

Its internal state causally affects environmental or resource transitions.

## Criterion C — Endogenous model

The predictive state is acquired, maintained, or updated within the declared physical boundary.

## Criterion D — Predictive content

The internal model predicts future-relevant states beyond current observation.

## Criterion E — Causal use

Matched model interventions change actions and outcomes through the model-to-policy path.

## Criterion F — Positive net contribution

At least one preregistered outcome satisfies:

\[
\mathcal W_{\mathrm{net}}>0
\]

or a positive causal-viability criterion, with all costs reported.

## Criterion G — Environmental breadth

The effect survives across a declared set of environments or after environmental change.

## Criterion H — No external decision oracle

The correct actions are not supplied by an uncounted external system.

This definition is relative, operational, and graded.

It does not assert that all agents satisfy one sharp metaphysical boundary.

---

# 11. Agency Ladder

## A0 — Passive persistence

The system persists through fixed physical stability.

Examples may include crystals or static attractors.

## A1 — Reactive regulation

Current sensory input changes present action.

A thermostat may qualify.

## A2 — Memory-dependent control

Past internal state changes present action.

## A3 — Predictive control

Internal state contains future-relevant information that causally improves outcome.

## A4 — Counterfactual control

The system evaluates multiple possible action-dependent futures.

## A5 — Self-modeling control

The model represents aspects of the system’s own future state or limitations.

## A6 — Policy revision

The system changes how it selects policies when its model repeatedly fails.

## A7 — Social modeling

The system predicts the states and actions of other agents.

## A8 — Institutional agency

Multiple systems create persistent shared constraints, records, or coordination structures that expand collective control.

Paper 2 addresses primarily the transition from A1–A2 to A3.

Reflection and higher-order agency are deferred to later papers.

---

# 12. Analytical Toy Model I: Predictive Resource Choice

Consider two locations:

\[
L,\ R.
\]

One contains a resource worth:

\[
W_F.
\]

The system may choose one location.

A random controller succeeds with probability:

\[
\frac12.
\]

A predictive model succeeds with probability:

\[
q>\frac12.
\]

Let model cost be:

\[
C_M.
\]

Let all other action costs be equal.

The additional expected work enabled by the model is:

\[
\Delta W_{\mathrm{enabled}}
=
\left(
q-\frac12
\right)W_F.
\]

The net model value is:

\[
\boxed{
\mathcal W_{\mathrm{net}}
=
\left(
q-\frac12
\right)W_F-C_M.
}
\]

The candidate threshold is:

\[
\boxed{
\Pi_A
=
\frac{
(q-\frac12)W_F
}{
C_M
}.
}
\]

The model is physically profitable when:

\[
\Pi_A>1.
\]

This threshold depends jointly on:

- prediction accuracy;
- resource value;
- model cost.

High prediction accuracy is not sufficient if the environment offers little benefit or the model is too expensive.

---

# 13. Analytical Toy Model II: Environmental Persistence

Let resource location follow a two-state Markov process:

\[
P(Y_{t+1}=Y_t)=r.
\]

When:

\[
r=\frac12,
\]

the previous location offers no predictive value.

When:

\[
r> \frac12,
\]

the environment has persistence.

An agent storing the previous location predicts:

\[
\hat Y_{t+1}=Y_t.
\]

Its prediction accuracy is:

\[
q=r.
\]

The model is net beneficial when:

\[
\boxed{
\left(
r-\frac12
\right)W_F>C_M.
}
\]

The minimum environmental predictability required is:

\[
\boxed{
r_c
=
\frac12+\frac{C_M}{W_F}.
}
\]

This produces an explicit boundary.

It predicts:

- expensive memory requires a more predictable environment;
- valuable resources support predictive systems at lower predictability;
- in a fully unpredictable environment, model maintenance is wasteful;
- increasing memory efficiency lowers the agency threshold.

This is a model-specific break-even boundary, not yet a universal law.

---

# 14. Analytical Toy Model III: Finite Memory Depth

Let the environment contain temporal dependencies up to order \(K\).

Let model depth be \(L\).

Assume predictive benefit rises and saturates:

\[
\Delta W(L)
=
W_{\max}
\left(
1-e^{-L/\ell}
\right).
\]

Let model cost rise approximately linearly:

\[
C_M(L)=cL.
\]

Then:

\[
\mathcal W_{\mathrm{net}}(L)
=
W_{\max}
\left(
1-e^{-L/\ell}
\right)-cL.
\]

The optimum satisfies:

\[
\frac{d\mathcal W_{\mathrm{net}}}{dL}
=
\frac{W_{\max}}{\ell}e^{-L/\ell}-c=0.
\]

Therefore:

\[
\boxed{
L^*
=
\ell
\ln
\left(
\frac{W_{\max}}{c\ell}
\right)
}
\]

when:

\[
W_{\max}>c\ell.
\]

Otherwise:

\[
L^*=0.
\]

This predicts a discontinuous onset of nonzero model depth in the simplified optimization problem, although finite populations and stochastic learning may smooth the transition.

---

# 15. The IF Causal-Work Principle

## 15.1 Weak form

> Internal information contributes to physical agency when matched interventions that selectively disrupt its predictive or policy-relevant content reduce the system’s net physical return or viability.

## 15.2 Cost-aware form

> Predictive information becomes physically self-sustaining when the additional work or persistence it enables exceeds the complete incremental cost of acquiring, storing, protecting, updating, and using it.

## 15.3 Strong threshold conjecture

> Across a meaningful class of resource-constrained adaptive systems, the onset of persistent predictive agency is organized by a dimensionless causal-work ratio near physical break-even:

\[
\Pi_A\approx1.
\]

## 15.4 Universality conjecture

> After appropriate nondimensionalization, different substrates exhibit common scaling behavior near the causal-work threshold.

The weak and cost-aware forms are operational definitions.

The strong threshold and universality forms are empirical conjectures.

---

# 16. Why Positive Causal Work Is Not Sufficient by Itself

A thermostat may use one bit of state to save heating energy.

It may therefore have:

\[
\mathcal W_C>0.
\]

Calling the thermostat a minimal agent may be acceptable under a broad graded definition, but it does not make it a predictive or reflective agent.

The stronger classification requires:

- internal predictive state;
- temporal horizon;
- environmental generalization;
- endogenous model maintenance;
- adaptive policy selection.

IF Theory therefore rejects the equation:

\[
\mathcal W_C>0
\quad\Rightarrow\quad
\text{full agency}.
\]

Instead:

\[
\mathcal W_C>0
\]

is evidence that selected information has physical causal value.

Agency classification requires the remaining criteria.

---

# 17. Simulation Architecture

## 17.1 Environment

The reference environment will contain:

- a spatial or graph-based domain;
- localized energy resources;
- resource depletion and regeneration;
- hazards;
- environmental states with tunable predictability;
- explicit action and transport costs.

Environmental predictability will be controlled through parameters such as:

- Markov persistence;
- periodicity;
- spatial correlation length;
- volatility;
- hidden-state switching;
- observation noise.

---

## 17.2 Candidate systems

The initial experiments will compare:

1. passive dissipative structures;
2. fixed reactive controllers;
3. one-step-memory controllers;
4. predictive finite-state controllers;
5. learned predictive controllers;
6. model-based planners.

All systems must obey the same physical resource ledger.

The model-based systems must pay explicit costs that increase with:

- sensor precision;
- memory capacity;
- model complexity;
- communication;
- planning depth.

---

## 17.3 Endogenous persistence

Systems receive no abstract reward points.

Their effective fitness arises through:

- maintaining internal energy;
- avoiding destructive states;
- continuing to act;
- repairing;
- reproducing where enabled.

Evolutionary runs may select systems by persistence and reproduction, but the simulator must not directly reward “intelligence,” “prediction,” or “agency.”

---

## 17.4 Agent-boundary detection

Initially, system boundaries may be declared for controlled tests.

Later experiments should compare declared boundaries with automatically detected candidates using:

- causal partitions;
- integrated information flow;
- transfer entropy;
- persistent connected components;
- resource-flow closure;
- intervention sensitivity;
- causal emergence.

A result that depends entirely on a hand-selected favorable boundary is weak.

---

# 18. Experimental Program

## Experiment 1 — Binary predictive choice

Validate the analytical threshold:

\[
\left(q-\frac12\right)W_F>C_M.
\]

Vary:

- \(q\);
- \(W_F\);
- \(C_M\);
- action cost;
- sensor noise.

---

## Experiment 2 — Predictive versus historical memory

Construct memories with matched size and past mutual information but different future-predictive information.

Test whether:

\[
\mathcal W_C
\]

tracks predictive information better than total historical storage.

---

## Experiment 3 — Scrambling hierarchy

Compare all intervention types.

A robust predictive model should show:

- large loss under predictive-variable scrambling;
- large loss under policy disconnection;
- smaller loss under irrelevant-memory scrambling;
- graded loss under temporal displacement.

---

## Experiment 4 — Environmental predictability sweep

Vary the persistence or temporal structure of the environment.

Measure:

- evolved model depth;
- causal-work value;
- survival;
- energy efficiency;
- prediction accuracy.

Test whether model-bearing systems disappear below a critical predictability.

---

## Experiment 5 — Model-cost sweep

Increase the physical cost of sensing, memory, and computation.

Test whether:

- model complexity decreases;
- predictive agency disappears;
- reactive control remains;
- a break-even boundary can be recovered.

---

## Experiment 6 — Environmental regime shift

Allow the environment’s transition law to change.

Compare:

- fixed predictor;
- adaptive predictor;
- reactive controller;
- no-memory controller.

Predictive agency should include recovery after model failure, not merely high performance in a stationary environment.

---

## Experiment 7 — Evolution without agency reward

Initialize random resource-processing structures.

Allow mutation and selection only through persistence and reproduction.

Test whether predictive internal states emerge near the calculated causal-work boundary.

---

## Experiment 8 — Cross-rule-family replication

Repeat the experiments using:

- finite-state Markov agents;
- cellular automata;
- reaction networks;
- recurrent neural controllers;
- graph-based organisms.

The strongest claim requires a common nondimensional relationship.

---

# 19. Phase-Transition Tests

The phrase “agency threshold” should not be used casually.

A smooth cost-benefit crossover is not necessarily a statistical-mechanical phase transition.

To support a genuine transition claim, the program will test for:

- order-parameter behavior;
- finite-size scaling;
- divergent or peaked susceptibility;
- critical slowing;
- hysteresis;
- bimodal state distributions;
- scaling collapse;
- robustness across system size.

Candidate order parameters include:

\[
\langle \mathcal W_C\rangle,
\]

\[
P(\Pi_A>1),
\]

\[
I(M_t;E_{t+\tau}\mid E_t),
\]

and the fraction of surviving systems whose model ablation reduces performance.

A null result—only a smooth crossover—remains scientifically informative.

---

# 20. Core Hypotheses

## CW-H1 — Intervention hypothesis

Scrambling future-relevant internal information while preserving relevant physical and statistical controls reduces net work or viability.

### Falsifier

Matched scrambling produces no selective reduction.

---

## CW-H2 — Predictive-specificity hypothesis

Causal-work value is more strongly associated with predictive information than with total memory size or information about the past.

### Falsifier

Irrelevant or nonpredictive memories provide equal net benefit after costs and architecture are matched.

---

## CW-H3 — Cost threshold hypothesis

Predictive models persist only when:

\[
\Delta W_{\mathrm{enabled}}>C_M.
\]

### Falsifier

Model-bearing systems remain selectively favored even when their complete physical cost persistently exceeds their physical benefit.

---

## CW-H4 — Finite-complexity hypothesis

When model cost rises with complexity and environmental predictability is finite, an optimal noninfinite model complexity exists.

### Falsifier

Additional model complexity is always net beneficial despite nonzero costs.

---

## CW-H5 — Adaptive-agency hypothesis

Systems capable of revising their models outperform fixed predictors after environmental regime shifts when the long-run value of adaptation exceeds its added cost.

### Falsifier

Adaptation provides no net benefit across the preregistered regime family.

---

## CW-H6 — Emergent-agency hypothesis

Predictive internal models arise through physical selection without an explicit prediction or intelligence reward in environments where:

\[
\Pi_A>1.
\]

### Falsifier

Predictive models emerge only when directly rewarded or manually installed.

---

## CW-H7 — Scaling hypothesis

Different substrates exhibit an approximately shared relationship between:

\[
\Pi_A
\]

and the persistence of predictive control.

### Falsifier

Every substrate requires unrelated thresholds or arbitrary rescaling.

---

# 21. Deterministic Jupyter-Notebook Program

## Notebook 02A — Causal-Work Analytical Baselines

Implement exact calculations for:

- binary resource choice;
- Markov-persistent resources;
- finite memory depth;
- model-cost break-even.

Validate simulation against closed-form equations.

---

## Notebook 02B — Intervention Library

Implement:

- erase;
- permute;
- cross-episode scramble;
- temporal displacement;
- irrelevant-model replacement;
- false-model substitution;
- policy disconnection.

Test that each intervention preserves its declared controls.

---

## Notebook 02C — Predictive Information Estimators

Estimate:

\[
I(M_t;E_{t+\tau}\mid E_t)
\]

using:

- exact distributions;
- plug-in estimates;
- bias-corrected estimates;
- k-nearest-neighbor estimators where appropriate.

Validate estimators on synthetic systems with known information.

---

## Notebook 02D — Physical Cost Ledger

Track:

- sensing energy;
- memory cost;
- computation cost;
- actuation;
- communication;
- repair;
- reset;
- external imports.

Fail the run when accounting does not close within tolerance.

---

## Notebook 02E — Causal-Work Ablation

Measure:

\[
\mathbf W_C
\]

and:

\[
\mathbf V_C
\]

for all intervention classes.

Generate causal diagrams and path-specific controls.

---

## Notebook 02F — Predictability–Cost Phase Map

Sweep:

\[
(r,C_M,W_F,\text{noise},\tau).
\]

Map regions where:

- no control persists;
- reactive control persists;
- predictive control persists;
- adaptive prediction persists.

---

## Notebook 02G — Finite-Size Scaling

Test whether the apparent agency boundary sharpens with:

- population size;
- system size;
- evaluation horizon.

Distinguish a true transition from a finite-system crossover.

---

## Notebook 02H — Evolution Without Intelligence Reward

Evolve controllers under physical energy balance and reproduction only.

Test whether model-bearing structures emerge near predicted regions.

---

## Notebook 02I — Cross-Substrate Replication

Repeat the causal-work analysis across at least three independently coded model classes.

---

## Notebook 02J — Adversarial Reproduction

A separate coding agent receives only:

- the paper;
- public configurations;
- raw outputs.

It must independently reproduce the principal result and attempt to destroy it.

---

# 22. Reproducibility Record

Every experiment will emit:

```yaml
experiment_id: if-causal-work-02
paper_version: null
git_commit: null
environment_hash: null
model_family: null
system_boundary: null
environment_family: null
time_horizon: null
random_seed: 65537

gross_resource_input_joules: null
useful_work_output_joules: null
sensor_cost_joules: null
memory_cost_joules: null
compute_cost_joules: null
action_cost_joules: null
repair_cost_joules: null
reset_cost_joules: null

predictive_information_bits: null
past_information_bits: null
causal_work_vector_joules: {}
causal_viability_vector: {}
agency_ratio: null

energy_residual_joules: null
entropy_residual: null
intervention_validation: {}
invariant_failures: []
result_hash: null
```

Canonical results will use deterministic CPU calculations where possible.

Stochastic results will report:

- seeds;
- sample counts;
- confidence intervals;
- convergence diagnostics;
- sensitivity to estimator choice.

---

# 23. Statistical Standards

## 23.1 Holdout environments

Model design and parameter selection will use training environments.

Primary claims will use held-out environmental transition laws.

---

## 23.2 Multiple comparisons

Large parameter sweeps create many opportunities for false discoveries.

The primary order parameter, threshold, and transition criteria must be frozen before final analysis.

---

## 23.3 Model comparison

Compare:

- no-memory controls;
- reactive controls;
- predictive models;
- alternative information measures;
- standard reinforcement-learning metrics;
- empowerment;
- semantic-information measures.

IF Theory must demonstrate incremental explanatory value.

---

## 23.4 Robustness

Vary:

- system boundary;
- time horizon;
- coarse-graining;
- cost model;
- information estimator;
- intervention type;
- viability definition.

A result surviving only one favorable specification is not strong.

---

# 24. Failure Modes

## 24.1 Hidden oracle

An external process supplies correct actions or labels without its physical cost entering the boundary.

## 24.2 Unmatched ablation

Removing memory also reduces architecture, action capacity, or available energy.

## 24.3 Circular utility

The output measure directly rewards possession of the model.

## 24.4 Cost omission

Training, sensing, external computation, or resetting is ignored.

## 24.5 Correlation mistaken for use

The model predicts the future but the policy does not depend on it.

## 24.6 Action without autonomy

An external controller makes the decisions.

## 24.7 Survival-score arbitrariness

The chosen viability function guarantees the desired conclusion.

## 24.8 Boundary manipulation

The system boundary is changed until the energy ratio becomes favorable.

## 24.9 Substrate overfitting

A threshold exists only in the original toy environment.

## 24.10 Terminological inflation

A simple control improvement is presented as consciousness or free will.

---

# 25. What Would Count as a Major Result?

## Level 1 — Valid computational measure

The interventions reliably distinguish useful predictive information from irrelevant memory.

This would justify the method but not a new law of nature.

## Level 2 — Robust threshold within one model family

A repeatable break-even boundary predicts when predictive controllers persist.

This would be a publishable artificial-life or information-thermodynamics result.

## Level 3 — Cross-model scaling

The same nondimensional relationship predicts agency onset across independently designed simulation classes.

This would be substantially more important.

## Level 4 — Laboratory transfer

The same relationship predicts transitions in chemical, active-matter, microbial, or robotic systems.

This could establish a new physics-of-agency research program.

## Level 5 — Universal bound

A theorem constrains predictive causal work for broad physical systems, and experiments approach or confirm the bound.

This would be potentially field-changing.

---

# 26. Novelty Assessment

The current novelty score must be lower than initially assumed because several recent papers propose closely neighboring physical measures of intelligence and agency. citeturn625612academia33turn625612academia34turn625612academia35turn625612academia36

The IF proposal is potentially distinctive only if it demonstrates all of the following together:

1. **Matched model intervention:** the selected internal information is experimentally disrupted rather than inferred from correlation alone.
2. **Net physical accounting:** the benefit is measured after complete incremental physical costs.
3. **Accessible-capacity interpretation:** information changes access to an existing nonequilibrium battery rather than being treated as energy.
4. **Agent emergence:** predictive systems arise without direct intelligence rewards.
5. **Cross-substrate scaling:** one relationship transfers across substantially different implementations.
6. **Clear negative cases:** systems with large information stores but no causal work are correctly rejected.
7. **Prospective prediction:** the threshold predicts a result not used to construct it.

Without those results, the IF Causal-Work Principle is best viewed as a synthesis and experimental protocol, not a novel fundamental law.

---

# 27. Relationship to Consciousness

Positive causal-work value does not imply consciousness.

A predictive controller may qualify as an agent while lacking:

- global access;
- self-modeling;
- counterfactual depth;
- report;
- subjective experience.

Later IF work may ask whether counterfactual self-models introduce another transition, but Paper 2 makes no phenomenal-consciousness claim.

The correct implication is:

\[
\text{predictive causal work}
\Rightarrow
\text{functional agency evidence},
\]

not:

\[
\text{predictive causal work}
\Rightarrow
\text{subjective awareness}.
\]

---

# 28. Relationship to Free Will

The framework evaluates whether internal models causally alter action and future outcomes.

That can establish a functional form of endogenous control.

It does not decide whether:

- determinism is metaphysically compatible with freedom;
- actions could have occurred differently under identical total physical conditions;
- moral responsibility follows from internal causation.

Those remain philosophical questions.

---

# 29. Relationship to IF Cosmology

The causal-work principle is not a dark-matter or dark-energy equation.

Its relevance to cosmology is indirect.

If IF Theory later proposes that cosmic organization alters access to a universe-wide nonequilibrium state, Paper 2 supplies a discipline:

> No informational contribution may be invoked unless a physical intervention, mechanism, cost, and measurable consequence are specified.

Cosmological structure formation cannot be called intelligent or agentic merely because it generates complexity.

---

# 30. Criteria for Rejection or Major Revision

The IF Causal-Work Principle should be rejected or substantially revised if:

1. causal-work effects cannot be separated from ordinary controller architecture;
2. matched information interventions cannot be constructed;
3. work benefits disappear under complete physical accounting;
4. predictive information performs no better than irrelevant stored information;
5. the threshold is entirely determined by an arbitrary reward definition;
6. evolved predictive systems require explicit intelligence rewards;
7. no relationship transfers across rule families;
8. existing semantic-information, empowerment, or physical-intelligence measures explain every result equally well with less machinery;
9. the concept adds terminology without making new predictions;
10. negative findings are repeatedly avoided by changing the system boundary or agency definition.

---

# 31. Conclusion

Information becomes scientifically relevant to agency not because it is mysterious, meaningful to a human observer, or abundant.

It becomes relevant when it is physically embodied, future-directed, used by action, and worth its cost.

The proposed IF criterion is:

\[
\boxed{
\mathcal W_C(M;\tau)
=
J_{\mathrm{intact}}(\tau)
-
J_{\mathrm{ablated}}(\tau).
}
\]

The corresponding break-even ratio is:

\[
\boxed{
\Pi_A
=
\frac{
\Delta W_{\mathrm{enabled}}
}{
C_M
}.
}
\]

The physical interpretation is:

\[
\boxed{
\Pi_A>1
\quad\Rightarrow\quad
\text{the predictive model enables more useful work than it costs.}
}
\]

That inequality does not, by itself, prove consciousness, reflection, autonomy in every sense, or a universal phase transition.

It supplies a testable boundary between information that merely exists and information that physically pays for its own continued use.

The strongest IF research question is therefore:

\[
\boxed{
\text{Across what classes of physical systems does net-positive}
\atop
\text{predictive information become a stable, self-maintaining cause}
\atop
\text{of action rather than a passive trace of the environment?}
}
\]

If no transferable relationship exists, the proposed universal principle fails.

If one relationship predicts the emergence of predictive control across artificial, chemical, biological, and engineered systems, IF Theory will have identified a plausible physical law of agency.

---

# References

1. Still, S., Sivak, D. A., Bell, A. J. and Crooks, G. E. “Thermodynamics of Prediction.” *Physical Review Letters* 109, 120604 (2012). citeturn775554search1turn775554search11

2. Kolchinsky, A. and Wolpert, D. H. “Semantic Information, Autonomous Agency and Non-equilibrium Statistical Physics.” *Interface Focus* 8, 20180041 (2018). citeturn775554search0turn775554search12

3. Hoel, E. P., Albantakis, L. and Tononi, G. “Quantifying Causal Emergence Shows That Macro Can Beat Micro.” *Proceedings of the National Academy of Sciences* 110, 19790–19795 (2013). citeturn775554search3turn775554search8

4. Horowitz, J. M. and Esposito, M. “Thermodynamics with Continuous Information Flow.” *Physical Review X* 4, 031015 (2014). citeturn775554search27

5. Perunov, N., Marsland, R. A. and England, J. L. “Statistical Physics of Adaptation.” *Physical Review X* 6, 021036 (2016). citeturn775554search39

6. Parrondo, J. M. R. “Thermodynamics of Information.” Review manuscript (2023). citeturn775554academia67

7. Takahashi, K. and Hayashi, Y. “Thermodynamic Limits of Physical Intelligence.” (2026). citeturn625612academia33

8. Hafez, W. et al. “A Mathematical Theory of Agency and Intelligence.” (2026). citeturn625612academia34

9. Fagan, P. D. “Toward a Physical Theory of Intelligence.” (2025–2026). citeturn625612academia35

10. Chattopadhyay, I. “Thermodynamic Measure of Intelligence.” (2026). citeturn625612academia36

11. Halpern, N. Y. “Toward Physical Realizations of Thermodynamic Resource Theories.” (2014–2016). citeturn775554search30

12. Marletto, C. “The Information-Theoretic Foundation of Thermodynamic Work Extraction.” (2020). citeturn775554academia68
