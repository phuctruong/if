<!-- extracted from ChatGPT (driver) on 2026-07-18 -->

# The Agency Threshold  
## Critical Conditions for the Evolution of Predictive Control in IF Universes

**Author:** Phuc Vinh Truong  
**Series:** IF Theory Working Papers  
**Paper:** 5  
**Date:** July 18, 2026  
**Status:** Theoretical and computational proposal awaiting preregistration, implementation, and falsification

---

## Abstract

Persistent structures may react to their environments without predicting them. A thermostat responds to current temperature; a chemical network follows local concentrations; an attractor returns toward a stable state after disturbance. Predictive agency requires something more: a system must physically maintain internal information about future-relevant conditions, use that information to select among actions, and obtain enough additional work or viability to repay the costs of sensing, memory, prediction, computation, and control.

This paper proposes the **IF Agency-Threshold Hypothesis**: predictive control becomes evolutionarily sustainable only when the causal benefit enabled by an internal model exceeds the complete physical cost of maintaining and using that model. For a predictive controller \(P\) and a matched reactive controller \(R\), define:

\[
\Delta J
=
J_P-J_R,
\]

where \(J\) is net physical return after all modeled costs. Define the dimensionless predictive-return number:

\[
\boxed{
\Pi_A
=
\frac{
\Delta W_{\mathrm{enabled}}
+
\chi_V\Delta V_{\mathrm{physical}}
}{
C_{\mathrm{sense}}
+
C_{\mathrm{memory}}
+
C_{\mathrm{prediction}}
+
C_{\mathrm{control}}
}.
}
\]

Because physical work and viability generally have different units, the preferred analysis reports them separately. The combined expression above is permitted only where viability has been converted into an explicitly defined physical continuation value \(\chi_V\Delta V_{\mathrm{physical}}\). In the primary experiments, the work-based ratio is:

\[
\boxed{
\Pi_A^{W}
=
\frac{
\Delta W_{\mathrm{enabled}}
}{
C_{\mathrm{model}}
}.
}
\]

Predictive control is physically profitable when:

\[
\Pi_A^{W}>1.
\]

The simplest evolutionary model produces a stability exchange at:

\[
\Pi_A^{W}=1.
\]

Below the threshold, model-bearing systems lose resources relative to reactive controls. Above it, predictive systems can increase in frequency. At finite population size, with mutation, noise, heterogeneous environments, and frequency-dependent interactions, the sharp boundary may become a smooth crossover, a hysteretic transition, or disappear entirely. The theory therefore distinguishes a **physical break-even threshold**, an **evolutionary invasion threshold**, and a genuine **collective phase transition**. These are not assumed to coincide.

The paper defines a primary agency order parameter, evolutionary protocols without intelligence rewards, intervention requirements, model-complexity transitions, finite-size tests, and cross-substrate validation. It also specifies conditions under which the agency-threshold claim must be rejected. The strongest possible result would be a transferable dimensionless relationship that prospectively predicts when predictive internal models emerge and persist across independently designed artificial, chemical, biological, and robotic substrates. A result confined to one toy environment would remain a useful computational observation but would not constitute a universal physical law of agency.

---

## Keywords

Agency; predictive control; artificial life; phase transition; bifurcation; information thermodynamics; semantic information; evolutionary dynamics; internal models; empowerment; causal intervention; resource-constrained intelligence.

---

# 1. Introduction

A system may persist without sensing.

A system may sense without remembering.

A system may remember without predicting.

A system may predict without using its prediction.

A system may use prediction while spending more resources than the prediction saves.

These distinctions define the problem of agency.

The thermodynamics of prediction distinguishes information about the past from information that predicts future environmental states. A system can retain extensive nonpredictive memory while dissipating more energy than a better-compressed predictive representation. Predictive information can therefore be physically relevant without implying intelligence or autonomy by itself. citeturn905371search0

Intervention-based semantic-information theory asks whether correlations between a system and its environment are causally necessary for continued viability. It does so by scrambling selected correlations and comparing the original and intervened systems. That framework already supplies a formal bridge from syntactic information to system-relative meaning and agency-like behavior. citeturn905371search1

Empowerment provides another neighboring idea. It measures the channel capacity from an agent’s actions to future sensory states, quantifying the range of future states the agent can reliably influence. Empowerment is useful for measuring potential control, but high potential influence does not prove that an internal predictive model generated a net physical benefit. citeturn905371search4

Information-bottleneck research has also demonstrated mathematically and computationally that changes in the tradeoff between compression and prediction can produce sharp representational transitions. Such transitions concern the onset of useful predictive representations; they do not automatically establish the emergence of autonomous physical agents. citeturn876473academia1turn876473academia2

The IF research question is therefore not:

> Can information help control a system?

That is already established.

The narrower question is:

\[
\boxed{
\text{Under what physical and environmental conditions does an}
\atop
\text{internally maintained predictive model become sufficiently useful}
\atop
\text{to pay for itself and persist through physical selection?}
}
\]

A related question is:

\[
\boxed{
\text{Does that transition possess a transferable critical structure,}
\atop
\text{or is “agency” merely a gradual, substrate-specific continuum?}
}
\]

Paper 2 defined causal-work value through matched model interventions. Paper 5 asks when systems that possess positive causal-work information arise and remain stable in populations.

---

# 2. Scope

This paper studies the emergence of **predictive physical agency**.

It does not attempt to establish:

- phenomenal consciousness;
- metaphysical free will;
- moral responsibility;
- human-level reasoning;
- subjective experience;
- cosmic agency;
- divine intention.

The primary systems are:

- finite-state agents;
- resource-constrained spatial agents;
- evolving cellular or graph-based structures;
- stochastic controllers;
- simple model-based planners.

The primary environments are:

- Markov resource processes;
- hidden-state environments;
- spatially correlated resource fields;
- periodically changing conditions;
- regime-switching environments;
- adversarially perturbed environments.

A system qualifies as predictive only relative to:

- a declared system boundary;
- an environment family;
- an evaluation horizon;
- a controller comparator;
- a physical cost model;
- a causal intervention.

---

# 3. The Three Thresholds

The phrase **agency threshold** can refer to three different phenomena.

They must not be conflated.

## 3.1 Physical break-even threshold

A predictive model reaches physical break-even when the additional work it enables equals its incremental physical cost:

\[
\boxed{
\Delta W_{\mathrm{enabled}}
=
C_{\mathrm{model}}.
}
\]

Equivalently:

\[
\boxed{
\Pi_A^{W}=1.
}
\]

This is a cost-benefit equality.

It is not automatically a phase transition.

---

## 3.2 Evolutionary invasion threshold

A predictive phenotype can invade a reactive population when its expected net reproductive or continuation rate exceeds that of the resident phenotype:

\[
\boxed{
f_P-f_R>0.
}
\]

If reproductive output is directly proportional to accumulated physical surplus, then:

\[
f_P-f_R
\propto
\Delta W_{\mathrm{enabled}}-C_{\mathrm{model}}.
\]

Under those restricted conditions, the invasion threshold may coincide with:

\[
\Pi_A^{W}=1.
\]

If reproduction depends on nonlinear survival, cooperation, density, or developmental constraints, the thresholds may differ.

---

## 3.3 Collective critical transition

A genuine critical transition would require population- or system-level evidence such as:

- nonanalytic behavior in an infinite-size limit;
- finite-size scaling;
- divergent or peaked susceptibility;
- critical slowing;
- correlation-length growth;
- hysteresis or bistability;
- scaling collapse;
- reproducible universality classes.

The existence of a physical break-even point does not prove a thermodynamic phase transition.

The project must permit the conclusion:

> Predictive control exhibits an evolutionary crossover but no universal critical transition.

---

# 4. Prior Art and Novelty Boundary

## 4.1 Thermodynamics of prediction

Still and colleagues showed that a system’s inefficiency is related to information retained about the past that fails to predict the future. This establishes a physical motivation for compressing memory toward future-relevant variables. citeturn905371search0

IF Theory cannot claim novelty for the proposition:

> Efficient systems should retain predictive rather than irrelevant historical information.

The IF extension tests whether the net benefit of that predictive information determines the evolutionary persistence of model-bearing systems.

---

## 4.2 Semantic information

Kolchinsky and Wolpert define information as semantic for a system when disrupting selected correlations reduces a viability function. They also discuss the automatic selection of system boundaries, timescales, and decompositions relevant to agency. citeturn905371search1turn905371search8

IF Theory cannot claim novelty for:

- interventionally defined useful information;
- viability-based informational value;
- information-relative agency.

Its proposed contribution is a cost-aware evolutionary threshold using net physical return and model ablation.

---

## 4.3 Empowerment

Empowerment measures potential control through action-to-future-state channel capacity. It has been used as an intrinsic objective capable of producing nontrivial behavior without task-specific rewards. citeturn905371search4turn905371search22

IF Theory distinguishes:

\[
\text{potential influence}
\]

from:

\[
\text{realized net causal-work benefit}.
\]

A system may have many reachable future states but lack:

- resources to exploit them;
- a predictive model;
- viable policies;
- physical efficiency.

Empowerment remains an important baseline.

---

## 4.4 Information-bottleneck transitions

The information bottleneck balances compressed representations against predictive relevance. Formal and empirical work has identified sharp transitions where nontrivial predictive representations become learnable or where new predictive components appear as the tradeoff parameter changes. citeturn876473academia1turn876473academia2

These results establish that predictive representation can undergo mathematically defined transitions.

They do not establish that the same boundary governs:

- physical self-maintenance;
- resource competition;
- evolutionary invasion;
- endogenous action.

IF Theory must test the bridge rather than assume it.

---

## 4.5 Good-regulator and internal-model principles

Control theory has long associated successful regulation with internal models of the regulated system. Recent embodied formulations examine when a physically situated agent must model its environment to regulate it effectively. citeturn905371search34

IF Theory cannot claim that good agents need models as an original general proposition.

Its question is:

> When does the physical benefit of an internal model exceed its complete cost, and is that boundary transferable?

---

## 4.6 Predictive representations in reinforcement learning

Predictive-information objectives have improved sample efficiency in reinforcement-learning systems, indicating that compressed future-relevant representations can support learning and control. citeturn876473academia3

Such agents are normally optimized using externally defined learning objectives and computing infrastructure.

The strongest IF experiment instead requires predictive control to emerge through local physical resource competition without a direct prediction, reward, or intelligence objective.

---

## 4.7 Provisional novelty claim

The potentially novel contribution is not the general existence of predictive control.

It is the following combined claim:

\[
\boxed{
\begin{gathered}
\text{A dimensionless, intervention-validated, physically accounted}
\\
\text{benefit-to-cost ratio prospectively predicts when internal}
\\
\text{predictive models emerge, invade, persist, and increase in}
\\
\text{complexity across multiple independently designed substrates.}
\end{gathered}
}
\]

This remains a conjecture until demonstrated.

---

# 5. Definitions

## 5.1 Reactive controller

A reactive controller selects action from present observation:

\[
A_t
=
\pi_R(O_t).
\]

It may contain fixed parameters but has no physically instantiated state carrying information beyond the current observation.

---

## 5.2 Memory-dependent controller

A memory-dependent controller uses:

\[
A_t
=
\pi_M(O_t,M_t),
\]

where:

\[
M_{t+1}
=
U_M(M_t,O_t,A_t).
\]

Memory may encode the past without predicting the future.

---

## 5.3 Predictive controller

A predictive controller contains internal state \(M_t\) satisfying:

\[
I(M_t;Y_{t+\tau}\mid O_t)>0
\]

for a future-relevant variable \(Y\).

It must also use that state causally:

\[
M_t
\rightarrow
A_t
\rightarrow
Y_{t+\tau}
\text{ or }
V_{t+\tau}.
\]

---

## 5.4 Adaptive predictive controller

An adaptive predictive controller updates its predictive mechanism when the environmental transition law changes.

It must outperform a frozen predictive controller in at least some held-out regime shifts after including the adaptation cost.

---

## 5.5 Counterfactual controller

A counterfactual controller evaluates at least two action-conditioned future distributions:

\[
P(Y_{t+\tau}\mid do(A_t=a),M_t)
\]

for multiple \(a\).

It selects action using those alternatives.

This is a higher agency level than simple one-step prediction.

---

## 5.6 Endogenous controller

The information and action-selection process must occur within the declared physical system boundary.

A remote oracle supplying correct actions does not create local agency unless the oracle’s complete physical costs and role are included as part of the agent.

---

# 6. Physical Return

Let useful work exported over horizon \(\tau\) be:

\[
W_{\mathrm{out}}(\tau).
\]

Let complete controller cost be:

\[
C_{\mathrm{ctrl}}
=
C_{\mathrm{sense}}
+
C_{\mathrm{memory}}
+
C_{\mathrm{prediction}}
+
C_{\mathrm{compute}}
+
C_{\mathrm{communication}}
+
C_{\mathrm{actuation}}
+
C_{\mathrm{repair}}
+
C_{\mathrm{reset}}.
\]

Define net work return:

\[
\boxed{
J_W(\tau)
=
\mathbb E
\left[
W_{\mathrm{out}}(\tau)
-
C_{\mathrm{ctrl}}(\tau)
\right].
}
\]

For predictive and reactive controllers:

\[
\Delta J_W
=
J_W^{P}-J_W^{R}.
\]

Positive predictive advantage requires:

\[
\boxed{
\Delta J_W>0.
}
\]

---

# 7. Causal Validation

A predictive controller may outperform a reactive controller for reasons unrelated to prediction.

The internal model must therefore pass the Paper 2 intervention family.

## 7.1 Scrambled model

Replace \(M_t\) with a matched state whose predictive relationship has been destroyed.

## 7.2 Temporally displaced model

Use:

\[
M_{t-\Delta}
\]

instead of \(M_t\).

## 7.3 Irrelevant model

Replace the predictive state with an equally large representation of an irrelevant variable.

## 7.4 Policy disconnection

Retain \(M_t\) but remove the path from model to action.

## 7.5 False model

Provide systematically wrong predictions at matched computational cost.

A system counts as predictively controlled only if the intact model outperforms these controls in the preregistered pattern.

---

# 8. Predictive-Return Number

Define gross enabled work:

\[
\Delta W_{\mathrm{enabled}}
=
W_{\mathrm{out}}^P-W_{\mathrm{out}}^R.
\]

Define incremental model cost:

\[
C_{\mathrm{model}}
=
C_{\mathrm{ctrl}}^P-C_{\mathrm{ctrl}}^R.
\]

Then:

\[
\boxed{
\Pi_A^W
=
\frac{
\Delta W_{\mathrm{enabled}}
}{
C_{\mathrm{model}}
}.
}
\]

Where:

\[
C_{\mathrm{model}}>0.
\]

Interpretation:

\[
\Pi_A^W<1:
\quad
\text{prediction is physically unprofitable},
\]

\[
\Pi_A^W=1:
\quad
\text{prediction reaches physical break-even},
\]

\[
\Pi_A^W>1:
\quad
\text{prediction produces a net work surplus}.
\]

The net surplus is:

\[
\boxed{
\mathcal W_{\mathrm{net}}
=
C_{\mathrm{model}}
\left(
\Pi_A^W-1
\right).
}
\]

---

# 9. Viability Return

Work output is not the only meaningful physical outcome.

Let:

\[
S(\tau)
\]

be survival probability or persistence to horizon \(\tau\).

Let:

\[
T_{\mathrm{life}}
\]

be expected lifetime.

Let:

\[
B_{\mathrm{op}}(\tau)
\]

be future operational battery capacity.

Report:

\[
\Delta S
=
S_P-S_R,
\]

\[
\Delta T_{\mathrm{life}}
=
T_{\mathrm{life}}^P-T_{\mathrm{life}}^R,
\]

\[
\Delta B_{\mathrm{op}}
=
B_{\mathrm{op}}^P-B_{\mathrm{op}}^R.
\]

These are not directly added to joules.

A combined objective is allowed only if the physical rules themselves convert continuation into expected future resource flow.

For example:

\[
V_{\mathrm{physical}}
=
\mathbb E
\left[
\int_0^\tau
P_{\mathrm{survive}}(t)
P_{\mathrm{capture}}(t)\,dt
\right].
\]

The conversion must be explicit.

---

# 10. Primary Agency Order Parameter

The fraction of systems possessing intact predictive causal value is:

\[
\boxed{
m_A
=
\frac{1}{N}
\sum_{i=1}^{N}
\mathbf 1
\left[
\mathcal W_{C,i}>0
\land
I(M_i;Y_{\mathrm{future}}\mid O_i)>I_{\min}
\land
\Delta A_i^{\mathrm{disconnect}}>A_{\min}
\right].
}
\]

Here:

- \(\mathcal W_{C,i}\) is interventionally measured causal-work value;
- \(I_{\min}\) is fixed from estimator null distributions;
- \(\Delta A_i^{\mathrm{disconnect}}\) measures the change in action behavior after policy disconnection;
- \(A_{\min}\) is preregistered.

The primary evolutionary order parameter is:

\[
\boxed{
x_P
=
\frac{
N_{\mathrm{predictive}}
}{
N_{\mathrm{population}}
}.
}
\]

Both are reported because a genetically or architecturally model-bearing system may not actually use prediction in a given environment.

---

# 11. Minimal Analytical Environment

Consider an environment with resource state:

\[
Y_t\in\{L,R\}.
\]

The state persists with probability:

\[
P(Y_{t+1}=Y_t)=r.
\]

A reactive controller sees no cue before committing to an action and succeeds with probability:

\[
q_R=\frac12.
\]

A one-step predictive controller stores \(Y_t\) and chooses it at \(t+1\), succeeding with probability:

\[
q_P=r.
\]

Let successful capture provide useful work:

\[
W_F.
\]

Let predictive-model cost per cycle be:

\[
C_M.
\]

Then:

\[
J_R
=
\frac12W_F-C_R,
\]

\[
J_P
=
rW_F-C_R-C_M.
\]

Therefore:

\[
\boxed{
\Delta J
=
\left(
r-\frac12
\right)W_F-C_M.
}
\]

The physical threshold is:

\[
\boxed{
r_c
=
\frac12+\frac{C_M}{W_F}.
}
\]

Prediction is profitable when:

\[
r>r_c.
\]

Equivalently:

\[
\Pi_A^W
=
\frac{
(r-\frac12)W_F
}{
C_M
}
>1.
\]

---

# 12. Evolutionary Stability

Let:

\[
x
\]

be the fraction of predictive controllers.

Assume, initially, frequency-independent returns.

Let:

\[
f_P=J_P,
\qquad
f_R=J_R.
\]

The replicator equation is:

\[
\dot x
=
x
\left(
f_P-\bar f
\right),
\]

where:

\[
\bar f
=
xf_P+(1-x)f_R.
\]

This reduces to:

\[
\boxed{
\dot x
=
x(1-x)\Delta J.
}
\]

If:

\[
\Delta J<0,
\]

then:

\[
x=0
\]

is stable and:

\[
x=1
\]

is unstable.

If:

\[
\Delta J>0,
\]

then:

\[
x=1
\]

is stable and:

\[
x=0
\]

is unstable.

At:

\[
\Delta J=0,
\]

the two phenotypes have equal expected return.

This is a stability exchange governed by:

\[
\Pi_A^W=1.
\]

It is an evolutionary bifurcation in the idealized deterministic population model.

It is not automatically a thermodynamic phase transition.

---

# 13. Mutation and Finite Populations

Let predictive controllers mutate to reactive controllers and vice versa at rate:

\[
\mu.
\]

A simple replicator–mutation equation is:

\[
\dot x
=
x(1-x)\Delta J
+
\mu(1-2x).
\]

For:

\[
\mu>0,
\]

the absorbing states disappear.

The sharp stability exchange becomes a smoother transition.

In finite populations, drift allows:

- predictive agents below break-even;
- loss of predictive agents above break-even;
- threshold broadening;
- fixation delays.

Therefore, the empirical order parameter should be:

\[
P_{\mathrm{fix}}(P)
\]

or stationary predictive frequency, not one deterministic outcome.

The width of the transition should scale with:

- population size;
- mutation rate;
- environmental variance;
- evaluation horizon.

---

# 14. Model Complexity Thresholds

Prediction is not binary.

Let model complexity be:

\[
K\in\{0,1,\ldots,K_{\max}\}.
\]

Let predictive accuracy be:

\[
q(K).
\]

Let model cost be:

\[
C(K).
\]

Net return is:

\[
J(K)
=
q(K)W_F-C(K)-C_R.
\]

The selected complexity is:

\[
\boxed{
K^*
=
\arg\max_K
\left[
q(K)W_F-C(K)
\right].
}
\]

As environmental predictability or resource value changes, \(K^*\) may change:

- smoothly;
- through discrete jumps;
- through hysteresis;
- through coexistence of multiple strategies.

The information-bottleneck literature provides precedent for sharp representational changes as prediction–compression tradeoffs vary, but IF Theory must test whether comparable transitions occur under physical resource selection rather than a chosen optimization multiplier. citeturn876473academia1turn876473academia2

---

# 15. Hidden-State Environments

A one-step Markov environment rewards simple memory.

To test genuine model formation, use a hidden Markov environment.

Let latent state be:

\[
H_t\in\{1,\ldots,K\}.
\]

Observations satisfy:

\[
P(O_t\mid H_t).
\]

Resources satisfy:

\[
P(Y_{t+\tau}\mid H_t).
\]

A reactive controller receives \(O_t\).

A predictive controller maintains belief:

\[
b_t(h)
=
P(H_t=h\mid O_{\leq t},A_{<t}).
\]

It updates:

\[
b_{t+1}
=
\mathcal B
\left(
b_t,O_{t+1},A_t
\right).
\]

A model becomes useful only when integrating observations improves future action enough to repay inference cost.

This allows testing:

- state-estimation depth;
- memory compression;
- uncertainty;
- model mismatch;
- belief revision.

---

# 16. Environmental Predictability

Agency should not be maximal in every environment.

## 16.1 Fully predictable environment

A fixed reflex may be sufficient.

The environment can be predictable yet not require an internal model.

## 16.2 Moderately structured environment

A predictive model may provide strong advantage.

## 16.3 Fully random environment

Future prediction is impossible.

Model maintenance becomes wasteful.

This suggests an intermediate relationship:

\[
\boxed{
\text{predictive agency may peak at intermediate environmental complexity,}
}
\]

not necessarily at maximum predictability.

The relevant quantity is **actionable predictability beyond current observation**:

\[
I_{\mathrm{actionable}}
=
I(Y_{t+\tau};O_{<t}\mid O_t,A_t).
\]

If this is zero, memory adds no predictive advantage.

---

# 17. Environmental Volatility and Adaptation

A fixed model can become harmful after a regime shift.

Let the environmental transition matrix change at time:

\[
t_s.
\]

A fixed predictive controller retains the old model.

An adaptive controller pays additional update cost:

\[
C_{\mathrm{adapt}}.
\]

The adaptive model is profitable when:

\[
\int_{t_s}^{t_s+\tau}
\left[
W_A(t)-W_F(t)
\right]dt
>
C_{\mathrm{adapt}}.
\]

This creates a second threshold:

\[
\boxed{
\tau_{\mathrm{stable}}
>
\tau_{\mathrm{break-even}}^{\mathrm{adapt}}.
}
\]

If environmental regimes change faster than adaptation repays its cost, flexible modeling is selected against.

---

# 18. Exploration and Model Acquisition

Predictive models require data.

An agent may sacrifice immediate work to obtain information.

Let exploration cost be:

\[
C_{\mathrm{explore}}.
\]

Let future model-enabled surplus be:

\[
G_{\mathrm{future}}.
\]

Exploration is physically justified when:

\[
\boxed{
G_{\mathrm{future}}
>
C_{\mathrm{explore}}.
}
\]

This distinguishes:

- reactive exploitation;
- passive learning;
- active information seeking.

Empowerment and intrinsic-motivation frameworks provide important comparison baselines because they can generate exploratory behavior without task-specific reward. citeturn905371search4turn905371search29

The IF question is whether information-seeking behavior emerges from long-run resource accounting without adding an intrinsic information reward.

---

# 19. Strong Definition of Predictive Agency

A candidate system qualifies as predictively agentic under Paper 5 only when:

1. **Persistence:** It maintains an operational identity.
2. **Action:** Its internal state changes environmental outcomes.
3. **Internal model:** It physically maintains future-relevant state.
4. **Causal use:** Model ablation changes action and outcome.
5. **Positive return:** The model’s benefit exceeds its full cost under at least part of the environment family.
6. **Endogeneity:** No external oracle supplies decisions.
7. **Breadth:** The benefit survives held-out conditions.
8. **Adaptability:** For higher agency levels, the model can revise after failure.
9. **Evolutionary viability:** The phenotype can persist without a direct agency reward.

A controller that meets only conditions 1–4 possesses causal predictive information but may not possess physically sustainable agency.

---

# 20. Core Hypotheses

## AT-H1 — Physical break-even hypothesis

Predictive controllers achieve positive net physical advantage when:

\[
\Pi_A^W>1.
\]

### Falsifier

Complete accounting shows no relationship between \(\Pi_A^W\) and net return.

---

## AT-H2 — Evolutionary invasion hypothesis

A rare predictive phenotype invades a reactive population when its causal-work surplus is positive after all developmental and maintenance costs.

### Falsifier

Predictive invasion requires direct reward or parameters unrelated to causal-work value.

---

## AT-H3 — Predictability threshold hypothesis

For a fixed model cost and resource value, predictive control disappears below a calculable level of actionable environmental predictability.

### Falsifier

Predictive models persist in environments where past state contains no useful future information and no alternative benefit exists.

---

## AT-H4 — Finite-complexity hypothesis

Selected model complexity remains finite under nonzero sensing, memory, and computation costs.

### Falsifier

Unbounded model complexity is favored despite saturating predictive benefit and increasing physical cost.

---

## AT-H5 — Model-jump hypothesis

As resource value, predictability, or model cost varies, selected model complexity can undergo discrete transitions.

### Falsifier

No repeatable complexity transitions occur beyond implementation artifacts.

---

## AT-H6 — Adaptation-timescale hypothesis

Adaptive prediction is selected only when environmental regimes remain stable long enough for model revision to repay its cost.

### Falsifier

Adaptation persists when regime duration is systematically shorter than its physical break-even time.

---

## AT-H7 — Causal-specificity hypothesis

Predictive-state scrambling and policy disconnection reduce performance more strongly than matched irrelevant-state interventions.

### Falsifier

Any memory state of equal size performs equally well.

---

## AT-H8 — Reward-independence hypothesis

Predictive control can emerge under selection based solely on resource capture, maintenance, and reproduction, without a prediction or intelligence reward.

### Falsifier

Predictive behavior appears only when explicitly rewarded.

---

## AT-H9 — Cross-substrate hypothesis

A dimensionless causal-return ratio predicts predictive-controller prevalence better than raw model size, raw mutual information, or substrate-specific parameters across independent implementations.

### Falsifier

Every substrate requires unrelated thresholds and definitions.

---

## AT-H10 — Criticality hypothesis

Under at least one well-defined limit, the onset of predictive control exhibits finite-size scaling or another accepted signature of a collective transition.

### Falsifier

All apparent sharp transitions resolve into smooth finite-system crossovers without transferable scaling.

---

# 21. Evolutionary Simulation Design

## 21.1 Population

The environment contains a population of resource-constrained systems.

Each system has:

- an energy or capacity store;
- sensors;
- a controller;
- an action interface;
- optional internal memory;
- reproduction machinery;
- physical maintenance costs.

---

## 21.2 Reproduction

Reproduction occurs only when accumulated surplus exceeds a declared threshold:

\[
B_i>B_{\mathrm{rep}}.
\]

Reproduction costs:

\[
C_{\mathrm{rep}}.
\]

Offspring inherit:

- controller architecture;
- mutable parameters;
- memory capacity;
- model-update rules.

They do not inherit the parent’s current environmental knowledge unless physical copying is implemented and costed.

---

## 21.3 Death

A system ceases functioning when:

\[
B_i\leq0
\]

or when its structural viability conditions fail.

No abstract fitness score is required.

---

## 21.4 Mutation

Mutations may change:

- sensor precision;
- memory length;
- model order;
- update rate;
- planning depth;
- action policy;
- resource allocation.

Mutation cost and developmental cost must be included where relevant.

---

## 21.5 No intelligence reward

The simulator may not directly reward:

- accurate prediction;
- information gain;
- memory;
- model complexity;
- empowerment;
- agency score.

Prediction survives only if it improves physical continuation or reproduction.

---

# 22. Primary Experiments

## Experiment 1 — Analytical threshold recovery

Use the binary Markov environment.

Sweep:

\[
r,\quad W_F,\quad C_M.
\]

Test whether the simulated break-even boundary agrees with:

\[
r_c
=
\frac12+\frac{C_M}{W_F}.
\]

This validates implementation.

---

## Experiment 2 — Evolutionary invasion

Initialize a reactive population.

Introduce a rare predictive mutant.

Measure invasion probability against:

\[
\Pi_A^W.
\]

Compare with finite-population fixation theory.

---

## Experiment 3 — De novo model evolution

Begin with randomly parameterized controllers lacking a designated predictive module.

Allow mutation in internal-state capacity and update rules.

Test whether future-relevant internal states emerge only where:

\[
\Pi_A^W>1.
\]

---

## Experiment 4 — Predictive versus historical memory

Construct controllers with equal memory capacity.

One stores predictive variables.

One stores irrelevant past variables.

One stores shuffled histories.

Compare:

- work;
- persistence;
- reproduction;
- causal ablation.

---

## Experiment 5 — Model-complexity ladder

Use environments of increasing hidden-state order.

Allow controller complexity:

\[
K=0,\ldots,K_{\max}.
\]

Map selected:

\[
K^*(\text{predictability},W_F,C_M).
\]

Test for discrete transitions.

---

## Experiment 6 — Regime shifts

Change the environmental transition law.

Compare:

- reactive control;
- frozen prediction;
- adaptive prediction;
- meta-learning control.

Measure adaptation break-even time.

---

## Experiment 7 — Exploration threshold

Hide the environmental structure initially.

Allow information-seeking actions that consume resources.

Test when active exploration evolves without an intrinsic information reward.

---

## Experiment 8 — Partial observability

Vary observation noise and hidden-state ambiguity.

Test whether predictive memory becomes useful only within a bounded region:

- low ambiguity: reaction is sufficient;
- intermediate ambiguity: prediction helps;
- extreme ambiguity: prediction cannot recover the state.

---

## Experiment 9 — Action necessity

Create conditions where prediction is accurate but no available action can alter the outcome.

IF predicts:

\[
I_{\mathrm{pred}}>0
\]

but:

\[
\mathcal W_C\approx0.
\]

This distinguishes forecasting from agency.

---

## Experiment 10 — Resource-value threshold

Hold predictive accuracy fixed.

Vary resource value.

Prediction should disappear when the available consequence is too small to repay modeling cost.

---

## Experiment 11 — Structural agents

Embed controllers in candidate structures discovered in Paper 3.

Test whether the same threshold predicts the emergence of internal predictive states in self-maintaining structures.

---

## Experiment 12 — Expansion coupling

Run predictive agents inside Paper 4 expanding domains.

Test whether the agency threshold shifts with:

- crowding;
- dilution;
- topology turnover;
- coordination time.

---

# 23. Critical-Transition Tests

## 23.1 Order parameter

Primary:

\[
x_P
=
\text{population fraction with intervention-validated predictive control}.
\]

Secondary:

\[
m_A,
\quad
\langle K^*\rangle,
\quad
\langle\mathcal W_C\rangle,
\quad
I_{\mathrm{pred}},
\quad
P_{\mathrm{fix}}.
\]

---

## 23.2 Control parameter

Possible control parameters include:

\[
\lambda_1
=
\Pi_A^W,
\]

\[
\lambda_2
=
\frac{
I_{\mathrm{actionable}}
}{
C_{\mathrm{model}}/W_F
},
\]

\[
\lambda_3
=
\frac{
\tau_{\mathrm{environment}}
}{
\tau_{\mathrm{model}}
}.
\]

The primary control variable is frozen before confirmatory analysis.

---

## 23.3 Susceptibility

Define response to a small change in model benefit:

\[
\chi_A
=
\frac{
\partial \langle x_P\rangle
}{
\partial \Pi_A^W
}.
\]

In simulations:

\[
\chi_A
\approx
\frac{
\langle x_P\rangle_{\Pi+\delta}
-
\langle x_P\rangle_{\Pi-\delta}
}{
2\delta
}.
\]

A peak may indicate transition sensitivity but is not sufficient by itself.

---

## 23.4 Critical slowing

After perturbing controller frequency or model parameters, measure return time:

\[
\tau_{\mathrm{return}}.
\]

Increasing return time near the threshold would support critical slowing.

---

## 23.5 Finite-size scaling

Run populations:

\[
N\in
\{N_1,N_2,\ldots,N_{\max}\}.
\]

Test whether transition width scales as:

\[
\Delta\Pi_A(N)
\propto
N^{-1/\nu}
\]

or another derived form.

Exponents must not be interpreted as universal without cross-model replication.

---

## 23.6 Hysteresis

Sweep the control parameter upward and downward.

Hysteresis may arise when:

- models require developmental investment;
- predictive agents change the environment;
- social learning reduces model cost;
- frequency-dependent benefits create bistability.

No hysteresis should be claimed without controlled sweep protocols and equilibrium-time checks.

---

# 24. Agency Phase Taxonomy

## A-P0 — Passive phase

Persistent structures exist without action-dependent control.

## A-P1 — Reactive phase

Current observations drive action, but no future-relevant internal model is required.

## A-P2 — Memory phase

Internal state affects action but contains primarily historical rather than predictive information.

## A-P3 — Predictive-correlational phase

Internal state predicts the future but ablation shows little causal contribution.

## A-P4 — Predictive-control phase

Prediction causally improves action, but model cost exceeds benefit.

## A-P5 — Sustainable-agency phase

Predictive control produces positive net work or continuation benefit:

\[
\Pi_A^W>1.
\]

## A-P6 — Adaptive-agency phase

Systems revise their models after environmental change.

## A-P7 — Counterfactual-agency phase

Systems compare action-dependent futures.

## A-P8 — Social-predictive phase

Systems model other agents.

## A-P9 — Institutional phase

Agents create persistent shared records or constraints that reduce individual modeling costs and stabilize collective action.

Paper 5 focuses on phases A-P1 through A-P6.

---

# 25. Deterministic Jupyter-Notebook Program

## Notebook 05A — Binary Threshold Derivation

Derive and numerically verify:

\[
r_c
=
\frac12+\frac{C_M}{W_F}.
\]

Produce exact and simulated phase maps.

---

## Notebook 05B — Replicator Dynamics

Implement:

\[
\dot x
=
x(1-x)\Delta J.
\]

Verify stability and break-even behavior.

---

## Notebook 05C — Replicator–Mutation Dynamics

Add mutation and quantify transition smoothing.

Compare deterministic equations with stochastic birth–death simulations.

---

## Notebook 05D — Finite-Population Fixation

Measure:

\[
P_{\mathrm{fix}}(P)
\]

across population sizes and causal-return values.

---

## Notebook 05E — Hidden Markov Environment

Implement exact Bayesian predictors of varying model order.

Validate beliefs against known latent states.

---

## Notebook 05F — Predictive-Information Estimation

Estimate:

\[
I(M_t;Y_{t+\tau}\mid O_t).
\]

Validate bias and uncertainty using systems with known distributions.

---

## Notebook 05G — Causal Model Ablations

Apply:

- erasure;
- scrambling;
- temporal displacement;
- irrelevant replacement;
- policy disconnection;
- false models.

---

## Notebook 05H — Physical Cost Model

Implement explicit costs for:

- sensing;
- memory;
- belief update;
- planning;
- action;
- copying;
- resetting.

---

## Notebook 05I — Model-Complexity Transitions

Sweep environmental complexity and model cost.

Track:

\[
K^*.
\]

Search for discrete representation jumps.

---

## Notebook 05J — De Novo Evolution

Evolve finite-state controllers without prediction rewards.

Measure whether predictive states emerge.

---

## Notebook 05K — Environmental Regime Shifts

Test fixed versus adaptive models.

Estimate adaptation break-even time.

---

## Notebook 05L — Exploration Without Intrinsic Reward

Allow costly information-seeking actions.

Determine when exploration emerges through future physical return.

---

## Notebook 05M — Critical Slowing and Susceptibility

Measure transition response, recovery time, and population variance.

---

## Notebook 05N — Finite-Size Scaling

Test whether the agency transition sharpens or remains a crossover.

---

## Notebook 05O — Paper 3 Structural Integration

Insert evolving controllers into resource-conserving IF structures.

---

## Notebook 05P — Paper 4 Expansion Integration

Measure agency thresholds across domain-growth regimes.

---

## Notebook 05Q — Cross-Substrate Replication

Repeat the core threshold in:

1. finite-state agents;
2. spatial cellular agents;
3. graph-based organisms;
4. stochastic chemical controllers;
5. simple robotic simulations.

---

## Notebook 05R — Adversarial Audit

A separate agent attempts to show that predictive control is caused by:

- reward leakage;
- hidden oracle access;
- unmatched architecture;
- uncounted training energy;
- biased information estimators;
- hand-selected environments;
- arbitrary agency thresholds.

---

# 26. Reproducibility Record

Each experiment emits:

```yaml
experiment_id: if-agency-threshold-05
paper_version: null
git_commit: null
environment_hash: null
implementation: null

environment_family: null
environment_parameters: {}
resource_value: null
environment_predictability: null
regime_duration: null

population_size: null
mutation_rate: null
initial_predictive_fraction: null
random_seed: 65537
time_horizon: null

reactive_controller_cost: null
sensor_cost: null
memory_cost: null
prediction_cost: null
planning_cost: null
actuation_cost: null
reset_cost: null

reactive_work_output: null
predictive_work_output: null
enabled_work: null
model_cost: null
agency_ratio: null

predictive_information: null
historical_information: null
causal_work_vector: {}
causal_viability_vector: {}

predictive_fraction_history: null
fixation_probability: null
selected_model_complexity: null
transition_width: null
return_time: null
susceptibility: null

invariant_failures: []
result_hash: null
```

---

# 27. Statistical Standards

## 27.1 Environment holdout

Environmental processes used to evolve controllers must be separated from those used to evaluate generalization.

---

## 27.2 Architecture matching

Predictive and reactive controllers should be matched as closely as possible in:

- action capacity;
- state count;
- update frequency;
- physical substrate;
- access to observations.

The incremental model cost must not be confounded with unrelated architecture changes.

---

## 27.3 Search correction

If many model classes and environment families are searched, confirmatory results require:

- held-out environments;
- frozen metrics;
- multiple-comparison correction;
- independent replication.

---

## 27.4 Transition preregistration

Before opening confirmatory sweeps, freeze:

- control parameter;
- order parameter;
- transition criterion;
- system-size sequence;
- scaling analysis;
- hysteresis protocol.

---

## 27.5 Null models

Required nulls include:

- random memory;
- past-only memory;
- perfect prediction with no action influence;
- action influence with no prediction;
- uncosted oracle upper bound;
- model-free controller with matched parameter count.

---

# 28. Failure Modes

## 28.1 Prediction reward leakage

The objective directly rewards prediction accuracy or mutual information.

## 28.2 Hidden fitness score

Reproduction depends on an abstract score rather than physical surplus or viability.

## 28.3 External training subsidy

Model training occurs outside the energy boundary.

## 28.4 Architecture mismatch

Predictive controllers have more actions, sensors, or computational updates than controls without paying for them.

## 28.5 Information-estimator bias

High-dimensional memory appears predictive because of finite-sample estimation error.

## 28.6 Correlation without causal use

The model predicts accurately, but actions do not depend on it.

## 28.7 Action without prediction

A complex policy performs well through memorized reflexes and is mislabeled predictive.

## 28.8 Environment overfitting

Agents succeed only in the exact process used during evolution.

## 28.9 Threshold-by-definition

The order parameter is constructed so that it changes at:

\[
\Pi_A=1.
\]

Evolutionary prevalence and causal behavior must be independently measured.

## 28.10 Phase-transition inflation

A logistic crossover is described as a critical phenomenon without scaling evidence.

## 28.11 System-boundary manipulation

Training, memory preparation, or oracle cost is excluded to force:

\[
\Pi_A>1.
\]

## 28.12 Intelligence inflation

One-step prediction is described as consciousness or free will.

---

# 29. What Would Count as Success?

## Level 1 — Analytical break-even recovery

Simulation reproduces the exact toy-model threshold.

## Level 2 — Evolutionary invasion

Causal-work surplus predicts invasion of predictive controllers.

## Level 3 — De novo predictive emergence

Predictive internal states evolve without direct intelligence rewards.

## Level 4 — Model-complexity prediction

The framework predicts which model complexity is selected.

## Level 5 — Adaptive threshold

Regime duration and adaptation cost predict when flexible models evolve.

## Level 6 — Cross-environment generalization

The ratio predicts outcomes in held-out environment families.

## Level 7 — Cross-substrate scaling

One nondimensional relationship organizes agency onset across independent substrates.

## Level 8 — Laboratory validation

The same law predicts the onset of predictive control in physical or biological systems.

## Level 9 — General bound

A theorem establishes a broad physical limit relating predictive information, control benefit, and model cost.

---

# 30. What Would Count as a Major Discovery?

A strong artificial-life paper would show:

> Predictive controllers evolve without a prediction reward when their interventionally measured physical benefit exceeds their full cost.

A field-creating result would show:

\[
\boxed{
\text{One dimensionless causal-return law prospectively predicts the}
\atop
\text{emergence and selected complexity of internal models across}
\atop
\text{independently designed computational and physical substrates.}
}
\]

A still stronger result would derive a universal inequality:

\[
\mathcal W_C
\leq
\mathcal F
\left(
I_{\mathrm{pred}},
B_{\mathrm{gross}},
C_{\mathrm{control}},
\tau
\right)
\]

and demonstrate systems approaching the bound.

---

# 31. Relationship to the Informational Battery

Paper 1 defined:

\[
B_{\mathrm{gross}},
\quad
B_{\mathrm{op}},
\quad
B_{\mathrm{latent}}.
\]

Predictive control can increase operational access:

\[
\Delta B_{\mathrm{op}}>0
\]

without increasing gross physical capacity.

Agency emerges when the accessibility gain repays its own cost:

\[
\boxed{
\Delta B_{\mathrm{op}}
>
C_{\mathrm{model}}.
}
\]

The model does not create energy.

It reduces the fraction of capacity that remains inaccessible.

---

# 32. Relationship to Emergent Structure

Paper 3 supplies candidate persistent structures.

Paper 5 tests whether some structures develop internal states whose disruption selectively impairs future resource access.

A self-repairing attractor is not predictive merely because it returns to a stable form.

The distinguishing intervention is:

\[
\text{scramble future-relevant internal information}
\]

while preserving:

- material;
- energy;
- architecture;
- current morphology.

If recovery or resource capture declines selectively, the internal information has causal value.

---

# 33. Relationship to Expansion

Paper 4 predicts a sustainable domain-growth window.

Expansion may alter agency thresholds by changing:

- encounter rates;
- memory usefulness;
- environmental stationarity;
- communication delay;
- resource density;
- model-update cost.

Possible prediction:

\[
\Pi_A^{W}
=
F
\left(
g,
\tau_{\mathrm{coord}},
\tau_{\mathrm{environment}}
\right).
\]

Very rapid substrate change may make deep models obsolete before they repay their cost.

Moderate growth may create enough novelty and opportunity to favor prediction.

---

# 34. Relationship to Consciousness

Predictive physical agency is not consciousness.

A one-step model may satisfy:

\[
\Pi_A^W>1
\]

without possessing:

- self-modeling;
- global access;
- competing policy coordination;
- deep counterfactual reasoning;
- subjective experience.

The progression remains:

\[
\boxed{
\text{reactive control}
\rightarrow
\text{predictive agency}
\rightarrow
\text{counterfactual agency}
\rightarrow
\text{functional consciousness hypotheses}.
}
\]

Each transition requires separate evidence.

---

# 35. Relationship to Free Will

Paper 5 may establish that internal models causally influence action.

This supports a functional sense of endogenous control:

\[
M_t
\rightarrow
A_t
\rightarrow
Y_{t+\tau}.
\]

It does not establish metaphysical indeterminism.

A deterministic agent can possess internal causal control under this framework.

Whether such control is sufficient for free will is a philosophical question beyond the physical threshold.

---

# 36. Criteria for Rejection or Major Revision

The agency-threshold hypothesis should be rejected or substantially revised if:

1. predictive advantage does not track complete benefit-to-cost accounting;
2. model interventions do not selectively affect outcomes;
3. prediction emerges only through direct rewards;
4. environment-specific parameters dominate all results;
5. selected model complexity cannot be predicted;
6. apparent thresholds vanish under architecture matching;
7. no result survives held-out environments;
8. cross-substrate scaling fails;
9. the transition is entirely an estimator artifact;
10. simpler semantic-information or empowerment measures predict all outcomes equally well;
11. physical and evolutionary thresholds cannot be meaningfully connected;
12. the project repeatedly redefines agency to preserve positive results.

---

# 37. Conclusion

Prediction is not automatically agency.

Agency is not automatically consciousness.

The IF Agency-Threshold Hypothesis begins with a narrower proposition:

\[
\boxed{
\text{An internal predictive model becomes physically sustainable when}
\atop
\text{the additional accessible work or continuation it enables exceeds}
\atop
\text{the complete cost of acquiring, maintaining, and using the model.}
}
\]

The physical break-even ratio is:

\[
\boxed{
\Pi_A^W
=
\frac{
\Delta W_{\mathrm{enabled}}
}{
C_{\mathrm{model}}
}.
}
\]

The candidate threshold is:

\[
\boxed{
\Pi_A^W=1.
}
\]

Below it, predictive machinery is a net burden.

Above it, prediction can become self-financing.

Whether this physical boundary produces:

- evolutionary invasion;
- stable predictive populations;
- discrete increases in model complexity;
- critical collective behavior;
- or a universal physical law

is an empirical question.

The theory succeeds only if the boundary predicts outcomes it was not designed around.

The strongest version of the claim is:

\[
\boxed{
\text{Predictive agency is a physical transition in accessibility:}
\atop
\text{matter begins maintaining models of possible futures when doing so}
\atop
\text{reliably reveals more usable capacity than the models consume.}
}
\]

If that relationship fails to transfer beyond a toy environment, the universal agency-threshold hypothesis fails.

If it predicts the emergence of internally modeled control across artificial, chemical, biological, and engineered systems, it may provide a measurable physical bridge from self-organizing matter to agency.

---

# References

1. Still, S., Sivak, D. A., Bell, A. J. and Crooks, G. E. “The Thermodynamics of Prediction.” *Physical Review Letters* 109, 120604 (2012). citeturn905371search0

2. Kolchinsky, A. and Wolpert, D. H. “Semantic Information, Autonomous Agency and Non-equilibrium Statistical Physics.” *Interface Focus* 8, 20180041 (2018). citeturn905371search1turn905371search8

3. Salge, C., Glackin, C. and Polani, D. “Empowerment—An Introduction.” (2013). citeturn905371search4

4. Mohamed, S. and Rezende, D. J. “Variational Information Maximisation for Intrinsically Motivated Reinforcement Learning.” (2015). citeturn905371search22

5. Wu, T. and Fischer, I. “Phase Transitions for the Information Bottleneck in Representation Learning.” (2020). citeturn876473academia1

6. Wu, T., Fischer, I., Chuang, I. L. and Tegmark, M. “Learnability for the Information Bottleneck.” (2019). citeturn876473academia2

7. Lee, K.-H. et al. “Predictive Information Accelerates Learning in Reinforcement Learning.” (2020). citeturn876473academia3

8. Virgo, N. “A Good Regulator Theorem for Embodied Agents.” (2025). citeturn905371search34

9. Tiomkin, S. et al. “Control Capacity of Partially Observable Dynamic Systems.” (2017). citeturn905371search30

10. Tiomkin, S. et al. “Intrinsic Motivation in Dynamical Control Systems.” (2022–2023). citeturn905371search29
