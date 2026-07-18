# The IF Causal-Work Principle
## When Predictive Information Becomes Physical Agency

**Author:** Phuc Vinh Truong
**Series:** IF Theory Canonical Papers
**Paper:** 2
**Layer:** SCIENCE
**Status:** canonical revision of extracted draft (2026-07-18)
**Supersedes:** canon/extracted/paper-02-extracted.md

---

## Abstract

Physical systems frequently contain information about their environments. Crystals preserve traces of their formation, thermostats respond to temperature, organisms retain memories, and intelligent agents construct models of possible futures. Correlation alone, however, does not establish agency. Information may be physically stored yet irrelevant to action, predictive yet unused, or useful but more costly to maintain than the physical benefit it enables.

This paper proposes the **IF Causal-Work Principle**, an intervention-based criterion for identifying when predictive information becomes part of a system's physical agency. A bounded system is evaluated under an intact condition and under matched interventions that erase, scramble, temporally displace, or falsify selected internal information while preserving relevant physical and statistical properties. The causal-work contribution is the change in net useful work after all incremental costs of sensing, memory, computation, communication, control, and action are included — costs that couple the information ledger to the energy and thermodynamic-entropy ledgers via Landauer-type terms (Paper 0 §6, Paper 1 §7).

For internal model \(M\), environment \(E\), and horizon \(\tau\), the central quantity is provisionally:

\[
\mathcal W_C(M;\tau)
=
J_{\mathrm{intact}}(\tau)
-
J_{\mathrm{ablated}}(\tau),
\qquad
J(\tau)
=
\mathbb E
\left[
W_{\mathrm{useful}}(\tau)
-
C_{\mathrm{total}}(\tau)
\right].
\]

A model carries positive causal-work value when \(\mathcal W_C>0\). Positive causal-work value alone is not sufficient for full agency: simple feedback controllers may satisfy it. IF Theory therefore defines **predictive physical agency** through a conjunction: bounded persistence, endogenous information maintenance, action-dependent environmental influence, positive causal-work value of predictive internal states, adaptive performance across multiple environments, and the absence of an external controller supplying the relevant decisions.

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

where \(\Delta W_{\mathrm{enabled}}\) is the additional useful work enabled by the intact model relative to a matched model-free or scrambled system. The strongest IF hypothesis is that predictive agency becomes selectively sustainable when \(\Pi_A>1\). The paper presents analytical toy models, intervention standards, simulation architectures, phase-transition tests (with an explicit kill condition), falsification conditions, and a deterministic Jupyter-notebook program. The prospective IF contribution is not the broad claim that intelligence converts information into work; it is the discovery — if it exists — of a transferable intervention-based boundary predicting when internally maintained information pays for itself physically and becomes constitutive of autonomous agency.

---

## Keywords

Agency; predictive information; information thermodynamics; causal intervention; work extraction; autonomy; semantic information; viability; artificial life; physical intelligence; phase transition.

---

# 1. Introduction

A physical system may contain information without being an agent. A rock records pressure and temperature. A crystal stores structural regularities. A camera stores images. A thermostat contains and uses information about present temperature. A bacterium senses chemical gradients. An animal retains memories and anticipates future resource locations. These systems differ not merely in how much information they contain, but in what that information *does*.

The central challenge is to distinguish: information that exists; information that predicts; information that causally changes action; and information whose physical benefit exceeds its cost.

Information thermodynamics has established that information-processing systems must be evaluated as physical systems — measurement, feedback, memory, work extraction, and erasure participate in thermodynamic ledgers rather than operating outside them. The thermodynamics of prediction (Still, Sivak, Bell & Crooks) further distinguishes predictive information from memory that records the past without helping anticipate the future; nonpredictive memory may contribute to dissipation without improving control. Kolchinsky and Wolpert developed an intervention-based account of semantic information for autonomous nonequilibrium systems, asking which syntactic correlations matter to continued viability by intervening on them. Causal-emergence research (Hoel, Albantakis, Tononi) investigates whether macrolevel descriptions can possess greater effective causal organization than microlevel ones, providing a possible method for identifying agents without assuming the programmer's preferred partition. Recent physical-intelligence work (2025–2026) includes measures based on goal-directed work per unit of irreversibly processed information, information or empowerment per joule, rare-valid future amplification, and interaction-level predictive structure (Takahashi & Hayashi; Hafez et al.; Fagan; Chattopadhyay).

Consequently, IF Theory cannot claim novelty for any of: intelligent systems are physical; information processing has thermodynamic costs; predictive information is more useful than indiscriminate memory; information may enable work extraction; agency requires interaction between action and outcome; viable systems can contain information meaningful to their persistence; or intelligence can be normalized by energy or information-processing cost.

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

This paper addresses **functional physical agency**. It does not claim to explain phenomenal consciousness, metaphysical free will, moral responsibility, subjective experience, human-level intelligence, cosmic purpose, or divine action (the layer firewall; interpretation lives in `canon/30-meaning/`).

The initial domain: finite-state stochastic systems; deterministic artificial environments; resource-constrained controllers; artificial-life simulations; adaptive agents; evolutionary populations. Later tests may extend to chemical reaction networks, active matter, microbial behavior, cellular regulatory systems, robots, and AI systems. No cross-substrate universality is assumed before empirical demonstration.

---

# 3. Prior Art and the Novelty Boundary

## 3.1 Thermodynamics of prediction

Still, Sivak, Bell, and Crooks separated stored information about past environmental states from information predictive of future states, relating thermodynamic inefficiency to nonpredictive retained information. IF Theory cannot claim that predictive information is generally more thermodynamically valuable than irrelevant historical memory as an original insight. The IF extension: evaluate the information through controlled ablations and quantify the resulting change in net physical work and autonomous persistence.

## 3.2 Semantic information and viability

Kolchinsky and Wolpert proposed that information becomes semantic when interventions removing it reduce a specified viability measure, explicitly addressing system–environment decomposition, intervention, timescale, and agency identification. IF Theory cannot claim novelty for defining meaningful information through intervention. Its proposed distinction is to keep two effects separate: **causal-work value** and **causal-viability value**. A system might obtain more work while reducing its long-term persistence; conversely, information might improve survival without increasing exported work. The two outcomes are reported independently, not collapsed into a single favorable score.

## 3.3 Causal emergence

Hoel, Albantakis, and Tononi showed that appropriately chosen macroscopic causal models can exhibit greater effective information than microscopic models when degeneracy and noise are reduced at the macrolevel. IF Theory may use causal emergence to identify candidate agent boundaries or macrostates, but cannot equate higher effective information with agency automatically. A macrodescription may be causally informative without self-maintenance, prediction, endogenous action, resource acquisition, or policy adaptation.

## 3.4 Empowerment and control information

Empowerment measures the channel capacity between an agent's actions and future sensory states; recent physical-intelligence work proposes empowerment per unit energetic cost (Takahashi & Hayashi). IF Theory cannot claim novelty for measuring action influence or control information per joule. The IF intervention asks a different question: *how much net useful physical output or persistence disappears when a particular internal predictive representation is selectively destroyed?* Empowerment measures potential influence; causal-work ablation measures the realized physical contribution of a selected internal model.

## 3.5 Goal-directed work per information cost

A recent physical theory of intelligence defines intelligence using goal-directed work produced per unit of irreversibly processed information (Fagan). A separate 2026 proposal defines thermodynamic intelligence as the lawful amplification of rare but valid futures, arguing recursive self-simulation is necessary under stated assumptions (Chattopadhyay). These strongly overlap with IF Theory's motivation. The IF program must demonstrate a specific advantage rather than presenting another renamed work-efficiency ratio. The prospective distinctions are:

1. a matched causal intervention on an internal model;
2. explicit separation of gross battery capacity from model-enabled accessibility (Paper 1);
3. comparison of intact, erased, scrambled, displaced, and false internal models;
4. a search for a transferable threshold across independently designed substrates;
5. automatic detection of candidate agents rather than assuming the agent boundary.

This distinction remains provisional and must survive a deeper formal literature review.

---

# 4. Conceptual Requirements for Physical Agency

A useful definition should not classify every causal system as an agent. A falling stone changes its environment; a heat engine extracts work; a thermostat uses information; a bacterium navigates; a planning organism evaluates counterfactual futures. IF Theory treats agency as graded and requires multiple conditions.

**4.1 Bounded organization.** A candidate system \(A\) must be distinguishable from environment \(E\) over a stated interval. The boundary (spatial, causal, thermodynamic, informational, or functional) must be selected through a documented rule, not because it makes the result favorable.

**4.2 Persistence.** The candidate must maintain organizational identity across time or component turnover. Let \(V_t\) define its viable organization; it must remain within or repeatedly return to a viability region \(\mathcal V\).

**4.3 Action.** The system must possess internal transitions that alter its coupling to the environment. With action variables \(A_t\), an intervention on \(A_t\) must change the distribution of future states:

\[
P(E_{t+\tau},V_{t+\tau}\mid do(A_t=a))
\neq
P(E_{t+\tau},V_{t+\tau}\mid do(A_t=a')).
\]

**4.4 Endogenous information.** The relevant internal state must be acquired, updated, preserved, or selected within the declared system boundary. A lookup table externally updated with the correct answer at every step does not establish autonomous prediction.

**4.5 Predictive content.** The internal model \(M_t\) must contain information about a future-relevant variable beyond the immediately available state: \(I(M_t;E_{t+\tau}\mid E_t)>0\). Predictive information alone remains insufficient.

**4.6 Causal use.** Changing \(M_t\) while controlling relevant alternatives must change actions and outcomes; the causal path must include \(M_t\rightarrow A_t\rightarrow E_{t+\tau}\text{ or }V_{t+\tau}\). A representation correlated with behavior but ignored by the policy does not contribute agency.

**4.7 Net physical benefit.** The benefit produced by the model must exceed the physical cost of maintaining and using it under at least some environments. This is the central IF condition.

**4.8 Generalization or adaptation.** A fixed reflex may be useful in one environment. Predictive agency requires performance across a defined environment family or adaptation when environmental statistics change; the system must not receive a newly hand-coded response for every tested condition.

---

# 5. Formal Setup

Let the total system be \(\Omega=(A,E,R)\), where \(A\) is the candidate agent, \(E\) the environment, and \(R\) physical reservoirs (fuel, heat, matter, work stores). The candidate agent contains:

\[
A_t
=
(X_t,M_t,U_t),
\]

with internal physical state \(X_t\), memory/predictive-model state \(M_t\), and action/control state \(U_t\). Let \(Y_t\) denote future-relevant environmental variables and \(W_{\mathrm{out}}[0,\tau]\) useful work transferred to a declared work reservoir. Physical costs within the same boundary:

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

The system boundary must include externally prepared low-entropy memory, remote computation, sensor power, and actuator power. Physical-intelligence metrics become misleading when boundary closure, reset, horizon, and imported low-entropy resources are omitted (Takahashi & Hayashi).

---

# 6. The Intervention Family

No single ablation is sufficient — destroying memory can also change energy, dynamics, architecture, and action capacity. IF Theory requires a family of matched interventions.

**6.1 Erasure.** Replace \(M_t\) with a default \(m_0\). Tests dependence on stored state but may alter state entropy and physical cost.

**6.2 Permutation scrambling.** Apply a bijection \(M_t\rightarrow\sigma(M_t)\) preserving marginal frequencies but destroying the intended memory–environment correspondence. Useful when labels are arbitrary and action mappings held fixed.

**6.3 Cross-episode scrambling.** Assign an internal model from another independent episode, \(M_t^{(i)}\rightarrow M_t^{(j)},\ i\neq j\). Approximately preserves model complexity and physical representation while destroying episode-specific predictive content.

**6.4 Temporal displacement.** Replace with a delayed or advanced model, \(M_t\rightarrow M_{t-\Delta}\) (or \(M_{t+\Delta}\), used only as a diagnostic upper bound). Tests whether temporal alignment matters.

**6.5 Predictive-variable scrambling.** Preserve information about past states while destroying information specifically predictive of future states:

\[
I(\widetilde M_t;E_{t-\tau_p})
\approx
I(M_t;E_{t-\tau_p}),
\qquad
I(\widetilde M_t;E_{t+\tau_f}\mid E_t)\ \text{reduced}.
\]

**6.6 Equal-capacity irrelevant model.** Replace the useful model with an equally large and similarly costly representation of an irrelevant variable. Distinguishes model size from predictive content.

**6.7 False-model intervention.** Provide systematically inaccurate predictions with matched confidence and computational cost. Determines whether the policy uses model content rather than merely model presence.

**6.8 Policy-disconnection intervention.** Preserve \(M_t\) but remove the causal edge \(M_t\rightarrow U_t\). Tests whether the information is used by action selection.

---

# 7. Definition of Causal-Work Value

Let \(\mathcal I_0\) denote the intact condition and \(\mathcal I_k\) an intervention. For each condition:

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

A robust effect requires positive values across multiple appropriate interventions. A single large erasure effect does not establish predictive causal work because erasure may damage the controller nonspecifically.

---

# 8. Gross Benefit, Model Cost, and the Agency Ratio

To study whether maintaining a model pays for itself, separate gross enabled work from incremental model cost. Let \(W_{\mathrm{use}}^{M}\) be useful work with the model and \(W_{\mathrm{use}}^{0}\) useful work under a matched controller lacking that model. Define gross model-enabled work:

\[
\Delta W_{\mathrm{enabled}}
=
W_{\mathrm{use}}^{M}
-
W_{\mathrm{use}}^{0}.
\]

Define incremental model cost (excluding work differences already counted):

\[
C_M
=
C_{\mathrm{total}}^{M}
-
C_{\mathrm{total}}^{0}.
\]

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

Where \(C_M>0\), define the candidate agency ratio:

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

Interpretation: \(\Pi_A<1\) — the model costs more than the work it enables; \(\Pi_A=1\) — physical break-even; \(\Pi_A>1\) — the model produces net-positive work value. This ratio is not automatically universal; its meaning depends on the boundary, horizon, model-free comparator, and definition of useful work.

---

# 9. Viability Must Remain Separate

An organism may spend energy to survive without exporting useful mechanical work; a predator may extract considerable work while increasing its risk of death. Work and viability therefore require separate measures. Let:

\[
P_{\mathcal V}(\tau)
=
P(V_t\in\mathcal V\text{ for }0\leq t\leq\tau)
\]

or another preregistered viability function. Define causal viability:

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

A candidate system qualifies as a **predictive physical agent relative to environment family \(\mathcal E\), horizon \(\tau\), and boundary \(\partial A\)** when all of the following hold:

- **A — Persistence.** Maintains an operational identity or viability distribution over the interval.
- **B — Endogenous action.** Its internal state causally affects environmental or resource transitions.
- **C — Endogenous model.** The predictive state is acquired, maintained, or updated within the declared boundary.
- **D — Predictive content.** The internal model predicts future-relevant states beyond current observation.
- **E — Causal use.** Matched model interventions change actions and outcomes through the model-to-policy path.
- **F — Positive net contribution.** At least one preregistered outcome satisfies \(\mathcal W_{\mathrm{net}}>0\) or a positive causal-viability criterion, with all costs reported.
- **G — Environmental breadth.** The effect survives across a declared set of environments or after environmental change.
- **H — No external decision oracle.** The correct actions are not supplied by an uncounted external system.

This definition is relative, operational, and graded. It does not assert that all agents satisfy one sharp metaphysical boundary.

---

# 11. Agency Ladder

- **A0 — Passive persistence.** Fixed physical stability (crystals, static attractors).
- **A1 — Reactive regulation.** Current sensory input changes present action (a thermostat may qualify).
- **A2 — Memory-dependent control.** Past internal state changes present action.
- **A3 — Predictive control.** Internal state contains future-relevant information that causally improves outcome.
- **A4 — Counterfactual control.** Evaluates multiple possible action-dependent futures.
- **A5 — Self-modeling control.** The model represents aspects of the system's own future state or limitations.
- **A6 — Policy revision.** Changes how it selects policies when its model repeatedly fails.
- **A7 — Social modeling.** Predicts the states and actions of other agents.
- **A8 — Institutional agency.** Multiple systems create persistent shared constraints, records, or coordination structures that expand collective control.

Paper 2 addresses primarily the transition from A1–A2 to A3. Reflection and higher-order agency are deferred to later papers.

---

# 12. Analytical Toy Model I: Predictive Resource Choice

Two locations \(L,R\); one contains a resource worth \(W_F\); the system chooses one. A random controller succeeds with probability \(\tfrac12\); a predictive model with \(q>\tfrac12\). Let model cost be \(C_M\) and other action costs equal. Additional expected work enabled:

\[
\Delta W_{\mathrm{enabled}}
=
\left(
q-\tfrac12
\right)W_F.
\]

Net model value:

\[
\boxed{
\mathcal W_{\mathrm{net}}
=
\left(
q-\tfrac12
\right)W_F-C_M.
}
\]

Candidate threshold:

\[
\boxed{
\Pi_A
=
\frac{
(q-\tfrac12)W_F
}{
C_M
}.
}
\]

The model is physically profitable when \(\Pi_A>1\). This depends jointly on prediction accuracy, resource value, and model cost. High accuracy is not sufficient if the environment offers little benefit or the model is too expensive.

---

# 13. Analytical Toy Model II: Environmental Persistence

Let resource location follow a two-state Markov process with \(P(Y_{t+1}=Y_t)=r\). At \(r=\tfrac12\) the previous location offers no predictive value; at \(r>\tfrac12\) the environment has persistence. An agent storing the previous location predicts \(\hat Y_{t+1}=Y_t\) with accuracy \(q=r\). The model is net beneficial when:

\[
\boxed{
\left(
r-\tfrac12
\right)W_F>C_M.
}
\]

The minimum environmental predictability required is:

\[
\boxed{
r_c
=
\tfrac12+\frac{C_M}{W_F}.
}
\]

This explicit boundary predicts: expensive memory requires a more predictable environment; valuable resources support predictive systems at lower predictability; in a fully unpredictable environment, model maintenance is wasteful; increasing memory efficiency lowers the agency threshold. This is a model-specific break-even boundary, not yet a universal law.

---

# 14. Analytical Toy Model III: Finite Memory Depth

Let the environment contain temporal dependencies up to order \(K\), model depth \(L\). Assume predictive benefit rises and saturates, \(\Delta W(L)=W_{\max}(1-e^{-L/\ell})\), and model cost rises approximately linearly, \(C_M(L)=cL\). Then:

\[
\mathcal W_{\mathrm{net}}(L)
=
W_{\max}
\left(
1-e^{-L/\ell}
\right)-cL,
\]

with optimum from \(\frac{W_{\max}}{\ell}e^{-L/\ell}-c=0\):

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
\qquad\text{when}\quad
W_{\max}>c\ell,
\]

and \(L^*=0\) otherwise. This predicts a discontinuous onset of nonzero model depth in the simplified optimization problem, although finite populations and stochastic learning may smooth the transition — a point directly relevant to the phase-transition tests of §19.

---

# 15. The IF Causal-Work Principle

**15.1 Weak form.** Internal information contributes to physical agency when matched interventions that selectively disrupt its predictive or policy-relevant content reduce the system's net physical return or viability.

**15.2 Cost-aware form.** Predictive information becomes physically self-sustaining when the additional work or persistence it enables exceeds the complete incremental cost of acquiring, storing, protecting, updating, and using it.

**15.3 Strong threshold conjecture.** Across a meaningful class of resource-constrained adaptive systems, the onset of persistent predictive agency is organized by a dimensionless causal-work ratio near physical break-even, \(\Pi_A\approx1\).

**15.4 Universality conjecture.** After appropriate nondimensionalization, different substrates exhibit common scaling behavior near the causal-work threshold.

The weak and cost-aware forms are operational definitions. The strong threshold and universality forms are empirical conjectures (see §19 kill condition).

---

# 16. Why Positive Causal Work Is Not Sufficient by Itself

A thermostat may use one bit of state to save heating energy and therefore have \(\mathcal W_C>0\). Calling it a minimal agent may be acceptable under a broad graded definition, but it does not make it a predictive or reflective agent. The stronger classification requires internal predictive state, temporal horizon, environmental generalization, endogenous model maintenance, and adaptive policy selection. IF Theory therefore rejects the equation \(\mathcal W_C>0\Rightarrow\text{full agency}\). Instead, \(\mathcal W_C>0\) is evidence that selected information has physical causal value; agency classification requires the remaining criteria (§10).

---

# 17. Simulation Architecture

**17.1 Environment.** A spatial or graph-based domain with localized energy resources; resource depletion and regeneration; hazards; environmental states with tunable predictability; explicit action and transport costs. Predictability is controlled via Markov persistence, periodicity, spatial correlation length, volatility, hidden-state switching, and observation noise.

**17.2 Candidate systems.** Compare: passive dissipative structures; fixed reactive controllers; one-step-memory controllers; predictive finite-state controllers; learned predictive controllers; model-based planners. All obey the same physical resource ledger. Model-based systems pay explicit costs increasing with sensor precision, memory capacity, model complexity, communication, and planning depth.

**17.3 Endogenous persistence.** Systems receive no abstract reward points. Effective fitness arises through maintaining internal energy, avoiding destructive states, continuing to act, repairing, and reproducing where enabled. Evolutionary runs may select by persistence and reproduction, but the simulator must not directly reward "intelligence," "prediction," or "agency" (the Conway gate; `TELEOLOGY_INJECTION` is forbidden).

**17.4 Agent-boundary detection.** Initially boundaries may be declared for controlled tests. Later, declared boundaries should be compared with automatically detected candidates using causal partitions, integrated information flow, transfer entropy, persistent connected components, resource-flow closure, intervention sensitivity, and causal emergence. A result depending entirely on a hand-selected favorable boundary is weak.

---

# 18. Experimental Program

1. **Binary predictive choice.** Validate \((q-\tfrac12)W_F>C_M\); vary \(q,W_F,C_M\), action cost, sensor noise.
2. **Predictive versus historical memory.** Construct memories with matched size and past mutual information but different future-predictive information; test whether \(\mathcal W_C\) tracks predictive information better than total historical storage.
3. **Scrambling hierarchy.** Compare all intervention types; a robust predictive model should show large loss under predictive-variable scrambling and policy disconnection, smaller loss under irrelevant-memory scrambling, graded loss under temporal displacement.
4. **Environmental predictability sweep.** Vary persistence/temporal structure; measure evolved model depth, causal-work value, survival, energy efficiency, prediction accuracy; test whether model-bearing systems disappear below a critical predictability.
5. **Model-cost sweep.** Increase sensing/memory/computation cost; test whether model complexity decreases, predictive agency disappears, reactive control remains, and a break-even boundary can be recovered.
6. **Environmental regime shift.** Change the environment's transition law; compare fixed predictor, adaptive predictor, reactive controller, no-memory controller; predictive agency should include recovery after model failure, not merely high stationary performance.
7. **Evolution without agency reward.** Initialize random resource-processing structures; allow mutation/selection only through persistence and reproduction; test whether predictive internal states emerge near the calculated causal-work boundary.
8. **Cross-rule-family replication.** Repeat using finite-state Markov agents, cellular automata, reaction networks, recurrent neural controllers, graph-based organisms. The strongest claim requires a common nondimensional relationship.

---

# 19. Phase-Transition Tests

The phrase "agency threshold" should not be used casually. A smooth cost-benefit crossover is not necessarily a statistical-mechanical phase transition. To support a genuine transition claim, the program tests for: order-parameter behavior; finite-size scaling; divergent or peaked susceptibility; critical slowing; hysteresis; bimodal state distributions; scaling collapse; robustness across system size. Candidate order parameters:

\[
\langle \mathcal W_C\rangle,
\qquad
P(\Pi_A>1),
\qquad
I(M_t;E_{t+\tau}\mid E_t),
\]

and the fraction of surviving systems whose model ablation reduces performance.

**Explicit kill condition (panel round 1, adopted).** If net viability (or the order parameter) scales smoothly and linearly with information capacity across all tested rule families — no discontinuity, no hysteresis, no critical slowing, no scaling collapse — the phase-transition claim (15.3–15.4) is FALSE. Agency is then a **gradient, not a state of matter**. This result is recorded in the SCOREBOARD kill log the session it fires. The causal-work measure (\(\mathcal W_C\), \(\Pi_A\)) survives as a valid measurement tool even if the transition dies; only the strong/universality conjectures fall. A null result — only a smooth crossover — remains scientifically informative.

---

# 20. Core Hypotheses

## CW-H1 — Intervention hypothesis
Scrambling future-relevant internal information while preserving relevant physical and statistical controls reduces net work or viability.
**Falsifier:** Matched scrambling produces no selective reduction.

## CW-H2 — Predictive-specificity hypothesis
Causal-work value is more strongly associated with predictive information than with total memory size or information about the past.
**Falsifier:** Irrelevant or nonpredictive memories provide equal net benefit after costs and architecture are matched.

## CW-H3 — Cost threshold hypothesis
Predictive models persist only when \(\Delta W_{\mathrm{enabled}}>C_M\).
**Falsifier:** Model-bearing systems remain selectively favored even when their complete physical cost persistently exceeds their physical benefit.

## CW-H4 — Finite-complexity hypothesis
When model cost rises with complexity and environmental predictability is finite, an optimal noninfinite model complexity exists.
**Falsifier:** Additional model complexity is always net beneficial despite nonzero costs.

## CW-H5 — Adaptive-agency hypothesis
Systems capable of revising their models outperform fixed predictors after environmental regime shifts when the long-run value of adaptation exceeds its added cost.
**Falsifier:** Adaptation provides no net benefit across the preregistered regime family.

## CW-H6 — Emergent-agency hypothesis
Predictive internal models arise through physical selection without an explicit prediction or intelligence reward in environments where \(\Pi_A>1\).
**Falsifier:** Predictive models emerge only when directly rewarded or manually installed.

## CW-H7 — Scaling hypothesis
Different substrates exhibit an approximately shared relationship between \(\Pi_A\) and the persistence of predictive control.
**Falsifier:** Every substrate requires unrelated thresholds or arbitrary rescaling. (This is the strong-form kill condition of §19.)

---

# 21. Deterministic Jupyter-Notebook Program

- **02A — Causal-Work Analytical Baselines.** Exact calculations for binary resource choice, Markov-persistent resources, finite memory depth, model-cost break-even; validate simulation against closed-form equations.
- **02B — Intervention Library.** Implement erase, permute, cross-episode scramble, temporal displacement, irrelevant-model replacement, false-model substitution, policy disconnection; test that each preserves its declared controls.
- **02C — Predictive Information Estimators.** Estimate \(I(M_t;E_{t+\tau}\mid E_t)\) via exact distributions, plug-in, bias-corrected, and k-NN estimators; validate on synthetic systems with known information.
- **02D — Physical Cost Ledger.** Track sensing, memory, computation, actuation, communication, repair, reset, external imports; fail the run when accounting does not close within tolerance.
- **02E — Causal-Work Ablation.** Measure \(\mathbf W_C\) and \(\mathbf V_C\) for all intervention classes; generate causal diagrams and path-specific controls.
- **02F — Predictability–Cost Phase Map.** Sweep \((r,C_M,W_F,\text{noise},\tau)\); map regions where no control / reactive control / predictive control / adaptive prediction persists.
- **02G — Finite-Size Scaling.** Test whether the apparent boundary sharpens with population size, system size, evaluation horizon; distinguish a true transition from a finite-system crossover (the §19 kill test).
- **02H — Evolution Without Intelligence Reward.** Evolve controllers under physical energy balance and reproduction only; test whether model-bearing structures emerge near predicted regions.
- **02I — Cross-Substrate Replication.** Repeat the causal-work analysis across at least three independently coded model classes.
- **02J — Adversarial Reproduction.** A separate coding agent receives only the paper, public configurations, and raw outputs, and must independently reproduce the principal result and attempt to destroy it.

---

# 22. Reproducibility Record

Every experiment emits:

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

Canonical results use deterministic CPU calculations where possible. Stochastic results report seeds, sample counts, confidence intervals, convergence diagnostics, and sensitivity to estimator choice.

---

# 23. Statistical Standards

**23.1 Holdout environments.** Model design and parameter selection use training environments; primary claims use held-out environmental transition laws.

**23.2 Multiple comparisons.** Large sweeps create many false-discovery opportunities; the primary order parameter, threshold, and transition criteria must be frozen before final analysis (the Popper gate).

**23.3 Model comparison.** Compare no-memory controls, reactive controls, predictive models, alternative information measures, standard RL metrics, empowerment, and semantic-information measures. IF Theory must demonstrate incremental explanatory value.

**23.4 Robustness.** Vary system boundary, time horizon, coarse-graining, cost model, information estimator, intervention type, viability definition. A result surviving only one favorable specification is not strong.

---

# 24. Failure Modes

1. **Hidden oracle** — an external process supplies correct actions/labels without its cost entering the boundary.
2. **Unmatched ablation** — removing memory also reduces architecture, action capacity, or energy.
3. **Circular utility** — the output measure directly rewards possession of the model.
4. **Cost omission** — training, sensing, external computation, or resetting is ignored.
5. **Correlation mistaken for use** — the model predicts the future but the policy does not depend on it.
6. **Action without autonomy** — an external controller makes the decisions.
7. **Survival-score arbitrariness** — the viability function guarantees the desired conclusion.
8. **Boundary manipulation** — the boundary is changed until the energy ratio becomes favorable.
9. **Substrate overfitting** — a threshold exists only in the original toy environment.
10. **Terminological inflation** — a simple control improvement is presented as consciousness or free will (`LAYER_COLLAPSE`).

---

# 25. What Would Count as a Major Result?

- **Level 1 — Valid computational measure.** Interventions reliably distinguish useful predictive information from irrelevant memory. Justifies the method, not a new law.
- **Level 2 — Robust threshold within one model family.** A repeatable break-even boundary predicts when predictive controllers persist. A publishable artificial-life or information-thermodynamics result.
- **Level 3 — Cross-model scaling.** The same nondimensional relationship predicts agency onset across independently designed simulation classes. Substantially more important.
- **Level 4 — Laboratory transfer.** The same relationship predicts transitions in chemical, active-matter, microbial, or robotic systems. Could establish a new physics-of-agency program.
- **Level 5 — Universal bound.** A theorem constrains predictive causal work for broad physical systems, and experiments approach or confirm the bound. Potentially field-changing.

---

# 26. Novelty Assessment

The current novelty score must be lower than initially assumed because several recent papers propose closely neighboring physical measures of intelligence and agency (Takahashi & Hayashi; Hafez et al.; Fagan; Chattopadhyay). The IF proposal is potentially distinctive only if it demonstrates all of the following together:

1. **Matched model intervention** — the selected internal information is experimentally disrupted rather than inferred from correlation.
2. **Net physical accounting** — the benefit is measured after complete incremental physical costs.
3. **Accessible-capacity interpretation** — information changes access to an existing nonequilibrium battery (Paper 1) rather than being treated as energy.
4. **Agent emergence** — predictive systems arise without direct intelligence rewards.
5. **Cross-substrate scaling** — one relationship transfers across substantially different implementations.
6. **Clear negative cases** — systems with large information stores but no causal work are correctly rejected.
7. **Prospective prediction** — the threshold predicts a result not used to construct it.

Without those results, the IF Causal-Work Principle is best viewed as a synthesis and experimental protocol, not a novel fundamental law.

---

# 27. Relationship to Consciousness

Positive causal-work value does not imply consciousness. A predictive controller may qualify as an agent while lacking global access, self-modeling, counterfactual depth, report, or subjective experience. Later IF work may ask whether counterfactual self-models introduce another transition, but Paper 2 makes no phenomenal-consciousness claim. The correct implication is \(\text{predictive causal work}\Rightarrow\text{functional agency evidence}\), not \(\text{predictive causal work}\Rightarrow\text{subjective awareness}\).

---

# 28. Relationship to Free Will

The framework evaluates whether internal models causally alter action and future outcomes. That can establish a functional form of endogenous control. It does not decide whether determinism is metaphysically compatible with freedom, whether actions could have occurred differently under identical total physical conditions, or whether moral responsibility follows from internal causation. Those remain philosophical questions (interpretation layer).

---

# 29. Relationship to IF Cosmology

The causal-work principle is not a dark-matter or dark-energy equation; its relevance to cosmology is indirect. If IF Theory later proposes that cosmic organization alters access to a universe-wide nonequilibrium state, Paper 2 supplies a discipline: *no informational contribution may be invoked unless a physical intervention, mechanism, cost, and measurable consequence are specified.* Cosmological structure formation cannot be called intelligent or agentic merely because it generates complexity.

---

# 30. Criteria for Rejection or Major Revision

Reject or substantially revise if: causal-work effects cannot be separated from ordinary controller architecture; matched information interventions cannot be constructed; work benefits disappear under complete physical accounting; predictive information performs no better than irrelevant stored information; the threshold is entirely determined by an arbitrary reward definition; evolved predictive systems require explicit intelligence rewards; no relationship transfers across rule families; existing semantic-information, empowerment, or physical-intelligence measures explain every result equally well with less machinery; the concept adds terminology without new predictions; or negative findings are repeatedly avoided by changing the system boundary or agency definition.

---

# 31. Conclusion

Information becomes scientifically relevant to agency not because it is mysterious, meaningful to a human observer, or abundant. It becomes relevant when it is physically embodied, future-directed, used by action, and worth its cost. The proposed IF criterion:

\[
\boxed{
\mathcal W_C(M;\tau)
=
J_{\mathrm{intact}}(\tau)
-
J_{\mathrm{ablated}}(\tau).
}
\]

The corresponding break-even ratio:

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

The physical interpretation:

\[
\boxed{
\Pi_A>1
\quad\Rightarrow\quad
\text{the predictive model enables more useful work than it costs.}
}
\]

That inequality does not, by itself, prove consciousness, reflection, autonomy in every sense, or a universal phase transition. It supplies a testable boundary between information that merely exists and information that physically pays for its own continued use. The strongest IF research question is therefore:

\[
\boxed{
\text{Across what classes of physical systems does net-positive}
\atop
\text{predictive information become a stable, self-maintaining cause}
\atop
\text{of action rather than a passive trace of the environment?}
}
\]

If no transferable relationship exists, the proposed universal principle fails (the §19 kill condition). If one relationship predicts the emergence of predictive control across artificial, chemical, biological, and engineered systems, IF Theory will have identified a plausible physical law of agency.

---

# References

1. Still, S., Sivak, D. A., Bell, A. J. and Crooks, G. E. "Thermodynamics of Prediction." *Physical Review Letters* 109, 120604 (2012).
2. Kolchinsky, A. and Wolpert, D. H. "Semantic Information, Autonomous Agency and Non-equilibrium Statistical Physics." *Interface Focus* 8, 20180041 (2018).
3. Hoel, E. P., Albantakis, L. and Tononi, G. "Quantifying Causal Emergence Shows That Macro Can Beat Micro." *PNAS* 110, 19790–19795 (2013).
4. Horowitz, J. M. and Esposito, M. "Thermodynamics with Continuous Information Flow." *Physical Review X* 4, 031015 (2014).
5. Perunov, N., Marsland, R. A. and England, J. L. "Statistical Physics of Adaptation." *Physical Review X* 6, 021036 (2016).
6. Parrondo, J. M. R. "Thermodynamics of Information." Review manuscript (2023).
7. Takahashi, K. and Hayashi, Y. "Thermodynamic Limits of Physical Intelligence." (2026).
8. Hafez, W. et al. "A Mathematical Theory of Agency and Intelligence." (2026).
9. Fagan, P. D. "Toward a Physical Theory of Intelligence." (2025–2026).
10. Chattopadhyay, I. "Thermodynamic Measure of Intelligence." (2026).
11. Halpern, N. Y. "Toward Physical Realizations of Thermodynamic Resource Theories." (2014–2016).
12. Marletto, C. "The Information-Theoretic Foundation of Thermodynamic Work Extraction." (2020).
