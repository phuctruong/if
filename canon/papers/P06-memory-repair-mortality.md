# Memory, Reflection, Repair, and Mortality
## Costly Self-Maintenance in IF Agents

**Author:** Phuc Vinh Truong
**Series:** IF Theory Canonical Papers
**Paper:** 6
**Layer:** SCIENCE
**Status:** canonical revision of extracted draft (2026-07-18)
**Supersedes:** canon/extracted/paper-06-extracted.md

---

## Abstract

Predictive agency creates a new physical problem. Once a system maintains memories and internal models, it must allocate finite resources among learning, deliberation, action, error correction, structural repair, reproduction, and continued existence. More memory is not always beneficial; deeper reflection may cost more than the decision is worth; perfect repair may be uneconomical; and indefinite maintenance may lose to reproduction, replacement, or lineage continuation.

This paper proposes a unified IF framework for four linked transitions: **memory** (when retained information becomes worth its acquisition and maintenance cost), **reflection** (when a system benefits from modeling and regulating its own cognition), **repair** (when damage detection and correction produce more future value than they consume), and **mortality** (when finite maintenance, accumulating damage, environmental hazard, and reproductive opportunity make indefinite persistence unsustainable). The theory does **not** claim aging or death is universally adaptive. Classical evolutionary theories — mutation accumulation, antagonistic pleiotropy, disposable soma — emphasize that the force of selection and the allocation of limited resources can permit imperfect late-life maintenance; the disposable-soma framework treats somatic maintenance and reproduction as competing investments, and later work stresses multiple interacting trade-offs rather than a single energy budget.

IF contributes a computationally explicit synthesis under the three-ledger discipline (energy / thermodynamic entropy / information, never merged — CLAUDE.md §1). An agent has a finite operational battery \(B_t\), memory complexity \(L_t\), reflective depth \(R_t\), accumulated damage \(D_t\), repair allocation \(u_t\), and reproductive allocation \(v_t\). Every cognitive and maintenance process carries a declared physical cost, and policies are evaluated through counterfactual ablation and held-out environments rather than through direct rewards for memory, reflection, repair, longevity, or intelligence (the Conway gate, CLAUDE.md §5). The central hypotheses are \(L^*=\arg\max_L[G_{\mathrm{prediction}}(L)-C_{\mathrm{memory}}(L)]\); reflect only when \(\operatorname{VOC}>C_{\mathrm{reflection}}\); active repair is most strongly selected over an intermediate recoverable damage range; and continued maintenance is favored only while its expected future causal value exceeds reproduction, replacement, or exit.

A finite lifespan counts as **emergent** only when identical local rules permit different lifespans under different resource, hazard, repair, and reproductive conditions — never when an age-death rule is installed. The strongest possible result is a transferable resource-allocation law predicting memory depth, reflective effort, repair investment, and lifespan across independently designed artificial substrates. A result confined to one engineered environment remains an artificial-life observation, not a universal law of biology.

---

## Keywords

Memory; reflection; metacognition; repair; aging; mortality; self-maintenance; resource allocation; artificial life; predictive agency; damage accumulation; disposable soma; value of computation.

---

# 1. Introduction

A predictive agent cannot spend all its resources predicting. It must also sense, act, preserve internal state, correct errors, repair damage, reproduce or replicate, and survive long enough for prediction to matter — a family of physical trade-offs. A system with no memory cannot exploit temporal structure; one with excessive memory may spend more maintaining irrelevant history than it gains; one that never reflects repeats costly mistakes; one that reflects before every trivial action drains its battery in deliberation; one that never repairs deteriorates; one that repairs every defect perfectly starves growth, reproduction, or immediate action.

The cost of cognition is not metaphorical. Neural signaling requires substantial metabolic expenditure — action potentials, postsynaptic currents, maintaining ionic gradients, transmitter recycling — establishing that biological information processing is physically budgeted (though no universal price per thought exists). Repair and longevity are likewise constrained: Kirkwood's disposable-soma proposal argued organisms may evolve less-than-perfect somatic maintenance because error prevention and repair compete with growth and reproduction, and modern reviews treat aging as multiple interacting genetic, physiological, ecological, and energetic trade-offs, not one programmed death mechanism. Computational metacognition supplies a neighboring framework for reflection: metacognitive systems monitor their own reasoning and intervene at a meta-level, improving behavior but adding computation, latency, and complexity.

The IF question is therefore whether memory, reflection, repair, and mortality can be *derived as resource-allocation outcomes* rather than installed as narrative labels — and, under what measurable conditions each process repays its cost. The project will not ask whether long memory, deep reflection, perfect repair, or immortality is inherently good.

---

# 2. Scope

Paper 6 studies resource allocation within persistent predictive agents. It does not prove that biological aging has one cause, that mortality is always adaptive, that death is necessary for evolution, that reflection implies subjective consciousness, that self-modeling produces free will, that longer life is morally superior, that reproduction is the only continuation, or that an artificial-agent result applies directly to humans (layer firewall, CLAUDE.md §6). Primary systems: finite-state predictive controllers, resource-conserving cellular agents, graph-based organisms, evolving artificial populations, stochastic damage-and-repair models. Primary outcomes: net physical work, operational battery capacity, persistence, reproductive/lineage output, prediction error, repair success, accumulated damage, selected memory complexity, selected reflective effort, lifespan distribution.

---

# 3. Prior Art and the Novelty Boundary

**3.1 Memory has physical cost.** Information must be physically instantiated; biological memory requires molecular/synaptic/electrical/structural processes, computational memory requires hardware state and maintenance. IF cannot claim novelty for "memory and computation cost energy." Its contribution must be a *predictive law* relating memory depth to environmental structure, resource value, and maintenance cost.

**3.2 Metacognition and metareasoning.** Computational metacognition monitors a system's own cognition to regulate reasoning; contemporary agent research explores self-assessment, capability-boundary estimation, delegation, and failure prediction, weighed against overhead. IF cannot claim novelty for self-monitoring, confidence estimation, cognitive control, or reasoning about reasoning. Its question is whether reflection emerges under a general **value-of-computation threshold** without a direct reflection reward.

**3.3 Damage, repair, and aging.** Aging is associated with accumulating molecular/cellular/systemic dysfunction, but no single damage variable explains every organism; evolutionary theories emphasize declining selection with age, late-acting deleterious effects, early–late trade-offs, and maintenance/reproduction allocation. Individual-based models show when repair, segregation, senescence, or replacement becomes advantageous under different ecological/spatial conditions, and results can reverse when assumptions change. IF cannot claim "aging is simply insufficient repair"; its contribution is a falsifiable multi-regime model distinguishing damage production, detectability, repairability, repair cost, repair side effects, replacement, reproduction, external hazard, and selection horizon.

**3.4 Reproduction–maintenance trade-offs.** Disposable-soma theory proposes finite resources create trade-offs among reproduction, growth, and somatic maintenance; later reviews stress energy allocation, nutrient signaling, genetic regulation, ecological context, and physiological constraints. IF treats the reproduction–repair trade-off as a hypothesis to test inside each artificial universe, not a universal assumption in the scoring function (avoiding `TELEOLOGY_INJECTION`).

**3.5 Computational aging and regeneration.** Agent and network models study trade-offs among accuracy, connection cost, degradation, regeneration, and lifespan, showing multiple Pareto-optimal maintenance strategies. IF cannot claim novelty for placing aging and computation in one simulation; the possible novelty is linking all four processes through the same operational-battery and causal-work accounting.

**3.6 Provisional novelty claim.** *One resource-accounted framework prospectively predicts selected memory depth, reflective effort, repair investment, and lifespan from independently measured environmental predictability, damage, hazard, and continuation value.* A stronger result would show dimensionless scaling across independent substrates. No novelty is established merely by giving established trade-offs IF terminology.

---

# 4. Unified Agent State

Let agent \(i\) at time \(t\) have state \(\mathcal A_i(t)=[B_i,M_i,\Theta_i,D_i,Q_i,P_i,H_i]\): operational battery \(B_i\), memory \(M_i\), predictive/self-model parameters \(\Theta_i\), accumulated damage \(D_i\), repair/quality-control machinery \(Q_i\), reproductive/successor state \(P_i\), historical identity/lineage record \(H_i\). Available power is allocated across

\[
u_A+u_M+u_R+u_Q+u_P+u_S=1
\]

(immediate action, memory acquisition/maintenance, reflection/metareasoning, repair/quality control, reproduction/successor construction, reserve storage). Every allocation has measurable consequences; the simulator may not provide separate unbounded budgets for cognition, repair, and reproduction (`PERPETUAL_RECHARGE` guardrail — no free budget).

---

# 5. Operational Battery Dynamics

With intake \(I_t\), baseline maintenance \(C_0\), and process costs \(C_M(L_t),C_R(R_t),C_Q(u_Q,D_t),C_P(u_P),C_A(a_t)\):

\[
B_{t+1}=B_t+I_t-C_0-C_M-C_R-C_Q-C_P-C_A-C_D(D_t),
\]

where \(C_D(D_t)\) is performance loss/leakage from accumulated damage. The agent ceases operation if \(B_t\le0\) or if damage exceeds a functional boundary.

---

# 6. Memory

**6.1 Memory complexity.** \(L\) may be stored time steps, internal states, predictive-state dimension, model parameters, description length, retained mutual information, or physical memory volume — no single measure assumed. The primary experiment uses controlled finite-state memory so exact capacity and cost are calculable.

**6.2 Memory benefit.** \(G_M(L)\) is additional gross work/continuation value from complexity \(L\) relative to a matched reactive controller; typically \(dG_M/dL\ge0\) initially with saturating \(d^2G_M/dL^2<0\). Memory that preserves irrelevant history adds little future benefit.

**6.3 Memory cost.** \(C_M(L)\) includes acquisition, writing, retention, retrieval, error correction, copying, reset, physical volume, and slowed decision time, with \(dC_M/dL>0\). Net value \(J_M(L)=G_M(L)-C_M(L)\); selected complexity \(L^*=\arg\max_L J_M(L)\).

**6.4 Analytical memory optimum.** With saturating benefit \(G_M(L)=G_{\max}(1-e^{-L/\ell})\) and linear cost \(C_M(L)=c_L L\), \(dJ_M/dL=(G_{\max}/\ell)e^{-L/\ell}-c_L\), giving the interior optimum

\[
\boxed{\,L^*=\ell\ln\!\left(\frac{G_{\max}}{c_L\ell}\right)\,}\quad\text{when } G_{\max}>c_L\ell,\qquad L^*=0 \text{ otherwise.}
\]

This predicts a threshold for the evolution of nonzero memory.

**6.5 Memory decay, forgetting, and nostalgia.** Forgetting may be beneficial. For stored item \(m_j\) with expected future value \(V_j(t)\) and maintenance cost \(c_j\), retain while \(V_j(t)>c_j\). When regimes change, old information may become irrelevant, misleading, or actively harmful. This connects directly to the break-even theorem's **nostalgia** term, \(\mathrm{nostalgia}=I_{\mathrm{mem}}-I_{\mathrm{pred}}\) — stored bits with no predictive power, the *self-deception* entry of the battery ledger and a pure thermodynamic liability (Still–Sivak–Crooks floor; `canon/00-foundations/04-break-even-theorem.md`). A reflective agent's first duty is deleting its own nostalgia (Founding Panel, Shannon). The strongest IF memory system evolves **selective forgetting** rather than indiscriminate retention — the same discipline as the repo's own LAI-6.

---

# 7. Reflection

**7.1 Operational definition.** Reflection is not verbal self-description. An agent reflects when it (1) represents aspects of its own model, confidence, policy, or limitation; (2) considers whether additional computation or information gathering is useful; (3) changes cognitive strategy because of that assessment; (4) produces a downstream change in action or outcome. Reflection is **control of cognition by an internal model of cognition**.

**7.2 Object level and meta level.** Object-level policy \(a_t=\pi_{\Theta_t}(o_t,M_t)\); the reflective controller chooses a cognitive operation \(r_t\in\{\text{act},\text{simulate},\text{retrieve},\text{inspect},\text{revise},\text{ask},\text{stop}\}\) via meta-policy \(r_t=\mu(\hat q_t,\hat c_t,\hat u_t,B_t,D_t)\) with estimated decision quality \(\hat q_t\), expected cognitive cost \(\hat c_t\), uncertainty/expected improvement \(\hat u_t\), remaining capacity \(B_t\), damage/reliability \(D_t\).

**7.3 Value of computation.** With immediate best action value \(V_{\mathrm{now}}\), expected value after operation \(r\) as \(\mathbb E[V_{\mathrm{after}}(r)]\), and physical/temporal cost \(C_{\mathrm{reflect}}(r)\):

\[
\operatorname{VOC}(r)=\mathbb E[V_{\mathrm{after}}(r)-V_{\mathrm{now}}]-C_{\mathrm{reflect}}(r).
\]

Reflection is rational under the model when \(\max_r\operatorname{VOC}(r)>0\); otherwise the agent acts without further reflection.

**7.4 Reflection threshold.** With decision stakes \(S\), uncertainty \(U\), error reduction \(\Delta e(R,U)\) from depth \(R\), and cost \(C_R(R)\), expected gain \(G_R(R)=S\Delta e(R,U)\), so reflection is beneficial when \(S\Delta e(R,U)>C_R(R)\). This predicts little reflection for low-stakes decisions; more when uncertainty and stakes are high; less when time is scarce, when the agent is damaged or energy-depleted, and past the point where improvement saturates.

**7.5 Reflection can be harmful.** Through delay, overfitting, indecision, repeated simulation, inaccurate self-assessment, memory contamination, excessive confidence correction, or missed action windows. The framework rejects "more reflection ⇒ more intelligence"; the expected relationship is often an inverted U, \(R_{\min}<R^*<R_{\max}\).

---

# 8. Self-Modeling

**8.1 Capability model.** With self-estimate \(\hat p_{\mathrm{success}}(x)\) and true \(p_{\mathrm{success}}(x)\), calibration error \(E_{\mathrm{cal}}=\mathbb E_x[(\hat p_{\mathrm{success}}(x)-p_{\mathrm{success}}(x))^2]\). A self-model has causal value only if ablating/scrambling it reduces performance in decisions such as whether to attempt, deliberate, seek help, delegate, repair, or reproduce.

**8.2 Damage awareness.** The agent may estimate its own damage \(\hat D_t\); repair policy \(u_Q(t)=\pi_Q(\hat D_t,B_t,\text{future value})\). A systematically wrong \(\hat D_t\) leads to under-repair, wasted resources on false alarms, dangerous continued operation, or reproduction of damaged organization — connecting reflection directly to maintenance.

---

# 9. Damage

**9.1 Damage state.** \(D_t\ge0\); a multidimensional model \(\mathbf D_t=[D_{\mathrm{struct}},D_{\mathrm{memory}},D_{\mathrm{controller}},D_{\mathrm{transport}},D_{\mathrm{replication}}]\) is preferable, the scalar used first for analytical clarity.
**9.2 Damage production.** Intrinsic \(\lambda_{\mathrm{int}}\), environmental \(\lambda_{\mathrm{ext}}(E_t)\), action-induced \(\lambda_{\mathrm{act}}(a_t)\); total \(\lambda_t=\lambda_{\mathrm{int}}+\lambda_{\mathrm{ext}}+\lambda_{\mathrm{act}}\).
**9.3 Repair.** With allocation \(u_Q\in[0,1]\) and efficiency \(\rho(D,Q,u_Q)\), \(D_{t+1}=D_t+\lambda_t-\rho(D_t,Q_t,u_Q)+\xi_t\) with stochastic \(\xi_t\); physical limits require \(0\le\rho\le D_t+\lambda_t\).
**9.4 Repair cost.** \(C_Q(u_Q,D_t)\) may include detection, diagnosis, replacement material, energy, downtime, verification, repair-induced errors, and redundant storage, with \(\partial C_Q/\partial u_Q>0\) and saturating benefit \(\partial^2\rho/\partial u_Q^2<0\).

---

# 10. The Repair Window

**10.1 Low-damage regime.** \(\lambda\approx0\): expensive repair machinery provides little benefit; selected strategy may be low repair capacity, passive robustness, minimal monitoring.
**10.2 Intermediate-damage regime.** \(\lambda_{\min}<\lambda<\lambda_{\max}\): repair can preserve enough future work and reproduction to repay its cost.
**10.3 Extreme-damage regime.** \(\lambda\gg\rho_{\max}\): repair cannot maintain bounded function; selection may favor rapid reproduction, redundancy, dormancy, escape, disposable structures, or lineage-level replacement.
**10.4 Repair-window hypothesis.** Expected investment \(u_Q^*(\lambda)=\arg\max_{u_Q}[V_{\mathrm{future}}(D(u_Q,\lambda))-C_Q(u_Q)]\); the primary prediction is **nonmonotonic** — weak at very low damage, strongest over an intermediate recoverable range, and declining when damage becomes economically or physically unrecoverable. Not universally guaranteed; some environments may produce monotonic investment.

---

# 11. Repair Versus Redundancy

An agent can protect function by preventing, detecting, repairing, storing redundant components, replacing components, or replicating the whole system. With redundancy allocation \(u_Z\) and cost \(C_Z(u_Z)\), the optimal maintenance strategy is \((u_Q^*,u_Z^*)=\arg\max[V_{\mathrm{future}}-C_Q(u_Q)-C_Z(u_Z)]\). Prediction: high repair efficiency favors repair; low detection reliability favors redundancy; catastrophic damage may favor distributed redundancy; cheap replacement may favor turnover over perfect preservation.

---

# 12. Error Detection and Repair Accessibility

Not all damage is detectable. Partition \(D_t=D_t^{\mathrm{visible}}+D_t^{\mathrm{hidden}}\); repair applies primarily to visible damage, \(\rho=\rho(D^{\mathrm{visible}},u_Q)\). Hidden damage may accumulate despite high repair investment — **repair capacity cannot correct damage the system cannot detect, localize, or represent**. A system may therefore increase reflection and diagnostic memory before increasing repair effort.

---

# 13. Repair Hysteresis

Two agents with equal current damage \(D_t\) may differ (gradual vs acute degradation, exhausted reserves, altered self-model, accumulated hidden errors). With repair state \(Q_t\), \(Q_{t+1}=Q_t+G_Q(u_Q)-\delta_Q Q_t-\omega_Q D_t\), the outcome depends on \((D_t,Q_t)\), not \(D_t\) alone — permitting repair fatigue, training of repair systems, irreversible thresholds, recovery debt, and path-dependent mortality.

---

# 14. Mortality

**14.1 Mortality is not one mechanism.** Finite lifespan may arise from stochastic damage, insufficient repair, hidden damage, catastrophic hazard, resource depletion, reproductive exhaustion, programmed termination, lineage-level replacement, competitive displacement, or loss of identity through component turnover — these must be distinguished.
**14.2 Functional death.** Permanent loss of the capacity to maintain a boundary, access resources, execute a control policy, restore itself, or continue its lineage; simple threshold \(D_t\ge D_{\mathrm{crit}}\), a more realistic criterion using multiple essential subsystems.
**14.3 Hazard.** With extrinsic hazard \(h_{\mathrm{ext}}(t)\) and intrinsic \(h_{\mathrm{int}}(D_t)\), total \(h(t)=h_{\mathrm{ext}}(t)+h_{\mathrm{int}}(D_t)\); survival \(S(t)=\exp[-\int_0^t h(s)\,ds]\).
**14.4 Continuation value.** With self-maintenance future value \(V_{\mathrm{self}}(t)\), reproduction value \(V_{\mathrm{offspring}}(t)\), and marginal maintenance cost \(C_{\mathrm{maint}}'(t)\), continued maintenance is favored while \(\Delta V_{\mathrm{self}}(t)>\Delta V_{\mathrm{offspring}}(t)+C_{\mathrm{maint}}'(t)\) under the physical evolutionary rules. This is not a moral valuation of lives — it is a model of resource allocation under lineage selection (layer firewall).

---

# 15. Emergent Mortality

A lifespan is **programmed** if the rule contains an age counter triggering death, a fixed maximum lifespan, a termination instruction, a direct reward for dying, or predetermined senescence. A lifespan is **emergent** when (1) no primitive age-death rule exists; (2) damage, repair, allocation, and reproduction follow local dynamics; (3) mortality arises from those dynamics; (4) lifespan changes predictably with environmental conditions; (5) identical genotypes/rule sets exhibit different lifespans under different resource and hazard conditions. A programmed age counter that silently terminates the system is the failure mode `programmed mortality` (§30.5).

**15.1 Mortality trade-off hypothesis.** Finite maintenance may evolve when repair has increasing marginal cost, extrinsic hazard limits expected future benefit, reproduction competes for the same resource, damage includes inaccessible components, or replacement is cheaper than indefinite repair. The claim is not "death is good" but *under finite resources, indefinite self-maintenance may cease to maximize lineage continuation or physical return.*

**15.2 Conditions favoring long life.** Low external hazard; valuable accumulated knowledge; expensive learning; costly reproduction; efficient repair; detectable damage; high mature-agent productivity; poor knowledge transfer to offspring.

**15.3 Conditions favoring rapid turnover.** High external hazard; cheap damage-avoidance through replacement; inefficient repair; rapidly changing environments; inexpensive offspring; obsolescent accumulated memory; lineage benefit from rapid generational turnover.

---

# 16. Memory and Mortality

Long-lived agents accumulate valuable models. With accumulated knowledge value \(K_{\mathrm{value}}(t)\) and age-related access impairment \(\eta_K(D_t)\), effective value \(K_{\mathrm{eff}}(t)=K_{\mathrm{value}}(t)\eta_K(D_t)\). An old agent may possess more information but use it less reliably, creating a three-way trade-off: preserve old agent vs repair old agent vs transfer knowledge to successor.

---

# 17. Reproduction as Memory Transfer

Reproduction need not transmit only structural parameters; a successor may inherit controller architecture, learned parameters, compressed environmental models, social records, external artifacts, or institutions. With transfer fraction \(\kappa\in[0,1]\) and copying cost \(C_{\mathrm{copy}}(\kappa)\), high-fidelity transfer may reduce turnover cost while poor transfer favors longer-lived individuals. Prediction: *selected lifespan decreases as low-cost, high-fidelity knowledge transfer improves, all else equal* — reversing if long-lived agents remain necessary to interpret or maintain shared knowledge.

---

# 18. Reflection and Repair

Reflection can allocate repair intelligently. A nonreflective agent follows fixed \(u_Q=u_0\); a reflective agent estimates damage severity, repair probability, future task value, remaining battery, and reproduction opportunity, choosing \(u_Q^*=\arg\max_{u_Q}\mathbb E[V_{\mathrm{future}}-C_Q(u_Q)]\). Reflection has repair value when \(G_{\mathrm{repair\ decision}}>C_{\mathrm{reflection}}\), especially when damage is heterogeneous, repair outcomes are uncertain, repair costs are nonlinear, or decisions are irreversible.

---

# 19. Repair and Reflection Can Fail Together

Damage to memory or self-modeling can impair repair decisions. With diagnostic accuracy \(q_D(D_t)\), \(dq_D/dD<0\), so repair effectiveness may fall as damage grows, \(\rho_{\mathrm{effective}}=q_D(D)\rho_{\max}(u_Q)\), and \(\dot D=\lambda-q_D(D)\rho_{\max}(u_Q)\). This positive feedback creates a tipping point: below it, repair maintains bounded damage; above it, damage impairs the very process required to repair damage.

---

# 20. Bounded and Runaway Damage

Consider \(\dot D=\lambda-\rho_{\max}\,u_Q/(K_D+D)\). A bounded stationary point satisfies \(\lambda=\rho_{\max}u_Q/(K_D+D^*)\), i.e. \(D^*=\rho_{\max}u_Q/\lambda-K_D\), which is physically meaningful only when \(\rho_{\max}u_Q>\lambda K_D\). Below that boundary damage runs away, giving three regimes: **bounded maintenance**, **slow drift**, **runaway deterioration**. A recent control-theoretic aging model likewise distinguishes bounded, drifting, and runaway regimes while cautioning that biological translation requires empirical identification of its variables.

---

# 21. Minimal Life-History Model

With intake \(I\), allocate \(u_Q\) to repair and \(u_P\) to reproduction, \(u_Q+u_P+u_A\le1\). Reproduction rate \(b(u_P,D)=b_{\max}u_P e^{-\alpha D}\); damage \(\dot D=\lambda-\rho u_Q\); intrinsic mortality \(h_{\mathrm{int}}(D)=h_0 e^{\beta D}\). Expected lineage output \(\mathcal R_0=\int_0^\infty S(t)b(t)\,dt\); selected allocation \(u_Q^*=\arg\max_{u_Q}\mathcal R_0\). This model can generate low repair and short lifespan, high repair and delayed reproduction, intermediate repair, non-aging bounded states, or catastrophic failure — the outcome depends on parameters rather than being assumed.

---

# 22. Reflection as an Allocation Decision

The reflective agent may allocate not only repair but cognition itself. At each step \(u_A+u_M+u_R+u_Q+u_P+u_S=1\), with meta-policy \(\mathbf u_t=\mu[B_t,D_t,\hat D_t,M_t,\text{environment},\text{future opportunities}]\). A successful IF agent learns when to remember, forget, think, act, repair, reproduce, and conserve resources. This is the first IF paper in which a provisional *self* becomes an allocation process rather than merely a bounded structure.

---

# 23. Core Hypotheses

**MRR-H1 — Finite-memory.** Under bounded predictability and increasing memory cost, selected complexity is finite, \(0\le L^*<\infty\). *Falsifier:* complexity grows without bound despite saturated gain and rising cost.

**MRR-H2 — Selective-forgetting.** Agents discard memories whose expected future causal value falls below maintenance and interference cost — deleting nostalgia. *Falsifier:* indiscriminate retention remains optimal across regime changes and nonzero storage costs.

**MRR-H3 — Reflection-threshold.** Reflection occurs when expected improvement exceeds cognitive and delay cost, \(\operatorname{VOC}>0\). *Falsifier:* reflective effort shows no relationship to stakes, uncertainty, time pressure, or cost.

**MRR-H4 — Finite-reflection.** Optimal reflection depth is finite and often nonmonotonic. *Falsifier:* more reflection always improves net return under increasing cost and bounded decision value.

**MRR-H5 — Self-model causal-value.** Scrambling capability/damage estimates selectively impairs delegation, stopping, repair, and reproduction decisions. *Falsifier:* the self-model predicts internal state but has no causal effect on policy or outcome.

**MRR-H6 — Repair-window.** Active repair is most strongly selected over an intermediate region where damage is consequential but recoverable. *Falsifier:* repair investment is universally monotonic or unrelated to recoverability and future value.

**MRR-H7 — Repair–redundancy substitution.** The selected balance shifts predictably with damage detectability, repair reliability, and replacement cost. *Falsifier:* strategy selection does not respond to these independently varied parameters.

**MRR-H8 — Damage-tipping.** When damage impairs detection/repair machinery, a threshold separates bounded maintenance from runaway deterioration. *Falsifier:* no tipping or nonlinear deterioration occurs even where the modeled feedback requires it.

**MRR-H9 — Emergent-mortality.** Finite lifespan can arise without an age-death rule from damage, imperfect repair, hazard, reproduction, and continuation value. *Falsifier:* finite lifespan occurs only because age or death is directly encoded.

**MRR-H10 — Hazard–maintenance.** Higher external hazard generally reduces selected long-term maintenance when that investment cannot repay before likely death. *Falsifier:* maintenance allocation remains invariant under large, otherwise matched hazard changes.

**MRR-H11 — Knowledge-longevity.** Expensive-to-acquire, poorly transferable knowledge favors longer individual maintenance. *Falsifier:* selected lifespan is unrelated to accumulated knowledge value and transfer fidelity.

**MRR-H12 — Cross-substrate.** Dimensionless return-on-maintenance variables predict memory, reflection, repair, and lifespan across independent substrates. *Falsifier:* every substrate requires unrelated laws and freely fitted thresholds.

---

# 24. Dimensionless Control Numbers

- **Memory-return** \(\Pi_M=G_M(L)/C_M(L)\); net beneficial when \(\Pi_M>1\).
- **Reflection-return** \(\Pi_R=\mathbb E[V_{\mathrm{after}}-V_{\mathrm{now}}]/C_{\mathrm{reflection}}\); net beneficial when \(\Pi_R>1\).
- **Repair-control** \(\Pi_Q=G_Q/C_Q\) (avoided future loss over repair cost); net beneficial when \(\Pi_Q>1\).
- **Damage-control ratio** \(\Gamma_D=\lambda/\rho_{\max}\); \(\Gamma_D<1\): maximum repair can exceed damage input; \(\Gamma_D>1\): damage exceeds maximal repair capacity.
- **Maintenance-horizon** \(\Gamma_H=\tau_H/\tau_Q\) (expected hazard-limited remaining lifetime over repair break-even time); long-horizon repair plausible when \(\Gamma_H>1\).
- **Knowledge-transfer** \(\Gamma_K=(K_{\mathrm{child}}/C_{\mathrm{copy}})/(K_{\mathrm{self}}/C_{\mathrm{maint}})\); compares knowledge continuity through succession against individual maintenance.

These are the P06 companions to the \(\Pi_A/\Pi_C\) ratios of Paper 5 — every one is a benefit-over-cost number with a break-even at 1, and every "benefit" names its ledger.

---

# 25. Experimental Program

1. **Memory-depth sweep** — vary capacity, cost, resource value, predictability; test \(L^*\).
2. **Useful vs irrelevant memory** — equal-capacity memories of predictive/irrelevant/scrambled/outdated/false content; Paper 2 ablation.
3. **Evolved forgetting** — mutate forgetting rates and retention policies; change regimes; is obsolete information selectively removed?
4. **Reflection under variable stakes** — matched uncertainty, different consequences; \(R^*\) rises with stakes until delay cost dominates.
5. **Reflection under time pressure** — constant stakes, shrinking action window; less deliberation, earlier stopping, more reactive reliance.
6. **Self-model calibration** — estimate competence, damage, confidence, resources; does better calibration improve allocation after modeling cost?
7. **Reflection ablation** — scramble the self-model, preserve object-level prediction; measure effects on stopping, delegation, repair, reproduction, timing.
8. **Damage-rate sweep** — vary \(\lambda\); repair allocation evolves; test low-repair / intermediate strong-repair / high-damage replacement or extinction regimes.
9. **Repairability sweep** — fixed damage rate, vary \(\rho_{\max}\); does repair disappear when \(\Gamma_D>1\)?
10. **Detection-limited repair** — hidden damage, varied diagnostic accuracy; do memory and reflection evolve before additional repair capacity?
11. **Repair vs redundancy** — invest in active repair, spares, distributed organization, or rapid replacement; map selected strategy.
12. **Repair fatigue** — repair machinery degrades with use; test hysteresis, recovery debt, runaway deterioration, rest periods.
13. **Emergent lifespan** — remove all age-triggered death; allow only damage, repair, allocation, reproduction, hazard; measure lifespan distributions.
14. **External hazard sweep** — vary hazard independent of internal damage; does lower horizon reduce long-term maintenance?
15. **Knowledge-value sweep** — increase time to learn a useful model; predict longer lifespan, more repair, more transfer.
16. **Knowledge-transfer sweep** — vary transfer fidelity/cost; test shifts among individual longevity, reproduction, institutional memory.
17. **Regime-change pressure** — make old knowledge obsolete at controlled rates; does rapid change favor forgetting, shorter life, faster reproduction, flexible self-models?
18. **Structural-agent integration** — embed memory/reflection/repair within Paper 3 structures; no abstract container beyond the detected structure.
19. **Expansion integration** — agents across Paper 4 regimes; do dilution, congestion, and turnover shift memory value, repair strategy, selected lifespan?
20. **Multi-agent care** — agents repair one another; does **agency-preserving cooperation** change individual maintenance investment, lifespan, division of labor, and lineage survival? (Input to the meaning-layer treatment of care; see §34.)

---

# 26. Phase Taxonomy

- **S0 — Stateless reactive:** no persistent memory or self-model.
- **S1 — Memory-bearing:** memory exists, reflection absent.
- **S2 — Reflective allocation:** conditional spending on additional cognition.
- **S3 — Stable-maintenance:** repair keeps damage statistically bounded.
- **S4 — Drifting-aging:** repair slows but does not halt long-term accumulation.
- **S5 — Runaway-damage:** damage impairs repair, accelerating decline.
- **S6 — Redundant-resilience:** function preserved mainly through spare capacity and distributed structure.
- **S7 — Replacement:** rapid reproduction/component replacement dominates repair.
- **S8 — Long-lived knowledge:** accumulated information makes individual preservation especially valuable.
- **S9 — Successor-memory:** knowledge continuity mainly through offspring, artifacts, institutions.
- **S10 — Social-maintenance:** agents preserve one another through cooperative repair and shared memory.

---

# 27. Deterministic Jupyter-Notebook Program

Each notebook carries the contract cell and seed 65537.

- **06A — Unified Resource Allocator:** \(u_A+u_M+u_R+u_Q+u_P+u_S=1\); validate exact budget closure.
- **06B — Memory Optimum:** reproduce \(L^*=\ell\ln(G_{\max}/c_L\ell)\); compare discrete and continuous solutions.
- **06C — Predictive and Obsolete Memory:** value of relevant/irrelevant/delayed/false/obsolete memories (nostalgia measured directly).
- **06D — Evolved Forgetting:** mutate retention/deletion policies; map forgetting vs environmental volatility.
- **06E — Reflection Value of Computation:** implement \(\operatorname{VOC}\); validate stopping against exact small decision trees.
- **06F — Reflection Stakes–Uncertainty Map:** sweep \(S\times U\times C_R\times\text{deadline}\); map selected depth.
- **06G — Self-Model Calibration:** track \(E_{\mathrm{cal}}\); self-model scrambling and policy-disconnection tests.
- **06H — Damage and Repair Dynamics:** deterministic and stochastic models; validate bounded/drifting/runaway regimes.
- **06I — Repair-Window Sweep:** \(\lambda\times\rho_{\max}\times C_Q\); test nonmonotonic investment.
- **06J — Detectability Limit:** visible/hidden partition; does diagnostic investment precede repair expansion?
- **06K — Repair Versus Redundancy:** Pareto fronts for repair cost, redundancy cost, performance, lifespan.
- **06L — Repair Hysteresis:** acute vs gradual histories ending at equal present damage.
- **06M — Life-History Optimization:** \(\mathcal R_0=\int_0^\infty S(t)b(t)\,dt\); analytical vs simulation optima.
- **06N — Emergent Mortality:** remove all lifespan timers; verify finite lifespans arise only from declared dynamics.
- **06O — Hazard–Maintenance Trade-off:** sweep external hazard; estimate \(\Gamma_H\).
- **06P — Knowledge and Longevity:** vary learning cost, value, transfer fidelity; measure lifespan and repair allocation.
- **06Q — Knowledge Transfer to Successors:** no transfer / genetic / copied learned state / external shared memory / institutional memory.
- **06R — Reflection–Repair Coupling:** does metacognitive allocation improve maintenance decisions after full cost accounting?
- **06S — Structural IF Agents:** transfer the models into Paper 3 resource-conserving organisms.
- **06T — Cross-Substrate Scaling:** do \(\Pi_M,\Pi_R,\Pi_Q,\Gamma_D,\Gamma_H\) organize behavior across independent implementations?
- **06U — Adversarial Audit:** a separate agent attempts to attribute results to lifespan timers, repair/reflection rewards, fitness leakage, hidden energy, arbitrary identity definitions, selected damage distributions, knowledge-transfer assumptions, or unreported failed strategies.

---

# 28. Reproducibility Record

Every run emits:

```yaml
experiment_id: if-memory-reflection-repair-mortality-06
paper_version: null
git_commit: null
environment_hash: null
implementation: null
random_seed: 65537
environment_predictability: null
environment_volatility: null
resource_input: null
external_hazard: null
damage_rate: null
damage_detectability: null
maximum_repair_rate: null
memory_capacity: null
memory_cost: null
retention_policy: null
predictive_information: null
obsolete_information: null
nostalgia: null
reflection_depth: null
reflection_cost: null
estimated_value_of_computation: null
realized_reflection_gain: null
self_model_calibration_error: null
repair_allocation: null
repair_cost: null
redundancy_allocation: null
damage_history: null
repair_history: null
reproduction_allocation: null
knowledge_transfer_fidelity: null
knowledge_transfer_cost: null
lifespan: null
cause_of_failure: null
lineage_output: null
final_operational_battery: null
memory_return_number: null
reflection_return_number: null
repair_return_number: null
damage_control_ratio: null
maintenance_horizon_number: null
energy_residual: null
resource_residual: null
invariant_failures: []
result_hash: null
```

---

# 29. Statistical Standards

**29.1 Lifespan is an outcome, not an independent sample generator** — time steps within one life are correlated; primary units are agent, lineage, genotype, environment, replicate.
**29.2 Competing risks** — report causes of termination separately (external hazard, energy depletion, structural damage, controller failure, reproductive exhaustion, competitive exclusion); a shorter lifespan from increased reproduction differs from one from coding failure.
**29.3 Censoring** — runs ending before agent death are right-censored, not assigned the final simulation time as lifespan.
**29.4 Held-out environments** — policies evolved under one damage/volatility distribution are evaluated on held-out distributions.
**29.5 Multiple strategy search** — many architectures explored → frozen strategies, metrics, parameter ranges, primary hypotheses.
**29.6 Pareto reporting** — where no strategy dominates, report Pareto fronts (work, reproduction, lifespan, repair, prediction, resilience, resource use) rather than inventing one composite fitness.

---

# 30. Failure Modes

Free memory (storage/retrieval with no physical cost); free reflection (unlimited simulation without energy or delay); reflection reward leakage; repair reward leakage; **programmed mortality** (an age counter silently terminates the system); damage chosen to force the window (distribution tuned after seeing results); identity by decree (a structure declared the same individual despite complete replacement, without an operational identity rule); reproduction counted as death (parent division labeled mortality even when identity continues); replacement counted as repair (whole system recreated externally, described as self-repair); hidden repair oracle (simulator reveals exact damage location); inaccessible damage ignored (perfect repair claimed because only detectable damage was modeled); longevity moralization (longer life described as inherently superior — a layer-firewall violation); aging overclaim (a toy damage variable presented as a complete biological theory); consciousness inflation (metacognitive policy described as phenomenal awareness).

---

# 31. What Would Count as Success?

- **Level 1 — Valid allocation model:** memory, reflection, repair, reproduction, action from one closed budget.
- **Level 2 — Finite memory and reflection optima:** independent analytical predictions match simulation.
- **Level 3 — Spontaneous repair allocation:** repair evolves without a repair reward.
- **Level 4 — Nontrivial repair window:** damage and recoverability prospectively predict the selected maintenance regime.
- **Level 5 — Emergent mortality:** finite lifespan without age-triggered death.
- **Level 6 — Knowledge–longevity relationship:** accumulated information and transfer fidelity predict maintenance and lifespan.
- **Level 7 — Cross-domain scaling:** dimensionless ratios predict allocation across held-out environment families.
- **Level 8 — Cross-substrate replication:** the same relationships in independent agent, cellular, graph, chemical models.
- **Level 9 — Biological or robotic validation:** the framework prospectively predicts measured maintenance decisions in real systems.

---

# 32. What Would Count as a Major Discovery?

A strong artificial-life result: *memory, reflection, repair, and lifespan arise from one closed physical budget without direct rewards for intelligence, self-preservation, or longevity.* A field-significant result: *transferable dimensionless return ratios predict how systems divide resources among remembering, thinking, repairing, reproducing, and remaining alive.* A deeper result: a general continuation inequality — *maintain the current agent while \(V_{\mathrm{self}}^{\mathrm{future}}-C_{\mathrm{maintenance}}>V_{\mathrm{successor}}^{\mathrm{future}}\)* — still requiring careful interpretation, describing physical and evolutionary allocation, not the moral worth of individuals.

---

# 33. Relationship to Functional Consciousness

Reflection and self-modeling are candidates for functional consciousness, but Paper 6 does not cross that boundary. A system may estimate uncertainty, monitor damage, allocate cognition, revise policy, and describe itself without subjective experience. Paper 6 establishes only a functional architecture, \(\text{self-model}\to\text{meta-policy}\to\text{changed cognition}\to\text{changed action}\). The next consciousness paper must compare this architecture with competing functional theories and make divergent predictions.

---

# 34. Relationship to Agency-Preserving Cooperation

Repair can be directed toward oneself, offspring, unrelated agents, shared infrastructure, or collective memory. Paper 6 studies self-maintenance first. The meaning-layer treatment (`canon/30-meaning/01-maxlove.md`) will ask when preserving the **agency of others** increases collective resilience, future action space, innovation, lineage continuity, and recovery from catastrophe. In the science layer the phenomenon is named **agency-preserving cooperation**, and its load-bearing formalization is the \(A_{\mathrm{future}}\) metric — cooperation measured by the action-space it preserves in others, \(A_{\mathrm{future}}=\sum\log|\text{viable actions}|\) (Founding Panel: "max love, formalized without circularity"). Cooperative care cannot be assumed beneficial — its transfer and opportunity costs must be included, exactly as every other allocation in this paper. The term "MaxLove" appears only in the meaning layer and the book; it never leaks into a science claim here (Founding Panel adjudication 3; `LAYER_COLLAPSE` guardrail).

---

# 35. Relationship to Meaning

An IF agent develops a primitive continuity problem: *which future organization should present resources preserve?* — current body, internal memory, policy, offspring, community, shared records, or broader future agency. Science can measure the consequences of these allocations; it cannot determine by itself which continuity should be valued morally. That determination is the meaning layer's, welded to the falsifiers below it, and is the reason the ladder was built.

---

# 36. Criteria for Rejection or Major Revision

Reject or substantially revise if: selected memory does not track predictive benefit and cost; reflection effort does not track expected decision improvement; self-model ablation has no selective causal effect; repair evolves only under direct repair rewards; the proposed repair window disappears under held-out damage models; lifespan remains entirely determined by programmed limits; knowledge value does not affect maintenance or succession; external hazard produces no predicted allocation response; dimensionless ratios fail across independent substrates; simpler life-history or metareasoning models explain all results with fewer assumptions; the framework cannot define identity through component turnover; or biological claims exceed what artificial simulations establish. Fired kills go in `SCOREBOARD.md` §Kill log the same session.

---

# 37. Conclusion

A physical agent must decide more than what to do next — it must decide, implicitly or explicitly, how much of itself to preserve. Memory preserves information; reflection regulates cognition; repair preserves organization; reproduction preserves lineage; mortality marks the failure, abandonment, or replacement of one continuation strategy. The IF framework proposes that none of these is free or universally optimal:

\[
L^*=\arg\max_L[G_M(L)-C_M(L)];\qquad \text{reflect when } \operatorname{VOC}>0;\qquad \text{repair when } \Pi_Q>1;
\]
\[
\text{bounded maintenance requires } \Gamma_D=\frac{\lambda}{\rho_{\max}}<1;\qquad \text{persist while } V_{\mathrm{self}}^{\mathrm{future}}-C_{\mathrm{maintenance}}>V_{\mathrm{successor}}^{\mathrm{future}}.
\]

These are hypotheses and model definitions, not established universal laws. The strongest conceptual claim: *a self is a costly continuity strategy — a physical system allocating finite capacity to preserve selected structure, information, policy, and future causal power through time.* If the simulations require explicit memory, reflection, repair, or death rewards, the proposed emergence fails. If one resource-accounted framework predicts all four across distinct substrates, IF Theory will have established a computational bridge from predictive agency to self-maintaining identity.

---

# References

1. Kirkwood, T. B. L. "Evolution of Ageing." *Nature* 270, 301–304 (1977).
2. Kirkwood, T. B. L. and Austad, S. N. "Why Do We Age?" *Nature* 408, 233–238 (2000).
3. Maklakov, A. A. and Chapman, T. "Evolution of Ageing as a Tangle of Trade-Offs: Energy versus Function." *Proceedings of the Royal Society B* 286, 20191604 (2019).
4. Mc Auley, M. T. "The Evolution of Ageing: Classic Theories and Emerging Ideas." *Biogerontology* (2025).
5. Attwell, D. and Laughlin, S. B. "An Energy Budget for Signaling in the Grey Matter of the Brain." *Journal of Cerebral Blood Flow & Metabolism* 21, 1133–1145 (2001).
6. Cox, M. T. et al. "Computational Metacognition." (2022).
7. Wang, C. and Shu, Y. "MetaCogAgent: A Metacognitive Multi-Agent LLM Framework with Self-Aware Task Delegation." (2026).
8. Krisko, A. and Radman, M. "Protein Damage, Ageing and Age-Related Diseases." *Open Biology* 9, 180249 (2019).
9. Ollé-Vila, A., Seoane, L. F. and Solé, R. "Aging, Computation, and the Evolution of Neural Regeneration Processes." (2019).
10. Ledberg, A. "Exponential Increase in Mortality with Age Is a Generic Property of a Simple Model System of Damage Accumulation and Death." (2020).
11. Barkman, T. "A Control-Theoretic Model of Damage Accumulation and Boundedness in Biological Aging." (2026).
12. Still, S., Sivak, D. A., Bell, A. J. and Crooks, G. E. "The Thermodynamics of Prediction." *Physical Review Letters* 109, 120604 (2012).
