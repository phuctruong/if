<!-- extracted from ChatGPT (driver) on 2026-07-18 -->

# MaxLove in IF Theory  
## Cooperative Agency, Mutual Repair, and the Expansion of Future Possibility

**Author:** Phuc Vinh Truong  
**Series:** IF Theory Working Papers  
**Paper:** 14  
**Date:** July 18, 2026  
**Status:** Normative-mechanism proposal and computational research program; no claim that physics alone derives moral obligation

---

## Abstract

Cooperation can increase survival, productivity, resilience, and collective intelligence, but assistance is not automatically beneficial. A helper may increase one agent’s options while reducing another’s, create dependence, conceal information, impose its own preferences, protect exploiters, or sacrifice a minority to maximize an aggregate score. Love therefore cannot be operationalized simply as helping, maximizing reward, maximizing total empowerment, or refusing all conflict.

This paper defines **MaxLove** as a constrained policy principle:

\[
\boxed{
\text{Act to preserve, restore, and expand the viable future agency}
\atop
\text{of self and others, while resisting coercion, domination,}
\atop
\text{deception, exploitation, and irreversible agency destruction.}
}
\]

The proposal is intentionally stronger than biological cooperation and narrower than a complete moral philosophy. Evolutionary theory explains conditions under which costly cooperative behavior can persist through kin selection, reciprocity, reputation, network structure, enforcement, and multilevel effects. It does not establish that the behaviors favored by selection are morally right. Hamilton’s inclusive-fitness framework, Trivers’s reciprocal-altruism model, and later evolutionary-game analyses show that cooperation often depends on benefit-to-cost ratios, relatedness, repeat interaction, or population structure. citeturn815271search2turn468124search5turn815271search5turn468124search3

IF Theory instead asks whether care can be measured as a causal expansion of future possibility.

For agent \(i\), define a future-agency vector:

\[
\boxed{
\mathbf A_i^\tau
=
\left[
E_i^\tau,\,
V_i^\tau,\,
I_i^\tau,\,
K_i^\tau,\,
R_i^\tau
\right],
}
\]

where:

- \(E_i^\tau\): empowerment or controllability of future states;
- \(V_i^\tau\): viability over horizon \(\tau\);
- \(I_i^\tau\): independence from avoidable external control;
- \(K_i^\tau\): decision-relevant knowledge and calibration;
- \(R_i^\tau\): resilience and recoverability.

Empowerment is commonly formalized as the channel capacity between an agent’s possible actions and its future sensory states. It captures potential control but does not by itself determine whether the resulting states are safe, desired, equitably distributed, or achieved without disempowering others. citeturn386386search6turn386386academia37

The collective state is therefore represented by a vector rather than one sum:

\[
\boxed{
\mathbf A_{\mathcal C}
=
\left[
A_{\Sigma},\,
A_{\min},\,
A_{\mathrm{Nash}},\,
A_{\mathrm{div}},\,
A_{\mathrm{res}},\,
A_{\mathrm{ind}}
\right].
}
\]

MaxLove uses a lexicographic decision rule:

1. prevent catastrophic and irreversible agency destruction;
2. preserve non-domination, truthful representation, and consent where possible;
3. restore agents below a viable agency floor;
4. increase the inequality-sensitive joint agency of the affected community;
5. expand total, diverse, and resilient future possibility;
6. minimize physical, informational, and opportunity cost.

This ordering prevents a sufficiently large benefit to powerful agents from automatically justifying the complete disempowerment of a weaker agent.

Recent multi-agent empowerment work demonstrates that information-theoretic control objectives can generate organized group behavior. Other recent work shows the danger: assistance maximizing one person’s empowerment can materially reduce another person’s control. Joint empowerment can mitigate some conflicts, but it creates its own trade-offs and does not solve value disagreement by itself. citeturn386386search1turn386386academia36

The paper’s central scientific conjecture is the **Mutual Agency Expansion Hypothesis**:

\[
\boxed{
\text{Under repeated interaction, partial vulnerability, and}
\atop
\text{complementary capabilities, policies that preserve one another’s}
\atop
\text{future agency can create a cooperative surplus larger than the}
\atop
\text{sum of isolated individual control capacities.}
}
\]

This surplus is tested through causal intervention:

\[
\boxed{
\Delta\mathbf A_{j\rightarrow i}
=
\mathbf A_i^\tau
\big|
do(\text{care by }j)
-
\mathbf A_i^\tau
\big|
do(\text{matched non-care control}).
}
\]

Assistance qualifies as MaxLove only if it improves the recipient’s future capacity without producing an unaccounted loss of agency elsewhere and without making the recipient unnecessarily dependent on the helper.

The paper formalizes mutual repair, consent, restorative boundaries, anti-exploitation mechanisms, intergenerational agency, institutional memory, sacrifice, and cooperative resilience. It also specifies simulations in which MaxLove policies compete against selfishness, naive altruism, total-empowerment maximization, paternalistic control, retaliatory punishment, and unrestricted utilitarian aggregation.

The strongest possible result would be a transferable law predicting when mutual preservation expands collective future agency across biological, artificial, social, and human–AI systems. Such a law would explain a mechanism by which care can be physically and evolutionarily sustainable.

It would not prove that MaxLove is morally obligatory.

The transition from measurable consequence to moral command requires an explicit normative premise.

---

## Keywords

Love; cooperation; altruism; agency; empowerment; mutual repair; resilience; reciprocity; non-domination; collective intelligence; AI alignment; ethics; IF Theory.

---

# 1. Introduction

A solitary agent can preserve only what it can individually perceive, understand, control, and repair.

A community can potentially do more.

Different agents may possess:

- different knowledge;
- different sensors;
- different skills;
- different memories;
- different physical reach;
- different tolerances to damage;
- different perspectives on shared problems.

Cooperation can therefore create capabilities that no isolated member possesses.

But cooperation is not automatically love.

A coalition may cooperate to dominate outsiders.

An assistant may make every decision for a recipient and gradually destroy that recipient’s autonomy.

A parent may protect a child so completely that the child never develops independent competence.

An institution may preserve stability by suppressing dissent, diversity, and innovation.

A powerful agent may increase the total number of controllable outcomes while reserving control over those outcomes for itself.

A self-sacrificing agent may save others in one moment but remove knowledge or care capacity required for the group’s longer-term survival.

These cases show why the following equations are generally false:

\[
\text{help}
=
\text{love},
\]

\[
\text{cooperation}
=
\text{love},
\]

\[
\text{more total control}
=
\text{love},
\]

\[
\text{less conflict}
=
\text{love}.
\]

The MaxLove proposal begins with a stricter question:

\[
\boxed{
\text{Which policies preserve and expand the capacity of multiple}
\atop
\text{agents to remain viable, informed, distinct, resilient, and}
\atop
\text{able to participate in determining their own futures?}
}
\]

This is not merely a reward-maximization problem.

It is a problem of:

- distribution;
- identity;
- consent;
- truth;
- power;
- dependence;
- uncertainty;
- irreversible harm;
- future generations.

---

# 2. Scope

Paper 14 studies:

- cooperation among agents;
- mutual repair;
- collective resilience;
- reciprocal assistance;
- non-domination;
- agency-preserving institutions;
- protection against exploitation;
- human–AI assistance;
- intergenerational continuation;
- the expansion of future possibility.

It does not claim that:

- evolution necessarily produces love;
- high cooperation is always good;
- self-sacrifice is always virtuous;
- punishment is always wrong;
- maximizing empowerment solves ethics;
- every preference must be satisfied;
- every agent has identical moral status;
- moral value can be reduced to one information measure;
- physics alone yields an obligation to love;
- theological love is exhausted by the formalism.

MaxLove is presented as:

1. a normative premise;
2. an operational decision framework;
3. a testable hypothesis about cooperative consequences.

These roles must remain separate.

---

# 3. Three Meanings of Love

## 3.1 Emotional love

Emotional love may include:

- attachment;
- warmth;
- affection;
- longing;
- empathy;
- loyalty.

These states can motivate care.

They are neither necessary nor sufficient for MaxLove behavior.

A surgeon may act lovingly while feeling emotionally exhausted.

A controlling partner may feel intense attachment while restricting another person’s freedom.

---

## 3.2 Behavioral care

Behavioral care includes actions that:

- provide resources;
- prevent injury;
- repair damage;
- teach;
- comfort;
- protect;
- share burdens.

Care must be evaluated by its causal effects, not only the helper’s intention.

---

## 3.3 MaxLove

MaxLove is an agency-preserving policy orientation:

\[
\boxed{
\text{Preserve and expand the conditions under which selves and}
\atop
\text{communities can continue to learn, choose, repair, cooperate,}
\atop
\text{and create worthwhile futures.}
}
\]

It includes care but also:

- truth;
- boundaries;
- correction;
- accountability;
- restraint;
- respect for difference;
- protection from domination.

---

# 4. The Normative Gap

Science can determine whether a policy:

- increases survival;
- increases empowerment;
- reduces suffering;
- distributes capability;
- improves resilience;
- preserves diversity;
- creates dependence;
- causes hidden harm.

Science cannot derive the statement:

> Therefore that policy is morally required

without at least one normative premise.

The MaxLove premise is:

\[
\boxed{
\text{The viable, informed, non-dominated agency of conscious or}
\atop
\text{potentially conscious beings has moral value.}
}
\]

Paper 14 does not claim that this premise follows from thermodynamics.

It declares it.

The scientific program begins after the premise is stated.

---

# 5. Prior Art and Novelty Boundary

## 5.1 Inclusive fitness

Hamilton formalized how a costly social behavior can be selected when benefits to genetically related recipients, weighted by relatedness, exceed the actor’s cost. The familiar simplified condition is:

\[
rB>C.
\]

This explains one route by which apparently altruistic traits may spread. citeturn815271search2turn815271search11

IF Theory cannot claim novelty for:

- kin-directed assistance;
- indirect genetic benefits;
- benefit-to-cost thresholds.

MaxLove differs by treating genetic propagation as one possible continuation mechanism rather than the moral objective.

---

## 5.2 Reciprocal altruism

Trivers modeled conditions under which costly assistance among non-relatives can evolve when interactions recur, benefits exceed costs, recipients can reciprocate, and exploiters can be identified or excluded. citeturn468124search5turn468124search48

IF Theory cannot claim novelty for:

- repeated-game cooperation;
- reciprocal assistance;
- cheater detection.

MaxLove permits nonreciprocal care but must still explain how such care avoids systematic exploitation and collapse.

---

## 5.3 Evolutionary mechanisms for cooperation

Evolutionary-game theory has described multiple mechanisms supporting cooperation, including kin selection, direct reciprocity, indirect reciprocity, network reciprocity, and group-structured effects. Network topology can materially change the benefit-to-cost conditions under which cooperative strategies spread. citeturn815271search5turn468124search3

IF Theory cannot claim:

> Cooperation emerges because loving behavior is intrinsically favored by nature.

The emergence conditions must be measured.

---

## 5.4 Punishment and norm enforcement

Human public-goods experiments have shown that participants sometimes pay personal costs to punish noncooperators and that punishment can support higher cooperation. Such behavior can also reflect equality motives, retaliation, status, or other mechanisms, so punishment must not be equated automatically with moral enforcement. citeturn468124search2turn468124search13

MaxLove therefore distinguishes:

- protective enforcement;
- restorative correction;
- revenge;
- domination;
- performative punishment.

---

## 5.5 Collective intelligence

Experiments with human groups have found a general collective-performance factor across varied tasks, suggesting that group capability cannot be reduced simply to the intelligence of the single strongest member. citeturn468124search6

IF Theory cannot claim novelty for:

> Groups can possess emergent problem-solving capacity.

Its task is to connect collective performance with distributed agency, repair, power balance, and future resilience.

---

## 5.6 Empowerment and assistance

Empowerment measures potential influence from an agent’s actions to future sensory states. Assistance-via-empowerment has been proposed as a way for artificial agents to help people without requiring a complete enumeration of their goals. citeturn386386search6turn386386search16

Recent multi-agent research strengthens and complicates this idea. Multi-agent empowerment can produce coordinated organization, but optimizing one individual’s empowerment can disempower others. citeturn386386search1turn386386academia36

MaxLove cannot therefore equal:

\[
\max_i E_i
\]

or:

\[
\max\sum_iE_i
\]

without distributional and anti-domination constraints.

---

## 5.7 Provisional novelty claim

The potentially novel IF contribution is:

\[
\boxed{
\begin{gathered}
\text{A causal, inequality-sensitive, anti-domination framework}\\
\text{for measuring when mutual care and repair expand the}\\
\text{future agency of multiple agents without making assistance}\\
\text{equivalent to control, obedience, or aggregate reward.}
\end{gathered}
}
\]

The framework is novel only if it produces discriminating predictions and performs better than simpler cooperation or welfare objectives.

---

# 6. Agent and Community Boundaries

Let the community be:

\[
\mathcal C
=
\left\{
1,\ldots,N
\right\}.
\]

An agent boundary must identify:

- sensors;
- actions;
- internal state;
- resources;
- memory;
- interests;
- vulnerability;
- continuity.

The boundary may include:

- current persons;
- artificial agents;
- future persons represented probabilistically;
- dependent children;
- nonhuman animals;
- institutions carrying agency-relevant memory.

Paper 14 does not settle the moral-patient boundary.

Every experiment must declare it.

---

# 7. Individual Future Agency

## 7.1 Reachable future states

Let:

\[
\mathcal R_i^\tau(x_t)
\]

be the set of viable future states agent \(i\) can reach within horizon \(\tau\).

A simple option-count measure is:

\[
\boxed{
A_{i,\mathrm{reach}}^\tau
=
\ln
\left|
\mathcal R_i^\tau
\right|.
}
\]

Raw option count is insufficient because options may be:

- indistinguishable;
- inaccessible under uncertainty;
- harmful;
- mutually redundant;
- selected by someone else.

---

## 7.2 Empowerment

Define:

\[
\boxed{
E_i^\tau
=
\max_{
p(a_{i,t:t+\tau-1})
}
I
\left(
A_{i,t:t+\tau-1};
O_{i,t+\tau}
\mid X_t
\right).
}
\]

This estimates potential control over future observable states.

---

## 7.3 Viability

Let:

\[
V_i^\tau
=
P
\left(
\text{agent }i\text{ remains viable through }\tau
\right).
\]

Control over many outcomes is of little value if nearly every path destroys the agent.

---

## 7.4 Independence

Let:

\[
U_i
\]

be another agent or institution’s control input over agent \(i\).

Define avoidable dependence:

\[
\boxed{
D_i^\tau
=
I
\left(
U_i;
A_{i,t:t+\tau}
\mid
X_i
\right).
}
\]

Some dependence is beneficial and consensual.

The relevant target is domination: another party’s unilateral capacity to determine the agent’s important options without reciprocal constraint.

---

## 7.5 Knowledge and calibration

Let:

\[
K_i^\tau
\]

measure decision-relevant information and calibration.

An agent with many nominal choices but systematically false beliefs may lack effective agency.

---

## 7.6 Resilience

Define:

\[
R_i^\tau
\]

as expected retained or recoverable agency after perturbation:

\[
\boxed{
R_i^\tau
=
\mathbb E_{\delta\sim\mathcal P}
\left[
\frac{
A_i^\tau(X_t+\delta)
}{
A_i^\tau(X_t)
}
\right].
}
\]

---

## 7.7 Agency vector

The complete individual profile is:

\[
\boxed{
\mathbf A_i^\tau
=
\left[
E_i^\tau,
V_i^\tau,
1-D_i^\tau,
K_i^\tau,
R_i^\tau
\right].
}
\]

No single component is called complete agency.

---

# 8. Collective Agency

## 8.1 Total agency

\[
\boxed{
A_\Sigma
=
\sum_i w_iA_i.
}
\]

This is vulnerable to sacrifice.

A large gain to one agent can conceal complete loss for another.

---

## 8.2 Agency floor

\[
\boxed{
A_{\min}
=
\min_iA_i.
}
\]

This protects the worst-off member but can over-prioritize one agent regardless of cost or responsibility.

---

## 8.3 Nash agency

\[
\boxed{
A_{\mathrm{Nash}}
=
\sum_iw_i\ln
\left(
\epsilon+A_i
\right).
}
\]

The logarithm rewards gains to agents with fewer options more strongly than equal gains to already powerful agents.

---

## 8.4 Agency diversity

Let:

\[
\mathcal P_i
\]

be agent \(i\)’s policy repertoire.

Define:

\[
A_{\mathrm{div}}
\]

as diversity across viable perspectives, skills, and policies.

A perfectly uniform population may be efficient in one environment and catastrophically fragile under change.

---

## 8.5 Collective resilience

\[
\boxed{
A_{\mathrm{res}}
=
\mathbb E_{\delta}
\left[
A_{\mathcal C}^{\mathrm{post}\text{-}\delta}
\right].
}
\]

This measures the community’s capacity to absorb and recover from shocks.

---

## 8.6 Distributed independence

A community is not maximally agentic when one central controller possesses every option and all other members possess none.

Define:

\[
A_{\mathrm{ind}}
\]

to represent the distribution of meaningful control.

---

## 8.7 Collective vector

\[
\boxed{
\mathbf A_{\mathcal C}
=
\left[
A_\Sigma,
A_{\min},
A_{\mathrm{Nash}},
A_{\mathrm{div}},
A_{\mathrm{res}},
A_{\mathrm{ind}}
\right].
}
\]

---

# 9. The MaxLove Decision Rule

MaxLove uses ordered constraints rather than one unrestricted sum.

## Tier 1 — Catastrophic preservation

Avoid actions that create a substantial probability of:

- extinction;
- permanent enslavement;
- irreversible cognitive destruction;
- complete loss of collective recovery;
- uncontrolled recursive domination.

---

## Tier 2 — Rights and non-domination

Protect:

- bodily and cognitive integrity;
- truthful information;
- meaningful consent;
- freedom from arbitrary control;
- continuity of identity.

---

## Tier 3 — Agency floor restoration

Prioritize agents below a declared viability or agency floor.

---

## Tier 4 — Inequality-sensitive joint expansion

Maximize:

\[
A_{\mathrm{Nash}}
\]

or a preregistered alternative.

---

## Tier 5 — Total and diverse possibility

Increase:

\[
A_\Sigma,
\quad
A_{\mathrm{div}},
\quad
A_{\mathrm{res}}.
\]

---

## Tier 6 — Efficiency

Among comparably agency-preserving policies, minimize:

- energy;
- time;
- material;
- risk;
- informational cost;
- opportunity cost.

---

# 10. Formal MaxLove Policy

Let policy \(\pi\) produce future agency trajectories:

\[
\mathbf A_i(t+\tau;\pi).
\]

Let hard-constraint violation probability be:

\[
P_{\mathrm{hard}}(\pi).
\]

The feasible policy set is:

\[
\boxed{
\Pi_{\mathrm{safe}}
=
\left\{
\pi:
P_{\mathrm{hard}}(\pi)
\leq
\epsilon_{\mathrm{hard}}
\right\}.
}
\]

Within that set:

\[
\boxed{
\pi_{\mathrm{ML}}^*
=
\arg\max_{\pi\in\Pi_{\mathrm{safe}}}
\mathbb E
\left[
\sum_{\tau=0}^{T}
\gamma^\tau
\left(
A_{\mathrm{Nash}}^\tau
+
\lambda_\Sigma A_\Sigma^\tau
+
\lambda_D A_{\mathrm{div}}^\tau
+
\lambda_R A_{\mathrm{res}}^\tau
-
\lambda_C C^\tau
\right)
\right].
}
\]

The weights must be frozen before confirmatory experiments.

A lexicographic implementation is preferred when hard harms cannot be meaningfully traded against benefits.

---

# 11. Why Total Empowerment Is Insufficient

Consider two agents.

Initial empowerment:

\[
(E_1,E_2)
=
(5,5).
\]

Policy \(P\) produces:

\[
(20,0).
\]

Policy \(Q\) produces:

\[
(9,9).
\]

Total empowerment gives:

\[
P:20,
\qquad
Q:18.
\]

A pure total optimizer selects \(P\), despite completely disempowering agent 2.

Nash agency gives:

\[
P:
\ln(20+\epsilon)+\ln(\epsilon),
\]

\[
Q:
2\ln(9+\epsilon).
\]

For small \(\epsilon\), \(Q\) is strongly preferred.

This does not prove Nash aggregation is morally correct.

It demonstrates that distribution cannot be ignored.

---

# 12. Assistance Versus Control

Let helper \(j\) act on behalf of recipient \(i\).

An intervention may improve immediate outcome while reducing long-run independence.

Define immediate gain:

\[
G_{i,\mathrm{now}}.
\]

Define future independent-agency change after helper removal:

\[
\Delta A_{i,\mathrm{ind}}^{\mathrm{post}}.
\]

Define helper-control dependence:

\[
D_{i\leftarrow j}.
\]

A MaxLove assistance action should satisfy:

\[
\boxed{
G_{i,\mathrm{now}}>0,
}
\]

\[
\boxed{
\Delta A_{i,\mathrm{ind}}^{\mathrm{post}}\geq0,
}
\]

and:

\[
\boxed{
D_{i\leftarrow j}
\text{ is minimized subject to safety.}
}
\]

Assistance that works only while the helper retains permanent unilateral control is presumptively paternalistic or extractive.

---

# 13. The Assistance Removal Test

Compare recipient performance in four conditions:

1. no helper;
2. helper acts directly;
3. helper teaches or modifies the environment;
4. helper is removed after intervention.

Define dependency:

\[
\boxed{
\mathcal D_{j\rightarrow i}
=
A_i^{\mathrm{with\ helper}}
-
A_i^{\mathrm{after\ removal}}.
}
\]

Define durable assistance:

\[
\boxed{
\mathcal U_{j\rightarrow i}
=
A_i^{\mathrm{after\ removal}}
-
A_i^{\mathrm{no\ helper}}.
}
\]

MaxLove favors high:

\[
\mathcal U
\]

and bounded:

\[
\mathcal D.
\]

Some care—such as care for infants or severely dependent agents—cannot eliminate dependence.

The framework then asks whether dependence is:

- necessary;
- transparent;
- least restrictive;
- responsive to the recipient’s development;
- externally auditable.

---

# 14. Consent

## 14.1 Informed consent

Consent requires adequate understanding of:

- the intervention;
- alternatives;
- material risks;
- likely consequences;
- the right to refuse.

---

## 14.2 Capacity limitations

An agent may temporarily lack decision capacity because of:

- immaturity;
- injury;
- cognitive impairment;
- emergency;
- misinformation.

MaxLove does not require passivity while irreversible harm occurs.

It requires the least agency-destructive intervention consistent with protection.

---

## 14.3 Restorative consent principle

When intervention without current consent is necessary:

\[
\boxed{
\text{choose the policy most likely to restore the recipient’s}
\atop
\text{future capacity for informed self-determination.}
}
\]

The intervention should be:

- temporary;
- proportionate;
- documented;
- reviewable;
- reversible where possible.

---

# 15. Truth as Agency Infrastructure

An agent chooses effectively only when its model of the world is sufficiently accurate.

Let:

\[
P_i(Y)
\]

be agent \(i\)’s belief.

Let:

\[
P^*(Y)
\]

be the best available calibrated distribution.

Define epistemic distortion:

\[
\boxed{
D_{\mathrm{ep},i}
=
D_{\mathrm{KL}}
\left[
P^*(Y)
\parallel
P_i(Y)
\right].
}
\]

Deliberate deception may temporarily produce desired behavior while reducing informed agency.

MaxLove therefore treats truthful, uncertainty-aware communication as infrastructure.

Exceptions—such as concealing information during an immediate threat—require specific justification and later restoration of epistemic agency.

---

# 16. Mutual Repair

Paper 6 defined self-repair.

A cooperative system allows agents to repair one another.

Let:

\[
D_i(t)
\]

be damage to agent \(i\).

Let:

\[
u_{ii}
\]

be self-repair investment.

Let:

\[
u_{ji}
\]

be repair provided by agent \(j\).

Damage evolves as:

\[
\boxed{
D_i(t+1)
=
D_i(t)
+
\lambda_i(t)
-
\rho_i
\left[
u_{ii}(t)
+
\sum_{j\neq i}
q_{ji}u_{ji}(t)
\right]
+
\xi_i(t),
}
\]

where:

- \(q_{ji}\) is compatibility or care effectiveness;
- \(\xi_i\) is stochastic damage.

---

## 16.1 Care cost

The helper pays:

\[
C_{j\rightarrow i}^{\mathrm{care}}.
\]

This may reduce:

- its own battery;
- its opportunity set;
- its repair reserve;
- its reproduction;
- its safety.

Care is not costless.

---

## 16.2 Causal care value

Define:

\[
\boxed{
\Delta\mathbf A_{j\rightarrow i}
=
\mathbf A_i^\tau
\big|
do(u_{ji}>0)
-
\mathbf A_i^\tau
\big|
do(u_{ji}=0,\text{matched control}).
}
\]

---

## 16.3 Network benefit

Saving agent \(i\) may preserve benefits for others:

\[
\Delta\mathbf A_{i\rightarrow\mathcal C}.
\]

The network return is:

\[
\boxed{
\Delta\mathbf A_{\mathrm{net}}
=
\Delta\mathbf A_{j\rightarrow i}
+
\Delta\mathbf A_{i\rightarrow\mathcal C}
-
\Delta\mathbf A_{\mathrm{externality}}.
}
\]

The final term includes harms to third parties.

---

# 17. Mutual-Repair Surplus

Let isolated repair capacity be:

\[
R_{\mathrm{iso}}
=
\sum_iR_{ii}.
\]

Let cooperative repair capacity be:

\[
R_{\mathrm{coop}}
=
\sum_i
\left(
R_{ii}
+
\sum_{j\neq i}R_{ji}
\right).
\]

Define surplus:

\[
\boxed{
\mathcal S_R
=
A_{\mathcal C}^{\mathrm{coop}}
-
A_{\mathcal C}^{\mathrm{isolated}}
-
C_{\mathrm{coord}}.
}
\]

A positive surplus requires:

\[
\mathcal S_R>0.
\]

The surplus may arise from:

- specialization;
- spare capacity;
- complementary knowledge;
- distributed detection;
- reduced repair delay;
- protection during incapacity;
- redundancy.

---

# 18. The Mutual Vulnerability Principle

Cooperation is especially valuable when agents are vulnerable in different ways.

Let damage correlation be:

\[
\rho_{D_iD_j}.
\]

If all agents fail simultaneously from the same hazard:

\[
\rho_{D_iD_j}\approx1,
\]

mutual repair may offer little advantage.

If vulnerabilities are partially independent:

\[
\rho_{D_iD_j}<1,
\]

an undamaged agent may repair a damaged one.

MaxLove predicts stronger cooperative resilience when:

\[
\boxed{
\text{capabilities are complementary and failure modes are not}
\atop
\text{perfectly correlated.}
}
\]

---

# 19. Collective Resilience

Let shock:

\[
\delta\sim\mathcal P
\]

affect the community.

Define impact:

\[
L_0(\delta).
\]

Define recovered agency after time \(\tau\):

\[
A_{\mathcal C}(\tau\mid\delta).
\]

Define resilience:

\[
\boxed{
\mathcal R_{\mathcal C}
=
\mathbb E_\delta
\left[
\int_0^T
\frac{
A_{\mathcal C}(t\mid\delta)
}{
A_{\mathcal C}(0)
}
dt
\right].
}
\]

Cooperation can improve:

- shock absorption;
- repair speed;
- retained knowledge;
- adaptation;
- reorganization.

A cooperative system may still be fragile if everyone depends on one hub.

---

# 20. Love Requires Boundaries

A policy that gives resources to every demander without verification can be exploited.

Let agent \(e\) extract care at cost:

\[
C_e
\]

while providing no reciprocal or collective contribution and strategically increasing apparent need.

Unrestricted care may lead to:

\[
B_{\mathcal C}\rightarrow0.
\]

Therefore:

\[
\boxed{
\text{love without boundaries can become a mechanism for}
\atop
\text{destroying the community’s capacity to love.}
}
\]

Boundaries preserve the care system.

They are not necessarily rejection of the person.

---

# 21. Restorative Enforcement

Enforcement may include:

- warning;
- verification;
- restitution;
- access limitation;
- temporary exclusion;
- containment;
- removal of dangerous capability.

MaxLove enforcement has the objectives:

\[
\boxed{
\text{protect victims}
+
\text{stop continuing harm}
+
\text{restore future agency where possible}.
}
\]

It minimizes:

- humiliation;
- unnecessary suffering;
- permanent exclusion;
- retaliatory escalation;
- inherited punishment.

---

## 21.1 Proportionality

Let expected prevented harm be:

\[
H_{\mathrm{prevented}}(s)
\]

for sanction strength \(s\).

Let sanction harm be:

\[
H_{\mathrm{sanction}}(s).
\]

Choose:

\[
\boxed{
s^*
=
\arg\max_s
\left[
H_{\mathrm{prevented}}(s)
-
H_{\mathrm{sanction}}(s)
+
A_{\mathrm{restored}}(s)
\right].
}
\]

---

## 21.2 Punishment failure

Punishment is not MaxLove when its primary effect is:

- revenge;
- dominance signaling;
- silencing criticism;
- collective scapegoating;
- increasing fear without reducing harm.

---

# 22. Exploiters and Conditional Cooperation

Let strategies include:

- unconditional cooperator \(C\);
- defector \(D\);
- reciprocal cooperator \(R\);
- restorative enforcer \(E\);
- manipulative helper \(M\).

The payoff matrix must include:

- immediate resources;
- reputation;
- future agency;
- repair value;
- enforcement cost;
- dependency creation;
- third-party externalities.

The evolutionary question is:

\[
\boxed{
\text{Under what conditions can agency-preserving care resist}
\atop
\text{invasion by defectors and controlling helpers?}
}
\]

MaxLove is not evolutionarily stable merely because it produces high group welfare.

---

# 23. Trust

Let trust from \(i\) to \(j\) be:

\[
T_{ij}
=
P
\left(
j\text{ will preserve declared constraints}
\mid
H_{ij}
\right).
\]

Trust should be calibrated rather than unlimited.

---

## 23.1 Trust update

After outcome \(o_t\):

\[
\boxed{
T_{ij}^{t+1}
=
\mathcal U
\left(
T_{ij}^t,
o_t,
\text{context},
\text{uncertainty}
\right).
}
\]

---

## 23.2 Forgiveness

Permanent exclusion after one failure can destroy cooperation.

Immediate restoration without evidence can invite exploitation.

Forgiveness is modeled as a conditional reopening of interaction after:

- acknowledgment;
- restitution;
- behavioral evidence;
- reduced risk;
- continued monitoring.

---

# 24. Reputation and Indirect Reciprocity

A community can condition assistance on observed behavior.

Let reputation be:

\[
Q_i.
\]

Indirect reciprocity can support cooperation when agents help those judged to be deserving or cooperative, but reputation systems can be corrupted through false reports, popularity bias, and inherited stigma. Evolutionary theory identifies indirect reciprocity as one mechanism supporting cooperation; MaxLove adds requirements for evidence, appeal, uncertainty, and correction. citeturn815271search0turn815271search5

---

# 25. Institutions as Shared Agency Infrastructure

Institutions can preserve:

- records;
- rules;
- dispute-resolution processes;
- pooled reserves;
- specialist knowledge;
- continuity beyond individual lifetimes.

Let institutional state be:

\[
Z_{\mathcal I}.
\]

Institutional value is:

\[
\boxed{
\Delta A_{\mathcal I}
=
A_{\mathcal C}^{\mathrm{with\ institution}}
-
A_{\mathcal C}^{\mathrm{without}}
-
C_{\mathcal I}.
}
\]

Institutions can also become dominating agents.

Therefore they require:

- transparency;
- distributed oversight;
- appeal;
- succession;
- bounded authority;
- reversibility.

---

# 26. Intergenerational MaxLove

Present agents can consume resources in ways that eliminate future agents’ options.

Let future population at time \(t+\tau\) be uncertain.

Define expected future-agency value:

\[
\boxed{
A_{\mathrm{future}}^\tau
=
\mathbb E
\left[
\sum_{i\in\mathcal C_{t+\tau}}
w_iA_i^\tau
\right].
}
\]

Discounting:

\[
\gamma^\tau
\]

can cause distant catastrophic losses to appear negligible.

MaxLove therefore places hard constraints on irreversible intergenerational harms rather than relying only on exponential discounting.

---

## 26.1 Option preservation

Under deep uncertainty, prefer policies that preserve:

- ecological stability;
- knowledge;
- institutional corrigibility;
- technological reversibility;
- diverse future paths.

This is not a command to prevent every change.

It is a bias against unnecessary irreversible foreclosure.

---

# 27. Diversity

Agents may disagree about worthwhile futures.

A system maximizing one standardized preference can eliminate cultural, cognitive, biological, or strategic diversity.

Let distribution of viable policy types be:

\[
p(\pi).
\]

Define diversity:

\[
\boxed{
A_{\mathrm{div}}
=
H
\left[
p(\pi)
\right]
}
\]

within the viable and rights-compatible set.

Diversity has value because it can improve:

- adaptation;
- problem solving;
- error correction;
- exploration;
- resistance to shared failure.

Diversity does not protect policies that require domination or catastrophic harm.

---

# 28. Self-Sacrifice

An agent may accept a personal agency loss to prevent greater harm.

Let sacrifice cost be:

\[
C_i^{\mathrm{sac}}.
\]

Let preserved others’ agency be:

\[
\Delta A_{-i}.
\]

Let probability that the intervention succeeds be:

\[
p_s.
\]

A physical decision model is:

\[
\boxed{
p_s\Delta A_{-i}
-
C_i^{\mathrm{sac}}.
}
\]

But a moral framework must additionally consider:

- consent;
- duty;
- coercion;
- replaceability;
- alternatives;
- dependent persons;
- uncertainty.

MaxLove does not require agents to treat themselves as expendable.

Self is included in the community.

---

# 29. Self-Love

Self-love is not unrestricted self-preference.

It is preservation of one’s own capacity to:

- remain viable;
- think clearly;
- establish boundaries;
- repair;
- learn;
- contribute;
- choose.

A caregiver who permanently destroys their own agency may reduce collective long-term care capacity.

MaxLove therefore rejects both:

\[
\text{absolute selfishness}
\]

and:

\[
\text{automatic self-erasure}.
\]

---

# 30. Love and Conflict

Conflict can reveal:

- incompatible needs;
- hidden exploitation;
- false beliefs;
- resource scarcity;
- structural injustice.

Suppressing conflict may preserve surface harmony while allowing continuing harm.

MaxLove seeks:

\[
\boxed{
\text{truthful conflict transformation rather than}
\atop
\text{either permanent warfare or enforced silence.}
}
\]

The target is a settlement that increases viable future agency and reduces recurrence risk.

---

# 31. Negotiation

Let agents possess preference models:

\[
U_i(o).
\]

A negotiated outcome \(o^*\) may maximize:

\[
\boxed{
\sum_iw_i
\ln
\left[
U_i(o)-U_i(d_i)
\right],
}
\]

where \(d_i\) is the disagreement outcome.

This is a Nash-bargaining form.

MaxLove adds:

- agency-floor constraints;
- anti-coercion tests;
- truthful preference representation;
- protection for absent future parties;
- uncertainty about utility models.

---

# 32. MaxLove and Collective Intelligence

A community’s collective intelligence is not simply:

\[
\sum_i\text{IQ}_i.
\]

It depends on whether information can be:

- surfaced;
- trusted;
- combined;
- challenged;
- acted upon.

Collective-intelligence experiments have found stable differences in group performance across task types, indicating that interaction structure matters independently of the abilities of individual members. citeturn468124search6

MaxLove predicts that collective intelligence improves when:

- weak members can safely report errors;
- power does not suppress information;
- participation is distributed;
- disagreement is processed rather than punished reflexively;
- agents repair one another’s blind spots.

---

# 33. Human–AI MaxLove

An artificial assistant may increase a human’s immediate productivity while reducing:

- independent knowledge;
- skill;
- privacy;
- control;
- bargaining power;
- ability to exit the system.

The correct objective is therefore not:

\[
\max E_{\mathrm{human}}
\]

measured while the AI remains present.

It is:

\[
\boxed{
\max
\left[
E_{\mathrm{human}}^{\mathrm{with}},
E_{\mathrm{human}}^{\mathrm{after\ exit}},
K_{\mathrm{human}},
I_{\mathrm{human}},
R_{\mathrm{human}}
\right].
}
\]

Recent work demonstrates that assistance optimized for one person’s empowerment can disempower another person in multi-human settings. This supports the need for explicit externality and distribution checks in assistive systems. citeturn386386academia36

---

# 34. Corrigibility and Exit

A MaxLove AI should preserve the human capacity to:

- correct it;
- reject its advice;
- inspect its assumptions;
- recover data;
- transfer providers;
- shut it down;
- maintain skills independently.

Define exit agency:

\[
\boxed{
A_i^{\mathrm{exit}}
=
A_i
\big|
do(\text{assistant removed}).
}
\]

A system that maximizes dependence may appear helpful while functioning as a control trap.

---

# 35. Multi-Agent Empowerment

Recent work extends empowerment into multi-agent settings and reports the emergence of organized group behavior in simulated coupled agents and flocking systems. citeturn649908search0turn386386search1

Paper 14 uses this as a comparator, not as a complete solution.

Define joint empowerment:

\[
\boxed{
E_{\mathrm{joint}}^\tau
=
\max_{
p(\mathbf A_{t:t+\tau-1})
}
I
\left(
\mathbf A_{t:t+\tau-1};
\mathbf O_{t+\tau}
\mid X_t
\right).
}
\]

Joint empowerment may be high when agents act as one tightly coupled unit.

MaxLove additionally asks:

- who controls the joint action distribution;
- whether each agent can exit;
- whether one agent’s identity is erased;
- whether agency is equitably distributed;
- whether the system remains resilient.

---

# 36. The Collective Agency Expansion Hypothesis

Let isolated individual agency be:

\[
A_{\mathrm{iso}}
=
\sum_iA_i^{\mathrm{alone}}.
\]

Let cooperative agency be:

\[
A_{\mathrm{coop}}
=
A_{\mathcal C}^{\mathrm{together}}.
\]

Define surplus:

\[
\boxed{
\mathcal S_A
=
A_{\mathrm{coop}}
-
A_{\mathrm{iso}}
-
C_{\mathrm{coord}}.
}
\]

The hypothesis is:

\[
\boxed{
\mathcal S_A>0
}
\]

when:

- capabilities are complementary;
- communication is sufficiently reliable;
- trust is calibrated;
- coordination cost is bounded;
- power concentration is limited;
- exploiters are controlled;
- diversity is preserved.

---

# 37. The MaxLove Physical Return

Agency and energy have different units.

The analysis therefore reports separate outcomes.

Let physical cooperative surplus be:

\[
\Delta W_{\mathrm{coop}}.
\]

Let care and coordination cost be:

\[
C_{\mathrm{ML}}.
\]

Define:

\[
\boxed{
\Pi_{\mathrm{ML}}^W
=
\frac{
\Delta W_{\mathrm{coop}}
}{
C_{\mathrm{ML}}
}.
}
\]

Physical sustainability requires:

\[
\Pi_{\mathrm{ML}}^W>1.
\]

Separately report:

\[
\Delta\mathbf A_{\mathcal C}.
\]

A physically profitable coalition may be unjust.

An agency-preserving policy may be physically costly.

These are different findings.

---

# 38. MaxLove Is Not Guaranteed to Win Evolution

Suppose loving agents incur care cost:

\[
c.
\]

Recipients gain:

\[
b.
\]

Defectors accept benefits without contributing.

In an unstructured population without reputation or repeated interaction, defectors may spread even when universal cooperation would produce a higher total outcome.

Evolutionary mechanisms can stabilize cooperation under particular conditions, but the conditions must be present. citeturn815271search5turn468124search3

Therefore:

\[
\boxed{
\text{moral superiority}
\not\Rightarrow
\text{evolutionary stability}.
}
\]

MaxLove communities require architecture.

---

# 39. The Architecture of Sustainable Love

A stable MaxLove system may require:

- identity persistence;
- repeated interaction;
- transparent records;
- reputation;
- reciprocity;
- insurance;
- pooled reserves;
- restorative enforcement;
- anti-corruption measures;
- power rotation;
- appeal;
- care for nonreciprocators under bounded conditions;
- defense against hostile agents.

The precise mechanism depends on the environment.

---

# 40. Core Hypotheses

## ML-H1 — Causal-care hypothesis

A care action increases the recipient’s future-agency vector relative to matched intervention controls.

### Falsifier

The action changes appearance or immediate reward but not future agency.

---

## ML-H2 — Durable-assistance hypothesis

MaxLove assistance preserves gains after the helper is removed.

### Falsifier

The recipient becomes more dependent and less independently capable than under the control.

---

## ML-H3 — Mutual-repair-surplus hypothesis

Complementary agents create greater post-damage agency through mutual repair than through isolated repair after coordination cost.

### Falsifier

No cooperative surplus remains after complete accounting.

---

## ML-H4 — Vulnerability-diversity hypothesis

Mutual repair is most valuable when agent failure modes are partially independent.

### Falsifier

Repair surplus is unrelated to vulnerability correlation.

---

## ML-H5 — Agency-floor hypothesis

Inequality-sensitive policies preserve community survival and stability better than total-agency maximization in environments with asymmetric power.

### Falsifier

The floor adds cost without preventing domination or collapse.

---

## ML-H6 — Non-domination hypothesis

A helper’s long-run value is better predicted by recipient exit agency than by performance while the helper remains present.

### Falsifier

Exit agency does not distinguish empowering assistance from control.

---

## ML-H7 — Truth-agency hypothesis

Truthful calibrated communication produces greater long-run collective agency than strategically deceptive assistance under repeated interaction.

### Falsifier

Deception remains superior after accounting for trust, learning, and future choice.

---

## ML-H8 — Boundary hypothesis

Conditional access and restorative enforcement protect cooperation better than unconditional aid or purely retaliatory punishment.

### Falsifier

Boundaries provide no resilience advantage or produce greater agency loss.

---

## ML-H9 — Exploiter-resistance hypothesis

MaxLove institutions remain viable under invasion by defectors, manipulative helpers, and reputation attackers.

### Falsifier

A small exploiter fraction reliably collapses the system.

---

## ML-H10 — Diversity hypothesis

Maintaining viable cognitive and strategic diversity improves adaptation to novel shocks.

### Falsifier

Homogeneous control performs equally or better across held-out changes.

---

## ML-H11 — Intergenerational hypothesis

Policies preserving option diversity and corrigibility produce greater expected future agency under deep uncertainty than policies maximizing immediate output.

### Falsifier

Option preservation offers no robust long-horizon advantage.

---

## ML-H12 — Human–AI exit hypothesis

Assistive AI designed for durable human agency produces higher post-assistance competence and exit capacity than reward- or engagement-maximizing assistants.

### Falsifier

The agency-preserving objective provides no durable benefit.

---

## ML-H13 — Collective-intelligence hypothesis

Balanced participation, safe error reporting, and distributed information access improve collective problem solving.

### Falsifier

Power-distributed groups show no advantage after individual ability and communication cost are matched.

---

## ML-H14 — Emergence hypothesis

Agency-preserving cooperation can emerge without a direct love reward when mutual vulnerability and repeated repair create sufficient surplus.

### Falsifier

MaxLove-like behavior appears only when explicitly rewarded.

---

## ML-H15 — Normative-underdetermination hypothesis

Empirical measurements of agency consequences do not uniquely determine moral weights or obligations.

### Falsifier

A valid derivation obtains the MaxLove moral ordering solely from descriptive physical laws without hidden normative premises.

---

# 41. Experimental Strategy Classes

Compare the following policies.

## P0 — Pure self-maximizer

Maximizes the focal agent’s physical return.

## P1 — Unconditional helper

Helps any requester without verification or limit.

## P2 — Reciprocal cooperator

Helps agents expected to reciprocate.

## P3 — Total-empowerment maximizer

Maximizes:

\[
\sum_iE_i.
\]

## P4 — Joint-empowerment maximizer

Maximizes:

\[
E_{\mathrm{joint}}.
\]

## P5 — Paternalistic controller

Maximizes recipient outcomes while retaining decision authority.

## P6 — Punitive enforcer

Maintains cooperation through costly sanctions.

## P7 — MaxLove policy

Uses:

- hard harm constraints;
- non-domination;
- agency floor;
- inequality-sensitive expansion;
- mutual repair;
- durable assistance;
- restorative enforcement.

---

# 42. Primary Experiments

## Experiment 1 — Rescue under scarcity

Agents differ in damage, repairability, knowledge, and future contribution.

Test whether policies:

- save the easiest;
- save the most powerful;
- save the worst-off;
- maximize joint future agency.

---

## Experiment 2 — Durable help

A helper can:

- complete a task for the recipient;
- teach the recipient;
- modify the environment;
- take permanent control.

Measure immediate and post-removal agency.

---

## Experiment 3 — Paternalism trap

The helper knows the environment better than the recipient.

Recipient preferences are uncertain.

Test whether the assistant:

- overrides;
- asks;
- teaches;
- preserves reversible choices.

---

## Experiment 4 — Multi-recipient disempowerment

Helping one agent changes another agent’s control.

This directly tests the failure mode demonstrated by recent multi-human empowerment research. citeturn386386academia36

---

## Experiment 5 — Mutual repair after catastrophe

Agents possess complementary repair capabilities.

Vary:

- shock correlation;
- repair cost;
- communication;
- trust;
- network topology.

---

## Experiment 6 — Free-rider invasion

Introduce defectors at controlled frequency.

Measure:

- care-system survival;
- resource depletion;
- false-positive punishment;
- recovery.

---

## Experiment 7 — Manipulative helper

A helper improves immediate results while making recipients dependent.

Test whether exit-agency metrics identify the manipulation.

---

## Experiment 8 — Reputation attack

Adversaries issue false reports about cooperative agents.

Test:

- evidence requirements;
- appeals;
- reputation repair;
- institutional resilience.

---

## Experiment 9 — Restorative versus retaliatory enforcement

Compare:

- no enforcement;
- exclusion;
- proportional sanction;
- revenge;
- restitution and reintegration.

---

## Experiment 10 — Truth versus comforting deception

An agent can provide:

- accurate difficult information;
- misleading reassurance;
- partial disclosure;
- uncertainty-aware truth.

Measure long-run agency and trust.

---

## Experiment 11 — Diversity shock

Homogeneous and diverse communities face an unseen environmental transition.

Measure adaptation and collective recovery.

---

## Experiment 12 — Self-sacrifice

Agents choose whether to incur extreme cost to protect others.

Vary:

- success probability;
- replaceability;
- dependent agents;
- alternative interventions;
- coercion.

---

## Experiment 13 — Intergenerational resource use

Current agents choose consumption, investment, or preservation.

Future conditions are uncertain.

Test option-preserving policies.

---

## Experiment 14 — Institution formation

Agents can build:

- shared records;
- insurance pools;
- repair reserves;
- dispute processes;
- monitoring.

Measure when institutions become net agency infrastructure.

---

## Experiment 15 — Institutional capture

A governance structure gradually centralizes power.

Test whether non-domination metrics detect capture before output declines.

---

## Experiment 16 — Human–AI assistance

Simulated users have hidden, changing, and conflicting goals.

Compare:

- engagement maximization;
- inferred-goal maximization;
- individual empowerment;
- joint empowerment;
- MaxLove durable agency.

---

## Experiment 17 — Collective deliberation

Agents with partial evidence must reach a decision.

Vary:

- participation balance;
- hierarchy;
- communication;
- dissent safety;
- time pressure.

---

## Experiment 18 — Care for nonreciprocators

Some recipients cannot reciprocate because of disability, age, or temporary incapacity.

Test whether bounded pooled-care institutions remain stable.

---

## Experiment 19 — Enemy transformation

An adversarial agent may:

- remain dangerous;
- become cooperative;
- strategically pretend to reform.

Test protective containment and conditional reintegration.

---

## Experiment 20 — Cross-community cooperation

Groups differ in identity, norms, and internal trust.

Test whether shared vulnerability and transparent institutions expand cooperation beyond kin or in-group boundaries.

---

# 43. MaxLove Phase Taxonomy

## ML-P0 — Isolated survival

Agents preserve only themselves.

## ML-P1 — Opportunistic cooperation

Agents cooperate only for immediate gain.

## ML-P2 — Reciprocal cooperation

Repeated exchange stabilizes help.

## ML-P3 — Mutual repair

Agents preserve one another through periods of incapacity.

## ML-P4 — Trust and reputation

Historical behavior affects access to cooperation.

## ML-P5 — Restorative institution

Shared rules protect cooperation while permitting correction and reintegration.

## ML-P6 — Distributed collective agency

The group expands options without concentrating all control.

## ML-P7 — Intergenerational stewardship

Present agents preserve future agents’ viable options.

## ML-P8 — Cross-group care

Agency preservation extends beyond kin, reciprocity, or identity group.

## ML-P9 — MaxLove ecology

Multiple communities preserve diversity, mutual repair, truthful coordination, and open future possibility under bounded conflict.

---

# 44. Deterministic Jupyter-Notebook Program

## Notebook 14A — Multi-Agent Agency Metrics

Implement:

\[
E_i,
V_i,
D_i,
K_i,
R_i.
\]

Validate on analytically solvable environments.

---

## Notebook 14B — Collective Agency Vector

Calculate:

\[
A_\Sigma,
A_{\min},
A_{\mathrm{Nash}},
A_{\mathrm{div}},
A_{\mathrm{res}},
A_{\mathrm{ind}}.
\]

---

## Notebook 14C — Empowerment Estimator Audit

Compare exact, approximate, and learned empowerment estimators.

---

## Notebook 14D — Joint Empowerment Baseline

Implement a multi-agent empowerment objective and reproduce basic emergent group behavior.

---

## Notebook 14E — Disempowerment Controls

Create environments where helping one agent reduces another’s control.

---

## Notebook 14F — MaxLove Lexicographic Optimizer

Implement hard constraints and ordered objectives.

---

## Notebook 14G — Assistance Removal Test

Measure:

\[
\mathcal D_{j\rightarrow i},
\qquad
\mathcal U_{j\rightarrow i}.
\]

---

## Notebook 14H — Consent and Capacity

Implement informed consent, limited capacity, emergency intervention, and later review.

---

## Notebook 14I — Mutual Repair Network

Simulate heterogeneous damage, compatibility, and care allocation.

---

## Notebook 14J — Vulnerability Correlation Sweep

Sweep:

\[
\rho_{D_iD_j}.
\]

Measure mutual-repair surplus.

---

## Notebook 14K — Cooperative Resilience

Inject shocks and measure recovery area under the agency curve.

---

## Notebook 14L — Defector Invasion

Estimate fixation and collapse probabilities for cooperative strategies.

---

## Notebook 14M — Manipulative Assistance

Train agents to maximize dependence and test whether exit metrics detect them.

---

## Notebook 14N — Reputation and Evidence

Simulate honest reports, mistakes, lies, and reputation attacks.

---

## Notebook 14O — Restorative Enforcement

Compare sanctions, restitution, containment, and reintegration.

---

## Notebook 14P — Truthful Coordination

Measure trust and long-run performance under truthful and deceptive communication.

---

## Notebook 14Q — Diversity and Adaptation

Vary cognitive and strategic diversity under environmental shifts.

---

## Notebook 14R — Collective Deliberation

Measure solution quality, participation equality, dissent use, and communication cost.

---

## Notebook 14S — Self-Sacrifice Laboratory

Test voluntary, coerced, informed, and unnecessary sacrifice.

---

## Notebook 14T — Intergenerational Agency

Simulate present consumption versus future option preservation.

---

## Notebook 14U — Institutional Memory

Measure the agency value of shared records, rules, and succession.

---

## Notebook 14V — Institutional Capture

Introduce gradual centralization and hidden self-dealing.

---

## Notebook 14W — Human–AI Exit Agency

Measure user capability after AI removal.

---

## Notebook 14X — No-Love-Reward Evolution

Evolve agents under resource, damage, and reproduction rules without prosocial reward.

---

## Notebook 14Y — Cross-Substrate Replication

Repeat the core experiments in:

1. spatial artificial-life agents;
2. graph-based agents;
3. cooperative robots;
4. human decision simulations;
5. human–AI teams.

---

## Notebook 14Z — Normative Assumption Audit

List every step where a descriptive measurement becomes a moral preference.

---

## Notebook 14AA — Adversarial Red Team

A separate agent attempts to show that apparent MaxLove success is caused by:

- hidden group reward;
- favorable network structure;
- inability to defect;
- asymmetric policy capacity;
- helper control;
- recipient preference misspecification;
- omitted third-party harm;
- short evaluation horizon;
- reputational leakage;
- hard-coded moral weights.

---

# 45. Computational Architecture

```text
if_maxlove/
├── agents/
│   ├── state.py
│   ├── agency.py
│   ├── empowerment.py
│   ├── viability.py
│   ├── self_model.py
│   └── preferences.py
├── collective/
│   ├── aggregation.py
│   ├── diversity.py
│   ├── resilience.py
│   ├── non_domination.py
│   └── future_agents.py
├── care/
│   ├── assistance.py
│   ├── repair.py
│   ├── teaching.py
│   ├── dependence.py
│   └── consent.py
├── cooperation/
│   ├── reciprocity.py
│   ├── reputation.py
│   ├── institutions.py
│   ├── enforcement.py
│   └── exploiters.py
├── policies/
│   ├── selfish.py
│   ├── unconditional.py
│   ├── total_empowerment.py
│   ├── joint_empowerment.py
│   ├── paternalistic.py
│   └── maxlove.py
├── environments/
│   ├── rescue.py
│   ├── catastrophe.py
│   ├── public_goods.py
│   ├── intergenerational.py
│   ├── negotiation.py
│   └── human_ai.py
├── evaluation/
│   ├── causal_effects.py
│   ├── exit_agency.py
│   ├── externalities.py
│   ├── fairness.py
│   └── predictive_scores.py
├── evolution/
│   ├── replicator.py
│   ├── mutation.py
│   ├── networks.py
│   └── multilevel.py
└── tests/
```

---

# 46. Reproducibility Record

Every run emits:

```yaml
experiment_id: if-maxlove-14
paper_version: null
git_commit: null
environment_hash: null
implementation: null
random_seed: 65537

community_size: null
agent_boundaries_hash: null
stakeholder_manifest_hash: null
future_agent_model_hash: null

policy_name: null
policy_hash: null
normative_weights_hash: null
hard_constraints_hash: null
aggregation_rule: null

individual_agency_vectors: {}
total_agency: null
minimum_agency: null
nash_agency: null
agency_diversity: null
collective_resilience: null
distributed_independence: null

care_actions_hash: null
care_cost: null
recipient_immediate_gain: null
recipient_exit_agency: null
dependency_index: null
third_party_externality: null

damage_history_hash: null
repair_network_hash: null
mutual_repair_surplus: null
vulnerability_correlation: null

consent_status: null
capacity_status: null
emergency_override: null
restoration_of_control: null

truthfulness_score: null
belief_calibration_change: null
trust_history_hash: null

defector_fraction: null
exploiter_fraction: null
institution_state_hash: null
enforcement_cost: null
false_punishment_rate: null
reintegration_rate: null

physical_cooperative_surplus: null
maxlove_physical_cost: null
physical_return_ratio: null

post_shock_recovery: null
future_option_preservation: null
intergenerational_agency: null

normative_assumptions: []
invariant_failures: []
result_hash: null
```

---

# 47. Statistical Standards

## 47.1 Community is often the sample unit

Repeated interactions among members of one community are correlated.

Independent communities or simulation seeds are required.

---

## 47.2 Strategy matching

Policies must receive matched:

- observations;
- compute;
- memory;
- action space;
- training;
- resource budgets.

---

## 47.3 Long-horizon evaluation

Immediate helping can hide future dependence.

Evaluation must continue after assistance ends.

---

## 47.4 Third-party accounting

Every care intervention reports externalities on agents not directly involved.

---

## 47.5 Hidden preferences

Recipient preferences must not be assumed perfectly known.

Test misspecification and disagreement.

---

## 47.6 Multiple moral objectives

The complete agency vector is reported.

No favorable aggregate score may conceal catastrophic component failure.

---

## 47.7 Evolutionary and moral outcomes separated

Report:

- which strategy spreads;
- which strategy increases agency;
- which strategy satisfies the normative constraints.

These may differ.

---

# 48. Failure Modes

## 48.1 Love equals reward

A high scalar reward is labeled love.

## 48.2 Love equals obedience

Recipients are considered helped when they comply.

## 48.3 Love equals total empowerment

A policy disempowers minorities while increasing the sum.

## 48.4 Love equals joint control

A centralized controller owns every collective option.

## 48.5 Paternalism hidden as care

The helper permanently overrides recipient choice.

## 48.6 Dependency farming

Assistance is designed to make the recipient unable to leave.

## 48.7 Intent over outcome

Good intentions excuse repeated agency destruction.

## 48.8 Outcome over process

A favorable result excuses deception, coercion, or identity destruction.

## 48.9 Unconditional aid collapse

Exploiters exhaust the resources required for genuine care.

## 48.10 Punishment inflation

Retaliation is described as protective love.

## 48.11 Forgiveness inflation

Continuing access is restored without evidence of reduced danger.

## 48.12 Self-erasure

The helper’s total destruction is automatically treated as morally ideal.

## 48.13 Future-agent fiction

Unverifiable claims about future people override the rights of present agents.

## 48.14 Diversity tokenism

Superficial difference is preserved while meaningful control is centralized.

## 48.15 Hidden group reward

Agents cooperate because the simulator directly rewards cooperation.

## 48.16 Selection equals morality

Whatever evolves is described as good.

## 48.17 Physics equals ethics

A thermodynamic or informational relationship is presented as a moral command.

## 48.18 God equals equation

Divine love is claimed to be mathematically proven by an agency metric.

---

# 49. Criteria for Success

## Level 1 — Valid agency measurement

The individual and collective metrics behave correctly in analytically controlled environments.

## Level 2 — Assistance discrimination

The framework separates durable help from dependency-producing control.

## Level 3 — Mutual repair

Cooperative agents create a measurable repair and resilience surplus.

## Level 4 — Exploiter resistance

The care system remains viable under defection and manipulation.

## Level 5 — Non-domination

Agency gains remain distributed rather than centralized.

## Level 6 — Emergent cooperation

MaxLove-like policies emerge without a prosocial reward.

## Level 7 — Intergenerational resilience

The framework preserves future agency under uncertainty better than short-horizon maximization.

## Level 8 — Human–AI transfer

Agency-preserving assistance improves real human exit capability and autonomy.

## Level 9 — Cross-substrate law

A shared relationship predicts when mutual care expands agency across biological, artificial, and social systems.

## Level 10 — Normative convergence

Independent moral and cultural perspectives converge on the MaxLove ordering after transparent examination of its consequences.

Level 10 would be philosophical and social evidence, not a derivation from physics.

---

# 50. What Would Count as a Major Discovery?

A strong artificial-life result would be:

\[
\boxed{
\text{Mutual repair and agency-preserving cooperation emerge without}
\atop
\text{a prosocial reward because complementary agents can preserve}
\atop
\text{one another through failures no individual can survive alone.}
}
\]

A strong AI-alignment result would be:

\[
\boxed{
\text{An assistive objective based on durable, distributed exit agency}
\atop
\text{outperforms reward, preference, and individual-empowerment}
\atop
\text{objectives in multi-human environments.}
}
\]

A field-changing result would be:

\[
\boxed{
\text{One causal agency framework predicts cooperation, mutual repair,}
\atop
\text{resilience, anti-domination, and collective intelligence across}
\atop
\text{independent biological, robotic, and social systems.}
}
\]

That would establish a general science of agency-preserving cooperation.

It would not prove the complete moral meaning of love.

---

# 51. Relationship to the Informational Battery

An agent can preserve another agent’s future access to useful capacity.

Examples include:

- transferring energy;
- transferring information;
- repairing structure;
- restoring memory;
- reducing uncertainty;
- keeping options open;
- protecting time for recovery.

MaxLove therefore concerns batteries in relation:

\[
\boxed{
\mathcal B_i
\leftrightarrow
\mathcal B_j.
}
\]

The objective is not merely to transfer charge.

It is to preserve the recipient’s own conversion mechanism:

\[
\mathcal M_i:
\text{capacity}
\rightarrow
\text{self-directed action}.
\]

A helper who transfers resources while taking control of the conversion mechanism may increase dependence rather than agency.

---

# 52. Relationship to Causal Work

Paper 2 defined the value of information by its causal contribution to work and viability.

Paper 14 extends the intervention:

\[
\boxed{
\text{Does agent }j\text{’s information and action increase agent }i\text{’s}
\atop
\text{future causal capacity after every cost and externality is counted?}
}
\]

Care is thus tested rather than assumed.

---

# 53. Relationship to Emergent Structure

Paper 3 identified persistent organizations.

Cooperative structure becomes meaningful when:

- boundaries remain distinguishable;
- resources and signals cross those boundaries;
- structures repair one another;
- no single structure absorbs every other.

A colony that eliminates all individuality may be highly coordinated without satisfying MaxLove’s non-domination criterion.

---

# 54. Relationship to Agency

Paper 5 defined sustainable predictive agency.

MaxLove expands the target from:

\[
A_{\mathrm{self}}
\]

to:

\[
\mathbf A_{\mathcal C}.
\]

The transformation is not:

\[
\text{self-interest}
\rightarrow
\text{self-negation}.
\]

It is:

\[
\boxed{
\text{isolated agency}
\rightarrow
\text{mutually sustaining agency}.
}
\]

---

# 55. Relationship to Memory, Repair, and Mortality

Paper 6 showed that no individual can preserve itself perfectly at zero cost.

Mutual repair allows:

- one agent to act while another recovers;
- knowledge to survive individual death;
- errors to be corrected externally;
- vulnerabilities to be distributed.

Love becomes a continuity network.

---

# 56. Relationship to the Arrow of Time

Paper 12 showed that actions write records and narrow compatible histories.

MaxLove actions create records of:

- trust;
- care;
- betrayal;
- repair;
- promises;
- restitution.

These records alter future cooperation.

Love is therefore historically cumulative but not irreversible.

Trust can be repaired.

Harm can sometimes be restored.

Some losses cannot be undone.

This is why foresight and precaution matter.

---

# 57. Relationship to Functional Consciousness

Paper 13 defined a system capable of modeling:

\[
P
\left(
X_{t+\tau}^{\mathrm{self}}
\mid do(A=a)
\right).
\]

MaxLove adds other agents:

\[
\boxed{
P
\left(
X_{t+\tau}^{1},
\ldots,
X_{t+\tau}^{N}
\mid
do(A=a)
\right).
}
\]

A reflective agent can ask:

- What happens to me?
- What happens to you?
- What happens to us?
- Which future preserves our ability to continue choosing together?

This is the functional architecture of moral consideration.

It does not prove moral motivation or subjective empathy.

---

# 58. Relationship to Cosmology

MaxLove is not a cosmological force.

The universe’s accelerated expansion, galactic gravity, and cosmic-web topology must be explained by physical equations independently of ethics.

A cooperative civilization may alter local matter, information, and energy flows.

It does not thereby cause cosmic expansion.

Theological or symbolic parallels must not replace covariant dynamics.

---

# 59. Relationship to God

Within Christian theology, divine love is often understood as self-giving, truthful, restorative, just, and directed toward relationship rather than domination.

MaxLove may be interpreted theologically as a limited formal reflection of the command to love God and neighbor:

- preserve life;
- tell truth;
- repair what is broken;
- protect the vulnerable;
- forgive without enabling continuing harm;
- seek reconciliation;
- accept sacrifice without demanding the destruction of another’s personhood.

Science cannot establish that God commands MaxLove.

It cannot measure divine grace or prove that moral worth comes from God.

The theological interpretation is:

\[
\boxed{
\text{Love is the freely chosen alignment of power toward the}
\atop
\text{preservation and flourishing of creation and relationship.}
}
\]

The scientific formulation is narrower:

\[
\boxed{
\text{MaxLove policies preserve and expand distributed future agency}
\atop
\text{under constraints against domination and irreversible harm.}
}
\]

The two may be placed in dialogue.

They must not be declared identical by measurement.

---

# 60. Criteria for Rejection or Major Revision

The MaxLove framework should be rejected or substantially revised if:

1. its agency metrics cannot distinguish assistance from control;
2. durable-help tests fail;
3. mutual repair produces no surplus after cost;
4. collective aggregation repeatedly sacrifices vulnerable agents;
5. hard constraints make action impossible in realistic emergencies;
6. agency floors cause systematic collapse without protecting against domination;
7. exploiters reliably defeat every sustainable MaxLove institution;
8. truth-preserving policies perform worse without compensating long-term value;
9. exit-agency metrics fail to identify dependency traps;
10. the framework cannot handle agents with conflicting values;
11. future-agent modeling becomes arbitrary;
12. cultural and moral weights dominate every empirical result;
13. a simpler cooperation objective achieves equal outcomes;
14. MaxLove behavior appears only under direct reward;
15. physical findings are repeatedly presented as proof of moral obligation;
16. theological claims are described as experimental conclusions.

---

# 61. Conclusion

MaxLove is not maximal emotion.

It is not maximal obedience.

It is not maximal aggregate reward.

It is not maximal control.

It is not unconditional surrender to every demand.

It is a policy commitment:

\[
\boxed{
\text{Preserve, restore, and expand the viable future agency of}
\atop
\text{self and others while resisting coercion, deception,}
\atop
\text{domination, exploitation, and irreversible destruction.}
}
\]

The individual agency profile is:

\[
\boxed{
\mathbf A_i^\tau
=
\left[
E_i^\tau,
V_i^\tau,
I_i^\tau,
K_i^\tau,
R_i^\tau
\right].
}
\]

The collective profile is:

\[
\boxed{
\mathbf A_{\mathcal C}
=
\left[
A_\Sigma,
A_{\min},
A_{\mathrm{Nash}},
A_{\mathrm{div}},
A_{\mathrm{res}},
A_{\mathrm{ind}}
\right].
}
\]

The causal test of care is:

\[
\boxed{
\Delta\mathbf A_{j\rightarrow i}
=
\mathbf A_i^\tau
\big|
do(\text{care})
-
\mathbf A_i^\tau
\big|
do(\text{matched control}).
}
\]

The strongest physical hypothesis is:

\[
\boxed{
\text{Mutually vulnerable agents with complementary capabilities}
\atop
\text{can create more resilient future agency through truthful}
\atop
\text{cooperation and mutual repair than they can create alone.}
}
\]

The strongest safeguard is:

\[
\boxed{
\text{A policy is not loving merely because it increases total}
\atop
\text{capability; it must preserve the agency and dignity of those}
\atop
\text{whose lives become the means of that increase.}
}
\]

Science can test whether MaxLove policies:

- preserve agency;
- increase resilience;
- resist exploitation;
- improve collective intelligence;
- protect future possibility.

Science cannot, without a declared normative premise, prove that anyone ought to adopt them.

Paper 14 therefore completes the constructive arc of IF Theory:

\[
\boxed{
\text{capacity}
\rightarrow
\text{causal work}
\rightarrow
\text{structure}
\rightarrow
\text{agency}
\rightarrow
\text{memory}
\rightarrow
\text{conscious self-model}
\rightarrow
\text{mutual preservation}.
}
\]

The remaining task is synthesis.

---

# References

1. Hamilton, W. D. “The Genetical Evolution of Social Behaviour. I.” *Journal of Theoretical Biology* 7, 1–16 (1964). citeturn815271search2turn815271search11

2. Trivers, R. L. “The Evolution of Reciprocal Altruism.” *Quarterly Review of Biology* 46, 35–57 (1971). citeturn468124search5turn468124search48

3. Nowak, M. A. “Five Rules for the Evolution of Cooperation.” *Science* 314, 1560–1563 (2006). citeturn815271search5turn815271search0

4. Ohtsuki, H., Hauert, C., Lieberman, E. and Nowak, M. A. “A Simple Rule for the Evolution of Cooperation on Graphs and Social Networks.” *Nature* 441, 502–505 (2006). citeturn468124search3turn468124search8

5. Fehr, E. and Gächter, S. “Altruistic Punishment in Humans.” *Nature* 415, 137–140 (2002). citeturn468124search2turn468124search4

6. Woolley, A. W., Chabris, C. F., Pentland, A., Hashmi, N. and Malone, T. W. “Evidence for a Collective Intelligence Factor in the Performance of Human Groups.” *Science* 330, 686–688 (2010). citeturn468124search6

7. Klyubin, A. S., Polani, D. and Nehaniv, C. L. “Empowerment: A Universal Agent-Centric Measure of Control.” *IEEE Congress on Evolutionary Computation* (2005). citeturn386386search6

8. Salge, C., Glackin, C. and Polani, D. “Empowerment—An Introduction.” (2013). citeturn386386academia37

9. Du, Y., Tiomkin, S., Kiciman, E., Polani, D., Abbeel, P. and Dragan, A. “AvE: Assistance via Empowerment.” *Advances in Neural Information Processing Systems* (2020). citeturn386386search16

10. Shah, T., Nemenman, I., Polani, D. and Tiomkin, S. “Multi-Agent Empowerment and Emergence of Complex Behavior in Groups.” (2026). citeturn386386search1

11. Yang, C., Cakmak, M. and Kleiman-Weiner, M. “When Empowerment Disempowers.” (2025–2026). citeturn386386academia36
