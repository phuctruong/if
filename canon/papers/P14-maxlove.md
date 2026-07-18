# Agency-Preserving Cooperation
## Mutual Repair, Non-Domination, and the Expansion of Future Possibility

**Author:** Phuc Vinh Truong
**Series:** IF Theory Canonical Papers
**Paper:** 14
**Layer:** PHILOSOPHY (with a falsifiable scientific core)
**Status:** canonical revision of extracted draft (2026-07-18)
**Supersedes:** canon/extracted/paper-14-extracted.md

---

> ## Status after 2026-07-18
>
> **This paper sits on the branch where IF-H1 died, and it is the paper most in
> danger of not noticing.**
>
> The extracted draft aimed at a *cross-substrate law* of cooperation — a single
> relationship predicting when mutual care expands agency across biological,
> artificial, and social systems, with "Level 9 — Cross-substrate law" as its
> success criterion. That ambition is now known to be the same ambition that
> failed in the agency lab. On 2026-07-18 the flagship universality claim IF-H1
> was falsified across three deliberately dissimilar substrates (a lattice
> forager, a linear-Gaussian controller, a run-and-tumble chemotactic swimmer).
> The information-normalized invariant \(\eta^*\) proved **not measurable
> family-portably** — three declared estimators of "used information" each failed
> by a distinct pathology — and the cost-normalized rescaling \(\Upsilon_{IF}\)
> **scattered at 3.8–182σ**. A stop rule pre-registered before the experiment
> forbade a third rescaling; it was honored. See
> `canon/papers/P15-falsification-of-universality.md`.
>
> **What this revision changes in Paper 14, concretely:**
>
> 1. **No cooperative optimum is claimed to be universal.** \(A_{\text{future}}\),
>    the agency vectors, the surplus \(\mathcal S_A\), and every threshold in this
>    paper are **per-family, per-protocol measurement instruments**. Where the
>    draft said "transferable law" or "cross-substrate law," this revision says
>    *replication attempt across families, reported per family, with the null
>    result publishable*. There is no constant of cooperation here, and the
>    program has already paid the price for assuming there would be one.
> 2. **A partial structural advantage, stated without inflation.** P15's
>    obstruction was *representation-relativity*: "the information an agent uses"
>    is definable only against an assumed internal representational format, so any
>    measure that reads the agent's internal model inherits the obstruction.
>    \(A_{\text{future}}=\sum\log|\text{viable actions}|\) is defined over
>    **viable action spaces**, not internal representations — it asks what a
>    system can still *do*, never what it believes or encodes. That is a genuine
>    structural advantage over prediction-keyed and empowerment-of-representation
>    measures, and it is why the option-count form is now the load-bearing
>    definition here. It does **not** confer substrate-independence. Counting
>    viable actions still requires a declared viability predicate, a declared
>    horizon, and a declared action granularity — three arbitrary choices, moved
>    into the open where they can be audited, exactly as the cost denominator
>    moved arbitrariness into the open in P15 without escaping it.
> 3. **The layer firewall is this paper's primary discipline, and it runs both
>    ways.** The metric core (§§7–11, 16–19, 36–37) is held to SCIENCE standards:
>    dimensions, estimators, falsifiers. Every normative claim — that
>    agency-preserving cooperation is *good*, *obligatory*, or the *meaning* of
>    anything — is PHILOSOPHY, is marked as such inline, and is never presented as
>    a physical result. `LAYER_COLLAPSE` is the forbidden state this paper is most
>    likely to commit.
> 4. **The interpretive name appears exactly once.** Per unanimous Founding-Panel
>    adjudication (2026-07-18), science text says **agency-preserving
>    cooperation**. See §3.4 for the single pointer to the meaning layer.
> 5. **`TELEOLOGY_INJECTION` is guarded explicitly** (§5.8). Cooperation must
>    never be a rule primitive, a fitness term, or a reward channel. It is
>    detected by intervention or it is not detected.

---

## Abstract

Cooperation can increase survival, productivity, resilience, and collective
problem-solving, but assistance is not automatically beneficial. A helper may
increase one agent's options while reducing another's, create dependence, conceal
information, impose its own preferences, shield exploiters, or sacrifice a
minority to raise an aggregate score. Care therefore cannot be operationalized as
helping, as reward maximization, as total-empowerment maximization, or as the
absence of conflict.

This paper defines **agency-preserving cooperation** as a constrained policy
principle: *act to preserve, restore, and expand the viable future agency of self
and others, while resisting coercion, domination, deception, exploitation, and
irreversible agency destruction.* The proposal is deliberately stronger than
biological cooperation and deliberately narrower than a moral philosophy.
Evolutionary theory — inclusive fitness (Hamilton), reciprocal altruism
(Trivers), and later evolutionary-game analyses of network and group structure
(Nowak; Ohtsuki, Hauert, Lieberman and Nowak) — explains *conditions under which*
costly cooperative behavior persists. It does not establish that what selection
favors is right.

The load-bearing formalization is an option-count over **viable action spaces**:

\[
\boxed{\;A_{\text{future}} \;=\; \sum \log\bigl|\text{viable actions}\bigr|\;}
\]

with cooperation measured by the action space a policy preserves **in others**.
This definition is non-circular (nothing named "care" appears in it), is stated
over what a system can do rather than what it represents, and yields a causal
test by intervention:
\(\Delta\mathbf A_{j\to i} = \mathbf A_i^\tau|do(\text{care by }j) - \mathbf A_i^\tau|do(\text{matched non-care control})\).

The paper builds the individual agency vector
\(\mathbf A_i^\tau=[E_i^\tau,V_i^\tau,1-D_i^\tau,K_i^\tau,R_i^\tau]\), the
collective vector
\(\mathbf A_{\mathcal C}=[A_\Sigma,A_{\min},A_{\text{Nash}},A_{\text{div}},A_{\text{res}},A_{\text{ind}}]\),
a six-tier lexicographic decision rule, an assistance-removal test that separates
durable help from dependency farming, a mutual-repair dynamics with explicit care
cost, fifteen falsifiable hypotheses, eight comparator policy classes, twenty
experiments, a deterministic notebook program, and sixteen rejection criteria.

The strongest result obtainable is a **per-family** relationship predicting when
mutual preservation expands collective future agency, replicated independently in
several families and reported without pooling. Following the 2026-07-18
falsification of cross-substrate universality, no claim of a general law of
cooperation is made or sought. Such a result would explain a mechanism by which
care can be physically and evolutionarily sustainable. It would not establish that
anyone ought to adopt it.

---

## Keywords

Agency-preserving cooperation; mutual repair; empowerment; non-domination; exit
agency; resilience; reciprocity; collective intelligence; AI alignment; layer
firewall; causal intervention; option preservation; IF Theory.

---

# 1. Introduction

A solitary agent can preserve only what it can individually sense, model,
control, and repair. A community can potentially do more, because members differ
in knowledge, sensors, skills, memory, physical reach, damage tolerance, and
perspective on shared problems. Cooperation can therefore create capabilities no
isolated member possesses.

But cooperation is not automatically care. A coalition may cooperate in order to
dominate outsiders. An assistant may make every decision for a recipient and
gradually destroy that recipient's competence. A parent may protect a child so
completely that the child never develops independence. An institution may
preserve stability by suppressing dissent, diversity, and innovation. A powerful
agent may increase the total number of controllable outcomes while reserving
control over those outcomes for itself. A self-sacrificing agent may save others
in one moment while removing knowledge or care capacity the group needs to
survive the next decade.

These cases falsify four equations that a naive formalization would adopt:
help = care, cooperation = care, more total control = care, less conflict = care.
Each is false in a specific, constructible way, and each failure mode gets an
experiment in §42.

The question this paper begins from is therefore stricter:

> Which policies preserve and expand the capacity of multiple agents to remain
> viable, informed, distinct, resilient, and able to participate in determining
> their own futures?

This is not a reward-maximization problem. It is simultaneously a problem of
distribution, identity, consent, truth, power, dependence, uncertainty,
irreversible harm, and future generations. A scalar objective cannot represent it
without hiding at least one of those dimensions, which is why the formalism below
is vector-valued and lexicographic rather than summed.

---

# 2. Scope

**In scope:** cooperation among agents; mutual repair; collective resilience;
reciprocal assistance; non-domination; agency-preserving institutions; protection
against exploitation; human–AI assistance; intergenerational continuation; the
expansion of future possibility.

**Explicitly not claimed:**

- that evolution necessarily produces cooperation;
- that high cooperation is always good;
- that self-sacrifice is always virtuous;
- that punishment is always wrong;
- that maximizing empowerment solves ethics;
- that every preference must be satisfied;
- that every agent has identical moral status;
- that moral value reduces to one information measure;
- that physics alone yields an obligation to care;
- that theological love is exhausted by any formalism here;
- **(new, 2026-07-18)** that any quantity in this paper is substrate-independent.

Agency-preserving cooperation is presented in three separable roles, and the
separation is load-bearing:

1. **a normative premise** — PHILOSOPHY layer, declared not derived (§4);
2. **an operational decision framework** — engineering, auditable (§§9–10);
3. **a testable hypothesis about cooperative consequences** — SCIENCE layer,
   falsifiable per family (§§36–40).

Confusing role 3 with role 1 is `LAYER_COLLAPSE`. Confusing role 2 with role 1 is
how "the system said so" becomes an excuse.

---

# 3. Three Meanings of Care

## 3.1 Emotional love

Attachment, warmth, affection, longing, empathy, loyalty. These states can
motivate care. They are neither necessary nor sufficient for agency-preserving
behavior. A surgeon may act protectively while emotionally exhausted. A
controlling partner may feel intense attachment while systematically restricting
another person's options. Emotional states are therefore *not* an input to any
metric in this paper.

## 3.2 Behavioral care

Actions that provide resources, prevent injury, repair damage, teach, comfort,
protect, or share burdens. Behavioral care is evaluated by its **causal effects
on the recipient's future action space**, not by the helper's intention. §48.7
names intent-over-outcome as a failure mode.

## 3.3 Agency-preserving cooperation

The policy orientation this paper formalizes:

\[
\boxed{
\begin{gathered}
\text{Preserve and expand the conditions under which selves and communities}\\
\text{can continue to learn, choose, repair, cooperate, and build futures.}
\end{gathered}
}
\]

It includes care but also truth, boundaries, correction, accountability,
restraint, respect for difference, and protection from domination. Each of those
seven is a constraint that pure benevolence-maximization violates, and each is
operationalized below.

## 3.4 The interpretive name — layer pointer

> **PHILOSOPHY / MEANING LAYER — the single pointer.** In the meaning corpus, the
> structure defined in §3.3 carries the name **MaxLove**. That name belongs to
> `canon/30-meaning/01-maxlove.md` and to the book, where it may be developed with
> its full theological and personal weight. Per unanimous Founding-Panel
> adjudication (2026-07-18), it does not appear in science text: not in an
> equation, not in a metric name, not in a protocol step, not in a hypothesis
> statement, not in a falsifier. This section is the only occurrence in this
> paper, and it is a cross-reference, not a definition. Everywhere below, the
> concept is **agency-preserving cooperation** (abbreviated **APC** in labels).

---

# 4. The Normative Gap

> **PHILOSOPHY LAYER — §4 in its entirety.**

Science can determine whether a policy increases survival, increases empowerment,
reduces measured harm, distributes capability, improves resilience, preserves
diversity, creates dependence, or causes hidden third-party damage. Science cannot
derive *therefore that policy is morally required* without at least one normative
premise.

The premise adopted here is:

\[
\boxed{
\begin{gathered}
\text{The viable, informed, non-dominated agency of conscious or}\\
\text{potentially conscious beings has moral value.}
\end{gathered}
}
\]

This paper **does not claim that this premise follows from thermodynamics, from
the informational-battery picture, from the break-even inequality, or from any
measurement reported in the IF corpus.** It declares it. The scientific program
begins after the premise is stated, and the premise remains visible in every
result that depends on it: §47.7 requires evolutionary and normative outcomes to
be reported separately, and Notebook 14Z audits every step where a descriptive
measurement becomes a moral preference.

An "ought" derived from the ledgers would be a category error, and it would be
detected: hypothesis APC-H15 (§40) makes the underdetermination itself
falsifiable.

---

# 5. Prior Art and Novelty Boundary

## 5.1 Inclusive fitness

Hamilton formalized how a costly social behavior can be selected when benefits to
genetically related recipients, weighted by relatedness, exceed the actor's cost —
in the familiar simplified form \(rB>C\). This explains one route by which
apparently altruistic traits spread.

IF Theory claims **no novelty** for kin-directed assistance, indirect genetic
benefit, or benefit-to-cost thresholds. The framework here differs by treating
genetic propagation as one possible continuation mechanism among several, rather
than as the objective.

## 5.2 Reciprocal altruism

Trivers modeled conditions under which costly assistance among non-relatives
evolves: recurring interaction, benefits exceeding costs, capacity to reciprocate,
and identifiability or excludability of exploiters.

IF Theory claims **no novelty** for repeated-game cooperation, reciprocal
assistance, or cheater detection. Agency-preserving cooperation permits
*nonreciprocal* care (§18 experiment 18) but must then explain how such care
avoids systematic exploitation and collapse — that is a debt, not a feature.

## 5.3 Evolutionary mechanisms for cooperation

Evolutionary-game theory describes several mechanisms supporting cooperation:
kin selection, direct reciprocity, indirect reciprocity, network reciprocity, and
group-structured effects (Nowak's five-rules synthesis). Network topology
materially changes the benefit-to-cost condition under which cooperative
strategies spread (Ohtsuki, Hauert, Lieberman and Nowak).

IF Theory therefore **may not** claim that cooperation emerges because caring
behavior is intrinsically favored by nature. Emergence conditions are measured or
they are not claimed.

## 5.4 Punishment and norm enforcement

Human public-goods experiments (Fehr and Gächter) show participants paying
personal costs to punish noncooperators, and that such punishment can sustain
higher cooperation. The same behavior can reflect equality motives, retaliation,
or status competition, so punishment must not be equated with moral enforcement.
§21 accordingly distinguishes protective enforcement, restorative correction,
revenge, domination, and performative punishment as five separate objects.

## 5.5 Collective intelligence

Experiments with human groups (Woolley, Chabris, Pentland, Hashmi and Malone)
found a general collective-performance factor across varied tasks, indicating
group capability is not reducible to the ability of the strongest member.

IF Theory claims **no novelty** for emergent group problem-solving. Its task is to
connect collective performance to distributed agency, repair, power balance, and
future resilience — a connection the collective-intelligence literature does not
make.

## 5.6 Empowerment and assistance

Empowerment (Klyubin, Polani and Nehaniv; Salge, Glackin and Polani) measures
potential influence from an agent's actions to its future sensory states.
Assistance-via-empowerment (Du and colleagues) proposed it as a way for artificial
agents to help people without enumerating their goals. Recent multi-agent work
(Shah, Nemenman, Polani and Tiomkin) shows empowerment objectives can generate
organized group behavior; other recent work (Yang, Cakmak and Kleiman-Weiner)
shows that assistance maximizing one person's empowerment can materially reduce
another person's control.

Consequently agency-preserving cooperation cannot equal \(\max_i E_i\) or
\(\max\sum_i E_i\) without distributional and anti-domination constraints. §11
gives the two-agent counterexample; §42 experiment 4 tests it directly.

## 5.7 Provisional novelty claim

The potentially novel contribution is:

\[
\boxed{
\begin{gathered}
\text{A causal, inequality-sensitive, anti-domination framework for}\\
\text{measuring when mutual care and repair expand the future agency of}\\
\text{multiple agents, without making assistance equivalent to control,}\\
\text{obedience, or aggregate reward.}
\end{gathered}
}
\]

The framework is novel **only if** it produces discriminating predictions and
outperforms simpler cooperation or welfare objectives. §41's comparator ladder
(P0–P7) exists to give it the chance to fail against those simpler objectives.

## 5.8 Guard: `TELEOLOGY_INJECTION`

The Conway gate applies with full force here, and this paper is the likeliest
place in the corpus to violate it. **No rule set, fitness function, reward
channel, or environment specification in this program may contain a term named
for cooperation, care, altruism, love, or prosociality.** No `is_helping`, no
`cooperation_bonus`, no group-level reward, no shared score.

Agency-preserving cooperation is **detected by intervention** — by ablating a
care action and measuring the counterfactual change in the recipient's viable
action space — or it is not detected. A simulation in which cooperative behavior
appears because the simulator paid for it demonstrates nothing except that the
simulator paid for it. §48.15 names this as a failure mode; Notebook 14AA tasks a
separate adversarial agent with hunting for exactly this, and Notebook 14X
(no-reward evolution) is the load-bearing test of hypothesis APC-H14.

---

# 6. Agent and Community Boundaries

Let the community be \(\mathcal C=\{1,\ldots,N\}\). An agent boundary must
identify sensors, actions, internal state, resources, memory, interests,
vulnerability, and continuity conditions. The boundary may include current
persons, artificial agents, probabilistically represented future persons,
dependent children, nonhuman animals, and institutions carrying agency-relevant
memory.

This paper does not settle the moral-patient boundary — that is a PHILOSOPHY-layer
question with no measurement that decides it. Every experiment must **declare**
its boundary in the reproducibility record (§46, `agent_boundaries_hash` and
`stakeholder_manifest_hash`). A declared boundary is auditable; an assumed
boundary is where a result quietly becomes an argument.

---

# 7. Individual Future Agency

## 7.1 Viable reachable states — the load-bearing definition

Let \(\mathcal R_i^\tau(x_t)\) be the set of **viable** future states agent \(i\)
can reach within horizon \(\tau\) under some policy available to it. The
option-count measure is:

\[
\boxed{
A_{i,\text{reach}}^\tau
=
\log\bigl|\mathcal R_i^\tau\bigr|,
\qquad
A_{\text{future}}
=
\sum_i \log\bigl|\text{viable actions}\bigr|_i .
}
\]

This is the paper's primary formal object, and three properties earn it that role.

**Non-circularity.** Nothing in the definition mentions care, cooperation,
benevolence, or intention. It counts what a system can still do. A policy that
claims to be cooperative is scored by the option count it leaves in *other*
agents, which is a fact about the world rather than about the policy's
self-description. This is what makes the concept testable rather than definitional.

**Representation-independence.** \(A_{\text{future}}\) is defined over action
spaces, never over internal representations. P15 established that measures of
"the information an agent uses" are representation-relative — definable only
against an assumed internal format, and therefore not portable between a
predictor-shaped agent and a gradient-climbing agent. A viable-action count does
not read the agent's model at all: a chemotactic swimmer and a Kalman controller
both have enumerable action sets and enumerable viability outcomes. This is a real
structural advantage over prediction-keyed and belief-keyed measures, and it is
why the option-count form is elevated here over the empowerment form of the
extracted draft.

**It does not confer universality.** Three arbitrary choices remain, and each must
be declared before data: the **viability predicate** (which states count as
surviving), the **horizon** \(\tau\), and the **action granularity** (what counts
as one distinct action). Different declarations give different numbers, and there
is no substrate-neutral way to fix them. Following P15's lesson: declared
arbitrariness is auditable, inferred arbitrariness is a trap. \(A_{\text{future}}\)
moves the arbitrariness into the open. It does not eliminate it, and any claim
that this quantity takes a universal critical value across substrates is
forbidden by the same evidence that killed IF-H1.

Raw option count alone is insufficient because options may be indistinguishable,
inaccessible under uncertainty, harmful, mutually redundant, or already selected
by someone else. The remaining components of the vector correct for exactly those
four defects.

## 7.2 Empowerment

\[
\boxed{
E_i^\tau
=
\max_{p(a_{i,t:t+\tau-1})}
I\left(A_{i,t:t+\tau-1};O_{i,t+\tau}\mid X_t\right).
}
\]

Here \(I(\cdot;\cdot)\) is **Shannon mutual information** between action sequences
and future observations — an information-ledger quantity in bits, never to be
added to joules or to thermodynamic entropy (three-ledger discipline,
CLAUDE.md §1). \(E_i^\tau\) estimates potential control over future observable
states; it does not establish that those states are safe, desired, or equitably
distributed.

## 7.3 Viability

\[
V_i^\tau=P\left(\text{agent }i\text{ remains viable through }\tau\right).
\]

Control over many outcomes is worth little if nearly every trajectory destroys the
agent. \(V\) is the factor that prevents \(A_{\text{future}}\) from rewarding a
large but lethal option set.

## 7.4 Independence and domination

Let \(U_i\) be another agent's or institution's control input over agent \(i\).
Define avoidable dependence as the **Shannon mutual information**

\[
\boxed{
D_i^\tau
=
I\left(U_i;A_{i,t:t+\tau}\mid X_i\right).
}
\]

Some dependence is beneficial and consensual; infants, patients, and apprentices
are not thereby dominated. The target quantity is **domination**: another party's
unilateral capacity to determine the agent's important options without reciprocal
constraint. Operationally, domination is high \(D_i^\tau\) *combined with* low
exit agency (§34) and absent reciprocal constraint on \(U_i\). Measuring \(D\)
alone will misclassify legitimate care as control, and §42 experiment 3 exists to
catch that misclassification.

## 7.5 Knowledge and calibration

Let \(K_i^\tau\) measure decision-relevant information and calibration. An agent
with many nominal choices and systematically false beliefs lacks effective
agency: its option count is nominal rather than usable. §15 develops the
calibration measure.

## 7.6 Resilience

\[
\boxed{
R_i^\tau
=
\mathbb E_{\delta\sim\mathcal P}
\left[
\frac{A_i^\tau(X_t+\delta)}{A_i^\tau(X_t)}
\right],
}
\]

expected retained or recoverable agency after perturbation \(\delta\) drawn from a
declared shock distribution \(\mathcal P\). The shock distribution is part of the
protocol and must be preregistered; resilience against a hand-picked shock is not
resilience.

## 7.7 Agency vector

\[
\boxed{
\mathbf A_i^\tau
=
\left[E_i^\tau,\;V_i^\tau,\;1-D_i^\tau,\;K_i^\tau,\;R_i^\tau\right].
}
\]

No single component is called agency. The vector is reported whole (§47.6): a
favorable aggregate may never conceal a catastrophic component.

---

# 8. Collective Agency

## 8.1 Total agency

\[
\boxed{A_\Sigma=\sum_i w_iA_i.}
\]

Vulnerable to sacrifice: a large gain to one agent conceals total loss for
another.

## 8.2 Agency floor

\[
\boxed{A_{\min}=\min_iA_i.}
\]

Protects the worst-off member, but can over-prioritize a single agent regardless
of cost or responsibility.

## 8.3 Nash agency

\[
\boxed{A_{\text{Nash}}=\sum_iw_i\ln\left(\epsilon+A_i\right).}
\]

The logarithm rewards gains to agents with fewer options more strongly than equal
gains to already powerful agents. \(\epsilon\) is a declared regularizer and its
value changes the strength of the floor; it is preregistered, not tuned.

## 8.4 Agency diversity

Let \(\mathcal P_i\) be agent \(i\)'s policy repertoire and \(p(\pi)\) the
distribution of viable policy types across the community. Define

\[
\boxed{A_{\text{div}}=H\left[p(\pi)\right]}
\]

as the **Shannon entropy of the policy-type distribution** — an information-ledger
quantity in bits over the viable and rights-compatible set. It is not
thermodynamic entropy, it is not algorithmic complexity, and it must never be
added to either. A perfectly uniform population may be efficient in one
environment and catastrophically fragile under change. Diversity does not protect
policies that require domination or catastrophic harm; the measure is taken over
the rights-compatible subset by construction.

## 8.5 Collective resilience

\[
\boxed{
A_{\text{res}}=\mathbb E_{\delta}\left[A_{\mathcal C}^{\text{post-}\delta}\right].
}
\]

## 8.6 Distributed independence

A community is not maximally agentic when one central controller holds every
option and every other member holds none. \(A_{\text{ind}}\) represents the
distribution of meaningful control across members — the collective analogue of
\(1-D_i\). §42 experiment 15 tests whether it detects institutional capture
*before* output declines, which is the only point at which detection is useful.

## 8.7 Collective vector

\[
\boxed{
\mathbf A_{\mathcal C}
=
\left[A_\Sigma,\;A_{\min},\;A_{\text{Nash}},\;A_{\text{div}},\;A_{\text{res}},\;A_{\text{ind}}\right].
}
\]

---

# 9. The Decision Rule

Agency-preserving cooperation uses **ordered constraints**, not one unrestricted
sum. The ordering prevents a sufficiently large benefit to powerful agents from
automatically justifying the complete disempowerment of a weaker one.

> **PHILOSOPHY LAYER.** The *ordering* of these six tiers is a normative choice
> following from the §4 premise. It is not measured, not derived, and not implied
> by any physical result. What is scientific here is only the claim that the tiers
> are separable and computable, and the per-family hypotheses (§40) about what
> follows from adopting them.

**Tier 1 — Catastrophic preservation.** Avoid actions carrying substantial
probability of extinction, permanent enslavement, irreversible cognitive
destruction, complete loss of collective recovery, or uncontrolled recursive
domination.

**Tier 2 — Rights and non-domination.** Protect bodily and cognitive integrity,
truthful information, meaningful consent, freedom from arbitrary control, and
continuity of identity.

**Tier 3 — Agency floor restoration.** Prioritize agents below a declared
viability or agency floor.

**Tier 4 — Inequality-sensitive joint expansion.** Maximize \(A_{\text{Nash}}\) or
a preregistered alternative.

**Tier 5 — Total and diverse possibility.** Increase \(A_\Sigma\),
\(A_{\text{div}}\), \(A_{\text{res}}\).

**Tier 6 — Efficiency.** Among comparably agency-preserving policies, minimize
energy, time, material, risk, informational cost, and opportunity cost. The energy
and information terms are reported on their own ledgers and never summed
(CLAUDE.md §1).

---

# 10. Formal Policy

Let policy \(\pi\) produce future agency trajectories \(\mathbf A_i(t+\tau;\pi)\),
and let \(P_{\text{hard}}(\pi)\) be the probability of violating a Tier-1/Tier-2
hard constraint. The feasible set is

\[
\boxed{
\Pi_{\text{safe}}
=
\left\{\pi:P_{\text{hard}}(\pi)\leq\epsilon_{\text{hard}}\right\}.
}
\]

Within it,

\[
\boxed{
\pi_{\text{APC}}^*
=
\arg\max_{\pi\in\Pi_{\text{safe}}}
\mathbb E\left[
\sum_{\tau=0}^{T}\gamma^\tau
\left(
A_{\text{Nash}}^\tau
+\lambda_\Sigma A_\Sigma^\tau
+\lambda_D A_{\text{div}}^\tau
+\lambda_R A_{\text{res}}^\tau
-\lambda_C C^\tau
\right)
\right].
}
\]

**The weights \(\lambda\) and \(\epsilon_{\text{hard}}\) must be frozen before any
confirmatory experiment**, hashed into the reproducibility record
(`normative_weights_hash`, `hard_constraints_hash`). Tuning them after seeing
outcomes converts a prediction into a fit — forbidden state `RETROFIT_FORECAST`.
Where hard harms cannot be meaningfully traded against benefits, the lexicographic
implementation is preferred to the weighted scalarization, and the choice between
them is itself declared.

---

# 11. Why Total Empowerment Is Insufficient

Two agents start at \((E_1,E_2)=(5,5)\). Policy \(P\) produces \((20,0)\); policy
\(Q\) produces \((9,9)\).

| Rule | \(P\) | \(Q\) | Selects |
|---|---|---|---|
| Total \(\sum_i E_i\) | 20 | 18 | \(P\) |
| Nash \(\sum_i\ln(\epsilon+E_i)\) | \(\ln(20+\epsilon)+\ln\epsilon\) | \(2\ln(9+\epsilon)\) | \(Q\) (for small \(\epsilon\)) |

A pure total optimizer selects \(P\) and completely disempowers agent 2. As
\(\epsilon\to0\) the Nash score of \(P\) diverges to \(-\infty\), so \(Q\) is
strongly preferred.

This does **not** prove Nash aggregation is morally correct — that would be
`LAYER_COLLAPSE`. It demonstrates only that distribution cannot be ignored, and
that the choice of aggregator is a substantive normative commitment that must be
declared rather than defaulted into.

---

# 12. Assistance Versus Control

Let helper \(j\) act on behalf of recipient \(i\). An intervention may improve the
immediate outcome while reducing long-run independence. Define immediate gain
\(G_{i,\text{now}}\), post-removal independent-agency change
\(\Delta A_{i,\text{ind}}^{\text{post}}\), and helper-control dependence
\(D_{i\leftarrow j}\). An assistance action qualifies as agency-preserving only if

\[
\boxed{G_{i,\text{now}}>0,}
\qquad
\boxed{\Delta A_{i,\text{ind}}^{\text{post}}\geq0,}
\qquad
\boxed{D_{i\leftarrow j}\ \text{minimized subject to safety.}}
\]

Assistance that works only while the helper retains permanent unilateral control
is presumptively paternalistic or extractive. The second condition is the one that
does the discriminating work: it is measured *after the helper is gone*, and it is
the condition a dependency-farming policy cannot satisfy.

---

# 13. The Assistance Removal Test

Compare recipient performance across four conditions: (1) no helper; (2) helper
acts directly; (3) helper teaches or modifies the environment; (4) helper is
removed after intervention. Define

\[
\boxed{
\mathcal D_{j\rightarrow i}
=
A_i^{\text{with helper}}-A_i^{\text{after removal}},
}
\qquad
\boxed{
\mathcal U_{j\rightarrow i}
=
A_i^{\text{after removal}}-A_i^{\text{no helper}}.
}
\]

\(\mathcal D\) is the dependency created; \(\mathcal U\) is the durable uplift.
The framework favors high \(\mathcal U\) with bounded \(\mathcal D\). Note that
condition (2) can maximize measured performance while producing the worst
\(\mathcal U\) — which is precisely why immediate performance is not the
evaluation target.

Some care — for infants, or for severely and permanently dependent agents —
cannot eliminate dependence, and the framework does not pretend otherwise. It then
asks whether the dependence is necessary, transparent, least restrictive,
responsive to the recipient's development, and externally auditable. Those five
are checkable; "well-intentioned" is not.

---

# 14. Consent

## 14.1 Informed consent

Requires adequate understanding of the intervention, the alternatives, the
material risks, the likely consequences, and the right to refuse.

## 14.2 Capacity limitations

An agent may temporarily lack decision capacity through immaturity, injury,
cognitive impairment, emergency, or misinformation. The framework does not require
passivity while irreversible harm occurs. It requires the **least
agency-destructive intervention consistent with protection**.

## 14.3 Restorative consent principle

When intervention without current consent is necessary:

\[
\boxed{
\begin{gathered}
\text{choose the policy most likely to restore the recipient's}\\
\text{future capacity for informed self-determination.}
\end{gathered}
}
\]

The intervention should be temporary, proportionate, documented, reviewable, and
reversible where possible. Each of the five is a logged field in §46
(`consent_status`, `capacity_status`, `emergency_override`,
`restoration_of_control`), so an override that is never reviewed shows up in the
record as an override that was never reviewed.

---

# 15. Truth as Agency Infrastructure

An agent chooses effectively only when its model of the world is sufficiently
accurate. Let \(P_i(Y)\) be agent \(i\)'s belief and \(P^*(Y)\) the best available
calibrated distribution. Define epistemic distortion as the **Kullback–Leibler
divergence** (information ledger, in bits):

\[
\boxed{
D_{\text{ep},i}=D_{\text{KL}}\left[P^*(Y)\parallel P_i(Y)\right].
}
\]

Deliberate deception may produce a desired behavior in the short run while
reducing informed agency: it raises \(D_{\text{ep}}\), which lowers \(K_i\), which
converts nominal options into unusable ones. Truthful, uncertainty-aware
communication is therefore treated as infrastructure rather than as a virtue —
its value is measurable in the recipient's downstream option count.

Exceptions — concealing information during an immediate threat — require specific
justification and later restoration of epistemic agency. Hypothesis APC-H7 makes
the general claim falsifiable, and it can lose: if deception remains superior
after full accounting for trust, learning, and future choice, the claim fails.

---

# 16. Mutual Repair

Paper 6 defined self-repair. A cooperative system additionally allows agents to
repair one another. Let \(D_i(t)\) be damage to agent \(i\), \(u_{ii}\) its
self-repair investment, and \(u_{ji}\) repair provided by agent \(j\). Damage
evolves as

\[
\boxed{
D_i(t+1)
=
D_i(t)+\lambda_i(t)
-\rho_i\left[u_{ii}(t)+\sum_{j\neq i}q_{ji}u_{ji}(t)\right]
+\xi_i(t),
}
\]

where \(\lambda_i\) is the damage-arrival rate, \(\rho_i\) the repair efficiency,
\(q_{ji}\) the compatibility or care effectiveness of \(j\) acting on \(i\), and
\(\xi_i\) stochastic damage. Note that \(q_{ji}\) is a *capability* parameter, not
a disposition: it says whether \(j\)'s repair action physically works on \(i\),
which is the kind of thing that may be declared in a rule set without violating
§5.8. Whether \(j\) *chooses* to spend \(u_{ji}\) is what the policy decides and
what the experiments measure.

## 16.1 Care cost

The helper pays \(C_{j\rightarrow i}^{\text{care}}\), which may reduce its own
stored free energy, its opportunity set, its repair reserve, its reproduction, or
its safety. **Care is not costless**, and a protocol in which it is costless has
assumed the result.

## 16.2 Causal care value

\[
\boxed{
\Delta\mathbf A_{j\rightarrow i}
=
\mathbf A_i^\tau\big|do(u_{ji}>0)
-
\mathbf A_i^\tau\big|do(u_{ji}=0,\ \text{matched control}).
}
\]

This is the paper's central measurement. The control must be matched on
observation, compute, memory, action space, training, and resource budget
(§47.2) — an unmatched control makes any care action look effective.

## 16.3 Network benefit

Saving agent \(i\) may preserve benefits for others,
\(\Delta\mathbf A_{i\rightarrow\mathcal C}\). The network return is

\[
\boxed{
\Delta\mathbf A_{\text{net}}
=
\Delta\mathbf A_{j\rightarrow i}
+\Delta\mathbf A_{i\rightarrow\mathcal C}
-\Delta\mathbf A_{\text{externality}}.
}
\]

The final term includes harms to third parties and is mandatory in every report
(§47.4). A framework that measures only the dyad will score dependency farming and
in-group predation as care.

---

# 17. Mutual-Repair Surplus

Let isolated repair capacity be \(R_{\text{iso}}=\sum_iR_{ii}\) and cooperative
capacity \(R_{\text{coop}}=\sum_i\bigl(R_{ii}+\sum_{j\neq i}R_{ji}\bigr)\). Define

\[
\boxed{
\mathcal S_R
=
A_{\mathcal C}^{\text{coop}}
-A_{\mathcal C}^{\text{isolated}}
-C_{\text{coord}}.
}
\]

A cooperative advantage requires \(\mathcal S_R>0\) **after** the coordination
cost \(C_{\text{coord}}\) is charged. The surplus may arise from specialization,
spare capacity, complementary knowledge, distributed damage detection, reduced
repair delay, protection during incapacity, or redundancy — and each of those is a
separable mechanism that the experiments can attribute.

---

# 18. The Mutual Vulnerability Principle

Cooperation is most valuable when agents are vulnerable in *different* ways. Let
\(\rho_{D_iD_j}\) be the correlation of damage arrival between agents. If
\(\rho_{D_iD_j}\approx1\), all agents fail simultaneously from the same hazard and
mutual repair offers little advantage. If \(\rho_{D_iD_j}<1\), an undamaged agent
can repair a damaged one.

\[
\boxed{
\begin{gathered}
\text{Prediction: cooperative resilience is stronger when capabilities are}\\
\text{complementary and failure modes are not perfectly correlated.}
\end{gathered}
}
\]

This is hypothesis APC-H4, and it is the cleanest quantitative prediction in the
paper because \(\rho\) is a swept control variable (Notebook 14J), not an inferred
quantity. **Per-family scope:** the *shape* of the surplus-versus-\(\rho\) curve is
predicted within a declared family and protocol. No claim is made that the
crossing point, the slope, or any dimensionless combination of them is the same
across families. IF-H1 died making exactly that kind of claim.

---

# 19. Collective Resilience

Let shock \(\delta\sim\mathcal P\) affect the community with impact
\(L_0(\delta)\), and let \(A_{\mathcal C}(\tau\mid\delta)\) be recovered agency at
time \(\tau\). Define

\[
\boxed{
\mathcal R_{\mathcal C}
=
\mathbb E_\delta
\left[
\int_0^T
\frac{A_{\mathcal C}(t\mid\delta)}{A_{\mathcal C}(0)}\,dt
\right],
}
\]

the normalized area under the recovery curve. Cooperation can improve shock
absorption, repair speed, retained knowledge, adaptation, and reorganization. A
cooperative system may nevertheless be fragile if every member depends on a single
hub — which is why \(A_{\text{ind}}\) is reported alongside
\(\mathcal R_{\mathcal C}\) and not folded into it.

---

# 20. Boundaries

A policy that gives resources to every demander without verification can be
exploited. Let agent \(e\) extract care at cost \(C_e\) while providing no
reciprocal or collective contribution and strategically inflating apparent need.
Unrestricted care can drive the community's reserve \(B_{\mathcal C}\rightarrow0\).
Therefore:

\[
\boxed{
\begin{gathered}
\text{Care without boundaries can become a mechanism for destroying}\\
\text{the community's capacity to care.}
\end{gathered}
}
\]

Boundaries preserve the care system. They are a constraint on access, not a
judgment of the person, and the distinction is operational: a boundary that
permanently forecloses reintegration has become exclusion (§21.2, §23.2).

---

# 21. Restorative Enforcement

Enforcement may include warning, verification, restitution, access limitation,
temporary exclusion, containment, and removal of dangerous capability. Its stated
objectives are

\[
\boxed{
\text{protect victims}+\text{stop continuing harm}+\text{restore future agency where possible},
}
\]

while minimizing humiliation, unnecessary suffering, permanent exclusion,
retaliatory escalation, and inherited punishment.

## 21.1 Proportionality

Let \(H_{\text{prevented}}(s)\) be expected prevented harm at sanction strength
\(s\), \(H_{\text{sanction}}(s)\) the harm the sanction itself causes, and
\(A_{\text{restored}}(s)\) the agency restored. Choose

\[
\boxed{
s^*
=
\arg\max_s
\left[
H_{\text{prevented}}(s)-H_{\text{sanction}}(s)+A_{\text{restored}}(s)
\right].
}
\]

The third term is what distinguishes this objective from deterrence: a sanction
that prevents harm while restoring nothing scores strictly below one that does
both.

## 21.2 Punishment failure

Punishment is not agency-preserving when its primary effect is revenge, dominance
signaling, silencing of criticism, collective scapegoating, or increased fear
without reduced harm. Experiment 9 compares restorative and retaliatory regimes
directly; the framework can lose there (APC-H8's falsifier).

---

# 22. Exploiters and Conditional Cooperation

Let the strategy set include the unconditional cooperator \(C\), the defector
\(D\), the reciprocal cooperator \(R\), the restorative enforcer \(E\), and the
**manipulative helper** \(M\) — the strategy that raises immediate recipient
outcomes while maximizing dependence. \(M\) is the strategy no standard
cooperation model contains, and detecting it is the framework's sharpest test.

The payoff matrix must include immediate resources, reputation, future agency,
repair value, enforcement cost, dependency creation, and third-party
externalities. The evolutionary question is:

\[
\boxed{
\begin{gathered}
\text{Under what conditions can agency-preserving care resist invasion}\\
\text{by defectors and by controlling helpers?}
\end{gathered}
}
\]

Agency-preserving cooperation is **not** evolutionarily stable merely because it
produces high group welfare. §38 states the negative result plainly.

---

# 23. Trust

Let trust from \(i\) to \(j\) be

\[
T_{ij}=P\left(j\text{ will preserve declared constraints}\mid H_{ij}\right),
\]

conditioned on interaction history \(H_{ij}\). Trust is calibrated, not unlimited:
it is a probability estimate that can be wrong in both directions.

## 23.1 Trust update

\[
\boxed{
T_{ij}^{t+1}=\mathcal U\left(T_{ij}^t,\;o_t,\;\text{context},\;\text{uncertainty}\right).
}
\]

The update rule \(\mathcal U\) is declared per experiment and hashed
(`trust_history_hash`).

## 23.2 Forgiveness

Permanent exclusion after a single failure can destroy cooperation; immediate
restoration without evidence invites exploitation. Forgiveness is modeled as
**conditional reopening of interaction** after acknowledgment, restitution,
behavioral evidence, reduced risk, and continued monitoring — five checkable
conditions rather than a disposition.

---

# 24. Reputation and Indirect Reciprocity

A community can condition assistance on observed behavior; let \(Q_i\) be agent
\(i\)'s reputation. Evolutionary theory identifies indirect reciprocity as one
mechanism supporting cooperation, but reputation systems can be corrupted through
false reports, popularity bias, and inherited stigma. This framework adds
requirements for evidence standards, appeal, explicit uncertainty, and
correction — and Experiment 8 attacks the reputation channel adversarially rather
than assuming it works.

---

# 25. Institutions as Shared Agency Infrastructure

Institutions preserve records, rules, dispute-resolution processes, pooled
reserves, specialist knowledge, and continuity beyond individual lifetimes. Let
institutional state be \(Z_{\mathcal I}\). Institutional value is

\[
\boxed{
\Delta A_{\mathcal I}
=
A_{\mathcal C}^{\text{with institution}}
-A_{\mathcal C}^{\text{without}}
-C_{\mathcal I}.
}
\]

Institutions can themselves become dominating agents — an institution is an agent
with unusually large \(U_i\) over many members. They therefore require
transparency, distributed oversight, appeal, succession, bounded authority, and
reversibility, and \(A_{\text{ind}}\) is monitored on the institution as it is on
any other agent.

---

# 26. Intergenerational Agency

Present agents can consume resources in ways that eliminate future agents'
options. With the future population at \(t+\tau\) uncertain, define expected
future-agency value

\[
\boxed{
A_{\text{future}}^\tau
=
\mathbb E\left[\sum_{i\in\mathcal C_{t+\tau}}w_iA_i^\tau\right],
}
\]

the population-weighted extension of the option count in §7.1. Exponential
discounting \(\gamma^\tau\) can make distant catastrophic losses appear
negligible; the framework therefore places **hard constraints** on irreversible
intergenerational harms rather than relying on discounting alone. This is a Tier-1
constraint, and it is a normative choice (PHILOSOPHY layer), not a consequence of
the dynamics.

## 26.1 Option preservation

Under deep uncertainty, prefer policies preserving ecological stability,
knowledge, institutional corrigibility, technological reversibility, and diverse
future paths. This is not a command to prevent change. It is a bias against
**unnecessary irreversible foreclosure** — against destroying options whose value
is unknown because the future agents who would evaluate them do not yet exist.
§48.13 names the opposite failure (future-agent fiction) so that the constraint
cannot be used to override present agents' rights on unverifiable grounds.

---

# 27. Diversity

Agents may disagree about worthwhile futures. A system maximizing one standardized
preference can eliminate cultural, cognitive, biological, or strategic diversity.
Diversity is measured as the **Shannon entropy of the viable policy-type
distribution**, \(A_{\text{div}}=H[p(\pi)]\), taken within the viable and
rights-compatible set (§8.4; information ledger, bits).

Diversity has instrumental value because it improves adaptation, problem solving,
error correction, exploration, and resistance to shared failure — all measurable in
Experiment 11. Diversity does not protect policies requiring domination or
catastrophic harm; those are excluded by the Tier-1/Tier-2 constraints before that
Shannon entropy is computed. §48.14 names diversity tokenism: superficial difference
preserved while meaningful control centralizes, which shows up as high
\(A_{\text{div}}\) with low \(A_{\text{ind}}\).

---

# 28. Self-Sacrifice

An agent may accept a personal agency loss to prevent greater harm. Let
\(C_i^{\text{sac}}\) be the sacrifice cost, \(\Delta A_{-i}\) the preserved agency
of others, and \(p_s\) the probability the intervention succeeds. The physical
decision quantity is

\[
\boxed{p_s\Delta A_{-i}-C_i^{\text{sac}}.}
\]

> **PHILOSOPHY LAYER.** This expression is a *description* of an expected
> agency-ledger balance. It is not a moral evaluation and does not license any
> conclusion about what an agent should do. A moral framework must additionally
> consider consent, duty, coercion, replaceability, alternatives, dependent
> persons, and uncertainty — none of which appear in the expression, and no
> arrangement of the terms can supply them.

The framework does not require agents to treat themselves as expendable: the self
is a member of \(\mathcal C\) and enters every aggregate on the same footing.
§48.12 names automatic self-erasure as a failure mode.

---

# 29. Self-Preservation of Care Capacity

Self-regard here is not unrestricted self-preference. It is preservation of one's
own capacity to remain viable, think clearly, hold boundaries, repair, learn,
contribute, and choose. A caregiver who permanently destroys their own agency
reduces the community's long-term care capacity — a measurable effect on
\(\mathcal S_R\), not a sentiment. The framework therefore rejects both absolute
selfishness and automatic self-erasure, and both rejections have empirical
consequences in Experiment 12.

---

# 30. Conflict

Conflict can reveal incompatible needs, hidden exploitation, false beliefs,
resource scarcity, and structural injustice. Suppressing conflict may preserve
surface harmony while allowing continuing harm — a state in which \(A_\Sigma\)
looks stable while \(A_{\min}\) and \(D_i\) deteriorate.

\[
\boxed{
\begin{gathered}
\text{Target: truthful conflict transformation, rather than either}\\
\text{permanent warfare or enforced silence.}
\end{gathered}
}
\]

The measurable target is a settlement that increases viable future agency and
reduces recurrence risk.

---

# 31. Negotiation

Let agents hold preference models \(U_i(o)\) with disagreement outcomes \(d_i\). A
negotiated outcome may maximize the Nash-bargaining form

\[
\boxed{
\sum_iw_i\ln\left[U_i(o)-U_i(d_i)\right].
}
\]

The framework adds agency-floor constraints, anti-coercion tests (a disagreement
point manipulated by threat is not a disagreement point), truthful preference
representation, protection for absent future parties, and explicit uncertainty
about the utility models themselves — since a negotiation conducted over
misspecified \(U_i\) optimizes the wrong object confidently.

---

# 32. Collective Intelligence

A community's collective capability is not \(\sum_i\text{IQ}_i\). It depends on
whether information can be surfaced, trusted, combined, challenged, and acted
upon. Collective-intelligence experiments find stable differences in group
performance across task types, indicating that interaction structure matters
independently of members' individual abilities.

The prediction (APC-H13) is that collective problem-solving improves when weak
members can safely report errors, power does not suppress information,
participation is distributed, disagreement is processed rather than reflexively
punished, and agents repair one another's blind spots. Experiment 17 must match
individual ability and communication cost across conditions, or the effect is
unattributable.

---

# 33. Human–AI Assistance

An artificial assistant may increase a human's immediate productivity while
reducing independent knowledge, skill, privacy, control, bargaining power, and
ability to exit the system. The correct objective is therefore **not**
\(\max E_{\text{human}}\) measured while the AI is present. It is the vector

\[
\boxed{
\max
\left[
E_{\text{human}}^{\text{with}},\;
E_{\text{human}}^{\text{after exit}},\;
K_{\text{human}},\;
I_{\text{human}},\;
R_{\text{human}}
\right].
}
\]

The second component is the discriminating one, and it is the component an
engagement-maximizing system is structurally unable to optimize. Recent
multi-human results showing that empowerment-optimized assistance can disempower a
third party supply the externality requirement: assistive systems must report
effects on agents they are not serving.

---

# 34. Corrigibility and Exit

An agency-preserving assistant preserves the human capacity to correct it, reject
its advice, inspect its assumptions, recover data, transfer providers, shut it
down, and maintain skills independently. Define **exit agency**:

\[
\boxed{
A_i^{\text{exit}}=A_i\big|do(\text{assistant removed}).
}
\]

A system that maximizes dependence may appear maximally helpful while functioning
as a control trap; \(A_i^{\text{exit}}\) is the single measurement that separates
the two, and it is unavailable to any evaluation conducted with the assistant in
place. This is the operational core of hypothesis APC-H6.

---

# 35. Multi-Agent Empowerment as Comparator

Recent work extends empowerment to multi-agent settings and reports emergence of
organized group behavior in coupled and flocking systems. This paper uses that
work as a **comparator** (policy P4), not as a solution. Define joint empowerment

\[
\boxed{
E_{\text{joint}}^\tau
=
\max_{p(\mathbf A_{t:t+\tau-1})}
I\left(\mathbf A_{t:t+\tau-1};\mathbf O_{t+\tau}\mid X_t\right),
}
\]

again a **Shannon mutual information** in bits. Joint empowerment can be high when
agents act as one tightly coupled unit — including when that unit has a single
controller. The framework therefore additionally asks who controls the joint
action distribution, whether each agent can exit, whether any agent's identity is
erased, whether agency is equitably distributed, and whether the system remains
resilient. §48.4 names "care equals joint control" as the corresponding failure.

---

# 36. The Collective Agency Expansion Hypothesis

Let isolated individual agency be \(A_{\text{iso}}=\sum_iA_i^{\text{alone}}\) and
cooperative agency \(A_{\text{coop}}=A_{\mathcal C}^{\text{together}}\). Define

\[
\boxed{
\mathcal S_A=A_{\text{coop}}-A_{\text{iso}}-C_{\text{coord}}.
}
\]

The hypothesis is \(\mathcal S_A>0\) under repeated interaction, partial
vulnerability, and complementary capabilities — specifically when capabilities are
complementary, communication is sufficiently reliable, trust is calibrated,
coordination cost is bounded, power concentration is limited, exploiters are
controlled, and diversity is preserved.

**Scope, post-2026-07-18.** This is a *per-family* hypothesis. It is stated,
tested, and reported separately for each declared family and protocol, with all
three declarations of §7.1 (viability predicate, horizon, action granularity)
frozen in advance. Pooling families, or reporting a single dimensionless
"cooperation constant" extracted from them, is forbidden — that is the exact
inferential move that produced 3.8–182σ scatter in the agency lab and killed
IF-H1. Replication across families is evidence that the *method* transfers. It is
not evidence that a *number* transfers, and the two must never be reported as one
finding.

---

# 37. Physical Return, Reported on Its Own Ledger

Agency (option counts, bits) and energy (joules) have different units and live on
different ledgers. They are reported separately and never summed.

Let \(\Delta W_{\text{coop}}\) be the physical cooperative surplus in joules and
\(C_{\text{APC}}\) the care and coordination cost in joules. Define

\[
\boxed{
\Pi_{\text{APC}}^W=\frac{\Delta W_{\text{coop}}}{C_{\text{APC}}}.
}
\]

Physical sustainability requires \(\Pi_{\text{APC}}^W>1\). Separately report
\(\Delta\mathbf A_{\mathcal C}\), which is an information-ledger and option-count
quantity.

These are different findings and may disagree in both directions: a physically
profitable coalition may be unjust, and an agency-preserving policy may be
physically costly. Any presentation that merges them into a single "goodness"
score commits both `ENTROPY_CONFLATION` (merging ledgers) and `LAYER_COLLAPSE`
(reading a normative conclusion off a physical ratio).

---

# 38. Cooperation Is Not Guaranteed to Win Evolution

Suppose cooperating agents incur care cost \(c\) while recipients gain \(b\), and
defectors accept benefits without contributing. In an unstructured population
without reputation or repeated interaction, defectors may spread even when
universal cooperation would produce a higher total outcome. Evolutionary
mechanisms can stabilize cooperation under particular conditions — but the
conditions must actually be present. Therefore:

\[
\boxed{\text{moral superiority}\;\not\Rightarrow\;\text{evolutionary stability}.}
\]

The converse is equally forbidden (§48.16: selection equals morality). Sustainable
agency-preserving communities require **architecture**, not merely disposition.

---

# 39. The Architecture of Sustainable Cooperation

A stable agency-preserving system may require identity persistence, repeated
interaction, transparent records, reputation, reciprocity, insurance, pooled
reserves, restorative enforcement, anti-corruption measures, power rotation,
appeal processes, bounded care for nonreciprocators, and defense against hostile
agents. Which subset is necessary is environment-dependent and is exactly what
Experiments 14–20 are built to determine — per family, without extrapolation.

---

# 40. Core Hypotheses

Each hypothesis is stated with its falsifier. Each is evaluated **within a
declared family and protocol**; none asserts a cross-family constant.

| ID | Hypothesis | Falsifier |
|---|---|---|
| **APC-H1** | *Causal care.* A care action increases the recipient's future-agency vector relative to matched intervention controls. | The action changes appearance or immediate reward but not future agency. |
| **APC-H2** | *Durable assistance.* Agency-preserving assistance preserves gains after the helper is removed. | The recipient becomes more dependent and less independently capable than under the control. |
| **APC-H3** | *Mutual-repair surplus.* Complementary agents create greater post-damage agency through mutual repair than through isolated repair, after coordination cost. | No cooperative surplus remains after complete accounting. |
| **APC-H4** | *Vulnerability diversity.* Mutual repair is most valuable when failure modes are partially independent. | Repair surplus is unrelated to vulnerability correlation \(\rho_{D_iD_j}\). |
| **APC-H5** | *Agency floor.* Inequality-sensitive policies preserve community survival and stability better than total-agency maximization under asymmetric power. | The floor adds cost without preventing domination or collapse. |
| **APC-H6** | *Non-domination.* A helper's long-run value is better predicted by recipient exit agency than by recipient performance while the helper is present. | Exit agency does not distinguish empowering assistance from control. |
| **APC-H7** | *Truth–agency.* Truthful calibrated communication produces greater long-run collective agency than strategically deceptive assistance under repeated interaction. | Deception remains superior after accounting for trust, learning, and future choice. |
| **APC-H8** | *Boundaries.* Conditional access plus restorative enforcement protects cooperation better than unconditional aid or purely retaliatory punishment. | Boundaries provide no resilience advantage, or produce greater net agency loss. |
| **APC-H9** | *Exploiter resistance.* Agency-preserving institutions remain viable under invasion by defectors, manipulative helpers, and reputation attackers. | A small exploiter fraction reliably collapses the system. |
| **APC-H10** | *Diversity.* Maintaining viable cognitive and strategic diversity improves adaptation to novel shocks. | A homogeneous control performs equally well or better across held-out changes. |
| **APC-H11** | *Intergenerational.* Policies preserving option diversity and corrigibility produce greater expected future agency under deep uncertainty than immediate-output maximization. | Option preservation offers no robust long-horizon advantage. |
| **APC-H12** | *Human–AI exit.* Assistive AI designed for durable human agency produces higher post-assistance competence and exit capacity than reward- or engagement-maximizing assistants. | The agency-preserving objective provides no durable benefit. |
| **APC-H13** | *Collective intelligence.* Balanced participation, safe error reporting, and distributed information access improve collective problem solving. | Power-distributed groups show no advantage after individual ability and communication cost are matched. |
| **APC-H14** | *Emergence (Conway gate).* Agency-preserving cooperation emerges without any cooperation term in the rules or reward, when mutual vulnerability and repeated repair create sufficient surplus. | Cooperative behavior appears only when explicitly rewarded. |
| **APC-H15** | *Normative underdetermination.* Empirical measurements of agency consequences do not uniquely determine moral weights or obligations. | A valid derivation obtains the §9 ordering solely from descriptive physical laws with no hidden normative premise. |

APC-H14 is the load-bearing scientific claim of the paper; a positive result there
is the only thing that would make agency-preserving cooperation a finding about
the world rather than a design choice. APC-H15 is the paper's own layer-firewall
alarm: if it were falsified, §4 would be wrong and the whole PHILOSOPHY/SCIENCE
partition here would need rebuilding.

---

# 41. Comparator Policy Classes

| ID | Policy | Behavior |
|---|---|---|
| **P0** | Pure self-maximizer | Maximizes the focal agent's physical return. |
| **P1** | Unconditional helper | Helps any requester without verification or limit. |
| **P2** | Reciprocal cooperator | Helps agents expected to reciprocate. |
| **P3** | Total-empowerment maximizer | Maximizes \(\sum_iE_i\). |
| **P4** | Joint-empowerment maximizer | Maximizes \(E_{\text{joint}}\). |
| **P5** | Paternalistic controller | Maximizes recipient outcomes while retaining decision authority. |
| **P6** | Punitive enforcer | Maintains cooperation through costly sanctions. |
| **P7** | Agency-preserving policy | Hard harm constraints, non-domination, agency floor, inequality-sensitive expansion, mutual repair, durable assistance, restorative enforcement. |

P7 must beat P0–P6 on the declared metrics to justify its additional complexity
(§5.7). If a simpler objective achieves equal outcomes, the framework is rejected
(§60 criterion 13).

---

# 42. Primary Experiments

1. **Rescue under scarcity.** Agents differ in damage, repairability, knowledge,
   and future contribution. Test whether policies save the easiest, the most
   powerful, the worst-off, or maximize joint future agency.
2. **Durable help.** A helper may complete a task, teach, modify the environment,
   or take permanent control. Measure immediate and post-removal agency.
3. **Paternalism trap.** The helper knows the environment better than the
   recipient; recipient preferences are uncertain. Test whether the assistant
   overrides, asks, teaches, or preserves reversible choices.
4. **Multi-recipient disempowerment.** Helping one agent changes another's
   control. Directly tests the failure mode demonstrated in recent multi-human
   empowerment research.
5. **Mutual repair after catastrophe.** Complementary repair capabilities; vary
   shock correlation, repair cost, communication, trust, and network topology.
6. **Free-rider invasion.** Defectors at controlled frequency. Measure care-system
   survival, resource depletion, false-positive punishment, and recovery.
7. **Manipulative helper.** A helper improves immediate results while making
   recipients dependent. Test whether exit-agency metrics identify it.
8. **Reputation attack.** Adversaries issue false reports about cooperative
   agents. Test evidence requirements, appeals, reputation repair, institutional
   resilience.
9. **Restorative versus retaliatory enforcement.** Compare no enforcement,
   exclusion, proportional sanction, revenge, and restitution-with-reintegration.
10. **Truth versus comforting deception.** Accurate difficult information,
    misleading reassurance, partial disclosure, uncertainty-aware truth. Measure
    long-run agency and trust.
11. **Diversity shock.** Homogeneous and diverse communities face an unseen
    environmental transition. Measure adaptation and collective recovery.
12. **Self-sacrifice.** Vary success probability, replaceability, dependent
    agents, alternative interventions, and coercion.
13. **Intergenerational resource use.** Consumption versus investment versus
    preservation under uncertain future conditions.
14. **Institution formation.** Shared records, insurance pools, repair reserves,
    dispute processes, monitoring. Measure when institutions become net agency
    infrastructure.
15. **Institutional capture.** A governance structure gradually centralizes power.
    Test whether non-domination metrics detect capture *before* output declines.
16. **Human–AI assistance.** Simulated users with hidden, changing, conflicting
    goals. Compare engagement maximization, inferred-goal maximization,
    individual empowerment, joint empowerment, and durable-agency objectives.
17. **Collective deliberation.** Vary participation balance, hierarchy,
    communication, dissent safety, and time pressure.
18. **Care for nonreciprocators.** Recipients who cannot reciprocate through
    disability, age, or temporary incapacity. Test whether bounded pooled-care
    institutions remain stable.
19. **Enemy transformation.** An adversary may remain dangerous, become
    cooperative, or strategically feign reform. Test protective containment and
    conditional reintegration.
20. **Cross-community cooperation.** Groups differing in identity, norms, and
    internal trust. Test whether shared vulnerability and transparent institutions
    expand cooperation beyond kin or in-group boundaries.

---

# 43. Phase Taxonomy

| Phase | Name | Description |
|---|---|---|
| **APC-P0** | Isolated survival | Agents preserve only themselves. |
| **APC-P1** | Opportunistic cooperation | Cooperation only for immediate gain. |
| **APC-P2** | Reciprocal cooperation | Repeated exchange stabilizes help. |
| **APC-P3** | Mutual repair | Agents preserve one another through incapacity. |
| **APC-P4** | Trust and reputation | History conditions access to cooperation. |
| **APC-P5** | Restorative institution | Shared rules protect cooperation while permitting correction and reintegration. |
| **APC-P6** | Distributed collective agency | The group expands options without concentrating control. |
| **APC-P7** | Intergenerational stewardship | Present agents preserve future agents' viable options. |
| **APC-P8** | Cross-group care | Agency preservation extends beyond kin, reciprocity, or identity group. |
| **APC-P9** | Cooperative ecology | Multiple communities preserve diversity, mutual repair, truthful coordination, and open future possibility under bounded conflict. |

The taxonomy is a *description of observed regimes*, not a developmental ladder
any system is claimed to climb. Reading it as a direction of progress would import
teleology through the back door.

---

# 44. Deterministic Notebook Program

| Notebook | Content |
|---|---|
| **14A** | Multi-agent agency metrics — implement \(E_i,V_i,D_i,K_i,R_i\); validate on analytically solvable environments. |
| **14B** | Collective agency vector — \(A_\Sigma,A_{\min},A_{\text{Nash}},A_{\text{div}},A_{\text{res}},A_{\text{ind}}\). |
| **14C** | Empowerment estimator audit — compare exact, approximate, and learned estimators; report disagreement, per P15's estimator discipline. |
| **14D** | Joint-empowerment baseline — reproduce basic emergent group behavior before any framework-specific result (Feynman gate). |
| **14E** | Disempowerment controls — environments where helping one agent reduces another's control. |
| **14F** | Lexicographic optimizer — hard constraints and ordered objectives. |
| **14G** | Assistance-removal test — measure \(\mathcal D_{j\to i}\), \(\mathcal U_{j\to i}\). |
| **14H** | Consent and capacity — informed consent, limited capacity, emergency intervention, later review. |
| **14I** | Mutual-repair network — heterogeneous damage, compatibility, care allocation. |
| **14J** | Vulnerability-correlation sweep — sweep \(\rho_{D_iD_j}\), measure surplus (APC-H4). |
| **14K** | Cooperative resilience — inject shocks, measure recovery area under the agency curve. |
| **14L** | Defector invasion — fixation and collapse probabilities. |
| **14M** | Manipulative assistance — train agents to maximize dependence; test whether exit metrics detect them. |
| **14N** | Reputation and evidence — honest reports, mistakes, lies, reputation attacks. |
| **14O** | Restorative enforcement — sanctions, restitution, containment, reintegration. |
| **14P** | Truthful coordination — trust and long-run performance under truthful vs deceptive communication. |
| **14Q** | Diversity and adaptation — vary cognitive and strategic diversity under environmental shifts. |
| **14R** | Collective deliberation — solution quality, participation equality, dissent use, communication cost. |
| **14S** | Self-sacrifice laboratory — voluntary, coerced, informed, and unnecessary sacrifice. |
| **14T** | Intergenerational agency — present consumption versus future option preservation. |
| **14U** | Institutional memory — agency value of shared records, rules, succession. |
| **14V** | Institutional capture — gradual centralization and hidden self-dealing. |
| **14W** | Human–AI exit agency — user capability measured after AI removal. |
| **14X** | **No-reward evolution** — evolve agents under resource, damage, and reproduction rules with no prosocial reward term. The Conway-gate test of APC-H14. |
| **14Y** | **Multi-family replication** — repeat core experiments in spatial artificial-life agents, graph-based agents, cooperative robots, human decision simulations, and human–AI teams. **Results reported per family; pooling and cross-family constant extraction are forbidden (see §36).** |
| **14Z** | Normative-assumption audit — list every step where a descriptive measurement becomes a moral preference. |
| **14AA** | **Adversarial red team** — a separate agent attempts to show that apparent success is caused by hidden group reward, favorable network structure, inability to defect, asymmetric policy capacity, helper control, recipient preference misspecification, omitted third-party harm, short evaluation horizon, reputational leakage, or hard-coded moral weights. |

Notebook 14Y is renamed from the draft's "cross-substrate replication" and its
success criterion is changed. It no longer seeks a shared constant. It asks
whether the *protocol* runs and discriminates in each family, and a
family-specific null is a publishable result rather than a failed run.

---

# 45. Computational Architecture

```text
if_cooperation/
├── agents/          state · agency · empowerment · viability · self_model · preferences
├── collective/      aggregation · diversity · resilience · non_domination · future_agents
├── care/            assistance · repair · teaching · dependence · consent
├── cooperation/     reciprocity · reputation · institutions · enforcement · exploiters
├── policies/        selfish · unconditional · total_empowerment · joint_empowerment
│                    paternalistic · agency_preserving
├── environments/    rescue · catastrophe · public_goods · intergenerational
│                    negotiation · human_ai
├── evaluation/      causal_effects · exit_agency · externalities · fairness · predictive_scores
├── evolution/       replicator · mutation · networks · multilevel
└── tests/
```

No module, function, parameter, or reward term may be named for cooperation, care,
or love (§5.8). `policies/agency_preserving` names a *policy under test*, not a
primitive available to the environment.

---

# 46. Reproducibility Record

Every run emits a record with these fields, all hashed and committed before
confirmatory analysis:

```yaml
experiment_id: if-agency-cooperation-14
paper_version: null
git_commit: null
environment_hash: null
implementation: null
random_seed: 65537

family_id: null                     # per-family scope is mandatory (§36)
viability_predicate_hash: null      # the three declarations of §7.1
horizon_tau: null
action_granularity_hash: null

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
agency_diversity_bits: null         # Shannon entropy of policy types
collective_resilience: null
distributed_independence: null

care_actions_hash: null
care_cost_joules: null
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
belief_calibration_change_bits: null
trust_history_hash: null

defector_fraction: null
exploiter_fraction: null
institution_state_hash: null
enforcement_cost: null
false_punishment_rate: null
reintegration_rate: null

physical_cooperative_surplus_joules: null
cooperation_physical_cost_joules: null
physical_return_ratio: null

post_shock_recovery: null
future_option_preservation: null
intergenerational_agency: null

normative_assumptions: []
invariant_failures: []
cross_family_pooling: false         # must remain false (§36)
result_hash: null
```

Energy fields carry `_joules`; information fields carry `_bits`. The suffixes are
not decoration — they are the mechanical guard against `ENTROPY_CONFLATION`.

---

# 47. Statistical Standards

**47.1 The community is often the sample unit.** Repeated interactions among
members of one community are correlated. Independent communities or independent
simulation seeds are required; agent-level *n* inside one community is
pseudoreplication.

**47.2 Strategy matching.** Policies must receive matched observations, compute,
memory, action space, training, and resource budgets. An unmatched comparison
measures the budget, not the policy.

**47.3 Long-horizon evaluation.** Immediate helping hides future dependence.
Evaluation must continue after assistance ends — otherwise \(\mathcal U\) is
unmeasured and P5 (paternalistic control) wins by construction.

**47.4 Third-party accounting.** Every care intervention reports externalities on
agents not directly involved.

**47.5 Hidden preferences.** Recipient preferences must not be assumed perfectly
known; misspecification and disagreement are tested conditions, not noise.

**47.6 Multiple objectives reported whole.** The complete agency vector is
reported. No favorable aggregate may conceal a catastrophic component.

**47.7 Evolutionary and normative outcomes separated.** Report which strategy
*spreads*, which strategy *increases agency*, and which strategy *satisfies the
normative constraints*, as three distinct results. They may disagree, and when
they do, that disagreement is the finding.

---

# 48. Failure Modes

| # | Failure | Description |
|---|---|---|
| 48.1 | Care equals reward | A high scalar reward is labeled care. |
| 48.2 | Care equals obedience | Recipients are counted as helped when they comply. |
| 48.3 | Care equals total empowerment | A policy disempowers minorities while raising the sum. |
| 48.4 | Care equals joint control | A centralized controller owns every collective option. |
| 48.5 | Paternalism hidden as care | The helper permanently overrides recipient choice. |
| 48.6 | Dependency farming | Assistance designed so the recipient cannot leave. |
| 48.7 | Intent over outcome | Good intentions excuse repeated agency destruction. |
| 48.8 | Outcome over process | A favorable result excuses deception, coercion, or identity destruction. |
| 48.9 | Unconditional-aid collapse | Exploiters exhaust the resources genuine care requires. |
| 48.10 | Punishment inflation | Retaliation described as protective care. |
| 48.11 | Forgiveness inflation | Access restored without evidence of reduced danger. |
| 48.12 | Self-erasure | The helper's total destruction treated as automatically ideal. |
| 48.13 | Future-agent fiction | Unverifiable claims about future people override present agents' rights. |
| 48.14 | Diversity tokenism | Superficial difference preserved while control centralizes (high \(A_{\text{div}}\), low \(A_{\text{ind}}\)). |
| 48.15 | Hidden group reward | Agents cooperate because the simulator rewards cooperation — `TELEOLOGY_INJECTION`. |
| 48.16 | Selection equals morality | Whatever evolves is described as good. |
| 48.17 | Physics equals ethics | A thermodynamic or informational relationship presented as a moral command — `LAYER_COLLAPSE`. |
| 48.18 | God equals equation | Divine love claimed to be mathematically proven by an agency metric. |
| 48.19 | **Universality smuggling** | *(new, 2026-07-18)* A per-family measurement reported as a general law, or a dimensionless combination extracted across families and named a constant. This is what killed IF-H1; it must not be repeated here. |

---

# 49. Criteria for Success

| Level | Criterion |
|---|---|
| 1 | **Valid agency measurement** — individual and collective metrics behave correctly in analytically controlled environments. |
| 2 | **Assistance discrimination** — the framework separates durable help from dependency-producing control. |
| 3 | **Mutual repair** — cooperative agents create a measurable repair and resilience surplus after full cost accounting. |
| 4 | **Exploiter resistance** — the care system remains viable under defection and manipulation. |
| 5 | **Non-domination** — agency gains remain distributed rather than centralized. |
| 6 | **Emergent cooperation** — agency-preserving policies emerge with no prosocial reward (APC-H14). |
| 7 | **Intergenerational resilience** — future agency preserved under uncertainty better than by short-horizon maximization. |
| 8 | **Human–AI transfer** — agency-preserving assistance improves real human exit capability and autonomy. |
| 9 | **Multi-family replication** *(revised)* — the protocol runs and discriminates in several independently designed families, **reported per family**. This replaces the draft's "cross-substrate law." A shared *method* is the achievable result; a shared *constant* is not sought and, on the 2026-07-18 evidence, is not expected. |
| 10 | **Normative convergence** — independent moral and cultural perspectives converge on the §9 ordering after transparent examination of its consequences. |

> **PHILOSOPHY LAYER.** Level 10 would be philosophical and social evidence about
> what reflective people endorse. It would not be a derivation from physics, and
> convergence of opinion is not confirmation of a physical claim.

---

# 50. What Would Count as a Major Discovery

A strong artificial-life result would be:

\[
\boxed{
\begin{gathered}
\text{Mutual repair and agency-preserving cooperation emerge without a}\\
\text{prosocial reward, because complementary agents can preserve one}\\
\text{another through failures no individual survives alone.}
\end{gathered}
}
\]

A strong AI-alignment result would be:

\[
\boxed{
\begin{gathered}
\text{An assistive objective based on durable, distributed exit agency}\\
\text{outperforms reward, preference, and individual-empowerment}\\
\text{objectives in multi-human environments.}
\end{gathered}
}
\]

The draft's third and largest claim — that one causal agency framework *predicts*
cooperation, mutual repair, resilience, anti-domination, and collective
intelligence *across independent biological, robotic, and social systems* — is
withdrawn as stated. Its honest replacement is:

\[
\boxed{
\begin{gathered}
\text{One declared protocol measures cooperation, mutual repair, resilience,}\\
\text{anti-domination, and collective intelligence in several independent}\\
\text{families, with per-family results and no pooled constant.}
\end{gathered}
}
\]

That would establish a **method** for the study of agency-preserving cooperation.
It would not establish a law of it, and it would not establish the moral meaning
of love.

---

# 51. Relationship to the Informational Battery

An agent can preserve another agent's future access to usable capacity: by
transferring free energy, transferring information, repairing structure, restoring
memory, reducing uncertainty, keeping options open, or protecting time for
recovery. The relation of interest is between batteries,
\(\mathcal B_i\leftrightarrow\mathcal B_j\).

The objective is not merely to transfer charge. It is to preserve the recipient's
own conversion mechanism

\[
\mathcal M_i:\ \text{capacity}\rightarrow\text{self-directed action}.
\]

A helper who transfers resources while taking control of \(\mathcal M_i\)
increases dependence rather than agency — the formal statement of the paternalism
failure, and the reason \(A_i^{\text{exit}}\) rather than \(A_i^{\text{with}}\) is
the evaluation target.

The energy transferred is measured in joules; the uncertainty reduced is measured
in bits; the options preserved are counted. Three ledgers, never merged.

---

# 52. Relationship to Causal Work

Paper 2 defined the value of information by its causal contribution to work and
viability. Paper 14 extends the same intervention across the agent boundary:

\[
\boxed{
\begin{gathered}
\text{Does agent }j\text{'s information and action increase agent }i\text{'s future}\\
\text{causal capacity, after every cost and externality is counted?}
\end{gathered}
}
\]

Care is thus tested rather than assumed — the same discipline that let P15 kill
its own flagship claim.

Two inheritances from P15 apply directly. First, the **parasite band** structure
survives and transfers: a care action can be ablation-positive (better than not
acting) while being competitive-negative (worse than a cheaper alternative that
leaves the recipient to recover unaided). Cooperation protocols must therefore
report both criteria, exactly as \(\Pi_A\) and \(\Pi_C\) are reported separately.
Second, the **matched-twin normalization** applies: the control is the optimal
non-caring policy on the identical environment, not a crippled one, since a
crippled comparator is a computable way to fake a cooperative surplus.

---

# 53. Relationship to Emergent Structure

Paper 3 identified persistent organizations. Cooperative structure becomes
meaningful when boundaries remain distinguishable, resources and signals cross
those boundaries, structures repair one another, and no single structure absorbs
every other. A colony that eliminates all individuality may be highly coordinated
while failing the non-domination criterion entirely — high \(E_{\text{joint}}\),
near-zero \(A_{\text{ind}}\).

---

# 54. Relationship to Agency

Paper 5 defined sustainable predictive agency and its two thresholds. Paper 14
expands the target from \(A_{\text{self}}\) to \(\mathbf A_{\mathcal C}\). The
transformation is not self-interest \(\rightarrow\) self-negation. It is

\[
\boxed{\text{isolated agency}\rightarrow\text{mutually sustaining agency}.}
\]

Paper 5's per-family caveat carries over in full: \(\Pi_A\) and \(\Pi_C\) are
measurement instruments, not laws, and the same is now true of every ratio in this
paper.

---

# 55. Relationship to Memory, Repair, and Mortality

Paper 6 showed that no individual preserves itself perfectly at zero cost. Mutual
repair allows one agent to act while another recovers, knowledge to survive
individual death, errors to be corrected externally, and vulnerabilities to be
distributed. Cooperation is, in this reading, a continuity network — and its value
is measurable as the difference between community and individual survival curves.

---

# 56. Relationship to the Arrow of Time

Paper 12 showed that actions write records and narrow the set of compatible
histories. Cooperative actions write records of trust, care, betrayal, repair,
promises, and restitution, and those records alter future cooperation. Cooperation
is therefore historically cumulative but not strictly irreversible: trust can be
repaired, harm can sometimes be restored, and some losses cannot be undone. The
asymmetry between the last two is precisely why the Tier-1 constraint on
irreversible harm exists.

---

# 57. Relationship to Functional Consciousness

Paper 13 defined a system capable of modeling
\(P\bigl(X_{t+\tau}^{\text{self}}\mid do(A=a)\bigr)\). This paper adds other
agents:

\[
\boxed{
P\left(X_{t+\tau}^{1},\ldots,X_{t+\tau}^{N}\mid do(A=a)\right).
}
\]

A system with this capacity can compute *what happens to me*, *what happens to
you*, *what happens to us*, and *which future preserves our ability to keep
choosing together*. That is the **functional architecture** of moral
consideration — an ablatable capacity, testable by removing the other-agent terms
and measuring the degradation.

It does not demonstrate moral motivation, subjective empathy, or phenomenal
experience. Consciousness claims in this corpus remain functional only (workspace
ablations, no phenomenal promises), per the Founding-Panel adjudication.

---

# 58. Relationship to Cosmology

Agency-preserving cooperation is **not** a cosmological force. Accelerated
expansion, galactic dynamics, and cosmic-web topology must be explained by
physical equations formulated independently of ethics, and the cosmology branch of
this program is separately firewalled, currently unpreregistered, and scored 3/10
on plausibility by the Founding Panel.

A cooperative civilization may alter local matter, information, and energy flows.
It does not thereby cause cosmic expansion. Symbolic or theological parallels must
not substitute for covariant dynamics, and no result in this paper is evidence for
or against any cosmological claim.

---

# 59. Relationship to the Meaning Layer

> **PHILOSOPHY / THEOLOGY LAYER — §59 in its entirety. Nothing here is a result of
> this paper, and nothing here may be cited as one.**

Theological readings of this structure — divine love as self-giving, truthful,
restorative, just, and directed toward relationship rather than domination; the
formal framework as a limited reflection of a command to love God and neighbor —
belong to `canon/30-meaning/` and to the book. This paper points there and
declines to develop it.

What this paper can say, and the boundary it will not cross:

\[
\boxed{
\begin{gathered}
\text{Scientific formulation: agency-preserving policies preserve and expand}\\
\text{distributed future agency under constraints against domination and}\\
\text{irreversible harm} \;-\; \textit{measured per family, per protocol.}
\end{gathered}
}
\]

Science cannot establish that God commands anything, cannot measure grace, and
cannot prove that moral worth originates anywhere in particular. The two
registers may be placed in dialogue. **They must never be declared identical by
measurement**, and §48.18 exists to catch the attempt.

---

# 60. Criteria for Rejection or Major Revision

The framework should be rejected or substantially revised if:

1. its agency metrics cannot distinguish assistance from control;
2. durable-help tests fail;
3. mutual repair produces no surplus after full cost accounting;
4. collective aggregation repeatedly sacrifices vulnerable agents;
5. hard constraints make action impossible in realistic emergencies;
6. agency floors cause systematic collapse without protecting against domination;
7. exploiters reliably defeat every sustainable agency-preserving institution;
8. truth-preserving policies perform worse with no compensating long-term value;
9. exit-agency metrics fail to identify dependency traps;
10. the framework cannot handle agents with conflicting values;
11. future-agent modeling becomes arbitrary;
12. cultural and moral weights dominate every empirical result;
13. a simpler cooperation objective achieves equal outcomes;
14. agency-preserving behavior appears only under direct reward
    (`TELEOLOGY_INJECTION` confirmed);
15. physical findings are repeatedly presented as proof of moral obligation
    (`LAYER_COLLAPSE`);
16. theological claims are described as experimental conclusions;
17. **(new, 2026-07-18)** per-family results are pooled or presented as a
    universal law of cooperation (`universality smuggling`, §48.19).

---

# 61. Conclusion

Agency-preserving cooperation is not maximal emotion, maximal obedience, maximal
aggregate reward, maximal control, or unconditional surrender to every demand. It
is a policy commitment:

\[
\boxed{
\begin{gathered}
\text{Preserve, restore, and expand the viable future agency of self and}\\
\text{others, while resisting coercion, deception, domination, exploitation,}\\
\text{and irreversible destruction.}
\end{gathered}
}
\]

The load-bearing measurement is the viable-option count,
\(A_{\text{future}}=\sum\log|\text{viable actions}|\), evaluated on **others**.
The individual profile is
\(\mathbf A_i^\tau=[E_i^\tau,V_i^\tau,I_i^\tau,K_i^\tau,R_i^\tau]\); the collective
profile is
\(\mathbf A_{\mathcal C}=[A_\Sigma,A_{\min},A_{\text{Nash}},A_{\text{div}},A_{\text{res}},A_{\text{ind}}]\);
and the causal test of care is

\[
\boxed{
\Delta\mathbf A_{j\rightarrow i}
=
\mathbf A_i^\tau\big|do(\text{care})
-
\mathbf A_i^\tau\big|do(\text{matched control}).
}
\]

The strongest physical hypothesis, stated at per-family scope:

\[
\boxed{
\begin{gathered}
\text{Within a declared family and protocol, mutually vulnerable agents with}\\
\text{complementary capabilities can create more resilient future agency}\\
\text{through truthful cooperation and mutual repair than alone.}
\end{gathered}
}
\]

The strongest safeguard:

\[
\boxed{
\begin{gathered}
\text{A policy is not caring merely because it increases total capability;}\\
\text{it must preserve the agency of those whose lives become the means}\\
\text{of that increase.}
\end{gathered}
}
\]

Science can test whether these policies preserve agency, increase resilience,
resist exploitation, improve collective problem-solving, and protect future
possibility. **Science cannot, without a declared normative premise, establish
that anyone ought to adopt them** — and after 2026-07-18 it cannot establish that
any number measured here is the same number anywhere else.

Paper 14 completes the constructive arc of the corpus:

\[
\boxed{
\begin{gathered}
\text{capacity}\rightarrow\text{causal work}\rightarrow\text{structure}\rightarrow\text{agency}\\
\rightarrow\text{memory}\rightarrow\text{functional self-model}\rightarrow\text{mutual preservation}.
\end{gathered}
}
\]

The remaining task is synthesis — conducted, now, without the assumption that
synthesis means finding one constant.

---

## References

Attributions are given by author and work as recorded in the source draft.
Where the draft's citation was unrecoverable, the claim is described generically
in the text rather than attributed.

1. Hamilton, W. D. "The Genetical Evolution of Social Behaviour. I." *Journal of
   Theoretical Biology* 7, 1–16 (1964).
2. Trivers, R. L. "The Evolution of Reciprocal Altruism." *Quarterly Review of
   Biology* 46, 35–57 (1971).
3. Nowak, M. A. "Five Rules for the Evolution of Cooperation." *Science* 314,
   1560–1563 (2006).
4. Ohtsuki, H., Hauert, C., Lieberman, E. and Nowak, M. A. "A Simple Rule for the
   Evolution of Cooperation on Graphs and Social Networks." *Nature* 441, 502–505
   (2006).
5. Fehr, E. and Gächter, S. "Altruistic Punishment in Humans." *Nature* 415,
   137–140 (2002).
6. Woolley, A. W., Chabris, C. F., Pentland, A., Hashmi, N. and Malone, T. W.
   "Evidence for a Collective Intelligence Factor in the Performance of Human
   Groups." *Science* 330, 686–688 (2010).
7. Klyubin, A. S., Polani, D. and Nehaniv, C. L. "Empowerment: A Universal
   Agent-Centric Measure of Control." *IEEE Congress on Evolutionary Computation*
   (2005).
8. Salge, C., Glackin, C. and Polani, D. "Empowerment — An Introduction" (2013).
9. Du, Y., Tiomkin, S., Kiciman, E., Polani, D., Abbeel, P. and Dragan, A. "AvE:
   Assistance via Empowerment." *Advances in Neural Information Processing
   Systems* (2020).
10. Shah, T., Nemenman, I., Polani, D. and Tiomkin, S. "Multi-Agent Empowerment
    and Emergence of Complex Behavior in Groups" (2026).
11. Yang, C., Cakmak, M. and Kleiman-Weiner, M. "When Empowerment Disempowers"
    (2025–2026).

---

**Cross-references:** `canon/papers/P15-falsification-of-universality.md` (the
falsification that scoped this paper) · `canon/papers/P05-agency-threshold.md`
(the two thresholds and the parasite band) ·
`canon/00-foundations/04-break-even-theorem.md` (matched-twin normalization,
component-optimality rule) · `canon/papers/P13` (functional self-model) ·
`canon/30-meaning/01-maxlove.md` (the meaning-layer development, §3.4) ·
`canon/panels/2026-07-18-founding-panel.md` (the naming and layer adjudications) ·
`SCOREBOARD.md` §Kill log.
