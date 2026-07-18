# The Arrow of Time in IF Theory
## Records, Irreversibility, Causal Memory, and the Growth of Historical Constraint

**Author:** Phuc Vinh Truong
**Series:** IF Theory Canonical Papers
**Paper:** 12
**Layer:** SCIENCE
**Status:** canonical revision of extracted draft (2026-07-18)
**Supersedes:** canon/extracted/paper-12-extracted.md

---

> ## Status after 2026-07-18
>
> **This paper sits on the agency branch — the branch where IF-H1 died.** IF-H1 held
> that a dimensionless combination evaluated at competitive break-even is
> substrate-independent. It was falsified across three substrates (ring-lattice
> forager, linear-Gaussian Kalman controller, run-and-tumble chemotactic swimmer)
> and written up in
> [`P15-falsification-of-universality.md`](P15-falsification-of-universality.md).
> It died twice: the information-denominated form
> \(\eta^*=\Delta W_{\mathrm{ablation}}/kT\,\Delta I_{\mathrm{use}}\) proved **not
> measurable family-portably** — three declared estimators of \(I_{\mathrm{use}}\)
> each failed by a distinct pathology (error-keyed estimators *invert*;
> prediction-keyed estimators *presuppose* predictor-shaped agents, reading
> \(\Delta I_{\mathrm{use}}\approx 0\) for chemotaxis despite a >3× work gap;
> outcome-keyed estimators go *negative* because plain mutual information cannot
> carry a sign) — one structural fact in three faces: \(I_{\mathrm{use}}\) is
> representation-relative, definable only against an assumed internal format. The
> cost-denominated form
> \(\Upsilon_{\mathrm{IF}}=\Theta^*C_{\mathrm{model}}/\nu_{\mathrm{active}}\)
> scattered at **3.8–182σ** at every cost level tested. A pre-registered stop rule
> was honored: no third rescaling was attempted.
>
> **Consequence for Paper 12 — applied as an editing pass, not a disclaimer.** The
> extracted draft promised that "the same path, record, and constraint measures
> organize temporal asymmetry across stochastic, cellular, agent, and quantum
> models," and listed cross-substrate collapse as both a success level and a
> notebook deliverable. Every such claim is rewritten here as a per-family,
> per-protocol measurement claim with the substrate named.
> \(\langle\Sigma\rangle\), \(\mathcal A_\Gamma\), \(\mathcal C_H\),
> \(\mathcal R_\delta\), \(\mathcal A_R\), and \(\eta_R\) are **instruments** to be
> re-declared, re-estimated, and re-validated inside each substrate — not
> invariants. Concretely: TA-H12 is demoted to TA-H12′, a methodological-portability
> hypothesis with its own falsifier (§27); Success Level 7 is rewritten (§38);
> Notebook 12S becomes *per-family scaling*, where absence of collapse is an
> expected outcome; §39's "law per unit thermodynamic entropy production" becomes a
> per-architecture regression; §36.8 forbids cross-family pooling; and §37.17 makes
> universality drift a named, checkable failure state.
>
> **What survives and remains available here:** \(\Pi_A\) and \(\Pi_C\) as
> per-family measurement instruments (a *method, not a law*); the **parasite band**
> (ablation-positive ≠ competitive-positive) as a structural consequence of the
> break-even inequality; the **rule/state dissociation** — ablating a model of the
> update law selectively destroys post-switch recovery, 6σ — as a within-family
> causal claim; and the break-even inequality as an accounting identity. Paper 12's
> memory-ablation experiments (§32 Exp. 9, Notebook 12M) inherit exactly this
> status: within-family causal tests whose thresholds are protocol-specific numbers,
> not constants of nature.
>
> **Independent of IF-H1**, this paper never claimed a universal constant, and its
> central negative result — record accumulation does *not* by itself explain a
> global one-sided arrow without an independently derived asymmetric boundary
> condition — is unchanged.

---

## Abstract

Most microscopic dynamical laws permit time-reversed solutions, yet macroscopic
processes display a pronounced direction. Heat flows from hotter to colder systems,
fuels are consumed, organisms age, measurements leave records, and observers
remember what they call the past rather than what they call the future. The
existence of time as a coordinate or ordering parameter must be distinguished from
this **arrow of time**, which concerns asymmetry between two temporal orientations.

This paper proposes an IF framework connecting four quantities:

1. **trajectory irreversibility** — the statistical distinguishability of forward and
   time-reversed histories;
2. **physical records** — stable, accessible states causally correlated with earlier
   events;
3. **causal memory** — compressed internal state that influences future dynamics
   because of historical experience;
4. **historical constraint** — the reduction in the set of prior histories compatible
   with the present record state.

For a trajectory \(\Gamma\) and its reversed counterpart \(\Gamma^\dagger\),
stochastic thermodynamics motivates the trajectory-level irreversibility

\[
\Sigma[\Gamma] = \ln\frac{P_F[\Gamma]}{P_R[\Gamma^\dagger]},
\]

whose expectation is a Kullback–Leibler divergence between path ensembles — a
**Shannon-ledger** quantity that, under the declared stochastic-thermodynamic
conditions, tracks **thermodynamic entropy production** in units of \(k_B\):

\[
\left\langle\Sigma\right\rangle
= D_{\mathrm{KL}}\!\left[P_F(\Gamma)\,\|\,P_R(\Gamma^\dagger)\right]\geq 0.
\]

The expression measures how strongly observed paths reveal a preferred temporal
orientation. It does not imply that microscopic equations contain an intrinsic
direction of time.

A physical record \(R_t\) of event \(X_{t-\tau}\) must be more than statistically
correlated with that event. It must be causally generated by it, persist under a
declared perturbation class, remain physically accessible, and improve retrodiction
beyond the present macrostate alone. The proposed record value is a vector, not a
scalar:

\[
\mathbf R = \left[I_{\mathrm{retro}},\;C_{\mathrm{record}},\;\tau_{\mathrm{life}},
\;\mathcal A_{\mathrm{access}},\;\mathcal R_{\mathrm{redundancy}}\right].
\]

Historical constraint is defined through Bayesian contraction over possible
histories, in the Shannon ledger:

\[
\mathcal C_H(t)
= D_{\mathrm{KL}}\!\left[P(H_{<t}\mid R_t)\,\|\,P(H_{<t})\right]
= I(H_{<t};R_t).
\]

A new record increases historical constraint when \(\Delta\mathcal C_H>0\). This does
*not* mean that total information in the universe necessarily increases. Fine-grained
Shannon information may be redistributed into correlations, records may decay, and
multiple records may be redundant rather than independent.

The central **IF Record-Arrow Hypothesis** is: *in an open nonequilibrium system, the
empirically forward direction is the orientation in which robust causally generated
records and historical constraints are typically produced while total thermodynamic
entropy production is nonnegative.*

The stronger conjecture — that irreversible record accumulation generates the cosmic
arrow without a special low-thermodynamic-entropy boundary condition — is **not
established**. Under reversible microscopic dynamics, ordinary statistical mechanics
still requires an asymmetric boundary condition, a low-thermodynamic-entropy past, or
another independently derived cosmological mechanism to explain why records
overwhelmingly point in one direction. IF Theory must not rename that unresolved
condition "the informational battery" and claim the problem solved.

The paper presents analytical toy models, record interventions, reversible and
stochastic simulations, quantum-record tests, preregistered hypotheses, and a
deterministic notebook program. It also clarifies that agents and choices can create
new local records and redirect existing free energy, but do not thereby create
energy, coordinate time, or the universe's fundamental temporal dimension.

---

## Keywords

Arrow of time; thermodynamic entropy production; physical records; causal memory;
historical constraint; irreversibility; Landauer principle; stochastic
thermodynamics; predictive information; quantum Darwinism; low-thermodynamic-entropy
boundary condition.

---

# 1. Introduction

Physical laws and human experience treat time differently.

In many classical and quantum theories, the microscopic equations admit a trajectory
and an appropriately transformed time-reversed trajectory. Yet ordinary macroscopic
experience is asymmetric: eggs break but do not spontaneously reassemble; fuel burns
but combustion products do not reconstruct fuel; radiation spreads outward from
sources; measurements leave records; organisms retain traces of earlier interactions;
memories concern what we call the past; causes appear before their recorded effects.

These observations are often summarized by the second law of thermodynamics, stated
strictly in the **thermodynamic-entropy** ledger:

\[
\Delta S^{\mathrm{thermo}}_{\mathrm{total}}\geq 0.
\]

But the second law alone does not eliminate the foundational question. If the
microscopic laws are time-reversal symmetric and a system is specified only by a
present low-thermodynamic-entropy macrostate, typical compatible microstates often
evolve toward higher thermodynamic entropy in *both* temporal directions away from
that state. Explaining why our universe exhibits a consistent one-sided thermodynamic
arrow ordinarily invokes a special low-thermodynamic-entropy boundary condition or
another cosmological asymmetry; this is the standard position in the foundational
literature (Goldstein, Tumulka and Zanghì; Price).

Records appear to point in the same direction as thermodynamic entropy production.
Photographs, fossils, memories, radiation fields, impact marks, computer logs, and
environmental correlations carry Shannon information about earlier events. Formal
work on memory systems has shown, however, that correlation with another time does
not by itself create a one-directional epistemic arrow: some memory constructions can
contain information about either temporal direction (Wolpert and Kipper; Mlodinow and
Brun). A physical account must explain why robust, naturally produced records
predominantly concern one side of the present.

IF Theory asks two questions, and takes only the first as tractable here:

> **Q1.** Can the production, persistence, redundancy, and causal use of physical
> records provide an operational account of the arrow experienced by embedded agents?

> **Q2.** Can record formation explain the *origin* of the cosmic arrow, or does it
> merely inherit an arrow established by boundary conditions and nonequilibrium
> resources?

The present paper treats the second possibility — inheritance, not origination — as
the default until the stronger claim is derived.

---

# 2. Scope

Paper 12 addresses statistical irreversibility, thermodynamic entropy production,
record formation, memory, causal history, retrodiction, temporal orientation in
artificial universes, and temporal orientation experienced by embedded agents.

It does **not** claim to derive: time as a fundamental spacetime coordinate; the
relativistic metric signature; the quantum measurement postulate; the initial state
of the universe; the low gravitational entropy of the early universe; metaphysical
passage or the moving present; free will; consciousness; backward causation; or the
creation of energy through choice.

The word **time** refers to a parameter or causal ordering already supplied by the
model. The phrase **arrow of time** refers to measurable asymmetry between
alternative orientations of histories.

**Layer.** SCIENCE throughout. Interpretive material (§45) is flagged and delegated
to `canon/30-meaning/`; it is never asserted as a physical result (`LAYER_COLLAPSE`).

**Universality scope (post-2026-07-18).** Every quantity below is declared **per
family and per protocol**. Where the extracted draft asserted that a measure
"organizes temporal asymmetry across substrates," this revision asserts only that the
same *procedure* can be re-instantiated in another substrate, with its own estimator
validation, and that the resulting numbers are not expected to agree.

---

# 3. Five Concepts That Must Not Be Confused

## 3.1 Temporal parameter

A dynamical model may use \(t\in\mathbb R\) or discrete updates \(t\in\mathbb Z\).
This provides ordering and duration. It does not by itself select a preferred
orientation.

## 3.2 Microscopic time-reversal symmetry

Let \(\Gamma=\{x_t\}_{t=0}^{T}\) be a trajectory and let \(\Theta\) reverse momenta
and other odd-parity variables. The reversed trajectory is

\[
\Gamma^\dagger=\left\{\Theta x_{T-t}\right\}_{t=0}^{T}.
\]

Microscopic reversibility means, approximately, that an allowed forward history
corresponds to an allowed reversed history under the properly reversed dynamics and
protocol. It does **not** mean the two histories have equal probability under the
actual boundary conditions.

## 3.3 Thermodynamic arrow

The thermodynamic arrow is the orientation in which typical macroscopic evolution
produces positive **thermodynamic entropy**, \(\langle\Sigma\rangle>0\). It is
statistical rather than an absolute prohibition against thermodynamic-entropy-reducing
fluctuations.

## 3.4 Record arrow

The record arrow is the orientation in which interactions naturally produce
persistent states containing causally grounded Shannon information about earlier
events.

## 3.5 Psychological or epistemic arrow

An embedded agent's epistemic arrow is the asymmetry between detailed records of what
it calls the past and probabilistic models of what it calls the future. The
psychological arrow may align with the thermodynamic arrow because reliable memory
formation normally occurs in a physical environment already possessing robust
thermodynamic asymmetry. That relationship must be demonstrated rather than assumed.

---

# 4. Prior Art and the Novelty Boundary

## 4.1 Entropy production as path asymmetry

Fluctuation-theorem research relates thermodynamic entropy production to the ratio of
forward-trajectory probability and the probability of the corresponding reversed
trajectory. Crooks derived a general fluctuation relation for microscopically
reversible stochastic dynamics, and later quenched-quantum experiments (Batalhão and
collaborators) directly connected produced thermodynamic entropy to the relative
Shannon entropy between forward and reversed processes.

IF Theory therefore **cannot** claim novelty for

\[
\Sigma[\Gamma]=\ln\frac{P_F[\Gamma]}{P_R[\Gamma^\dagger]}.
\]

The possible IF contribution is to connect path irreversibility with the creation,
survival, accessibility, and causal usefulness of physical records — within a single
declared family at a time.

## 4.2 Physical memory and the epistemic arrow

Formal analyses of memory systems distinguish present states that provide information
about states at other times. They show that some abstract memory systems can be
time-symmetric, while ordinary records require additional physical conditions.
Mlodinow and Brun argued that generic robust memories align with a well-defined
thermodynamic arrow when their state does not require fine-tuning to match the
recorded system.

IF Theory cannot claim novelty for "memory generally points along the thermodynamic
arrow." Its proposed addition is an **intervention-based record criterion** and a
**measurable historical-constraint ledger**.

## 4.3 Information erasure

Landauer-type results connect logically irreversible memory erasure with
thermodynamic cost under specified physical conditions. Experiments using colloidal
and feedback-trap memories (Jun, Gavrilov and Bechhoefer) have measured work
approaching \(k_BT\ln 2\) for quasistatic erasure of one symmetric bit, while
finite-time and nonequilibrium realizations incur additional cost; Sagawa and Ueda
established the minimal-cost accounting for measurement and erasure jointly.

IF Theory cannot claim that writing or erasing records is thermodynamically free. It
must model the physical implementation and distinguish **writing, preserving,
reading, copying, correcting, and erasing** as separate ledger entries. This is the
direct guard against `PERPETUAL_RECHARGE`: no operation in this paper reduces
thermodynamic entropy anywhere without an accounted energy input and an exported
waste-heat term.

## 4.4 Predictive causal states

Computational mechanics (Shalizi and Crutchfield) defines causal states by grouping
histories that yield the same distribution over futures. The resulting
\(\epsilon\)-machine is a minimal predictive representation under the framework's
assumptions, and its statistical complexity measures the Shannon information stored
in those predictive states.

IF Theory cannot claim novelty for minimal predictive state, predictive equivalence
classes, or statistical complexity. Paper 12 instead distinguishes **predictive
memory**, which helps forecast, from **historical records**, which help identify what
occurred.

## 4.5 Environmental records and quantum Darwinism

Quantum Darwinism studies how interactions distribute multiple records of selected
system observables through environmental fragments. Work in this framework (Ollivier,
Poulin and Zurek; Blume-Kohout and Zurek; Riedel and Zurek) has shown that redundant
environmental records can make certain information independently accessible to
multiple observers and can help account for effective classical objectivity.

IF Theory cannot claim novelty for the environment storing records, redundancy
supporting objectivity, or decoherence selecting robust observables. Its proposed
contribution is to compare record redundancy, thermodynamic cost, path
irreversibility, and causal-history contraction **within one accounting framework, in
one substrate at a time**.

## 4.6 Correlations can reverse local arrows

Standard thermodynamic reasoning typically assumes suitable initial independence or
weak correlation between interacting subsystems. Carefully prepared correlations can
enable behavior such as local heat flow from colder to hotter systems (Partovi;
Jennings and Rudolph), showing that a local apparent arrow can depend on hidden
correlation resources.

This is a decisive warning for IF Theory. Apparent thermodynamic-entropy reduction or
reversed record flow may **consume preexisting correlations** rather than violate
physical law. Any IF result reporting local reversal without naming the consumed
correlation resource is in the forbidden state `PERPETUAL_RECHARGE`.

## 4.7 Provisional novelty claim

The possible IF contribution is the combined proposal: *quantify the local arrow
through trajectory asymmetry; then measure, within a declared substrate, how
irreversible interactions generate robust records, how those records constrain
compatible histories, and how embedded agents convert those constraints into causal
memory.*

The stronger cosmic claim remains unearned unless IF Theory derives the required
asymmetric boundary conditions. The *cross-substrate* portion of the original novelty
claim is withdrawn: this combination is offered as a portable **protocol**, and its
portability is itself a falsifiable hypothesis (TA-H12′, §27), not an assumption.

---

# 5. Trajectory Irreversibility

Let the forward protocol be \(\lambda_F(t)\) and its reversed protocol

\[
\lambda_R(t)=\Theta_\lambda\,\lambda_F(T-t).
\]

Let \(P_F[\Gamma]\) be the forward path probability and \(P_R[\Gamma^\dagger]\) the
reverse path probability. Define stochastic irreversibility:

\[
\Sigma[\Gamma]=\ln\frac{P_F[\Gamma]}{P_R[\Gamma^\dagger]}.
\]

A trajectory with \(\Sigma[\Gamma]>0\) is more probable in the declared forward
experiment than its reverse is in the reverse experiment. A negative value is possible
for an individual fluctuation. The mean is a Shannon-ledger divergence:

\[
\langle\Sigma\rangle=D_{\mathrm{KL}}\!\left[P_F(\Gamma)\,\|\,P_R(\Gamma^\dagger)\right]\geq 0,
\]

and equals the **thermodynamic entropy production** in units of \(k_B\) only under
the stochastic-thermodynamic conditions in which the fluctuation relation is derived.
The identification is a theorem with hypotheses, not a definition; where the
hypotheses are not verified, \(\langle\Sigma\rangle\) is reported as a Shannon-ledger
distinguishability only.

## 5.1 Length of the local arrow

Define a dimensionless distinguishability using the Jensen–Shannon divergence:

\[
\mathcal A_\Gamma=D_{\mathrm{JS}}\!\left[P_F(\Gamma),\,P_R(\Gamma^\dagger)\right],
\qquad 0\leq\mathcal A_\Gamma\leq\ln 2 .
\]

- \(\mathcal A_\Gamma=0\): forward and reversed trajectories are statistically
  indistinguishable;
- large \(\mathcal A_\Gamma\): the orientation can be inferred reliably.

This is an epistemic (Shannon-ledger) measure of distinguishability. **It is not an
additional thermodynamic entropy and must never be added to one** (three-ledger
discipline, CLAUDE.md §1).

## 5.2 Optimal arrow classifier

Given trajectory \(\Gamma\), classify whether it was generated forward or backward.
For equal priors the ideal posterior is

\[
P(F\mid\Gamma)=\frac{P_F(\Gamma)}{P_F(\Gamma)+P_R(\Gamma^\dagger)}
=\frac{1}{1+e^{-\Sigma[\Gamma]}} .
\]

A successful learned arrow classifier must converge toward this likelihood-ratio
benchmark in analytically solvable models. The benchmark is per model family.

---

# 6. Physical Records

## 6.1 Statistical record

Let \(X_{t-\tau}\) be an earlier event and \(R_t\) a present subsystem. A statistical
record satisfies

\[
I\!\left(R_t;X_{t-\tau}\right)>0 .
\]

This is necessary but insufficient. Correlation may arise from a common cause,
selection bias, initial fine-tuning, a future boundary condition, deterministic global
constraints, or direct causal recording.

## 6.2 Causal record

\(R_t\) is a causal record of \(X_{t-\tau}\) when an intervention on the earlier event
changes the later record distribution:

\[
P\!\left(R_t\mid do(X_{t-\tau}=x)\right)\neq P\!\left(R_t\mid do(X_{t-\tau}=x')\right).
\]

The relevant causal channel must be physically specified. This is the Conway-gate
form of the record criterion: recordhood is **detected by intervention**, never
declared (CLAUDE.md §5, `TELEOLOGY_INJECTION`).

## 6.3 Accessible record

The record is accessible to system \(A\) if an allowed operation can extract useful
information from it:

\[
I_{\mathrm{acc}}^{A}\!\left(X_{t-\tau};R_t\right)>0 .
\]

A microscopic correlation dispersed across the environment may exist but be
inaccessible to any realistic local decoder. Accessibility is relative to the declared
operation class; it is a per-substrate quantity.

## 6.4 Robust record

Let \(\mathcal P\) be a declared perturbation class. A record is robust if it retains
retrodictive information under typical perturbations:

\[
\mathcal R_{\mathrm{robust}}
=\mathbb E_{\pi\sim\mathcal P}\!\left[I\!\left(X_{t-\tau};\pi(R_t)\right)\right].
\]

The perturbation class must be declared before measurement. A perfectly protected
record under an empty perturbation class is trivial.

## 6.5 Persistent record

Define record lifetime at tolerance \(\epsilon\):

\[
\tau_R(\epsilon)=\inf\left\{\Delta t:\;I\!\left(X_{t-\tau};R_{t+\Delta t}\right)<\epsilon\right\}.
\]

Long record lifetime may require continuing maintenance and error correction — paid in
energy, with waste heat exported.

---

# 7. Retrodictive and Predictive Information

Let the present macrostate be \(X_t\) and the record state \(R_t\). Define incremental
retrodictive and predictive information (both Shannon ledger):

\[
I_{\mathrm{retro}}(\tau)=I\!\left(R_t;X_{t-\tau}\mid X_t\right),
\qquad
I_{\mathrm{pred}}(\tau)=I\!\left(R_t;X_{t+\tau}\mid X_t\right).
\]

A memory may contain both. Define the epistemic-record asymmetry:

\[
\mathcal A_R(\tau)=I_{\mathrm{retro}}(\tau)-I_{\mathrm{pred}}(\tau).
\]

\(\mathcal A_R>0\) means the state carries more incremental past information;
\(\mathcal A_R<0\) means it is more predictive than retrodictive. **This does not
define the thermodynamic arrow, and it does not do so in any substrate.** A predictive
model may properly carry more future-relevant information than historical detail.

A caution inherited from P15: both quantities are estimator-dependent, their values
turning on the assumed representational format of \(R_t\). The P15 trilemma applies
verbatim. Any \(\mathcal A_R\) reported here ships with its estimator declared and
validated on a null system of known zero information (§36.5).

---

# 8. Record Redundancy

Suppose the environment contains fragments \(E_1,E_2,\ldots,E_N\), each of which may
carry information about system observable \(X\). For tolerated information loss
\(\delta\), define the smallest fragment size \(f_\delta\) satisfying

\[
I(X;E_f)\geq(1-\delta)H(X),
\]

where \(H(X)\) is the Shannon entropy of the observable. Define redundancy:

\[
\mathcal R_\delta=\frac{N}{f_\delta}.
\]

High redundancy means many disjoint fragments independently carry nearly complete
information about the selected observable. Redundancy contributes to robustness,
accessibility, agreement among observers, and persistence after local damage. It does
**not** mean every copy contains independent new information.

## 8.1 Independent and redundant constraint

If two records \(R_1\) and \(R_2\) contain the same information, then \(I(H;R_1,R_2)\)
may be only slightly larger than \(I(H;R_1)\). **Record count is not
historical-information count.** The correct joint constraint is

\[
\mathcal C_H=I\!\left(H;R_1,\ldots,R_N\right),
\]

evaluated jointly and never as a sum over records. Violating this is failure mode
§37.10.

---

# 9. Historical Constraint

Let \(H_{<t}\) denote the complete or coarse-grained history before \(t\). Before
reading a record the observer has prior \(P(H_{<t})\); after observing \(R_t=r\) it
has posterior \(P(H_{<t}\mid r)\). Define realized historical constraint:

\[
\mathcal C_H(r)=D_{\mathrm{KL}}\!\left[P(H_{<t}\mid r)\,\|\,P(H_{<t})\right].
\]

Average historical constraint is the mutual information

\[
\left\langle\mathcal C_H\right\rangle=I\!\left(H_{<t};R_t\right),
\]

measuring how much the present record contracts Shannon uncertainty over possible
histories.

## 9.1 Compatible-history volume

For finite histories with approximately uniform prior, define

\[
\Omega_H=\#\{\text{histories compatible with background constraints}\},
\qquad
\Omega_H(R_t)=\#\{\text{histories compatible with }R_t\},
\]

so that

\[
\mathcal C_H=\ln\frac{\Omega_H}{\Omega_H(R_t)} .
\]

This counting form is a **coarse-grained/combinatorial** quantity, appropriate only
under its declared ensemble and coarse-graining. It is not a Boltzmann thermodynamic
entropy and must not be substituted into a thermodynamic balance.

## 9.2 Historical-constraint rate

Define \(\dot{\mathcal C}_H = d\mathcal C_H/dt\). The rate may be positive during
record creation, zero during stable preservation, and negative during record
degradation or erasure. IF Theory therefore does **not** claim
\(\dot{\mathcal C}_H\geq 0\) for every subsystem or interval. The proposed arrow
concerns typical net production across an appropriate open system *plus* its
record-bearing environment.

---

# 10. The Record Ledger

For a declared domain \(\Omega\):

\[
\frac{d\mathcal C_H^\Omega}{dt}
=\dot{\mathcal C}_{\mathrm{write}}
+\dot{\mathcal C}_{\mathrm{copy}}
+\dot{\mathcal C}_{\mathrm{infer}}
-\dot{\mathcal C}_{\mathrm{decay}}
-\dot{\mathcal C}_{\mathrm{erase}}
-\dot{\mathcal C}_{\mathrm{overwrite}}
-\dot{\mathcal C}_{\mathrm{redundant}} .
\]

The final term corrects for counting duplicate records as independent history.
Inference alone may change one observer's knowledge without creating a new physical
record. The ledger therefore distinguishes physical record production, copying,
epistemic extraction, destruction, and redundancy.

This is a **Shannon-ledger** balance. It is coupled to, but never merged with, the
energy and thermodynamic-entropy ledgers; the coupling appears explicitly and only
through the cost terms of §11.

---

# 11. Record Cost

Let \(C_{\mathrm{write}}\), \(C_{\mathrm{maint}}\), \(C_{\mathrm{read}}\),
\(C_{\mathrm{copy}}\), \(C_{\mathrm{erase}}\) be the energy-ledger costs of writing,
maintaining, reading, copying, and erasing. Total record cost:

\[
C_R=C_{\mathrm{write}}+C_{\mathrm{maint}}+C_{\mathrm{read}}+C_{\mathrm{copy}}+C_{\mathrm{erase}} .
\]

**No universal conversion exists from record bits to joules** independent of the
physical implementation, temperature, error tolerance, speed, and allowed operations.
This is the same lesson P15 learned in the agency lab from the other direction: an
information denominator is not portable across implementations, and a *declared* cost
is auditable where an *inferred* information content is not.

## 11.1 Record efficiency

Define useful historical constraint per unit cost:

\[
\eta_R=\frac{\Delta\mathcal C_H}{C_R},
\]

with units of information per energy. **This is not a thermodynamic efficiency and is
not bounded by one.** It is a comparative engineering or evolutionary metric, valid
for ranking architectures *within* a declared implementation class. Cross-class
comparison of \(\eta_R\) values is not licensed by anything in this paper.

---

# 12. Causal Memory

A record becomes memory for an agent when it enters future control. Let the agent's
internal memory be \(M_t\), generated by

\[
M_{t+1}=U\!\left(M_t,O_t,A_t\right),
\]

and influencing action

\[
A_t=\pi\!\left(O_t,M_t\right).
\]

A causal memory requires the intervention chain

\[
M_{t-\tau}\rightarrow A_t\rightarrow Y_{t+\Delta} ,
\]

i.e. ablating or scrambling memory must alter future action or outcome. Such tests are
**within-family causal claims**, of the same logical type as the surviving rule/state
dissociation (6σ): they establish that a specific internal variable does physical work
in a specific substrate, and nothing about other substrates.

## 12.1 Historical memory

Historical memory optimizes reconstruction of earlier events:

\[
M_t^{\mathrm{hist}}\approx\arg\max_M I(M;X_{<t}).
\]

## 12.2 Predictive memory

Predictive memory optimizes future-relevant information:

\[
M_t^{\mathrm{pred}}\approx\arg\max_M I(M;X_{>t}).
\]

## 12.3 Causal-state memory

A minimal predictive representation groups pasts producing identical conditional
future distributions:

\[
x_{<t}\sim x'_{<t}\iff P(X_{>t}\mid x_{<t})=P(X_{>t}\mid x'_{<t}),
\qquad S_t=\epsilon\!\left(X_{<t}\right),
\]

with statistical complexity \(C_\mu=H(S_t)\), a Shannon entropy of the causal-state
distribution. An efficient agent may forget historical details that do not alter
future action. Therefore **historical detail ≠ predictive value** — and, per P15's
chemotaxis existence proof, an agent can perform substantial causal work while
carrying essentially no recoverable predictive information about the variable it
exploits. Neither \(C_\mu\) nor any of the memory measures above should be read as a
proxy for agency.

---

# 13. The IF Record-Arrow Vector

No single scalar completely characterizes time asymmetry. Define

\[
\mathbf A_{\mathrm{IF}}
=\left[\langle\Sigma\rangle,\;\mathcal A_\Gamma,\;\dot{\mathcal C}_H,
\;\mathcal R_\delta,\;\mathcal A_R,\;J_{\mathrm{memory}}\right],
\]

where \(\langle\Sigma\rangle\) is mean trajectory irreversibility, \(\mathcal A_\Gamma\)
forward–reverse distinguishability, \(\dot{\mathcal C}_H\) historical-constraint
production, \(\mathcal R_\delta\) record redundancy, \(\mathcal A_R\)
retrodictive–predictive asymmetry, and \(J_{\mathrm{memory}}\) the causal value of
memory.

The vector prevents IF Theory from hiding contradictory behavior inside one weighted
score. It is **explicitly not** a candidate invariant — its components live in
different ledgers and are estimator-dependent to different degrees. No scalar
reduction of it is proposed as substrate-independent. It is reported per family,
componentwise, with each component's estimator named.

---

# 14. The IF Record-Arrow Hypothesis

## 14.1 Weak form

> In ordinary open nonequilibrium systems, the direction of positive mean
> thermodynamic entropy production is also typically the direction in which robust
> causal records of earlier states are generated.

## 14.2 Constraint form

> Record-forming interactions reduce Shannon uncertainty over compatible histories in
> the thermodynamic-entropy-producing orientation:

\[
\langle\Sigma\rangle>0\quad\Rightarrow\quad\mathbb E\!\left[\Delta\mathcal C_H\right]>0
\]

for a **declared class of record-forming processes in a declared substrate**. It is not
expected for every process, nor with the same slope or magnitude in another substrate.

## 14.3 Agent form

> Embedded agents call the record-producing orientation "past to future" because their
> memories, models, and self-maintenance machinery are physically assembled along that
> direction.

## 14.4 Strong cosmic form

> The universe's global arrow is generated by the irreversible accumulation of physical
> records.

This strong form is **not established**. It risks circularity unless IF Theory explains
why record formation begins asymmetrically from time-symmetric laws and boundary
conditions. The cosmology branch is firewalled from the agency branch and remains
unpreregistered and untested; nothing here may be cited as cosmological support.

---

# 15. The Boundary-Condition Problem

Suppose a system is observed in a low-thermodynamic-entropy macrostate at \(t=0\). For
a typical compatible microstate selected without an asymmetric boundary condition,
thermodynamic entropy may increase as \(|t|\) in both directions away from that state.
Records can then form in two opposite orientations.

To explain the consistent arrow observed across our accessible universe, one ordinarily
requires a low-thermodynamic-entropy boundary at one temporal end; a time-asymmetric
law; asymmetric cosmological geometry; asymmetric causal boundary conditions; or
another independently derived mechanism. **The existence of records at later times does
not by itself explain why the boundary was low thermodynamic entropy.**

## 15.1 IF honesty condition

IF Theory must not reason:

1. the past is where the records point;
2. records point to the past;
3. therefore records explain why there is a past.

A successful account must derive a measurable asymmetry *before* using the word
**past**.

## 15.2 Required IF derivation

The strong theory must derive at least one of: the probability of a low-record initial
state, \(P(\text{low-record initial state})\), from a deeper law; a fundamental
dynamical asymmetry \(P(\Gamma)\neq P(\Gamma^\dagger)\); or a cosmological attractor
producing two arrows away from a central low-complexity boundary.

Until then, records explain the **manifestation and amplification** of the arrow, not
its ultimate origin. That sentence is the paper's load-bearing negative result, and it
is unaffected by the IF-H1 falsification.

---

# 16. Minimal Record-Writing Model

Let \(X\in\{0,1\}\) be a source bit and \(R\in\{0,1\}\) a memory bit initialized in the
standard state \(R=0\). A reversible copy operation is

\[
(X,R)\mapsto(X,R\oplus X).
\]

If \(R\) begins in a known standard state, the final memory records \(X\), and the full
mapping is reversible. **Record formation does not logically require erasure during
each copy.** However, preparing a reusable memory in the known initial state requires
resetting it, and that reset has thermodynamic consequences under the relevant physical
conditions.

## 16.1 Reversible record formation

Before copying, \(I(X;R)=0\); after copying, \(I(X;R)=H(X)\). The joint Shannon entropy
may remain constant under reversible dynamics — information has moved into correlation.
Therefore

\[
\text{record creation}\;\not\Rightarrow\;\text{increase in fine-grained total information}.
\]

## 16.2 Reusable memory cycle

A complete memory cycle is: prepare, write, preserve, read, erase or overwrite,
reprepare. **Thermodynamic cost must be assessed over the full cycle**, not only the
reversible copy step. Reporting only the reversible step is failure mode §37.9 and is
the standard route into `PERPETUAL_RECHARGE`.

---

# 17. Minimal Irreversible Record Model

Let the record be stabilized through coupling to a dissipative environment, with rate
\(k_+\) into matching states and \(k_-\) away. Detailed balance may imply

\[
\frac{k_+}{k_-}=e^{\beta\Delta E}.
\]

With record fidelity \(q=P(R=X)\), the stored Shannon information for unbiased binary
\(X\) is

\[
I(X;R)=\ln 2-H_{\mathrm b}(q),
\qquad H_{\mathrm b}(q)=-q\ln q-(1-q)\ln(1-q),
\]

where \(H_{\mathrm b}\) is the binary Shannon entropy. Increasing stability generally
requires a larger barrier, stronger dissipation during writing, slower switching, active
error correction, or some combination. **There is no free infinitely stable record.**

---

# 18. Record Formation and Entropy Production

Let \(\Delta I_{XR}\) be the mutual information produced between source and record. A
generalized second-law relation for measurement and feedback may include informational
terms; a schematic inequality is

\[
\beta\left(W-\Delta F\right)\geq-\Delta I_{XR},
\]

with the exact sign and form depending on the process, definitions, and initial
correlations. Paper 12 will **not** use one generic information-thermodynamic
inequality without deriving its conditions for the implemented model. Applying such an
inequality outside its derived hypotheses is precisely the `ENTROPY_CONFLATION`
failure: it silently equates a Shannon-ledger increment with a thermodynamic-entropy
budget.

## 18.1 Record-cost conjecture

For a fixed physical memory architecture and error requirement, define the minimal
thermodynamic entropy production \(\Sigma_{\min}(I_R,\tau_R,\epsilon)\). The IF
conjecture is monotonicity in stored information and in lifetime:

\[
\frac{\partial\Sigma_{\min}}{\partial I_R}\geq 0,
\qquad
\frac{\partial\Sigma_{\min}}{\partial\tau_R}\geq 0
\]

**over a declared architecture class.** The conjecture is explicitly *not* universal
across all possible memories, because reversible writing and alternative physical
encodings may alter the trade-off. Post-2026-07-18 this scoping is mandatory, not
optional: a monotonicity measured in one memory architecture is a statement about that
architecture.

---

# 19. Record Decay

Let record fidelity obey

\[
\frac{dq}{dt}=-\gamma\left(q-\tfrac12\right)
\quad\Longrightarrow\quad
q(t)=\tfrac12+\left[q(0)-\tfrac12\right]e^{-\gamma t}.
\]

The stored Shannon information decays as

\[
I_R(t)=\ln 2-H_{\mathrm b}\!\left[q(t)\right].
\]

A passive record has finite lifetime. Active maintenance may reduce the effective
\(\gamma\) at continuing energetic cost, with waste heat exported to the reservoir.
This connects Paper 12 with Paper 6's repair framework.

---

# 20. Records as Repair Substrates

Repair requires a reference. A system can restore damaged organization only if it
retains information about prior structure, invariants, target dynamics, viable-state
boundaries, and error syndromes. Let \(R_t^{\mathrm{self}}\) be a record of the
system's viable organization, with repair policy

\[
A_t^{\mathrm{repair}}=\pi_Q\!\left(D_t,R_t^{\mathrm{self}}\right).
\]

Scrambling the self-record should impair repair while preserving the same physical
resources — a marginal-preserving intervention, in the same protocol family as the
agency-lab ablations. Thus records create causal continuity:

\[
\text{past organization}\rightarrow\text{present record}\rightarrow\text{future restoration}.
\]

The measured repair deficit is a per-substrate effect size. It is a within-family
causal claim, and it is the strongest class of claim this program currently supports.

---

# 21. Historical Constraint and Identity

A persistent agent's identity can be represented as a constrained lineage of states
rather than a fixed collection of matter. Let \(Z_t\) be the present agent state and
\(\mathcal H_A(Z_t)\) the set of histories consistent with its memories, architecture,
and causal continuity. Define identity constraint:

\[
\mathcal C_{\mathrm{id}}(t)=-\ln P\!\left[\mathcal H_A(Z_t)\right].
\]

A stronger record-supported identity has fewer compatible arbitrary histories. This does
**not** imply that identity is merely information: the information must be physically
instantiated and causally connected.

---

# 22. Choices and the Arrow of Time

An agent's choice can create new records: a selected action changes the environment,
leaves traces, is stored in internal memory, is recorded by observers, and renders
alternatives counterfactual rather than actual.

Let available policies be \(\Pi_t=\{\pi_1,\ldots,\pi_n\}\) and let the agent select
\(\pi^*\), whose trajectory creates records \(R_{t+\tau}\). Historical constraint
increases because the records make one realized branch more probable than alternatives:

\[
\Delta\mathcal C_H=I\!\left(\pi^*;R_{t+\tau}\right).
\]

This is a meaningful local sense in which choice **writes history**.

## 22.1 What choice does not do

Choice does **not** create new energy, a new temporal coordinate, an additional
conserved physical quantity, an infinite reservoir of work, or a violation of unitary or
reversible dynamics. The selected action redirects existing physical capacity and
establishes new correlations and records, paying the write and maintenance costs of §11
and exporting waste heat. Therefore

\[
\text{choice can create historical specificity without creating energy.}
\]

Any IF text stating or implying otherwise is in the forbidden state
`PERPETUAL_RECHARGE`.

## 22.2 Source-of-time hypothesis

The strong statement "conscious choice creates time" is **not supported** by Paper 12.
A defensible restricted statement is:

> Agents create local irreversible records that deepen the experienced distinction
> between remembered past and open future.

The physical arrow must already supply the conditions under which those records are
stably formed, unless a deeper IF derivation proves otherwise.

---

# 23. Counterfactual Openness

The past appears fixed because records constrain it. The future appears open because
multiple future trajectories remain compatible with the current state. Let
\(\mathcal H_-(t)\) be histories compatible with present records and
\(\mathcal H_+(t)\) future continuations compatible with the present physical state.
Define the conditional Shannon entropies

\[
S_-(t)=H\!\left(H_{<t}\mid R_t\right),
\qquad
S_+(t)=H\!\left(H_{>t}\mid X_t\right),
\]

and the epistemic openness gap

\[
\mathcal O_t=S_+(t)-S_-(t).
\]

Ordinarily \(\mathcal O_t>0\). This is an **agent-relative Shannon-ledger uncertainty
asymmetry**, not a thermodynamic quantity, and it does not prove ontological
indeterminism: a fully deterministic universe can have \(\mathcal O_t>0\) for an
embedded observer with incomplete state access.

---

# 24. Record Arrow in Expanding IF Universes (toy)

**This entire section is an explicitly labeled toy computational study on the Paper 4
artificial domain. It is not cosmology, it is not derived from a covariant action, and
no result here may be transferred to physical expansion** (forbidden state
`COMMANDED_EXPANSION`). The growth schedule \(g\) is *imposed by the experimenter as a
control variable*, not produced by the model's own dynamics.

Paper 4 introduced an expanding artificial domain. Expansion may affect record
formation in competing ways:

**Low expansion** — records overwrite one another; available storage saturates;
environmental fragments repeatedly interact; redundancy may be destroyed.

**Intermediate expansion** — records separate spatially; interference decreases;
multiple fragments preserve independent access; historical constraint may grow.

**Excessive expansion** — causal contact weakens; records become inaccessible; copies
cannot be compared; maintenance and decoding fail.

This suggests a possible record-accessibility window

\[
g_{\min}^{R}<g<g_{\max}^{R},
\]

which is a computational hypothesis about a toy domain and is not an explanation of
cosmological expansion.

---

# 25. Quantum Records

## 25.1 Decoherence

A system \(S\) interacts with environment \(E\):

\[
\left(\sum_i c_i|s_i\rangle\right)|E_0\rangle\rightarrow\sum_i c_i|s_i\rangle|E_i\rangle .
\]

When environmental states become distinguishable, \(\langle E_i|E_j\rangle\approx 0\),
interference between corresponding system states becomes locally inaccessible.

## 25.2 Environmental records

Fragments of \(E\) may carry information about selected system observables. Record
redundancy can permit multiple observers to infer the same outcome without directly
interacting with the original system.

## 25.3 Unitary reversibility

The global state may still evolve unitarily. In principle, reversing the complete
system–environment interaction can erase the apparent record and restore coherence. In
practice the information may be distributed across enormous environmental degrees of
freedom. Operational irreversibility arises from dispersion, control limitations,
environmental size, coarse-graining, unavailable phases, and record redundancy — that
is, from a **coarse-grained/observational** ledger, not from a change in the
fine-grained von Neumann entropy, which is invariant under unitary evolution. Keeping
those two ledgers apart is mandatory here.

## 25.4 IF quantum claim

Paper 12 does **not** claim that decoherence alone solves the measurement problem, the
Born rule, the selection of one experienced outcome, or the cosmic
low-thermodynamic-entropy condition. It uses quantum-record theory as a model of how
environmental records can become robust and multiply accessible — one substrate among
several, studied on its own terms.

---

# 26. Local Reversal Experiments

A local arrow can be weakened or reversed through carefully prepared protocols:
reversing a driven Hamiltonian; feedback control; using prepared correlations; erasing
environmental records; recombining decohered branches in a controlled system. Quantum-
control work has shown that monitored-system trajectories can be engineered to become
more consistent with reversed dynamics under active control. Such experiments
demonstrate **local controllability, not reversal of the universe's complete arrow.**

The IF prediction is:

> Local arrow reversal consumes low-thermodynamic-entropy control, correlation,
> measurement, or work resources, and exports records and waste heat elsewhere.

The complete ledger must include the controller. A reported reversal whose ledger omits
the controller is not a result; it is failure mode §37.13.

---

# 27. Core Hypotheses

Each hypothesis is declared **per family and per protocol**. Confirmation in one family
is not evidence about another; this is the direct methodological consequence of the
2026-07-18 falsification.

## TA-H1 — Path-asymmetry hypothesis

Mean thermodynamic entropy production equals or bounds the distinguishability between
forward and reversed path ensembles in the declared stochastic models.

**Falsifier.** The implemented trajectory statistic fails exact fluctuation-relation
benchmarks.

## TA-H2 — Causal-record hypothesis

Naturally generated robust records contain interventionally grounded information about
earlier events.

**Falsifier.** Record correlations disappear under causal controls, or are explained
entirely by common initial conditions.

## TA-H3 — Record-alignment hypothesis

In ordinary nonequilibrium systems *of the declared family*, robust records are
preferentially generated in the positive-thermodynamic-entropy-production orientation.

**Falsifier.** Equally robust records form generically in both directions without
fine-tuned boundary conditions.

## TA-H4 — Historical-constraint hypothesis

Record formation increases \(\mathcal C_H=I(H_{<t};R_t)\) after correcting for
redundancy and common causes.

**Falsifier.** The record does not reduce Shannon uncertainty over compatible histories.

## TA-H5 — Redundancy–robustness hypothesis

Independent environmental copies increase resilience to local record destruction.

**Falsifier.** Redundancy provides no survival or accessibility advantage under
held-out perturbations.

## TA-H6 — Record-cost hypothesis

Increasing fidelity, lifetime, and accessibility generally requires greater physical
cost **within a fixed memory architecture**.

**Falsifier.** Arbitrarily reliable and permanent records are obtained at vanishing cost
under the same physical constraints.

## TA-H7 — Causal-memory hypothesis

An agent's records become memory only when intervening on them changes future policy or
outcome.

**Falsifier.** Memory ablation has no selective causal consequence.

## TA-H8 — Epistemic-openness hypothesis

Embedded agents possess lower conditional Shannon uncertainty about recorded histories
than about future continuations: \(\mathcal O_t>0\).

**Falsifier.** The asymmetry disappears under realistic agent and environment models.

## TA-H9 — Local-reversal-cost hypothesis

Apparent local reversal consumes prepared correlations, work, control, or
low-thermodynamic-entropy resources when the complete system boundary is included.

**Falsifier.** A repeatable net reversal occurs without compensating resource
consumption or thermodynamic-entropy export.

## TA-H10 — Expansion-record-window hypothesis (toy domain)

In the Paper 4 toy domain, costly imposed domain growth produces a finite region
maximizing accessible record redundancy and lifetime.

**Falsifier.** Record performance is monotonic in \(g\), or unrelated to it, after
density and resource controls.

## TA-H11 — Boundary-condition necessity hypothesis

Record accumulation alone does not create a global one-sided arrow from generic
time-symmetric equilibrium boundary conditions.

**Falsifier.** A rigorous model demonstrates a unique global arrow from symmetric
generic conditions without hidden asymmetry or postselection.

## TA-H12′ — Methodological-portability hypothesis (replaces TA-H12)

*The former TA-H12 asserted that the same path, record, and constraint measures
"organize temporal asymmetry across stochastic, cellular, agent, and quantum models."
That is a universality claim of exactly the type falsified on 2026-07-18, and it is
withdrawn.*

The replacement claim is procedural and much weaker: **the measurement protocol** —
declare the reverse experiment, declare the record variable in advance, declare the
perturbation class, validate the estimator on a null system, close the cost ledger over
a full memory cycle — **can be instantiated in each substrate and yields internally
valid, reproducible per-family numbers.** No agreement of values across families is
predicted, and none will be reported as support.

**Falsifier.** The protocol cannot be instantiated in some substrate without a
substrate-specific redefinition that changes what is being measured — the P15 pathology,
in which an estimator presupposes the very structure it is supposed to test. Should this
occur, TA-H12′ dies and the affected measure is declared substrate-local.

---

# 28. Analytical Toy Model I: Measurement and Record

Let \(X\sim\operatorname{Bernoulli}(\tfrac12)\), with memory copying correctly with
probability \(q\), so \(P(R=X)=q\) and

\[
I(X;R)=\ln 2-H_{\mathrm b}(q).
\]

Suppose fidelity depends on dissipated work:

\[
q(W)=\frac{1}{1+e^{-\beta(W-W_0)}},
\qquad
I_R(W)=\ln 2-H_{\mathrm b}\!\left[q(W)\right].
\]

The exact function is **architecture-specific**. The experiment tests whether the
information–cost curve is monotonic, saturating, speed-dependent, or error-dependent —
for the architecture actually implemented.

---

# 29. Analytical Toy Model II: Redundant Records

Let \(N\) conditionally independent copies each record \(X\) with error
\(\epsilon<\tfrac12\). A majority decoder has error

\[
\epsilon_N=\sum_{k=\lceil N/2\rceil}^{N}\binom{N}{k}\epsilon^k(1-\epsilon)^{N-k},
\]

and the decoded information is

\[
I_N=\ln 2-H_{\mathrm b}(\epsilon_N).
\]

**Note on capacity growth.** The monotone increase of \(I_N\) with \(N\) is a *derived*
combinatorial consequence of the majority-decoder error expression above, under the
stated conditional-independence assumption — it is not an imposed or commanded expansion
of capacity, and it buys nothing free: robustness increases with \(N\) while physical
cost scales as

\[
C_N=N\,C_{\mathrm{copy}}+N\,C_{\mathrm{maint}} .
\]

The optimal redundancy is therefore finite:

\[
N^*=\arg\max_N\left[V_R(I_N)-C_N\right],
\]

and infinite redundancy is not generally optimal. Any presentation of
\(I_N\rightarrow I_{N+k}\) growth without the accompanying \(C_N\) term is in the
forbidden state `COMMANDED_EXPANSION`.

---

# 30. Analytical Toy Model III: History Contraction

Suppose there are initially \(2^T\) possible binary histories of length \(T\), and a
perfect record stores the last \(m\) states, so compatible histories fall to
\(2^{T-m}\). Then

\[
\mathcal C_H=\ln\frac{2^T}{2^{T-m}}=m\ln 2 .
\]

If each recorded bit is independently wrong with probability \(\epsilon\), the effective
constraint is reduced:

\[
\mathcal C_H=m\left[\ln 2-H_{\mathrm b}(\epsilon)\right].
\]

Both expressions are Shannon-ledger quantities over a declared uniform history ensemble.

---

# 31. Analytical Toy Model IV: Two-Sided Boundary Conditions

Let a reversible system have a low-thermodynamic-entropy constraint at \(t=0\), with no
condition distinguishing positive from negative time. A typical
thermodynamic-entropy profile is

\[
S^{\mathrm{thermo}}(t)=S_0+\alpha|t| .
\]

Records form away from the minimum on both sides. Observers at \(t>0\) call increasing
\(t\) the future; observers at \(t<0\) call decreasing \(t\) the future. This model
demonstrates:

> Record formation can align *local* arrows without producing one globally preferred
> orientation.

This is the cleanest statement of the paper's core negative result, and it is a
constructive counterexample to the strong cosmic form (§14.4).

---

# 32. Experimental Program

**Experiment 1 — Reversible copy.** Implement reversible bit copying. Measure mutual
information, fine-grained Shannon entropy, correlation, work, and erasure cost over the
full cycle.

**Experiment 2 — Dissipative record stabilization.** Vary energy barrier, writing speed,
temperature, error tolerance, and record lifetime. Map the record-cost surface for the
implemented architecture.

**Experiment 3 — Forward–reverse classification.** Generate stochastic trajectories,
train a classifier to infer temporal orientation, and compare with the exact
likelihood-ratio result of §5.2.

**Experiment 4 — Record-arrow alignment.** Run nonequilibrium processes with record
media and measure \(\langle\Sigma\rangle\), \(\dot{\mathcal C}_H\), and
\(\mathcal A_R\). Test whether their signs align within the family.

**Experiment 5 — Time-symmetric boundary conditions.** Impose a
low-thermodynamic-entropy central boundary with reversible dynamics and test for two
arrows pointing away from the minimum (§31).

**Experiment 6 — Fine-tuned future record.** Prepare a record correlated with a future
event through boundary conditioning, and test whether the record is fragile to small
perturbations.

**Experiment 7 — Record redundancy.** Create multiple environmental copies, destroy
random subsets, and measure residual accessibility and retrodictive accuracy.

**Experiment 8 — Record degradation.** Allow copies to decay; measure lifetime,
maintenance cost, and historical-constraint loss.

**Experiment 9 — Agent memory.** Embed records in Paper 5 agents. Scramble memories with
marginal-preserving interventions and measure prediction, action, repair, and survival.
Results are **within-family causal claims**, reported with the substrate named; the
rule/state dissociation is the template.

**Experiment 10 — Choice and trace production.** Allow agents to choose among policies;
measure how actions change environmental records, historical constraint, thermodynamic
entropy production, and future action space.

**Experiment 11 — Local arrow reversal.** Use feedback or initial correlations to produce
locally reverse-looking behavior. Expand the accounting boundary until compensating costs
are found or the hypothesis fails.

**Experiment 12 — Expanding domain (toy).** Run record media under Paper 4 imposed growth
schedules and test the predicted record-accessibility window. Toy domain only.

**Experiment 13 — Quantum circuit records.** Simulate a system interacting with many
environment qubits; measure decoherence, fragment information, redundancy, and
reversibility under global control.

**Experiment 14 — Coarse-graining sensitivity.** Repeat arrow and record inference at
multiple spatial and temporal resolutions. A physical conclusion must not depend
entirely on one arbitrary coarse-graining.

---

# 33. Phase Taxonomy

| Phase | Name | Signature |
|---|---|---|
| T0 | Reversible equilibrium | \(\langle\Sigma\rangle\approx 0\); forward and reverse indistinguishable |
| T1 | Fluctuation | Individual directional fluctuations; no persistent arrow |
| T2 | Dissipative | Positive mean path irreversibility; no durable records |
| T3 | Record | Persistent records of earlier events form |
| T4 | Redundant-record | Multiple accessible environmental records exist |
| T5 | Causal-memory | Records guide future action |
| T6 | Historical-identity | Persistent agents use records for continuity and repair |
| T7 | Institutional-record | Shared archives outlive individual agents |
| T8 | Saturated-record | New records overwritten or rendered inaccessible |
| T9 | Fragmented-record | Records persist physically but cannot be jointly accessed |

The taxonomy is a descriptive vocabulary for phases observed within a substrate. It is
not claimed that the phase boundaries occur at the same parameter values, or in the same
order, in different substrates.

---

# 34. Deterministic Notebook Program

| Notebook | Content |
|---|---|
| 12A — Time-Reversal Conventions | Even/odd variables; reversed protocols; trajectory reversal; boundary-condition transformation |
| 12B — Fluctuation-Theorem Baselines | Validate \(\Sigma[\Gamma]=\ln P_F/P_R\); recover exact results in finite Markov systems |
| 12C — Arrow Classifier | Train forward–reverse classifiers; compare learned log odds with exact \(\Sigma[\Gamma]\) |
| 12D — Reversible Copy Gate | Track marginal Shannon entropy, joint Shannon entropy, mutual information, work |
| 12E — Full Memory Cycle | Prepare, write, read, erase, reprepare; verify the complete cost ledger |
| 12F — Record Fidelity and Lifetime | Sweep \(T,\;W,\;\epsilon,\;\tau_R\) |
| 12G — Causal Record Intervention | Distinguish direct records, common-cause correlations, fine-tuned boundary correlations, predictive correlations |
| 12H — Historical Constraint Estimator | Estimate \(I(H_{<t};R_t)\); validate on known history ensembles |
| 12I — Redundancy and Damage | Measure \(\mathcal R_\delta\) under random fragment loss |
| 12J — Time-Symmetric Boundary Universe | Create two arrows away from a central low-thermodynamic-entropy condition; test observer-relative orientation |
| 12K — Correlation-Fueled Reversal | Prepare correlated subsystems; track correlation consumption, heat flow, total thermodynamic entropy |
| 12L — Computational-Mechanics Memory | Infer \(S_t=\epsilon(X_{<t})\); compare statistical complexity, historical information, predictive information |
| 12M — Agent Memory Ablation | Transfer Paper 12 records into Paper 5 and Paper 6 agents; measure causal value **within family** |
| 12N — Choice and Historical Constraint | Quantify how selected actions produce distinguishable traces |
| 12O — Expansion–Record Window (toy) | Sweep imposed domain growth; measure record density, redundancy, accessibility, lifetime, overwrite rate |
| 12P — Quantum Darwinism Circuit | Simulate redundant environmental records; compute fragment mutual information and redundancy |
| 12Q — Global Reversal Audit | Attempt to reverse system, record, environment, controller; identify where irreversibility enters |
| 12R — Coarse-Graining Audit | Repeat all metrics under multiple macrostate partitions |
| **12S — Per-Family Scaling** | *Revised.* Within each substrate separately, test whether record creation and robustness scale against \(\langle\Sigma\rangle,\;C_R,\;\tau_R,\;\mathcal R_\delta\). **No cross-family collapse is predicted; failure of collapse is an expected outcome, not a program failure.** Any collapse observed is logged as an unexplained observation and not pursued without an independent pre-registration. |
| 12T — Adversarial Audit | A separate agent attempts to show the arrow arises from update order, random-number direction, boundary leakage, hidden dissipation, asymmetric initialization, future information leakage, record-detector bias, coarse-graining choice, or omitted controller cost |

---

# 35. Reproducibility Record

Every run emits:

```yaml
experiment_id: if-arrow-of-time-12
paper_version: null
git_commit: null
environment_hash: null
implementation: null
random_seed: 65537

substrate_family: null            # required post-2026-07-18: no cross-family pooling
protocol_id: null
microdynamics_time_reversal_symmetric: null
forward_protocol_hash: null
reverse_protocol_hash: null
initial_boundary_condition: null
final_boundary_condition: null
coarse_graining_hash: null

mean_thermodynamic_entropy_production: null
path_kl_divergence: null
jensen_shannon_arrow: null
arrow_classifier_accuracy: null

record_source_variable: null
record_state_variable: null
record_causal_channel: null
retrodictive_information: null
predictive_information: null
record_asymmetry: null
record_estimator_id: null         # required: which I_use-style estimator, per P15
record_estimator_null_validation: null
record_lifetime: null
record_accessibility: null
record_redundancy: null

historical_constraint: null
historical_constraint_change: null
redundancy_corrected_constraint: null

write_cost: null
maintenance_cost: null
read_cost: null
copy_cost: null
erase_cost: null

memory_ablation_effect: null
choice_trace_information: null
local_reversal_resource_cost: null

boundary_residuals: []
fluctuation_theorem_residual: null
invariant_failures: []
result_hash: null
```

---

# 36. Statistical Standards

**36.1 Trajectories are the sample units.** Time points from one trajectory are
correlated; primary uncertainty uses independent trajectories or justified block methods.

**36.2 The reverse experiment must be defined correctly.** The reverse ensemble requires
more than reading stored data backward: it may require a reversed driving protocol,
reversed odd-parity variables, reversed boundary distributions, and reversed feedback
rules.

**36.3 Record selection must be preregistered.** The record variable may not be chosen
after observing which subsystem most strongly correlates with the designated past.
Preregistration means a timestamped commit before the data exists (`RETROFIT_FORECAST`).

**36.4 Boundary conditions must be reported.** Every arrow experiment reports initial
macrostate, initial correlations, final conditioning, reservoirs, and control systems.

**36.5 Mutual-information bias.** Information estimators must be validated on null
systems with known zero information. Given the P15 estimator trilemma, this is not a
formality: an unvalidated information estimator is the single most likely source of a
false positive in this program.

**36.6 Multiple candidate records.** Testing many possible record variables requires
holdout data or multiplicity correction.

**36.7 Coarse-graining robustness.** The direction of the claimed arrow must survive a
reasonable family of physically motivated coarse-grainings.

**36.8 No cross-family pooling.** Results from different substrate families are never
pooled, averaged, or fitted jointly. Each family is analyzed and reported separately.

---

# 37. Failure Modes

| # | Failure mode | Description |
|---|---|---|
| 37.1 | Time-arrow conflation | The existence of parameter \(t\) is described as proof of a thermodynamic direction |
| 37.2 | Entropy-arrow circularity | Forward is *defined* as the direction of thermodynamic-entropy increase, then that increase is claimed to explain forward |
| 37.3 | Record-arrow circularity | Past is defined as the direction records point, then records are said to prove the past |
| 37.4 | Hidden low-entropy boundary | A highly ordered initial state is inserted and later described as emergent |
| 37.5 | Reverse-protocol error | The backward data sequence is treated as the physical reverse experiment |
| 37.6 | Correlation without causation | A common cause creates correlation between event and record |
| 37.7 | Future leakage | The simulation or estimator accesses future state during record formation |
| 37.8 | Free memory preparation | All records begin in blank low-thermodynamic-entropy states with no preparation cost |
| 37.9 | Partial-cycle accounting | Writing is reversible, so the memory system is declared cost-free while reset is omitted |
| 37.10 | Redundancy double counting | A million copies of one bit are counted as a million independent bits of history |
| 37.11 | Coarse-graining manufacture | The arrow appears only under one hand-selected macrostate partition |
| 37.12 | Local–global confusion | A reversed local heat flow is described as reversal of the universe's arrow |
| 37.13 | Correlation-resource omission | Prepared correlations enabling reverse-looking behavior are excluded from the ledger |
| 37.14 | Agency inflation | An agent leaving traces is said to create time itself |
| 37.15 | Cosmological inflation | Record growth is presented as a cause of cosmic expansion without a covariant derivation |
| **37.16** | **Ledger conflation** | Any sentence using "entropy" without naming thermodynamic, Shannon, algorithmic, or coarse-grained/observational — or adding a bit-valued quantity to a joule-valued one (`ENTROPY_CONFLATION`) |
| **37.17** | **Universality drift** | A per-family measured value is quoted, compared, or extrapolated as if substrate-independent. This is the failure mode that killed IF-H1; it is now a named, checkable state |

---

# 38. Criteria for Success

| Level | Criterion |
|---|---|
| 1 | **Correct irreversibility measurement** — the implementation reproduces exact fluctuation relations and known reversible limits |
| 2 | **Valid record measure** — the framework distinguishes causal records from accidental correlations |
| 3 | **Record-arrow alignment** — record creation aligns statistically with the thermodynamic-entropy-producing orientation, *demonstrated separately in each model family studied* |
| 4 | **Historical-constraint accounting** — record generation, redundancy, decay, and erasure close a measurable constraint ledger |
| 5 | **Agent integration** — causal memories demonstrably guide prediction, repair, and policy in resource-constrained agents, by intervention |
| 6 | **Controlled local reversal** — apparent arrow reversal is predicted quantitatively from consumed control or correlation resources |
| 7 | **Protocol portability** (*revised*) — the measurement protocol instantiates in stochastic, artificial-life, agent, and quantum substrates without substrate-specific redefinition of what is measured. **Replaces the former "cross-substrate scaling" level: shared dimensionless relationships across substrates are no longer a success criterion, are not expected, and would require independent pre-registration before being claimed** |
| 8 | **Boundary derivation** — a deeper IF theory derives the asymmetric condition from which the cosmic record arrow follows |

Only Level 8 would address the ultimate origin problem. No level is currently achieved
above Level 2 in any substrate.

---

# 39. What Would Count as a Major Discovery?

A strong information-thermodynamics result would be a quantitative relation predicting
how much robust historical constraint a physical process creates per unit thermodynamic
entropy production **within a declared architecture class** — a per-architecture
regression with a stated domain of validity, not a law of nature. The distinction is
not pedantry: it is exactly the distinction whose neglect produced the IF-H1
falsification.

A strong artificial-life result would be embedded agents spontaneously constructing
memory systems whose record orientation, lifetime, and causal value are predicted from
the environment's irreversible path statistics *in that environment family*.

A foundational result would be a time-symmetric microscopic IF theory that derives a
unique cosmological boundary or dynamical mechanism producing the observed one-sided
thermodynamic and record arrows.

Without the final step, IF Theory explains how the arrow is recorded and used, not why
the universe began with the capacity to point one way.

---

# 40. Relationship to the Informational Battery

Record media require writable states, stabilizing barriers,
low-thermodynamic-entropy preparation, maintenance, and accessible decoding. These are
components of an informational battery

\[
\mathcal B_R=\left(B_{\mathrm{physical}},\;I_{\mathrm{accessible}},\;\mathcal M_{\mathrm{conversion}}\right),
\]

with \(B_{\mathrm{physical}}\) in the energy ledger, \(I_{\mathrm{accessible}}\) in the
Shannon ledger, and \(\mathcal M_{\mathrm{conversion}}\) the declared, implementation-
specific map between them. The three are tracked separately and never summed.

A blank memory is a form of operational capacity. Writing discharges some of that
flexibility into a specific record. Resetting restores reuse capacity at a physical
cost. Thus:

> A record is not free information; it is a constrained physical state produced from a
> writable nonequilibrium capacity.

The battery is a bookkeeping device for an already-existing nonequilibrium resource. It
does not generate that resource, and naming the unexplained cosmological boundary
condition "the informational battery" would be the §37.4 failure under a new label.

---

# 41. Relationship to Agency

Paper 5 defined agency through predictive causal work. Paper 12 adds historical depth.
An agent becomes history-dependent when

\[
P(A_t\mid O_t,M_t)\neq P(A_t\mid O_t),
\]

and becomes reflective when it can inspect and revise how past records influence present
decisions. **Reflection is not free**: inspecting and revising records consumes read,
compute, and rewrite budget, and exports waste heat like any other operation. No rule in
this framework permits reflection or intelligence to reduce thermodynamic entropy
without paying energy.

**Agency does not create the thermodynamic arrow.** It exploits the arrow to construct
memory, models, plans, and commitments.

Scoping: Paper 5's \(\Pi_A\) and \(\Pi_C\) remain available here as **per-family
measurement instruments**, as does the parasite band (ablation-positive ≠
competitive-positive), a structural consequence of the break-even inequality. Neither
confers a universal constant on record-bearing agents, and no threshold measured in a
record experiment may be quoted as substrate-independent.

---

# 42. Relationship to Mortality

Paper 6 defined the self as a costly continuity strategy. Records allow identity to
persist through component replacement, damage, learning, reproduction, and institutional
transfer. Mortality destroys some records and transfers others. A lineage can preserve
historical constraint beyond an individual lifespan through inherited structure, copied
memory, artifacts, social archives, and environmental modification. This creates a
hierarchy:

\[
\text{molecular record}\rightarrow\text{organism memory}\rightarrow\text{social record}\rightarrow\text{institutional history}.
\]

Each level pays its own maintenance cost; none is a free carry.

---

# 43. Relationship to Cosmology

Cosmic expansion may alter available phase space, gravitational clumping, radiation
dilution, horizon accessibility, and record preservation. **None of this shows that
record accumulation causes expansion.**

A cosmological IF theory must derive both \(H(a)\) and \(\dot{\mathcal C}_H(a)\) from
one covariant action and then make a **prospective** prediction. A correlation between
them is insufficient.

The cosmology branch runs parallel-under-discipline and is firewalled from the agency
branch: it is currently unpreregistered and untested, and its plausibility is assessed
internally as low against very high stakes. Nothing in Paper 12 may be cited as
cosmological evidence, and the §24 toy expansion study in particular is not cosmology.

---

# 44. Relationship to Consciousness

Conscious experience appears temporally structured, but Paper 12 addresses only
**functional** memory. A system may possess records, retrodiction, prediction,
self-history, and temporal ordering without phenomenal consciousness, and this paper
makes no phenomenal claim whatsoever. The later consciousness work must test, by
workspace ablation, whether globally accessible counterfactual self-models add
explanatory or predictive power beyond record-bearing agency. Functional claims only; no
phenomenal promises.

---

# 45. Interpretive Layer — Pointer Only

**Layer firewall (CLAUDE.md §6).** The scientific framework of this paper can measure
irreversibility, records, causal continuity, historical constraint, memory, and temporal
asymmetry. It **cannot** experimentally identify divine purpose, providence, ultimate
meaning, directionality of life, or God as the source of time.

A theological or philosophical reading — that an ordered creation, lawful causation, and
meaningful history express intention, or that the growth of historical constraint is
what makes a life *a story* — is a **meaning-layer** interpretation. It lives in
[`canon/30-meaning/`](../30-meaning/) and is **not asserted here as a physical result**.
No measurement in this paper supports or refutes it, and no result in this paper should
be quoted in support of it. Where the meaning layer discusses agency-preserving
cooperation, the science layer uses that phrase and no other.

---

# 46. Criteria for Rejection or Major Revision

The Paper 12 framework should be rejected or substantially revised if:

1. path irreversibility cannot be measured consistently;
2. record correlations fail causal interventions;
3. record creation does not align with thermodynamic entropy production under the
   stated conditions;
4. historical constraint is dominated by redundancy or estimator bias;
5. complete memory-cycle accounting contradicts the proposed trade-offs;
6. reliable records form generically in both orientations without special conditions;
7. agent memory has no causal effect under marginal-preserving ablation;
8. local arrow reversals require no compensating resource;
9. coarse-graining changes the inferred arrow arbitrarily;
10. the measurement protocol cannot be instantiated in a new substrate without
    redefining what is measured (TA-H12′ falsifier);
11. record accumulation is repeatedly used to conceal an unexplained
    low-thermodynamic-entropy boundary;
12. the framework claims that choices create energy or fundamental time without a
    physical derivation.

Item 10 replaces the extracted draft's "cross-substrate relationships fail," which after
2026-07-18 is no longer a rejection criterion — cross-substrate disagreement is now the
expected default.

---

# 47. Conclusion

Time and the arrow of time are not the same concept. A dynamical law may contain a time
parameter while remaining symmetric under reversal. A macroscopic arrow appears when
forward and reversed histories cease to be equally probable under the actual conditions:

\[
\Sigma[\Gamma]=\ln\frac{P_F[\Gamma]}{P_R[\Gamma^\dagger]} .
\]

Physical records deepen that asymmetry. A causal record is a stable present state
generated by an earlier event, robust under a declared perturbation class, and
accessible to a later system. Its historical constraint is the Shannon-ledger quantity

\[
\mathcal C_H=I(H_{<t};R_t).
\]

Records create an experienced distinction between a past narrowed by surviving traces
and a future represented by multiple compatible continuations. Agents use those records
to predict, repair, preserve identity, and choose actions that create additional traces —
paying, at every step, in energy and exported waste heat.

The strongest defensible IF conclusion is:

> The arrow experienced by embedded agents is physically amplified by the irreversible
> production of robust causal records and their accumulation into historical constraint.

The stronger statement remains unresolved:

> Record accumulation alone explains why the universe has one global arrow rather than
> two or none.

Without a derivation of the asymmetric boundary condition, IF Theory has **not** solved
the arrow-of-time problem. It has instead defined a measurable research program
connecting

\[
\text{thermodynamic entropy production}\rightarrow\text{records}\rightarrow\text{memory}
\rightarrow\text{historical identity}\rightarrow\text{agency},
\]

whose every link is falsifiable within a named substrate and none of which is claimed
to hold with the same numbers across substrates. That bridge is valuable precisely
because it can fail at each step — and on 2026-07-18, on the neighboring branch of this
program, a comparable bridge did fail, publicly and by a pre-registered rule. This
paper is written to be capable of the same.

---

# References

Attribution is by author and result. Where the extracted draft's citation metadata was
transcript residue, it has been removed rather than reconstructed.

1. Crooks, G. E. "Entropy Production Fluctuation Theorem and the Nonequilibrium Work
   Relation for Free Energy Differences." *Physical Review E* 60, 2721–2726 (1999).
2. Batalhão, T. B. et al. "Irreversibility and the Arrow of Time in a Quenched Quantum
   System." *Physical Review Letters* 115, 190601 (2015).
3. Goldstein, S., Tumulka, R. and Zanghì, N. "Is the Hypothesis About a Low Entropy
   Initial State of the Universe Necessary for Explaining the Arrow of Time?" (2016).
4. Price, H. "The Thermodynamic Arrow: Puzzles and Pseudo-Puzzles." (2004).
5. Wolpert, D. H. and Kipper, J. "Memory Systems, the Epistemic Arrow of Time, and the
   Second Law." *Entropy* 26, 170 (2024).
6. Mlodinow, L. and Brun, T. A. "On the Relation Between the Psychological and
   Thermodynamic Arrows of Time." *Physical Review E* 89, 052102 (2014).
7. Jun, Y., Gavrilov, M. and Bechhoefer, J. "High-Precision Test of Landauer's Principle
   in a Feedback Trap." *Physical Review Letters* 113, 190601 (2014).
8. Sagawa, T. and Ueda, M. "Minimal Energy Cost for Thermodynamic Information Processing:
   Measurement and Information Erasure." *Physical Review Letters* 102, 250602 (2009).
9. Shalizi, C. R. and Crutchfield, J. P. "Computational Mechanics: Pattern and
   Prediction, Structure and Simplicity." *Journal of Statistical Physics* 104, 817–879
   (2001).
10. Ollivier, H., Poulin, D. and Zurek, W. H. "Environment as a Witness: Selective
    Proliferation of Information and Emergence of Objectivity in a Quantum Universe."
    *Physical Review A* 72, 042113 (2005).
11. Blume-Kohout, R. and Zurek, W. H. "Quantum Darwinism: Entanglement, Branches, and the
    Emergent Classicality of Redundantly Stored Quantum Information." *Physical Review A*
    73, 062310 (2006).
12. Riedel, C. J. and Zurek, W. H. "Quantum Darwinism in an Everyday Environment: Huge
    Redundancy in Scattered Photons." *Physical Review Letters* 105, 020404 (2010).
13. Jennings, D. and Rudolph, T. "Entanglement and the Thermodynamic Arrow of Time."
    *Physical Review E* 81, 061130 (2010).
14. Partovi, M. H. "Entanglement versus Stosszahlansatz: Disappearance of the
    Thermodynamic Arrow in a High-Correlation Environment." *Physical Review E* 77,
    021110 (2008).

---

**Cross-references:** [`canon/papers/P15-falsification-of-universality.md`](P15-falsification-of-universality.md)
(the IF-H1 kill and the estimator trilemma) ·
[`canon/papers/P05-agency-threshold.md`](P05-agency-threshold.md) (\(\Pi_A\), \(\Pi_C\),
the parasite band) · `canon/00-foundations/04-break-even-theorem.md` (the break-even
inequality as an accounting identity) · `canon/papers/P06-memory-repair-mortality.md` (repair and
costly continuity) · `canon/papers/P04-expansion-complexity-window.md` (the toy growth schedule
used in §24) · `canon/30-meaning/` (interpretive layer — pointer only, §45) ·
`SCOREBOARD.md` §Kill log.
