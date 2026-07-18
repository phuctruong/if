# Emergent Structure in Resource-Conserving IF Universes
## Minimal Local Dynamics, Objective Structure Detection, and Tests Against Designed Emergence

**Author:** Phuc Vinh Truong
**Series:** IF Theory Canonical Papers
**Paper:** 3
**Layer:** SCIENCE
**Status:** canonical revision of extracted draft (2026-07-18)
**Supersedes:** canon/extracted/paper-03-extracted.md

---

## Abstract

IF Theory proposes that persistent organization, life-like self-maintenance, and eventually agency might emerge from local physical interactions governed by explicit resource constraints. Demonstrating visually interesting patterns is insufficient. Cellular automata, reaction–diffusion systems, continuous artificial-life substrates, and trained neural cellular automata already produce complex morphologies, locomotion, regeneration, and self-organized behavior. The scientific challenge is to determine whether organization emerges without being inserted through target templates, organism-specific variables, hand-designed seeds, hidden energy sources, or reward functions that directly favor the desired result — the constitutional "no intelligence in the primitives" gate (Conway gate; CLAUDE.md §5, forbidden state `TELEOLOGY_INJECTION`).

This paper specifies a family of **resource-conserving IF universes** designed to test that question. Primitive elements possess local physical states, conserved material, finite high-grade resource capacity, degraded resource or waste, and globally shared local transition rules. They do **not** possess primitive variables named life, organism, boundary, fitness, memory, reflection, consciousness, cooperation, or love.

Two reference implementations are proposed. **IF-RC0** is a deterministic accounting model with exact conservation of material and abstract capacity tokens — for software validation, exhaustive rule search, and reproducible phase mapping. **IF-RC1** is a stochastic-thermodynamic extension whose transition rates satisfy local detailed-balance constraints, permitting explicit calculation of work, heat, and thermodynamic-entropy production. RC0 may demonstrate *computational* (ledger) resource conservation only; only RC1 may support *thermodynamic* claims. The three-ledger discipline (energy / thermodynamic entropy / information, never merged — CLAUDE.md §1) is load-bearing here: an abstract RC0 capacity token is neither a joule nor a bit, and calling it one is the forbidden state `ENTROPY_CONFLATION`.

Structures are identified through a preregistered, target-independent pipeline based on localization, temporal persistence, boundary stability, material and causal continuity, resource throughput, and perturbation response. Self-repair is measured against an undamaged counterfactual twin and against passive-relaxation controls. Replication requires dynamically independent descendants that inherit a reproducible organization, not merely temporary fragmentation or spreading activity.

The principal hypotheses are that localized persistent structures occupy reproducible regions of rule space; that some structures maintain themselves through resource throughput rather than static stability alone; that repair can arise without a repair objective; and that detected structures can exhibit stronger causal and predictive closure than arbitrary matched regions. Each hypothesis carries an explicit falsifier (CLAUDE.md §2).

The proposed novelty is not the generation of life-like cellular-automaton patterns — Lenia and mass-conserving Flow-Lenia already generate spatially localized artificial lifeforms and evolutionary dynamics, and neural cellular automata can grow and repair prescribed forms. The possible IF contribution is a combined protocol requiring exact resource accounting, target-free discovery, objective post hoc detection, causal interventions, held-out replication, and agreement across substantially different substrates. If those requirements cannot be met, the claim that IF structures emerge autonomously must be rejected.

---

## Keywords

Artificial life; cellular automata; self-organization; resource conservation; morphogenesis; causal emergence; self-repair; replication; nonequilibrium systems; phase diagrams; reproducibility.

---

# 1. Introduction

A small collection of local rules can generate patterns that appear organized. Turing's reaction–diffusion framework showed mathematically that diffusion plus local chemical interaction can destabilize a uniform state and produce spatial patterning. Cellular automata later supplied discrete substrates in which persistent objects, oscillators, moving patterns, and self-reproduction could be studied through local updates. These traditions establish that global pattern need not require global command.

Modern artificial-life systems extend the result. Lenia is a continuous cellular-automaton family supporting many localized, resilient, motile patterns; Flow-Lenia adds mass conservation and localized rule parameters, enabling interacting forms and emergent evolutionary dynamics. Neural cellular automata can be trained to grow a specified target image and regenerate it after damage. Automated search (diversity search, curiosity-driven exploration, foundation-model proposal) can discover previously unseen artificial-life behavior.

These achievements set a strict novelty boundary. It is **not** sufficient to show a beautiful moving pattern, a pattern that survives many updates, a trained system that reconstructs a target, a mass-conserving automaton, a parameter sweep containing organism-like forms, or an AI agent that labels a simulation "alive." The scientific question is narrower:

> Can persistent, resource-processing, perturbation-resistant structures arise from minimal local laws **without** their forms, boundaries, repair objectives, or fitness functions being prescribed?

and a second, dependent question:

> Can those structures be detected and tracked by a reproducible procedure that does not depend on human visual preference?

This paper defines the computational system and experimental controls required to answer both.

---

# 2. Scientific Scope

Paper 3 concerns the emergence of **structure** and preliminary life-like properties. It does not claim to establish biological life, predictive agency, consciousness, moral behavior, cosmic expansion, fundamental particle physics, quantum gravity, or divine purpose. The layer firewall (CLAUDE.md §6) holds: any interpretive reading of a discovered structure as "alive," "caring," or "meaningful" belongs in `canon/30-meaning/` and may not leak back into this science doc.

A persistent structure discovered here may become a *candidate* for later agency experiments; it is not an agent merely because it moves, persists, or repairs. The intended progression is

```
local dynamics → localized structure → self-maintenance → repair → replication → agency tests,
```

each arrow a separate operational test (agency itself is Papers 2 and 5).

---

# 3. Prior Art and the Novelty Boundary

**3.1 Reaction–diffusion pattern formation.** Turing's morphogenesis model showed a homogeneous chemical state can become unstable under coupled reaction and diffusion. Spontaneous symmetry breaking and pattern formation therefore do not by themselves demonstrate life or agency. IF cannot claim novelty for producing spots, stripes, waves, or oscillations (`NOVELTY_INFLATION` guardrail).

**3.2 Classical and continuous cellular automata.** Lenia generalized CA dynamics to continuous states, time, and neighborhoods, producing localized patterns with movement, resilience, and complex morphology. IF cannot classify a simulation as novel merely because it contains a persistent moving "creature."

**3.3 Mass-conserving artificial life.** Flow-Lenia is a mass-conserving Lenia extension demonstrating localized patterns, coexisting forms, parameter localization, and emergent evolution. Resource/mass conservation alone is not an IF innovation. The IF standard is stronger: conservation *plus* a declared capacity ledger, a high-grade/degraded resource distinction, target-free discovery, and intervention-based tests of maintenance and repair.

**3.4 Trained morphogenesis and regeneration.** Growing neural cellular automata learn a shared local rule that grows and regenerates a target — but the desired form is present in the training loss. IF distinguishes **targeted regeneration** (distributed control of a prescribed form) from **spontaneous self-repair** (emergence of the goal of preserving a form the rule search never named).

**3.5 Automated discovery of artificial life.** Diversity search, curiosity-driven exploration, and foundation-model methods have uncovered diverse CA behavior. IF may use these for *exploration*, but AI-generated aesthetic or semantic judgments cannot be the primary scientific metric: the discovery system must be separated from the confirmatory evaluator (CLAUDE.md §5 corollary — agency is detected, not declared).

**3.6 Conservation laws in cellular automata.** Conservation laws in CA can be defined and tested mathematically; lattice-gas and reversible-CA frameworks give exact local transport and microscopic reversibility. IF cannot infer physical legitimacy from merely *using* a CA — its specific update rules must satisfy their declared conservation identities.

**3.7 Statistical adaptation.** Nonequilibrium statistical physics shows driven systems can become statistically associated with structures adapted to their forcing. This does not establish agency, but shows drive and dissipation can bias which organized states form. IF must therefore separate adaptation-like selection among physical states from genuine endogenous control.

**3.8 Provisional novelty claim.** The prospective contribution is not any single ingredient but the combined protocol:

> minimal globally shared local laws + exact resource accounting + no target morphology + automatic structure detection + counterfactual repair testing + held-out replication + cross-substrate confirmation.

This is a proposed methodological synthesis; novelty exists only if the resulting experiments discover a robust phenomenon not already explained by simpler artificial-life systems.

---

# 4. Design Principles

**4.1 Nothing life-like is primitive.** The primitive state may contain material, energetic/capacity state, local configuration, position or graph connectivity, and interaction channels. It may **not** contain `is_alive`, `organism_id`, `body`, `boundary`, `fitness`, `repair_target`, `memory_score`, `intelligence`, `reflection`, `consciousness`, `cooperation_bonus`, or `love`. Candidate organisms are interpretations generated *after* the dynamics run. (Direct enforcement of the Conway gate.)

**4.2 Locality.** Each update depends only on a declared local causal neighborhood. For site/node \(i\):

\[
z_i(t+1)=T_\theta\!\left[z_i(t),\{z_j(t):j\in\mathcal N_i(t)\},\xi_i(t)\right],
\]

with local state \(z_i\), causal neighborhood \(\mathcal N_i\), globally shared rule \(\theta\), and optional declared noise \(\xi_i\). No update may query the complete grid, the location of a target shape, whether a cell belongs to a detected organism, or a global score instructing it to repair.

**4.3 Translation and orientation neutrality.** Primary reference models should not privilege an absolute location; where possible rules should also be rotation-equivariant, reflection-equivariant, and permutation-invariant over equivalent neighbors. Any broken symmetry must be declared.

**4.4 Exact resource accounting.** Every material and capacity flow must be attributable to local transport, local transformation, environmental input, or environmental export. Numerical clipping must not silently create or destroy resources (`PERPETUAL_RECHARGE` guardrail).

**4.5 Discovery and confirmation must be separated.** Search may identify interesting rules in a discovery set; scientific claims must be evaluated using held-out seeds, perturbations, and parameter neighborhoods, independently written evaluators, and frozen classification criteria.

**4.6 No single beauty score.** The simulator must not classify structures through one opaque "life-likeness" score. It reports a vector of independently interpretable measurements

\[
\mathbf S=[L,P,B,T_R,C,R,D,N]
\]

(localization, persistence, boundary strength, throughput, causal closure, repair, descendant formation, novelty).

---

# 5. General IF Universe

Let the simulated universe at discrete time \(t\) be \(\mathcal U_t=(G_t,Z_t,\mathcal R_t,\Lambda)\), where \(G_t=(V_t,E_t)\) is a lattice or interaction graph, \(Z_t=\{z_i(t)\}\) the local states, \(\mathcal R_t\) the global resource ledgers, and \(\Lambda\) the external boundary conditions. For Paper 3 the domain is fixed; **dynamic growth of space is deferred to Paper 4** — this paper does not activate or partition domain (avoiding the `COMMANDED_EXPANSION` forbidden state, which Paper 4 addresses explicitly).

Each local state is \(z_i=(m_i^1,\ldots,m_i^K,f_i,w_i,s_i)\), with conserved material \(m_i^a\ge0\) of channel \(a\), high-grade resource \(f_i\ge0\), degraded resource/waste \(w_i\ge0\), and finite local configuration \(s_i\in\mathcal S\). "Species" here denotes a simulation channel, not a biological species.

---

# 6. Two Reference Implementations

**6.1 IF-RC0: deterministic resource accounting.** For exact software testing, deterministic replay, large rule sweeps, phase classification, debugging, and proof of resource conservation, using abstract capacity units. Its claim is *the programmed resource ledger closes* — **not** *the model is a complete physical thermodynamic universe*.

**6.2 IF-RC1: stochastic-thermodynamic extension.** Local transitions as stochastic processes with declared state energies, heat reservoirs, chemical/resource potentials, local detailed-balance relations, and trajectory-level work and heat. RC1 supports claims about thermodynamic-entropy production, nonequilibrium maintenance, work extraction, and physical cost. Any RC0 result must be re-evaluated in RC1 before thermodynamic interpretation — the ledger firewall between the information/accounting ledger (RC0) and the energy + thermodynamic-entropy ledgers (RC1) is not optional.

---

# 7. IF-RC0 Dynamics

**7.1 Conserved material transport.** Let \(J_{ij}^a(t)\) be the net material flux of channel \(a\) from \(i\) to \(j\), with antisymmetry \(J_{ij}^a=-J_{ji}^a\). The update

\[
m_i^a(t+1)=m_i^a(t)-\sum_{j\in\mathcal N_i}J_{ij}^a(t)
\]

gives \(\sum_i m_i^a(t+1)=\sum_i m_i^a(t)\) under closed boundaries, so \(M_a=\sum_i m_i^a=\text{const}\).

**7.2 Pairwise flux rule.** A general local flux \(J_{ij}^a=\operatorname{clip}[\kappa_a(\mu_i^a-\mu_j^a),-J_{\max},J_{\max}]\), with local potential \(\mu_i^a=F_\theta^a(z_i,\operatorname{Agg}\{z_j:j\in\mathcal N_i\})\). Each edge flux is computed once and applied with opposite signs to preserve antisymmetry; aggregation must be local and symmetry-compatible.

**7.3 Resource conversion.** Let \(c_i(t)\ge0\) be high-grade resource consumed locally and \(0\le\eta_i(t)\le1\) the fraction credited to declared useful work. Then

\[
f_i(t+1)=f_i(t)-c_i(t)+I_i^f(t),\qquad
w_i(t+1)=w_i(t)+[1-\eta_i(t)]c_i(t)-O_i^w(t),
\]
\[
W_{\mathrm{out}}(t+1)=W_{\mathrm{out}}(t)+\sum_i\eta_i(t)c_i(t),
\]

with external input \(I_i^f\) and declared waste sink \(O_i^w\). For a closed RC0 universe \(I_i^f=O_i^w=0\), giving the abstract capacity balance

\[
\sum_i f_i(t)+\sum_i w_i(t)+W_{\mathrm{out}}(t)=C_{\mathrm{total}}.
\]

This is a **simulation** conservation law (information/accounting ledger). Physical interpretation requires RC1.

**7.4 Configuration-state transitions.** A transition \(s_i(t)\to s_i(t+1)\) has cost \(c_s[s_i(t),s_i(t+1)]\ge0\) and may occur only when \(f_i(t)\ge c_s\). The globally shared rule proposes the transition without knowing whether the node belongs to a structure.

**7.5 Driven mode.** Long-lived self-maintaining structures generally require continuing access to a gradient: fuel enters through fixed/stochastic source regions, waste exits through declared sinks, source and sink rules are external boundary conditions, and all imported/exported capacity is logged. A structure may reorganize how effectively it intercepts these flows but cannot alter the ledger.

---

# 8. IF-RC1 Thermodynamic Dynamics

For local microstates \(x,y\), a transition \(x\to y\) at rate \(k_{xy}\) coupled to a heat bath at inverse temperature \(\beta\) satisfies a local-detailed-balance relation of the form \(\ln(k_{xy}/k_{yx})=\beta[W_{xy}-\Delta E_{xy}]\) (exact expression depending on modeled reservoirs and sign conventions). For each trajectory \(\Gamma\) record \(W[\Gamma],\,Q[\Gamma],\,\Delta S_{\mathrm{sys}}[\Gamma],\,\Delta S_{\mathrm{env}}[\Gamma]\). Mean total **thermodynamic-entropy** production must satisfy

\[
\langle\Delta S_{\mathrm{tot}}\rangle=\langle\Delta S_{\mathrm{sys}}+\Delta S_{\mathrm{env}}\rangle\ge0.
\]

A structure may reduce its internal entropy or hold a narrow state distribution **only** while exporting thermodynamic entropy or consuming free energy from the environment. (This is the `PERPETUAL_RECHARGE` guardrail at the trajectory level: order maintained locally is always paid for in exported waste.)

---

# 9. Initial Conditions

**9.1 Homogeneous null.** All sites in the same state, no noise — tests whether the deterministic rule preserves exact symmetry. A translation-symmetric rule should not spontaneously break perfect symmetry absent noise, asynchronous updates, numerical-asymmetry instability, or asymmetric boundaries; unexpected structure here may indicate a bug.

**9.2 Perturbed homogeneous state.** \(z_i(0)=z_0+\epsilon_i\), \(\epsilon_i\) sampled from a declared distribution — tests spontaneous amplification of generic fluctuations.

**9.3 Generic sparse seeds.** A few sites receive random material/configuration perturbations, **not** designed to resemble the eventual structure, tested over many seed realizations.

---

# 10. What Counts as a Structure?

Visual inspection is insufficient. A candidate is a temporally tracked region \(A_t\subseteq V\) satisfying preregistered criteria.

**10.1 Activity field.** \(a_i(t)=d_z[z_i(t),z_{\mathrm{bg}}(t)]\) with fixed state-space distance \(d_z\) and background \(z_{\mathrm{bg}}\) estimated *without* a target pattern. Detection threshold from null simulations: \(a_i>a_{\mathrm{thr}}\), where \(a_{\mathrm{thr}}\) may be the \(1-\alpha\) quantile of the null activity distribution.

**10.2 Spatial localization.** Effective size \(N_{\mathrm{eff}}=(\sum_i a_i)^2/\sum_i a_i^2\); normalized localization \(L=1-N_{\mathrm{eff}}/|V|\). Localization alone does not imply structure — a single static spike is localized.

**10.3 Connected candidate regions.** Group active sites by connected components, density-based clustering, persistent-homology features, or graph communities. The primary method is frozen before confirmation; alternatives are robustness checks.

**10.4 Temporal tracking.** Match regions at \(t\) and \(t+1\) by a cost incorporating material overlap, predicted displacement, state-distribution similarity, optimal transport, and shape-independent composition. A moving structure should not lose identity merely by changing location; matching may use the Hungarian algorithm or a min-cost-flow lineage graph.

**10.5 Persistence.** Lifetime \(\tau_A\) qualifies as persistent only if \(\tau_A>\tau_{\mathrm{null}}\), a preregistered high quantile of matched null-dynamics lifetimes. No universal lifetime threshold is assumed.

**10.6 Boundary strength.** With internal edges \(\mathcal E_{\mathrm{in}}\) and crossing edges \(\mathcal E_{\mathrm{out}}\), \(C_{\mathrm{in}}=\sum_{\mathcal E_{\mathrm{in}}}|J_{ij}|\), \(C_{\mathrm{cross}}=\sum_{\mathcal E_{\mathrm{out}}}|J_{ij}|\), and \(B_A=C_{\mathrm{in}}/(C_{\mathrm{in}}+C_{\mathrm{cross}})\). High internal coupling can indicate integration, but an impermeable inert object may also score high — interpret with throughput and persistence.

**10.7 Resource throughput.** With imported fuel \(F_{\mathrm{in}}^A\), exported waste \(W_{\mathrm{out}}^A\), and internal conversion \(C_A\), a self-maintaining structure exhibits sustained throughput \(T_A=\frac1\tau\int_t^{t+\tau}C_A(t')\,dt'>0\). A static crystal-like pattern may persist with \(T_A\approx0\) — that is persistent structure, not metabolic self-maintenance.

---

# 11. Structure Classification Vector

For every candidate report

\[
\mathbf S_A=[L_A,\tau_A,B_A,T_A,M_A,R_A,D_A,C_A]
\]

(localization, persistence, boundary strength, throughput, motility, repair response, descendant/replication evidence, causal-closure evidence). No universal scalar is initially formed — this prevents arbitrary weights from converting a weak result into a high "life score."

---

# 12. Phase Taxonomy

- **P0 — Extinction:** activity/nonequilibrium structure decays to the null background.
- **P1 — Homogeneous equilibrium:** the domain approaches a spatially homogeneous stationary state.
- **P2 — Frozen pattern:** persistent organization with negligible throughput or adaptation.
- **P3 — Distributed turbulence:** high activity, no stable localized structures.
- **P4 — Transient localization:** localized structures form but remain within the null lifetime distribution.
- **P5 — Persistent localization:** localized structures survive substantially longer than matched null structures.
- **P6 — Throughput-maintained structures:** persistent structures consume resources and export degraded products while maintaining organization.
- **P7 — Motile structures:** displacement not explained by diffusion or global drift.
- **P8 — Repair-capable structures:** after controlled damage, return toward the undamaged counterfactual trajectory more strongly than matched passive controls.
- **P9 — Replicating structures:** dynamically independent descendants inheriting reproducible organization.
- **P10 — Candidate adaptive structures:** behavior modified across environments in a way later shown to depend causally on internal state.

Paper 3 may identify phases P0–P9. Predictive agency remains the subject of Papers 2 and 5.

---

# 13. Measuring Motility

For center of mass \(\mathbf r_A(t)=\sum_{i\in A_t}a_i(t)\mathbf r_i/\sum_{i\in A_t}a_i(t)\), define mean squared displacement \(\operatorname{MSD}_A(\Delta t)=\langle|\mathbf r_A(t+\Delta t)-\mathbf r_A(t)|^2\rangle\), compared against passive diffusion, environmental drift, randomized-phase controls, and background material transport. Directed motility requires displacement beyond these nulls; movement alone does not establish agency.

---

# 14. Measuring Self-Repair

**14.1 Counterfactual twin design.** At perturbation time \(t_d\), clone the complete universe state; run an undamaged control \(U^{(0)}\) and a damaged \(U^{(D)}\), applying a localized intervention only to candidate \(A\) in \(U^{(D)}\), with identical environmental inputs, noise streams (where possible), boundary conditions, and update rules.

**14.2 Damage interventions.** Deletion of a material fraction, randomization of local configuration states, boundary puncture, component displacement, resource deprivation, targeted removal of high-flux nodes, random removal of matched size. Magnitude recorded as \(d\in[0,1]\).

**14.3 Macrostate distance.** \(D_A(U^{(D)}_t,U^{(0)}_t)\) after optimizing over irrelevant translation, rotation, and labeling symmetries; components may include material-distribution Wasserstein distance, state-distribution divergence, boundary mismatch, throughput mismatch, dynamical-mode mismatch.

**14.4 Recovery score.**

\[
R_A(\tau)=1-\frac{D_A(U^{(D)}_{t_d+\tau},U^{(0)}_{t_d+\tau})}{D_A(U^{(D)}_{t_d},U^{(0)}_{t_d})}.
\]

\(R_A=1\): complete return to the undamaged macrotrajectory; \(R_A=0\): no reduction; \(R_A<0\): divergence increased.

**14.5 Passive-recovery controls.** Self-repair must outperform passive diffusion, equilibrium relaxation, matched nonpersistent patterns, randomized local rules, and undriven material aggregation. A structure returning to an attractor because every state relaxes there demonstrates stability, not necessarily active repair.

**14.6 No repair optimization.** For the strongest claim the rule-search objective must not include target-image loss, post-damage reconstruction, structure-specific recovery, or damage episodes; repair is evaluated only after rule selection. A separate experiment may deliberately evolve repair, but must be labeled **selected repair**, not spontaneous repair.

---

# 15. Measuring Replication

Replication is easily confused with growth, fragmentation, diffusion, repeated environmental nucleation, or periodic pattern generation. A claim requires all of:

**15.1 Parent identification** — a persistent candidate \(A\) exists before descendant formation.
**15.2 Material or causal lineage** — measurable transfer from parent process to descendants.
**15.3 Organizational inheritance** — descendants reproduce a dynamical organization, not merely share material: with parent signature \(\Sigma_A\), require \(d_\Sigma(\Sigma_{\mathrm{child}},\Sigma_{\mathrm{parent}})<\epsilon_\Sigma\).
**15.4 Independent persistence** — after separation, parent and child continue as independently tracked structures for a minimum null-adjusted interval.
**15.5 Repetition** — at least one descendant retains the capacity to produce another under comparable conditions. Without repeated lineage, the event is reproduction-like fragmentation, not demonstrated replication.

---

# 16. Causal Individuality

A structure may be spatially localized but causally dominated by its environment. IF tests whether a candidate is a meaningful macro-unit.

**16.1 Predictive closure.** With candidate macrostate \(A_t\) and local environment \(E_t\), compare \(I(A_t;A_{t+\tau})\) against \(I(E_t;A_{t+\tau}\mid A_t)\). Greater internal predictive continuity indicates more self-determination, though this remains correlational.

**16.2 Intervention tests.** Matched interventions on internal candidate states, nearby environmental states, and arbitrary equal-size regions; measure changes in candidate persistence, future macrostate, throughput, and movement. A genuine unit shows a reproducible intervention structure distinct from arbitrary partitions.

**16.3 Causal emergence.** Micro- and macro-state causal models compared via intervention-based effective-information measures. A macrodescription predicting interventions more selectively than the noisy/degenerate microdescription may possess causal-emergence evidence — justifying treating the structure as a useful causal unit, not proving life.

---

# 17. Rule Complexity

A million-parameter neural rule may generate impressive structures while providing little evidence that *minimal* laws suffice. Every rule reports a complexity measure — number of local state variables, neighborhood radius, transition-table size, parameter count, description length, compressed source length, or circuit complexity. Let \(K(\theta)\) be a declared rule-complexity estimate; comparisons report both behavior and \(K(\theta)\). The objective is not the shortest rule but to avoid hiding the organism inside a vast rule table (`RULE_TABLE_ORGANISM`, §24.11).

---

# 18. Search Strategy

**18.1 Discovery stage.** Random/Latin-hypercube sampling, Bayesian optimization, evolutionary novelty search, quality-diversity, curiosity-driven exploration, AI-assisted proposal. Automated ALife search (foundation models, curiosity-driven AI scientists) is used as a discovery tool, **not** as proof.

**18.2 Discovery objectives.** Permitted broad objectives: behavioral diversity, temporal nonstationarity, localization diversity, multiscale entropy, compression complexity, novelty relative to prior runs. For the strongest repair claim, discovery may not optimize the repair metric; for the strongest replication claim, discovery may not directly optimize descendant count.

**18.3 Confirmation stage.** Before confirmatory runs, freeze the rule, the detector, the classification criteria, and parameter ranges; reserve unseen seeds/perturbations; assign an independent evaluator. No manual deletion of failed seeds.

**18.4 Neighborhood robustness.** \(\rho_\theta=P_{\theta'\sim\mathcal N(\theta,\Sigma)}[\text{behavior persists}]\). A phenomenon at one isolated floating-point setting may be interesting but physically fragile.

---

# 19. Core Hypotheses

**ES-H1 — Spontaneous-localization.** Some resource-conserving local-rule families generate persistent localized structures from generic perturbations without target morphology or hand-designed seeds. *Falsifier:* all persistent localized structures require specially constructed initial patterns, explicit target optimization, or isolated parameter values that fail under minimal perturbation.

**ES-H2 — Throughput-maintenance.** Some persistent structures maintain organization through continuing resource throughput rather than passive static stability. *Prediction:* removing the resource gradient causes loss of dynamic maintenance after a characteristic depletion time. *Falsifier:* persistence is unaffected by resource removal (structure is static, or resource dependence misidentified).

**ES-H3 — Spontaneous-repair.** Some structures recover from novel damage despite the rule search never evaluating damage or reconstruction. *Falsifier:* recovery disappears under held-out damage types, matched passive-relaxation controls, translation-invariant macrostate comparison, or unseen seeds.

**ES-H4 — Causal-individuality.** Automatically detected persistent structures exhibit stronger internal predictive and intervention-based closure than arbitrary matched regions. *Falsifier:* candidate boundaries provide no more causal coherence than random or geometrically similar partitions.

**ES-H5 — Replication.** Some resource-conserving rules generate persistent structures capable of producing dynamically independent descendants with inherited organization. *Falsifier:* apparent replication reduces to fragmentation, diffusion, repeated external nucleation, or nonheritable pattern repetition.

**ES-H6 — Robust-phase.** Persistent and repair-capable structures occupy finite regions of parameter space, not isolated fine-tuned points. *Falsifier:* the phenomenon disappears under small parameter, numerical, or seed changes.

**ES-H7 — Cross-substrate.** At least one qualitative phase boundary transfers between IF-RC0, IF-RC1, and an independently designed conservative substrate. *Falsifier:* every result depends on implementation-specific artifacts.

**ES-H8 — Thermodynamic-maintenance.** In RC1, dynamically maintained organization requires positive environmental thermodynamic-entropy production or consumption of nonequilibrium free energy. *Falsifier:* the structure persistently restores internal order without an accounted compensating flow (a fired `PERPETUAL_RECHARGE`).

---

# 20. Primary Experiments

1. **Conservation audit** — random states/transitions; verify \(\Delta M_a=0\) per channel and \(\Delta(F+W+W_{\mathrm{out}})=0\) for closed RC0. Must pass before pattern search.
2. **Null dynamics** — homogeneous deterministic, randomized, zero-resource, disabled-interaction, and shuffled-rule controls; establish null distributions for localization, lifetime, apparent repair, lineage events, complexity.
3. **Rule-space phase survey** — sweep parameters, classify all outcomes into P0–P9; produce phase diagrams, not selected runs.
4. **Seed independence** — thousands of generic seeds per candidate rule; report formation probability, time, diversity, failure modes.
5. **Gradient dependence** — vary resource-input rate, sink rate, spatial source distribution, environmental volatility; test whether dynamic structures occupy a bounded nonequilibrium region.
6. **Perturbation and repair** — held-out damage classes; recovery vs undamaged twins, passive patterns, randomized-rule controls, nonpersistent structures.
7. **Causal boundary** — intervene on interiors, boundaries, nearby environment, and random matched regions; test distinctive causal significance.
8. **Replication and lineage** — lineage graphs from tracked candidates; require organizational similarity and descendant independence.
9. **Rule simplification** — remove parameters, quantize, reduce radius, prune channels; determine necessary components.
10. **Cross-implementation reproduction** — independently reimplement the rule family; compare phase boundaries, structure statistics, repair behavior, conservation.

---

# 21. Deterministic Jupyter-Notebook Program

Each notebook carries the constitutional contract cell (`Prediction · Baseline · Data · Pass criterion · Falsifier`) and seed 65537.

- **03A — RC0 State and Ledger:** local state, pairwise antisymmetric transport, resource conversion, closed/driven boundaries, exact ledger assertions. Output: \(\max_t|\Delta C_{\mathrm{total}}(t)|\).
- **03B — Conservation Property Tests:** randomized property-based tests for mass conservation, capacity conservation, nonnegativity, translation invariance, neighbor relabeling, zero-coupling limits.
- **03C — Null Universe Catalog:** null distributions for all structure metrics; later detection thresholds originate here.
- **03D — Structure Detector:** activity estimation, thresholding, connected components, optimal-transport tracking, lineage construction, localization/persistence; validated on synthetic moving objects with known ground truth.
- **03E — Minimal Rule Sweep:** low-complexity rule families; complete phase maps, not screenshots.
- **03F — Resource-Gradient Sweep:** behavior over input rate × degradation rate × transport rate.
- **03G — Counterfactual Damage Twins:** paired damaged/undamaged universes; compute \(R_A(\tau)\).
- **03H — Passive-Relaxation Controls:** whether apparent repair exceeds ordinary attractor return.
- **03I — Causal Boundary Tests:** internal vs environmental intervention effects; detected candidates vs matched random regions.
- **03J — Replication and Lineage:** parent formation, split events, descendant persistence, inherited signatures, multigeneration continuation.
- **03K — RC1 Thermodynamic Validation:** small stochastic version with exact state energies and detailed-balance transitions; validate work, heat, entropy production, equilibrium distribution, fluctuation statistics.
- **03L — Cross-Substrate Replication:** IF-RC0, IF-RC1, and an independent mass-conserving continuous or lattice-gas model.
- **03M — Adversarial Audit:** a red-team agent attempts to explain reported structures via threshold choice, numerical clipping, periodic boundaries, finite precision, hand-picked seeds, detector leakage, implicit targets, or uncounted resources.

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

Raw trajectories or deterministic replay instructions are retained for every published figure.

---

# 23. Statistical Standards

**23.1 Unit of analysis** — run, candidate structure, rule, parameter neighborhood, or lineage. Thousands of time steps from one run are not thousands of independent samples.
**23.2 Seed multiplicity** — every confirmatory rule tested across a preregistered number of independent seeds; report success probability, CI, median formation time, failure distribution.
**23.3 Multiple-search correction** — if millions of rules are explored, one extreme pattern is expected; confirmatory evidence comes from held-out neighborhoods, independent reruns, preregistered tests, and second-substrate replication.
**23.4 Detector robustness** — repeat with alternative thresholds, clustering methods, tracking costs, macrostate distances; the result is strong only if qualitative classification is stable.
**23.5 Negative results** — report rule-space regions with no structure, fragile structure, false repair, detector failures, or conservation violations. A phase map of only successes is not credible.

---

# 24. Failure Modes

Target leakage (target in loss, parameters, seed, detector, or intervention design); free resources (clipping/normalization/boundary updates create material or capacity); detector hallucination (tracker joins unrelated fluctuations); periodic-boundary artifact (moving pattern interacts with its own wake); numerical attractor (behavior only at one precision/resolution/update order); human cherry-picking; repair-by-attractor (all damaged states relax to the same fixed pattern); fragmentation called replication; static order called life; search-objective contamination (search optimizes the property later claimed spontaneous); rule-table organism (rule encodes the structure implicitly); thermodynamic overclaim (RC0 tokens described as literal energy/entropy without RC1 — the `ENTROPY_CONFLATION` failure).

---

# 25. What Would Count as Success?

- **Level 1 — Valid conservative substrate:** exact ledgers, reproducible phase diagrams. Necessary but not novel.
- **Level 2 — Target-free persistent localization:** robust localized structures from generic perturbations. An ALife result, related phenomena already exist.
- **Level 3 — Spontaneous repair:** held-out recovery emerges without a repair objective and exceeds passive controls. More significant.
- **Level 4 — Causal individuality:** interventionally meaningful boundaries and macrostate closure. Links self-organization to objective individuality.
- **Level 5 — Reproducible replication:** heritable, dynamically independent descendants across generations without a reproduction objective. A strong origins-of-life result.
- **Level 6 — Cross-substrate law:** one dimensionless phase boundary predicts structure, maintenance, or repair across independent conservative substrates. The most important outcome.

---

# 26. Novelty Assessment

*Already established (not novel IF claims):* local rules can form global patterns; CA can contain persistent moving structures; continuous automata produce organism-like forms; trained CA regenerate targets; mass-conserving automata support evolution; automated search discovers ALife behavior.

*Potential IF novelty (only in the discovered regularity, not the vocabulary):* exact resource accounting; no target morphology; no structure-specific reward; automatic detection; held-out spontaneous repair; interventionally meaningful individuality; multigeneration replication; finite parameter robustness; cross-substrate scaling.

---

# 27. Relationship to the Informational Battery

Paper 1 separated \(B_{\mathrm{gross}}\) from \(B_{\mathrm{op}}\). Paper 3 creates systems in which persistent structures may alter access to resource flows. For a candidate \(A\), later analysis may ask whether its organization increases \(B_{\mathrm{op}}^A\) relative to scrambled or destroyed controls. Paper 3 does not yet call that agency — it establishes the structures on which Paper 2's causal-work interventions can be performed.

---

# 28. Relationship to Agency

A persistent self-repairing structure may lack predictive agency (a stable reaction–diffusion pattern, an autocatalytic cycle, an attractor that reconstructs, a fixed feedback loop). Agency requires evidence that internal information is *used causally* to select actions and improve future outcomes. The progression

\[
\text{persistence}\not\Rightarrow\text{repair}\not\Rightarrow\text{agency}\not\Rightarrow\text{consciousness}
\]

must be tested at each arrow, not assumed. Papers 5 and 6 supply the agency and self-maintenance tests; the causal-work break-even that separates parasite from contributor is stated in `canon/00-foundations/04-break-even-theorem.md`.

---

# 29. Relationship to Biological Life

A successful IF structure would be an artificial-life candidate, not proof that biological life arose through identical rules. Biological relevance would require contact with chemistry, catalysis, compartment formation, heredity, mutation, selection, metabolism, and experimentally measurable reaction networks. The computational substrate is a laboratory for identifying principles, not a substitute for chemistry.

---

# 30. Relationship to Cosmology

This paper does **not** simulate the Big Bang. A lattice or graph with local resource dynamics may illuminate generic relationships among symmetry breaking, structure formation, gradients, persistence, and (in Paper 4) expansion. It does not reproduce spacetime, gravity, quantum fields, CMB fluctuations, primordial nucleosynthesis, or cosmic expansion — those require a separate physical bridge (the cosmology lab, `canon/20-cosmology/`, with its own falsification calendar). Presenting a toy structural result as cosmology is the forbidden state `LAYER_COLLAPSE`.

---

# 31. Criteria for Rejection or Major Revision

Reject or substantially revise if: the resource ledger cannot be made exact; all interesting structures require hand-designed seeds; persistent structures occupy only isolated numerical points; detected structures depend strongly on subjective thresholds; repair disappears under counterfactual-twin analysis; apparent replication is fragmentation or repeated nucleation; causal boundaries are no better than arbitrary partitions; rule complexity encodes the observed behavior; RC0 findings disappear in disciplined RC1 systems; no result survives independent implementation; or simpler existing substrates explain every observation with less machinery. A fired kill goes in `SCOREBOARD.md` §Kill log the same session (CLAUDE.md §7).

---

# 32. Conclusion

The existence of a beautiful pattern is not the scientific result. The result must be a reproducible relationship among local laws, resource constraints, nonequilibrium drive, persistent localization, maintenance, repair, causal individuality, and replication. The proposed IF standard:

> A structure is emergent only when its identity, boundary, maintenance, and recovery are discovered *after* the dynamics, rather than encoded in the primitive state or objective.

The first computational question is whether simple resource-conserving local rules possess finite regions of rule space in which persistent, throughput-maintained, perturbation-resistant structures arise from generic fluctuations. A positive answer would not yet establish life or agency; it would establish a disciplined artificial universe in which those higher transitions can be tested. A negative answer would be equally important — it would show the proposed IF substrate lacks the generative capacity the larger theory requires. The next paper asks whether allowing the causal domain itself to grow creates a reproducible intermediate regime favorable to organization: *The Expansion–Complexity Window in Resource-Conserving Causal Networks* (Paper 4).

---

# References

1. Turing, A. M. "The Chemical Basis of Morphogenesis." *Philosophical Transactions of the Royal Society of London B* 237, 37–72 (1952).
2. Chan, B. W.-C. "Lenia — Biology of Artificial Life." *Complex Systems* 28, 251–286 (2019).
3. Chan, B. W.-C. "Lenia and Expanded Universe." Artificial Life Conference Proceedings (2020).
4. Mordvintsev, A., Randazzo, E., Niklasson, E. and Levin, M. "Growing Neural Cellular Automata." *Distill* (2020).
5. Plantec, E. et al. "Flow-Lenia: Emergent Evolutionary Dynamics in Mass-Conservative Continuous Cellular Automata." *Artificial Life* 31, 228–248 (2025).
6. Hamon, G. et al. "Discovering Sensorimotor Agency in Cellular Automata Using Diversity Search." (2024).
7. Kumar, A. et al. "Automating the Search for Artificial Life with Foundation Models." (2024).
8. Michel, T. et al. "Exploring Flow-Lenia Universes with a Curiosity-Driven AI Scientist." (2025).
9. Pivato, M. "Conservation Laws in Cellular Automata." *Nonlinearity* (2002).
10. Toffoli, T., Capobianco, S. and Mentrasti, P. "When — and How — Can a Cellular Automaton Be Rewritten as a Lattice Gas?" (2007).
11. Perunov, N., Marsland, R. and England, J. "Statistical Physics of Adaptation." *Physical Review X* 6, 021036 (2016).
12. Hoel, E. P., Albantakis, L. and Tononi, G. "Quantifying Causal Emergence Shows That Macro Can Beat Micro." *PNAS* 110, 19790–19795 (2013).
