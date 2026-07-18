<!-- extracted from ChatGPT (driver) on 2026-07-18 -->

# The Expansion–Complexity Window  
## Costly Domain Growth, Causal Coordination, and Sustainable Organization in Resource-Conserving IF Universes

**Author:** Phuc Vinh Truong  
**Series:** IF Theory Working Papers  
**Paper:** 4  
**Date:** July 18, 2026  
**Status:** Theoretical and computational proposal awaiting implementation, preregistration, and falsification

---

## Abstract

Physical and artificial systems often develop while the space available to them changes. Embryos grow while spatial patterns form; ecological populations expand into new territory; developmental graph systems create new nodes and connections; and the physical universe exhibits cosmological expansion. Domain growth can create room for new organization, but it can also dilute interactions, fragment communication, destabilize existing patterns, and consume resources.

This paper proposes the **IF Expansion–Complexity Window Hypothesis**: in resource-conserving artificial universes, persistent causal complexity is maximized over an intermediate range of domain-growth rates. Insufficient growth produces crowding, resource competition, boundary saturation, and loss of structural diversity. Excessive growth causes dilution, falling encounter rates, topological fragmentation, and domain change faster than structures can coordinate or adapt. Between these extremes may exist a finite window in which growth relieves congestion without destroying causal integration.

The central conjecture is not merely that “moderate expansion is best.” It is that the window can be predicted using measurable competing timescales. Let \(g\) denote the fractional rate of domain growth, \(r_{\mathrm{occ}}\) the rate at which active organization occupies available capacity, \(r_{\mathrm{release}}\) the rate at which capacity is returned through decay or turnover, and \(\tau_{\mathrm{coord}}\) the characteristic time required for a persistent structure to propagate information or reorganize across itself. A provisional necessary condition for sustainable complexity is:

\[
\boxed{
r_{\mathrm{occ}}-r_{\mathrm{release}}
<
g
<
\frac{\gamma_c}{\tau_{\mathrm{coord}}},
}
\]

where \(\gamma_c\) is an empirically determined coherence threshold.

The lower bound prevents progressive saturation. The upper bound requires expansion to remain slow enough for causal coordination. The window exists only when the lower bound lies below the upper bound.

Domain growth is not free. New nodes, sites, edges, or spatial degrees of freedom require an explicit cost paid from a declared resource ledger. New space initially contains no unaccounted matter, fuel, or information. In lattice implementations, existing material is conservatively remapped during growth. In graph implementations, a node may divide only by partitioning its existing material and paying the costs of creating and maintaining new connectivity.

The hypothesis will be tested in three reference systems: scheduled conservative lattice growth, endogenous developmental graph growth, and a stochastic-thermodynamic extension. Complexity will not be represented by a single visual score. The preregistered evidence vector includes persistent-structure diversity, causal integration, predictive depth, multiscale dynamical complexity, repair capacity, lineage depth, and useful resource throughput.

The paper explicitly distinguishes this artificial-universe hypothesis from cosmology. An intermediate complexity window in a growing simulation would not demonstrate that cosmic expansion exists to create life or that the observed universe follows IF rules. It would establish a computational relationship between growth, dilution, coordination, and organization that might later motivate—but cannot substitute for—a physical theory.

---

## Keywords

Domain growth; artificial life; cellular automata; causal networks; complexity; criticality; dilution; fragmentation; self-organization; resource conservation; developmental systems; phase transition.

---

# 1. Introduction

Growth changes the conditions under which patterns and organisms form.

A chemical pattern generated on a fixed surface may behave differently when the surface stretches. A population that is stable in a confined habitat may collapse from congestion, spread through newly available territory, or fragment when dispersal becomes too fast. A distributed computational network may gain expressive capacity when new nodes become connected, yet lose coordination if growth outruns communication and maintenance.

Research on reaction–diffusion systems has long recognized that a time-dependent domain changes diffusion, concentration, and pattern selection. Reviews of reaction–diffusion kinetics on growing domains emphasize that domain growth occurs on timescales relevant to biological pattern formation and changes the dynamics relative to otherwise similar fixed-domain systems. A 2024 analysis of Turing pattern formation on growing domains further identified growth rate as a parameter controlling bifurcation and pattern robustness. citeturn856725view0turn856725view1

Artificial developmental systems also permit topology itself to grow. Developmental Graph Cellular Automata begin from small seed graphs and allow nodes, using local information, to divide, remain stable, or be removed. Recent work has trained such systems to grow directed computational reservoirs, demonstrating that distributed local graph-growth rules can produce functional network structures. citeturn856725view2

Network science supplies another relevant result. Connectivity can be too low for global influence to propagate, while high connectivity can suppress functional diversity through overly constrained collective behavior. Recent work on “functional percolation” reported that structural percolation coincides with a sharp expansion of realizable information-processing functions and that functional diversity can peak close to the connectivity transition. citeturn856725view6turn856725view7

These findings support the plausibility of an intermediate organizational regime. They do not establish the IF hypothesis.

The familiar “edge of chaos” idea has proposed that computation and complexity may be enhanced near transitions between ordered and chaotic dynamics. However, early attempts to establish a privileged computational region through cellular-automaton rule parameters were challenged by later reanalyses, which found that the original evidence and interpretation were not generally reliable. IF Theory therefore must not assume that every measure of complexity peaks at criticality or treat a visually attractive intermediate regime as proof of a universal law. citeturn856725view4turn856725view5

The present paper asks a more constrained question:

\[
\boxed{
\text{When the available causal domain grows at a physically accounted cost,}
\atop
\text{is there a reproducible interval of growth rates that maximizes}
\atop
\text{persistent, resource-supported causal organization?}
}
\]

The proposed answer is an empirical hypothesis, not a conclusion:

\[
\boxed{
g_{\min}<g<g_{\max}.
}
\]

The scientific work lies in deriving the bounds, defining complexity independently of the desired answer, and attempting to make the window disappear.

---

# 2. Scope

Paper 4 concerns artificial systems in which:

- the number of available sites or graph nodes can change;
- connectivity can change locally;
- material and resource flows are explicitly accounted;
- persistent structures can form and be detected;
- domain growth may be scheduled or endogenous.

It does not claim to model:

- physical spacetime;
- general relativity;
- the Big Bang;
- dark energy;
- cosmological inflation;
- the observed Hubble expansion;
- consciousness;
- divine purpose.

The word **expansion** in this paper means an increase in the active causal domain of a specified artificial system.

The paper’s central claim is conditional:

> If structures require both available capacity and sufficient causal coordination, then excessive confinement and excessive expansion may suppress complexity through different mechanisms.

The simulations will determine whether such a window actually appears.

---

# 3. Prior Art and the Novelty Boundary

## 3.1 Growing reaction–diffusion domains

A growing domain changes the equations governing diffusion and reaction. In continuum models, transforming from physical to comoving coordinates generally introduces growth-related advection or dilution terms. Concentrations can fall as volume grows, wavelengths can shift, and pattern modes can appear or disappear.

Research on growing-domain reaction–diffusion systems has investigated these effects in developmental and biological contexts. Growth rate can alter whether patterns form and whether they remain robust as the system enlarges. citeturn856725view0turn856725view1

IF Theory therefore cannot claim novelty for the proposition:

> Domain growth affects pattern formation.

Its stronger question is whether the competition between growth and causal coordination produces a transferable sustainable-complexity window across discrete, graph-based, and stochastic substrates.

---

## 3.2 Developmental graph cellular automata

Graph cellular automata replace a fixed lattice with nodes connected by an evolving graph. Developmental Graph Cellular Automata allow nodes to make local growth decisions and can expand a graph from a small seed. Recent implementations have optimized growing graphs for computational-reservoir properties and task performance. citeturn856725view2

IF Theory therefore cannot claim novelty for:

- locally controlled graph growth;
- growth from a seed;
- functional structures produced by developmental rules.

The IF standard adds:

- explicit growth cost;
- conservative material partition;
- no unaccounted capacity in new nodes;
- target-free confirmatory measurements;
- and a predicted two-sided growth window.

---

## 3.3 Mass-conserving artificial life

Mass-conserving continuous cellular automata such as Flow-Lenia already support localized structures and emergent evolutionary dynamics. This demonstrates that conservative artificial-life substrates can generate complex behavior without ordinary birth-and-death cell rules. citeturn510876search14

IF Theory cannot claim that conservation plus artificial life is new.

The expansion study must show that a growth-rate relationship adds explanatory and predictive content beyond what fixed-domain conservative systems already demonstrate.

---

## 3.4 Criticality and the edge of chaos

Langton’s edge-of-chaos proposal suggested that complex computation might occur near a transition between ordered and chaotic dynamics. Subsequent work challenged the universality of the claim and showed that conclusions could depend on the rule parameter, task, evolutionary process, and interpretation of the original experiments. citeturn856725view4turn856725view5

Paper 4 therefore does not assume:

\[
\text{intermediate growth}
=
\text{edge of chaos}
=
\text{maximum computation}.
\]

Growth rate, dynamical instability, connectivity, resource density, and information flow are separate control variables.

The analysis will determine whether the apparent expansion window coincides with:

- an absorbing-state transition;
- percolation;
- a dynamical critical point;
- a smooth crossover;
- or no critical phenomenon.

---

## 3.5 Functional percolation

Recent simulations of cascade dynamics on random networks found that the appearance of a giant connected component can coincide with sharp increases in realizable functional complexity, response diversity, output entropy, and directed information flow. Functional diversity peaked near the connectivity transition in that model, while transfer entropy continued increasing beyond it. citeturn856725view6turn856725view7

This result is relevant because excessive expansion can lower effective connectivity and fragment causal propagation.

However, functional percolation studies connectivity at a given network size or ensemble. IF Theory studies a resource-constrained network whose size, density, and topology evolve together.

---

## 3.6 Provisional novelty claim

The possible contribution of Paper 4 is:

\[
\boxed{
\begin{gathered}
\text{A resource-conserving, cost-aware theory predicts both a lower}
\\
\text{and an upper bound on domain-growth rate from independently}
\\
\text{measured crowding and coordination timescales, then tests whether}
\\
\text{multiple persistent-complexity measures peak inside those bounds}
\\
\text{across independently designed artificial substrates.}
\end{gathered}
}
\]

The broad expectation of an intermediate optimum is not sufficient novelty.

A scientific contribution requires:

1. an operational derivation of the bounds;
2. prospective prediction of the window;
3. confirmation on held-out rules and substrates;
4. separation of expansion from energy injection and final domain size;
5. a dimensionless scaling collapse or another transferable relationship.

---

# 4. Central Hypothesis

## 4.1 Expansion–Complexity Window Hypothesis

There exists a finite interval of domain-growth rates:

\[
\boxed{
g_{\min}<g<g_{\max}
}
\]

within which resource-conserving IF universes support more persistent causal organization than comparable slower-growing or faster-growing universes.

The lower and upper failures arise through different mechanisms.

### Below the window

- available sites become saturated;
- structures compete for the same local resources;
- waste accumulates;
- boundaries collide;
- new structures cannot nucleate;
- diversity declines;
- large structures may monopolize the domain;
- perturbations propagate globally because there is no spatial slack.

### Above the window

- matter and resources become diluted;
- encounter rates decline;
- local neighborhoods turn over faster than adaptation;
- communication paths lengthen or disconnect;
- structures lose causal closure;
- offspring or fragments fail to establish;
- growth consumes too much available capacity;
- the substrate changes before patterns can stabilize.

### Inside the window

- new capacity becomes available before severe saturation;
- interaction density remains sufficient for coordination;
- structures can grow, separate, and diversify;
- resource gradients remain exploitable;
- causal paths persist long enough for maintenance and repair;
- growth does not consume the entire available battery.

---

# 5. Domain, Matter, and Capacity

Expansion must not silently create physical content.

Let the IF universe at time \(t\) be:

\[
\mathcal U_t=
\left(
G_t,Z_t,\mathcal R_t,\Lambda
\right),
\]

where:

- \(G_t=(V_t,E_t)\) is the active causal domain;
- \(Z_t\) contains local material and configuration states;
- \(\mathcal R_t\) contains resource ledgers;
- \(\Lambda\) specifies external reservoirs and boundary conditions.

Define:

\[
N_t=|V_t|
\]

as the active domain size.

Let:

\[
M_t=\sum_{i\in V_t}m_i
\]

be total conserved material.

Let:

\[
F_t=\sum_{i\in V_t}f_i
\]

be remaining high-grade capacity.

Let:

\[
W_t=\sum_{i\in V_t}w_i
\]

be degraded capacity or waste.

Let:

\[
X_t
\]

be exported work.

New domain does not imply new material:

\[
\Delta N_t>0
\centernot\Rightarrow
\Delta M_t>0.
\]

New domain does not imply new fuel:

\[
\Delta N_t>0
\centernot\Rightarrow
\Delta F_t>0.
\]

Any resource imported with new space must be declared as an environmental input and compared against a matched no-expansion resource-input control.

---

# 6. Growth Rate

For discrete time, define fractional domain growth:

\[
\boxed{
g_t=
\frac{
N_{t+1}-N_t
}{
N_t
}.
}
\]

Over an interval:

\[
\bar g_{[t_0,t_1]}
=
\frac{
\ln N_{t_1}-\ln N_{t_0}
}{
t_1-t_0
}.
\]

For continuous approximations:

\[
g(t)=\frac{\dot N}{N}.
\]

Growth may be:

- constant;
- episodic;
- stochastic;
- boundary-triggered;
- density-responsive;
- locally endogenous.

The complete growth history matters. Two universes can have equal final size and different organizational outcomes because one expanded gradually while the other expanded in bursts.

---

# 7. Growth Costs

Creating a new causal degree of freedom must have a declared cost.

Let:

- \(c_V\) be the cost of activating one new node;
- \(c_E\) be the cost of creating one edge;
- \(c_R\) be the cost of rewiring or maintaining connectivity;
- \(c_D\) be the cost of transporting or redistributing matter during growth.

For a growth event:

\[
\boxed{
C_{\mathrm{grow}}
=
c_V\Delta N
+
c_E\Delta |E|
+
c_RN_{\mathrm{rewired}}
+
C_{\mathrm{transport}}.
}
\]

The cost is paid from:

- local structure resources;
- a global expansion reservoir;
- an external environment.

The selected source must be explicit.

A run fails accounting if:

\[
C_{\mathrm{grow}}
\]

is not matched by a corresponding ledger decrease or external input.

---

# 8. Dormant-Substrate Interpretation

The simplest computational interpretation avoids literal creation of new physical substance.

Begin with a finite maximal graph:

\[
G_{\max}.
\]

At time \(t\), only a subset:

\[
V_t\subseteq V_{\max}
\]

is active.

Inactive nodes are dormant degrees of freedom that:

- contain no available material;
- contain no fuel;
- do not update;
- do not communicate.

Expansion activates dormant nodes at a cost.

This interpretation permits exact finite accounting and avoids claiming that the program creates spacetime from nothing.

It is a computational reference model, not a fundamental cosmology.

---

# 9. Conservative Lattice Expansion

## 9.1 Site activation

A lattice universe begins with an active connected region inside a larger dormant lattice.

Growth activates a layer or selected boundary sites.

A newly active site begins with:

\[
m_i=0,\qquad f_i=0,\qquad w_i=0
\]

unless imported resources are separately logged.

---

## 9.2 Conservative remapping

If expansion stretches an existing region rather than activating empty boundary space, material must be remapped conservatively.

Let a parent cell \(i\) split into \(i_1,\ldots,i_k\).

Require:

\[
\sum_{\ell=1}^{k}m_{i_\ell}=m_i,
\]

\[
\sum_{\ell=1}^{k}f_{i_\ell}\leq f_i-C_{\mathrm{grow},i},
\]

\[
\sum_{\ell=1}^{k}w_{i_\ell}
=
w_i+
C_{\mathrm{diss},i}.
\]

No state variable is copied as physical material unless duplication cost is paid.

Configuration state may be inherited by both daughters only when its physical memory-copying cost is included.

---

## 9.3 Dilution

For approximately uniform physical stretching in \(d\) dimensions, a conserved density \(\rho\) obeys a dilution contribution:

\[
\left.
\frac{d\rho}{dt}
\right|_{\mathrm{growth}}
=
-dg_L\rho,
\]

where \(g_L\) is the linear expansion rate and:

\[
g=dg_L
\]

is the volume or node-number growth rate.

The discrete implementation must reproduce the corresponding conservation behavior.

---

# 10. Conservative Graph Expansion

## 10.1 Node division

In the graph implementation, node \(i\) may divide into:

\[
i\rightarrow(i',j').
\]

Its material is partitioned:

\[
m_{i'}+m_{j'}=m_i.
\]

Its available capacity is partitioned after paying division and edge costs:

\[
f_{i'}+f_{j'}
=
f_i-C_{\mathrm{division}}-C_{\mathrm{edges}}.
\]

Waste increases by the dissipated portion.

---

## 10.2 Edge inheritance

The parent’s edges may be:

- assigned to one daughter;
- divided between daughters;
- retained by both at an additional edge cost;
- removed.

The local graph rule determines this using only neighborhood information.

---

## 10.3 Empty-domain growth

A global process may activate new empty nodes adjacent to the existing graph.

These nodes contain no material but may later receive material through transport.

This separates:

\[
\text{growth of available domain}
\]

from:

\[
\text{growth of organized matter}.
\]

---

## 10.4 Connectivity maintenance

As \(N\) increases, keeping average degree constant requires edge creation.

If edges are not added sufficiently quickly:

\[
\langle k\rangle
\]

falls and the graph may fragment.

If edges proliferate too rapidly, maintenance cost rises and collective dynamics may become overly constrained.

Expansion must therefore specify both:

\[
g_N=\frac{\dot N}{N}
\]

and:

\[
g_E=\frac{\dot{|E|}}{|E|}.
\]

---

# 11. Reference Implementations

## 11.1 IF-X0: scheduled lattice growth

IF-X0 uses:

- a dormant maximal lattice;
- externally scheduled activation;
- empty new sites;
- explicit activation cost;
- resource-conserving local dynamics from Paper 3.

Purpose:

- isolate the causal effect of growth rate;
- compare equal final sizes;
- construct phase diagrams;
- establish analytical baselines.

The schedule is imposed, so X0 does not demonstrate endogenous expansion.

---

## 11.2 IF-X1: density-responsive lattice growth

Growth probability depends on local boundary pressure:

\[
p_{\mathrm{activate},i}
=
\sigma
\left[
\alpha
\left(
\phi_i-\phi_c
\right)
-
\beta C_{\mathrm{grow},i}
\right],
\]

where:

- \(\phi_i\) is local occupation or flux pressure;
- \(\phi_c\) is a threshold;
- \(\sigma\) is a bounded response function.

The global growth history emerges from local boundary states.

The growth rule still has no access to complexity, structure identity, or survival score.

---

## 11.3 IF-X2: developmental graph growth

IF-X2 begins with a small active graph.

Local nodes may:

- remain;
- divide;
- create an edge;
- remove an edge;
- deactivate.

All growth actions partition local material and pay local costs.

This implementation is conceptually related to Developmental Graph Cellular Automata, which allow local nodes to decide whether to divide or remain based on their own and neighboring states. The IF version differs by making material and expansion costs central constraints rather than optimizing a prescribed graph function. citeturn856725view2

---

## 11.4 IF-X3: stochastic-thermodynamic growth

IF-X3 associates each activation, division, transport, and rewiring event with:

- a state-energy change;
- work input;
- heat exchange;
- reservoir coupling;
- transition rates satisfying a declared local detailed-balance relation.

X3 is required before claims are made about physical entropy production during expansion.

---

# 12. Occupation and Crowding

Let:

\[
Q_t
\]

be total occupied or materially active capacity.

Define occupation fraction:

\[
\boxed{
\phi_t=
\frac{Q_t}{K N_t},
}
\]

where \(K\) is the local carrying capacity per site.

Suppose active material or organization increases at effective rate:

\[
r_{\mathrm{occ}}
=
\frac{\dot Q}{Q}.
\]

Suppose turnover releases capacity at rate:

\[
r_{\mathrm{release}}.
\]

Then approximately:

\[
\frac{d\ln\phi}{dt}
=
r_{\mathrm{occ}}
-
r_{\mathrm{release}}
-
g.
\]

To prevent persistent growth in occupation fraction:

\[
g
\gtrsim
r_{\mathrm{occ}}-r_{\mathrm{release}}.
\]

This motivates the lower expansion bound:

\[
\boxed{
g_{\min}
\approx
r_{\mathrm{occ}}-r_{\mathrm{release}}.
}
\]

This is not assumed exact. It is a preregistered first-order prediction.

---

# 13. Coordination Timescale

A persistent structure requires causal influence to propagate across itself.

Define:

\[
\tau_{\mathrm{coord}}
\]

as a characteristic coordination time.

Possible estimators include:

- median time for a perturbation at one boundary to influence the opposite boundary;
- inverse spectral gap of the structure’s interaction graph;
- mixing time of relevant local signals;
- time to recover from a standardized perturbation;
- lag maximizing cross-structure transfer entropy;
- minimal intervention-response latency.

During one coordination interval, the domain grows fractionally by:

\[
\Gamma
=
g\tau_{\mathrm{coord}}.
\]

Define the **growth–coordination number**:

\[
\boxed{
\Gamma=g\tau_{\mathrm{coord}}.
}
\]

Interpretation:

\[
\Gamma\ll1:
\quad
\text{the domain changes slowly relative to coordination},
\]

\[
\Gamma\sim1:
\quad
\text{domain change and coordination occur on comparable timescales},
\]

\[
\Gamma\gg1:
\quad
\text{the substrate changes substantially before coordination completes}.
\]

The upper bound is:

\[
\boxed{
g_{\max}
\approx
\frac{\gamma_c}{\tau_{\mathrm{coord}}},
}
\]

where \(\gamma_c\) is determined from held-out systems rather than selected separately for every rule.

---

# 14. The IF Expansion–Complexity Inequality

Combining the crowding and coordination constraints gives:

\[
\boxed{
r_{\mathrm{occ}}-r_{\mathrm{release}}
<
g
<
\frac{\gamma_c}{\tau_{\mathrm{coord}}}.
}
\]

A window exists only when:

\[
\boxed{
r_{\mathrm{occ}}-r_{\mathrm{release}}
<
\frac{\gamma_c}{\tau_{\mathrm{coord}}}.
}
\]

This yields an immediate negative prediction:

> Some rule families possess no sustainable expansion window because their occupation pressure exceeds their capacity to coordinate.

This is scientifically preferable to assuming that every universe has a life-supporting expansion rate.

---

# 15. Additional Dimensionless Controls

## 15.1 Expansion-cost number

Let:

\[
P_{\mathrm{grow}}
\]

be average power spent on growth.

Let:

\[
P_{\mathrm{avail}}
\]

be total high-grade resource power available.

Define:

\[
\boxed{
\Gamma_C=
\frac{P_{\mathrm{grow}}}{P_{\mathrm{avail}}}.
}
\]

If:

\[
\Gamma_C\rightarrow1,
\]

growth consumes nearly all available capacity, leaving little for maintenance or structure formation.

---

## 15.2 Dilution number

Let:

\[
\tau_{\mathrm{capture}}
\]

be the time required for structures to intercept or concentrate dispersed resources.

Define:

\[
\boxed{
\Gamma_D=g\tau_{\mathrm{capture}}.
}
\]

When:

\[
\Gamma_D\gg1,
\]

resources dilute faster than structures can capture them.

---

## 15.3 Connectivity number

Let:

\[
k_t
\]

be mean active degree and:

\[
k_c
\]

the relevant percolation or functional-connectivity threshold.

Define:

\[
\boxed{
\Gamma_K=\frac{k_t}{k_c}.
}
\]

When:

\[
\Gamma_K<1,
\]

large-scale causal propagation may be fragmented.

The threshold must be estimated for the actual graph and dynamics, not borrowed blindly from an Erdős–Rényi network.

---

# 16. What Counts as Complexity?

Paper 4 rejects one visual complexity score.

Complexity is represented by a preregistered vector:

\[
\boxed{
\mathbf C(t)=
\left[
C_{\mathrm{phen}},
C_{\mathrm{causal}},
C_{\mathrm{temporal}},
C_{\mathrm{multi}},
C_{\mathrm{repair}},
C_{\mathrm{lineage}},
C_{\mathrm{throughput}}
\right].
}
\]

---

## 16.1 Persistent phenotype diversity

Candidate structures are represented through feature vectors including:

- size;
- boundary organization;
- resource throughput;
- motility;
- dynamical spectrum;
- perturbation response;
- causal signature.

Cluster the features using a method frozen before final evaluation.

Let phenotype frequencies be:

\[
p_1,\ldots,p_K.
\]

Define effective phenotype diversity:

\[
\boxed{
C_{\mathrm{phen}}
=
\exp
\left[
-\sum_{k=1}^{K}p_k\ln p_k
\right].
}
\]

Only structures exceeding the null-adjusted persistence threshold are included.

---

## 16.2 Causal complexity

For detected structure \(A\), compare intervention-response distributions across macrostates.

Possible measures include:

- effective information;
- intervention selectivity;
- decision-tree depth of response functions;
- causal-state complexity;
- path-specific influence.

The primary causal estimator must be fixed before confirmatory analysis.

---

## 16.3 Temporal predictive complexity

Let \(S_t^A\) denote the candidate macrostate.

Measure:

\[
C_{\mathrm{temporal}}(\tau)
=
I(S_t^A;S_{t+\tau}^A).
\]

A frozen structure may have high predictability but low behavioral repertoire.

Therefore, also report:

\[
H(S_{t+\tau}^A)
\]

and:

\[
I(S_t^A;S_{t+\tau}^A)
\]

separately.

---

## 16.4 Multiscale dynamical complexity

Measure how much structure is present across spatial and temporal scales.

Possible estimators include:

- multiscale entropy;
- compression curves;
- persistent homology;
- wavelet entropy;
- excess entropy;
- statistical complexity.

No estimator may be interpreted as thermodynamic entropy unless its physical connection is separately established.

---

## 16.5 Repair complexity

Measure:

- number of distinct damage classes recovered from;
- maximum recoverable damage;
- recovery time;
- resource cost of recovery;
- restoration of dynamics rather than appearance alone.

---

## 16.6 Lineage complexity

For replication-capable structures, measure:

- lineage depth;
- branching;
- heritable phenotype diversity;
- innovation rate;
- extinction rate.

Repeated external nucleation does not count as lineage.

---

## 16.7 Resource-supported throughput

Persistent organization must be related to resource use.

Report:

\[
C_{\mathrm{throughput}}
=
\frac{
\text{persistent causal organization}
}{
\text{resource consumed}
}
\]

only after defining the numerator explicitly.

This efficiency does not replace absolute complexity. A trivial static structure may be resource-efficient but uninteresting.

---

# 17. Primary Outcome Standard

The expansion-window hypothesis will not be accepted because one metric has a convenient interior maximum.

Before confirmatory runs, designate:

- one primary metric;
- six secondary metrics;
- one joint criterion.

The proposed primary metric is:

\[
\boxed{
C_{\mathrm{primary}}
=
C_{\mathrm{phen}}
\times
\operatorname{median}
\left[
C_{\mathrm{causal}}
\right],
}
\]

after both terms are nondimensionalized against fixed null distributions.

Because this product is partly conventional, all components must also be reported.

The joint criterion is satisfied only if:

1. the primary metric has a statistically supported interior optimum;
2. at least four of six secondary measures improve inside the predicted interval relative to both sides;
3. the result survives held-out rules and seeds;
4. the interval overlaps the bounds predicted from occupation and coordination measurements;
5. the peak is not explained solely by final domain size, total resource input, or detector behavior.

---

# 18. Phase Taxonomy

## X0 — Confined extinction

Insufficient space or resources cause activity to disappear.

## X1 — Congested freeze

Structures fill the domain and become static or mutually blocked.

## X2 — Congested turbulence

High density produces unstable collisions, waste accumulation, and short-lived organization.

## X3 — Sustainable expansion

Persistent structures form, separate, maintain throughput, and diversify.

## X4 — Dilution-dominated expansion

Material and resources become too sparse for stable organization.

## X5 — Causal fragmentation

The domain remains active, but structures lose large-scale coordination because topology changes too quickly or connectivity falls below threshold.

## X6 — Growth-starved organization

Expansion cost consumes the resource budget needed for maintenance.

## X7 — Runaway expansion

Growth continues despite declining occupation and organization.

## X8 — Self-limiting expansion

Local growth slows or stops when pressure falls, producing a dynamically maintained domain size.

## X9 — Pulsed expansion

Growth and structural activity alternate through reproducible cycles.

---

# 19. Distinguishing Competing Mechanisms

An observed high-growth collapse could have several causes.

## 19.1 Pure dilution

Test by expanding while preserving connectivity and compensating interaction range.

If complexity remains low, resource density may be the limiting mechanism.

---

## 19.2 Connectivity loss

Hold resource density fixed while changing edge density.

If complexity is restored, fragmentation rather than dilution was decisive.

---

## 19.3 Growth cost

Provide a matched external resource subsidy equal to growth cost.

If complexity returns, the collapse resulted from battery depletion.

---

## 19.4 Coordination failure

Hold average degree and resource density fixed, but increase topology-turnover rate.

If complexity falls, structures cannot adapt quickly enough to changing neighborhoods.

---

## 19.5 Final-size effect

Compare runs with identical final \(N\) but different growth histories.

If outcomes depend only on final size, there is no rate-specific window.

---

## 19.6 Available-time effect

Fast-growing universes may spend less time at each intermediate size.

Compare at equal total update count, equal physical time, and equal time since reaching final size.

---

# 20. Core Hypotheses

## EC-H1 — Interior-optimum hypothesis

At least one preregistered persistent-causal-complexity measure has a robust interior optimum as a function of growth rate.

### Falsifier

Complexity is:

- monotonic;
- flat;
- boundary-maximized;
- or dependent on a post hoc metric.

---

## EC-H2 — Two-mechanism hypothesis

Low-growth and high-growth failures arise through measurably different mechanisms:

\[
\text{crowding below},
\qquad
\text{dilution or fragmentation above}.
\]

### Falsifier

Both sides are explained by the same trivial resource shortage or simulation artifact.

---

## EC-H3 — Predictive-bound hypothesis

The observed window overlaps bounds predicted before the final growth sweep:

\[
r_{\mathrm{occ}}-r_{\mathrm{release}}
<
g
<
\frac{\gamma_c}{\tau_{\mathrm{coord}}}.
\]

### Falsifier

The fitted optimum bears no transferable relationship to independently measured occupation and coordination timescales.

---

## EC-H4 — Equal-final-size hypothesis

Growth history affects complexity even when initial state, final domain size, total material, total imported resource, and run duration are matched.

### Falsifier

Only final size matters.

---

## EC-H5 — Cost-aware hypothesis

The optimal growth rate shifts predictably when node and edge creation costs change.

### Prediction

Increasing \(c_V\) or \(c_E\) lowers the sustainable upper growth rate and may eliminate the window.

### Falsifier

Growth cost has no systematic effect or enters only through a coding artifact.

---

## EC-H6 — Connectivity hypothesis

Complexity collapses when effective connectivity crosses a functional threshold, even when resource density is held constant.

### Falsifier

No relation exists between causal fragmentation and measured connectivity.

---

## EC-H7 — Cross-substrate scaling hypothesis

The growth–coordination number:

\[
\Gamma=g\tau_{\mathrm{coord}}
\]

organizes the high-growth transition across lattice, graph, and stochastic implementations better than raw \(g\).

### Falsifier

Each substrate requires unrelated scaling variables and thresholds.

---

## EC-H8 — Endogenous-regulation hypothesis

Local growth rules can evolve or self-organize toward the sustainable window without receiving a direct complexity reward.

### Falsifier

Self-regulation occurs only when the fitness function explicitly rewards the target growth rate or complexity measure.

---

## EC-H9 — No-universal-window hypothesis

Some rule families will have no nonempty window because:

\[
r_{\mathrm{occ}}-r_{\mathrm{release}}
\geq
\frac{\gamma_c}{\tau_{\mathrm{coord}}}.
\]

This is a positive prediction of failure.

### Falsifier

Every rule can be assigned a favorable window after arbitrary adjustment, indicating that the inequality lacks restrictive content.

---

# 21. Experimental Program

## Experiment 1 — Fixed-domain baseline

Run Paper 3 rule families on several fixed domain sizes.

Establish how complexity depends on:

- size;
- density;
- resource input;
- boundary condition.

No expansion claim is made here.

---

## Experiment 2 — Constant growth-rate sweep

For:

\[
g\in\{0,g_1,\ldots,g_{\max}\},
\]

run matched universes with:

- equal initial material;
- equal resource schedule;
- equal maximal domain;
- equal total duration;
- frozen structure detector.

Generate full phase diagrams.

---

## Experiment 3 — Equal final size, different histories

Compare:

- early rapid growth;
- late rapid growth;
- constant growth;
- pulsed growth;
- sigmoid growth.

All runs end with the same:

\[
N_{\mathrm{final}}.
\]

This isolates rate and timing effects.

---

## Experiment 4 — Growth-cost sweep

Sweep:

\[
c_V,\quad c_E,\quad c_R.
\]

Measure whether the complexity optimum shifts according to the fraction of capacity consumed by growth:

\[
\Gamma_C.
\]

---

## Experiment 5 — Density-compensated expansion

During fast expansion, maintain average material density using declared external material input.

Compare against:

- no compensation;
- resource-only compensation;
- material-only compensation;
- full compensation.

This separates dilution from topology change.

---

## Experiment 6 — Connectivity-compensated expansion

Adjust edge creation so mean degree remains constant.

Then independently vary:

- edge turnover;
- path length;
- clustering;
- modularity.

This identifies causal-fragmentation mechanisms.

---

## Experiment 7 — Coordination-time prediction

For each candidate rule:

1. measure \(\tau_{\mathrm{coord}}\) at low growth;
2. predict \(g_{\max}\);
3. freeze the prediction;
4. conduct the growth sweep.

---

## Experiment 8 — Crowding-bound prediction

Measure:

\[
r_{\mathrm{occ}}
\]

and:

\[
r_{\mathrm{release}}
\]

in a large but fixed domain.

Predict:

\[
g_{\min}.
\]

Then test whether occupation and diversity degrade below the bound.

---

## Experiment 9 — Local endogenous growth

Allow boundary sites or graph nodes to activate new domain using only local states.

Do not reward complexity or survival directly.

Measure whether growth settles within the independently predicted sustainable interval.

---

## Experiment 10 — Pulsed expansion

Introduce repeated growth bursts.

Test whether structures:

- recover;
- synchronize;
- fragment;
- become more robust;
- experience hysteresis.

---

## Experiment 11 — Damage during expansion

Apply standardized damage at matched expansion phases.

Test whether repair capability varies with:

- crowding;
- domain turnover;
- resource dilution;
- growth cost.

---

## Experiment 12 — Replication during expansion

Measure whether expansion changes:

- replication probability;
- descendant establishment;
- lineage diversity;
- extinction.

An interior optimum may occur because slow growth prevents offspring separation while fast growth prevents resource capture.

---

## Experiment 13 — Cross-substrate validation

Repeat the primary experiments in:

1. IF-X0 lattice growth;
2. IF-X2 graph growth;
3. IF-X3 stochastic growth;
4. one independently designed conservative substrate.

---

# 22. Phase-Transition Analysis

An interior optimum does not automatically imply a phase transition.

The analysis will test separately for:

- absorbing-state transitions;
- percolation transitions;
- fragmentation transitions;
- hysteretic first-order transitions;
- smooth ecological crossovers.

Possible order parameters include:

\[
P_{\mathrm{persist}}
=
P(\text{persistent structure exists}),
\]

\[
S_{\max}/N
=
\text{fraction in largest causal component},
\]

\[
C_{\mathrm{phen}},
\]

\[
\chi_C
=
N
\left(
\langle C^2\rangle-\langle C\rangle^2
\right),
\]

and correlation length.

Evidence for criticality requires:

- finite-size scaling;
- susceptibility behavior;
- critical slowing;
- scaling collapse;
- robust exponents where applicable.

The paper will explicitly permit the conclusion:

> A broad optimum exists, but no critical phase transition was detected.

---

# 23. Deterministic Jupyter-Notebook Program

## Notebook 04A — Dormant Domain and Activation Ledger

Implement:

- maximal dormant lattice;
- active-domain mask;
- activation cost;
- empty-node initialization;
- exact material and capacity assertions.

---

## Notebook 04B — Conservative Cell Splitting

Implement domain stretching through conservative site division.

Verify:

\[
\Delta M=0
\]

and complete resource accounting across random split operations.

---

## Notebook 04C — Graph Node Division

Implement:

- material partition;
- edge inheritance;
- edge costs;
- local division decisions;
- graph-isomorphism-invariant tests.

---

## Notebook 04D — Fixed-Domain Size Baselines

Map complexity against domain size without growth.

This establishes whether later effects are merely size effects.

---

## Notebook 04E — Constant Growth Sweep

Generate the first complete:

\[
g\times\text{rule parameter}
\]

phase map.

---

## Notebook 04F — Equal-Final-Size Growth Histories

Compare constant, early, late, pulsed, and sigmoid schedules.

---

## Notebook 04G — Occupation Lower-Bound Estimator

Estimate:

\[
r_{\mathrm{occ}},
\qquad
r_{\mathrm{release}},
\qquad
g_{\min}.
\]

Validate on analytically solvable population models.

---

## Notebook 04H — Coordination Upper-Bound Estimator

Estimate:

\[
\tau_{\mathrm{coord}}
\]

using:

- intervention propagation;
- spectral gap;
- transfer-entropy lag;
- perturbation recovery.

Freeze:

\[
g_{\max}
\]

before the final sweep.

---

## Notebook 04I — Dilution Controls

Run density-matched, resource-matched, and uncompensated universes.

---

## Notebook 04J — Connectivity and Fragmentation Controls

Track:

- mean degree;
- giant component;
- path length;
- modularity;
- causal reach;
- transfer entropy.

---

## Notebook 04K — Growth-Cost Decomposition

Separate:

- node cost;
- edge cost;
- transport cost;
- maintenance cost.

---

## Notebook 04L — Complexity Metric Validation

Validate each complexity metric against synthetic systems with known:

- randomness;
- periodicity;
- frozen order;
- modular hierarchy;
- causal depth.

---

## Notebook 04M — Joint Window Test

Apply the preregistered primary and secondary outcome criteria.

No metric changes are allowed after opening held-out results.

---

## Notebook 04N — Finite-Size Scaling

Test whether observed boundaries stabilize as:

\[
N_{\max}
\]

increases.

---

## Notebook 04O — Endogenous Growth

Test local density-responsive and resource-responsive expansion without a complexity objective.

---

## Notebook 04P — Cross-Substrate Collapse

Plot results against:

\[
\Gamma=g\tau_{\mathrm{coord}},
\qquad
\Gamma_C,
\qquad
\Gamma_D,
\qquad
\Gamma_K.
\]

Test whether raw substrate differences collapse onto shared curves.

---

## Notebook 04Q — Adversarial Audit

A separate agent attempts to explain the window through:

- final size;
- total resources;
- threshold leakage;
- detector artifacts;
- boundary effects;
- time normalization;
- search-objective contamination;
- numerical instability.

---

# 24. Reproducibility Record

Each run emits:

```yaml
experiment_id: if-expansion-complexity-04
paper_version: null
git_commit: null
environment_hash: null
implementation: IF-X0

rule_family: null
rule_parameters: {}
seed: 65537

max_domain_size: null
initial_domain_size: null
final_domain_size: null
growth_schedule: null
growth_history_hash: null

node_activation_cost: null
edge_creation_cost: null
rewiring_cost: null
transport_cost: null
total_growth_cost: null

initial_material: null
final_material: null
external_material_input: null
material_residual: null

initial_capacity: null
external_capacity_input: null
final_fuel: null
final_waste: null
exported_work: null
capacity_residual: null

occupation_rate: null
release_rate: null
predicted_g_min: null
coordination_time: null
predicted_g_max: null

mean_degree_history: null
giant_component_history: null
occupation_fraction_history: null

primary_complexity_metric: null
secondary_complexity_metrics: {}
phase_classification: null

invariant_failures: []
result_hash: null
```

Every published figure must be regenerable from a manifest and deterministic replay or declared stochastic seed ensemble.

---

# 25. Statistical Standards

## 25.1 Growth rate is the experimental unit

Time points from one expanding universe are not independent growth-rate samples.

The primary unit is a complete run under one frozen schedule and seed.

---

## 25.2 Held-out rules

The window may first be discovered using one group of rule families.

Its predicted bounds must then be tested on held-out rules not used to select:

- metrics;
- coefficients;
- thresholds;
- growth schedules.

---

## 25.3 Multiple metrics

The joint criterion and correction method must be preregistered.

No paper may search dozens of metrics and report only those with interior peaks.

---

## 25.4 Uncertainty in predicted bounds

Because:

\[
r_{\mathrm{occ}},
r_{\mathrm{release}},
\tau_{\mathrm{coord}}
\]

are estimated, the predicted interval has uncertainty.

Report:

\[
P(g_{\min}<g<g_{\max})
\]

rather than treating estimated bounds as exact.

---

## 25.5 Model comparison

Compare at minimum:

- fixed domain;
- cost-free domain growth;
- costly growth;
- resource-injecting growth;
- equal-final-size fixed domain;
- density-controlled growth;
- connectivity-controlled growth.

---

# 26. Failure Modes

## 26.1 Free-space fallacy

New nodes arrive with uncounted fuel or material.

## 26.2 Duplication through interpolation

A remapping algorithm copies mass or internal state without paying for it.

## 26.3 Final-size confounding

The apparent growth-rate effect is actually a domain-size effect.

## 26.4 Total-resource confounding

Faster expansion changes cumulative resource input.

## 26.5 Time-allocation confounding

Slow-growth universes receive more effective development time at intermediate sizes.

## 26.6 Detector-density bias

The structure detector performs differently at different densities.

## 26.7 Metric-by-construction

The complexity score explicitly rewards intermediate occupation.

## 26.8 Search leakage

Growth rates were selected after examining confirmatory results.

## 26.9 Boundary artifact

Structures benefit specifically from periodic or reflecting boundaries.

## 26.10 Hidden global control

The growth scheduler has access to global complexity or structure labels.

## 26.11 Criticality inflation

A smooth maximum is described as a critical phase transition without scaling evidence.

## 26.12 Cosmological overclaim

A computational domain-growth optimum is presented as an explanation of the observed expansion of the universe.

---

# 27. What Would Count as Success?

## Level 1 — Valid costly-growth substrate

Expansion occurs without violating the declared material and capacity ledgers.

## Level 2 — Reproducible interior optimum

A preregistered persistent-complexity metric peaks at an intermediate growth rate.

## Level 3 — Mechanistic decomposition

Low-growth failure is traced to congestion, while high-growth failure is independently traced to dilution, fragmentation, coordination failure, or growth cost.

## Level 4 — Predictive bounds

Measurements made before the growth sweep predict the approximate sustainable interval.

## Level 5 — Cross-rule generalization

The bounds predict behavior in held-out rule families.

## Level 6 — Cross-substrate scaling

Results collapse under dimensionless growth, coordination, cost, and connectivity variables.

## Level 7 — Endogenous regulation

Local systems without a direct complexity reward stabilize their own growth near the predicted sustainable region.

The last result would be particularly important: expansion would become part of the evolving organization rather than an externally selected schedule.

---

# 28. What Would Count as a Major Result?

A field-significant result would not be:

> “We found pretty structures at \(g=0.03\).”

A major result would be:

\[
\boxed{
\text{A dimensionless inequality derived from independently measurable}
\atop
\text{occupation and coordination dynamics prospectively predicts}
\atop
\text{the existence, location, and disappearance of sustainable}
\atop
\text{complexity windows across distinct conservative substrates.}
}
\]

An even stronger result would show that local growth rules spontaneously regulate:

\[
\Gamma=g\tau_{\mathrm{coord}}
\]

near a common range without directly optimizing complexity.

That could suggest a general principle of developmental or ecological organization.

---

# 29. Relationship to the Informational Battery

Domain growth changes the accessibility of physical capacity.

Too little domain may trap:

\[
B_{\mathrm{gross}}
\]

in congested or inaccessible configurations.

Appropriate growth may increase:

\[
B_{\mathrm{op}}
\]

by opening pathways, separating structures, and reducing destructive interference.

Excessive growth may decrease:

\[
B_{\mathrm{op}}
\]

through dilution and fragmentation, even when gross capacity remains present.

Thus:

\[
\boxed{
\text{Expansion can change accessibility without creating energy.}
}
\]

This is an **accessibility transformation**, not free recharge.

Because growth itself costs capacity:

\[
\Delta B_{\mathrm{op}}
>
C_{\mathrm{grow}}
\]

is required for expansion to provide net operational benefit.

---

# 30. Relationship to Agency

Paper 4 does not assume that structures choose the growth rate.

Scheduled growth is environmental.

Density-responsive local growth may be endogenous but still purely reactive.

Agency requires the Paper 2 intervention standard:

- an internal model of future growth consequences;
- policy alternatives;
- causal use of that model;
- net benefit after model cost.

A self-regulating expanding structure is therefore not automatically reflective or conscious.

---

# 31. Relationship to Biological Development

A successful IF expansion window could motivate comparisons with:

- organism growth;
- tissue morphogenesis;
- colony expansion;
- ecological range growth;
- vascular development;
- network development.

Such comparisons would require biological models and data.

The artificial simulation cannot establish that real organisms optimize the same dimensionless variables.

---

# 32. Relationship to Cosmology

The observed universe expands according to a relativistic spacetime geometry constrained by cosmological observations.

Paper 4 changes the number or connectivity of sites in an artificial causal domain.

These are not equivalent statements.

A positive Paper 4 result would show:

> Certain artificial systems develop greater persistent complexity under an intermediate rate of costly domain growth.

It would not show:

> The universe expands in order to create life.

It would not show:

> Dark energy is informational complexity.

It would not show:

> Cosmic expansion follows the IF growth rule.

A cosmological IF theory would have to derive:

- an effective metric;
- covariant field equations;
- the expansion history;
- structure growth;
- gravitational lensing;
- the cosmic microwave background;
- and distinctive observational predictions.

Paper 4 supplies only a computational intuition and a possible complexity-selection principle.

---

# 33. Criteria for Rejection or Major Revision

The expansion–complexity hypothesis should be rejected or substantially revised if:

1. no interior optimum appears under preregistered metrics;
2. the optimum is explained entirely by final domain size;
3. total resources were not matched;
4. the result disappears after correcting detector-density bias;
5. low- and high-growth failure mechanisms cannot be distinguished;
6. independently measured timescales do not predict the interval;
7. no dimensionless relationship transfers across rule families;
8. the window occurs only at isolated numerical settings;
9. expansion cost eliminates all apparent benefits;
10. endogenous growth does not approach the predicted region;
11. simpler growing-domain models explain the result completely;
12. the project repeatedly changes its definition of complexity to preserve the claim.

---

# 34. Conclusion

Expansion is neither automatically creative nor automatically destructive.

It changes:

- density;
- interaction rates;
- connectivity;
- resource accessibility;
- communication time;
- boundary pressure;
- and the physical cost of maintaining the substrate.

The IF Expansion–Complexity Window Hypothesis proposes that these effects create two competing bounds.

The lower bound is set by saturation:

\[
g_{\min}
\approx
r_{\mathrm{occ}}-r_{\mathrm{release}}.
\]

The upper bound is set by causal coordination:

\[
g_{\max}
\approx
\frac{\gamma_c}{\tau_{\mathrm{coord}}}.
\]

The central candidate inequality is:

\[
\boxed{
r_{\mathrm{occ}}-r_{\mathrm{release}}
<
g
<
\frac{\gamma_c}{\tau_{\mathrm{coord}}}.
}
\]

This is not yet a law.

It is a prediction to be tested.

The theory fails if complexity does not occupy an interior interval, if the interval cannot be predicted, or if it arises only through uncounted resources and chosen metrics.

The strongest possible computational result is:

\[
\boxed{
\text{Sustainable complexity occurs when domain growth relieves}
\atop
\text{occupation pressure without outrunning causal coordination,}
\atop
\text{and this balance is governed by transferable dimensionless ratios.}
}
\]

Such a result would not explain cosmological expansion.

It would establish a disciplined artificial-life principle linking growth, resource accessibility, causal coherence, and persistent organization.

The next paper will examine the proposed transition from reactive structures to predictive agents:

\[
\boxed{
\textit{The Agency Threshold: Critical Conditions for the Evolution}
\atop
\textit{of Predictive Control in IF Universes.}
}
\]

---

# References

1. Escudero, C., Yuste, S. B., Abad, E. and Le Vot, F. “Reaction-Diffusion Kinetics in Growing Domains.” (2018). citeturn856725view1

2. Nishihara, S. and Ohira, T. “The Bifurcation Growth Rate for Robust Pattern Formation in a Reaction-Diffusion System on a Growing Domain.” (2024). citeturn856725view0

3. Barandiaran, M. and Stovold, J. “Growing Reservoirs with Developmental Graph Cellular Automata.” (2025). citeturn856725view2

4. Plantec, E. et al. “Flow-Lenia: Emergent Evolutionary Dynamics in Mass-Conservative Continuous Cellular Automata.” *Artificial Life* 31, 228–248 (2025). citeturn510876search14

5. Langton, C. G. “Computation at the Edge of Chaos: Phase Transitions and Emergent Computation.” *Physica D* 42, 12–37 (1990).

6. Mitchell, M., Hraber, P. and Crutchfield, J. P. “Revisiting the Edge of Chaos: Evolving Cellular Automata to Perform Computations.” (1993). citeturn856725view4

7. Mitchell, M., Crutchfield, J. P. and Hraber, P. T. “Dynamics, Computation, and the ‘Edge of Chaos’: A Re-Examination.” (1993). citeturn856725view5

8. Wilkerson, G. J. “Functional Percolation: A Perspective on Criticality of Form and Function.” (2025–2026). citeturn856725view6turn856725view7

9. Dorogovtsev, S. N., Goltsev, A. V. and Mendes, J. F. F. “Critical Phenomena in Complex Networks.” (2007). citeturn956301search30

10. Waldegrave, R. et al. “Developmental Graph Cellular Automata.” Artificial Life Conference Proceedings (2023). citeturn510876search37
