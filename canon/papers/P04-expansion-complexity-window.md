# The Expansion–Complexity Window
## Costly Domain Growth, Causal Coordination, and Sustainable Organization in Resource-Conserving IF Universes

**Author:** Phuc Vinh Truong
**Series:** IF Theory Canonical Papers
**Paper:** 4
**Layer:** SCIENCE
**Status:** canonical revision of extracted draft (2026-07-18)
**Supersedes:** canon/extracted/paper-04-extracted.md

---

## Abstract

Physical and artificial systems often develop while the space available to them changes. Embryos grow while spatial patterns form; ecological populations expand into new territory; developmental graph systems create new nodes and connections; and the physical universe exhibits cosmological expansion. Domain growth can create room for new organization, but it can also dilute interactions, fragment communication, destabilize existing patterns, and consume resources.

This paper proposes the **IF Expansion–Complexity Window Hypothesis**: in resource-conserving artificial universes, persistent causal complexity is maximized over an intermediate range of domain-growth rates. Insufficient growth produces crowding, resource competition, boundary saturation, and loss of structural diversity. Excessive growth causes dilution, falling encounter rates, topological fragmentation, and domain change faster than structures can coordinate or adapt. Between these extremes may exist a finite window in which growth relieves congestion without destroying causal integration.

The central conjecture is not merely that "moderate expansion is best." It is that the window can be predicted from measurable competing timescales. Let \(g\) be the fractional rate of domain growth, \(r_{\mathrm{occ}}\) the rate at which active organization occupies available capacity, \(r_{\mathrm{release}}\) the rate at which capacity is returned through decay or turnover, and \(\tau_{\mathrm{coord}}\) the characteristic time for a persistent structure to propagate information or reorganize across itself. A provisional necessary condition for sustainable complexity is

\[
\boxed{\;r_{\mathrm{occ}}-r_{\mathrm{release}}\;<\;g\;<\;\frac{\gamma_c}{\tau_{\mathrm{coord}}}\;}
\]

where \(\gamma_c\) is an empirically determined coherence threshold. The lower bound prevents progressive saturation; the upper bound keeps expansion slow enough for causal coordination; the window exists only when the lower bound lies below the upper.

Domain growth is not free. New nodes, sites, edges, or spatial degrees of freedom require an explicit cost paid from a declared resource ledger, and new space initially contains no unaccounted matter, fuel, or information — the three-ledger discipline (CLAUDE.md §1) applied to spacetime itself. In lattice implementations, existing material is conservatively remapped during growth; in graph implementations, a node may divide only by partitioning its material and paying the costs of new connectivity.

The paper explicitly distinguishes this artificial-universe hypothesis from cosmology. An intermediate complexity window in a growing simulation would **not** demonstrate that cosmic expansion exists to create life, that dark energy is informational complexity, or that the observed universe follows IF rules. Presenting the toy result as cosmology is the forbidden state **`COMMANDED_EXPANSION`** (CLAUDE.md): "the board gets bigger at threshold" is a *toy* expansion, labeled as such, until an effective metric and covariant field equations are derived in the cosmology lab. What Paper 4 establishes is a computational relationship between growth, dilution, coordination, and organization that might later motivate — but cannot substitute for — a physical theory.

---

## Keywords

Domain growth; artificial life; cellular automata; causal networks; complexity; criticality; dilution; fragmentation; self-organization; resource conservation; developmental systems; phase transition.

---

# 1. Introduction

Growth changes the conditions under which patterns and organisms form. A chemical pattern on a fixed surface behaves differently when the surface stretches; a population stable in a confined habitat may collapse from congestion, spread through new territory, or fragment when dispersal is too fast; a distributed computational network may gain expressive capacity when new nodes connect yet lose coordination if growth outruns communication and maintenance.

Reaction–diffusion research has long recognized that a time-dependent domain changes diffusion, concentration, and pattern selection, on timescales relevant to biological pattern formation; growth rate can control bifurcation and pattern robustness. Artificial developmental systems permit topology itself to grow: Developmental Graph Cellular Automata begin from small seed graphs and let nodes divide, remain, or be removed using local information, and have been trained to grow directed computational reservoirs. Network science adds that connectivity can be too low for global influence to propagate while high connectivity can suppress functional diversity — recent "functional percolation" work reports structural percolation coinciding with a sharp expansion of realizable information-processing functions, with functional diversity peaking near the connectivity transition. These findings support the *plausibility* of an intermediate organizational regime; they do not establish the IF hypothesis.

The familiar "edge of chaos" idea proposed that computation and complexity are enhanced near transitions between ordered and chaotic dynamics. Later reanalyses challenged the universality of that claim and showed the original evidence was not generally reliable. IF must therefore not assume every complexity measure peaks at criticality, nor treat a visually attractive intermediate regime as proof of a universal law (`NOVELTY_INFLATION` guardrail). The constrained question is:

> When the available causal domain grows at a physically accounted cost, is there a reproducible interval of growth rates \(g_{\min}<g<g_{\max}\) that maximizes persistent, resource-supported causal organization?

The proposed answer is an empirical hypothesis, not a conclusion. The scientific work lies in deriving the bounds, defining complexity independently of the desired answer, and attempting to make the window disappear.

---

# 2. Scope

Paper 4 concerns artificial systems in which the number of available sites or graph nodes can change, connectivity can change locally, material and resource flows are explicitly accounted, persistent structures can form and be detected, and domain growth may be scheduled or endogenous. It does **not** claim to model physical spacetime, general relativity, the Big Bang, dark energy, cosmological inflation, the observed Hubble expansion, consciousness, or divine purpose (layer firewall, CLAUDE.md §6). Here **expansion** means an increase in the active causal domain of a specified artificial system. The central claim is conditional: *if structures require both available capacity and sufficient causal coordination, then excessive confinement and excessive expansion may suppress complexity through different mechanisms.* The simulations determine whether such a window actually appears.

---

# 3. Prior Art and the Novelty Boundary

**3.1 Growing reaction–diffusion domains.** A growing domain changes the equations governing diffusion and reaction; transforming to comoving coordinates introduces growth-related advection or dilution terms, concentrations fall as volume grows, wavelengths shift, and modes appear or disappear. IF cannot claim novelty for "domain growth affects pattern formation." Its stronger question is whether the competition between growth and causal coordination produces a *transferable* sustainable-complexity window across discrete, graph-based, and stochastic substrates.

**3.2 Developmental graph cellular automata.** Graph CA replace a fixed lattice with nodes on an evolving graph; Developmental Graph CA make local growth decisions and expand from a seed, and have been optimized for reservoir properties. IF cannot claim novelty for locally controlled graph growth, growth from a seed, or functional structures from developmental rules. IF adds explicit growth cost, conservative material partition, no unaccounted capacity in new nodes, target-free confirmatory measurements, and a predicted two-sided growth window.

**3.3 Mass-conserving artificial life.** Flow-Lenia supports localized structures and emergent evolution without birth/death rules. Conservation plus artificial life is not new; the expansion study must add explanatory and predictive content *beyond* fixed-domain conservative systems.

**3.4 Criticality and the edge of chaos.** Langton's proposal was challenged; conclusions depended on rule parameter, task, evolutionary process, and interpretation. Paper 4 does not assume intermediate growth = edge of chaos = maximum computation. Growth rate, dynamical instability, connectivity, resource density, and information flow are separate control variables; the analysis determines whether the apparent window coincides with an absorbing-state transition, percolation, a dynamical critical point, a smooth crossover, or no critical phenomenon.

**3.5 Functional percolation.** Cascade-dynamics simulations found a giant connected component can coincide with sharp increases in realizable functional complexity, response diversity, output entropy, and directed information flow, with functional diversity peaking near the connectivity transition. This is relevant because excessive expansion can lower effective connectivity and fragment causal propagation. But functional-percolation studies connectivity at a fixed size/ensemble; IF studies a resource-constrained network whose size, density, and topology evolve together.

**3.6 Provisional novelty claim.** The possible contribution: *a resource-conserving, cost-aware theory predicts both a lower and an upper bound on domain-growth rate from independently measured crowding and coordination timescales, then tests whether multiple persistent-complexity measures peak inside those bounds across independently designed substrates.* The broad expectation of an intermediate optimum is not sufficient. A contribution requires an operational derivation of the bounds, prospective prediction of the window, confirmation on held-out rules/substrates, separation of expansion from energy injection and final size, and a dimensionless scaling collapse.

---

# 4. Central Hypothesis

There exists a finite interval \(g_{\min}<g<g_{\max}\) within which resource-conserving IF universes support more persistent causal organization than comparable slower- or faster-growing universes. The lower and upper failures arise through different mechanisms.

**Below the window:** sites saturate; structures compete for local resources; waste accumulates; boundaries collide; new structures cannot nucleate; diversity declines; large structures monopolize; perturbations propagate globally for lack of spatial slack.

**Above the window:** matter and resources dilute; encounter rates decline; neighborhoods turn over faster than adaptation; communication paths lengthen or disconnect; structures lose causal closure; offspring/fragments fail to establish; growth consumes too much capacity; the substrate changes before patterns stabilize.

**Inside the window:** new capacity becomes available before severe saturation; interaction density remains sufficient for coordination; structures grow, separate, and diversify; resource gradients remain exploitable; causal paths persist long enough for maintenance and repair; growth does not consume the entire battery.

---

# 5. Domain, Matter, and Capacity

Expansion must not silently create physical content. Let \(\mathcal U_t=(G_t,Z_t,\mathcal R_t,\Lambda)\) with active causal domain \(G_t=(V_t,E_t)\), local states \(Z_t\), resource ledgers \(\mathcal R_t\), external reservoirs \(\Lambda\). Define active domain size \(N_t=|V_t|\), total conserved material \(M_t=\sum_{i}m_i\), remaining high-grade capacity \(F_t=\sum_i f_i\), degraded capacity \(W_t=\sum_i w_i\), and exported work \(X_t\). Crucially,

\[
\Delta N_t>0\;\not\Rightarrow\;\Delta M_t>0,\qquad \Delta N_t>0\;\not\Rightarrow\;\Delta F_t>0.
\]

New domain implies neither new material nor new fuel. Any resource imported with new space must be declared as an environmental input and compared against a matched no-expansion resource-input control. (New space arriving with free fuel is the "free-space fallacy," §26.1 — a `PERPETUAL_RECHARGE` at the level of the domain.)

---

# 6. Growth Rate

Discrete fractional domain growth \(g_t=(N_{t+1}-N_t)/N_t\); interval average \(\bar g_{[t_0,t_1]}=(\ln N_{t_1}-\ln N_{t_0})/(t_1-t_0)\); continuous \(g(t)=\dot N/N\). Growth may be constant, episodic, stochastic, boundary-triggered, density-responsive, or locally endogenous. The complete growth history matters: two universes with equal final size can differ organizationally because one grew gradually and the other in bursts.

---

# 7. Growth Costs

Each new causal degree of freedom has a declared cost. With node-activation cost \(c_V\), edge-creation cost \(c_E\), rewiring/maintenance cost \(c_R\), and matter-transport cost \(c_D\), a growth event costs

\[
C_{\mathrm{grow}}=c_V\Delta N+c_E\Delta|E|+c_R N_{\mathrm{rewired}}+C_{\mathrm{transport}}.
\]

Paid from local structure resources, a global expansion reservoir, or an external environment (source made explicit). A run fails accounting if \(C_{\mathrm{grow}}\) is not matched by a corresponding ledger decrease or external input.

---

# 8. Dormant-Substrate Interpretation

The simplest interpretation avoids literal creation of new substance. Begin with a finite maximal graph \(G_{\max}\); at time \(t\) only \(V_t\subseteq V_{\max}\) is active. Inactive nodes are dormant degrees of freedom that contain no available material or fuel, do not update, and do not communicate. Expansion activates dormant nodes at a cost. This permits exact finite accounting and avoids claiming the program creates spacetime from nothing — a computational reference model, **not** a fundamental cosmology.

---

# 9. Conservative Lattice Expansion

**9.1 Site activation.** A lattice universe begins with an active connected region inside a larger dormant lattice; growth activates a layer or selected boundary sites. A newly active site starts \(m_i=0,\,f_i=0,\,w_i=0\) unless imported resources are separately logged.

**9.2 Conservative remapping.** If expansion stretches an existing region, material is remapped conservatively: for parent cell \(i\) split into \(i_1,\ldots,i_k\),

\[
\sum_\ell m_{i_\ell}=m_i,\quad \sum_\ell f_{i_\ell}\le f_i-C_{\mathrm{grow},i},\quad \sum_\ell w_{i_\ell}=w_i+C_{\mathrm{diss},i}.
\]

No state variable is copied as physical material unless duplication cost is paid; configuration state may be inherited by both daughters only when its memory-copying cost is included.

**9.3 Dilution.** For approximately uniform stretching in \(d\) dimensions, a conserved density \(\rho\) obeys \(\left.d\rho/dt\right|_{\mathrm{growth}}=-d\,g_L\,\rho\), with linear rate \(g_L\) and node-number rate \(g=d\,g_L\). The discrete implementation must reproduce the corresponding conservation behavior.

---

# 10. Conservative Graph Expansion

**10.1 Node division.** Node \(i\to(i',j')\) with \(m_{i'}+m_{j'}=m_i\) and \(f_{i'}+f_{j'}=f_i-C_{\mathrm{division}}-C_{\mathrm{edges}}\); waste increases by the dissipated portion.
**10.2 Edge inheritance.** Parent edges may be assigned to one daughter, divided, retained by both at an extra edge cost, or removed — determined by the local graph rule from neighborhood information only.
**10.3 Empty-domain growth.** A global process may activate new empty nodes adjacent to the graph; they carry no material but may later receive it, separating *growth of available domain* from *growth of organized matter*.
**10.4 Connectivity maintenance.** Keeping average degree constant as \(N\) grows requires edge creation; too few edges lets \(\langle k\rangle\) fall and the graph fragment, too many raises maintenance cost and over-constrains dynamics. Expansion must specify both \(g_N=\dot N/N\) and \(g_E=\dot{|E|}/|E|\).

---

# 11. Reference Implementations

- **IF-X0 — scheduled lattice growth:** dormant maximal lattice, externally scheduled activation, empty new sites, explicit activation cost, Paper 3 resource-conserving local dynamics. Isolates the causal effect of growth rate; compares equal final sizes; builds phase diagrams. The imposed schedule means X0 does not demonstrate *endogenous* expansion.
- **IF-X1 — density-responsive lattice growth:** activation probability \(p_{\mathrm{activate},i}=\sigma[\alpha(\phi_i-\phi_c)-\beta C_{\mathrm{grow},i}]\), with local occupation/flux pressure \(\phi_i\), threshold \(\phi_c\), bounded response \(\sigma\). Global growth history emerges from local boundary states; the rule has no access to complexity, structure identity, or survival score.
- **IF-X2 — developmental graph growth:** a small active graph whose local nodes may remain, divide, create/remove an edge, or deactivate; all growth actions partition local material and pay local costs. Related to Developmental Graph CA but with material and expansion cost as central constraints, not an optimized graph function.
- **IF-X3 — stochastic-thermodynamic growth:** each activation, division, transport, and rewiring event has a state-energy change, work input, heat exchange, reservoir coupling, and transition rates satisfying declared local detailed balance. Required before any claim about physical thermodynamic-entropy production during expansion.

---

# 12. Occupation and Crowding

Let \(Q_t\) be total occupied/materially active capacity and occupation fraction \(\phi_t=Q_t/(K N_t)\) with per-site carrying capacity \(K\). With active organization increasing at \(r_{\mathrm{occ}}=\dot Q/Q\) and turnover releasing at \(r_{\mathrm{release}}\),

\[
\frac{d\ln\phi}{dt}=r_{\mathrm{occ}}-r_{\mathrm{release}}-g.
\]

To prevent persistent growth in occupation fraction, \(g\gtrsim r_{\mathrm{occ}}-r_{\mathrm{release}}\), motivating the lower bound

\[
\boxed{\,g_{\min}\approx r_{\mathrm{occ}}-r_{\mathrm{release}}\,}
\]

— not assumed exact, but a preregistered first-order prediction.

---

# 13. Coordination Timescale

A persistent structure requires causal influence to propagate across itself. Define \(\tau_{\mathrm{coord}}\) by any of: median time for a boundary perturbation to influence the opposite boundary; inverse spectral gap of the interaction graph; mixing time of local signals; standardized-perturbation recovery time; lag maximizing cross-structure transfer entropy; minimal intervention-response latency. During one coordination interval the domain grows fractionally by \(\Gamma=g\tau_{\mathrm{coord}}\), the **growth–coordination number**:

\[
\boxed{\,\Gamma=g\tau_{\mathrm{coord}}\,}
\]

\(\Gamma\ll1\): domain changes slowly relative to coordination; \(\Gamma\sim1\): comparable timescales; \(\Gamma\gg1\): the substrate changes substantially before coordination completes. The upper bound is

\[
\boxed{\,g_{\max}\approx\frac{\gamma_c}{\tau_{\mathrm{coord}}}\,}
\]

with \(\gamma_c\) determined from held-out systems, not fitted per rule.

---

# 14. The IF Expansion–Complexity Inequality

Combining crowding and coordination constraints:

\[
\boxed{\;r_{\mathrm{occ}}-r_{\mathrm{release}}\;<\;g\;<\;\frac{\gamma_c}{\tau_{\mathrm{coord}}}\;}
\]

A window exists only when \(r_{\mathrm{occ}}-r_{\mathrm{release}}<\gamma_c/\tau_{\mathrm{coord}}\). This yields an immediate negative prediction: *some rule families possess no sustainable expansion window* because their occupation pressure exceeds their capacity to coordinate — scientifically preferable to assuming every universe has a life-supporting expansion rate.

---

# 15. Additional Dimensionless Controls

**15.1 Expansion-cost number** \(\Gamma_C=P_{\mathrm{grow}}/P_{\mathrm{avail}}\) (growth power over available high-grade power); as \(\Gamma_C\to1\), growth consumes nearly all capacity, leaving little for maintenance or structure formation.
**15.2 Dilution number** \(\Gamma_D=g\tau_{\mathrm{capture}}\), with \(\tau_{\mathrm{capture}}\) the time for structures to intercept dispersed resources; \(\Gamma_D\gg1\) means resources dilute faster than they can be captured.
**15.3 Connectivity number** \(\Gamma_K=k_t/k_c\), mean active degree over percolation/functional threshold; \(\Gamma_K<1\) may fragment large-scale causal propagation. \(k_c\) must be estimated for the actual graph and dynamics, not borrowed from an Erdős–Rényi network.

---

# 16. What Counts as Complexity?

One visual complexity score is rejected. Complexity is a preregistered vector

\[
\mathbf C(t)=[C_{\mathrm{phen}},C_{\mathrm{causal}},C_{\mathrm{temporal}},C_{\mathrm{multi}},C_{\mathrm{repair}},C_{\mathrm{lineage}},C_{\mathrm{throughput}}].
\]

**16.1 Persistent phenotype diversity.** Represent candidate structures by feature vectors (size, boundary organization, throughput, motility, dynamical spectrum, perturbation response, causal signature); cluster with a method frozen before evaluation; with frequencies \(p_1,\ldots,p_K\), effective diversity \(C_{\mathrm{phen}}=\exp[-\sum_k p_k\ln p_k]\), including only structures past the null-adjusted persistence threshold.
**16.2 Causal complexity.** Compare intervention-response distributions across macrostates (effective information, intervention selectivity, decision-tree depth, causal-state complexity, path-specific influence); the primary estimator fixed before confirmatory analysis.
**16.3 Temporal predictive complexity.** \(C_{\mathrm{temporal}}(\tau)=I(S_t^A;S_{t+\tau}^A)\); a frozen structure has high predictability but low repertoire, so also report \(H(S_{t+\tau}^A)\) separately.
**16.4 Multiscale dynamical complexity.** Multiscale entropy, compression curves, persistent homology, wavelet entropy, excess entropy, statistical complexity. No estimator may be interpreted as thermodynamic entropy unless its physical connection is separately established (three-ledger discipline; `ENTROPY_CONFLATION` guardrail).
**16.5 Repair complexity.** Distinct damage classes recovered from, maximum recoverable damage, recovery time, resource cost, restoration of dynamics rather than appearance.
**16.6 Lineage complexity.** Lineage depth, branching, heritable phenotype diversity, innovation rate, extinction rate; repeated external nucleation does not count.
**16.7 Resource-supported throughput.** \(C_{\mathrm{throughput}}=(\text{persistent causal organization})/(\text{resource consumed})\), reported only after the numerator is explicitly defined; efficiency does not replace absolute complexity.

---

# 17. Primary Outcome Standard

The hypothesis will not be accepted because one metric has a convenient interior maximum. Before confirmatory runs, designate one primary metric, six secondary metrics, and one joint criterion. Proposed primary:

\[
C_{\mathrm{primary}}=C_{\mathrm{phen}}\times\operatorname{median}[C_{\mathrm{causal}}],
\]

after both terms are nondimensionalized against fixed null distributions; because the product is partly conventional, all components are also reported. The joint criterion is satisfied only if (1) the primary metric has a statistically supported interior optimum; (2) at least four of six secondary measures improve inside the predicted interval relative to both sides; (3) the result survives held-out rules and seeds; (4) the interval overlaps the bounds predicted from occupation and coordination measurements; (5) the peak is not explained solely by final domain size, total resource input, or detector behavior.

---

# 18. Phase Taxonomy

- **X0 — Confined extinction:** insufficient space/resources; activity disappears.
- **X1 — Congested freeze:** structures fill the domain, become static or mutually blocked.
- **X2 — Congested turbulence:** high density → unstable collisions, waste accumulation, short-lived organization.
- **X3 — Sustainable expansion:** persistent structures form, separate, maintain throughput, diversify.
- **X4 — Dilution-dominated expansion:** material/resources too sparse for stable organization.
- **X5 — Causal fragmentation:** domain active but structures lose coordination (topology changes too fast, or connectivity below threshold).
- **X6 — Growth-starved organization:** expansion cost consumes the maintenance budget.
- **X7 — Runaway expansion:** growth continues despite declining occupation and organization.
- **X8 — Self-limiting expansion:** local growth slows/stops as pressure falls, giving a dynamically maintained domain size.
- **X9 — Pulsed expansion:** growth and structural activity alternate through reproducible cycles.

---

# 19. Distinguishing Competing Mechanisms

A high-growth collapse could have several causes, separated by controls: **pure dilution** (expand while preserving connectivity and compensating interaction range — if complexity stays low, density is limiting); **connectivity loss** (hold density fixed, change edge density — restoration implicates fragmentation); **growth cost** (subsidize equal to growth cost — recovery implicates battery depletion); **coordination failure** (hold degree and density fixed, raise topology-turnover — falling complexity implicates adaptation speed); **final-size effect** (equal final \(N\), different histories — outcomes depending only on final size mean no rate-specific window); **available-time effect** (compare at equal total update count, equal physical time, and equal time since reaching final size).

---

# 20. Core Hypotheses

**EC-H1 — Interior-optimum.** At least one preregistered persistent-causal-complexity measure has a robust interior optimum in \(g\). *Falsifier:* complexity is monotonic, flat, boundary-maximized, or dependent on a post hoc metric.

**EC-H2 — Two-mechanism.** Low- and high-growth failures arise through measurably different mechanisms (crowding below, dilution/fragmentation above). *Falsifier:* both sides are explained by the same trivial resource shortage or simulation artifact.

**EC-H3 — Predictive-bound.** The observed window overlaps bounds predicted *before* the final growth sweep, \(r_{\mathrm{occ}}-r_{\mathrm{release}}<g<\gamma_c/\tau_{\mathrm{coord}}\). *Falsifier:* the fitted optimum bears no transferable relationship to independently measured occupation and coordination timescales.

**EC-H4 — Equal-final-size.** Growth history affects complexity even when initial state, final domain size, total material, total imported resource, and run duration are matched. *Falsifier:* only final size matters.

**EC-H5 — Cost-aware.** The optimal growth rate shifts predictably when node/edge creation costs change. *Prediction:* increasing \(c_V\) or \(c_E\) lowers the sustainable upper rate and may eliminate the window. *Falsifier:* growth cost has no systematic effect or enters only through a coding artifact.

**EC-H6 — Connectivity.** Complexity collapses when effective connectivity crosses a functional threshold, even at constant resource density. *Falsifier:* no relation between causal fragmentation and measured connectivity.

**EC-H7 — Cross-substrate scaling.** \(\Gamma=g\tau_{\mathrm{coord}}\) organizes the high-growth transition across lattice, graph, and stochastic implementations better than raw \(g\). *Falsifier:* each substrate requires unrelated scaling variables and thresholds.

**EC-H8 — Endogenous-regulation.** Local growth rules can evolve or self-organize toward the sustainable window without a direct complexity reward. *Falsifier:* self-regulation occurs only when the fitness function explicitly rewards the target growth rate or complexity (`TELEOLOGY_INJECTION`).

**EC-H9 — No-universal-window.** Some rule families have no nonempty window because \(r_{\mathrm{occ}}-r_{\mathrm{release}}\ge\gamma_c/\tau_{\mathrm{coord}}\) — a positive prediction of failure. *Falsifier:* every rule can be assigned a favorable window after arbitrary adjustment, showing the inequality lacks restrictive content.

---

# 21. Experimental Program

1. **Fixed-domain baseline** — Paper 3 rule families on several fixed sizes; complexity vs size, density, resource input, boundary. No expansion claim.
2. **Constant growth-rate sweep** — \(g\in\{0,g_1,\ldots,g_{\max}\}\) with equal initial material, resource schedule, maximal domain, total duration, and a frozen detector; full phase diagrams.
3. **Equal final size, different histories** — early-rapid, late-rapid, constant, pulsed, sigmoid growth, all ending at the same \(N_{\mathrm{final}}\); isolates rate/timing.
4. **Growth-cost sweep** — sweep \(c_V,c_E,c_R\); does the optimum shift with \(\Gamma_C\)?
5. **Density-compensated expansion** — maintain average density via declared external input; compare none/resource-only/material-only/full compensation; separates dilution from topology change.
6. **Connectivity-compensated expansion** — hold mean degree constant, then vary edge turnover, path length, clustering, modularity; identifies fragmentation mechanisms.
7. **Coordination-time prediction** — measure \(\tau_{\mathrm{coord}}\) at low growth, predict \(g_{\max}\), freeze the prediction, then run the sweep.
8. **Crowding-bound prediction** — measure \(r_{\mathrm{occ}},r_{\mathrm{release}}\) in a large fixed domain, predict \(g_{\min}\), test degradation below the bound.
9. **Local endogenous growth** — boundary sites/nodes activate new domain from local states only, no complexity/survival reward; does growth settle within the predicted interval?
10. **Pulsed expansion** — repeated bursts; do structures recover, synchronize, fragment, become robust, or show hysteresis?
11. **Damage during expansion** — standardized damage at matched phases; repair vs crowding, turnover, dilution, cost.
12. **Replication during expansion** — replication probability, descendant establishment, lineage diversity, extinction; an interior optimum may arise because slow growth prevents offspring separation while fast growth prevents resource capture.
13. **Cross-substrate validation** — IF-X0 lattice, IF-X2 graph, IF-X3 stochastic, plus one independently designed conservative substrate.

---

# 22. Phase-Transition Analysis

An interior optimum does not automatically imply a phase transition. Test separately for absorbing-state, percolation, fragmentation, hysteretic first-order, and smooth ecological crossovers. Possible order parameters: \(P_{\mathrm{persist}}=P(\text{persistent structure exists})\), \(S_{\max}/N\) (fraction in largest causal component), \(C_{\mathrm{phen}}\), susceptibility \(\chi_C=N(\langle C^2\rangle-\langle C\rangle^2)\), and correlation length. Evidence for criticality requires finite-size scaling, susceptibility behavior, critical slowing, scaling collapse, and robust exponents where applicable. The paper explicitly permits the conclusion: *a broad optimum exists, but no critical phase transition was detected* — a smooth maximum sold as a critical transition without scaling evidence is `criticality inflation`, §26.11.

---

# 23. Deterministic Jupyter-Notebook Program

Each notebook carries the contract cell (`Prediction · Baseline · Data · Pass criterion · Falsifier`) and seed 65537.

- **04A — Dormant Domain and Activation Ledger:** maximal dormant lattice, active mask, activation cost, empty-node init, exact material/capacity assertions.
- **04B — Conservative Cell Splitting:** domain stretching via conservative division; verify \(\Delta M=0\) and full resource accounting across random splits.
- **04C — Graph Node Division:** material partition, edge inheritance, edge costs, local division decisions, isomorphism-invariant tests.
- **04D — Fixed-Domain Size Baselines:** complexity vs domain size without growth (are later effects merely size effects?).
- **04E — Constant Growth Sweep:** first complete \(g\times\text{rule-parameter}\) phase map.
- **04F — Equal-Final-Size Growth Histories:** constant, early, late, pulsed, sigmoid schedules.
- **04G — Occupation Lower-Bound Estimator:** estimate \(r_{\mathrm{occ}},r_{\mathrm{release}},g_{\min}\); validate on analytically solvable population models.
- **04H — Coordination Upper-Bound Estimator:** estimate \(\tau_{\mathrm{coord}}\) (intervention propagation, spectral gap, transfer-entropy lag, perturbation recovery); freeze \(g_{\max}\) before the final sweep.
- **04I — Dilution Controls:** density-matched, resource-matched, uncompensated universes.
- **04J — Connectivity and Fragmentation Controls:** mean degree, giant component, path length, modularity, causal reach, transfer entropy.
- **04K — Growth-Cost Decomposition:** node, edge, transport, maintenance costs.
- **04L — Complexity Metric Validation:** validate each metric against synthetic systems with known randomness, periodicity, frozen order, modular hierarchy, causal depth.
- **04M — Joint Window Test:** apply the preregistered primary and secondary criteria; no metric changes after opening held-out results.
- **04N — Finite-Size Scaling:** whether observed boundaries stabilize as \(N_{\max}\) increases.
- **04O — Endogenous Growth:** local density-/resource-responsive expansion without a complexity objective.
- **04P — Cross-Substrate Collapse:** plot vs \(\Gamma=g\tau_{\mathrm{coord}}\), \(\Gamma_C\), \(\Gamma_D\), \(\Gamma_K\); test whether substrate differences collapse onto shared curves.
- **04Q — Adversarial Audit:** a separate agent attempts to explain the window via final size, total resources, threshold leakage, detector artifacts, boundary effects, time normalization, search-objective contamination, or numerical instability.

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

**25.1 Growth rate is the experimental unit** — time points from one expanding universe are not independent growth-rate samples; the primary unit is a complete run under one frozen schedule and seed.
**25.2 Held-out rules** — the window may be discovered on one rule group; predicted bounds are then tested on held-out rules not used to select metrics, coefficients, thresholds, or schedules.
**25.3 Multiple metrics** — the joint criterion and correction method are preregistered; no searching dozens of metrics and reporting only interior peaks.
**25.4 Uncertainty in predicted bounds** — because \(r_{\mathrm{occ}},r_{\mathrm{release}},\tau_{\mathrm{coord}}\) are estimated, report \(P(g_{\min}<g<g_{\max})\) rather than treating bounds as exact.
**25.5 Model comparison** — at minimum: fixed domain, cost-free growth, costly growth, resource-injecting growth, equal-final-size fixed domain, density-controlled growth, connectivity-controlled growth.

---

# 26. Failure Modes

Free-space fallacy (new nodes with uncounted fuel/material); duplication through interpolation (remapping copies mass/state unpaid); final-size confounding; total-resource confounding; time-allocation confounding (slow growth = more development time at intermediate sizes); detector-density bias; metric-by-construction (score rewards intermediate occupation); search leakage (growth rates chosen after seeing results); boundary artifact; hidden global control (scheduler reads global complexity/labels); criticality inflation; **cosmological overclaim** — a computational domain-growth optimum presented as an explanation of the observed expansion of the universe (the `COMMANDED_EXPANSION` forbidden state).

---

# 27. What Would Count as Success?

- **Level 1 — Valid costly-growth substrate:** expansion without violating material/capacity ledgers.
- **Level 2 — Reproducible interior optimum:** a preregistered complexity metric peaks at intermediate \(g\).
- **Level 3 — Mechanistic decomposition:** low-growth failure traced to congestion, high-growth failure independently traced to dilution, fragmentation, coordination failure, or growth cost.
- **Level 4 — Predictive bounds:** measurements before the sweep predict the approximate interval.
- **Level 5 — Cross-rule generalization:** the bounds predict behavior in held-out rule families.
- **Level 6 — Cross-substrate scaling:** results collapse under dimensionless growth, coordination, cost, connectivity variables.
- **Level 7 — Endogenous regulation:** local systems without a direct complexity reward stabilize their own growth near the predicted region — expansion becomes part of the evolving organization rather than an external schedule.

---

# 28. What Would Count as a Major Result?

Not "we found pretty structures at \(g=0.03\)." A major result: *a dimensionless inequality derived from independently measurable occupation and coordination dynamics prospectively predicts the existence, location, and disappearance of sustainable complexity windows across distinct conservative substrates.* Stronger still: local growth rules spontaneously regulate \(\Gamma=g\tau_{\mathrm{coord}}\) near a common range without directly optimizing complexity — a candidate general principle of developmental or ecological organization.

---

# 29. Relationship to the Informational Battery

Domain growth changes the *accessibility* of physical capacity. Too little domain may trap \(B_{\mathrm{gross}}\) in congested or inaccessible configurations; appropriate growth may increase \(B_{\mathrm{op}}\) by opening pathways, separating structures, and reducing destructive interference; excessive growth may decrease \(B_{\mathrm{op}}\) through dilution and fragmentation even while gross capacity remains present. Thus **expansion can change accessibility without creating energy** — an *accessibility transformation*, not free recharge. Because growth itself costs capacity, net operational benefit requires \(\Delta B_{\mathrm{op}}>C_{\mathrm{grow}}\).

This is where the **capacity-growth reading** of the core idea attaches. The Founding Panel and the geo-canon provenance (Paper 14 → `canon/30-meaning/02-recharge-role-and-heat-death.md`) frame the recharge role not as restoring lost order but as *expanding the capacity of what can be aligned*, \(I_N\to I_{N+k}\). Paper 4's \(B_{\mathrm{op}}\) shifts are the toy, auditable image of that reading: a growing domain can raise operational capacity above its pre-growth ceiling. The panel's discipline is explicit (Einstein/Round 2): the capacity-growth reading is *more distinctive* than the entropy-drives-expansion claim that died against prior art, but it is wilder and has no covariant formulation — so it stays a labeled toy here, and the cosmological version lives on a separate falsification calendar. Reading Paper 4's \(I_N\to I_{N+k}\) analogue as cosmology fires `COMMANDED_EXPANSION`.

---

# 30. Relationship to Agency

Paper 4 does not assume structures choose the growth rate. Scheduled growth is environmental; density-responsive local growth may be endogenous but still purely reactive. Agency requires the Paper 2 intervention standard: an internal model of future growth consequences, policy alternatives, causal use of that model, and net benefit after model cost. A self-regulating expanding structure is therefore not automatically reflective or conscious.

---

# 31. Relationship to Biological Development

A successful window could motivate comparisons with organism growth, tissue morphogenesis, colony expansion, ecological range growth, vascular development, and network development. Such comparisons require biological models and data; the simulation cannot establish that real organisms optimize the same dimensionless variables.

---

# 32. Relationship to Cosmology

The observed universe expands according to a relativistic spacetime geometry constrained by cosmological observations; Paper 4 changes the number or connectivity of sites in an artificial causal domain. These are not equivalent. A positive Paper 4 result would show *certain artificial systems develop greater persistent complexity under an intermediate rate of costly domain growth*. It would **not** show that the universe expands in order to create life, that dark energy is informational complexity, or that cosmic expansion follows the IF growth rule. A cosmological IF theory would have to derive an effective metric, covariant field equations, the expansion history, structure growth, gravitational lensing, the CMB, and distinctive observational predictions (the `SECTOR_SPLIT_FIT` and Noether one-state gates apply there). Paper 4 supplies only a computational intuition and a possible complexity-selection principle.

---

# 33. Criteria for Rejection or Major Revision

Reject or substantially revise if: no interior optimum under preregistered metrics; the optimum is explained entirely by final domain size; total resources were not matched; the result disappears after correcting detector-density bias; low/high-growth failure mechanisms cannot be distinguished; independently measured timescales do not predict the interval; no dimensionless relationship transfers across rule families; the window occurs only at isolated numerical settings; expansion cost eliminates all apparent benefit; endogenous growth does not approach the predicted region; simpler growing-domain models explain the result completely; or the project repeatedly changes its definition of complexity to preserve the claim. Fired kills go in `SCOREBOARD.md` §Kill log the same session.

---

# 34. Conclusion

Expansion is neither automatically creative nor automatically destructive. It changes density, interaction rates, connectivity, resource accessibility, communication time, boundary pressure, and the cost of maintaining the substrate. The IF Expansion–Complexity Window Hypothesis proposes that these effects create two competing bounds — a lower bound set by saturation, \(g_{\min}\approx r_{\mathrm{occ}}-r_{\mathrm{release}}\), and an upper bound set by causal coordination, \(g_{\max}\approx\gamma_c/\tau_{\mathrm{coord}}\) — with central candidate inequality

\[
\boxed{\;r_{\mathrm{occ}}-r_{\mathrm{release}}\;<\;g\;<\;\frac{\gamma_c}{\tau_{\mathrm{coord}}}\;.}
\]

This is not yet a law; it is a prediction to be tested. The theory fails if complexity does not occupy an interior interval, if the interval cannot be predicted, or if it arises only through uncounted resources and chosen metrics. The strongest possible computational result is: *sustainable complexity occurs when domain growth relieves occupation pressure without outrunning causal coordination, and this balance is governed by transferable dimensionless ratios.* Such a result would **not** explain cosmological expansion; it would establish a disciplined artificial-life principle linking growth, resource accessibility, causal coherence, and persistent organization. The next paper examines the proposed transition from reactive structures to predictive agents: *The Agency Threshold: Critical Conditions for the Evolution of Predictive Control in IF Universes* (Paper 5).

---

# References

1. Escudero, C., Yuste, S. B., Abad, E. and Le Vot, F. "Reaction-Diffusion Kinetics in Growing Domains." (2018).
2. Nishihara, S. and Ohira, T. "The Bifurcation Growth Rate for Robust Pattern Formation in a Reaction-Diffusion System on a Growing Domain." (2024).
3. Barandiaran, M. and Stovold, J. "Growing Reservoirs with Developmental Graph Cellular Automata." (2025).
4. Plantec, E. et al. "Flow-Lenia: Emergent Evolutionary Dynamics in Mass-Conservative Continuous Cellular Automata." *Artificial Life* 31, 228–248 (2025).
5. Langton, C. G. "Computation at the Edge of Chaos: Phase Transitions and Emergent Computation." *Physica D* 42, 12–37 (1990).
6. Mitchell, M., Hraber, P. and Crutchfield, J. P. "Revisiting the Edge of Chaos: Evolving Cellular Automata to Perform Computations." (1993).
7. Mitchell, M., Crutchfield, J. P. and Hraber, P. T. "Dynamics, Computation, and the 'Edge of Chaos': A Re-Examination." (1993).
8. Wilkerson, G. J. "Functional Percolation: A Perspective on Criticality of Form and Function." (2025–2026).
9. Dorogovtsev, S. N., Goltsev, A. V. and Mendes, J. F. F. "Critical Phenomena in Complex Networks." *Reviews of Modern Physics* 80, 1275 (2008).
10. Waldegrave, R. et al. "Developmental Graph Cellular Automata." Artificial Life Conference Proceedings (2023).
