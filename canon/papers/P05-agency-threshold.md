# The Agency Threshold
## Critical Conditions for the Evolution of Predictive Control in IF Universes

**Author:** Phuc Vinh Truong
**Series:** IF Theory Canonical Papers
**Paper:** 5
**Layer:** SCIENCE
**Status:** canonical revision of extracted draft (2026-07-18)
**Supersedes:** canon/extracted/paper-05-extracted.md

> **Founding-Panel update (2026-07-18).** This revision replaces the single break-even
> ratio of the extracted draft with the **two-threshold formulation**: an *ablation*
> criterion \(\Pi_A\) and a *competitive* criterion \(\Pi_C\), separated by a
> thermodynamic **parasite band**. The band was observed in notebook 04 v0.1
> (\(p^*_1=0.64\), \(p^*_2\approx0.995\)) before any derivation existed, and is
> derived as *necessary* in `canon/00-foundations/04-break-even-theorem.md`. The
> universality claim IF-H1 is restated as the invariant \(\Theta^*\equiv\Pi_A|_{\Pi_C=1}\)
> agreeing across ≥3 rule families. Cooperation, where it appears, is named
> **agency-preserving cooperation** (the meaning-layer name "MaxLove" lives only in
> `canon/30-meaning/01-maxlove.md`).

---

## Abstract

Persistent structures may react to their environments without predicting them. A thermostat responds to current temperature; a chemical network follows local concentrations; an attractor returns toward a stable state after disturbance. Predictive agency requires more: a system must physically maintain internal information about future-relevant conditions, use that information to select actions, and obtain enough additional work or viability to repay the costs of sensing, memory, prediction, computation, and control — costs that couple the information ledger to the energy and thermodynamic-entropy ledgers via Landauer-type terms (Paper 0, Paper 2; three-ledger discipline, CLAUDE.md §1).

This paper proposes the **IF Agency-Threshold Hypothesis**: predictive control becomes sustainable only when the causal benefit of an internal model exceeds the complete physical cost of maintaining and using it. But "exceeds the cost" is ambiguous, and the ambiguity is the discovery. The **ablation** criterion charges only the model's own compute and asks whether the intact model beats its scrambled self; the **competitive** criterion charges the full apparatus and credits a reactive twin's free harvest of predictive information. These give two different thresholds, and between them lies a **parasite band**: a regime where a model pays for itself against ablation yet loses to a memoryless competitor. First observed empirically (\(p^*_1=0.64\), \(p^*_2\approx0.995\)) and then derived as necessary, the band is the program's first original result.

For a predictive controller and a matched reactive controller define \(\Delta J=J_P-J_R\), where \(J\) is net physical return after all modeled costs. The paper defines two dimensionless ratios, a primary agency order parameter, evolutionary protocols without intelligence rewards, intervention requirements, model-complexity transitions, finite-size tests, and cross-substrate validation, and specifies the conditions under which the claim must be rejected. The strongest possible result is the invariant \(\Theta^*\): the fraction of full apparatus cost concentrated in belief-maintenance at competitive break-even, agreeing across independently designed substrates. A result confined to one toy environment would remain a useful computational observation, not a universal physical law of agency.

---

## Keywords

Agency; predictive control; artificial life; phase transition; parasite band; information thermodynamics; semantic information; evolutionary dynamics; internal models; empowerment; causal intervention; resource-constrained intelligence.

---

# 1. Introduction

A system may persist without sensing, sense without remembering, remember without predicting, predict without using its prediction, or use prediction while spending more than the prediction saves. These distinctions define the problem of agency.

The thermodynamics of prediction distinguishes information about the past from information predictive of future environmental states; a system can retain extensive nonpredictive memory while dissipating more than a better-compressed predictive representation. Intervention-based semantic-information theory asks whether system–environment correlations are causally necessary for continued viability, scrambling selected correlations and comparing intact and intervened systems. Empowerment measures the channel capacity from actions to future sensory states — potential control, not proof that an internal model produced a net physical benefit. Information-bottleneck research shows that changes in the compression–prediction tradeoff can produce sharp representational transitions, concerning the onset of useful predictive representations but not the emergence of autonomous physical agents.

The IF question is therefore not "can information help control a system?" (established), but:

> Under what physical and environmental conditions does an internally maintained predictive model become useful enough to pay for itself and persist through physical selection — and does that transition possess a transferable critical structure, or is "agency" merely a gradual, substrate-specific continuum?

Paper 2 defined causal-work value through matched model interventions. Paper 5 asks when systems with positive causal-work information arise and remain stable in populations, and how the answer depends on *which* accounting boundary — ablation or competition — is used.

---

# 2. Scope

This paper studies the emergence of **predictive physical agency**. It does not establish phenomenal consciousness, metaphysical free will, moral responsibility, human-level reasoning, subjective experience, cosmic agency, or divine intention (layer firewall, CLAUDE.md §6). Primary systems: finite-state agents, resource-constrained spatial agents, evolving cellular/graph structures, stochastic controllers, simple model-based planners. Primary environments: Markov resource processes, hidden-state environments, spatially correlated resource fields, periodically changing conditions, regime-switching environments, adversarially perturbed environments. A system qualifies as predictive only relative to a declared system boundary, an environment family, an evaluation horizon, a controller comparator, a physical cost model, and a causal intervention.

---

# 3. The Two Thresholds and the Parasite Band

The single ratio of the extracted draft conflated two distinct questions. Both must be tracked, and the constitutional discipline (name the ledger, never merge boundaries) forces them apart.

## 3.1 The apparatus-boundary normalization

Never draw an absolute boundary around "the model." Following the break-even theorem (`canon/00-foundations/04-break-even-theorem.md`), define everything as a **difference against the canonical twin \(A_0\)** — the work-maximizing *memoryless* policy on the identical environment and sensor (the POMDP collapsed to the MDP on current observation, unique up to ties):

\[
\Delta C \equiv C[A]-C[A_0],\qquad \Delta W \equiv W[A]-W[A_0].
\]

Everything shared (outer wall, reservoirs, actuators) cancels in the difference; what remains is exactly the persistent state plus the compute that reads/writes it. The guardrail: \(A_0\) must be *optimal* — a crippled reference is a computable, detectable violation (a way to fake \(\Pi_C>1\), §28.11). "Where is the boundary?" becomes "solve this MDP."

## 3.2 The ablation criterion \(\Pi_A\)

The ablation criterion asks whether the intact internal model beats a scrambled version of itself, charging only the model's own compute:

\[
\boxed{\;\Pi_A=\frac{kT\,[\,I_{\mathrm{pred}}-I_{\mathrm{scr}}\,]}{C_{\mathrm{model}}}\;}\qquad(\text{internal causal efficiency}),
\]

with \(I_{\mathrm{pred}}=I(M_t;X_{t+\tau})\) the memory's information about the horizon the action couples to, \(I_{\mathrm{scr}}\) its scrambled counterpart, and \(C_{\mathrm{model}}\) the incremental compute of maintaining and reading \(M\). This is the intervention-based ratio of Paper 2. Ablation break-even is \(\Pi_A=1\); at that point the model earns back exactly the compute it costs *relative to destroying its predictive content in place*.

## 3.3 The competitive criterion \(\Pi_C\)

The competitive criterion asks whether the model-bearing system beats the optimal memoryless twin \(A_0\), charging the **full** apparatus difference and crediting the twin's *free* harvest of predictive information from the current sensor:

\[
\boxed{\;\Pi_C=\frac{kT\,[\,I_{\mathrm{pred}}-I_{\mathrm{react}}\,]}{\Delta C_{\mathrm{full}}}\;}\qquad(\text{architectural efficiency}),
\]

with \(I_{\mathrm{react}}=I(Y_t;X_{t+\tau})\) the predictive information the memoryless twin gets free from the current sensor. Competitive break-even is \(\Pi_C=1\).

## 3.4 The parasite band is a theorem, not a bug

Because \(\Pi_A\) ignores both \(I_{\mathrm{react}}\) and the non-memory overhead \(C_{\mathrm{overhead}}\) while \(\Pi_C\) charges them, competitive break-even occurs at strictly higher predictability than ablation break-even whenever the reactive twin extracts anything:

\[
\text{band width}=\beta\cdot C_{\mathrm{overhead}}+\big[\,I_{\mathrm{react}}-\mathrm{nostalgia}(p^*_1)\,\big]>0,
\]

where **nostalgia** \(=I_{\mathrm{mem}}-I_{\mathrm{pred}}\) is the self-deception term (stored bits with no predictive power, each a pure thermodynamic liability). Notebook 04 v0.1 observed exactly this — an ablation threshold \(p^*_1=0.64\) and a competitive threshold \(p^*_2\approx0.995\) — **before** the derivation existed. Simulation surprised; the math then showed it was necessary.

Between the thresholds sits the **parasite band**: predictabilities where a model is "smart" (beats its scrambled self, \(\Pi_A>1\)) yet net-negative in competition (loses to the optimal memoryless twin, \(\Pi_C<1\)). This is why a nominally intelligent system can be a thermodynamic parasite. Writing this up as a paper-grade note is the program's first original result (Founding Panel, Round 2, adjudication 2).

## 3.5 The invariant \(\Theta^*\) (rung-274177 seal condition)

Raw predictability \(p\) is a bad coordinate — predictive information diverges as \(p\to1\). Neither threshold alone is the invariant; the candidate universal is their ratio in the right coordinate:

\[
\boxed{\;\Theta^*\equiv\Pi_A\ \text{evaluated where}\ \Pi_C=1\;}
\]

— the fraction of full apparatus cost concentrated in belief-maintenance, at competitive break-even. Dimensionless, boundary-free by the twin-difference construction. **IF-H1 (restated):** \(\Theta^*\) takes the same critical value across ≥3 unrelated rule families (ring world / Kalman–LQG world / chemotaxis world). Agreement → IF has a constant; scatter → IF has curve fits. See `canon/00-foundations/04-break-even-theorem.md` for the full inequality (Sagawa–Ueda ceiling × Still floor × twin difference) and the two missing lemmas (the DPI-for-interventions lemma \(R\le0\), and the signed usable-information functional \(J\)).

## 3.6 Three phenomena, not to be conflated

Distinct from the two thermodynamic thresholds are two further transitions:

- **Evolutionary invasion threshold.** A predictive phenotype invades a reactive population when its expected net reproductive/continuation rate exceeds the resident's, \(f_P-f_R>0\). If reproductive output is proportional to accumulated surplus, this may coincide with \(\Pi_C=1\); under nonlinear survival, cooperation, density, or developmental constraints it may differ.
- **Collective critical transition.** A genuine critical transition requires population-level evidence — nonanalyticity in an infinite-size limit, finite-size scaling, divergent/peaked susceptibility, critical slowing, correlation-length growth, hysteresis/bistability, scaling collapse, reproducible universality classes. The existence of a break-even point does **not** prove a thermodynamic phase transition. The project must permit the conclusion: *predictive control exhibits an evolutionary crossover but no universal critical transition* (avoiding `phase-transition inflation`, §28.10).

---

# 4. Prior Art and Novelty Boundary

**4.1 Thermodynamics of prediction.** Still et al. related inefficiency to information retained about the past that fails to predict the future. IF cannot claim novelty for "efficient systems should retain predictive rather than irrelevant historical information." The IF extension tests whether the *net* benefit determines evolutionary persistence — and separates ablation from competition.

**4.2 Semantic information.** Kolchinsky and Wolpert define information as semantic when disrupting selected correlations reduces a viability function, and discuss automatic selection of system boundaries, timescales, and decompositions. IF cannot claim novelty for interventionally defined useful information; its contribution is a cost-aware, twin-normalized evolutionary threshold using net physical return and model ablation.

**4.3 Empowerment.** Empowerment measures potential control via action-to-future-state channel capacity and can generate behavior without task rewards. IF distinguishes potential influence from realized net causal-work benefit: many reachable futures do not imply resources, a model, viable policies, or physical efficiency to exploit them. Empowerment remains an important baseline.

**4.4 Information-bottleneck transitions.** Sharp transitions where predictive representations become learnable establish that predictive representation can undergo mathematically defined transitions — but not that the same boundary governs physical self-maintenance, resource competition, evolutionary invasion, or endogenous action. IF must test the bridge, not assume it.

**4.5 Good-regulator and internal-model principles.** Control theory associates successful regulation with internal models; embodied formulations ask when a situated agent must model its environment. IF cannot claim that good agents need models as an original proposition; its question is when the physical benefit exceeds full cost and whether that boundary is transferable.

**4.6 Predictive representations in RL.** Predictive-information objectives improve sample efficiency, but such agents are optimized with external objectives and infrastructure. The strongest IF experiment requires predictive control to emerge through local physical resource competition *without* a direct prediction, reward, or intelligence objective.

**4.7 Provisional novelty claim.** Not the existence of predictive control, but: *a dimensionless, intervention-validated, twin-normalized benefit-to-cost structure — with a derived parasite band and a candidate invariant \(\Theta^*\) — prospectively predicts when internal predictive models emerge, invade, persist, and increase in complexity across multiple independently designed substrates.* A conjecture until demonstrated.

---

# 5. Definitions

**5.1 Reactive controller.** \(A_t=\pi_R(O_t)\); fixed parameters, no physically instantiated state carrying information beyond the current observation. (The optimal reactive controller is the twin \(A_0\) of §3.1.)
**5.2 Memory-dependent controller.** \(A_t=\pi_M(O_t,M_t)\), \(M_{t+1}=U_M(M_t,O_t,A_t)\); memory may encode the past without predicting the future.
**5.3 Predictive controller.** Internal state with \(I(M_t;Y_{t+\tau}\mid O_t)>0\) for a future-relevant \(Y\), used causally \(M_t\to A_t\to Y_{t+\tau}\text{ or }V_{t+\tau}\).
**5.4 Adaptive predictive controller.** Updates its predictive mechanism when the environmental transition law changes; must beat a frozen predictive controller on held-out regime shifts after including adaptation cost.
**5.5 Counterfactual controller.** Evaluates ≥2 action-conditioned future distributions \(P(Y_{t+\tau}\mid do(A_t=a),M_t)\) and selects using them — a higher agency level than one-step prediction.
**5.6 Endogenous controller.** Information and action-selection occur within the declared boundary; a remote oracle supplying correct actions creates no local agency unless the oracle's full physical costs and role are included.

---

# 6. Physical Return

Useful work exported over horizon \(\tau\) is \(W_{\mathrm{out}}(\tau)\); complete controller cost is

\[
C_{\mathrm{ctrl}}=C_{\mathrm{sense}}+C_{\mathrm{memory}}+C_{\mathrm{prediction}}+C_{\mathrm{compute}}+C_{\mathrm{communication}}+C_{\mathrm{actuation}}+C_{\mathrm{repair}}+C_{\mathrm{reset}}.
\]

Net work return \(J_W(\tau)=\mathbb E[W_{\mathrm{out}}(\tau)-C_{\mathrm{ctrl}}(\tau)]\); predictive advantage \(\Delta J_W=J_W^P-J_W^R\); positive predictive advantage requires \(\Delta J_W>0\). The boundary must include externally prepared low-entropy memory, remote computation, sensor power, and actuator power (Paper 2 §5).

---

# 7. Causal Validation

A predictive controller may outperform a reactive one for reasons unrelated to prediction, so the internal model must pass the Paper 2 intervention family: **scrambled model** (matched state, predictive relationship destroyed — the numerator of \(\Pi_A\)); **temporally displaced model** (\(M_{t-\Delta}\)); **irrelevant model** (equally large representation of an irrelevant variable); **policy disconnection** (retain \(M_t\), remove model→action path); **false model** (systematically wrong predictions at matched cost). A system counts as predictively controlled only if the intact model outperforms these controls in the preregistered pattern.

---

# 8. The Two Ratios, Operationally

Define gross enabled work \(\Delta W_{\mathrm{enabled}}=W_{\mathrm{out}}^P-W_{\mathrm{out}}^R\) and incremental model cost \(C_{\mathrm{model}}=C_{\mathrm{ctrl}}^P-C_{\mathrm{ctrl}}^R\). The **ablation ratio** compares the intact model to its scrambled self, charging \(C_{\mathrm{model}}\); the **competitive ratio** compares the model-bearing system to the optimal memoryless twin \(A_0\), charging the full apparatus difference \(\Delta C_{\mathrm{full}}\) and crediting \(I_{\mathrm{react}}\):

\[
\Pi_A=\frac{kT[I_{\mathrm{pred}}-I_{\mathrm{scr}}]}{C_{\mathrm{model}}},\qquad
\Pi_C=\frac{kT[I_{\mathrm{pred}}-I_{\mathrm{react}}]}{\Delta C_{\mathrm{full}}}.
\]

Interpretation of each: \(<1\) unprofitable, \(=1\) break-even, \(>1\) net surplus. The competitive net surplus is \(\mathcal W_{\mathrm{net}}=\Delta C_{\mathrm{full}}(\Pi_C-1)\). The **parasite band** is precisely \(\{\,\Pi_A>1\ \wedge\ \Pi_C<1\,\}\).

---

# 9. Viability Return

Work output is not the only meaningful outcome. With survival probability \(S(\tau)\), expected lifetime \(T_{\mathrm{life}}\), and future operational battery \(B_{\mathrm{op}}(\tau)\), report \(\Delta S=S_P-S_R\), \(\Delta T_{\mathrm{life}}=T_{\mathrm{life}}^P-T_{\mathrm{life}}^R\), \(\Delta B_{\mathrm{op}}=B_{\mathrm{op}}^P-B_{\mathrm{op}}^R\) **separately** — these are not added to joules. A combined objective is allowed only if the physical rules themselves convert continuation into expected future resource flow, e.g. \(V_{\mathrm{physical}}=\mathbb E[\int_0^\tau P_{\mathrm{survive}}(t)P_{\mathrm{capture}}(t)\,dt]\); the conversion must be explicit (three-ledger discipline; adding bits to joules is `ENTROPY_CONFLATION`).

---

# 10. Primary Agency Order Parameter

Fraction of systems with intact predictive causal value:

\[
m_A=\frac1N\sum_{i=1}^N\mathbf 1\!\left[\mathcal W_{C,i}>0\ \wedge\ I(M_i;Y_{\mathrm{future}}\mid O_i)>I_{\min}\ \wedge\ \Delta A_i^{\mathrm{disconnect}}>A_{\min}\right],
\]

with interventionally measured causal-work value \(\mathcal W_{C,i}\), estimator-null-derived \(I_{\min}\), disconnection response \(\Delta A_i^{\mathrm{disconnect}}\), preregistered \(A_{\min}\). Primary evolutionary order parameter \(x_P=N_{\mathrm{predictive}}/N_{\mathrm{population}}\). Both are reported because an architecturally model-bearing system may not actually use prediction in a given environment.

---

# 11. Minimal Analytical Environment

Resource state \(Y_t\in\{L,R\}\) persisting with probability \(P(Y_{t+1}=Y_t)=r\). A reactive controller sees no cue before committing and succeeds with \(q_R=\tfrac12\); a one-step predictive controller stores \(Y_t\) and chooses it at \(t+1\), succeeding with \(q_P=r\). With capture work \(W_F\) and predictive-model cost \(C_M\):

\[
J_R=\tfrac12 W_F-C_R,\qquad J_P=rW_F-C_R-C_M,\qquad \Delta J=\left(r-\tfrac12\right)W_F-C_M.
\]

The **ablation** threshold in this toy is \(r_c=\tfrac12+C_M/W_F\), i.e. \(\Pi_A=(r-\tfrac12)W_F/C_M>1\). The **competitive** threshold sits higher whenever the memoryless twin harvests any reactive predictive information \(I_{\mathrm{react}}>0\) — recovering the parasite band in the analytic model. (Notebook 05A derives both; the ring-world notebook 04 is where \(p^*_1,p^*_2\) were first measured.)

---

# 12. Evolutionary Stability

Let \(x\) be the predictive fraction with frequency-independent returns \(f_P=J_P,\,f_R=J_R\). The replicator equation \(\dot x=x(f_P-\bar f)\), \(\bar f=xf_P+(1-x)f_R\), reduces to \(\dot x=x(1-x)\Delta J\). For \(\Delta J<0\), \(x=0\) is stable; for \(\Delta J>0\), \(x=1\) is stable; at \(\Delta J=0\) the phenotypes have equal return. This stability exchange is governed by the **competitive** break-even \(\Pi_C=1\) (invasion is a contest against the resident memoryless twin, not against a scrambled self). It is an evolutionary bifurcation in the idealized deterministic model, **not** automatically a thermodynamic phase transition.

---

# 13. Mutation and Finite Populations

With predictive↔reactive mutation rate \(\mu\), \(\dot x=x(1-x)\Delta J+\mu(1-2x)\); for \(\mu>0\) the absorbing states disappear and the sharp exchange becomes a smoother transition. In finite populations, drift allows predictive agents below break-even, loss above break-even, threshold broadening, and fixation delays. The empirical order parameter should be \(P_{\mathrm{fix}}(P)\) or the stationary predictive frequency, not one deterministic outcome; transition width should scale with population size, mutation rate, environmental variance, and evaluation horizon.

---

# 14. Model Complexity Thresholds

Prediction is not binary. With complexity \(K\in\{0,\ldots,K_{\max}\}\), accuracy \(q(K)\), cost \(C(K)\), net return \(J(K)=q(K)W_F-C(K)-C_R\), the selected complexity is \(K^*=\arg\max_K[q(K)W_F-C(K)]\). As predictability or resource value changes, \(K^*\) may change smoothly, in discrete jumps, with hysteresis, or through coexistence. The information-bottleneck literature gives precedent for sharp representational changes as prediction–compression tradeoffs vary, but IF must test whether comparable transitions occur under *physical resource selection* rather than a chosen optimization multiplier.

---

# 15. Hidden-State Environments

A one-step Markov environment rewards simple memory; to test genuine model formation, use a hidden Markov environment with latent \(H_t\in\{1,\ldots,K\}\), observations \(P(O_t\mid H_t)\), resources \(P(Y_{t+\tau}\mid H_t)\). A reactive controller receives \(O_t\); a predictive controller maintains belief \(b_t(h)=P(H_t=h\mid O_{\le t},A_{<t})\) updated \(b_{t+1}=\mathcal B(b_t,O_{t+1},A_t)\). A model becomes useful only when integrating observations improves future action enough to repay inference cost — allowing tests of state-estimation depth, memory compression, uncertainty, model mismatch, belief revision.

---

# 16. Environmental Predictability

Agency should not be maximal in every environment. A fully predictable environment may need only a reflex; a fully random one makes model maintenance wasteful; a moderately structured one may give strong predictive advantage. This suggests **predictive agency may peak at intermediate environmental complexity**, not at maximum predictability. The relevant quantity is *actionable predictability beyond current observation*, \(I_{\mathrm{actionable}}=I(Y_{t+\tau};O_{<t}\mid O_t,A_t)\); if zero, memory adds no advantage. (This is why the twin normalization matters: \(I_{\mathrm{react}}\) captures exactly what the memoryless twin already gets for free.)

---

# 17. Environmental Volatility and Adaptation

A fixed model can become harmful after a regime shift at \(t_s\). A fixed predictive controller retains the old model; an adaptive controller pays update cost \(C_{\mathrm{adapt}}\) and is profitable when \(\int_{t_s}^{t_s+\tau}[W_A(t)-W_F(t)]\,dt>C_{\mathrm{adapt}}\), giving a second threshold \(\tau_{\mathrm{stable}}>\tau_{\mathrm{break\text{-}even}}^{\mathrm{adapt}}\). If regimes change faster than adaptation repays its cost, flexible modeling is selected against.

---

# 18. Exploration and Model Acquisition

Predictive models require data; an agent may sacrifice immediate work for information. With exploration cost \(C_{\mathrm{explore}}\) and future model-enabled surplus \(G_{\mathrm{future}}\), exploration is justified when \(G_{\mathrm{future}}>C_{\mathrm{explore}}\), distinguishing reactive exploitation, passive learning, and active information seeking. Empowerment and intrinsic-motivation frameworks are comparison baselines; the IF question is whether information-seeking emerges from long-run resource accounting *without* an intrinsic information reward.

---

# 19. Strong Definition of Predictive Agency

A candidate qualifies only when: (1) **persistence** — maintains operational identity; (2) **action** — internal state changes outcomes; (3) **internal model** — physically maintains future-relevant state; (4) **causal use** — ablation changes action and outcome; (5) **positive return** — benefit exceeds full cost under at least part of the environment family, i.e. \(\Pi_C>1\) somewhere (not merely \(\Pi_A>1\)); (6) **endogeneity** — no external oracle supplies decisions; (7) **breadth** — benefit survives held-out conditions; (8) **adaptability** — model revises after failure (higher levels); (9) **evolutionary viability** — the phenotype persists without a direct agency reward. A controller meeting only 1–4 possesses causal predictive information but may sit in the parasite band — physically sustainable agency additionally requires clearing \(\Pi_C=1\).

---

# 20. Core Hypotheses

**AT-H1 — Ablation break-even.** Predictive controllers achieve positive net advantage against their scrambled selves when \(\Pi_A>1\). *Falsifier:* complete accounting shows no relationship between \(\Pi_A\) and ablation-referenced net return.

**AT-H1′ — Competitive break-even and the parasite band.** Against the optimal memoryless twin, net advantage requires \(\Pi_C>1\), and there exists a nonempty regime \(\{\Pi_A>1,\Pi_C<1\}\) whose width is \(\beta C_{\mathrm{overhead}}+[I_{\mathrm{react}}-\mathrm{nostalgia}(p^*_1)]>0\). *Falsifier:* competitive and ablation thresholds coincide across families, i.e. no reactive twin ever harvests free predictive information and \(C_{\mathrm{overhead}}=0\) — the band collapses.

**AT-H2 — Evolutionary invasion.** A rare predictive phenotype invades a reactive population when its causal-work surplus is positive after all developmental and maintenance costs (i.e. near \(\Pi_C=1\)). *Falsifier:* predictive invasion requires direct reward or parameters unrelated to causal-work value.

**AT-H3 — Predictability threshold.** For fixed model cost and resource value, predictive control disappears below a calculable level of actionable predictability. *Falsifier:* predictive models persist where past state contains no useful future information and no alternative benefit exists.

**AT-H4 — Finite-complexity.** Selected model complexity is finite under nonzero sensing/memory/computation costs. *Falsifier:* unbounded complexity is favored despite saturating benefit and rising cost.

**AT-H5 — Model-jump.** As resource value, predictability, or model cost varies, selected complexity can undergo discrete transitions. *Falsifier:* no repeatable transitions beyond artifacts.

**AT-H6 — Adaptation-timescale.** Adaptive prediction is selected only when regimes remain stable long enough for revision to repay its cost. *Falsifier:* adaptation persists when regime duration is systematically shorter than its break-even time.

**AT-H7 — Causal-specificity.** Predictive-state scrambling and policy disconnection reduce performance more than matched irrelevant-state interventions. *Falsifier:* any equal-size memory state performs equally well.

**AT-H8 — Reward-independence.** Predictive control can emerge under selection on resource capture, maintenance, and reproduction alone, without a prediction/intelligence reward. *Falsifier:* predictive behavior appears only when explicitly rewarded (a fired `TELEOLOGY_INJECTION`).

**AT-H9 — Cross-substrate invariant (IF-H1).** \(\Theta^*=\Pi_A|_{\Pi_C=1}\) takes the same critical value across ≥3 unrelated rule families (ring / Kalman–LQG / chemotaxis), predicting predictive-controller prevalence better than raw model size, raw mutual information, or substrate-specific parameters. *Falsifier:* \(\Theta^*\) scatters beyond bootstrap error across families — IF has curve fits, not a constant. (This is the rung-274177 seal condition.)

**AT-H10 — Criticality.** Under at least one well-defined limit, the onset of predictive control exhibits finite-size scaling or another accepted collective-transition signature. *Falsifier:* all apparent sharp transitions resolve into smooth finite-system crossovers without transferable scaling.

---

# 21. Evolutionary Simulation Design

**21.1 Population.** Resource-constrained systems, each with an energy/capacity store, sensors, controller, action interface, optional memory, reproduction machinery, and physical maintenance costs.
**21.2 Reproduction.** Only when accumulated surplus exceeds \(B_{\mathrm{rep}}\), costing \(C_{\mathrm{rep}}\). Offspring inherit controller architecture, mutable parameters, memory capacity, and model-update rules — not the parent's current environmental knowledge unless physical copying is implemented and costed.
**21.3 Death.** Ceases when \(B_i\le0\) or structural viability fails. No abstract fitness score.
**21.4 Mutation.** May change sensor precision, memory length, model order, update rate, planning depth, action policy, resource allocation; mutation and developmental costs included where relevant.
**21.5 No intelligence reward.** The simulator may not directly reward prediction accuracy, information gain, memory, model complexity, empowerment, or an agency score. Prediction survives only if it improves physical continuation or reproduction. (Enforces the Conway gate.)

---

# 22. Primary Experiments

1. **Analytical threshold recovery** — binary Markov environment; sweep \(r,W_F,C_M\); test the simulated ablation boundary against \(r_c=\tfrac12+C_M/W_F\) **and** locate the competitive boundary above it.
2. **Parasite-band mapping** — measure \(p^*_1\) (ablation) and \(p^*_2\) (competitive) directly; verify band width matches \(\beta C_{\mathrm{overhead}}+[I_{\mathrm{react}}-\mathrm{nostalgia}(p^*_1)]\); reproduce the notebook-04 observation \(p^*_1=0.64,\,p^*_2\approx0.995\) in a second family.
3. **Evolutionary invasion** — rare predictive mutant into a reactive population; invasion probability vs \(\Pi_C\); compare with finite-population fixation theory.
4. **De novo model evolution** — randomly parameterized controllers with no designated predictive module; test whether future-relevant internal states emerge only where \(\Pi_C>1\).
5. **Predictive vs historical memory** — equal-capacity memories storing predictive vs irrelevant vs shuffled histories; compare work, persistence, reproduction, causal ablation.
6. **Model-complexity ladder** — environments of increasing hidden-state order; map \(K^*(\text{predictability},W_F,C_M)\); test discrete transitions.
7. **Regime shifts** — reactive vs frozen vs adaptive vs meta-learning control; measure adaptation break-even time.
8. **Exploration threshold** — hidden structure; costly information-seeking; when does exploration evolve without an intrinsic information reward?
9. **Partial observability** — vary observation noise/ambiguity: low → reaction suffices, intermediate → prediction helps, extreme → prediction cannot recover the state.
10. **Action necessity** — accurate prediction but no action can alter the outcome; IF predicts \(I_{\mathrm{pred}}>0\) yet \(\mathcal W_C\approx0\), distinguishing forecasting from agency.
11. **Resource-value threshold** — fixed accuracy, varied resource value; prediction disappears when the consequence is too small to repay modeling cost.
12. **Structural agents** — embed controllers in Paper 3 structures; does the same threshold predict emergence of internal predictive states in self-maintaining structures?
13. **Expansion coupling** — predictive agents inside Paper 4 expanding domains; does the threshold shift with crowding, dilution, topology turnover, coordination time?
14. **Update-law ablation (04e; Conway/Dennett).** Does the agent's internal state carry mutual information about the *rules* (not just the state), and does ablating specifically that information selectively destroy adaptation to rule-changes? This targets the **self-reflection threshold** — model-of-the-update-law — the operational form of \(I_N\to I_{N+k}\).

---

# 23. Critical-Transition Tests

**23.1 Order parameter.** Primary \(x_P\) (population fraction with intervention-validated predictive control); secondary \(m_A,\langle K^*\rangle,\langle\mathcal W_C\rangle,I_{\mathrm{pred}},P_{\mathrm{fix}}\).
**23.2 Control parameter.** \(\lambda_1=\Pi_C\) (competitive; the invasion-relevant coordinate), \(\lambda_2=I_{\mathrm{actionable}}/(C_{\mathrm{model}}/W_F)\), \(\lambda_3=\tau_{\mathrm{environment}}/\tau_{\mathrm{model}}\); the primary is frozen before confirmatory analysis.
**23.3 Susceptibility.** \(\chi_A=\partial\langle x_P\rangle/\partial\Pi_C\approx(\langle x_P\rangle_{\Pi+\delta}-\langle x_P\rangle_{\Pi-\delta})/2\delta\); a peak may indicate transition sensitivity but is not sufficient alone.
**23.4 Critical slowing.** Perturb frequency/parameters, measure return time \(\tau_{\mathrm{return}}\); increasing return time near threshold supports critical slowing.
**23.5 Finite-size scaling.** Populations \(N\in\{N_1,\ldots,N_{\max}\}\); test transition width \(\Delta\Pi(N)\propto N^{-1/\nu}\) or another derived form; exponents are not universal without cross-model replication.
**23.6 Hysteresis.** Sweep the control parameter up and down; hysteresis may arise from developmental investment, agents changing the environment, social learning reducing model cost, or frequency-dependent bistability. No claim without controlled sweeps and equilibrium-time checks.

---

# 24. Agency Phase Taxonomy

- **A-P0 — Passive:** persistent structures, no action-dependent control.
- **A-P1 — Reactive:** current observations drive action; no future-relevant model required.
- **A-P2 — Memory:** internal state affects action but is primarily historical.
- **A-P3 — Predictive-correlational:** state predicts the future but ablation shows little causal contribution.
- **A-P4 — Predictive-control (parasite band):** prediction causally improves action (\(\Pi_A>1\)) but full cost exceeds benefit (\(\Pi_C<1\)).
- **A-P5 — Sustainable-agency:** predictive control clears \(\Pi_C>1\) — positive net work or continuation benefit against the memoryless twin.
- **A-P6 — Adaptive-agency:** models revised after environmental change.
- **A-P7 — Counterfactual-agency:** action-dependent futures compared.
- **A-P8 — Social-predictive:** systems model other agents.
- **A-P9 — Institutional:** persistent shared records/constraints reduce individual modeling cost and stabilize collective action.

Paper 5 focuses on A-P1 through A-P6. (The parasite band is exactly the A-P4/A-P5 boundary.)

---

# 25. Deterministic Jupyter-Notebook Program

Each notebook carries the contract cell and seed 65537.

- **05A — Two-Threshold Derivation:** derive and numerically verify \(r_c=\tfrac12+C_M/W_F\) (ablation) and the competitive threshold; produce exact and simulated phase maps with the parasite band shaded.
- **05B — Replicator Dynamics:** \(\dot x=x(1-x)\Delta J\); verify stability and break-even at \(\Pi_C=1\).
- **05C — Replicator–Mutation Dynamics:** add mutation; quantify transition smoothing; compare deterministic vs stochastic birth–death.
- **05D — Finite-Population Fixation:** \(P_{\mathrm{fix}}(P)\) across population sizes and causal-return values.
- **05E — Hidden Markov Environment:** exact Bayesian predictors of varying order; beliefs validated against known latent states.
- **05F — Predictive-Information Estimation:** \(I(M_t;Y_{t+\tau}\mid O_t)\); bias/uncertainty validated on known distributions.
- **05G — Causal Model Ablations:** erasure, scrambling, temporal displacement, irrelevant replacement, policy disconnection, false models.
- **05H — Physical Cost Model:** explicit costs for sensing, memory, belief update, planning, action, copying, resetting — plus the twin \(A_0\) reference solve.
- **05I — Model-Complexity Transitions:** sweep environmental complexity and model cost; track \(K^*\); search for discrete jumps.
- **05J — De Novo Evolution:** finite-state controllers without prediction rewards; do predictive states emerge only past \(\Pi_C=1\)?
- **05K — Environmental Regime Shifts:** fixed vs adaptive models; adaptation break-even time.
- **05L — Exploration Without Intrinsic Reward:** costly information-seeking; when does exploration emerge through future physical return?
- **05M — Critical Slowing and Susceptibility:** transition response, recovery time, population variance.
- **05N — Finite-Size Scaling:** does the transition sharpen or remain a crossover?
- **05O — Paper 3 Structural Integration:** evolving controllers inside resource-conserving structures.
- **05P — Paper 4 Expansion Integration:** agency thresholds across domain-growth regimes.
- **05Q — Cross-Substrate \(\Theta^*\) Test:** ring, Kalman–LQG, chemotaxis families; compare \(\Theta^*\) against bootstrap error (the IF-H1 seal test; the Kalman companion is `04f_kalman_theta_star.ipynb` in the theorem doc).
- **05R — Adversarial Audit:** a separate agent attempts to attribute predictive control to reward leakage, hidden oracle access, unmatched architecture, uncounted training energy, biased estimators, hand-selected environments, or arbitrary agency thresholds.

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
full_apparatus_cost: null
ablation_ratio_Pi_A: null
competitive_ratio_Pi_C: null
theta_star: null
parasite_band_width: null
predictive_information: null
reactive_information: null
nostalgia: null
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

**27.1 Environment holdout** — processes used to evolve controllers separated from those used to evaluate generalization.
**27.2 Architecture matching** — predictive and reactive controllers matched in action capacity, state count, update frequency, physical substrate, and observation access; incremental model cost not confounded with unrelated architecture changes. (The twin \(A_0\) is the disciplined form of this matching.)
**27.3 Search correction** — many model classes/environments searched → held-out environments, frozen metrics, multiple-comparison correction, independent replication.
**27.4 Transition preregistration** — freeze control parameter, order parameter, transition criterion, system-size sequence, scaling analysis, hysteresis protocol before opening confirmatory sweeps.
**27.5 Null models** — random memory; past-only memory; perfect prediction with no action influence; action influence with no prediction; uncosted oracle upper bound; **optimal memoryless twin \(A_0\)** (the competitive reference).

---

# 28. Failure Modes

Prediction reward leakage; hidden fitness score; external training subsidy; architecture mismatch; information-estimator bias (high-dimensional memory looks predictive from finite-sample error); correlation without causal use; action without prediction (memorized reflexes mislabeled predictive); environment overfitting; **threshold-by-definition** (order parameter constructed to change at \(\Pi=1\) — evolutionary prevalence and causal behavior must be independently measured); phase-transition inflation; **system-boundary manipulation** (training/memory-preparation/oracle cost excluded to force \(\Pi_C>1\) — defeated by the mandatory \(A_0\) twin normalization); **crippled-twin fraud** (a suboptimal \(A_0\) fakes \(\Pi_C>1\) — computable and detectable); intelligence inflation (one-step prediction described as consciousness or free will).

---

# 29. What Would Count as Success?

- **Level 1 — Two-threshold recovery:** simulation reproduces the exact toy ablation threshold and the higher competitive threshold, with the parasite band between.
- **Level 2 — Evolutionary invasion:** competitive surplus \(\Pi_C\) predicts invasion of predictive controllers.
- **Level 3 — De novo predictive emergence:** predictive states evolve without direct intelligence rewards, only past \(\Pi_C=1\).
- **Level 4 — Model-complexity prediction:** the framework predicts selected complexity.
- **Level 5 — Adaptive threshold:** regime duration and adaptation cost predict when flexible models evolve.
- **Level 6 — Cross-environment generalization:** the ratio predicts held-out environment families.
- **Level 7 — \(\Theta^*\) invariance:** one nondimensional value organizes agency onset across ≥3 independent substrates (the seal).
- **Level 8 — Laboratory validation:** the same law predicts onset of predictive control in physical/biological systems.
- **Level 9 — General bound:** a theorem establishes a broad physical limit relating predictive information, control benefit, and model cost (the \(J\)-functional / \(R\)-lemma program of the theorem doc).

---

# 30. What Would Count as a Major Discovery?

A strong artificial-life result: *predictive controllers evolve without a prediction reward when their interventionally measured physical benefit exceeds their full cost against a memoryless twin.* A field-creating result: *one dimensionless invariant \(\Theta^*\) prospectively predicts the emergence and selected complexity of internal models across independently designed computational and physical substrates.* Stronger still, a universal inequality \(\mathcal W_C\le\mathcal F(I_{\mathrm{pred}},B_{\mathrm{gross}},C_{\mathrm{control}},\tau)\) with systems shown approaching the bound — the break-even theorem's target.

---

# 31. Relationship to the Informational Battery

Paper 1 defined \(B_{\mathrm{gross}},B_{\mathrm{op}},B_{\mathrm{latent}}\). Predictive control can raise operational access \(\Delta B_{\mathrm{op}}>0\) without raising gross capacity; agency emerges when the accessibility gain repays its cost, \(\Delta B_{\mathrm{op}}>C_{\mathrm{model}}\) (ablation) and, more strictly, against the twin (competitive). The model does not create energy — it reduces the fraction of capacity that remains inaccessible.

---

# 32. Relationship to Emergent Structure

Paper 3 supplies candidate persistent structures; Paper 5 tests whether some develop internal states whose disruption selectively impairs future resource access. A self-repairing attractor is not predictive merely because it returns to a stable form. The distinguishing intervention: scramble future-relevant internal information while preserving material, energy, architecture, and current morphology; if recovery or resource capture declines selectively, the information has causal value.

---

# 33. Relationship to Expansion

Paper 4 predicts a sustainable domain-growth window; expansion may alter agency thresholds by changing encounter rates, memory usefulness, environmental stationarity, communication delay, resource density, and model-update cost. Possible prediction: \(\Pi_C=F(g,\tau_{\mathrm{coord}},\tau_{\mathrm{environment}})\). Very rapid substrate change may make deep models obsolete before they repay their cost; moderate growth may create enough novelty and opportunity to favor prediction.

---

# 34. Relationship to Consciousness

Predictive physical agency is not consciousness. A one-step model may satisfy \(\Pi_C>1\) without self-modeling, global access, competing-policy coordination, deep counterfactual reasoning, or subjective experience. The progression

\[
\text{reactive control}\to\text{predictive agency}\to\text{counterfactual agency}\to\text{functional consciousness hypotheses}
\]

requires separate evidence at each arrow. (The update-law ablation of §22.14 is the first operational rung above prediction — the self-reflection threshold.)

---

# 35. Relationship to Free Will

Paper 5 may establish that internal models causally influence action, \(M_t\to A_t\to Y_{t+\tau}\) — a functional sense of endogenous control. It does not establish metaphysical indeterminism; a deterministic agent can possess internal causal control under this framework. Whether such control suffices for free will is a philosophical question beyond the physical threshold (interpretation layer).

---

# 36. Criteria for Rejection or Major Revision

Reject or substantially revise if: predictive advantage does not track complete twin-normalized benefit-to-cost accounting; model interventions do not selectively affect outcomes; prediction emerges only through direct rewards; environment-specific parameters dominate all results; selected complexity cannot be predicted; apparent thresholds vanish under architecture matching; no result survives held-out environments; \(\Theta^*\) scatters across substrates; the transition is entirely an estimator artifact; simpler semantic-information or empowerment measures predict all outcomes equally well; ablation and competitive thresholds cannot be meaningfully connected; or the project repeatedly redefines agency to preserve results. Fired kills go in `SCOREBOARD.md` §Kill log the same session.

---

# 37. Conclusion

Prediction is not automatically agency; agency is not automatically consciousness. The IF Agency-Threshold Hypothesis begins narrowly: *an internal predictive model becomes physically sustainable when the additional accessible work or continuation it enables exceeds the complete cost of acquiring, maintaining, and using the model.* But "the cost" has two boundaries. Against its scrambled self a model breaks even at the ablation ratio \(\Pi_A=1\); against the optimal memoryless twin it breaks even higher, at the competitive ratio \(\Pi_C=1\); and between them lies the **parasite band** — where a model is smart yet net-negative, a thermodynamic parasite. The band was seen before it was derived (\(p^*_1=0.64,\,p^*_2\approx0.995\)) and is the program's first original result. The candidate universal is neither threshold but their ratio in the right coordinate,

\[
\boxed{\;\Theta^*\equiv\Pi_A|_{\Pi_C=1}\;,}
\]

the fraction of full apparatus cost concentrated in belief-maintenance at competitive break-even. IF-H1: \(\Theta^*\) agrees across ≥3 rule families → IF has a constant; it scatters → IF has curve fits. Whether the boundary produces evolutionary invasion, stable predictive populations, discrete complexity jumps, critical collective behavior, or a universal law is empirical; the theory succeeds only if the boundary predicts outcomes it was not designed around. The strongest version: *predictive agency is a physical transition in accessibility — matter begins maintaining models of possible futures when doing so reliably reveals more usable capacity than the models consume, measured against the twin that harvests the free predictive information for nothing.* If this fails to transfer beyond a toy environment, the universal agency-threshold hypothesis fails. If \(\Theta^*\) predicts the emergence of internally modeled control across artificial, chemical, biological, and engineered systems, IF Theory will have found a measurable physical bridge from self-organizing matter to agency.

---

# References

1. Still, S., Sivak, D. A., Bell, A. J. and Crooks, G. E. "The Thermodynamics of Prediction." *Physical Review Letters* 109, 120604 (2012).
2. Sagawa, T. and Ueda, M. "Fluctuation Theorem with Information Exchange." *Physical Review Letters* 104, 090602 (2010).
3. Kolchinsky, A. and Wolpert, D. H. "Semantic Information, Autonomous Agency and Non-equilibrium Statistical Physics." *Interface Focus* 8, 20180041 (2018).
4. Salge, C., Glackin, C. and Polani, D. "Empowerment — An Introduction." (2013).
5. Mohamed, S. and Rezende, D. J. "Variational Information Maximisation for Intrinsically Motivated Reinforcement Learning." (2015).
6. Wu, T. and Fischer, I. "Phase Transitions for the Information Bottleneck in Representation Learning." (2020).
7. Wu, T., Fischer, I., Chuang, I. L. and Tegmark, M. "Learnability for the Information Bottleneck." (2019).
8. Lee, K.-H. et al. "Predictive Information Accelerates Learning in Reinforcement Learning." (2020).
9. Virgo, N. "A Good Regulator Theorem for Embodied Agents." (2025).
10. Barato, A. C. and Seifert, U. "Thermodynamic Bounds on Information Flow." (2014).
11. Tiomkin, S. et al. "Intrinsic Motivation in Dynamical Control Systems." (2022–2023).
