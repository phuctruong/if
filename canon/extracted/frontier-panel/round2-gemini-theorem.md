# Frontier Panel Round 2 — Gemini: The IF Break-Even Inequality (2026-07-18)

(via button ↻ continued)
T1. The Theorem Statement (The IF Break-Even Inequality)

Theorem: No agent can extract strictly positive net competitive work ΔW
net
	​

 from an interventionally-preserved internal information state M above its full apparatus cost C
model
	​

 except when the predictive mutual information rate 
I
˙
pred
	​

=I(M
t
	​

;E
t+τ
	​

) strictly bounds the sum of the metabolic erasure cost and the structural impedance mismatch.

Formal Condition:

ΔW
net
	​

≡W
intact
	​

−W
memoryless
	​

−C
model
	​

≤kT(
I
˙
pred
	​

−
I
˙
nostalgia
	​

)−ΔW
shock
	​


Equality holds if and only if the agent operates at the reversible Landauer limit (
Q
˙
	​

diss
	​

=kTln(2)
H
˙
mem
	​

) and the environmental statistics are strictly stationary over the prediction horizon τ.

Here, C
model
	​

 encompasses all sensing, compute, and memory costs. 
I
˙
nostalgia
	​

 is the information the agent retains about the past that has zero predictive power for the future (as defined by Still et al.). ΔW
shock
	​

≡W
memoryless
	​

−W
scrambled
	​

 represents the "adaptation trap": the baseline energetic penalty an agent suffers when its functionally integrated control loops are broken, compared to a structurally simpler agent evolved without those loops.

T2. The Canonical Apparatus-Boundary Normalization

To close the "measure is a knob" loophole, the system boundary cannot be designated by the observer; it must be topologically defined by the thermodynamic fluxes of the model.

The Rule (The Landauer-Flux Blanket):
The canonical boundary ∂Ω is the minimal contiguous spatiotemporal Markov blanket where the cross-surface heat flux 
Q
˙
	​

∂Ω
	​

 satisfies the equality condition for the erasure of the enclosed structural states: 
Q
˙
	​

∂Ω
	​

≥kTln(2)
H
˙
(S
internal
	​

).

Any physical substrate (sensor, membrane, register) whose state-changes contribute to the non-equilibrium heat dissipation required to maintain the predictive correlations must be classified as S
internal
	​

. If you attempt to arbitrarily expand the boundary to include the environment, the local heat flux decoupled from the agent's computation violates the equality, forcing the boundary back to the agent's true dissipative skin.

T3. The Dimensionless Invariant

The two observed thresholds, the interventional ablation limit p
1
∗
	​

 and the competitive viability limit p
2
∗
	​

, are united by a dimensionless ratio representing the Parasitic Burden (χ).

Given your observables, the ablation yield is Π
A
	​

=
C
model
	​

W
intact
	​

−W
scrambled
	​

	​

 and the competitive yield is Π
C
	​

=
C
model
	​

W
intact
	​

−W
memoryless
	​

	​

.
We construct the invariant χ:

χ=Π
A
	​

−Π
C
	​

=
C
model
	​

W
memoryless
	​

−W
scrambled
	​

	​


The Hypothesis: For IF-H1 universality to hold, the critical threshold ratio Υ
IF
	​

 must be an invariant of the system's dimensionality and the predictability gradient p:

Υ
IF
	​

=(
p
1
∗
	​

p
2
∗
	​

−p
1
∗
	​

	​

)(
χ
∗
1
	​

)≈const

The true invariant is built from both: it defines the exact width of the "thermodynamic parasite band" normalized by the entropic cost of maintaining the structurally dependent loops (χ
∗
).

T4. The Proof-Sketch Strategy

Existing Machinery:

Sagawa-Ueda Fluctuation Theorems: Provides the absolute upper bound on work extraction under feedback: W
ext
	​

≤−ΔF+kTI.

Still et al. (Thermodynamics of Prediction): Proves that non-predictive information (
I
˙
nostalgia
	​

) strictly necessitates thermodynamic dissipation.

Barato-Seifert Transducer Framework: Handles the bipartite continuous-time thermodynamics of an agent coupled to an environment.

The Genuinely Missing Lemma (The Interventional Shock Lemma):
Existing theorems treat the agent's capacity C as given and compare against a generic non-feedback protocol. IF Theory requires formalizing the transition between structurally distinct agents (W
scrambled
	​

 vs W
memoryless
	​

).

We must prove that scrambling a causally adapted memory state induces a transient dissipative shock ΔW
shock
	​

>0 that is mathematically bounded by the Kullback-Leibler divergence between the scrambled internal policy distribution and the optimal memoryless reactive policy:

ΔW
shock
	​

≥kT⋅D
KL
	​

(P(Action∣M
scrambled
	​

)∥P(Action∣M
optimal_memoryless
	​

))

This lemma bridges the gap between interventional ablation (breaking a complex machine) and competitive evaluation (comparing to a simpler machine), proving why the thermodynamic parasite band exists.

T5. Minimal Deterministic Refutation Notebook

The Experiment: The "Geometric Resonance Cheat"

Setup:

Environment: A deterministic 1D cyclic cellular automaton with a globally periodic, non-stochastic wave function that shifts right by v cells per timestep. Predictability p=1.0.

Agent 1 (Intact): Possesses memory M of the last 3 cell states, computes the next wave position, and moves. Pays C
model
	​

 per step.

Agent 2 (Memoryless): Reads only the current cell, moves randomly if 0, stays if 1.

Agent 3 (The Cheat): An agent with a physical morphology exactly matching the spatial wavelength of the environment. It acts like a "gear" locking into the wave. It possesses NO dynamic memory bits (thus 
I
˙
mem
	​

=0, Landauer cost =0) but extracts massive work due to physical impedance matching.

The Metric: Run the system and track ΔW
net
	​

 for the Cheat.

What Kills the Theorem: If the Cheat agent extracts ΔW
net
	​

>0 while bypassing the C
model
	​

 and kT
I
˙
pred
	​

 constraints entirely because its "information" is stored permanently in its static morphology rather than an interventionally-preservable active state, then T1 fails. It proves that static structural matching can dominate dynamic predictive agency, rendering the IF Battery ledger incomplete.

How does the IF Theory framework formally distinguish between dynamic information encoded in non-equilibrium active memory (which accrues Landauer costs) and static information embedded in the agent's physical morphology (which does not)?