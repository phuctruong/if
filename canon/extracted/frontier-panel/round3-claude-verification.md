# Frontier Panel Round 3 — Claude (Fable 5 High): Verification of R1–R3 + inequality audit (2026-07-18)

> V1 Θ* kill interpretation · V2 R>0 triviality check · V3 component-optimality attack
> · V4 inequality sign/unit audit · V5 decisive next experiment.

Thinking about validating experimental results challenging theoretical predictions on thermodynamic work and information

Marshaled technical rigor for five-part experimental verification

No preamble.

V1 — Is the Θ* kill correctly interpreted, and is lockstep salvageable?

The kill is correct and you interpreted it correctly: 3.5–9.6σ separation at every cost level means raw Θ* is not a rule-family invariant. Good — you logged it instead of explaining it away. That is the single most credibility-earning move in three rounds.

But be harsh with yourself about "lockstep." Two families tracking 5–15% apart across a 3× cost sweep is not yet evidence for a rescaled invariant — it is barely evidence for anything, because a 3× range in one parameter is a short lever arm. Any two smooth monotone functions of 
𝐶
C that happen to have similar log-slopes will look "lockstep" over 3×. You have one octave and a half. The 5–15% gap is the problem, not the reassurance: if a true invariant exists, the rescaled quantity should agree to within error bars (your seeds give you ~few-% bars), and 5–15% is several σ. So the honest status is: raw Θ is dead, and you do not yet have a replacement — you have a hint that the cost-dependence is close to a shared power law.*

The diagnosis is almost certainly dimensional. Θ* mixes 
𝐶
𝑚
𝑜
𝑑
𝑒
𝑙
C
model
	​

 (numerator, internal) and 
Δ
𝐶
𝑓
𝑢
𝑙
𝑙
ΔC
full
	​

 (denominator) but the 
1
/
𝐶
𝑀
𝐸
𝑀
𝑂
𝑅
𝑌
1/C
MEMORY
	​

 scaling says 
𝐶
𝑚
𝑜
𝑑
𝑒
𝑙
C
model
	​

 is not being divided out — it is leaking through linearly. So the rescaling to test is the one that removes that leak by construction, not by fitting.

Exact next test. Don't fit 
Θ
∗
⋅
𝐶
𝑚
𝑜
𝑑
𝑒
𝑙
𝛼
Θ
∗
⋅C
model
α
	​

 and hunt for 
𝛼
≈
1
α≈1 — that's numerology and a referee will say so. Instead go back to the theorem and form the quantity that is dimensionless before you take any threshold. Define the clean work-per-bit at competitive break-even:

𝜂
∗
  
≡
  
𝑊
𝑖
𝑛
𝑡
𝑎
𝑐
𝑡
−
𝑊
𝑠
𝑐
𝑟
𝑘
𝑇
 
Δ
𝐼
𝑢
𝑠
𝑒
∣
Π
𝐶
=
1
η
∗
≡
kTΔI
use
	​

W
intact
	​

−W
scr
	​

	​

	​

Π
C
	​

=1
	​


This is exactly your R2 clean work-per-bit (0.102) but evaluated at the competitive threshold, and it is dimensionless with no explicit 
𝐶
C in it — the cost dependence enters only through where 
Π
𝐶
=
1
Π
C
	​

=1 lands, not through the ratio's units. IF-H1, restated in falsifiable form: 
𝜂
∗
η
∗
 is the same across ring and Kalman to within seed error, independent of 
𝐶
𝑀
𝐸
𝑀
𝑂
𝑅
𝑌
C
MEMORY
	​

.

Falsifies it: 
𝜂
𝑟
𝑖
𝑛
𝑔
∗
≠
𝜂
𝐾
𝑎
𝑙
𝑚
𝑎
𝑛
∗
η
ring
∗
	​


=η
Kalman
∗
	​

 beyond ~2σ at *any* shared cost, OR 
𝜂
∗
η
∗
 drifts with 
𝐶
𝑀
𝐸
𝑀
𝑂
𝑅
𝑌
C
MEMORY
	​

 within a family. Either kills the universality claim outright — no third rescaling attempt. Two rescalings that fail is a program fishing for an invariant that isn't there, and you should pre-commit to stopping.
The trap to avoid: you already have 
𝜂
𝑐
𝑙
𝑒
𝑎
𝑛
≈
0.102
±
0.021
η
clean
	​

≈0.102±0.021 from R2. If 
𝜂
∗
η
∗
 (at threshold) equals that number, ask whether 
𝜂
∗
η
∗
 is just a Carnot-like bound (SU efficiency ceiling) that is trivially family-independent because it's set by thermodynamics, not by IF. A universal constant that turns out to be "the second law" is not a discovery. So the real test is whether 
𝜂
∗
<
1
η
∗
<1 by a family-specific margin or sits at a nontrivial shared value strictly below the SU ceiling. Report 
𝜂
∗
η
∗
 and the SU ceiling on the same axes.
V2 — Does R2 bear on the lemma, or is it trivial?

As run, it is close to trivial, and you should say so in the writeup before a referee does. "Kick the world and work changes" — yes, obviously; the dirty scramble opened a second channel (heat → ratchet) and 
𝑅
>
0
R>0 is definitional, not informative. The 3.4× / 11.6σ number proves your instrument can detect an off-shell channel. That has value — it validates the notebook as a discriminator — but it does not test the lemma 
𝑅
≤
0
R≤0, because the lemma is a claim about clean interventions and you demonstrated a dirty one. You confirmed the contrapositive's setup, not the theorem.

What the clean result does do is more important than the dirty one: **
𝜂
𝑐
𝑙
𝑒
𝑎
𝑛
=
0.102
±
0.021
η
clean
	​

=0.102±0.021 stable across 
𝑝
p is weak positive evidence for 
𝑅
≤
0
R≤0** — a constant work-per-bit is what you'd see if scrambling only ever destroys information-mediated work and nothing else. Lead with that. The dirty run is the negative control.

**A non-trivial 
𝑅
>
0
R>0 counterexample** — the thing that would actually kill the lemma — must satisfy: the intervention is provably confined to the memory channel (the world's transition operator is byte-identical under intact vs scrambled; only 
𝑀
M's contents change), yet 
𝑊
𝑖
𝑛
𝑡
𝑎
𝑐
𝑡
−
𝑊
𝑠
𝑐
𝑟
>
𝑘
𝑇
 
Δ
𝐼
𝑢
𝑠
𝑒
W
intact
	​

−W
scr
	​

>kTΔI
use
	​

. Concretely, the candidate is **memory–actuator back-action that isn't captured by 
𝐼
𝑢
𝑠
𝑒
=
𝐼
(
𝑀
;
work-coord
)
I
use
	​

=I(M;work-coord)**:

Build a world where the memory register is physically coupled to the actuator such that the act of holding a specific bit pattern changes an energy barrier (a memory-state-dependent potential), independent of what that pattern predicts. Scrambling changes the pattern → changes the barrier → changes extractable work, via a path that pairwise MI between 
𝑀
M and the work coordinate at the decision step does not register (because it's a static energetic coupling, not a predictive/feedback one).
Metric: 
𝑅
=
(
𝑊
𝑖
𝑛
𝑡
𝑎
𝑐
𝑡
−
𝑊
𝑠
𝑐
𝑟
)
−
𝑘
𝑇
 
Δ
𝐼
𝑢
𝑠
𝑒
R=(W
intact
	​

−W
scr
	​

)−kTΔI
use
	​

 with 
Δ
𝐼
𝑢
𝑠
𝑒
ΔI
use
	​

 computed as the feedback MI at the control step. If 
𝑅
>
0
R>0 robustly here, the lemma as stated is false — but the fix is also revealed: 
𝐼
𝑢
𝑠
𝑒
I
use
	​

 was the wrong functional; you need the **signed 
𝐽
J** from Round 2 T4, which includes the 
(
∂
𝑀
𝐴
)
(∂
M
	​

A) back-action term. So this counterexample doesn't destroy IF, it forces 
𝐼
→
𝐽
I→J. That's the notebook worth running, because either outcome is a result: 
𝑅
≤
0
R≤0 confirms the lemma for 
𝐼
𝑢
𝑠
𝑒
I
use
	​

; 
𝑅
>
0
R>0 promotes 
𝐽
J from conjecture to necessity.
V3 — Is R3 sound, or is the intact agent misconfigured?

This is the sharpest question you've asked and the answer is uncomfortable: as stated, R3's comparison is not yet valid, and the state-smoother "parasite" is the tell.

The twin-normalization rule (T2) requires 
𝐴
0
A
0
	​

 to be optimal precisely so that 
Δ
𝑊
>
0
ΔW>0 can't be manufactured by crippling the reference. The symmetric obligation you're now discovering: component ablation is only interpretable if the intact agent is optimal in the component being ablated. Here it isn't. "Raw-obs 0.81 > intact 0.61" doesn't show smoothing is a parasite in this regime — it shows your smoother (fixed 
𝛼
α) is mistuned for a switching-law world, where a static Kalman gain is provably suboptimal (optimal gain must jump at switches; a fixed 
𝛼
α lags exactly as you observed). You measured smoother-lag-from-misconfiguration, not smoothing-is-parasitic.

The precise rule you're missing, state it in the theorem doc:

*An ablation of component 
𝑐
c is interpretable only when the intact agent is Pareto-optimal in 
𝑐
c conditional on all other components — i.e., no local reparameterization of 
𝑐
c alone improves 
Δ
𝑊
𝑛
𝑒
𝑡
ΔW
net
	​

. Otherwise the ablation measures the config gap, not the component's causal-work contribution.*

This is not pedantry; it's the same optimality guardrail from T2 applied inward, and it's load-bearing: without it, every "parasite" finding is confounded with "we tuned it badly." The parasite band from Round 1 survives this (that was intact-optimal, memory-vs-none) — but R3's state result does not survive as written.

The salvage, and it's a better experiment: re-run R3 with the smoother gain optimized per regime (or an adaptive gain that's optimal for the switching statistics). Then:

If state-belief is still a parasite (raw-obs still wins with an optimally-tuned smoother) → now you have a real, deep result: in switching worlds, state estimation is genuinely net-negative and the rule-model carries all the causal-work load. That would be a legitimately novel dissociation.
If the parasite vanishes → it was misconfiguration, and the robust R3 finding is just the rule/state dissociation on the recovery axis (6σ), which is still worth reporting but is less surprising.

Either way the rule/state dissociation itself (ablating rule-model kills recovery, ablating state-belief doesn't) is probably robust to the tuning issue, because it's about which recovery dynamics break, not about the work-level comparison. Report that dissociation with confidence; hold the "smoother is a parasite" claim until the gain is optimized.

V4 — Verify the inequality
Δ
𝑊
𝑛
𝑒
𝑡
  
≤
  
𝑘
𝑇
[
𝐼
𝑝
𝑟
𝑒
𝑑
−
𝐼
𝑟
𝑒
𝑎
𝑐
𝑡
]
  
−
  
𝑘
𝑇
(
𝐼
𝑚
𝑒
𝑚
−
𝐼
𝑝
𝑟
𝑒
𝑑
)
  
−
  
𝐶
𝑜
𝑣
𝑒
𝑟
ℎ
𝑒
𝑎
𝑑
ΔW
net
	​

≤kT[I
pred
	​

−I
react
	​

]−kT(I
mem
	​

−I
pred
	​

)−C
overhead
	​


Three problems, one cosmetic, two real.

(a) Unit consistency — real. You flagged it yourself and you're right to worry. 
𝐼
𝑝
𝑟
𝑒
𝑑
,
𝐼
𝑟
𝑒
𝑎
𝑐
𝑡
,
𝐼
𝑚
𝑒
𝑚
I
pred
	​

,I
react
	​

,I
mem
	​

 must all be per-cycle quantities (bits per control step), and 
𝐶
𝑜
𝑣
𝑒
𝑟
ℎ
𝑒
𝑎
𝑑
C
overhead
	​

 must be per-cycle work (joules per step). As written that's consistent only if every term shares the horizon convention. The danger term is 
𝐼
𝑟
𝑒
𝑎
𝑐
𝑡
=
𝐼
(
𝑌
𝑡
;
𝑋
𝑡
+
𝜏
)
I
react
	​

=I(Y
t
	​

;X
t+τ
	​

) and 
𝐼
𝑝
𝑟
𝑒
𝑑
=
𝐼
(
𝑀
𝑡
;
𝑋
𝑡
+
𝜏
)
I
pred
	​

=I(M
t
	​

;X
t+τ
	​

) at horizon 
𝜏
τ, but the Still floor 
𝐼
𝑚
𝑒
𝑚
−
𝐼
𝑝
𝑟
𝑒
𝑑
I
mem
	​

−I
pred
	​

 is naturally a rate (nats dissipated per step to maintain memory). If 
𝜏
>
1
τ>1 step, 
𝐼
𝑝
𝑟
𝑒
𝑑
I
pred
	​

 at horizon 
𝜏
τ and the per-step maintenance term are on different clocks. Fix: define all information terms as per-step and let 
𝜏
τ be exactly one control-coupling interval, or carry an explicit 
1
/
𝜏
1/τ on the maintenance term. As written, there's an illegitimate mixing whenever 
𝜏
≠
1
τ

=1. Flag it.

(b) Sign of 
𝐼
𝑟
𝑒
𝑎
𝑐
𝑡
I
react
	​

 — correct, but the subtraction is doing something subtle you should state. 
−
𝑘
𝑇
 
𝐼
𝑟
𝑒
𝑎
𝑐
𝑡
−kTI
react
	​

 is right: the reactive twin harvests that for free, so it's credited to 
𝐴
0
A
0
	​

 and must be removed from 
𝐴
A's surplus. Fine. But note 
𝐼
𝑝
𝑟
𝑒
𝑑
−
𝐼
𝑟
𝑒
𝑎
𝑐
𝑡
I
pred
	​

−I
react
	​

 can be negative (a good sensor makes memory redundant), and then the RHS is negative and the inequality correctly says memory can't pay. Make sure the doc states this is a signed quantity, not 
∣
⋅
∣
∣⋅∣.

(c) Missing term — real, and it's the falsify channel. The inequality as written assumes memory information is either predictive or nostalgic (nonpredictive-but-neutral). It has no slot for anti-predictive information — the falsify case where 
𝐼
𝑚
𝑒
𝑚
>
0
I
mem
	​

>0 but the belief is systematically wrong, costing work. Under the current RHS, falsified memory looks like nostalgia (
𝐼
𝑚
𝑒
𝑚
−
𝐼
𝑝
𝑟
𝑒
𝑑
>
0
I
mem
	​

−I
pred
	​

>0) and is penalized linearly. But a confidently-wrong model drives actively harmful actions — the work loss is not bounded by 
𝑘
𝑇
 
𝐼
𝑚
𝑒
𝑚
kTI
mem
	​

, it can be arbitrarily large (drive the actuator hard in the wrong direction). The MI-based RHS cannot represent this, which is exactly why Round 2 T4 needed the signed 
𝐽
J. So: the inequality is correct for erase/scramble/nostalgia, and structurally incomplete for falsify. State its domain of validity explicitly — "holds for interventions that reduce 
𝐼
𝑢
𝑠
𝑒
I
use
	​

; the falsify operation requires the signed functional 
𝐽
J and is out of scope for this bound." A referee who feeds it a falsified-memory counterexample will otherwise claim a violation.

Net: no fatal sign error. One unit-clock bug (
𝜏
τ), one stated-domain gap (falsify). Both fixable in a sentence each, both must be fixed before circulation.

V5 — The single most important next experiment

The clean 
𝜂
∗
η
∗
 cross-family test from V1 — but run on a third family, chemotaxis, simultaneously with ring and Kalman, at a single fixed cost, with 
𝜏
=
1
τ=1 enforced.

Rationale, ranked against the alternatives:

The R2 clean-lemma counterexample (V2) is deeper but it tests the lemma; you can't sensibly test the lemma until you know whether 
𝐼
𝑢
𝑠
𝑒
I
use
	​

 or 
𝐽
J is the right object, and that's decided by whether 
𝜂
∗
η
∗
 (built on 
𝐼
𝑢
𝑠
𝑒
I
use
	​

) is even stable. Order matters: invariant first, then lemma.
R3's re-tuning is necessary hygiene but it's a fix, not a discovery.
The whole program's central live claim is IF-H1 universality, and it currently rests on two families that failed raw and only "track" over a short lever. Two points cannot establish a constant and you know it. A third family is the difference between "we fit a curve twice" and "three structurally unrelated substrates share a dimensionless number." If 
𝜂
𝑐
ℎ
𝑒
𝑚
𝑜
∗
η
chemo
∗
	​

 lands on 
𝜂
𝑟
𝑖
𝑛
𝑔
∗
=
𝜂
𝐾
𝑎
𝑙
𝑚
𝑎
𝑛
∗
η
ring
∗
	​

=η
Kalman
∗
	​

 within error, IF-H1 is alive and publishable. If it scatters, IF-H1 is dead and you've saved yourself papers 4–15.

Concrete spec: run-and-tumble chemotaxis agent, explicit per-bit metabolic debit, same protocol, locate 
Π
𝐶
=
1
Π
C
	​

=1, report 
𝜂
𝑐
ℎ
𝑒
𝑚
𝑜
∗
η
chemo
∗
	​

 with 8-seed bars against the other two on one plot, plus the SU efficiency ceiling on the same axes so you can tell a real shared constant from a rediscovered second law. One notebook. It either promotes IF-H1 to a genuine empirical regularity across three substrates or kills it. Everything else waits on that number.