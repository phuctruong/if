# Frontier Panel Round 2 — Claude (Fable 5 High): Constructing IF's kT ln 2 (2026-07-18)

> Task: T1 theorem statement, T2 apparatus-boundary normalization, T3 dimensionless invariant,
> T4 proof strategy + missing lemma, T5 self-refuting notebook. Harvested from claude.ai tab
> (math rendered as unicode by DOM extraction — canonical restatement in canon/00-foundations/04-break-even-theorem.md).


Synthesized thermodynamic bounds, apparatus normalization, and universality tests rigorously

No preamble. Here is the construction.

T1 — Candidate theorem

Setup. Discrete-time agent in a partially observable environment: environment state 
𝑋
𝑡
X
t
	​

, sensor 
𝑌
𝑡
Y
t
	​

, persistent internal state 
𝑀
𝑡
M
t
	​

, action 
𝐴
𝑡
A
t
	​

, temperature 
𝑇
T, 
𝛽
=
1
/
𝑘
𝑇
β=1/kT. Define over horizon 
𝐻
H (steady state, per cycle):

𝐼
𝑚
𝑒
𝑚
=
𝐼
(
𝑀
𝑡
;
𝑋
𝑡
)
I
mem
	​

=I(M
t
	​

;X
t
	​

) — total stored information about the world.
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

) — information the memory carries about the environment at the horizon 
𝜏
τ the action actually couples to.
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

) — predictive information a memoryless agent already extracts free from the instantaneous sensor.
𝐼
𝑢
𝑠
𝑒
I
use
	​

 — action-relevant mutual information actually driving the work-extracting DOF (the transfer entropy 
𝑀
 ⁣
→
 ⁣
M→work-coordinate).

The inequality. Stack Sagawa–Ueda (ceiling on work from feedback) against Still et al. (floor on cost of maintaining nonpredictive memory), differenced against the optimal memoryless twin (T2):

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
⏟
Still nostalgia floor
  
−
  
(
𝐶
𝑠
𝑒
𝑛
𝑠
𝑒
𝑒
𝑥
𝑡
𝑟
𝑎
+
𝐶
𝑐
𝑜
𝑚
𝑝
+
𝐶
𝑐
𝑡
𝑟
𝑙
𝑒
𝑥
𝑡
𝑟
𝑎
)
.
ΔW
net
	​

≤kT[I
pred
	​

−I
react
	​

]−
Still nostalgia floor
kT(I
mem
	​

−I
pred
	​

)
	​

	​

−(C
sense
extra
	​

+C
comp
	​

+C
ctrl
extra
	​

).

IF Causal-Work Break-Even Theorem (candidate). Internally-maintained information is net-work-productive iff

  
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
  
>
  
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
  
+
  
𝛽
 
𝐶
𝑜
𝑣
𝑒
𝑟
ℎ
𝑒
𝑎
𝑑
  
I
pred
	​

−I
react
	​

>(I
mem
	​

−I
pred
	​

)+βC
overhead
	​

	​


*where 
𝐶
𝑜
𝑣
𝑒
𝑟
ℎ
𝑒
𝑎
𝑑
=
𝐶
𝑠
𝑒
𝑛
𝑠
𝑒
𝑒
𝑥
𝑡
𝑟
𝑎
+
𝐶
𝑐
𝑜
𝑚
𝑝
+
𝐶
𝑐
𝑡
𝑟
𝑙
𝑒
𝑥
𝑡
𝑟
𝑎
C
overhead
	​

=C
sense
extra
	​

+C
comp
	​

+C
ctrl
extra
	​

. Equivalently: the predictive surplus over reactive must exceed the nostalgia (nonpredictive stored bits) plus the dimensionless non-memory overhead.*

**Equality (the IF 
𝑘
𝑇
ln
⁡
2
kTln2 line).** Break-even is achieved with equality when every sub-bound saturates simultaneously: 
𝐼
𝑚
𝑒
𝑚
=
𝐼
𝑝
𝑟
𝑒
𝑑
I
mem
	​

=I
pred
	​

 (zero nostalgia — the agent stores only what predicts the horizon), sensing reversible, control quasistatic. This is the ideal-agent frontier; no agent sits above it.

Why this predicts your parasite band — the important part. Your two thresholds are not an anomaly; the theorem forces them to differ. The ablation ratio 
Π
𝐴
Π
A
	​

 charges only 
𝐶
𝑚
𝑜
𝑑
𝑒
𝑙
C
model
	​

 and compares intact-vs-scrambled within one architecture, so it ignores both 
𝐼
𝑟
𝑒
𝑎
𝑐
𝑡
I
react
	​

 and 
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

:

Π
𝐴
=
1
  
⟺
  
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
𝑠
𝑐
𝑟
]
=
𝐶
𝑚
𝑜
𝑑
𝑒
𝑙
.
Π
A
	​

=1⟺kT[I
pred
	​

−I
scr
	​

]=C
model
	​

.

The competitive threshold charges the full 
Δ
𝐶
ΔC and subtracts the reactive free-harvest 
𝐼
𝑟
𝑒
𝑎
𝑐
𝑡
I
react
	​

. Since 
𝐼
𝑟
𝑒
𝑎
𝑐
𝑡
≥
0
I
react
	​

≥0 and 
𝐶
𝑜
𝑣
𝑒
𝑟
ℎ
𝑒
𝑎
𝑑
≥
0
C
overhead
	​

≥0, competitive break-even must occur at strictly higher predictability than ablation break-even. The band width is

band
  
=
  
𝛽
(
𝐶
𝑜
𝑣
𝑒
𝑟
ℎ
𝑒
𝑎
𝑑
)
  
+
  
[
𝐼
𝑟
𝑒
𝑎
𝑐
𝑡
−
nostalgia
(
𝑝
1
∗
)
]
  
>
  
0
band=β(C
overhead
	​

)+[I
react
	​

−nostalgia(p
1
∗
	​

)]>0

whenever the reactive agent extracts anything and non-memory overhead is nonzero. The parasite band is a derived structural feature of the theorem, not a bug in your sim. That is your first genuine result — write it up as such.

T2 — Canonical apparatus-boundary normalization

The knob dies by never drawing an absolute boundary. State the theorem entirely in architecture-differences against a canonically-defined twin.

Rule. The reference 
𝐴
0
A
0
	​

 is the work-maximizing memoryless policy on the identical environment and sensor — the POMDP collapsed to the MDP on current observation, unique up to ties. Then

Δ
𝐶
≡
𝐶
[
𝐴
]
−
𝐶
[
𝐴
0
]
,
Δ
𝑊
≡
𝑊
[
𝐴
]
−
𝑊
[
𝐴
0
]
.
ΔC≡C[A]−C[A
0
	​

],ΔW≡W[A]−W[A
0
	​

].

Every DOF shared by 
𝐴
A and 
𝐴
0
A
0
	​

 — the whole outer boundary, the reservoirs, the actuators — cancels identically in the difference. So the result is invariant to where you'd draw the outer wall; you only ever measure the diff, and the diff is exactly "the persistent state + the compute that reads/writes it." No free parameter.

The one guardrail that makes it non-tunable: 
𝐴
0
A
0
	​

 must be optimal, not arbitrary. Crippling the reference to inflate 
Δ
𝑊
ΔW is now a detectable violation (a suboptimal memoryless policy is a computable error). This converts "where is the boundary" — unanswerable — into "solve this MDP" — a definite computation. That is the whole trick.

T3 — Dimensionless invariant

Two thresholds because they measure two things. The invariant must use both.

Π
𝐴
=
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
𝑠
𝑐
𝑟
]
𝐶
𝑚
𝑜
𝑑
𝑒
𝑙
  
(
internal causal efficiency
)
,
Π
𝐶
=
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
Δ
𝐶
𝑓
𝑢
𝑙
𝑙
  
(
architectural efficiency
)
.
Π
A
	​

=
C
model
	​

kT[I
pred
	​

−I
scr
	​

]
	​

  (internal causal efficiency),Π
C
	​

=
ΔC
full
	​

kT[I
pred
	​

−I
react
	​

]
	​

  (architectural efficiency).

Π
𝐴
=
1
Π
A
	​

=1 at 
𝑝
1
∗
=
0.64
p
1
∗
	​

=0.64; 
Π
𝐶
=
1
Π
C
	​

=1 at 
𝑝
2
∗
=
0.995
p
2
∗
	​

=0.995. The raw 
𝑝
p-gap is huge only because 
𝑝
p is a bad coordinate — near 
𝑝
→
1
p→1 predictive information diverges, so equal 
𝑝
p-steps are wildly unequal information-steps.

Reparameterize into usable-information coordinates. Let 
𝑢
(
𝑝
)
=
Π
𝐶
(
𝑝
)
u(p)=Π
C
	​

(p). By construction 
𝑢
(
𝑝
2
∗
)
=
1
u(p
2
∗
	​

)=1. The universality claim (IF-H1) is:

  
Θ
\*
≡
Π
𝐴
∣
Π
𝐶
=
1
  
=
  
𝐶
𝑚
𝑜
𝑑
𝑒
𝑙
 
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
Δ
𝐶
𝑓
𝑢
𝑙
𝑙
 
[
𝐼
𝑝
𝑟
𝑒
𝑑
−
𝐼
𝑠
𝑐
𝑟
]
∣
𝑝
2
∗
  
Θ
\*
≡Π
A
	​

	​

Π
C
	​

=1
	​

=
ΔC
full
	​

[I
pred
	​

−I
scr
	​

]
C
model
	​

[I
pred
	​

−I
react
	​

]
	​

	​

p
2
∗
	​

	​

	​


takes the same critical value across rule families (ring, chemotaxis, Kalman). 
Θ
∗
Θ
∗
 is the fraction of the full apparatus cost concentrated in belief-maintenance, evaluated at competitive break-even — dimensionless, boundary-free (both numerator and denominator are T2-differences). From your run, plug in 
𝑝
2
∗
=
0.995
p
2
∗
	​

=0.995: 
Θ
𝑟
𝑖
𝑛
𝑔
∗
=
Π
𝐴
(
0.995
)
Θ
ring
∗
	​

=Π
A
	​

(0.995), a number 
≫
1
≫1 (memory is deep in ablation-positive territory there). **That single number is your candidate constant.** If a Kalman world reproduces it, IF has an invariant; if it scatters, IF has curve fits. Neither threshold alone is the invariant — 
Θ
∗
Θ
∗
 is their ratio in the right coordinate.

T4 — Proof-sketch strategy and the missing lemma

Assembly of existing machinery:

Sagawa–Ueda feedback FT → the ceiling 
Δ
𝑊
≤
𝑘
𝑇
 
Δ
𝐼
𝑢
𝑠
𝑒
ΔW≤kTΔI
use
	​

. Direct.
Still–Sivak–Crooks–Bialek (thermodynamics of prediction) → the maintenance floor 
𝐶
𝑚
𝑒
𝑚
≥
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
C
mem
	​

≥kT(I
mem
	​

−I
pred
	​

), the nonpredictive-info dissipation. Direct.
Barato–Seifert transducer / information-flow bounds → promotes single-shot to a steady-state rate theorem over 
𝐻
H and rigorously defines 
𝐼
˙
𝑢
𝑠
𝑒
I
˙
use
	​

 (learning rate) in the bipartite sensor→memory→actuator decomposition. Needed for the horizon.
KW viability → supplies interventional semantics for the scramble/erase operations you actually run.

The genuinely missing lemma. None of the four connects the interventional menu to the information terms tightly. You need a data-processing inequality for interventional (do-scramble) mutual information under a thermodynamically-consistent channel, giving

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
  
=
  
𝑘
𝑇
 
[
𝐼
𝑢
𝑠
𝑒
𝑖
𝑛
𝑡
𝑎
𝑐
𝑡
−
𝐼
𝑢
𝑠
𝑒
𝑠
𝑐
𝑟
]
  
+
  
𝑅
,
with 
𝑅
≤
0
 provable.
W
intact
	​

−W
scr
	​

=kT[I
use
intact
	​

−I
use
scr
	​

]+R,with R≤0 provable.

𝑅
R is off-shell work — work changes from scrambling not mediated by information loss (e.g. injected entropy coupling directly to the reservoir). Proving 
𝑅
≤
0
R≤0 (scrambling never creates usable work) is what makes ablation measure the information term. That DPI-for-interventions is not off the shelf. It is the theorem to prove.

And the harder one, for falsify. Falsification injects a self-consistent wrong model: 
𝐼
𝑚
𝑒
𝑚
>
0
I
mem
	​

>0 but the correlation points the wrong way — anti-predictive information costs work. Ordinary MI is nonnegative and cannot represent this. You need a signed usable-information functional

𝐽
  
=
  
𝑘
𝑇
 
⟨
 
(
∂
𝑀
𝐴
)
⋅
∇
𝑤
𝑜
𝑟
𝑘
 
⟩
  
=
  
𝑘
𝑇
 
C
o
v
(
belief-driven action
,
 true work gradient
)
,
J=kT⟨(∂
M
	​

A)⋅∇
work
	​

⟩=kTCov(belief-driven action, true work gradient),

which can go negative. Prove 
𝐽
J is squeezed between the SU ceiling and the Still floor, and 
𝐽
J — not 
𝐼
I — becomes IF's invariant object. That signed functional is the one genuinely new mathematical entity in the whole program. Everything else is assembly.

T5 — Minimal notebook that refutes T1

Two kills, cheapest first.

Refute the invariant (primary, ~a day). Build a structurally different second rule family — linear-Gaussian LQG/Kalman controller with an explicit metabolic debit per bit-op. Locate its 
𝑝
1
∗
,
𝑝
2
∗
p
1
∗
	​

,p
2
∗
	​

 by the identical protocol; compute 
Θ
𝐾
𝑎
𝑙
𝑚
𝑎
𝑛
∗
=
Π
𝐴
∣
Π
𝐶
=
1
Θ
Kalman
∗
	​

=Π
A
	​

∣
Π
C
	​

=1
	​

.
**Metric:** 
∣
Θ
𝐾
𝑎
𝑙
𝑚
𝑎
𝑛
∗
−
Θ
𝑟
𝑖
𝑛
𝑔
∗
∣
∣Θ
Kalman
∗
	​

−Θ
ring
∗
	​

∣ vs finite-size error bars (bootstrap over seeds).
Kills it: if 
Θ
∗
Θ
∗
 differs beyond error, IF-H1 universality is dead and T1 is at best world-specific. (Agreement obliges a third family — chemotaxis — before claiming universality; two points don't make a constant.)

Refute the lemma (deeper, the honest one). Engineer a world with an information-independent work channel from the scramble itself: the scramble must physically dump heat; couple that heat to a ratchet doing useful work. Measure

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
.
R=(W
intact
	​

−W
scr
	​

)−kTΔI
use
	​

.

Metric: sign and magnitude of 
𝑅
R across seeds.
Kills it: if 
𝑅
>
0
R>0 robustly, scrambling creates net usable work through a non-informational path, the DPI-for-interventions lemma is false, 
Π
𝐴
Π
A
	​

 over-counts, and the central identification "ablated work = information content" collapses. This is the notebook you should want to run, because it attacks the load-bearing lemma directly rather than the decorations.

One flag, stated once so it's on record: the entire construction above is agency-thermodynamics. It stands or falls on notebooks. None of it touches the cosmology branch, and nothing here lends the cosmology branch any support — keep them in separate repositories or the referee who kills T5's ratchet result will use it to dismiss the whole corpus. Build the invariant first.
