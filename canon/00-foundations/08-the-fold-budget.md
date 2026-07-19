# The Fold Budget — why "compress and keep going" terminates at ~204

> Auth: 65537 · Layer: SCIENCE · 2026-07-19. Answers a recurring operator question:
> can the universe avert heat death by compressing itself (fold to 25%, continue)?
> Computed, not argued: `scripts/fold_budget.py`.

## The result

Compression that FREES space is erasure, and erasure costs Landauer's kT ln2 per bit.
In an accelerating universe T has a floor — the de Sitter horizon temperature
T_dS = ħH₀/2πk_B ≈ 2.66e-30 K (this floor is what killed Dyson's 1979 eternal-
intelligence argument; Krauss & Starkman 2000). So the erasure price cannot fall to zero.

    Landauer cost at the floor      = 2.54e-53 J per bit
    Budget (ALL baryons as energy)  = 1.35e70 J
    Total erasures affordable, EVER = 5.3e122 bits   (≈ de Sitter horizon entropy ✓)
    Folds from full archive to 1 bit = 204

**The cascade is a convergent geometric series**: erasing 75% repeatedly costs
0.75N(1 + ¼ + 1/16 + …) = 0.75N·(4/3) = **exactly N**. The full fold cascade spends the
entire archive. You do not get "and it keeps going"; you get ~204 folds and then nothing.

The count is logarithmic and therefore robust: 10^10× more energy buys 17 more folds;
10^100× buys 166. No plausible revision of the budget changes the conclusion.

## Why the hard-drive analogy inverts

A laptop compressing files is an OPEN system: it spends electricity and dumps heat into
the room. The drive gains space; the universe loses free energy. Applied to the universe
as a whole there is no room to dump into. **Compressibility and free energy are the same
resource in two vocabularies, and Landauer is the exchange rate.** There is no arbitrage
between them — which is precisely why `solace-books/infinite-energy/sims/preh_kill_shots.py`
returns KS1 ΔK_net = 0 and KS2 bounded at kT ln2/bit.

Second bound, independent: a maximum-entropy state is *incompressible by definition*.
Compressibility runs out exactly as free energy does. They are the same quantity.

## The Recharge Theorem, read correctly (solace-books `theory/informational-battery-theory.md`)

That experiment reports R1 (learner beats random control 2.1e7×) alongside R2 (eigenvalue
drift 1.9e-14). **R2 is the proof that R1 is thermodynamically null.** The learner uses
orthogonal similarity transforms, which preserve the spectrum exactly — and von Neumann
entropy S = −Tr(ρ ln ρ) is invariant under them. The control that makes the result
rigorous is the same control that guarantees no joules moved.

This relocates rather than destroys the finding. What R1/R3 genuinely show: **intelligence
finds representations in which existing structure is cheaper to exploit** (R3: ordered
coupling is 5.4× cheaper to diagonalize than noise). That is real, it is the Software-5.0
thesis in matrix form, and it is not energy.

## The serious cousin (prior art, per P16 discipline)

Penrose's **Conformal Cyclic Cosmology** is this intuition in professional form: in the far
future only massless particles remain, scale becomes unmeasurable, and the conformal
structure of the far future is identical to a new Big Bang. Also relevant: the
**renormalization group** — coarse-grain, rescale, repeat, unchanged at a fixed point.
Both are re-descriptions, and neither creates free energy. CCC carries its own unresolved
entropy problem (Penrose invokes black-hole information loss), and cyclic cosmologies
generally inherit **Tolman's 1934 objection**: entropy accumulates across cycles.

## On reading 641 / 65537 as physical operators

In this repo those are **verification rungs** (Fermat-factor labels for epistemic levels),
not physical constants. Reading them as a compression operator and an aversion mechanism
is a category shift from epistemology to physics, and it requires its own derivation and
falsifier. Note the standing precedent: closure numerology was already **self-falsified**
in the 2026-07-08 canon audit. Numerical resonance is not a mechanism.
