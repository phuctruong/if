# Testability Audit of the Cosmology Branch

> Layer: SCIENCE (meta). Phase-3 track C3, 2026-07-18. **Verdict: not yet testable at
> galaxy scale.** This document exists because admitting that is worth more than a fit.

## Question

Does the IF Unified Geometry Hypothesis, as it currently stands in canon, specify an
**implementable** galaxy-scale law — a computable g_obs(ρ_baryon) that could be run
against SPARC tomorrow?

## Finding: no. The action is a target class, not a theory.

P07 §5.2 gives the covariant target class

\[
S=\int d^4x\sqrt{-g}\left[\frac{M_{\mathrm{Pl}}^2}{2}R+M^4\,\mathcal L_{\mathrm{IF}}(X,\mathcal A,\mathcal K_1,\mathcal K_2)\right]+S_b[g_{\mu\nu},\Psi_b]
\]

and then states plainly, in its own words: **"This action is a broad target class, not a
unique theory."** The IF Lagrangian \(\mathcal L_{\mathrm{IF}}\) is a *free function*.
Everything downstream inherits that freedom:

- \(\mu_{\mathrm{IF}}(x,b)\) — the effective-gravity interpolation — has its **limits**
  specified (→1 in the high-acceleration regime, →x in the deep regime) but not its form.
- \(a_{\mathrm{IF}}\) is described as "an acceleration scale derived from that state," but
  no derivation from \(\mathcal L_{\mathrm{IF}}\) is carried out; it functions as a symbol
  awaiting a theory, not a number the theory produces.
- The b(z) prediction lattice (`02-prediction-lattice.md`) is a *specification of what a
  theory would have to link*, not a set of computed predictions.

**Consequence:** there is nothing to fit. Any galaxy-scale "IF fit" run today would first
require choosing \(\mathcal L_{\mathrm{IF}}\) — and that choice, made after seeing the
data, is `RETROFIT_FORECAST` by construction. The Noether gate would also be unenforceable:
with a free function, μ, η, w_IF and a_IF *can* be tuned independently, which is precisely
the failure mode the gate exists to catch.

P08 independently reaches a compatible conclusion about its own strongest-looking
evidence, listing why the high-redshift acceleration-scale result "is not yet a successful
IF prediction" — beginning with *the IF formula was not preregistered before those data*.

## The inherited deficit (the honest prior)

The **previous** generation's IF galaxy law — the log-potential Φ = ln(r/r₀+1) — *was*
implementable, and was tested at full scale under fair rules
(`_archive/evidence/sparc_fair_benchmark/`, n=175, one fitted M/L per galaxy for both IF
and MOND, NFW allowed M/L+V200+concentration):

| Model | median χ²/dof | median BIC |
|---|---:|---:|
| IF (archived log-potential) | **7.13** | **85.9** |
| MOND | 3.71 | 50.9 |
| NFW halo | 1.14 | 19.8 |

It lost to both competitors, and lost on BIC too — so it was not a parameter-count story.
That law is dead (kill log, 2026-07-18). The current hypothesis is *not* that law, but it
also is not yet anything a benchmark can score.

## What this audit produces instead of a fit: a preregistered bar

Per the standing panel instruction — *state the one observable it must reproduce, state
the number that kills it, and run it* — the honest deliverable now is the **bar**, frozen
before any \(\mathcal L_{\mathrm{IF}}\) is chosen:

**PREREGISTERED GALAXY-SCALE ADMISSION CRITERIA (frozen 2026-07-18, before any new fit)**

Any future IF galaxy law, to be reported at all, must be evaluated on the full 175-galaxy
SPARC sample under the archived fairness rules (one fitted stellar M/L per galaxy; all IF
shape parameters global, not per-galaxy) and must state, in this order:

1. **Interesting-at-all threshold:** median χ²/dof ≤ **3.71** — i.e., at least matching
   MOND at equal per-galaxy freedom. Above this, the law is reported as a failure.
2. **Outright-win threshold:** median χ²/dof ≤ **1.14** *and* median BIC ≤ **19.8** —
   i.e., beating an NFW halo that is allowed strictly more freedom.
3. **Held-out requirement:** thresholds must be met on a 30% held-out galaxy set with all
   global parameters fixed on the other 70%.
4. **No-escape clause:** if the law requires any per-galaxy IF parameter beyond stellar
   M/L, it is reported as having failed the unification claim regardless of χ².

These numbers come from an *existing, already-run, adversarially fair* benchmark, which is
what makes them a legitimate preregistration rather than a moving target.

## Status assignment

| Claim | Status |
|---|---|
| IF unified geometry as a *research target class* | Alive, unimplemented |
| IF unified geometry as a *testable galaxy law* | **Not yet testable — no computable g_obs** |
| Archived log-potential IF galaxy law | **Falsified** (SPARC, n=175) |
| b(z) prediction lattice | Specification only; no computed predictions exist |
| Expansion–growth consistency test (notebook 07) | Blocked on the same free function |

## What would change this

A specific choice of \(\mathcal L_{\mathrm{IF}}\), committed in a timestamped commit
*before* any SPARC evaluation, yielding a closed-form or numerically computable
μ_IF(x,b) and a derived a_IF. Until such a commit exists, the correct public statement
about the cosmology branch is: **"specified, not yet implemented, not yet tested"** — and
the Sagan gate forbids any stronger phrasing in any abstract, talk, or book chapter.
