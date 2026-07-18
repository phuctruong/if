# Quantity Audit — Rung 641

> Layer: SCIENCE (meta). Phase-1 seal condition: *every defined quantity has dimensions,
> an estimator, and a falsifier — no metaphors wearing math.* Audited 2026-07-18 against
> `01-constitution.md` … `04-break-even-theorem.md` + P15.

## The audit table

| Quantity | Dimensions | Estimator | Falsifier | Verdict |
|---|---|---|---|---|
| **B** (informational battery) | J (capacity) + bits (correlations), **never summed** | free-energy accounting + declared MI on structured correlations | Maxwell-demon ledger violates 2nd law under ablation; or no stable coarse-graining window | ✅ PASS |
| **Energy ledger** | J | simulation-explicit; property-tested (`assert_energy_conserved`) | drift > ε in any run → run fails | ✅ PASS |
| **Thermodynamic-entropy ledger** | J/K | macrostate partition, stated coarse-graining | `assert_second_law` fires | ✅ PASS |
| **Information ledger** | bits | declared per use (see I_use caveat) | `assert_landauer_debit` fires | ⚠️ CONDITIONAL — see §2 |
| **Nostalgia** (I_mem − I_pred) | bits | difference of two MI estimates | ≤ 0 across all regimes → the self-deception term is empty | ⚠️ inherits I_use problem |
| **W_C** (causal work) | J | W_intact − W_scrambled − C_model, all measured directly | scrambling produces no work change | ✅ PASS |
| **Π_A** (ablation ratio) | dimensionless | ΔW_ablation / C_model — **both declared, neither inferred** | Π_A ≡ 1 everywhere → no threshold structure | ✅ PASS |
| **Π_C** (competitive ratio) | dimensionless | ΔW_vs_twin / ΔC_full, twin = optimal memoryless policy | no crossing exists in any family | ✅ PASS |
| **Parasite band** | dimensionless region | {Π_A > 1 ∧ Π_C < 1}, located by interpolation | the two thresholds coincide in every family | ✅ PASS (observed 3×) |
| **C_model, ΔC_full** | J | declared metabolic debit (design parameter) | — (declared, not inferred) | ✅ PASS |
| **A₀** (canonical twin) | policy | work-maximizing memoryless policy = solve the collapsed MDP | a suboptimal reference is computably detectable | ✅ PASS |
| **I_use** | bits | **THREE ESTIMATORS TRIED, ALL FAILED** (P15 §3) | — | ⛔ **FAILS THE GATE** |
| **η\*** | dimensionless | built on I_use | — | ⛔ **DEAD** (inherits I_use) |
| **Θ\***, **Υ_IF** | dimensionless | Π_A at Π_C = 1, ± rescaling | cross-family scatter | ⛔ **FALSIFIED** as invariants; survive as per-family descriptors |
| **J** (signed usable-information) | J | **not yet constructed** | — | 🔓 OPEN — conjectured; P15 §3 raised its necessity |
| **b(z)** (cosmic IF state) | model-dependent | **none — ℒ_IF is a free function** | — | ⛔ **FAILS** (see `../20-cosmology/03-testability-audit-2026-07-18.md`) |
| **μ_IF, η_slip, w_IF, a_IF** | dimensionless / dimensionless / dimensionless / m·s⁻² | limits specified, forms unspecified | — | ⛔ **FAIL** — symbols awaiting a theory |
| **A_future** (retained action space) | dimensionless (log-count) | Σᵢ log\|viable actions\| — counts what a system *can do*, never what it encodes | wins everywhere or nowhere; or dominated by implementation choices | ✅ PASS (and structurally immune to the P15 obstruction) |
| **Π_A^W (agency threshold)** | dimensionless | per-family only | — | ✅ PASS at per-family scope; ⛔ universality retired |

## §2 The one conditional and the two failures

**The information ledger is conditionally sound.** Where information enters as a *declared*
design quantity (bits erased, memory register size, Landauer debits), it is dimensionally
clean and property-testable. Where it enters as an *inferred* quantity about what an agent
"uses," P15 showed it is representation-relative and not portable. **The ledger passes for
accounting and fails for attribution.** Every future use must state which mode it is in.

**b(z) and the cosmology symbols fail outright** — not because they are wrong but because
they are unspecified. A free function is not a quantity.

## Rung 641 verdict: **SEALED, with two declared exclusions**

The gate asked whether every quantity has dimensions, an estimator, and a falsifier. The
audit's honest answer:

- **The agency-branch foundations pass** (B, ledgers, W_C, Π_A, Π_C, A₀, parasite band,
  A_future). These are the quantities the program actually computes with.
- **I_use / η\* / Θ\* / Υ_IF fail or are falsified** — and are now *recorded as such* in
  canon rather than quietly used. A quantity that failed the gate and is marked failed is
  not a violation of the gate; an unmarked one is.
- **The cosmology symbols fail** and the branch is labeled "specified, not yet
  implemented, not yet tested."

**Rung 641 is sealed on the condition that these exclusions remain visible.** The seal
certifies the *audit*, not universal passage — and no document may use a ⛔ quantity as
though it had passed. Enforcement: `scripts/verify.sh` gate 6 (added 2026-07-18) greps
canon for uses of `η*`, `Θ*`, `Υ_IF`, and `b(z)` outside a falsification/limitation
context.
