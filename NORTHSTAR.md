# NORTHSTAR: Phuc_Forecast — IF-Theory

> "Formalize Information Force Theory into a testable, reproducible physics simulation framework."

## Mission

IF-Theory is the **physics simulation + information theory** project — formalizing
the mathematical foundations of Information Force Theory (IF Theory) through exact
proofs, reproducible simulations, and testable predictions.

## North Star Metric

**Proof Completeness**: # of IF Theory theorems with verified, machine-checkable proofs
at rung 274177 (stability + replay-stable + null edge sweep).

Secondary metrics:
- Simulation reproducibility (byte-identical across seeds and platforms)
- Theorem coverage (% of IF Theory axioms with corresponding proofs)
- Zero CONVERGENCE_CLAIM_WITHOUT_R_P_CERTIFICATE violations

## Model Strategy

| Model | Role | When |
|-------|------|------|
| **haiku** | Main session coordinator, proof inventory | Always-on |
| **sonnet** | Coder (simulation code), Planner (proof strategy) | Implementation |
| **opus** | Mathematician (formal proofs, convergence) | Proof work (primary) |

## Rung Target: 274177

Physics proofs require stability — rung 274177:
- Seed sweep (min 3 seeds, deterministic simulation results)
- Replay stability (proofs verify identically on replay)
- Null edge sweep (zero mass, zero energy, boundary conditions)

## What Aligns with This Northstar

- Formal proofs with machine-checkable structure
- Exact arithmetic (Fraction/Decimal) in all simulation paths
- Convergence certificates for iterative physics methods
- Reproducible simulation scripts (pinned seed + initial conditions)

## What Does NOT Align

- "It looks physically reasonable" claims without formal proof
- Float in any verification or proof path
- Convergence claims without halting certificate

## See Also

- `CLAUDE.md` — prime-math + prime-coder loaded
- `ripples/project.md` — IF Theory constraints
- `skills/prime-math.md` — exact arithmetic for proofs
