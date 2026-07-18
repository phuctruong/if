# IF-Theory — Stillwater Ripple
# Generated: 2026-02-21 | stillwater v1.5.0
# This file overrides base Stillwater behavior for this project.
# Keep it under 50 lines. Everything else goes in README.md.

PROJECT: IF-Theory
DOMAIN: physics simulation / information theory
RUNG_TARGET: 274177
NORTHSTAR: Phuc_Forecast
ECOSYSTEM: PUBLIC
LANGUAGE: Python

KEY_CONSTRAINTS:
  - never-worse on standard test suite
  - IF Theory: information as the first force; energy/matter are derived, not primary
  - Exact arithmetic in all physics calculations (Fraction/Decimal, no float)
  - Deterministic: same seed + parameters must produce identical simulation
  - Proof-grade claims require rung 274177 (seed sweep + replay stability)

ENTRY_POINTS:
  - src/if_theory/  (physics simulation engine)
  - pytest -q tests/

FORBIDDEN_IN_THIS_PROJECT:
  - Float arithmetic in any verification path
  - Non-deterministic simulation (seeds must be reproducible)
  - Physics claims without rung-gated evidence artifacts
  - Violating IF Theory first-principles axioms

SEE_ALSO: README.md  # IF Theory axioms, simulation architecture, research roadmap