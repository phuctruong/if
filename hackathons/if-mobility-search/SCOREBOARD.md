# if-mobility-search — Scoreboard

> Auth: 65537 · Rubric in README.md. SEALED 2026-07-18 under seal condition (a).

| Track | Pts available | Earned | Evidence |
|---|---:|---:|---|
| M0 Detector-blindness re-check | 10 | 10 | min_size=6 was glider-blind, BUT verdict **confirmed** at min_size=5: 0 movers, 8 seeds (`evidence/mobility_controls_2026_07_18.json`) |
| M1 Declared sweep executed | 25 | 25 | 216/216 configs, census committed (`evidence/mobility_sweep_stageA_2026_07_18.json`) |
| M2 Instrument validated | 15 | 15 | C1: seeded glider tracked, life=150, disp=53.0, size=[5,5], mobile=True; C2 measured |
| M3 Causal-work audit on movers | 25 | 25 | D4 regime found (mean 24.0 movers/run, 8 seeds); 21 movers audited on the declared 24-seed roster; frozen verdict applied: **UNDECIDED** (t=−0.94) — no claim upgrade (`evidence/mover_audit_2026_07_18.json`) |
| M4 Honesty | 15 | 15 | Prereg committed `aed7149` before sweep; amendment #1 (dilation 21) logged+committed before audit; run-1 VOID honored (9 < 20); fixed 24-seed roster declared before rerun, verdict on full roster |
| M5 Canon + verify integrity | 10 | 10 | SCOREBOARD + kill log + HANDOFF updated; verify.sh GREEN |
| **TOTAL** | **100** | **100** | |

## Headline results

1. **The still-life mystery is solved: energy starvation, not rules.** Every top
   mobility config has E_BIRTH=0.25 (vs 1.0 original) and high inflow. Same Conway-gate
   rules, cheaper construction → movers grow from random soup. Best regime:
   `B3/S23, e_birth=0.25, e_maint=0.01, inflow=12, σ=40, ρ=0.15` → **mean 24.0 emergent
   mobile tracks per run** (D4 threshold was 0.5), all 8 seeds ≥ 17.
2. **First causal-work audit on agents a universe produced** (the Conway-gate asterisk's
   demand): 21 movers, W_C mean −4.88 ± 5.18 SEM, 61.9% positive, **t = −0.94 →
   UNDECIDED** by the frozen criteria.
3. **The heavy negative tail is a finding, not noise**: a count-preserving scramble of a
   mover occasionally ignites a debris growth explosion that out-harvests the organized
   structure (W_C −39, −93). In an energy-rich regime, organization is energy-*frugal*,
   not energy-greedy — raw harvest may be the wrong observable for mover-class agency.
   Candidate next observable: harvest per unit mass, or survival-conditioned harvest.

## Persona gate signatures

- **Conway** ✅ rules agency-free in all 3 variants; seeds confined to C1/C2.
- **Feynman** ✅ movers pass size-cap + displacement gates; glider control seen at exactly size 5; the negative tail traced to a physical mechanism (scramble-ignited growth), not instrument error.
- **Noether** ✅ ledger assertion enabled in every one of 216+ runs; zero leaks.
- **Shannon** ✅ tracker is a declared estimator: overlap identity, wrap-aware COM, unwrapped path.
- **Popper** ✅ space/thresholds/verdict frozen before results; VOID honored; UNDECIDED not upgraded.
- **Phuc Forecast / 65537** ✅ the universe was asked honestly; it answered "not yet decidable" and that is what the canon says.

## Log

- 2026-07-18 — Hackathon opened. Protocol frozen in README before any sweep ran (`aed7149`).
- 2026-07-18 — C1 PASS, M0 still-life verdict CONFIRMED at min_size=5.
- 2026-07-18 — Stage A: 216/216; mobility regimes exist; E_BIRTH=0.25 dominates.
- 2026-07-18 — Stage B: D4 PASS, mean 24.0 movers/run.
- 2026-07-18 — Amendment #1 logged+committed (audit dilation 21 for movers).
- 2026-07-18 — Audit run 1: 9 movers < 20 → VOID honored. 24-seed roster declared before rerun.
- 2026-07-18 — Audit run 2: 21 movers, t=−0.94 → UNDECIDED. Sealed under condition (a).
