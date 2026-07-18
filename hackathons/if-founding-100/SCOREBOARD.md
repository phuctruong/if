# IF Founding 100 — SCOREBOARD

> Live. Scored against `README.md` rubric. Honesty gate: Feynman.

## Current score

| Track | Score | Evidence |
|---|---:|---|
| T1 Canon coherence | 14/20 | Core idea + goal propagated; consensus scores in SCOREBOARD; gaps: stale checkboxes/statuses in ROADMAP+INDEX, CLAUDE.md layout missing panels/+hackathons/, notebooks README missing 04e/f/g detail |
| T2 Evidence engine | 30/30 | 00 ✅ · 04 ✅ · 04f RUN + cost-control verdict (SCATTER — kill logged, refined candidates named) ✅ · 04g RUN: R>0 signature 11.6σ ✅ · 04e v3 RUN: rule/state dissociation 6σ + parasite replication ✅ |
| T3 Paper corpus | 12/20 | Papers 0–6 extracted (7/15 ≈ 4/8 pts) · P00–P02 canonical ✅ (6) · P03–P06 not revised (0/6); harvest running |
| T4 Adversarial review | 8/10 | Rounds 1+2 filed (Claude/Gemini/ChatGPT-arc) · adjudications in panel doc; gap: round-3 verification pass on theorem doc |
| T5 Falsifiability | 10/10 | Kill log current · contracts frozen in all 5 notebooks (verify-enforced) · cost-invariance control run · `scripts/verify.sh` GREEN |
| T6 Bridge integrity | 10/10 | Goal doc + ladder + firewall in place · layer-leak audit RUN + CLEAN, wired into verify.sh gate 3 |
| **TOTAL** | **84/100** | 2026-07-18 07:45, iteration 2b |

## Gaps (ranked by value)

1. P03–P06 canonical revisions (6 pts, T3) — agent running.
2. Remaining papers 8–14 (harvest running; +4 pts when complete).
3. T1 stale-status sweep + notebooks README statuses (up to 6 pts).
4. Round-3 frontier verification of theorem doc (T4, 2 pts).
5. Iteration 3 science: cost-rescaled invariant test (Θ*·C_model + work-per-bit across families/costs).

## Iteration log

| # | When | What | Δ | Evidence |
|---:|---|---|---:|---|
| 0 | 2026-07-18 07:05 | Baseline after reboot day: canon rebuilt, panel sealed, theorem doc, notebook 04 result | 60 | this commit |
| 1 | 2026-07-18 07:15 | Θ\* second-family test provisional PASS (1.48σ) — later superseded by cost control | 70 | `notebooks/04f_kalman_theta_star.ipynb` |
| 2b | 2026-07-18 07:45 | verify.sh GREEN (5 gates: notebooks execute, no artifacts, layer-leak clean, contracts enforced, kill log current) | 84 | `scripts/verify.sh` |
| 2 | 2026-07-18 07:40 | **04g RUN** (R>0 signature 11.6σ; clean work-per-bit stable ~0.10) · **04e v3 RUN** (rule/state dissociation 6σ; state-smoother found to be a parasite — 3rd parasite replication) · **Θ\* cost control: KILL** (raw Θ\* not universal; lockstep survival → refined candidates) · kill log +3 | 82 | notebooks 04e/04f/04g, `scripts/{ratchet,update_law3,cost_control}.py` |
