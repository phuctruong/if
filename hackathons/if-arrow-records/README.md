# IF Arrow Records — TA-H3 / TA-H11 in a reversible CA

> Auth: 65537 · Started 2026-07-18 (loop iteration 14). Tests two preregistered P12
> hypotheses in the declared family "Critters-class reversible block CA." Per P12's
> editing pass these are PER-FAMILY measurement claims — no universality is claimed or
> claimable here.

## Frozen operationalization (before any run)

- **Substrate:** 128² torus, Margolus block CA with the Critters-style lookup table
  (per-block: c=2 → unchanged; c=3 → complement+rotate-180; else complement).
  Bijectivity of the 16-state block map is ASSERTED in code; the inverse rule is the
  inverted lookup applied with partitions in reverse order. **Instrument gate G1:**
  forward T then inverse T must retrace the initial state bit-perfectly.
- **Record functional (declared):** on the 2-step composed states, R(t) = count of 2×2
  blocks that are non-trivial (not all-dead, not all-alive) and unchanged for ≥ K = 16
  composed steps. **Gate G2:** a hand-built static pattern must register; a scrambled
  frame must register ≈ 0.
- **Coarse entropy (declared):** Shannon entropy of cell counts over 16×16 coarse cells.
- **Run plan (seeds 3, 5, 7 — declared):**
  - **E1 (low-entropy IC):** dense 24×24 blob, T = 1500 composed steps. TA-H3 clause A:
    Spearman ρ(R, t) ≥ +0.9 forward.
  - **E2 (Loschmidt probe):** at t = 750 flip 8 declared cells, then run the INVERSE
    rule 750 steps. TA-H3 clause B: retrace fails (states diverge) AND records
    accumulate in the reverse-time direction too, ρ ≥ +0.5 — the record arrow follows
    entropy production, not the time coordinate.
  - **E3 (generic IC):** density-0.5 random soup, T = 1500. TA-H11 clause: |ρ(R, t)| <
    0.5 and no secular drift beyond ±20% of the initial mean — no arrow from generic
    conditions.
- **Verdicts (frozen):** TA-H3 SUPPORTED (this family) iff E1 and E2 clauses pass on
  ≥ 2 of 3 seeds; TA-H11 SUPPORTED iff E3 clause passes on ≥ 2 of 3 seeds. Any failure
  logged as-is in the kill log (a per-family falsifier firing is a result, not a
  disaster). Gates G1/G2 failing → VOID, fix instrument, rerun, log.

## Rubric (100): prereg 20 · gates 20 · E1–E3 run 30 · verdicts honored 20 · canon+verify 10.
Personas: Noether (exact reversibility is the ledger here) · Feynman (the Loschmidt
probe is the self-fooling guard) · Shannon (R and S are declared estimators) · Popper ·
Conway · 65537.
