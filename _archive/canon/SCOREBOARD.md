# IF Theory Scoreboard
<!-- Auth: 65537 | Tenant: if | Law: LAI-13 Never-Worse | Date: 2026-05-19 -->

```
DNA: ratchet = prediction_pass × replications × theorem_closure × falsifiers_tried
```

## Day-0 baseline (2026-05-19)

This is the reset state at canon parity. Floors only move up (LAI-13).

| Dimension | Floor (Day-0) | Target (90 days) | Target (12 months) |
|---|---|---|---|
| Prediction-pass rate on geo canon corpus | 90/90 (internal eval, not peer-reviewed) | 90/90 sustained | 90/90 + ≥ 3 external reproductions |
| Independent reproductions of σ₈ → r₀ inversion | 0 | 1 | 3 |
| Mersenne uniqueness lemma closure | partial (p ≤ 19 exact, larger p approx) | exact for p ≤ 10^6 | full Schoenfeld bound for all 52 known M_p |
| Machine-checkable proof rung | none | rung 65537 on Mersenne harness | rung 274177 on full theorem |
| arXiv preprints published | 0 | 0 (drafting) | 1 (Mersenne) + 1 (σ₈ inversion) |
| Citation graph entries (cosmology/quantum literature) | UNKNOWN | UNKNOWN | UNKNOWN |
| GitHub stars (phuctruong/if) | UNKNOWN — operator to record | +30 | +200 |
| Falsifier registry entries | 0 | 1 | 5 |
| phuc.net cross-published articles | UNKNOWN — operator to verify | 2 | 5 |
| solaceagi.com pages footer-linking IF Theory | 0 | 1 (AGI framing page) | all (footer-wide) |
| BAO 2.1σ tension disclosed in every citation | YES (load-bearing discipline) | YES | YES |
| `POSTDICTION_NOTICE` flags in `predictions/` | 3 (hubble_tension, s8_tension, cmb_cold_spot) | 3 | 3 (never removed silently) |

## Ratchet floors (LAI-13: monotonic up — never demote silently)

The following four counts are the *Never-Worse* ratchet. Once they tick up,
they do not move back down without an audit event written to `evidence/`.

| Ratchet | Floor at Day-0 | Notes |
|---|---|---|
| `prediction_pass_count` (geo canon corpus passes) | 90 | If a prediction regresses (e.g., new data refutes), audit row required; otherwise floor only moves up as new predictions ship + pass. |
| `replication_count` (external runs of σ₈ → r₀) | 0 | Append-only. Each external researcher's reproduction logs a row in `evidence/replications/`. |
| `theorem_closure_count` (lemmas closed in Mersenne tower) | 1 (P ≤ 19 exact) | Floor moves up as larger-p closures land. Schoenfeld bound = +1. Full theorem = +1. |
| `falsifier_attempt_count` (external attempts to kill IF Theory) | 0 | Append-only. Counts attempts whether they succeed or fail. Failure-to-falsify is a win for IF Theory; success is the science working. |

## Floor commitments (LAI-13)

- **BAO 2.1σ tension disclosure is non-negotiable.** Every citation of the
  DESI DR1 BAO global fit must include χ²/dof = 1.72, p = 0.034, 2.1σ. Any
  citation that hides this is a blocking regression.
- **`POSTDICTION_NOTICE` flags are non-negotiable.** `hubble_tension.py`,
  `s8_tension.py`, and `cmb_cold_spot.py` are postdictions (shape, not
  amplitude). Removing the notice silently is a blocking regression.
- **C_XI = 62 cannot move** without re-deriving it from a closed Mersenne
  tower uniqueness lemma. If the lemma fails for some p, IF Theory's
  normalization story collapses; that is a hard kill, not a silent demote.
- **Float forbidden in proof paths.** Fraction / Decimal exact only.
- **Convergence claims require halting certificates** at rung 274177.

## Day-0 template scoring (fills in as work ticks)

```
[ ] prediction_pass_count           = 90 / 90
[ ] replication_count               = 0
[ ] theorem_closure_count           = 1 / 52
[ ] falsifier_attempt_count         = 0
[ ] arxiv_preprints                 = 0 / 2
[ ] github_stars                    = UNKNOWN (operator records)
[ ] phucnet_articles                = UNKNOWN
[ ] solaceagi_footer_pages          = 0 / all
[ ] bao_tension_disclosure_audit    = PASS (this session)
[ ] postdiction_notices_present     = 3 / 3
[ ] never_worse_ratchet_armed       = YES (floor = 90,0,1,0)
```

## Verification commands (when scoring updates)

```bash
# count POSTDICTION_NOTICE flags (must be 3)
grep -r "POSTDICTION_NOTICE" ~/projects/if/predictions/ | wc -l

# Mersenne tower theorem live status
python ~/projects/if/mersenne_tower_theorem.py | grep -E "VERIFIED|OPEN|PARTIAL"

# σ₈ → r₀ inversion live audit
python ~/projects/if/audits/validate_from_first_principles.py

# geo canon prediction-pass count
grep -l "PREDICTION_PASS" ~/projects/geo/canon/papers/*.md | wc -l

# external replications registry
ls ~/projects/if/evidence/replications/ 2>/dev/null | wc -l
```

## Honest caveat (per Phuc's no-academic-deference memory)

The 90/90 prediction-pass on the geo canon is **internal evaluation by the
same operator who built the framework**. It is not external peer review.
This is cited as a load-bearing internal signal, not as proof. The
ratchet's job is to make this internal scoring auditable; the falsifier
registry's job is to bring external numbers in. Until at least 3 external
reproductions land, the 90/90 number travels with the caveat.
