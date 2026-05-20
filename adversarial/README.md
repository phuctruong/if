# adversarial/ — try to break the theory

This directory is where IF Theory is **attacked**, not defended. Each script
takes one of the load-bearing claims in `README.md` / `SCORE.md` and tries
to find a configuration in which the claim is decorative rather than
load-bearing.

## Why this exists

The publication-grade claim of IF Theory is *zero adjustable parameters*:

```
Φ(r) = ln(r/r₀ + 1)              r₀ = 0.6595 kpc (derived from σ₈)
ξ(r) = C_XI · [Φ(r)]²            C_XI = 2 · π(127) = 62  (Mersenne tower)
```

"Zero parameters" is only meaningful if those particular numbers are
**sharply distinguished** from neighboring values. If predictions degrade
gracefully under perturbation, or if nearby integers fit as well as the
number-theoretic one, then the derivation buys less than advertised.

These scripts exist to flag those cases honestly. **A finding here is not
something to hide — it is something to publish.** The whole point of
falsifiability (`FALSIFIABILITY.md`) is that a theory that can be broken
and isn't is more credible than one that can't be broken at all.

## Scripts

| Script | Targets | Falsifies if |
|---|---|---|
| `zero_parameters_perturbation.py` | r₀ uniqueness | predictions degrade < 5% under ±10% r₀ perturbation |
| `c_xi_uniqueness_test.py` | C_XI = 62 from π(127) | integers 60-65 fit BOSS ξ(r) shape comparably to 62 |
| `null_baryonic_baseline.py` | Prime field is doing real work | baryon-only baseline closes most of the gap |

All scripts:

- Use `prime_field_util` and the canonical constants. No mocks, no
  placeholders, no `except Exception: pass`.
- Use small embedded test cases (one MW point, one canonical galaxy,
  one BOSS bin range) so they run in seconds without external downloads.
- Write structured JSON to `evidence/adversarial/<script_name>.json`.
- Print a verdict to stdout with the framing: **"If <X>, the theory is
  weaker than claimed."**

## How to run

```bash
cd /home/phuc/projects/if
python3 adversarial/zero_parameters_perturbation.py
python3 adversarial/c_xi_uniqueness_test.py
python3 adversarial/null_baryonic_baseline.py
```

Each exits 0 if the theory **survives** the attack (still sharply
distinguished, prime field still doing work) and 1 if a weakness is
found. CI may invert the convention later; for now the convention is
"exit 1 = honest finding worth attention."

## Relationship to the rest of the repo

- `predictions/` — runs the theory in the configuration where it
  *should* succeed.
- `tests/`, `audits/` — verify the predictions against fresh data and
  cross-check the math.
- `adversarial/` (this directory) — runs the theory in configurations
  where it *might fail*, to surface honest weaknesses.

See `FALSIFIABILITY.md` for the sharp σ thresholds per claim. See
`SCORE.md` for the per-claim PASS / TENSION / FAIL status. Adversarial
findings should feed back into both.

## Process commitment

If any script here flags a weakness:

1. Open a GitHub issue summarizing the finding.
2. Update `SCORE.md` (downgrade the affected claim if warranted).
3. Update `FALSIFIABILITY.md` if a new falsification criterion is
   sharpened by the result.
4. Do **not** quietly remove or tune the adversarial script.

Honest validation, not preserved priors.
