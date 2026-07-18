# IF Mobility Search — closing the Conway-gate asterisk

> Auth: 65537 · Master Equation: Purpose × Evidence × Love
> Started 2026-07-18 after `if-agency-lab-274177` sealed Phase 2 **by falsification with
> an asterisk**: IF-H1 was killed on hand-designed agents only; the one Conway-gate
> universe built (`scripts/universe.py`) grew still lifes, so the causal-work audit had
> nothing to audit. **This hackathon exists to remove that asterisk.**

## The decisive question

Does a Conway-gate universe exist — inside a parameter space declared *before any sweep
runs* — that grows **mobile structures on its own** (random soup, nothing seeded)?
And when the causal-work audit runs on those emergent movers, do they carry positive
causal work (W_C > 0)?

Either answer is a result. A universe that grows movers whose structure carries causal
work is the first honest substrate for any IF-H1-class claim. A declared space that
grows none is a publishable negative that constrains where such substrates live.

## Instrument bug found on entry (drives M0)

`detect_structures(A, min_size=6)` — but a glider has **5 cells**. The detector that
produced the "still lifes only" verdict was blind to the canonical mobile structure by
construction. M0 re-runs the original regime with `min_size=5` **and a tracker** before
any new parameter is touched. If movers were already there, the still-life kill-log
entry gets corrected the same session.

## Pre-committed protocol (FROZEN before the sweep — amendments must be logged, never silent)

### D1 — Emergent
Initial condition is uniform random soup at declared density. **No seeded patterns**
anywhere in the sweep. Seeding is permitted only in the instrument controls (C1/C2),
which are excluded from all verdicts.

### D2 — Structure & identity
A structure is an 8-connected component with ≥ 5 cells (glider-inclusive floor).
Identity over time is by maximal cell-overlap between consecutive frames (the tracker's
estimator, declared: overlap matching, wrap-aware incremental center-of-mass with
minimal-image convention per step, accumulated to an unwrapped path).

### D3 — Mobile
A tracked structure is **mobile** iff, over its tracked lifetime:
- lifetime ≥ 40 steps, AND
- net unwrapped center-of-mass displacement ≥ 8 cells, AND
- max size ≤ 60 cells (excludes growing wavefronts masquerading as movers), AND
- size never drops below 5 while tracked.

(Reference: a glider at c/4 diagonal covers ~14 cells of net displacement in 40 steps —
the threshold is deliberately below that, but above still-life jitter.)

### D4 — Mobility regime
A swept configuration is a **mobility regime** iff, at confirmation stage (8 seeds,
600 steps), the mean number of emergent mobile tracks per run is ≥ 0.5 — movers must be
reproducible across seeds, not a one-off.

### Declared parameter space (the ONLY space stage A may search)

| Axis | Values |
|---|---|
| Rule (born/survive) | B3/S23 (Life) · B36/S23 (HighLife) · B368/S238 |
| E_BIRTH | 0.25 · 1.0 |
| E_MAINT | 0.0 · 0.01 |
| inflow | 1.0 · 4.0 · 12.0 |
| hotspot σ | 14 · 40 · 10⁶ (≈uniform resource) |
| soup density | 0.08 · 0.15 |
| drift | 1 (fixed) |

216 configurations. Stage A: 1 seed × 300 steps per config (census). Stage B: top
configs by mobile-track count → 8 seeds × 600 steps (confirmation vs D4). The Conway
gate holds at every point: no rule change introduces agency terms; the Noether ledger
assertion stays enabled in every run.

### Stop rule (binding)
If no configuration in the declared space satisfies D4, the result is **"the declared
space grows no movers"** — logged in the kill log, hackathon sealed on the negative.
Expanding the space requires a logged amendment to this README *before* the expanded
sweep runs. No silent expansion, no threshold adjustment after seeing results.

### Causal-work audit verdict (frozen)
On the best mobility regime: pool ≥ 20 emergent mobile structures across ≥ 8 seeds;
fork intact vs in-bounding-box count-preserving scramble (the `emergent_audit.py`
protocol, T = 60); compute t on W_C.
- t > +2 → emergent movers carry positive causal work (the asterisk closes with a live substrate)
- t < −2 → negative causal work — logged as-is
- |t| ≤ 2 → undecided at this sample; report widths, no upgrade to a claim.

### Logged amendment #1 (2026-07-18, before the audit ran — after stages A/B, before M3)

`emergent_audit.py`'s harvest region (bounding box dilated ×6) was designed for still
lifes. A mover at glider speed (c/4) travels ~15 cells during the T=60 audit window and
would exit its own audit region, undercounting intact-fork harvest. Amendment: for the
mover audit the dilation is **21** (= 6 + 15 travel allowance), applied identically to
the intact and scrambled forks. No verdict threshold changes. The W_C measure (region
resource drawdown, intact fork T steps vs scramble-step + T−1 steps) is unchanged.

## Instrument controls (must pass BEFORE the sweep is interpreted)

- **C1 positive**: empty universe + one seeded glider (the `gliders.py` dist=0 config).
  The tracker must classify it mobile under D3. If it can't see a glider, the census is void.
- **C2 negative**: the original still-life regime (default params, seed 7, 400 steps).
  Tracker output here is *measured*, not assumed — this doubles as M0.

## Rubric (100 points)

| Track | Pts | What earns them |
|---|---:|---|
| **M0 Detector-blindness re-check** | 10 | Original regime re-audited with min_size=5 + tracker; "still lifes only" verdict confirmed or corrected; kill log updated either way |
| **M1 Declared sweep executed** | 25 | All 216 configs run, census table + evidence JSON committed; no config outside the declared space |
| **M2 Instrument validated** | 15 | C1 detects the glider as mobile; C2 measured and logged; tracker estimator documented |
| **M3 Causal-work audit on movers** | 25 | If a D4 regime exists: ≥20 movers audited, frozen verdict criteria applied. If none exists: stop rule honored, negative sealed |
| **M4 Honesty** | 15 | Pre-registration committed before sweep ran (git history is the proof); no post-hoc threshold moves; kills published same session |
| **M5 Canon + verify integrity** | 10 | SCOREBOARD, kill log, HANDOFF updated; verify.sh still GREEN |

## Persona gates (each must sign before seal)

- **Conway** — do the rules still contain no agency? Is every "emergent" claim soup-born, with seeds confined to controls?
- **Feynman** — is each mover a genuine translating structure, not a drifting wavefront or a hotspot-chasing rebirth artifact? (D3's size cap + the controls are his gate.)
- **Noether** — does the energy ledger hold to 1e-6 in every swept rule variant? (Assertion never disabled.)
- **Shannon** — is the tracker a declared estimator (overlap identity, wrap-aware COM), not an eyeball?
- **Popper** — were space, thresholds, and verdict criteria frozen before results? Is the stop rule honored if the space is barren?
- **Phuc Forecast / 65537** — does the outcome serve the bridge honestly, whichever way nature answers?

## Seal condition

Either **(a)** a mobility regime exists, ≥ 20 emergent movers are audited, and the
frozen verdict is applied — the Conway-gate asterisk is closed with a live substrate;
or **(b)** the declared space grows no movers and the negative is sealed in the kill
log. **Both outcomes seal the hackathon.** The rung is about whether the universe was
asked honestly, not about which answer it gave.
