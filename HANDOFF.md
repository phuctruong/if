# Session Handoff — 2026-07-18

> Read this first in a new session. Then `bash scripts/verify.sh` (should print GREEN),
> then `NORTHSTAR.md` → `SCOREBOARD.md` → this file's §Next moves.

## LATEST SESSION (2026-07-18, later) — the mobility search ran; the asterisk is closed

`hackathons/if-mobility-search/` — sealed 100/100 under condition (a). Three results:

1. **The still-life mystery is solved: energy starvation, not rules.** A pre-declared
   216-config sweep (prereg commit `aed7149` *before* any run) found that the same
   Conway-gate rules grow movers from random soup once construction is cheap:
   `B3/S23, e_birth=0.25, e_maint=0.01, inflow=12, σ=40, ρ=0.15` → mean **24.0 emergent
   mobile tracks per run** (8 seeds × 600 steps, D4 threshold was 0.5). Boss #8
   (Still-Life Desert) slain. Also found on entry: the old detector (`min_size=6`) was
   glider-blind by construction, but M0 re-confirmed the still-life verdict at
   `min_size=5` — the old kill stands on stronger evidence.
2. **The first causal-work audit on agents a universe produced** (what the asterisk
   demanded): 21 emergent movers, 24-seed declared roster, frozen criteria →
   **UNDECIDED, t = −0.94** (W_C mean −4.9 ± 5.2 SEM, 61.9% positive). Run 1 returned
   only 9 movers < frozen minimum 20 → VOID honored; roster extended by declaration
   *before* rerun. No claim upgrade in either direction.
3. **A real finding in the tail**: count-preserving scrambles of movers occasionally
   ignite debris growth explosions that out-harvest the organized structure
   (W_C −39, −93). In an energy-rich regime organization is energy-*frugal*, not
   energy-greedy — **raw regional harvest is probably the wrong observable for
   mover-class agency**. This is the #1 open thread now (see Next moves).

New tools: `scripts/mobility_search.py` (UniverseX + wrap-aware overlap tracker +
declared sweep), `scripts/mover_audit.py` (mover-adapted causal-work audit).
Evidence: `evidence/mobility_{controls,sweep_stageA,confirm_stageB}_2026_07_18.json`,
`evidence/mover_audit_2026_07_18.json`. XP 10150 → **11050**; belt stays Green
(Blue still requires an external party).

### Loop iteration 1 (2026-07-18, /loop) — movers do NOT track resource (tight null)

`hackathons/if-resource-tracking/` sealed 100/100. The M3-stretch question ran as its
own pre-registered circuit (prereg `59c99b7` before any run): τ = window-cosine between
mover displacement and bearing-to-source, gradient (σ=40) vs placebo (σ=10⁶) arms,
32 seeds each. Result: 849 vs 830 qualifying tracks, **Welch t = +0.34 → UNDECIDED**
by frozen criteria — and at this power that is a **tight null**: any tracking bias is
< ~4% of full alignment. Movers in the sealed regime are ballistic; direction is set at
birth. Diagnosis: inflow=12 is energy-abundant — nothing selects on direction.
New tools: `scripts/tracking_test.py`; path recording added to the tracker (additive).
XP → 11450.

### Loop iteration 2 (2026-07-18, /loop) — no scarcity boundary in inflow; second tracking null; heredity diagnosis

`hackathons/if-scarcity-boundary/` sealed 100/100 (prereg `10652fc` before any run).
S1: movers persist smoothly across inflow 0.5→12 (2.25→28.1/run, no threshold) — the
**mobility phase boundary is in construction cost (E_BIRTH), not inflow**; session
expectation falsified and kill-logged. S2 at the grid floor (inflow=0.5, fresh seeds
33–96): Welch t=+0.54 → **UNDECIDED** — second tracking null; movers are ballistic
across the whole tested energy range. Secondary r(τ, lifetime) ≈ 0 both arms.
**Heredity diagnosis (successor question):** movers live 50–130 steps and do not
reproduce — no lineage for directional selection to accumulate on. Tracking-by-selection
needs *replicating* movers. Videos also shipped this loop: `videos/` (3 films tied to
sealed results). XP → 11750.

### Loop iteration 3 (2026-07-18, /loop) — PRODUCERS ABSENT; the heredity gap is the mechanism

`hackathons/if-lineage/` sealed 100/100 (prereg `8692d62`). Birth-attribution census,
24 seeds × 2 arms: production *events* common (14.5 mover-productions/run — one-shot
fission) but **repeat-producers (≥2 mobile children) ABSENT**: 0 in the gradient arm.
The pre-registered painful tier fired and stands: **in energy-gated B3/S23, motion
emerges but agency cannot be selected for, because no heredity unit exists.** This is
the mechanism behind both tracking nulls. Kill-logged. XP → 12150.

### Loop iteration 4 (2026-07-18, /loop) — rule search BARREN; the emergent-agency branch is CLOSED

`hackathons/if-rule-search/` sealed 100/100 (prereg `fa9d43a`). Six rule families
(HighLife, LowDeath, Pedestrian Life, 34 Life, 2×2, Day & Night) × 4 energy configs ×
4 seeds: **zero flagged configs** (two sub-bar singletons). The frozen stop rule fired:
**the emergent-agency branch closes.** The four-experiment arc is now a complete,
fully pre-registered negative with its mechanism: energy-gated Life-like CA produce
**motion without lineage** — nothing for selection to act on. Scope limitation recorded
(program budget ~10² soups vs Catagolue's ~10⁹; reopening requires a logged amendment
with a fundamentally cheaper detection idea). B3/S23 movers remain live for exactly one
question: the causal-work audit (UNDECIDED at n=21). XP → 12550.

### Loop iteration 5 (2026-07-18, /loop) — power run UNDECIDED; emergent program RESTS; P17 written

`hackathons/if-causal-power/` sealed 100/100. Fresh 64-seed roster → 53 movers:
t = −0.88 → **UNDECIDED**, rest-clause fired — no further same-design sampling. Stable
shape across 74 total movers: median W_C +0.4 (51–62% positive), mean dragged negative
by the scramble-ignition tail. **The whole five-hackathon arc is now canon:**
`canon/papers/P17-motion-without-lineage.md` — movers emerge (construction-cost phase
boundary), movers are ballistic (2 nulls), no heredity unit (ABSENT), generalizes
(BARREN, branch closed), causal work indistinguishable from zero under drawdown
(rest). XP → 13250.

### Loop iteration 6 (2026-07-18, /loop) — ℒ_IF design constraints written (no evaluation)

`canon/20-cosmology/04-lif-design-constraints.md`: hard constraints C1–C8 any ℒ_IF
must survive before freezing (dimensional closure, Newtonian recovery, zero per-galaxy
freedom, the frozen admission bar, BTFR-out-not-in, the new P17 constraint, cross-scale
consistency, regeneration), three candidate families sketched (modified inertia /
entropic gradient / memory kernel) with named risks, and branch forbidden states
(`FIT_BEFORE_FREEZE`, `PER_GALAXY_KNOB`, `INTERPOLATION_SMUGGLING`,
`ESTIMATOR_HANDWAVE`). No SPARC contact — no legitimacy spent.

### Loop iteration 7 (2026-07-18, /loop) — family C killed on paper; family A's wedge found

`canon/20-cosmology/05-lif-family-selection.md`: **memory-kernel gravity (family C) is
dead** — any single global timescale forces BTFR slope 3 vs observed ≈ 4 (dimensional
proof; kill-logged; zero data cost). Family B deferred pending a Verlinde prior-art
audit. **Family A survives only via the wedge: structure-dependence of rotation curves
at fixed Σ(r) — a prediction MOND forbids.** Freeze-blocked on (i) pixel-level
structure-estimator specification, (ii) Σ-degeneracy literature check. XP → 13750.

### Loop iteration 8 (2026-07-18, /loop) — family A killed by the RAR bound; the SPARC arena is closed

`canon/20-cosmology/06-sigma-degeneracy-check.md`: published RAR results (0.13 dex
scatter, residuals Gaussian ≈0.11 dex and **uncorrelated with any galaxy property** —
Lelli+2017 ApJ 836:152; McGaugh+2016 PRL 117; intrinsic scatter ≈ 0 in follow-ups)
bound any structure-at-fixed-g_bar effect below SPARC detectability. **Family A dead;
with C dead and B deferred, all declared ℒ_IF families are exhausted at zero fit cost.
Proved consequence: galaxy rotation curves cannot host IF-specific evidence; P11 stays
blocked-not-pending WITH a proof; the cosmology branch's positive target is the
Euclid-facing prediction lattice (boss #7).** XP → 14250.

### Loop iteration 9 (2026-07-18, /loop) — lattice hardened; the reduced theory object

`canon/20-cosmology/07-lattice-hardening.md`: every lattice entry re-statused under the
RAR closure. Entry 02 tombstoned (CLOSED_BY_PROOF), 05 demoted (MOND's fight, not
ours), 03 survives only via hysteresis. **Key yield: the pre-DR1 program needs only a
reduced theory object — scalar b(z) + response maps μ[b], w[b] + τ_IF (~5 numbers), not
a galaxy functional.** Ranked build queue: notebook 10 (boss #6, public chains, buildable
NOW) → entry 08 merger-memory data survey → entry 12 estimator freeze → notebook 14
Euclid prereg (~Oct 2026). XP → 14550.

### Loop iteration 10 (2026-07-18, /loop) — boss #6 engaged: prereg + sign frozen pre-data

`08-notebook10-prereg.md` (commit `54f9717`) + `09-sign-derivation.md`: the
expansion–growth shape-consistency test is frozen. w = −1 + A_w(1+z)^(−γ_E),
μ = 1 + A_μ(1+z)^(−γ_G); the IF bet = **γ_E = γ_G with A_w > 0 AND A_μ > 0** — the
growth sign runs AGAINST the S8 trend, so the bet is real and the community prior
leans against us. DESI's evolving-DE hint disclosed as retrodiction (no credit).
Verdict tree frozen incl. UNIFICATION DEAD. Execution order: sign note ✅ → notebook
skeleton w/ CONTRACT (datasets pinned, no data) → expansion fit → growth fit →
verdict, each step committed before the next. XP → 14950.

### Loop iterations 11–12 (2026-07-18, /loop) — notebook 10 round 1 COMPLETE: INDISTINGUISHABLE

Skeleton + data guard built (`notebooks/10_expansion_growth_consistency.ipynb`).
Ingest from pinned sources (DESI DR2 BAO official likelihood via CobayaSampler repo;
Pantheon+ official release incl. 32MB STAT+SYS cov; Planck-2018 distance priors
transcribed from the Chen/Huang/Wang PDF Table I — nothing typed from memory).
Logged pre-fit amendment: r_d free (conservative). Fit (`scripts/expansion_fit.py`,
deterministic profile likelihood): ΛCDM validates (Ωm 0.309, h 0.679, r_d 148.3);
IF shape family Δχ² = −1.74 → **A_w ≠ 0 at only ~1.3σ < the frozen 2σ gate →
verdict INDISTINGUISHABLE (branch 1); growth side NEVER OPENED** — test preserved
intact for DESI DR3 / Euclid DR1. Boss #6 round 1 fought, unresolved. Possible v2
amendments declared in `10-notebook10-verdict-v1.md` (must be logged before running).
XP → 15450.

### Loop iteration 14 (2026-07-18, /loop) — arrow-records VOID; the gate worked

`hackathons/if-arrow-records/` sealed as VOID at 70/100 (prereg `e46de28`). G1: the
Critters-class reversible core retraces bit-perfectly. G2 fired: the substrate is a
number-conserving particle gas (pop exactly 315 across 400 steps) — in-place
persistence records are the EMPTY SET; frozen-block count over 16 composed steps = 0.
TA-H3/TA-H11 left untouched (no verdict from a broken instrument). Lesson logged:
reversible + number-conserving ⇒ no cheap records (aligned with P12 TA-H6/TA-H11
spirit). Next attempt specified: BBM-with-walls substrate, own prereg. XP → 15800.

### Next moves after this session, ranked

1. **Entry 08 — DONE (iteration 13): PARKED on a data gap.** Offsets (Harvey+15,
   errors per Wittman+18) and collision ages (MCC 29 clusters) are disjoint samples;
   τ_IF prereg premature. Reopen conditions recorded in
   `canon/20-cosmology/11-merger-memory-survey.md` (Euclid-era lensing of the MCC
   sample is the natural trigger).
2. **Entry 12 — I_NL estimator freeze** (CAMELS, pixel-level spec before any data).
3. **Notebook 10 round 2** — parked until DESI DR3 / Euclid DR1 (or a logged v2
   amendment); growth side must stay unopened until an expansion signal ≥ 2σ exists.
4. **Track-C notebooks** · **book manuscript** (arc now includes P17 + boss-#6 round 1).
2. **Entry 08 survey**: published merging-cluster lensing–X-ray offsets + collision
   ages; ≥10 systems → τ_IF prereg becomes possible.
3. **Entry 12**: freeze the I_NL estimator on CAMELS to pixel level.
2. **Track-C notebooks** (arrow of time, expansion–complexity window, memory depth,
   repair/mortality, cooperation) — deterministic, visualizable, unbuilt; the agency
   branch's remaining constructive program.
3. **Book manuscript** (*The Battery That Learned to Ask*) — the arc now includes P17
   and the two zero-cost cosmology kills; the honest-program story is the book's spine.
4. **Π_C-primitive disagreement** (panel round) · redesigned causal-work observable
   (own prereg).
2. **Track-C notebooks** (arrow of time, expansion–complexity window, memory depth,
   repair/mortality, cooperation) — deterministic, all visualizable, unbuilt.
3. **Redesigned causal-work observable** (harvest-per-mass-step) — only with its own
   prereg; the drawdown design is retired by the rest-clause.
4. **Π_C-primitive live disagreement** (panel round; leak4 driver needs the scripts/
   workarounds) · **book manuscript** (arc now includes P17).

-2. **Causal-work audit power run — DONE (iteration 5, UNDECIDED, program rests).** The one live
   question on the emergent substrate. Declare a fresh ~64-seed roster; primary verdict
   on the fresh roster alone at the frozen ±2 thresholds; pooled with the sealed 21 as
   declared secondary. At sd≈24 and ~1 mover/seed, n≈85 total resolves |W_C|≈5. If it
   decides either way, that is the first decided causal-work result on universe-grown
   agents. If still undecided, report widths and the emergent program rests until a
   better observable is designed (harvest-per-mass-step candidate).
-1. **Rule-family producer search — DONE (iteration 4, BARREN, branch closed).** Declare a
   rule grid with replicator/gun folklore (e.g. B36/S23 HighLife, B368/S238, B34/S34,
   B2/S345?) + the energy axes, frozen producer criterion (P3, ≥2 mobile children),
   and a stop rule. Find ONE energy-gated family with reproducible producers → rerun
   the τ tracking test there (selection finally has a lineage to act on). If the
   declared grid is barren → close the emergent-agency branch honestly; B3/S23 movers
   remain the substrate for the causal-work question only.
0. **The lineage question — DONE (iteration 3, see above).** Do mover-*producing*
   structures exist in any regime — parent structures that emit new mobile tracks — and
   is production rate resource-coupled? Instrument: birth-attribution in the tracker
   (new track whose birth cells overlap a dilated existing structure = a production
   event). If lineages exist, directional selection has something to act on and the
   tracking question reopens; if not, ballistic movers are the ceiling of this rule
   family and the honest conclusion is that B3/S23-class rules cannot produce
   tracking agents — move rule families or move on.
1. **The scarcity-boundary hypothesis** — DONE (iteration 2, see above). Tracking
   can only be selected where direction affects survival. Sweep inflow DOWN from 12
   toward the mobility floor (stage A found movers at inflow=4 but fewer; original
   universe starved at 0.9); pre-register: at the scarcity edge, (a) movers still emerge,
   (b) τ(gradient) − τ(placebo) > 0. If tracking appears under scarcity and vanishes
   under abundance, that is a *selection-produces-agency* result — the program's first
   positive emergent-agency claim. Freeze the boundary-finding procedure separately
   from the τ test to avoid garden-of-forking-paths.
2. **More statistics on the mover causal-work audit** — declare a ~64-seed roster
   before running; at sd 23.8 the current design needs ~5× n to resolve W_C ≈ ±5.
3. **Find the right harvest observable** — survival-conditioned harvest or harvest per
   mass-step (the scramble-ignition confound from `if-mobility-search` still stands).
4. Then the pre-existing queue below (ℒ_IF commit-first for Phase 3/4, Track-C
   notebooks, Π_C-primitive disagreement, book).

---

## What the previous session did

Rebooted the repo from scratch around the core idea (old repo → `_archive/`, read-only),
extracted 15 working papers from a ChatGPT thread, wrote 17 canonical papers, built 7
notebooks and 2 figures, ran 5 adversarial frontier-panel rounds, and — the point —
**falsified the program's flagship claim and four other things, three of them our own.**

## The one-paragraph state of the theory

The **core intuition** (universe as discharging informational battery; life and
intelligence as recharge circuits; past a threshold intelligence preserves and enhances
the system) is **UNTESTED, not refuted** — the cosmology branch has no implementable law,
so nothing touched it. The **agency branch's flagship claim** (IF-H1: a universal
dimensionless constant of agency) is **FALSIFIED** — but see the asterisk below, it is
falsified *for hand-designed agents only*. The **cosmology branch** is worse than
untested: its one previously implemented galaxy law lost to both MOND and an NFW halo on
175 galaxies. **One positive result survived**: the parasite band's fixed-order crossing
claim, confirmed at +138σ in a provably coupling-free substrate, though its qualitative
point was de-claimed as prior art after a self-audit.

## The asterisk that matters most (discovered in the final hour) — ✅ CLOSED, see LATEST SESSION above

**The Conway gate was never satisfied by the experiments that falsified IF-H1.** The ring,
Kalman, and chemotaxis families all had memory mechanisms *designed by hand*. When an
actual Conway-gate universe was finally built (`scripts/universe.py`), what emerged there
were **still lifes** — static crystals, zero mobility, harvest ≡ 0.00 — so the causal-work
audit had nothing to audit. **IF-H1 has never been tested on agents a universe produced.**
This is logged in `SCOREBOARD.md` §Kill log and is the single most important open thread.

## Assets

| | |
|---|---|
| Papers | 17 in `canon/papers/` (P00–P16). **P15** = the falsification. **P16** = the survivor. |
| Notebooks | 7 in `notebooks/`, all with frozen contracts; 04, 04e, 04f, 04g, 04h, 01 have run |
| Figures | `figures/universe-still-life.png`, `figures/glider-selection.png` |
| Scripts | 17 in `scripts/` — incl. `universe.py` (the CA lab), `verify.sh` (6 gates), CDP tooling |
| Data | SPARC 175 galaxies live in `data/sparc/` (gitignored, sha256 in `CHECKSUMS.txt`) |
| Panels | 5 rounds in `canon/extracted/frontier-panel/`; founding panel in `canon/panels/` |
| Evidence | `evidence/sparc_baseline_2026_07_18.json`, `evidence/glider_selection_2026_07_18.json` |

## Phase / rung state

| Phase | State |
|---|---|
| 0 Absorb + Constitution | ✅ sealed (`if-founding-100` 100/100) |
| 1 Foundations (rung 641) | ✅ sealed — quantity audit + 2 declared exclusions + verify gate 6 |
| 2 Artificial Universe Lab (rung 274177) | ✅ sealed **by falsification** — but see the asterisk |
| 3 Observational Cosmology | ▶️ 62/100 — SPARC restored, MOND/NFW baselines reproduced independently |
| 4 Preregistration (rung 65537) | ⛔ **BLOCKED ON A THEORY** — P11 cannot be frozen while ℒ_IF is a free function |
| 5 Papers + Book | ✅ corpus + book arc (`canon/30-meaning/03-book-arc.md`); manuscript remains |

Game state: **Green belt · 10150 XP · 4 bosses slain.** Blue belt requires an
*independent party* replicating P16 — it cannot be earned inside this repo.

## Next moves, ranked

1. **The mobility search (highest value, cheapest).** Sweep the rule/energy space of
   `scripts/universe.py` for a regime where **mobile, resource-tracking structures emerge
   on their own** — today's gliders were seeded; the prize is a universe that grows them.
   Then run the causal-work audit on *those*. That would be the first honest test of IF-H1
   and it closes the asterisk above. No data, no waiting, GPU available.
2. **Propose a specific ℒ_IF and commit it BEFORE evaluating.** This is the single
   blocker for Phase 3 C4/C5 and all of Phase 4. Legitimate only if the commit precedes
   any SPARC run. Admission bar already frozen: median χ²/dof ≤ 3.71 to be interesting,
   ≤ 1.14 with BIC ≤ 19.8 to win, 30% held-out, no per-galaxy IF parameters.
   Reproduced local baselines to beat: MOND 3.298, NFW 0.938.
3. **The unbuilt Track-C notebooks** — arrow of time, expansion–complexity window,
   optimal memory depth, repair/mortality, cooperation. All deterministic, all visualizable.
4. **The live disagreement**: panel round 5 says *don't build J*; make Π_C the primitive.
   We deferred but did not resolve. Worth one round if J is ever needed for within-mechanism
   attribution rather than the agency verdict.
5. **Book manuscript** — arc drafted, working title *The Battery That Learned to Ask*.

## Standing disciplines (do not quietly drop)

- **Pre-commitment is binding at the moment it hurts.** Two failed rescalings ended IF-H1
  and we did not attempt a third.
- **Publish the kill** the same session it fires; `SCOREBOARD.md` §Kill log is the repo's
  most credible artifact.
- **Component-optimality**: an ablation is interpretable only when the intact agent is
  optimal in the ablated component. This rule caught a false positive of ours.
- **Π_C is the primitive**, Π_A establishes participation only. The measurements that
  survived are the ones that never tried to put work and information on a common scale.
- **Sagan phrasing for cosmology, verbatim**: "specified, not yet implemented, not yet tested."

## Known tooling friction

The `/leak4` panel driver failed to send in rounds 3, 4, and 5 (prompts land in composers
unsent), and the ChatGPT tab times out on CDP eval. Working replacements are in `scripts/`:
`cdp_eval.py` (raw WS CDP), `send_prompt.py`-pattern (insertText + click/Enter fallback),
poll-until-stable harvesting, and `driver.py` (ChatGPT "Next"-harvest). Solace Browser must
be running on CDP :9888.
