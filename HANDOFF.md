# Session Handoff — 2026-07-18

> Read this first in a new session. Then `bash scripts/verify.sh` (should print GREEN),
> then `NORTHSTAR.md` → `SCOREBOARD.md` → this file's §Next moves.

## What this session did

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

## The asterisk that matters most (discovered in the final hour)

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
