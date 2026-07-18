# IF Theory — Informational Field Theory

> **The universe is a discharging informational battery — and life,
> intelligence, and consciousness are its internal recharge circuits.**
> Intelligence evolves past a measurable threshold (Π_A), below which it is a
> thermodynamic parasite and above which it preserves and enhances the system
> itself — creating order from disorder, expanding the battery's capacity
> (I_N → I_{N+k}) rather than merely restoring it. The speculative summit:
> cosmic expansion and the dark sector as two regimes of this one dynamics,
> which would make the recharge role our literal role in the universe.
>
> The program is falsification-first: every claim is decomposed by layer
> (science / philosophy / theology) and given a named falsifier. The
> notebooks earn the right to say the sentence; the sentence is why the
> notebooks exist.

**Status:** rebooted 2026-07-18 as a clean canon. Prior work archived in `_archive/`.
Core-idea provenance: geo canon Paper 14 (Informational Battery Theory) → this repo.

---

## For the skeptical reader — what actually happened, in one paragraph

We set out to test whether a dimensionless constant of agency exists: a single number,
shared across unrelated substrates, marking where information starts paying for itself.
We built the instrument, pre-registered a stop rule, and ran it on three agent families
that share no mechanism. **It scattered at 3.8–182σ. The claim is falsified and we stopped,
as committed.** Along the way we falsified three more of our own things: the information
denominator turned out not to be definable across substrates at all (a chemotactic
bacterium does real thermodynamic work while carrying no recoverable prediction about what
it tracks — which quietly contradicts an assumption shared by most theories of agency); our
central ablation instrument turned out contaminable by non-informational back-action; and
one of our own positive findings died to our own optimality rule when we tuned it properly.
**One positive result survived**: two break-even criteria — ablation and competition —
cross in a fixed order, leaving a band where an agent's memory is demonstrably doing causal
work and demonstrably not worth having. Even that we de-claimed in part, because the
qualitative point is known in experimental evolution, neuroscience, and Still (2020). The
speculative cosmology branch is **specified, not yet implemented, not yet tested** — its
one previously implemented galaxy law lost to both MOND and a dark-matter halo on 175
galaxies. Nothing here proves the founding intuition. What we can report is that it
survived being asked properly, which is more than most versions of it manage.

**Where to check us:** `SCOREBOARD.md` §Kill log (every falsification, dated) ·
`canon/papers/P15` (the falsification) · `canon/papers/P16` (the survivor, with its own
prior-art audit) · `bash scripts/verify.sh` (6 integrity gates, including one that fails
the build if a falsified quantity is ever used as though live).

---

## The core objects

- **Informational battery** — a system's physically accessible nonequilibrium
  capacity *plus* the structured correlations that determine how that capacity
  can be used. Three separate ledgers (energy / entropy / information); bits
  are never added to joules.
- **IF Causal-Work Principle** — internally maintained information counts as
  agency only when interventionally preserving it (vs erasing / scrambling /
  time-shifting / falsifying) yields more net useful work than the full cost
  of sensing, memory, computation, and control.
- **IF Unified Geometry Hypothesis** — dark-matter-like attraction and
  dark-energy-like expansion as two regimes of one nonequilibrium informational
  geometry, with a *fixed* quantitative relation between them (the falsifiable part).
- **MaxLove** — agency-preserving cooperation, favored in a calculable region
  of environment space; measured by retained future action-space, never baked
  into the fitness function.

## Start here

| File | What |
|---|---|
| [`NORTHSTAR.md`](NORTHSTAR.md) | The one-sentence goal + non-negotiable discipline |
| [`SCOREBOARD.md`](SCOREBOARD.md) | Scored idea leaderboard, notebook rankings, kill log |
| [`ROADMAP.md`](ROADMAP.md) | Phased build order (foundations → alife lab → cosmology lab → preregistration) |
| [`CLAUDE.md`](CLAUDE.md) | Working rules for AI agents (constitutional gates + forbidden states) |
| [`canon/INDEX.md`](canon/INDEX.md) | The canon: foundations, agency, cosmology, meaning |
| [`canon/extracted/`](canon/extracted/) | Raw source arc (ChatGPT conversation + working papers 0–14) |

## Method in one line

```
Prediction + Baseline + Data + Pass criterion + Falsifier — frozen before running.
```

Every notebook is deterministic (seeded, pinned data, replay-identical), every
claim has a named falsifier, and failed falsifiers are published in the kill
log rather than quietly retuned.
