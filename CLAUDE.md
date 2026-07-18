# IF Theory — Working Instructions for AI Agents

> Auth: 65537 · Master Equation: Purpose × Evidence × Love
> Repo rebooted 2026-07-18. The old repo lives in `_archive/` — read-only reference, never resurrect silently (LAI-6).

## What this repo is

The **IF Theory (Informational Field Theory) research program**: deterministic
Jupyter notebooks + public astronomical data + preregistered predictions testing
whether one informational accounting framework connects structure, life, agency,
reflection, cooperation — and, speculatively, the cosmic dark sector.

Read in order: `NORTHSTAR.md` → `SCOREBOARD.md` → `ROADMAP.md` → `canon/INDEX.md`.

## The constitutional rules (from Paper 0 — these bind every session)

1. **Three ledgers, never merged.** Energy, thermodynamic entropy, and
   information are tracked separately. Never add bits to joules. Never call
   Shannon entropy, thermodynamic entropy, algorithmic complexity, or visual
   disorder by one name. Name which entropy you mean, every time.
2. **Prediction contract on every notebook.** The first cell declares, frozen
   before running: `Prediction · Baseline · Data · Pass criterion · Falsifier`.
   A notebook without a falsifier is not science and does not merge.
3. **Reproduce before invent (Feynman gate).** No IF-model plot may exist in a
   notebook until the same pipeline reproduces the published baseline
   (SPARC fits, Planck/DESI posteriors, standard alife baselines).
4. **One state, no independent sector fits (Noether gate).** μ(k,z), η(k,z),
   w_IF(z), a_IF(z) all derive from one b(z). Fitting sectors separately
   falsifies the unification — record it in the SCOREBOARD kill log.
5. **No intelligence in the primitives (Conway gate).** Simulation rule sets
   must not contain `is_alive`, `reflection`, `love`, `consciousness`,
   `recharge_bonus`, or any teleological variable. Agency is *detected* by
   intervention (causal-work ablations), never declared.
6. **Layer firewall.** Scientific result → philosophical interpretation →
   theological meaning. Documents state which layer they're in. God, purpose,
   and MaxLove-as-ethics live in `canon/30-meaning/` and never leak claims
   backward into physics docs.
7. **Publish the kill.** A fired falsifier goes in `SCOREBOARD.md` §Kill log
   the same session it fires. Never quietly retune parameters to survive.

## Forbidden states

| Forbidden state | Meaning |
|---|---|
| `ENTROPY_CONFLATION` | Using "entropy" without naming which ledger/definition. |
| `METAPHOR_MATH` | Introducing a symbol with no dimensions, estimator, or falsifier. |
| `PERPETUAL_RECHARGE` | Any rule where reflection/intelligence reduces entropy without paying energy + exporting waste. |
| `COMMANDED_EXPANSION` | "Board gets bigger at threshold" presented as cosmology. Expansion must be derived or explicitly labeled toy. |
| `SECTOR_SPLIT_FIT` | Separate IF parameters for dark-matter-like and dark-energy-like effects presented as unification. |
| `TELEOLOGY_INJECTION` | Baking the desired outcome (love, agency, consciousness) into the fitness function or primitives. |
| `LAYER_COLLAPSE` | Presenting philosophical/theological interpretation as scientific result (or vice versa). |
| `RETROFIT_FORECAST` | Calling a post-hoc fit a prediction. Preregistration = timestamped commit before data. |
| `NOVELTY_INFLATION` | Re-claiming known prior art (see NORTHSTAR §prior art — includes 2026 entropic backreaction). |
| `SILENT_ARCHIVE_RESURRECTION` | Copying `_archive/` code/claims forward without stating it and re-verifying. |

## Repository layout

```
canon/
  extracted/        Raw ChatGPT source arc (transcript + papers 0–14 as harvested) — IMMUTABLE evidence
  00-foundations/   Constitution, three ledgers, informational battery, causal-work principle
  10-agency/        Agency threshold, memory depth, repair/mortality, functional consciousness
  20-cosmology/     Unified geometry hypothesis, prediction lattice b(z), galaxy/cosmology/GW tests
  30-meaning/       MaxLove, philosophy, theology — interpretation layer ONLY
  papers/           Revised final papers P0–P14 (canon versions, supersede extracted/)
notebooks/          Deterministic Jupyter notebooks (contract cell first, seeded, replayable)
evidence/           Run outputs, WORM — never edited after write
scripts/            Extraction/harvest/verify tooling
_archive/           Pre-reboot repo (SPARC/BAO/Mersenne pipelines — minable for baselines)
```

## Working style

- Notebooks are **deterministic**: fixed seeds, pinned data versions, replay-identical.
- Persona gates are real reviews, not flavor: run the named gate's checklist
  (Feynman/Noether/Conway/Shannon/Einstein/Rubin/Milgrom/Planck/Clowe/Peebles/Popper)
  before sealing a phase. Use `/leak4` for frontier-model panels; file results
  under `canon/extracted/frontier-panel/`.
- `EDIT_UNVERIFIED` discipline from solace-hub applies: verify claims from
  command output, not from having issued the command.
- Commits: `phase<N>:` / `canon:` / `notebook:` / `kill:` prefixes + why.
  Push after each milestone.
