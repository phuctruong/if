# IF Theory Notebooks

Every notebook is a deterministic experiment under the **prediction contract**.
The first markdown cell MUST declare, frozen before any code runs:

```
PREDICTION   the falsifiable statement, with the equation
BASELINE     what the null/competitor model is (halo fits, ΛCDM, KW, FEP, ...)
DATA         pinned source + version (or seeded synthetic universe)
PASS         the quantitative pass criterion, decided in advance
FALSIFIER    the observation that kills the claim — and gets published in SCOREBOARD.md §Kill log
```

Rules:
- Fixed seeds everywhere. Rerun = identical output (deterministic replay).
- Ledger integrity cells: property-based conservation checks that raise on violation (Noether gate).
- No teleological variables in rule sets (Conway gate).
- Baselines reproduce published results before any IF cell exists (Feynman gate).
- Data versions pinned; survey statistics validated on mocks first (Rubin gate).

## Build order

See `../ROADMAP.md` Phase 2 (artificial-universe lab) and Phase 3 (cosmology lab).
Track C notebooks run with zero external data; Track A/B notebooks mine
`../_archive/` for the previous generation's SPARC/BAO pipelines (state provenance).

## Naming

`NN_short_name.ipynb`, numbered per the roadmap. Divergence-hunt companions:
`04a_kw_vs_if_divergence.ipynb`, `04c_fep_cost_audit.ipynb`,
`04d_maxwell_demon_landauer.ipynb` (panel round-1 additions).
