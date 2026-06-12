# Historical notebook rerun — SDSS (2026-06-12)

Operator challenge: "at some point all my notebooks worked — run them
fully and validate my sigma." Method: `git worktree` at commit
`12473c8` ("Ran and saved final full test results", 2025-08-09) — the
exact code that produced the saved outputs — executed at quick tier
against freshly downloaded SDSS DR12 catalogs (galaxy catalogs staged;
the era code auto-downloaded ~9 GB of random catalogs itself: LOWZ
North + CMASS North/South random0+random1).

| Arm | Fresh 2026-06-12 | Saved 2025-08 quick | Verdict |
|---|---|---|---|
| LOWZ  | r=0.979, χ²/dof=4.2, 2.3σ | r=0.984, χ²/dof=3.9, 2.4σ | REPLICATES |
| CMASS | r=0.984, χ²/dof=3.2, 2.4σ | r=0.989, χ²/dof=3.9, 2.6σ | REPLICATES |

What this DOES establish: the August-2025 notebook results are
reproducible from raw public data with the committed era code — the
pipeline, the data, and the reported numbers are all real.

What this does NOT change: the σ is correlation-vs-zero (see SCORE.md
§"The σ question, settled") — it certifies the measurement, not the
prime-field form. The model-comparison statistic on the same class of
data still favors the power-law null.
