# Historical notebook rerun — Euclid Q1 (2026-06-12)

Era worktree 12473c8, quick tier. The era euclid_util discovered and
downloaded its own 5 SPE/MER tile pairs from IRSA and generated
synthetic randoms, exactly as in 2025-08.

| Metric | Fresh 2026-06-12 | Saved 2025-08 quick |
|---|---|---|
| Tiles | 5 (today's discovery order) | 5 (August's set) |
| Per-tile r | 0.754 – 0.966 | — |
| Mean r | 0.891 | 0.962 |
| Mean σ | 3.1 | 3.8 |

Verdict: REPLICATES-WITH-SPREAD — qualitatively consistent; the mean is
sensitive to which 5 tiles the discovery picks (today's set ≠ August's),
and one weak tile (r=0.754, χ²/dof=168) dragged the fresh mean. The
best fresh tile (r=0.966/3.9σ) matches the saved mean. Euclid remains
the most fragile of the three arms: synthetic randoms ARE the
measurement in Landy-Szalay, and tile-to-tile variance is large at
quick tier. The 102-tile full run (~5h+, ~9GB) is the only
tight comparison; not executed in this pass.
