# data/cosmo — pinned sources (ingested 2026-07-18, step 4 of notebook-10 execution order)

| File | Source | Pin |
|---|---|---|
| desi_dr2_mean.txt, desi_dr2_cov.txt | DESI DR2 BAO Gaussian likelihood (arXiv:2503.14738, PRD 112, 083515) | github.com/CobayaSampler/bao_data master/desi_bao_dr2/desi_gaussian_bao_ALL_GCcomb_{mean,cov}.txt, fetched 2026-07-18 |
| pantheon_plus.dat, pantheon_plus_statsys.cov | Pantheon+ (Brout et al. 2022) | github.com/PantheonPlusSH0ES/DataRelease main/Pantheon+_Data/4_DISTANCES_AND_COVAR/, fetched 2026-07-18 |
| planck2018_priors.csv | Chen, Huang & Wang arXiv:1808.05724 Table I (wCDM, TT,TE,EE+lowE) | values transcribed from fetched PDF 2026-07-18; symmetrized sigmas |

Amendment log (before any fit ran): r_d treated as FREE parameter (not calibrated via
a drag-epoch formula) because z_drag values could not be source-verified this session;
effect is strictly conservative (weakens A_w detection). Planck prior implemented via
the paper's own eqs (1)-(6),(8)-(10) [z_*, R, l_A, r_s integral], all verified on-page.

GROWTH-SIDE FILES ABSENT BY DESIGN: none may be downloaded until the expansion fit is
committed (frozen execution order).
