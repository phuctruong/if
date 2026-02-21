# ROADMAP: IF-Theory — Information as First Force

> NORTHSTAR: Formalize Information Force Theory into a testable, reproducible physics simulation framework.
> Proof Completeness: # of IF Theory theorems with verified, machine-checkable proofs at rung 274177.
> All simulation paths: Fraction/Decimal only (zero float tolerance). Real data only — no synthetic data.

## Phase Summary

| Phase | Name | Rung Gate | Key Deliverable |
|-------|------|-----------|----------------|
| 0 | Data Integrity Audit | 274177 | All SDSS/DESI/Euclid data verified real, all notebooks reproducible |
| 1 | Milky Way Validation (C-Track) | 274177 | v(10 kpc) = 220±20 km/s from IF Theory field equations |
| 2 | Academic Paper | 274177 | ApJ submission ready by March 2026 |
| 3 | Euclid DR2 Predictions | 274177 | Falsifiable predictions for upcoming Euclid data releases |
| 4 | PhucNet Integration | 641 | IF Theory books + YouTube content guide published |

---

## Phase 0: Data Integrity Audit

**Goal:** Verify every data source is real (not synthetic). Ensure all notebooks are reproducible on fresh checkout.

**Why this matters:** Physics claims built on synthetic data are unfalsifiable. Every IF Theory result must be traceable to public survey data (SDSS, DESI, Euclid DR1) with documented provenance.

**Tasks:**
- [ ] Inventory all data files: list every file in `data/`, `bao_data/`, `euclid_data/`, `results/`
- [ ] For each data file: document source URL, download date, and SHA-256 checksum
- [ ] Verify SDSS data provenance: confirm it matches SDSS DR17 public release
- [ ] Verify DESI data provenance: confirm it matches DESI EDR or DR1 public release
- [ ] Verify Euclid data provenance: confirm it matches Euclid DR1 public release
- [ ] Run all notebooks in fresh environment: `jupyter nbconvert --to notebook --execute *.ipynb`
- [ ] Document any notebooks that fail to execute and why
- [ ] Confirm zero synthetic/generated data in any notebook that produces theorem-supporting results
- [ ] Produce `evidence/data-audit.json`: checksums, sources, reproducibility status for all data

**Acceptance Criteria (Rung 274177):**
- [ ] All data files have documented SHA-256 checksum + public source URL
- [ ] All notebooks execute successfully on fresh checkout (or failures documented with mitigation)
- [ ] Zero synthetic data used in any result that supports an IF Theory claim
- [ ] `evidence/data-audit.json` complete with provenance chain for every data file
- [ ] Seed sweep: 3 seeds × 2 replays — deterministic simulation results

---

## Phase 1: Milky Way Validation (C-Track)

**Goal:** Derive v(10 kpc) = 220±20 km/s from IF Theory field equations using real SDSS/DESI rotation curve data.

**Context:** This is the C-Track validation — the Milky Way galactic rotation curve prediction. IF Theory must reproduce the observed ~220 km/s circular velocity at 10 kpc from the galactic center without dark matter tuning parameters.

**Tasks:**
- [ ] Implement IF Theory field equations in `src/if_field.py` — exact arithmetic (Fraction/Decimal throughout)
- [ ] Load Milky Way rotation curve data from SDSS/DESI (verified in Phase 0)
- [ ] Run field equation solver: compute v(r) for r = 1–25 kpc
- [ ] Verify v(10 kpc) falls within 220±20 km/s — record in `evidence/milky-way-validation.json`
- [ ] Produce residual plot: IF Theory prediction vs observed SDSS/DESI data points
- [ ] Halting certificate: if iterative solver used, prove convergence with R_P certificate
- [ ] Null edge sweep: v(0) = 0 (no singularity at center), v(∞) → 0 boundary condition
- [ ] Seed sweep: 3 seeds — deterministic v(10 kpc) result each time

**Acceptance Criteria (Rung 274177):**
- [ ] v(10 kpc) = 220±20 km/s confirmed from IF Theory equations (not tuned to match)
- [ ] Residual plot produced and saved to `evidence/milky-way-residuals.png`
- [ ] Halting certificate attached if iterative method used
- [ ] Null edge sweep passes (center + infinity boundary conditions)
- [ ] Seed sweep: 3 seeds × 2 replays — byte-identical v(10 kpc) result
- [ ] Zero float in any verification path (Fraction/Decimal only)

---

## Phase 2: Academic Paper

**Goal:** Submit IF Theory paper to ApJ by March 2026, covering dark energy + BAO predictions.

**Target:** The Astrophysical Journal (ApJ). Focus: IF Theory derivation of dark energy equation of state + BAO scale prediction vs DESI DR1 measurements.

**Tasks:**
- [ ] Draft paper outline: Abstract, Introduction, IF Theory field equations, Methods, Results, Discussion
- [ ] Write Methods section: describe IF Theory field equations with full mathematical derivation
- [ ] Write Results section: Milky Way v(10 kpc) validation + BAO scale prediction
- [ ] Produce all figures: rotation curve comparison, BAO power spectrum, equation-of-state w(z)
- [ ] Write Abstract: 250 words, must include falsifiable prediction for Euclid DR2
- [ ] Internal review: run through phuc-forecast adversarial lens — identify weakest claims
- [ ] Address reviewer prep: list top 5 objections a referee would raise + IF Theory responses
- [ ] Format for ApJ: LaTeX template, figure captions, bibliography (AASTeX 6.3)
- [ ] Submit to arXiv preprint first (target: February 28, 2026)
- [ ] Submit to ApJ (target: March 15, 2026)

**Acceptance Criteria (Rung 274177):**
- [ ] All figures reproducible from raw data + IF Theory code (no manual editing)
- [ ] Abstract contains at least one falsifiable prediction for Euclid DR2
- [ ] Internal adversarial review complete — all high-risk claims addressed
- [ ] LaTeX compiles clean with `pdflatex` on fresh checkout
- [ ] arXiv submission URL recorded in `evidence/paper-submission.json`

---

## Phase 3: Euclid DR2 Predictions

**Goal:** Produce falsifiable, timestamped IF Theory predictions for Euclid DR2 before data release.

**Context:** Euclid DR2 is expected in 2026. IF Theory must make specific, falsifiable predictions BEFORE the data is released — this is what makes it science. Predictions must be timestamped (git commit + arXiv).

**Tasks:**
- [ ] Identify key Euclid DR2 observables: weak lensing power spectrum, BAO peak position, galaxy clustering
- [ ] Compute IF Theory prediction for BAO peak position at z=0.5, 1.0, 1.5
- [ ] Compute IF Theory prediction for weak lensing σ8 × Ωm^0.5 (S8 parameter)
- [ ] Compute IF Theory prediction for dark energy equation-of-state w(z=1)
- [ ] Write `predictions/euclid-dr2-predictions.json` with: predicted value + uncertainty + IF Theory derivation
- [ ] Git-tag the prediction commit: `git tag euclid-dr2-prediction-v1`
- [ ] Post to arXiv: "IF Theory Predictions for Euclid DR2" (short paper, 4 pages)
- [ ] After DR2 release: compare predictions vs observations — record in `evidence/euclid-dr2-validation.json`

**Acceptance Criteria (Rung 274177):**
- [ ] Predictions file `predictions/euclid-dr2-predictions.json` committed and tagged before DR2 release
- [ ] Each prediction has: central value, uncertainty, IF Theory equation reference, derivation notebook
- [ ] arXiv preprint submitted with predictions (timestamped record)
- [ ] All predictions computed with exact arithmetic — zero float in prediction path
- [ ] Validation notebook ready (runs when DR2 data is available)

---

## Phase 4: PhucNet Integration

**Goal:** Publish IF Theory content on phuc.net — technical books and YouTube-ready content guide.

**Context:** IF Theory results feed into phucnet (phuc.net) as long-form accessible content. Two deliverables: a technical book for phucnet/books/ and a YouTube content guide for theorem announcements.

**Tasks:**
- [ ] Write `books/if-theory/index.md`: chapter outline for full IF Theory book
- [ ] Write Chapter 1: "What Is Information Force?" — accessible introduction (target: high school physics level)
- [ ] Write Chapter 2: "The Field Equations" — mathematical core with worked examples
- [ ] Write Chapter 3: "Milky Way Validation" — C-Track result explained for general audience
- [ ] Produce phucnet article: "Mersenne Tower Theorem — What It Is and Why It Matters" (virality-optimized)
- [ ] Produce YouTube content guide: 5 video outlines for IF Theory theorem announcements
- [ ] Cross-link: each book chapter links to the corresponding proof notebook + arXiv paper
- [ ] Publish to phuc.net (coordinate with phucnet project ROADMAP Phase 1)

**Acceptance Criteria (Rung 641):**
- [ ] IF Theory book: Chapters 1–3 complete and readable by non-specialist
- [ ] phucnet article on Mersenne Tower Theorem: published with canonical URL
- [ ] YouTube content guide: 5 video outlines covering major IF Theory results
- [ ] All technical claims in book cross-reference the rung 274177 proof notebooks
- [ ] Workspace clean after publish: `phuc-cleanup` run, no sensitive data in published content

---

## See Also

- `NORTHSTAR.md` — proof completeness metric + ecosystem integration
- `CLAUDE.md` — prime-math + prime-coder loaded
- `skills/prime-math.md` — exact arithmetic for proofs
- `/home/phuc/projects/pvideo/NORTHSTAR.md` — pvideo (IF Theory physics engine)
- `/home/phuc/projects/phucnet/NORTHSTAR.md` — phucnet (publication channel)
- `/home/phuc/projects/stillwater/NORTHSTAR.md` — verification governance layer
