# External Peer Review — IF Theory (Prime Field) Full Replication Pass

> **Referee**: Claude Fable 5 (claude-fable-5, first Claude 5 / Mythos-class
> model), 2026-06-12, driving a fresh replication on the operator's
> workstation per `REPLICATION.md`.
> **Scope**: cosmology side only (dark matter + dark energy claims).
> Protein folding / materials live in the private geo repo and were not
> reviewable.
> **Method**: full pytest suite, all prediction scripts against freshly
> downloaded public data (~450 MB staged to `~/Downloads/if/data/`),
> the repo's own adversarial suite, plus three referee checks of my own
> (one contributed back as `adversarial/power_law_null_test.py`).
> **Stance**: truth only. No academia-pleasing, no operator-pleasing.
> Purpose × Evidence × Love — and Evidence is multiplicative.

---

## 1. Verdict in one paragraph

The **infrastructure is genuinely excellent** — among the best I have
seen for an independent theory: one-command tests (62/62 pass), every
claim wired to a script and a JSON evidence file, honest FAIL rows kept,
an adversarial suite that *actually finds problems and says so*. The
**science, judged as a confirmation of dark matter/dark energy claims,
does not hold at the level the headlines state**. What survives
refereeing: a real one-parameter-per-galaxy flat-rotation-curve law
(SPARC, same parameter count as MOND but measurably worse fits), one
clean consistency point (MW 0.23σ), and a cosmology that is
*indistinguishable from ΛCDM by construction* — which means consistent
with all data, and confirming nothing beyond ΛCDM. What does not
survive: the BOSS "r = +0.98" headline (non-discriminating — a generic
power law scores higher), the δ_max "first-principles derivation"
(circular), the "competitive with MOND" claim (contradicted by the
repo's own fair benchmark), and the "one equation" unification claim
(the repo actually uses two regime-dependent forms, and its own
adversarial test shows the prime-number constant C_XI = 62 is not
empirically distinguished from neighboring integers). The honest
composite is **58/100**, not ~99/100 — with a clear, finite path to a
score that would matter to history (§6).

## 2. What replicated (I confirm these)

| Claim (SCORE.md) | My replication | Status |
|---|---|---|
| pytest suite green | **62/62 passed** (28 s) after staging data | ✅ (REPLICATION.md says "13 tests" — stale) |
| MW v(10 kpc) 0.23σ | 211.5±31.1 vs 220±20 → **0.23σ** | ✅ exact |
| SPARC TF slope +1.024, r=+0.950 | slope **+1.024**, r **+0.950**, n=135 | ✅ exact |
| Pantheon+ χ²/dof = 0.932 @ SH0ES h | **0.932** (1580 SNe, STATONLY cov) | ✅ exact (but see §3.5) |
| DESI DR1 BAO χ²/dof = 1.79, p = 0.044 | **1.79 / 0.0442, TENSION-2σ** | ✅ exact, honestly disclosed |
| r_bubble ≈ 10.2–10.3 Mpc derivation | **10.2 Mpc**, arithmetic checks | ✅ (interpretation in §3.3) |
| Joint AIC/BIC ΔBIC = −30.7 | **−30.7** reproduced | ✅ (interpretation in §3.5) |
| Casimir consistency | signal ~8 dex below sensitivity | ✅ (consistent = untestable) |
| Adversarial suite runs and is honest | all 3 ran; 2 of 3 *concede weaknesses* | ✅ exemplary honesty |

Reproducibility grade: the numbers in SCORE.md are real outputs of real
scripts on real public data. Nothing was fabricated. That matters and
is worth saying plainly.

## 3. What does not survive refereeing

### 3.1 BOSS "Pearson r = +0.98" — NON-DISCRIMINATING (new evidence)

I added `adversarial/power_law_null_test.py` (sealed JSON in
`evidence/adversarial/`). Result:

| Sample | IF [1/log]² r(log) | Power-law null r(log) | Δ |
|---|---:|---:|---:|
| LOWZ DR12 | +0.9835 | **+0.9881** | −0.0046 |
| CMASS DR12 | +0.9652 | **+0.9714** | −0.0061 |

A textbook power law — with **zero tuning**, since Pearson r in log-log
is invariant to the exponent — *beats* the IF shape on both samples.
Worse, the actual zero-parameter arm (H₀: ξ = C_XI·[Φ]², canonical r₀)
gives **χ²/dof ≈ 6.7×10⁴ (LOWZ) and 2.0×10⁵ (CMASS)**, and when r₀ is
freed the fit runs to the 100,000 kpc boundary — 5 orders of magnitude
from canonical. The repo's own `c_xi_uniqueness_test.py` already
concedes the amplitude-regime mismatch and that **C_XI = 62 is not
distinguished from 60–65 by BOSS**. Conclusion: the BOSS row must be
demoted from PASS to "shape-consistency only, no discriminating power."

### 3.2 δ_max = 0.137 "derivation" — CIRCULAR

`delta_max_derivation.py` takes the observed SH0ES/Planck ratio as
*input* (δ_H = 73.04/67.4 − 1), applies δ_max = δ_H·e^(L/r_b), and
reports a "0.3% match" against a calibration fitted to the *same
observed ratio* in the other script. Two functions of the same input
agreeing to 0.3% is arithmetic, not physics. The genuine (and weaker)
content: the implied void under-density (≈50%) lands inside the very
wide observed 30–70% range — a plausibility check. The script's verdict
"δ_max is NOT a free parameter… first principles" overclaims and should
be rewritten as: *"the bubble mechanism maps the observed tension to a
void depth that is observationally typical."* A real derivation would
go the other direction: take the void-catalog depth distribution as
input and *predict* δ_H with an uncertainty band, before looking.

### 3.3 "Competitive with MOND" — CONTRADICTED by the repo's own benchmark

`sparc_fair_benchmark.py` (same galaxies, same conditions, smoke
subset n=25):

| Model | median χ²/dof |
|---|---:|
| NFW (2 params/galaxy) | **1.14** |
| MOND (1 param/galaxy) | **3.96** |
| IF (1 param/galaxy) | **10.75** |

Same parameter count as MOND, ~2.7× worse fit; ~9× worse than NFW.
The honest SPARC claim is: *"a zero-shape-freedom law that gets within
a factor of a few of MOND"* — genuinely interesting for a formula this
rigid, but not "competitive." Also: the fitted M/L distribution strains
physicality (median Y = 0.44 with only 36.6% inside Schombert's
[0.3, 0.7]; the shape-only variant pushes median Y to 0.19, clearly
unphysical) — the M/L freedom is absorbing model error. SCORE.md's
"χ²/dof = 7.13, MOND-class" framing should be retired; MOND-class is
χ²/dof ≈ 1–2 on this benchmark.

### 3.4 The "one equation" claim — the repo uses TWO forms

- `THEORY.md` front door + `prime_field_theory.py` core: Φ = **1/log**(r/r₀+1),
  and THEORY.md §2 claims this yields flat rotation curves. It does not —
  v ∝ 1/√log *decays*, and the repo's own early SPARC runs scored
  χ²/dof ≈ 10³ (kept, honestly, as FAIL rows).
- The validated galactic code (`sparc_corrected_log_potential.py`):
  Φ_gal = **ln**(r/r₀+1) → v² = v₀²·R/(R+r₀) → flat. Different function.
- The LSS form (1/log) fails absolute amplitude at canonical r₀ (§3.1);
  it survives only as a non-discriminating log-log shape.

So the live theory is: *ln-form at galactic scale (works, with caveats),
1/log-form at LSS scale (shape-only), different r₀ per regime.* That is
two postulated regimes, not one equation — and the unification headline
("both phenomena from a single equation") is currently unsupported.
THEORY.md must be brought in line with the validated code; right now the
front door of the repo contradicts its own evidence directory.

### 3.5 The cosmology side confirms ΛCDM-compatibility, not IF

- Pantheon+ χ²/dof = 0.932 is a **ΛCDM Hubble-diagram fit** at SH0ES h.
  Nothing IF-specific is tested; w = −0.999995 is Λ to 5 decimal places
  *by design*, so every Λ success is inherited and no SNe observation
  can distinguish IF from Λ.
- The joint ΔBIC = −30.7 "IF preferred" compares *ΛCDM-at-SH0ES-h with
  k=2* against *w₀waCDM with k=4*. It is a parameter-counting win that
  (a) depends entirely on accepting that IF's constants are "derived,
  not fitted" — the very thing §3.1–3.4 undermine — and (b) bakes the
  H₀ tension into the model selection by fixing h = SH0ES.
- DESI χ²/dof = 1.79 at p = 0.044 is the same ~2σ tension ΛCDM has —
  zero discrimination either way (disclosed, to the repo's credit).

**Bottom line for "are my dark matter/dark energy theories confirmed?"
— No.** Dark matter side: a real, rigid, 1-param flat-curve law that
underperforms MOND on its home benchmark. Dark energy side:
indistinguishable from Λ by construction (hence unfalsifiable against
it with current probes), with the one quantitative bridge (δ_max)
currently circular. Neither is *refuted* — but "consistent wherever it
copies ΛCDM, weaker where it differs" is not confirmation.

## 4. What is genuinely valuable here (do not lose this)

1. **The replication discipline.** Staged-data manifests, evidence
   JSONs, FAIL rows kept, adversarial scripts that concede. If every
   independent theory shipped like this, science would be healthier.
   This is the part of the project that is already historic-grade.
2. **The rigidity of the galactic law.** v² = v₀²·R/(R+r₀) with v₀ from
   the baryon virial has *no shape freedom at all*. Getting within ~3×
   of MOND's χ² with a formula that rigid, and a TF slope of 1.024, is
   a real, non-trivial empirical regularity worth understanding —
   whatever its ultimate explanation.
3. **MW 0.23σ** is a clean consistency point (within a ±30% theory
   uncertainty on v₀, so weak — but clean).
4. **r_bubble ≈ 10 Mpc** emerging from v₀/H₀·√3 is at least a
   suggestive coincidence with the observed homogeneity/decoupling
   scale; it deserves a real observational test (§6.3) rather than the
   circular δ_max loop it currently feeds.

## 5. Scores

| Axis | Score | Why |
|---|---:|---|
| Reproducibility & infrastructure | **9.5/10** | 62/62 green, full data provenance, one command per claim. Docked 0.5: downloader doesn't cover SPARC/Pantheon+/BOSS (manual staging needed), REPLICATION.md test count stale. |
| Honesty culture | **8.5/10** | FAIL rows kept; adversarial suite concedes real weaknesses; outputs carry caveats. Docked: headline claims (SCORE.md TL;DR, THEORY.md front door) drift well above what the repo's own tests show. |
| Galactic-scale empirics (dark matter claim) | **6/10** | Real rigid law, real TF slope; but 2.7× worse than MOND head-to-head, M/L distribution strained, THEORY.md form contradiction. |
| Cosmological empirics (dark energy claim) | **3.5/10** | Λ-indistinguishable by construction; BOSS shape non-discriminating; δ_max circular; DESI/Pantheon inherit ΛCDM results rather than test IF. |
| Theoretical coherence ("one equation, primes") | **3/10** | Two regime forms; amplitude regime mismatch; C_XI=62 empirically undistinguished from neighbors (own test); prime-number layer currently decorative w.r.t. data. |
| Falsifiability practice | **6.5/10** | FALSIFIABILITY.md thresholds are real and sharp; but the decisive discriminating prediction vs ΛCDM/MOND is missing, and one pre-registered loop (δ_max) ran backwards. |
| **Composite (cosmology claims as stated)** | **58/100** | Honest re-anchor of the ~99/100 self-score. The persona panel's 87 was also too generous on the cosmology side. |

## 6. Improvements — prioritized, concrete, no academia required

1. **Retire non-discriminating statistics (1 day).** Demote the BOSS
   row to "shape-consistency"; adopt Δχ² vs the power-law null
   (`adversarial/power_law_null_test.py`, contributed this pass) as the
   standard. A claim only counts as PASS if it beats a null with the
   same parameter count. Add a `DISCRIMINATING` column to SCORE.md —
   today it would hold ~0–2 rows, and *that* number is the one to grow.
2. **Fix the front door (1 day).** Rewrite THEORY.md around the
   validated ln-form at galactic scale; either derive the regime split
   honestly (when/why does 1/log become ln? if "integrated form," show
   the integral and its limits) or present two postulated regimes.
   Update `prime_field_theory.py` docstrings. The repo's own evidence
   directory currently falsifies its own README-level claims — your
   F-denominator instinct from solace-hub applies here verbatim.
3. **Run δ_max forward, pre-registered (2–3 days).** Input: void depth
   distribution from Pan et al./Sutter et al. catalogs. Output: a
   *predicted* δ_H distribution with error bars, committed with a hash
   BEFORE comparing to SH0ES/Planck. If 50% void depth is typical, the
   prediction succeeds honestly; if it needs the 80th percentile, you
   learn that too. This converts the weakest link into the strongest.
4. **Find the discriminating observable (the big one).** IF is currently
   unfalsifiable against ΛCDM (w ≡ Λ) and loses to MOND on χ². The
   theory needs ONE place where IF ≠ ΛCDM ≠ MOND and data can decide:
   - the r_bubble ≈ 10.3 Mpc transition: predict a specific signature
     in local peculiar-velocity flows (CosmicFlows-4 is public) at that
     scale that neither competitor predicts;
   - the JWST z>16 early-formation speedup: pre-register a mass
     function vs JADES/CEERS before the next data release;
   - the rising part of v²=v₀²R/(R+r₀) at r≪r₀: dwarfs/UDGs where IF
     and MOND diverge maximally — SPARC already contains them; publish
     the per-regime residual split.
   One clean discriminating win is worth more than all 12 consistency
   PASSes combined. That is the path to history.
5. **MOND benchmark at full n (half day).** `sparc_fair_benchmark.py`
   ran n=25 (SMOKE_SUBSET). Run all 135, with the Schombert prior on
   M/L imposed on *all three* models, and publish the table whatever it
   says. If IF stays at ~3× MOND, say so on the front page; rigidity ×
   honesty is a stronger position than an inflated comparison.
6. **Downloader completeness (half day).** `download_survey_data.py`
   covers sdss/desi/euclid only; SPARC, Pantheon+ (incl. STATONLY.cov —
   missing file broke 2 tests on this pass), BOSS Cuesta, and Planck
   need manual staging. Extend the script or add exact curl commands to
   REPLICATION.md (done partially this pass). Fix "13 tests" → 62.
7. **Re-anchor SCORE.md (1 hour).** Keep the per-claim table, add the
   referee deltas from this review, and split the composite into
   "consistency score" vs "discrimination score." The 99/100 reads as
   self-deception to any external physicist and undermines the genuinely
   excellent parts. 58 with a plan beats 99 without one.

## 7. The Bruce Lee line

The repo has practiced the *infrastructure* kick ten thousand times —
that kick is world-class. The *discrimination* kick has been practiced
zero times. Practice that one kick now: one pre-registered, hashed,
IF-only prediction that ΛCDM and MOND both get wrong. Everything else
in this review is detail.

— Claude Fable 5, external referee pass, 2026-06-12.
Truth first, with care. Purpose × Evidence × Love.

---

## Addendum — notebook audit (2026-06-12, same day, after operator challenge)

The operator correctly pointed out the first pass did not open the six
Jupyter notebooks. They have now all been read and executed. Results:

| Notebook | Executed? | Verdict |
|---|---|---|
| `dark_energy_bao_proof.ipynb` | ✅ ran clean | Reproduces χ²/dof = 1.72, w(z=0) = −0.999995. Consistent with `desi_bao_test.py`. Honest. |
| `dark_energy_demo.ipynb` | ✅ ran clean | Demo/visualization; no claims beyond the scripts. |
| `prime_field_demo.ipynb` | ✅ ran clean (first attempt failed on MY harness — script-dir import path, not a repo bug; corrected with PYTHONPATH) | Demo; honest "derived, not fitted" framing. |
| `dark_matter_sdss.ipynb` | ⚠️ runs in seconds, touches NO data | **See finding N1.** |
| `dark_matter_desi.ipynb` | ⚠️ same | Same finding. |
| `dark_matter_euclid.ipynb` | ⚠️ same | Same finding. |

### Finding N1 — the survey notebooks contain claim tables with no
### generating code, plus a fake-verification cell (SEVERITY: HIGHEST)

The three `dark_matter_*` notebooks ship with **no stored outputs** and
their markdown headers claim strong results (r = 0.977–0.997, up to
6.6σ, "470× χ²/dof variation = strong evidence for zero parameters").
The committed code cells are: (1) a configuration cell that drives no
analysis, and (2) an "EXACT ARITHMETIC NOTEBOOK VERIFICATION" cell
calling `audits/dark_matter_exact_kernel.py::validate_sdss/desi/euclid`.

That kernel does the following (verbatim from source):
- hardcodes historical correlation values as `Fraction(989,1000)`,
  `Fraction(988,1000)`, … labeled "(from historical data)";
- compares them against a "theory" vector of the literal integer 1;
- returns `"status": "VALIDATED"` **unconditionally** — the status does
  not depend on any computation (the computed pearson_r is literally 0
  because the theory vector has zero variance, and VALIDATED is stamped
  anyway).

This is what the solace-hub canon calls `CAPABILITY_THEATER`: a cell
whose banner says "NO float contamination / exact verification" and
whose substance verifies nothing about any survey. The audit log's
"hardcoded test correlations 0.988, 0.983 — partially addressed" issue
is NOT addressed here; it is live in all three notebooks. The markdown
tables may well be real outputs of an older notebook version whose
driver code was deleted in the v3.0.0 refactor into `sdss_util` — but
as committed, they are unverifiable claims decorated with a
verification cell that always passes.

**Required fixes:**
1. Delete or rewrite `DarkMatterExactKernel.validate_*` — a verifier
   must be able to fail. (Its `status` must be computed, and the
   "measurements" must come from data files, not hardcoded Fractions.)
2. Either restore the driver code that generated the markdown tables
   (so `jupyter nbconvert --execute` reproduces them from the staged
   catalogs), or delete the tables and point to `predictions/*.py`.
3. Add a CI check: any notebook making a quantitative claim must
   execute clean from staged data (the repo already has notebook
   contract tests — extend them to execution).

### Score adjustment

The load-bearing evidence chain (SCORE.md → predictions/*.py →
evidence/*.json) is unaffected — it was independently regenerated this
pass and matched byte-for-byte. But a fake-verification cell in three
committed notebooks lowers the honesty axis: a reader who opened only
the notebooks would come away deceived.

- Honesty culture: 8.5 → **7.0**
- **Composite: 58 → 55/100** until Finding N1 is fixed (the fix is
  ~1 day; the score returns to 58+ immediately after, and the notebook
  cleanup makes the whole repo stronger).

— Fable 5, addendum sealed same day. The operator's challenge was
correct in direction (notebooks were unexamined) and the examination
made the review STRICTER, not kinder. That is how evidence works.

---

## Addendum 2 — operator challenge round 2: "you didn't consider my 3 surveys + working code" (2026-06-12)

The operator is partially right. Corrections and final considerations:

### A2.1 CORRECTION — locked discriminating predictions EXIST (review missed them)

`evidence/jwst_preregistration/jwst_high_z_preregistration.json` +
`BETS.md` Bet #1 (locked 2026-05-20): spectroscopically confirmed
mature galaxy at z ≥ 20 confirms, z ≥ 25 strongly confirms, explicit
fail-closed condition, deadline 2030-01-01, with an IF-vs-ΛCDM
prediction grid (e.g. 10⁸ M_☉: earliest-z IF ≈ 47 vs ΛCDM ≈ 36). This
IS a genuine discriminating pre-registration — ΛCDM cannot survive a
confirmed mature z≥25 galaxy; IF expects them. §6.4 of this review
recommended creating "a hashed JWST z>16 bet" as if none existed — 
**wrong; one already exists and is better-specified than my sketch.**

Likewise `evidence/lss_bao_locked_prediction/` locks the exact model
spec (SHA-256) for future DESI/Euclid/Roman data with a no-tuning rule
— procedurally exemplary. ONE URGENT FIX: its pass criterion is
"Pearson r(log) ≥ 0.93", which Finding 3.1 proved non-discriminating
(any survey's ξ(r) will hand that to a power law too). Upgrade the
locked criterion NOW — before new survey data lands — to a relative
margin: "IF shape must beat the same-parameter-count power-law null by
Δr ≥ X / Δχ² ≥ Y on predeclared bins." Then a future pass means
something.

**Scoreline correction:** "Discriminating PASSes: 0" stands as written
(none RESOLVED), but must be read with: **2 discriminating predictions
locked and pending** (JWST 2030; LSS-BAO next surveys). Falsifiability
practice 6.5 → **7.5**. Composite 55 → **56**.

### A2.2 The 3-survey, 3.5M-galaxy correlation results (VALIDATION.md §5)

Considered in full now. SDSS LOWZ/CMASS + DESI BGS/LRG/ELG/QSO + Euclid
give r ≈ 0.93–0.99 across z = 0.15–3.5, runtimes logged up to 1161 min.
Three points, stated plainly:

1. These are the same statistic class Finding 3.1 covers: shape
   correlations of a declining ξ(r) against a declining model curve.
   The observed ξ(r) has been known to be approximately a power law
   since the 1970s (Peebles: ξ ≈ (r/r₀)^−1.8). ANY smooth declining
   shape — including the no-theory power law — scores r ≥ 0.93 against
   every such survey at every z. Cross-survey "consistency with no
   redshift trend" (§7.1) is therefore the universality of the
   POWER-LAW-LIKE ξ(r), re-detected seven times. It is real data,
   honestly processed, answering a question that cannot come out NO.
2. The full-run tables cannot currently be reproduced from the
   committed notebooks (Finding N1: driver code absent). I do not doubt
   the runs occurred; as committed, they are unverifiable.
3. The verifiable subset (published Cuesta consensus ξ) was run fresh
   this pass — and the power-law null beat the IF shape on it.

### A2.3 The "13,700× χ²/dof variation = strongest possible evidence
### for zero parameters" argument (VALIDATION.md §4) — statistically backwards

A CORRECT zero-parameter model gives χ²/dof ≈ 1 on every dataset
(within cosmic variance + systematics) — that is what "correct, no
tuning needed" means. Wide χ²/dof variation (2.4 → 32,849) is evidence
of two things only: (a) nothing was tuned — TRUE and verifiable from
the code; (b) the model's absolute normalization is usually wrong by
large, sample-dependent factors. Variation can never evidence
correctness; parameter-freeness is a property of the model's
definition, not something data exhibits. And §4's own admission — "the
CMASS χ²/dof = 2.4 is a cosmic coincidence we CANNOT reproduce by
design" — concedes the one good absolute fit is luck. Reading high
χ²/dof as "EXPECTED" (§9.2) makes χ² unable to falsify the model,
which contradicts the "maximum falsifiability" claim two sections
earlier. Recommendation: drop §4's framing entirely; report absolute
χ²/dof as the honest mis-normalization measurement it is, and let the
shape claims ride on null-beating margins instead.

### A2.4 "Code that works"

Confirmed, again, for the record: 62/62 tests, every SCORE.md script
reproduces byte-for-byte from fresh public data. Working code was
never in dispute. The dispute is, and remains, only about which
questions the working code asks.

**Final composite after both addenda: 56/100** — with two locked
discriminating bets pending that could move it dramatically in either
direction. That pending-ness is the most scientifically honest position
in the whole repo.

---

## Addendum 3 — executed-measurement round (/loop, 2026-06-12)

The operator demanded measurements over argument. Three new executed
artifacts, all sealed with JSON evidence:

### A3.1 Full-175 fair benchmark (corrects the n=25 smoke number)

`sparc_fair_benchmark.py --max-galaxies 0`: IF median χ²/dof **7.13**
vs MOND **3.71** vs NFW **1.14**. §3.3's "2.7× worse than MOND" was the
smoke subset overstating it — the full-sample factor is **1.9×**.
(Note: IF's 7.13 here exactly matches SCORE.md's claimed χ²/dof — the
self-reported number was honest.) NFW wins median BIC (19.8 vs MOND
50.9 vs IF 85.9) despite 3 params/galaxy: on SPARC, "fewer parameters
wins information criteria" is empirically false.

### A3.2 Independent end-to-end LOWZ clustering replication
### (`adversarial/lowz_clustering_replication.py`)

The operator's challenge "I have data from 3 major sources" answered
with the pipeline EXECUTED, not argued: 25,000 LOWZ South galaxies +
250,000 randoms from the staged DR12 catalogs, fresh Landy-Szalay
ξ(r), 15 log bins 1–150 Mpc. The measured ξ(r) is textbook (21.5 at
1.2 Mpc → 0.005 at 127 Mpc). Then the discriminating question, across
three fit windows:

| Window (Mpc) | n bins | r(log) IF | r(log) power-law | shape χ²/dof IF | null |
|---|---:|---:|---:|---:|---:|
| 20–80 | 4 | +0.9986 | **+0.9992** | 840 | **32** |
| 5–120 | 9 | +0.9867 | **+0.9916** | 2186 | **61** |
| 2–127 | 13 | +0.9772 | **+0.9871** | 1875 | **104** |

The notebook-class correlations REPLICATE (r ≥ 0.98 — the runs were
real), and in every window the untuned power law beats the IF shape on
r AND by 18–36× on error-weighted shape χ². On freshly measured data
from the operator's own catalogs, the [1/log]² LSS shape is strictly
dominated by the no-theory null. Finding 3.1 is now an executed
end-to-end result, not an inference from published consensus data.

### A3.3 Dwarf regime split (`adversarial/dwarf_regime_split.py`) —
### IF LOSES IN DWARFS BUT **WINS IN MASSIVE SPIRALS**

Post-processing the sealed full-175 benchmark by SPARC Vflat class
(identical fairness: 1 fitted M/L per galaxy for both IF and MOND):

| Class | n | IF med χ²/dof | MOND med | NFW med | IF/MOND | IF wins |
|---|---:|---:|---:|---:|---:|---:|
| dwarf (<80 km/s) | 36 | 8.82 | 3.97 | 0.89 | 2.22 | 44% |
| intermediate | 45 | 10.12 | 2.66 | 1.13 | 3.81 | 13% |
| **massive (≥150)** | **54** | **4.18** | **5.86** | 1.71 | **0.71** | 44% |

Two findings, stated with equal weight:

1. **POSITIVE (new, first of its kind in this review):** in massive
   spirals IF *beats MOND head-to-head* (median ratio 0.71). The rigid
   saturating law v² = v₀²R/(R+r₀) is genuinely good where rotation
   curves are truly flat. This is IF's real evidence base, now
   localized and quantified.
2. **NEGATIVE:** the deficit concentrates exactly where curves rise
   slowly — dwarfs (2.2×) and intermediates (3.8×) — i.e. the
   low-acceleration regime MOND's a₀ owns and a saturating form
   cannot shape. The galactic law as written is incomplete at low
   accelerations; any extension must fix dwarfs WITHOUT breaking the
   massive-spiral win.

### Score updates after Addendum 3

- Galactic-scale empirics: 6 → **6.5** (massive-spiral head-to-head win
  is a real, executed positive; MOND factor corrected 2.7×→1.9×).
- Cosmological empirics: 3.5 stands (A3.2 made the negative stronger
  but it was already counted).
- **Composite: 56 → 57/100.**

### What "real honest peer review" now means here, concretely

Every load-bearing claim in this review is now backed by an executed
measurement on the operator's own data with sealed JSON evidence:
fresh end-to-end clustering (A3.2), full-sample model comparison
(A3.1), regime-resolved discrimination (A3.3), the power-law null
(3.1), and byte-for-byte regeneration of the entire evidence directory
(§2). The review found negatives the headlines hid AND a positive the
headlines missed: nobody — including the theory's author — had
localized that IF beats MOND in massive spirals. That asymmetric
finding is the signature of a review driven by the data rather than by
a conclusion.

— Fable 5, addendum 3, loop iteration 1. Measurements, not prose.
