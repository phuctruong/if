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
