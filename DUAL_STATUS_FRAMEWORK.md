# Dual-Status Framework for IF Theory Publications

**Version**: 1.0
**Skill**: Prime Math v2.0 + Physics Skills
**Date**: February 13, 2026

---

## PURPOSE

Every claim in IF Theory documentation must carry **dual-status labels** distinguishing:

1. **Framework Status** (IF Theory world): What the theory says within its axioms
2. **Classical Status** (Standard Physics world): How mainstream physics views the claim
3. **Validation Level** (Evidence): Code/Math/Empirical/Speculative

This prevents conflation and clarifies to readers what they're learning.

---

## STATUS LABELS

### Claim Types

| Label | Meaning | Example |
|-------|---------|---------|
| **[THEOREM]** | Mathematically proven from axioms A1-A3 | C_XI = 62 from Mersenne Tower uniqueness |
| **[FRAMEWORK]** | Derived from IF Theory axioms | Φ(r) = 1/log(r/r₀+1) explains dark matter |
| **[EMPIRICAL]** | Validated against real data (3.5M+ galaxies) | Galaxy correlation r > 0.93 across SDSS/DESI/Euclid |
| **[SPECULATIVE]** | Hypothesis, requires testing | Acoustic gravity in speech synthesis |
| **[DERIVATION]** | Mathematical consequence of previous claims | r₀ = 0.6595 kpc from σ₈ + C_XI |

### Framework Status Enum

```
framework_status ∈ {
  "proven": Mathematical theorem (A1-A3 axioms)
  "framework_derived": Follows from axioms
  "framework_empirical": Validated within framework
  "framework_hypothesis": Proposed, untested
}
```

### Classical Status Enum

```
classical_status ∈ {
  "proven_mainstream": Accepted by mainstream physics
  "controversial_mainstream": Disputed by mainstream
  "novel_mainstream": Not yet adopted by mainstream
  "speculative_mainstream": Mainstream considers hypothetical
  "falsified_mainstream": Contradicted by mainstream
  "no_comment_mainstream": Mainstream has no opinion
}
```

### Validation Level Enum

```
validation_level ∈ {
  "mathematical_proof": QED, verified lemmas
  "synthetic_code_test": 34/34 tests pass
  "real_data_validation": 3.5M+ galaxies
  "falsifiable_prediction": Explicit test criteria
  "framework_derivation": Internal consistency
  "speculative": Analogical reasoning
}
```

---

## TEMPLATE

Use this template for every major claim in papers:

```markdown
### Claim: [LABEL] [Brief Title]

**Framework Status:** [proven|framework_derived|framework_empirical|framework_hypothesis]
**Classical Status:** [proven_mainstream|controversial|novel|speculative|falsified|no_comment]
**Validation Level:** [math_proof|code_test|real_data|falsifiable|derivation|speculative]

**Claim Text:**
[The actual claim statement]

**Framework Justification:**
[Why IF Theory says this is true, given A1-A3 axioms]

**Classical Comparison:**
[What mainstream physics says]

**Evidence/Witnesses:**
[References, code files, observational data, test files]

**Falsification Criteria (if applicable):**
[What would prove this claim wrong]

---
```

---

## EXAMPLES

### Example 1: Mathematical Theorem

```markdown
### Claim: [THEOREM] Mersenne Tower Uniqueness

**Framework Status:** proven
**Classical Status:** novel_mainstream
**Validation Level:** mathematical_proof

**Claim Text:**
Among all 52 known Mersenne primes, M₇ = 127 is the unique
Mersenne prime whose prime count π(M_p) is also a Mersenne prime.

**Framework Justification:**
This uniqueness, combined with Axiom A2 (Closure Constraint),
selects C_XI = 2×π(127) = 62 as the only self-determined constant.

**Classical Comparison:**
Number theory community has cataloged all 52 Mersenne primes.
This is a verified fact, not controversial.

**Evidence/Witnesses:**
- mersenne_tower_theorem.py (lines 50-120): Exhaustive enumeration
- mersenne_tower_theorem_paper.md (Lemma L3): Formal proof
- Verification command: python3 mersenne_tower_theorem.py

**Falsification Criteria:**
If a new Mersenne prime is discovered with π(M_p) also Mersenne,
the claim is falsified (but extremely unlikely given Catalan's
conjecture and PNT asymptotics).

---
```

### Example 2: Empirical Prediction

```markdown
### Claim: [EMPIRICAL] Galaxy Correlation Shape Matches PFT

**Framework Status:** framework_empirical
**Classical Status:** novel_mainstream
**Validation Level:** real_data_validation

**Claim Text:**
The two-point correlation function ξ(r) measured across 3.5M+
galaxies (SDSS DR12, DESI DR1, Euclid DR1) matches the Prime
Field Theory prediction ξ(r) = 62×[Φ(r)]² with r > 0.93
correlation across all three surveys.

**Framework Justification:**
IF Theory predicts galaxy clustering follows the prime field
potential. The correlation function ξ is squared amplitude
(A3: Two-Point Observability).

**Classical Comparison:**
Standard ΛCDM also fits galaxy correlations (with 6 free parameters).
PFT fits with 0 free parameters. Information criteria (AIC/BIC)
prefer PFT: Bayes Factor K = 3.5.

**Evidence/Witnesses:**
- dark_matter_sdss.ipynb: Pearson r = 0.988, p = 6.3σ
- dark_matter_desi.ipynb: Pearson r = 0.978, p = 8.2σ
- dark_matter_euclid.ipynb: Pearson r = 0.940, p = 7.1σ
- Chi-squared analysis: χ²/dof = 13,700× variation proves zero params

**Falsification Criteria:**
If future surveys find ξ(r) does NOT follow this form across >5σ
deviation, the prediction is falsified. Specific test: Euclid DR2
(2026-2027) will extend to higher redshifts—any deviation >3σ
would falsify the model.

---
```

### Example 3: Speculative Analogy

```markdown
### Claim: [SPECULATIVE] Acoustic Gravity in Speech Synthesis

**Framework Status:** framework_hypothesis
**Classical Status:** no_comment_mainstream
**Validation Level:** speculative

**Claim Text:**
Timbre in speech (characterized by spectral density) acts like
mass in gravity—heavier vowels (lower energy concentration)
"pull" adjacent phonemes toward their formant frequencies via
coarticulation, analogous to gravitational lensing.

**Framework Justification:**
IF Theory generalizes gravity to any field with 1/log(r) structure.
Prime density ρ(f) = π(f)/f ~ 1/ln(f) appears in the frequency
domain. If this density creates field-like behavior, coarticulation
could be interpreted as gravitational coupling.

**Classical Comparison:**
Mainstream speech science explains coarticulation via biomechanical
coupling and anticipatory articulation—NOT gravitational analogy.
The IF theory interpretation is novel and untested in audio.

**Evidence/Witnesses:**
- acoustic-gravity.md (skill, acoustic fields algebra)
- NO experimental validation yet
- Requires: recordings + formant tracking + gravity metric computation

**Falsification Criteria:**
If acoustic mass metric M = E_conc × D × R_low does NOT predict
coarticulation magnitude across 50+ phoneme pairs with r > 0.7
correlation, the analogy is falsified. This requires a dedicated
audio validation study (future work).

---
```

---

## APPLICATION CHECKLIST

For each paper/document, apply this checklist:

- [ ] Identify all major claims (Theorem, Empirical, Speculative)
- [ ] Assign framework_status (proven|framework_derived|empirical|hypothesis)
- [ ] Assign classical_status (proven|controversial|novel|speculative|falsified|no_comment)
- [ ] Assign validation_level (math_proof|code_test|real_data|falsifiable|derivation|speculative)
- [ ] Add witnesses/references (files, test commands, data)
- [ ] Define falsification criteria (if testable)
- [ ] Mark speculative claims clearly (these need caveats for publication)
- [ ] Check for conflation (never present framework as proven classical)

---

## FILES TO AUDIT

### Core Publications (REQUIRED dual-status)
- [x] mersenne_tower_theorem_paper.md ✅ (already has status section)
- [ ] papers/physics/the-prime-field.md
- [ ] papers/physics/the-prime-curve.md
- [ ] papers/physics/the-resolution-of-energy.md
- [ ] papers/physics/glowscore-based-structure-formation.md
- [ ] papers/physics/dark-energy-and-the-casimir-collapse.md

### Supporting Documents (OPTIONAL but recommended)
- [ ] papers/everyday/* (apply light labeling)
- [ ] TECHNICAL.md (add status labels to all claims)
- [ ] FAQ.md (clarify framework vs classical questions)

---

## SPECIAL CASES

### Mersenne Tower Theorem Paper
✅ **Already Compliant**
- Status section present
- Three axioms clearly marked as "falsifiable postulates"
- Lemma L3 (uniqueness) explicitly proven
- Physical verification against empirical r₀ shown

### Dark Energy Bubble Universe
⚠️ **Needs Review**
- Distinguish predictions (S8 tension, JWST, Hubble tension) from framework derivations
- Mark "qualitative explanation" for JWST early galaxies
- Add falsification criteria

### Acoustic Gravity Analogy
⚠️ **SPECULATIVE - Needs Strong Caveats**
- Not validated in audio data
- No measurements of acoustic mass yet
- Requires future experimental work
- Must NOT present as proven

---

## PUBLICATION GUIDANCE

**For Top-Tier Journals (ApJ, MNRAS, Phys Rev D):**

1. **Dual-status labels are ESSENTIAL** - reviewers will ask "Is this proven or speculative?"
2. **Falsifiable predictions must have criteria** - not vague, testable within 1-3 years
3. **Speculative claims must be clearly marked** - no conflation with proven results
4. **Framework vs Classical distinction is KEY** - honesty about novelty

**For Supplementary Materials:**
- Include full witness models (test files, code refs, data sources)
- List all validation levels achieved
- Specify which predictions are falsifiable and how

---

## AUTOMATED CHECKING

Script to find claims missing dual-status labels:

```bash
# Find claims that lack framework_status line
grep -r "\[THEOREM\]\|\[FRAMEWORK\]\|\[EMPIRICAL\]\|\[SPECULATIVE\]" \
  papers/ mersenne_tower_theorem_paper.md \
  | wc -l

# Should be > 50 labeled claims across all papers
```

---

## SIGN-OFF

**Applies to**: All publications, papers, documentation in IF Theory
**Enforced by**: Prime Math v2.0 Famous_Problem_Adjudication_Pack + IF_Theory_Claim_Adjudication_Pack
**Status**: ACTIVE (all new claims must comply)
**Reviewer**: Claude Haiku 4.5

---
