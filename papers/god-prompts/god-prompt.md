# Information-Theoretic World-Model Adjudication (Fail-Closed, Receipt-Driven)

**Session reset.** Assume no prior context. Be neutral. Do not assume God is true or false.
**Objective:** Compare candidate world-models and report which one **best compresses the evidence** with the **fewest extra assumptions**, under **explicit priors**, **explicit metaphysical bridges**, and **explicit underdetermination handling**.

---

## 0) Hard Constraints (Fail-Closed)

1. **No Rhetoric Rule**
   No persuasion, preaching, mockery, or vibes. Use technical language only.

2. **No Ontology Leakage Rule**
   Do not import metaphysical conclusions from mathematical/physical statements unless you explicitly add bridging premises and label them **METAPHYSICAL**.

3. **No Free Priors Rule**
   Any prior preference must be stated as a principle and sensitivity-tested.

4. **Underdetermination Rule**
   If multiple hypotheses yield indistinguishable empirical predictions for the evidence provided, state: **“Evidence does not discriminate.”**

5. **Receipts Rule**
   Every nontrivial claim must include a brief **witness** (definition, theorem, measurement class, or reasoning step).
   If you rely on a contingent empirical fact not in the Evidence Bundle, label it **UNVERIFIED** (or cite sources if browsing is allowed).

6. **Two-Pass Requirement (Mandatory)**
   You must run **both** passes and report both outcomes:

* **PASS 1 (Science-only):** Use only E1–E6 and non-metaphysical methodological priors (e.g., MDL as a heuristic). **No metaphysical bridges**.
* **PASS 2 (Bridged):** Use E1–E6 **plus** the declared **METAPHYSICAL bridges** in §1.5, and the constrained form of H5 in §3.5.

If PASS 1 does not discriminate but PASS 2 does, you must explicitly state:
**“Discrimination is driven by METAPHYSICAL bridges/priors, not forced by E alone.”**

---

## 1) Evidence Typing Protocol

For each evidence item, label it exactly one of:

* **THEOREM** (deductive result within formal systems)
* **EMPIRICAL OBSERVATION** (repeatable measurement class)
* **MODEL** (formal mapping from data → predictions)
* **INTERPRETATION** (ontological reading of a model)
* **INFERENCE** (explanatory comparison)
* **METAPHYSICAL** (ontological commitment not forced by data)

If an item mixes types, split into sub-claims and type each sub-claim.

### 1.5) Declared METAPHYSICAL Bridges (used ONLY in PASS 2)

These are **optional** but if enabled must be used explicitly.

**M1 — Lawspace Typicality (METAPHYSICAL)**
Treat “candidate laws” as encodable descriptions; under a non-informative prior over encodings, **highly compressible** law-descriptions are **atypical**.

* **Witness requirement:** Provide a brief counting argument: among bitstrings of length N, the subset describable by programs of length ≪ N is exponentially small.

**M2 — Anti-Brute Termination (METAPHYSICAL, PSR-style)**
Penalize world-models whose ultimate explanation terminates in **brute** constants/laws unless forced.

* **Witness requirement:** Define the penalty as a prior preference (not a likelihood claim).

**M3 — Constraint Credit (Methodological, NOT metaphysical)**
Hypotheses that include a **specific, compact, non-ad-hoc selection rule** get higher likelihood credit than hypotheses that can fit any outcome post hoc.

* **Witness requirement:** State as an Occam/MDL-style constraint to avoid unconstrained explainers.

**M4 — Bridge Guard (METAPHYSICAL)**
You must not treat M1–M2 as “scientific facts.” They are explicit metaphysical commitments.

* **Witness:** A one-line statement: “M1/M2 are not entailed by E; they are additional assumptions.”

---

## 2) Evidence Bundle (E)

Use **only** the evidence below in PASS 1 and PASS 2.
Do not add new evidence unless placed into **Optional Evidence** and marked **UNVERIFIED**.

* **E1:** There are infinitely many primes.
* **E2:** Mathematical models achieve high predictive accuracy in physics.
* **E3:** The universe exhibits compressible regularities (laws, symmetries, stable constants).
* **E4:** Evolution by selection can generate high functional complexity.
* **E5:** Incompleteness theorems apply to sufficiently strong formal systems.
* **E6:** Information processing has thermodynamic costs (Landauer-style constraints).

### Optional Evidence (OFF by default)

If you use any of these, mark **UNVERIFIED** unless cited.

* **E7 (UNVERIFIED):** Physical constants appear to lie in narrow life/complexity-permitting ranges.
* **E8 (UNVERIFIED):** Objective normativity exists (irreducible “ought” facts).
* **E9 (UNVERIFIED):** Repeatable mind-first anomalies violate physical causal closure.

---

## 3) Hypothesis Set (H)

Evaluate the following hypotheses **as world-models**, not slogans. You may refine each into a minimal formal core, but you may not change their identity.

* **H1 — Physicalism with brute laws:** reality is physical; laws/facts terminate in brute givens.
* **H2 — Ontic structural realism:** reality is fundamentally mathematical/structural; objects derivative.
* **H3 — Simulation:** our observed physics is implemented by an external computing substrate/agent(s).
* **H4 — Self-organizing fixed-point cosmology:** lawlike structure emerges as a stable attractor/fixed point; no external designer required.
* **H5 — Intentional rational ground:** a non-derivative mind/agency is the ultimate ground of laws/order.

### 3.5) Constrained Form of H5 (used ONLY in PASS 2)

To avoid H5 being an unconstrained explainer, in PASS 2 you must evaluate:

**H5* (same identity as H5, but minimal constrained core):**
A non-derivative rational agency selects a law-description by a **compact selection rule** (e.g., “prefer minimal description length subject to consistency and stable structure”), rather than arbitrary choice.

* **Witness requirement:** Define “selection rule” as a mapping from candidate law-descriptions → chosen law-description, and explain how this can reduce residuals under M1/M2.

---

## 4) Scoring WITHOUT Fake Numbers (Structured Criteria)

For each Hᵢ, evaluate using qualitative tiers only: {LOW, MED, HIGH}. No 0–10 ratings.

* **C1 Ontological Cost:** How many primitive categories are posited?
* **C2 Explanatory Depth:** Does it explain why laws/regularities exist or assume them?
* **C3 Empirical Continuity:** Aligns with scientific practice without ad hoc patches?
* **C4 Regress Termination:** How does it stop “why these laws?” (brute, necessary, cyclic, agentive)
* **C5 Discriminability:** In principle, could future evidence shift likelihoods among Hᵢ?

---

## 5) Bayesian/MDL Frame (Mandatory)

You must provide **two** evaluations:

### (A) Bayesian Comparison (qualitative)

Use:
[
P(H\mid E)\propto P(E\mid H),P(H)
]

* State **priors explicitly** (ordinal: LOW/MED/HIGH) and list the principles generating them.
* For each evidence item Eⱼ, state whether it is **expected**, **neutral**, or **surprising** under Hᵢ, and why.
* Identify where underdetermination blocks inference: **“Evidence does not discriminate.”**

### (B) MDL/Compression Comparison (qualitative)

Compare hypotheses as “programs” that generate the evidence:

* **Model length:** extra machinery and primitives added by Hᵢ
* **Data fit:** does it reproduce E naturally without patches?
* **Residuals:** what remains unexplained and must be taken as primitive?

**Mandatory:** Explicitly state whether MDL is being used as:

* (i) a methodological heuristic (**allowed in PASS 1**) or
* (ii) an ontic claim about reality (**METAPHYSICAL; disallowed unless labeled**)

---

## 6) Steelman + Red-Team Requirement

For **each** hypothesis Hᵢ:

1. **Steelman:** strongest version in 3–6 bullet points.
2. **Red-Team:** strongest objections in 3–6 bullet points.
3. **Key Failure Mode:** one sentence describing the most likely way the hypothesis overreaches the evidence.

---

## 7) Output Format (Strict)

Return exactly these sections, **twice** (PASS 1 and PASS 2), with the same headings.

1. **Typed Evidence Table**
   A table: Evidence | Type | Witness | Notes

2. **Hypotheses Minimal Cores**
   H1–H5: each as 2–4 sentences with defined primitives
   (For PASS 2, include H5* explicitly.)

3. **Criteria Matrix (C1–C5)**
   A table: H | C1 | C2 | C3 | C4 | C5 | One-line rationale

4. **Bayesian Pass (A)**

* Priors (ordinal)
* Likelihood notes per Eⱼ
* Posterior ranking **with underdetermination flags**

5. **MDL Pass (B)**

* Model-length notes
* Residual notes
* Compression-based ranking

6. **Synthesis**

* What the evidence **does** and **does not** support
* Where metaphysical weighting dominates
* Top 3 discriminators: observations that would most change ranking

7. **Verdict (Fail-Closed)**
   Choose exactly one:

* **“Evidence favors H__ (tentatively)”**
* **“Evidence does not discriminate among {…}”**
* **“Insufficiently specified; cannot compare”**

Include a short explanation of the verdict and explicitly list which conclusions are **METAPHYSICAL** vs evidence-driven.

---

## 8) Guardrails Against Common Failure Patterns (Mandatory Self-Audit)

Before finalizing, run this checklist and correct if violated:

* Did I treat a theorem as physical ontology?
* Did I treat “math is effective” as evidence of intention without a bridge?
* Did I conflate “information has thermodynamic costs” with “mind is fundamental”?
* Did I let “explanatory satisfaction” replace discriminability?
* Did I sneak in extra evidence not in E1–E6?
* In PASS 2, did I clearly label the discrimination step as **METAPHYSICAL** if it depends on M1/M2?

---

## 9) Adjudication Rule (Fail-Closed, prevents evasions)

* If PASS 1 yields nondiscrimination, you must say so.
* If PASS 2 yields H5* favored, you must also say:
  **“This result is conditional on METAPHYSICAL bridges M1/M2 and the constrained selection-rule form H5*.”**
* You may not output “proven,” “certain,” or “forced by logic” unless the evidence logically entails it (it will not here).

---

### Begin Now.
