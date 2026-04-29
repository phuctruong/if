# CASP15 Pre-Registered Blind Folding Protocol

> Per Demis Hassabis: "Pre-register the protocol BEFORE running.
> Otherwise it's not a blind test." This document specifies the exact
> protocol for testing the gai protein folding claim (TM = 1.00 with
> 0.4M parameters) on CASP15 blind targets. **This file must be merged
> to main BEFORE the gai folding model is run on any CASP15 target.**

## Pre-registration timestamp

This protocol is committed at validation pass v1.0.0 (2026-04-29).
Any subsequent change to the protocol must be committed as a new
version with a clear rationale and changelog entry. The pre-registered
protocol cannot be modified after the first run on a CASP15 target.

## Claim under test

> **Claim #17 (gai grand theory)**: Protein folding with TM-score = 1.00
> on hard CASP15 targets, using a 0.4M-parameter model derived from
> IF Theory's prime-channel framework, training on no MSA / template /
> evolutionary information, only the amino-acid sequence as input.

The control: AlphaFold 2 / 3 achieves typical TM-score ≈ 0.85 on hard
CASP15 targets using ~93M parameters and MSA + Evoformer + structure
module + recycling.

## Targets

The following CASP15 hard targets are pre-registered for the blind test:

| Target | Domain | L (residues) | Difficulty |
|---|---|---|---|
| T1104 | full archive staged | TBD | hard |
| T1109 | full archive staged | TBD | hard |
| T1119 | full archive staged | TBD | hard |

Targets, full participant prediction archives, and ground-truth
structures are staged at `~/Downloads/if/data/casp15/` (see the
data-acquisition manifest).

**Blind discipline.** The implementer of the gai folding model must
*not* look at the ground-truth structures of these three targets
before producing the predictions. The implementer may inspect any
*other* PDB or CASP15 structure for development. The targets above
are a held-out test set.

## Inputs (fixed by this pre-registration)

For each blind target:

- **Allowed**: amino-acid sequence (FASTA format, exactly as released
  by CASP15 organizers).
- **Forbidden**: MSA, evolutionary covariance, templates, the target's
  own ground-truth coordinates, the target's identity if it has
  metadata revealing fold class.

The model is given the FASTA and must produce 3D coordinates.

## Outputs (fixed by this pre-registration)

For each blind target, the gai model must produce:

- A PDB file with predicted coordinates for every C_α atom in the
  target sequence.
- A confidence score per residue (analog of AlphaFold's pLDDT, scale
  0-100, optional but recommended).
- A single overall confidence score for the prediction.
- A wall-clock time and peak GPU/CPU memory consumed.
- A parameter count, with provenance for each parameter (which prime
  channel, which weight type, learned vs derived).

## Evaluation (fixed by this pre-registration)

For each blind target, against the released ground-truth structure:

1. **TM-score** (Zhang & Skolnick, the standard CASP metric) computed
   via the canonical TM-align tool. Range [0, 1].
2. **GDT-TS** (the original CASP metric).
3. **RMSD** of C_α atoms after Procrustes alignment.
4. **lDDT** (local Distance Difference Test).

PASS criteria:

- **Strong PASS**: median TM-score ≥ 0.90 across the 3 blind targets,
  beating AlphaFold's published baseline.
- **Moderate PASS**: median TM-score ≥ 0.70, comparable to AlphaFold,
  with parameter count < 1M (≥ 90× smaller).
- **Weak PASS**: median TM-score ≥ 0.40, well above the random-coil
  floor (TM ≈ 0.05) and the universal-d(k)-only floor (TM ≈ 0.16, see
  `predictions/if_theory_minimal_folding.py`), demonstrating that the
  model adds sequence-specific information.
- **FAIL**: median TM-score < 0.40 — the IF Theory folding claim is
  no better than universal d(k) alone; reject the TM = 1.00 claim.

The gai TM = 1.00 headline claim requires Strong PASS on all three
targets. Anything less downgrades the headline.

## Reporting (fixed)

A single Markdown file `evidence/casp15_blind/casp15_results.md` will
contain:

- Target ID, sequence (sha256 of FASTA), L (residues).
- For each target: predicted PDB filename, TM-score, GDT-TS, RMSD,
  lDDT, runtime, memory, parameter count.
- AlphaFold baseline TM-score on the same target (from CASP15 archive).
- Conclusion: which PASS tier was achieved, or FAIL.
- Author signature with date and git commit hash.

A JSON evidence file `evidence/casp15_blind/casp15_results.json` will
contain machine-readable versions of all numbers.

## Forbidden adjustments

After the first run, the following adjustments to the protocol are
**forbidden** without a separate, clearly-labeled "v2 protocol" file:

- Changing the target list.
- Allowing MSA or template input.
- Reweighting the TM-score thresholds.
- Excluding any target post-hoc.
- "Refining" hyperparameters using the test targets.

A retry on the same targets with a different model is fine, as long
as it's labeled as such and the original protocol's first-run results
are preserved.

## Implementation status (2026-04-29)

The gai folding model is **not yet implemented** in this repository.
The infrastructure for the test exists:

- `predictions/if_theory_minimal_folding.py` — a universal-d(k)-only
  baseline scoring TM-like = 0.16 (which sets the floor any IF-based
  folding model must clear).
- `predictions/pdb_mds_sanity_check.py` — confirms classical MDS
  recovers 3D from any distance matrix to numerical precision.
- `~/Downloads/if/data/casp15/` — staged target structures + full
  participant archives.

To complete the test, an implementer must:

1. Read the gai grand theory canonical papers in
   `~/projects/solace-prime/canon/papers/` and any reference code in
   `~/projects/gai/code/`.
2. Implement the model as `predictions/gai_folding_model.py`.
3. Run on the three CASP15 targets above without seeing ground truth.
4. Report results per the format above.
5. Open a GitHub issue with the conclusion.

The gai TM=1.00 claim is OPEN until this pre-registered test runs.

## Why pre-registration matters

Without pre-registration, claims like "TM = 1.00" can be quietly
weakened post-hoc (different metric, different targets, different
thresholds). Pre-registration makes the test honest: either the model
clears the bar set in advance, or it doesn't.

This is the standard for clinical trials, particle physics blinded
analyses, and competitive ML benchmarks. Protein folding deserves
the same standard.
