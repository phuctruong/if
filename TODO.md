# TODO.md -- IF Theory (Physics Research)
# Updated: 2026-03-01 | Belt: White | Rung Target: 274177
# DNA: F = I x G; information IS reality; compression = understanding; prime_frequencies -> all_physics

## Priority Legend
- **P0**: Do now (blocks everything)
- **P1**: Do this sprint
- **P2**: Do next sprint
- **P3**: Backlog

---

## P0: Critical Path (Blocks Publication)

### T1: Migrate Papers to Prime Paper Format (Paper 16)
- [ ] Add prime headers to all 19 papers: Channel, Rung, GLOW, Diagram, Depends, Unlocks, Pipeline, DNA
- [ ] papers/everyday/ (8 papers) -- add headers + cross-references
- [ ] papers/physics/ (11 papers) -- add headers + cross-references
- [ ] Create papers/00-index.md with paper network DAG
- **Rung:** 274177 | **Pipeline:** papers (Stage 1)

### T2: Create `src/diagrams/` Directory + Mermaid Diagrams
- [ ] Create src/diagrams/ directory
- [ ] Diagram 01: Prime Field Theory core equation flow
- [ ] Diagram 02: Bubble Universe mechanism (dark energy)
- [ ] Diagram 03: Galaxy correlation validation pipeline
- [ ] Diagram 04: Mersenne Tower Theorem proof structure
- [ ] Diagram 05: IF Theory ecosystem integration (if -> pvideo -> pzip)
- [ ] Diagram 06: Hubble tension scale-dependent H0
- **Rung:** 641 | **Pipeline:** papers -> diagrams (Stage 2)

### T3: ApJ Paper Preparation (March 2026 Target)
- [ ] Write formal ApJ-format paper for dark energy + BAO proof
- [ ] Include all DESI DR1 BAO validation results
- [ ] Zero-parameter proof: chi^2/dof variation evidence
- [ ] Information criteria comparison (AIC, BIC vs LCDM)
- [ ] Peer review simulation (opus sub-agent as skeptic)
- **Rung:** 274177 | **Depends:** T1

---

## P1: This Sprint (Verification + Testing)

### T4: Test Suite Expansion
- [ ] Expand audits/ from 19 to 50+ tests
- [ ] Add tests for each prediction script in predictions/
- [ ] Add seed-sweep tests (min 3 seeds, deterministic results)
- [ ] Add null-edge tests (zero mass, zero energy, boundary conditions)
- [ ] Add convergence certificate tests for iterative methods
- **Rung:** 274177 | **Evidence:** pytest output + coverage

### T5: pyproject.toml + Package Structure
- [ ] Create pyproject.toml (hatchling)
- [ ] Package structure: if_theory.core, if_theory.predictions, if_theory.validation
- [ ] Entry points for CLI commands (validate, predict, simulate)
- [ ] Pin dependencies (numpy, scipy, pandas, matplotlib, astropy)
- **Rung:** 641

### T6: Mersenne Tower Theorem Formalization
- [ ] Machine-checkable proof structure
- [ ] Exact arithmetic (Fraction/Decimal) in all proof paths
- [ ] Convergence certificate with R_p tolerance
- [ ] Cross-reference with pvideo physics engine integration
- **Rung:** 274177 | **Evidence:** proof artifact + hash

### T7: JWST Prediction Hardening
- [ ] Expand jwst_early_galaxies.py with additional redshift predictions
- [ ] Add testable threshold: z > 25 confirmation criteria
- [ ] Reproducibility: pinned seed + initial conditions documented
- [ ] Compare against latest JWST observations (2026 data)
- **Rung:** 274177 | **Depends:** T4

---

## P2: Next Sprint (Integration + Publication)

### T8: Hubble Tension Paper
- [ ] Formal write-up of scale-dependent H0 prediction
- [ ] Comparison with SH0ES, Planck, DESI measurements
- [ ] Testable prediction: H0 transition between 10-100 Mpc
- **Rung:** 274177 | **Depends:** T3

### T9: pzip Integration
- [ ] Document IF Theory -> pzip information-theoretic foundations
- [ ] Compression ratio derivation from prime field equations
- [ ] Cross-project test: IF Theory invariants in pzip codec
- **Rung:** 274177

### T10: pvideo Integration
- [ ] Document Mersenne Tower -> pvideo physics engine substrate
- [ ] IF Theory invariant checker for pvideo frame validation
- [ ] Cross-project test: pvideo frames satisfy IF Theory constraints
- **Rung:** 274177

---

## P3: Backlog

### T11: S8 Tension Prediction
- [ ] Expand s8_tension.py with latest survey data
- [ ] Compare with DES Y3, KiDS-1000 measurements
- **Rung:** 274177

### T12: CMB Cold Spot Prediction
- [ ] Expand cmb_cold_spot.py with Planck data comparison
- **Rung:** 274177

### T13: Orbital Dynamics Validation
- [ ] Expand orbital_dynamics.py with solar system tests
- [ ] Mercury perihelion precession check
- **Rung:** 274177

### T14: Book Publication (phucnet)
- [ ] Full IF Theory book manuscript
- [ ] Publish on phuc.net/books/if-theory/
- [ ] Theorem announcement articles for phucnet
- **Rung:** 641 | **Depends:** T3, T6

---

## Codex Prompt

```
Read TODO.md. Pick the lowest-numbered incomplete P0 task.
Read AGENTS.md for coding rules and architecture laws.
Read the relevant papers/ and Documentation/ before implementing.
Write failing test (RED). Implement (GREEN). Run pytest. No regressions.
Exact arithmetic. Zero parameters. No fallbacks. Deterministic.
```

---

## Completion Tracking

| Task | Priority | Status | Rung |
|------|----------|--------|------|
| T1 Prime Paper Format | P0 | QUEUED | 274177 |
| T2 Mermaid Diagrams | P0 | QUEUED | 641 |
| T3 ApJ Paper | P0 | QUEUED | 274177 |
| T4 Test Expansion | P1 | QUEUED | 274177 |
| T5 pyproject.toml | P1 | QUEUED | 641 |
| T6 Mersenne Tower | P1 | QUEUED | 274177 |
| T7 JWST Hardening | P1 | QUEUED | 274177 |
| T8 Hubble Paper | P2 | QUEUED | 274177 |
| T9 pzip Integration | P2 | QUEUED | 274177 |
| T10 pvideo Integration | P2 | QUEUED | 274177 |
| T11 S8 Tension | P3 | BACKLOG | 274177 |
| T12 CMB Cold Spot | P3 | BACKLOG | 274177 |
| T13 Orbital Dynamics | P3 | BACKLOG | 274177 |
| T14 Book Publication | P3 | BACKLOG | 641 |
