# IF Theory Roadmap
<!-- Auth: 65537 | Tenant: if | Parent: phuclabs | Date: 2026-05-19 -->

```
DNA: roadmap = empirical_backbone × theorem_closure × cross_links × invite_falsifiers
```

This roadmap is **90-day-phased** (A → B → C → D) for the next year of IF
Theory. Dates flagged **UNKNOWN** require operator confirmation. The Solace
workers (if attached) tick autonomously against this roadmap; absent workers,
the operator drives.

---

## Phase A — Integrate the 15-paper geo canon as IF's empirical backbone (Q3 2026)

**Owner:** Phuc Truong (sole author at session time)

**Why first:** the geo canon at `~/projects/geo/canon/` already contains 15
physics papers (P5–P19) plus the L1–L6 laws plus the prime-onion construction
(PO01–PO68) plus the T1–T6 theory series, scoring ~90/90 prediction-pass
internally with 5 domains at 100% canonical-only factorization. Before IF
Theory ships any new paper, this backbone must be **canonically wired in** so
external readers see the empirical scale of the program, not just the
flagship σ₈ → r₀ inversion.

### P0 deliverables

1. **`if/canon/geo-backbone-INDEX.md`** — table of contents mapping each P-,
   K-, L-, T-, PO-paper to its IF Theory claim. Status, prediction, falsifier.
2. **Cross-citation pass**: every `papers/physics/*.md` file in IF Theory
   that references an empirical claim must cite the corresponding geo canon
   paper number (P5, K1, L3, T2, etc.) with status (CANONICAL / PARTIAL / OPEN).
3. **L6 Master Theorem alignment** — `core/constants.py` must reference the
   geo canon L6 list of canonical primes (2, 3, 5, 7, 13, 17, 19, 23, 29, 31,
   37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, …, 127, …).
   65537 is in the L6 list — note in code comments.
4. **Honest framing disclosure**: the geo canon scoring is internal evaluation.
   Add this caveat to every cross-reference in IF Theory.

### Success criteria

- 15-paper backbone INDEX.md live in `canon/`
- Every empirical claim in IF Theory papers cross-cites geo canon by number
- L6 canonical-prime list referenced in `core/constants.py` comments
- Honesty caveat present in every external-facing artifact

---

## Phase B — Publish the Mersenne Tower Theorem with machine-checkable proof (Q4 2026)

**Owner:** Phuc Truong + invited number-theorist collaborators (UNKNOWN — open call via `proposal-for-if.md`)

**Why second:** the Mersenne tower observation (π(M₇) = π(127) = 31 = M₅)
is exact and machine-verified for small p, but uniqueness across all 52
known Mersenne primes is currently OPEN. The C_XI = 62 normalization in the
σ₈ → r₀ inversion **depends on this uniqueness**. Closing the lemma is the
single highest-leverage open problem for IF Theory's credibility.

### P0 deliverables

1. **Exact π(M_p) computation** via Meissel–Mertens–Lehmer or Deléglise–Rivat
   for every Mersenne prime exponent p ≤ ~10^6. Move the `mersenne_tower_theorem.py`
   harness from approximation (x/ln(x)) to exact for tractable p.
2. **Schoenfeld/Dusart-bound argument** for larger Mersenne primes where
   exact computation is infeasible. Formal explicit-bound proof of uniqueness.
3. **arXiv preprint draft** of the Mersenne Tower Theorem paper at
   `papers/physics/mersenne-tower-theorem.md`. Section by section: motivation,
   exact verification, Schoenfeld bound, uniqueness lemma, corollary
   (C_XI = 62 follows).
4. **Machine-checkable proof** via Lean / Coq / Isabelle (TBD which) for the
   uniqueness lemma. Rung 274177 (seed sweep + replay stability + null edge
   sweep) on the proof harness.
5. **Cross-link** the Mersenne theorem to geo canon L6 (emergent primes) — the
   tower-closure property is what makes 127 a canonical prime in the geo L6 list.

### Success criteria

- Uniqueness lemma closed for all 52 known Mersenne primes
- Machine-checkable proof at rung 274177
- arXiv preprint published with stable DOI
- Geo canon L6 cross-link live

---

## Phase C — Cross-link to phuc.net articles + solaceagi.com substrate docs (Q1 2027)

**Owner:** Phuc Truong + Solace workers (Hana for drafting, atlas for orchestration if attached)

**Why third:** the cross-promotion strategy (`cross-promotion-from-solaceagi.md`)
is the channel by which IF Theory reaches non-physicist audiences. phuc.net
hosts the long-form book. solaceagi.com hosts the AGI framing that depends on
IF Theory. Both must link back. Both must respect the honesty discipline.

### P0 deliverables

1. **phuc.net article series** — 5-8 articles tracking the technical work in
   accessible language. Each cross-links the corresponding IF Theory paper +
   the geo canon backing.
2. **solaceagi.com footer** — every page carries a "Built on IF Theory — read
   more at phuc.net/if-theory" link. Strategy doc: `cross-promotion-from-solaceagi.md`.
3. **Solace AGI definition page** — `~/projects/solace-hub/canon/standards/informational-field-ai-framing.md`
   gets a "Why this is more than metaphor" subsection linking to the IF Theory
   primer. Honest: cite as research program, not proven physics.
4. **Customer-twin "Why Solace" pages** — every active customer-twin site
   (gatan, metalmark, simplemdg, maxsalesgroup, …) gets a short "Why Solace"
   page citing IF Theory as substrate. Operator-approved copy.
5. **Hana drip footnote** — outreach to Pre-Series-A AI infra rounds includes
   a footnote: "Solace's canon-update-RSI thesis is structurally derivable
   from IF Theory's compression-is-cognition claim. Citation: github.com/phuctruong/if."

### Success criteria

- ≥ 5 phuc.net articles live with IF Theory cross-links
- solaceagi.com footer live on all pages
- AGI definition page cross-links IF Theory primer
- ≥ 4 customer-twin "Why Solace" pages live citing IF Theory honestly

---

## Phase D — Invite external physicists to falsify (Q2 2027 — ongoing)

**Owner:** Phuc Truong + external falsifiers (zero yet at session time)

**Why fourth:** a theory's worth is its risk. The falsifiability list in
`README.md` is the manual for killing IF Theory. The honest-discipline test
is whether external physicists actually try.

### P0 deliverables

1. **`proposal-for-if.md`** — public call for falsifiers (NOT a sales pitch).
   Frame: "IF Theory is open. Here's what it predicts. Here's how you'd
   falsify it. Reach out if you want to try." Shipped this session.
2. **Falsifier registry** — `falsifiers/<name>.md` per external researcher
   who runs at least one falsification attempt. Tracks attempt, method,
   result, public reproducible artifact (notebook, repo, paper).
3. **JWST z > 25 prediction tag** — git-tag the prediction BEFORE 2027 JWST
   data drops. Honest: if no mature galaxies appear at z > 25, IF Theory dies.
   Tag it so the deathblow is unambiguous.
4. **σ₈ → r₀ external reproduction** — at least one external physicist who
   runs the inversion and reports their own r₀. Goal: ≥ 3 by Q4 2027.
5. **Independent BAO global fit** — external researcher runs the DESI DR1
   live audit and reproduces (or refutes) χ²/dof = 1.72.

### Success criteria

- ≥ 3 external σ₈ → r₀ reproductions
- ≥ 1 external BAO global fit
- JWST z > 25 prediction git-tagged with stable hash
- Falsifier registry has ≥ 5 entries (pass, fail, or in-progress)

---

## Out-of-scope (explicitly)

- **Selling anything**. IF Theory is OSS research. No revenue. No customers.
  Solace City is the commercial entity; IF Theory is the intellectual foundation.
- **Press cycle**. No "Nobel-worthy" adjectives in any IF Theory artifact
  until the 5 concrete `README.md` deliverables are hit.
- **pvideo / pzip implementation work**. Those are sibling projects under
  phuclabs. Their IF Theory dependencies are tracked in their own ROADMAPs.

## Honest-unknowns ledger

| Item | Status | Resolver |
|---|---|---|
| Marketing site URL (`if.phuc.net`) | UNKNOWN | operator |
| Mersenne uniqueness lemma — large-p Schoenfeld bound | OPEN | invited number-theorist |
| JWST z > 25 data | accumulating 2026–2028 | external (NASA / JWST team) |
| arXiv submission date | target Q3 2026, not submitted | author |
| Independent reproductions of σ₈ → r₀ | 0 at session time | external physicists |
| Machine-checkable proof framework choice (Lean / Coq / Isabelle) | UNKNOWN | author |
