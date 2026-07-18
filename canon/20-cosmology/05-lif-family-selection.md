# ℒ_IF Family Selection — worked on paper, zero data contact

> Auth: 65537 · Layer: SCIENCE · Written 2026-07-18 (loop iteration 7).
> Applies the C1–C8 constraints (`04-lif-design-constraints.md`) to the three candidate
> families **in prose only**. No SPARC contact. One family dies here; one is deferred;
> one survives with conditions. Sagan phrasing still governs: *specified, not yet
> implemented, not yet tested.*

## 1. Family C (memory-kernel gravity) — KILLED ON PAPER

Form: g(r) = g_N(r) · K(τ_dyn/τ_IF), one global timescale τ_IF, K → 1 for
τ_dyn ≪ τ_IF (C2 ✓ by construction).

**BTFR derivation (C5).** Take τ_dyn = 2πr/v and deep-regime K(x) ≈ x^α. A flat
rotation curve requires v²/r = (GM/r²)(2πr/(v τ_IF))^α, i.e.

    v^(2+α) = GM (2π/τ_IF)^α · r^(α−1)

r-independence (flatness) forces **α = 1**, which then gives

    v³ = 2πGM/τ_IF   →   **M ∝ v³**.

The same cubic results for the alternative timescale τ = v/g (any single global *time*
constant can only combine with G and M into a velocity cubed — dimensional analysis,
not a modeling choice). The observed baryonic Tully–Fisher slope is ≈ 4 (MOND's
v⁴ = G M a₀ requires an *acceleration* constant); slope 3 is excluded by the data this
law would eventually face. The only repair — making K's argument an acceleration ratio
g/a₀ — is MOND's μ-function renamed: forbidden as `INTERPOLATION_SMUGGLING`.

**Verdict: family C is falsified before touching data.** (Numerical footnote: the
normalization would have needed τ_IF ≈ 3×10⁸ yr — a suggestive cosmic timescale, which
is exactly why the slope check mattered *before* anyone fell in love with it.)

## 2. Family B (entropic-gradient potential) — DEFERRED on prior-art grounds

Φ = Φ_N + λ∇S_config is Verlinde-adjacent (emergent/entropic gravity). The P16
discipline (self-audit before claiming) applies with full force: the entropic-gravity
literature already contains both the idea and known galaxy-scale difficulties. Working
this family means first writing the honest prior-art map (Verlinde 2011/2016, its
lensing/dwarf tests) and identifying a *divergence point* that is IF's own. Until that
audit exists, family B may not be frozen. Deferred, not dead.

## 3. Family A (information-density modified dynamics) — SURVIVES, with teeth

Naive form: response to gravity modified by f(I(r)/I₀). If I(r) is simply baryon
surface density Σ renamed, the theory collapses into Milgrom's Σ-form
(`INTERPOLATION_SMUGGLING` again) — the known MOND fact that departures begin below the
critical surface density Σ† = a₀/G would be *re-labeled*, not explained.

**The survival condition — where IF must differ from MOND:** let the modification
depend on informational structure that is NOT a function of the axisymmetric mass
profile alone. Then the family makes a prediction MOND *forbids*:

> **Two galaxies with identical baryonic surface-density profiles Σ(r) but different
> structural order (spiral coherence, clumpiness, azimuthal organization) must have
> measurably different rotation curves.** Under MOND, identical Σ(r) → identical
> g_N(r) → identical curves, exactly.

This is the falsifiable wedge, and it is the *only* content that makes family A a
theory rather than a re-parameterization. It comes with obligations:

- **C1 obligation:** the structure functional must be a declared estimator on
  photometry before freezing. Candidate estimators (to be chosen and frozen, not
  hedged): (a) azimuthal mutual information between image sectors at fixed radius;
  (b) a compression-ratio functional on the residual image after axisymmetric
  subtraction. Either is computable from SPARC's photometry; neither may be tuned
  after seeing kinematics (`ESTIMATOR_HANDWAVE`, `FIT_BEFORE_FREEZE`).
- **P17 obligation (C6):** no narrative in which this structure "accumulates wherever
  energy flows" — P17 measured that accumulation is not free. The law must treat
  structure as an observed input, not an assumed attractor.
- **Honest prior check:** if, on paper, the estimator provably reduces to a function of
  Σ(r) for real disk galaxies (structure correlating too tightly with surface density),
  the wedge closes and family A dies the same death as C. This check — correlation of
  candidate structure measures with Σ across published photometric samples — is
  *literature work*, allowed before freeze because it touches no kinematics.

## 4. Decision

Family C: **dead** (kill-logged). Family B: **deferred** pending prior-art audit.
Family A: **the working family**, freeze-blocked until (i) one structure estimator is
chosen and specified to the pixel level, (ii) the Σ-degeneracy literature check passes,
(iii) the exact functional + fitting procedure is written into a prereg commit. Only
then: one SPARC run against the frozen admission bar (interesting ≤ 3.71; win ≤ 1.14
with BIC ≤ 19.8; 30% held-out; ≤ 1 fitted M/L per galaxy).

The next concrete step is (i)+(ii) — estimator specification and the Σ-degeneracy
check. Both are prose-and-literature circuits; neither spends legitimacy.
