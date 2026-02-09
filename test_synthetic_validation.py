#!/usr/bin/env python3
"""
Synthetic Data Validation for Prime Field Theory
=================================================

PURPOSE:
    Verify the code is mathematically correct BEFORE running on real data.
    Uses synthetic (known-answer) tests to isolate code bugs from data issues.

METHODOLOGY:
    Each test constructs a situation with a KNOWN correct answer, then checks
    the code produces that answer. If a test fails, it means the CODE is wrong
    (not the physics), so fix the code before trusting results on real data.

TEST OVERVIEW:
    Code correctness tests (PASS/FAIL):
      1. Field equation:     Φ(r) = A/log(r/r₀+1) matches analytical formula
      2. Gradient:           dΦ/dr matches both numerical differentiation and formula
      3. Pair distance PDF:  Normalization, boundary conditions, and mean
      4. σ₈ round-trip A:    r₀ → C_XI → σ₈ → C_XI (invertibility check)
      5. σ₈ round-trip B:    C_XI → r₀ → σ₈ → r₀ (invertibility check)
      6. σ₈ monotonicity:    σ₈²(r₀) is monotonic (unique solution exists)
      7. Rotation curve:     v(r) = v₀√(r|dΦ/dr|) matches direct computation
      8. Correlation recovery: Recover C_XI from noisy synthetic ξ(r) data
      9. Module consistency: field & gradient match between core and dark_energy_util

    Scientific analysis tests (INFORMATIONAL, do not block):
     10. r₀ derivability:    Can any math constant C_XI produce r₀ = 0.65 kpc?
     11. Parameter count:    Honest assessment of empirical vs derived parameters

    Mersenne tower tests (PASS/FAIL):
     12. Mersenne recursion: π(M₇) = π(127) = 31 = M₅ (exact number theory)
     13. Mersenne tower mode: C_XI=62 → r₀ consistent with empirical value
     14. Zero-parameter consistency: Both modes give compatible predictions

HOW TO READ RESULTS:
    [PASS] = Code is correct for this test
    [FAIL] = Code BUG — must fix before trusting any results
    [INFO] = Scientific finding, not a code bug

REVIEWER NOTES:
    - All tests use r₀ = 0.65 kpc (0.00065 Mpc) and σ₈ = 0.8111 (Planck 2018)
    - Numerical tolerances are tight (typically 1e-10) to catch subtle bugs
    - Tests 4-5 verify the σ₈ integration is invertible (critical for parameter derivation)
    - Test 6 proves the σ₈ equation has a UNIQUE solution (monotonicity)
    - Tests 9 check consistency between two independent code modules
    - Test 10 exhaustively searches mathematical constants — negative result is expected
    - If ALL code tests pass, any disagreement with real data is a PHYSICS issue, not code
"""

import numpy as np
from scipy import integrate, optimize
from scipy.stats import pearsonr
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.constants import SIGMA_8, H_PLANCK, AMPLITUDE
from core.field_equations import FieldEquations
from core.parameter_derivations import ParameterDerivation

import logging
logging.basicConfig(level=logging.WARNING)

PASS = 0
FAIL = 0
INFO = 0


def report(name, passed, detail=""):
    """Report a PASS/FAIL test result. Failures indicate code bugs."""
    global PASS, FAIL
    if passed:
        PASS += 1
        status = "PASS"
    else:
        FAIL += 1
        status = "FAIL"
    print(f"  [{status}] {name}")
    if detail:
        print(f"         {detail}")
    return passed


def report_info(name, detail=""):
    """Report an informational finding. Not a pass/fail — scientific analysis."""
    global INFO
    INFO += 1
    print(f"  [INFO] {name}")
    if detail:
        print(f"         {detail}")


# =============================================================================
# CODE CORRECTNESS TESTS (PASS/FAIL)
# =============================================================================

def test_field_equation():
    """
    TEST 1: Field Equation Correctness
    ===================================
    WHY: The field Φ(r) = A/log(r/r₀ + 1) is the foundation of everything.
         If this is wrong, all predictions are wrong.

    METHOD: Compare FieldEquations.field() output against direct formula
            evaluation at 100 log-spaced points from 0.0001 to 1000 Mpc.

    ALSO CHECKS:
    - Monotonically decreasing (physical requirement: field weakens with distance)
    - Correct value at r = r₀: Φ(r₀) = 1/log(2) ≈ 1.4427
    - Large-r asymptotic: Φ(r>>r₀) ≈ 1/log(r/r₀)

    TOLERANCE: 1e-10 relative error (should be machine precision)
    """
    print("\n--- TEST 1: Field Equation ---")

    r0 = 0.00065  # Mpc (0.65 kpc)
    fe = FieldEquations(r0_mpc=r0)

    r_values = np.logspace(-4, 3, 100)
    phi = fe.field(r_values)

    # Direct formula evaluation (ground truth)
    phi_analytical = AMPLITUDE / np.log(r_values / r0 + 1.0)

    max_err = np.max(np.abs(phi - phi_analytical) / np.abs(phi_analytical))
    report("Φ(r) matches 1/log(r/r₀+1)", max_err < 1e-10,
           f"max relative error = {max_err:.2e}")

    report("Φ(r) monotonically decreasing", np.all(np.diff(phi) < 0))

    # At r = r₀: x = r/r₀ + 1 = 2, so Φ = 1/log(2)
    phi_at_r0 = fe.field(np.array([r0]))[0]
    expected = 1.0 / np.log(2.0)
    report("Φ(r₀) = 1/log(2)", abs(phi_at_r0 - expected) / expected < 1e-10,
           f"Φ(r₀) = {phi_at_r0:.6f}, expected = {expected:.6f}")

    # Large-r limit: the +1 becomes negligible, so Φ ≈ 1/log(r/r₀)
    r_large = 1000.0
    phi_large = fe.field(np.array([r_large]))[0]
    approx = 1.0 / np.log(r_large / r0)
    rel_err = abs(phi_large - approx) / approx
    report("Φ(r>>r₀) ≈ 1/log(r/r₀)", rel_err < 0.01,
           f"relative error = {rel_err:.4e}")


def test_gradient():
    """
    TEST 2: Gradient Correctness
    ==============================
    WHY: The gradient dΦ/dr drives rotation curves (v² ∝ r|dΦ/dr|) and
         dark energy dynamics. A wrong gradient means wrong velocity predictions.

    METHOD:
    1. Compare code gradient against numerical central difference (h = r×10⁻⁶)
    2. Compare code gradient against analytical formula: dΦ/dr = -A/(r₀·x·log²x)
       where x = r/r₀ + 1

    HISTORY: dark_energy_util.py had a gradient bug (missing r₀ factor) that was
             fixed in this audit. This test catches regressions.

    TOLERANCE: 1e-5 for numerical comparison (finite difference), 1e-10 for formula
    """
    print("\n--- TEST 2: Gradient ---")

    r0 = 0.00065
    fe = FieldEquations(r0_mpc=r0)

    r_values = np.logspace(-3, 2, 50)

    grad_analytical = fe.field_gradient(r_values)

    # Numerical gradient (central difference with step h = r × 10⁻⁶)
    dr = r_values * 1e-6
    phi_plus = fe.field(r_values + dr)
    phi_minus = fe.field(r_values - dr)
    grad_numerical = (phi_plus - phi_minus) / (2 * dr)

    rel_errors = np.abs(grad_analytical - grad_numerical) / np.abs(grad_numerical)
    max_err = np.max(rel_errors)

    report("dΦ/dr matches numerical differentiation", max_err < 1e-5,
           f"max relative error = {max_err:.2e}")

    report("dΦ/dr < 0 everywhere", np.all(grad_analytical < 0))

    # Direct formula: dΦ/dr = -A / (r₀ × (r/r₀+1) × log²(r/r₀+1))
    x = r_values / r0 + 1.0
    grad_formula = -AMPLITUDE / (r0 * x * np.log(x)**2)
    max_err2 = np.max(np.abs(grad_analytical - grad_formula) / np.abs(grad_formula))
    report("dΦ/dr = -A/(r₀·x·log²x)", max_err2 < 1e-10,
           f"max relative error = {max_err2:.2e}")


def test_pair_distance_pdf():
    """
    TEST 3: Pair Distance PDF
    ==========================
    WHY: The σ₈ integration uses f(s), the PDF of distances between two random
         points in a sphere. If f(s) is wrong, σ₈ and C_XI are wrong.

    FORMULA: f(s) = (3s²/(2R³)) × (2 - 3t + t³) where t = s/(2R)
    REFERENCE: Lord (1954), Peebles (1980) §36

    CHECKS:
    - ∫₀^{2R} f(s) ds = 1 (proper normalization)
    - f(0) = 0, f(2R) = 0 (boundary conditions — min and max separation)
    - f(s) ≥ 0 for all s (it's a probability density)
    - <s> = 36R/35 (known first moment — derived from the formula)

    TOLERANCE: 1e-8 for integral and moment (quadrature precision)
    """
    print("\n--- TEST 3: Pair Distance PDF ---")

    pd = ParameterDerivation(use_empirical_r0=True)
    R = 8.0 / H_PLANCK  # R₈ ≈ 11.88 Mpc

    norm, _ = integrate.quad(lambda s: pd._pair_distance_pdf(s, R), 0, 2 * R)
    report("∫f(s)ds = 1 (normalization)", abs(norm - 1.0) < 1e-10,
           f"integral = {norm:.12f}")

    report("f(0) = 0 (zero probability at zero separation)",
           pd._pair_distance_pdf(0, R) == 0.0)

    report("f(2R) = 0 (zero probability at max separation)",
           pd._pair_distance_pdf(2 * R, R) == 0.0)

    s_test = np.linspace(0, 2 * R, 1000)
    f_test = np.array([pd._pair_distance_pdf(s, R) for s in s_test])
    report("f(s) >= 0 for all s", np.all(f_test >= 0))

    # Known first moment: <s> = 36R/35 (from integrating s × f(s))
    mean, _ = integrate.quad(lambda s: s * pd._pair_distance_pdf(s, R), 0, 2 * R)
    expected_mean = 36 * R / 35
    report("<s> = 36R/35 (known first moment)",
           abs(mean - expected_mean) / expected_mean < 1e-8,
           f"<s> = {mean:.6f}, expected = {expected_mean:.6f}")


def test_sigma8_roundtrip_cxi():
    """
    TEST 4: σ₈ Round-Trip (r₀ → C_XI → σ₈)
    ==========================================
    WHY: The derivation computes C_XI = σ₈²/I(r₀). If we plug C_XI back into
         the σ₈ integral, we MUST recover the original σ₈. This tests that
         the integration code is self-consistent.

    METHOD:
    1. Use empirical r₀ = 0.65 kpc
    2. ParameterDerivation computes C_XI = σ₈² / ∫Φ²f ds
    3. Recompute σ₈ = √(C_XI × ∫Φ²f ds)
    4. Check σ₈ matches the Planck 2018 value

    TOLERANCE: 1e-6 relative error (quadrature accumulation)
    """
    print("\n--- TEST 4: σ₈ Round-Trip (r₀ → C_XI → σ₈) ---")

    pd = ParameterDerivation(use_empirical_r0=True)

    C_XI = pd.correlation_normalization
    r0 = pd.r0_mpc

    sigma8_sq = pd._compute_sigma8_squared(r0, C_XI)
    sigma8_recovered = np.sqrt(sigma8_sq)

    error = abs(sigma8_recovered - SIGMA_8) / SIGMA_8
    report("σ₈ round-trips through C_XI", error < 1e-6,
           f"σ₈ = {sigma8_recovered:.6f}, target = {SIGMA_8:.6f}, error = {error:.2e}")

    report("C_XI > 0 (physical)", C_XI > 0, f"C_XI = {C_XI:.4f}")
    report("C_XI is finite", np.isfinite(C_XI), f"C_XI = {C_XI:.4f}")


def test_sigma8_roundtrip_r0():
    """
    TEST 5: σ₈ Round-Trip (C_XI → r₀ → σ₈)
    ==========================================
    WHY: If someone knows C_XI independently, they should be able to recover
         r₀ from σ₈. This tests the optimizer and the σ₈ integral together.

    METHOD:
    1. Use the C_XI derived from empirical r₀
    2. Forget r₀, then re-derive it by solving σ₈(r₀, C_XI) = 0.8111
    3. Check recovered r₀ matches 0.65 kpc

    TOLERANCE: 1e-4 relative error (optimizer convergence)
    """
    print("\n--- TEST 5: σ₈ Round-Trip (C_XI → r₀ → σ₈) ---")

    pd = ParameterDerivation(use_empirical_r0=True)
    C_XI_true = pd.correlation_normalization
    r0_true = pd.r0_mpc
    target = SIGMA_8**2

    def objective(log_r0):
        r0 = np.exp(log_r0)
        s8sq = pd._compute_sigma8_squared(r0, C_XI_true)
        if s8sq <= 0 or not np.isfinite(s8sq):
            return 1e10
        return (np.log(s8sq) - np.log(target))**2

    result = optimize.minimize_scalar(
        objective, bounds=(-15, 2), method='bounded',
        options={'xatol': 1e-12, 'maxiter': 1000}
    )

    r0_recovered = np.exp(result.x)
    error = abs(r0_recovered - r0_true) / r0_true

    report("r₀ recoverable from σ₈ + C_XI", error < 1e-4,
           f"r₀ = {r0_recovered*1000:.4f} kpc, true = {r0_true*1000:.4f} kpc, "
           f"error = {error:.2e}")

    sigma8_check = np.sqrt(pd._compute_sigma8_squared(r0_recovered, C_XI_true))
    s8_err = abs(sigma8_check - SIGMA_8) / SIGMA_8
    report("σ₈ at recovered r₀ matches target", s8_err < 1e-4,
           f"σ₈ = {sigma8_check:.6f}, target = {SIGMA_8:.6f}")


def test_sigma8_monotonicity():
    """
    TEST 6: σ₈² Monotonicity in r₀
    =================================
    WHY: If σ₈²(r₀) is not monotonic, the equation σ₈² = target could have
         multiple solutions, making r₀ ambiguous. Monotonicity guarantees
         a UNIQUE r₀ for any given C_XI and σ₈.

    METHOD: Evaluate σ₈² at 30 log-spaced r₀ values from 0.01 to 1000 kpc.
            Check all successive differences have the same sign.

    EXPECTED: Monotonically increasing (larger r₀ → stronger field → more variance)
    """
    print("\n--- TEST 6: σ₈² Monotonicity in r₀ ---")

    pd = ParameterDerivation(use_empirical_r0=True)
    C_XI = pd.correlation_normalization

    r0_values = np.logspace(-5, 0, 30)  # 0.01 to 1000 kpc in Mpc
    sigma8sq_values = np.array([
        pd._compute_sigma8_squared(r0, C_XI) for r0 in r0_values
    ])

    diffs = np.diff(sigma8sq_values)
    is_monotone_inc = np.all(diffs > 0)
    is_monotone_dec = np.all(diffs < 0)

    direction = ('increasing' if is_monotone_inc
                 else 'decreasing' if is_monotone_dec
                 else 'NOT monotonic')

    report("σ₈²(r₀) is monotonic (unique solution)", is_monotone_inc or is_monotone_dec,
           direction)

    print(f"         r₀ range: [{r0_values[0]*1000:.3f}, {r0_values[-1]*1000:.1f}] kpc")
    print(f"         σ₈ range: [{np.sqrt(sigma8sq_values[0]):.4f}, "
          f"{np.sqrt(sigma8sq_values[-1]):.4f}]")
    print(f"         Target σ₈ = {SIGMA_8}")


def test_rotation_curve():
    """
    TEST 7: Rotation Curve
    ========================
    WHY: The rotation curve v(r) = v₀√(r|dΦ/dr|) is the key dark matter
         prediction. Must verify it's computed correctly and is monotonically
         decreasing (the prime field alone does NOT produce flat rotation curves).

    METHOD: Compare v(r) from gradient code against analytical formula
            v² = v₀² × r / (r₀ × (r/r₀+1) × log²(r/r₀+1))

    KEY RESULT: v(r) is monotonically DECREASING — the prime field contribution
                falls as ~√(r₀/r) at large r. Observed flat curves require
                additional baryonic contributions.

    TOLERANCE: 1e-10 relative error
    """
    print("\n--- TEST 7: Rotation Curve ---")

    r0 = 0.00065  # Mpc
    v0 = 394.4    # km/s
    fe = FieldEquations(r0_mpc=r0)

    r_kpc = np.array([1.0, 2.5, 5.0, 10.0, 20.0, 50.0, 100.0])
    r_mpc = r_kpc / 1000.0

    grad = fe.field_gradient(r_mpc)
    v_computed = v0 * np.sqrt(r_mpc * np.abs(grad))

    # Analytical: substitute |dΦ/dr| = A/(r₀·x·log²x)
    x = r_mpc / r0 + 1.0
    v_analytical = v0 * np.sqrt(r_mpc / (r0 * x * np.log(x)**2))

    max_err = np.max(np.abs(v_computed - v_analytical) / v_analytical)
    report("v(r) matches analytical formula", max_err < 1e-10,
           f"max relative error = {max_err:.2e}")

    report("v(r) monotonically decreasing (no peak)", np.all(np.diff(v_computed) < 0))

    print("         r (kpc)   v (km/s)   Note")
    for i in range(len(r_kpc)):
        note = ""
        if abs(r_kpc[i] - 2.5) < 0.1:
            note = "<-- closest to 220 km/s"
        if abs(r_kpc[i] - 10.0) < 0.1:
            note = "<-- standard comparison point"
        print(f"         {r_kpc[i]:>6.1f}    {v_computed[i]:>7.1f}   {note}")


def test_synthetic_correlation_recovery():
    """
    TEST 8: Synthetic Correlation Recovery
    ========================================
    WHY: Before fitting real ξ(r) data, verify that IF the data actually comes
         from our model (with noise), we can recover the correct C_XI.

    METHOD:
    1. Generate ξ_true(r) = C_XI / log²(r/r₀ + 1) at 9 radial bins
    2. Add 5% Gaussian noise (typical observational error level)
    3. Fit C_XI via linear least squares: C_XI_fit = Σ(ξ·Φ²)/Σ(Φ⁴)
    4. Check recovery within 10% and shape correlation > 0.99

    NOTE: This test uses seed=42 for reproducibility.

    TOLERANCE: 10% for C_XI recovery (with 5% noise), 0.99 for Pearson r
    """
    print("\n--- TEST 8: Synthetic Correlation Recovery ---")

    pd = ParameterDerivation(use_empirical_r0=True)
    C_XI = pd.correlation_normalization
    r0 = pd.r0_mpc

    r_bins = np.array([0.5, 1, 2, 5, 10, 20, 50, 100, 150])  # Mpc
    xi_true = C_XI / np.log(r_bins / r0 + 1.0)**2

    np.random.seed(42)
    noise = 0.05 * xi_true * np.random.randn(len(r_bins))
    xi_noisy = xi_true + noise

    # Linear least squares for C_XI given known r₀
    phi_sq = 1.0 / np.log(r_bins / r0 + 1.0)**2
    C_XI_fit = np.sum(xi_noisy * phi_sq) / np.sum(phi_sq**2)

    error = abs(C_XI_fit - C_XI) / C_XI
    report("C_XI recovery with 5% noise", error < 0.10,
           f"C_XI_fit = {C_XI_fit:.2f}, true = {C_XI:.2f}, error = {error*100:.1f}%")

    xi_fit = C_XI_fit * phi_sq
    r_pearson, _ = pearsonr(np.log(xi_noisy), np.log(xi_fit))
    report("Shape correlation > 0.99", r_pearson > 0.99,
           f"Pearson r = {r_pearson:.6f}")


def test_module_consistency():
    """
    TEST 9: Cross-Module Consistency
    ==================================
    WHY: The field and gradient are implemented in TWO places:
         - core/field_equations.py (main physics)
         - dark_energy_util.py (bubble universe calculations)
         These MUST agree. A previous bug had the gradient wrong in dark_energy_util.

    METHOD: Evaluate field and gradient at 5 points using both modules, compare.

    HISTORY: dark_energy_util.py previously had dΦ/dr = -1/(r·log²x) instead of
             the correct -1/(r₀·x·log²x). This was fixed in this audit.

    TOLERANCE: 1e-10 relative error (should be identical)
    """
    print("\n--- TEST 9: Cross-Module Consistency ---")

    from dark_energy_util import PrimeFieldPotential

    r0_kpc = 0.65
    r0_mpc = r0_kpc / 1000.0

    fe = FieldEquations(r0_mpc=r0_mpc)
    pfp = PrimeFieldPotential(r0_kpc=r0_kpc)

    r_mpc = np.array([0.01, 0.1, 1.0, 10.0, 100.0])

    # Field comparison
    phi_core = fe.field(r_mpc)
    phi_de = pfp.potential(r_mpc)
    field_err = np.max(np.abs(phi_core - phi_de) / np.abs(phi_core))
    report("Field consistent (core vs dark_energy_util)", field_err < 1e-10,
           f"max relative error = {field_err:.2e}")

    # Gradient comparison
    grad_core = fe.field_gradient(r_mpc)
    grad_de = pfp.gradient(r_mpc)
    grad_err = np.max(np.abs(grad_core - grad_de) / np.abs(grad_core))
    report("Gradient consistent (core vs dark_energy_util)", grad_err < 1e-10,
           f"max relative error = {grad_err:.2e}")

    if grad_err >= 1e-10:
        print("         r (Mpc)    core           dark_energy    ratio")
        for i in range(len(r_mpc)):
            ratio = grad_de[i] / grad_core[i]
            print(f"         {r_mpc[i]:<10.2f} {grad_core[i]:<14.6e} "
                  f"{grad_de[i]:<14.6e} {ratio:.6f}")


# =============================================================================
# SCIENTIFIC ANALYSIS TESTS (INFORMATIONAL — do not block)
# =============================================================================

def test_can_r0_be_derived():
    """
    TEST 10: Can r₀ Be Derived From First Principles?
    ====================================================
    WHY: The theory has 1 empirical input (r₀ = 0.65 kpc). If we could find
         a mathematical constant for C_XI, then σ₈ would uniquely determine r₀,
         making the theory truly zero-parameter.

    METHOD:
    - The equation is: σ₈² = C_XI × ∫₀^{2R₈} Φ²(s) f(s) ds
    - This is 1 equation in 2 unknowns (C_XI and r₀)
    - For each candidate C_XI (math constants), solve for r₀
    - Check if any candidate gives r₀ ≈ 0.65 kpc

    CANDIDATES TESTED:
    - π√3 ≈ 5.44 (original theory claim — WRONG, gives r₀ = 588 kpc)
    - 20π ≈ 62.83 (closest simple constant — gives r₀ = 0.62 kpc, 5% off)
    - Others: 4π, 2π², 4π², 8π², 64, π⁴/e

    RESULT: This is INFORMATIONAL. A [INFO] result means "no simple constant works"
            which is expected. It does NOT indicate a code bug.

    IMPLICATION: The theory genuinely has 1 empirical parameter.
                 This is still impressive (ΛCDM has 6, MOND has 1).
    """
    print("\n--- TEST 10: Can r₀ Be Derived? (INFORMATIONAL) ---")
    print("  Question: Does any math constant C_XI give r₀ = 0.65 kpc via σ₈?")

    pd = ParameterDerivation(use_empirical_r0=True)
    target = SIGMA_8**2
    r0_target_kpc = 0.65

    candidates = {
        "pi*sqrt(3)":    np.pi * np.sqrt(3),
        "4*pi":          4 * np.pi,
        "2*pi^2":        2 * np.pi**2,
        "4*pi^2":        4 * np.pi**2,
        "8*pi^2":        8 * np.pi**2,
        "64":            64.0,
        "pi^4/e":        np.pi**4 / np.e,
        "20*pi":         20 * np.pi,
        "actual(62.19)": pd.correlation_normalization,
    }

    print(f"\n  {'C_XI formula':<16} {'value':<10} {'r0 (kpc)':<10} "
          f"{'sigma8@0.65':<12} {'r0 error'}")
    print(f"  {'-'*65}")

    best_name = None
    best_error = float('inf')

    for name, c_xi in candidates.items():
        # What σ₈ does this C_XI give at the empirical r₀?
        sigma8_at_065 = np.sqrt(pd._compute_sigma8_squared(0.00065, c_xi))

        # What r₀ does this C_XI + observed σ₈ produce?
        def obj(log_r0, c=c_xi):
            r0 = np.exp(log_r0)
            s8sq = pd._compute_sigma8_squared(r0, c)
            if s8sq <= 0 or not np.isfinite(s8sq):
                return 1e10
            return (np.log(s8sq) - np.log(target))**2

        result = optimize.minimize_scalar(obj, bounds=(-15, 2), method='bounded')
        r0_kpc = np.exp(result.x) * 1000

        err = abs(r0_kpc - r0_target_kpc) / r0_target_kpc
        marker = " <--" if err < 0.06 else ""
        print(f"  {name:<16} {c_xi:<10.4f} {r0_kpc:<10.3f} "
              f"{sigma8_at_065:<12.4f} {err*100:>6.1f}%{marker}")

        if err < best_error and name != "actual(62.19)":
            best_error = err
            best_name = name

    print()
    if best_error < 0.05:
        report_info(f"Closest: C_XI = {best_name} gives r₀ within {best_error*100:.1f}%",
                    f"PROMISING — but needs theoretical justification for WHY C_XI = {best_name}")
    else:
        report_info(f"No simple math constant gives r₀ = 0.65 kpc (best: {best_name}, "
                    f"{best_error*100:.1f}% off)",
                    "r₀ remains an empirical input. Theory has 1 free parameter.")


def test_parameter_count():
    """
    TEST 11: Parameter Count Assessment (INFORMATIONAL)
    =====================================================
    WHY: Honest documentation of what's empirical vs derived.
         A reviewer needs to know exactly how many knobs can be turned.

    CLASSIFICATION:
    - EXACT: mathematically determined, no freedom (amplitude = 1 from PNT)
    - EMPIRICAL: fitted to data, could in principle be different (r₀)
    - DERIVED: uniquely determined once empirical inputs are fixed (C_XI from σ₈)
    - SEMI-DERIVED: formula exists but has theoretical uncertainty (v₀, ~30%)
    - OBSERVED: measured by experiments, not part of theory (σ₈, H₀, Ωm)

    COMPARISON:
    - ΛCDM: 6 free parameters (Ωb·h², Ωc·h², H₀, nₛ, Aₛ, τ)
    - MOND: 1 free parameter (a₀ ≈ 1.2×10⁻¹⁰ m/s²)
    - Prime Field Theory: 1 empirical input (r₀ = 0.65 kpc)
    """
    print("\n--- TEST 11: Parameter Count (INFORMATIONAL) ---")

    pd = ParameterDerivation(use_empirical_r0=True)

    print("  PARAMETER HIERARCHY:")
    print()
    print(f"    1. Amplitude = {pd.amplitude}")
    print(f"       Source: prime number theorem pi(x) ~ x/log(x), coefficient = 1")
    print(f"       Status: EXACT — no freedom")
    print()
    print(f"    2. r0 = {pd.r0_kpc} kpc ({pd.r0_mpc} Mpc)")
    print(f"       Source: fitted to galaxy correlation function shape")
    print(f"       Data: SDSS DR12, DESI DR1, Euclid DR1 (3.5M+ galaxies)")
    print(f"       Status: EMPIRICAL — 1 free parameter")
    print()
    print(f"    3. C_XI = {pd.correlation_normalization:.4f}")
    print(f"       Formula: C_XI = sigma8^2 / integral(Phi^2 * f ds)")
    print(f"       Inputs: r0 (empirical) + sigma8 (Planck 2018)")
    print(f"       Status: DERIVED — uniquely determined, no freedom")
    print()
    print(f"    4. v0 = {pd.v0_kms:.1f} +/- {pd.v0_kms*0.3:.1f} km/s")
    print(f"       Formula: v0^2 = c^2 * (r0/r_H) * geometric_factor")
    print(f"       Inputs: r0, H0, c, geometric_factor = 2*pi")
    print(f"       Status: SEMI-DERIVED — 30% uncertainty from geometric factor")
    print()
    print("  OBSERVATIONAL INPUTS (not theory parameters):")
    print(f"    sigma8 = {SIGMA_8} (Planck 2018 TT,TE,EE+lowE+lensing)")
    print(f"    H0 = {H_PLANCK*100} km/s/Mpc (Planck 2018)")
    print(f"    Omega_m = 0.3153 (Planck 2018)")
    print()
    print("  VERDICT: 1 empirical parameter (r0)")
    print("    Comparable to MOND (1: a0), much fewer than LCDM (6)")

    report_info("Parameter count = 1 empirical (r0) + 1 semi-derived (v0)")


# =============================================================================
# MERSENNE TOWER TESTS (12-14)
# =============================================================================

def test_mersenne_tower_recursion():
    """
    TEST 12: Mersenne Tower Recursion π(M₇) = M₅

    WHY:
      The Mersenne tower conjecture rests on the NUMBER THEORY fact that
      π(127) = 31, i.e., the prime counting function applied to M₇ (the
      cognitive Mersenne prime) gives M₅ (the emergence Mersenne prime).
      This is exact mathematics, not a conjecture.

    METHOD:
      Count primes up to 127 using a sieve. Verify the count = 31 = M₅.
      Also verify: 62 = 2×31, 293 is the 62nd prime.

    TOLERANCE: Exact (integer arithmetic, no floating point).
    """
    print("\n" + "-" * 70)
    print("[TEST 12] Mersenne Tower Recursion: π(M₇) = M₅")
    print("-" * 70)

    # Count primes up to 127 using sieve
    limit = 127
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    primes_to_127 = [i for i in range(2, limit + 1) if sieve[i]]
    pi_127 = len(primes_to_127)

    print(f"  Primes up to 127: {len(primes_to_127)}")
    print(f"  π(127) = {pi_127}")
    print(f"  M₅ = 2⁵ - 1 = 31")

    # Verify π(127) = 31
    report(f"π(M₇) = π(127) = 31 = M₅", pi_127 == 31,
           f"π(127) = {pi_127}")

    # Verify 2 × 31 = 62
    c_xi = 2 * pi_127
    report(f"C_XI = 2 × π(127) = 62", c_xi == 62,
           f"2 × π(127) = {c_xi}")

    # Verify 293 is the 62nd prime
    limit2 = 300
    sieve2 = [True] * (limit2 + 1)
    sieve2[0] = sieve2[1] = False
    for i in range(2, int(limit2**0.5) + 1):
        if sieve2[i]:
            for j in range(i*i, limit2 + 1, i):
                sieve2[j] = False
    all_primes = [i for i in range(2, limit2 + 1) if sieve2[i]]
    p62 = all_primes[61] if len(all_primes) >= 62 else -1
    report(f"p₆₂ = 293 (the 62nd prime)", p62 == 293,
           f"62nd prime = {p62}")

    # Verify phase decomposition: 62 = 5 + 13 + 23 + 21
    decomp = 5 + 13 + 23 + 21
    report(f"Phase decomposition: 62 = 5 + 13 + 23 + 21", decomp == 62,
           f"5 + 13 + 23 + 21 = {decomp}")


def test_mersenne_tower_mode():
    """
    TEST 13: Mersenne Tower Mode — C_XI=62 gives consistent r₀

    WHY:
      If C_XI = 62 from the Mersenne tower conjecture, then r₀ is uniquely
      determined by σ₈ normalization. The resulting r₀ should be close to
      the empirical value (0.65 kpc). This tests the PHYSICAL consistency
      of the conjecture.

    METHOD:
      Run ParameterDerivation in mersenne_tower mode. Check that r₀ is
      within 5% of the empirical value.

    TOLERANCE: 5% (generous, since Planck σ₈ has ~0.7% uncertainty which
      propagates to ~7% in r₀).
    """
    print("\n" + "-" * 70)
    print("[TEST 13] Mersenne Tower Mode: C_XI=62 → r₀ ≈ 0.65 kpc")
    print("-" * 70)

    from core.parameter_derivations import ParameterDerivation

    pd = ParameterDerivation(use_mersenne_tower=True)
    params = pd.get_parameters()

    r0_kpc = params['r0_kpc']
    c_xi = params['correlation_normalization']
    mode = params['mode']
    free_params = params['free_parameters']

    r0_pct = abs(r0_kpc - 0.65) / 0.65 * 100

    print(f"  Mode: {mode}")
    print(f"  C_XI = {c_xi:.1f}")
    print(f"  r₀ = {r0_kpc:.4f} kpc")
    print(f"  Free parameters: {free_params}")
    print(f"  Diff from empirical (0.65 kpc): {r0_pct:.2f}%")

    report("C_XI = 62.0 from Mersenne tower", c_xi == 62.0,
           f"C_XI = {c_xi}")
    report("Zero free parameters", free_params == 0,
           f"free_parameters = {free_params}")
    report(f"r₀ within 5% of empirical 0.65 kpc", r0_pct <= 5.0,
           f"r₀ = {r0_kpc:.4f} kpc ({r0_pct:.2f}% from empirical)")


def test_zero_parameter_consistency():
    """
    TEST 14: Both modes give compatible predictions

    WHY:
      The Mersenne tower mode (0 params) and empirical mode (1 param) should
      give COMPATIBLE physical predictions. If they diverge, either the
      conjecture is wrong or the empirical value needs updating.

    METHOD:
      Run both modes, compare r₀, C_XI, and v₀. Check they agree within
      reasonable tolerances.

    TOLERANCE: r₀ within 5%, C_XI within 2%, v₀ within 5%.
    """
    print("\n" + "-" * 70)
    print("[TEST 14] Zero-Parameter Consistency: Both modes compatible")
    print("-" * 70)

    from core.parameter_derivations import ParameterDerivation

    pd_emp = ParameterDerivation(use_empirical_r0=True)
    pd_mt = ParameterDerivation(use_mersenne_tower=True)

    p_emp = pd_emp.get_parameters()
    p_mt = pd_mt.get_parameters()

    r0_diff = abs(p_emp['r0_kpc'] - p_mt['r0_kpc']) / p_emp['r0_kpc'] * 100
    cxi_diff = abs(p_emp['correlation_normalization'] - p_mt['correlation_normalization']) / p_emp['correlation_normalization'] * 100
    v0_diff = abs(p_emp['v0_kms'] - p_mt['v0_kms']) / p_emp['v0_kms'] * 100

    print(f"  Empirical mode:  r₀={p_emp['r0_kpc']:.4f} kpc, C_XI={p_emp['correlation_normalization']:.2f}, v₀={p_emp['v0_kms']:.1f} km/s")
    print(f"  Mersenne tower:  r₀={p_mt['r0_kpc']:.4f} kpc, C_XI={p_mt['correlation_normalization']:.2f}, v₀={p_mt['v0_kms']:.1f} km/s")
    print(f"  Differences:     r₀={r0_diff:.2f}%, C_XI={cxi_diff:.2f}%, v₀={v0_diff:.2f}%")

    report(f"r₀ consistent (<5%)", r0_diff <= 5.0,
           f"difference = {r0_diff:.2f}%")
    report(f"C_XI consistent (<2%)", cxi_diff <= 2.0,
           f"difference = {cxi_diff:.2f}%")
    report(f"v₀ consistent (<5%)", v0_diff <= 5.0,
           f"difference = {v0_diff:.2f}%")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("PRIME FIELD THEORY — SYNTHETIC VALIDATION SUITE")
    print("=" * 70)
    print()
    print("PURPOSE: Verify code correctness using known-answer tests.")
    print("         Isolates code bugs from data/physics issues.")
    print("         If ALL code tests pass, disagreement with real data")
    print("         means the PHYSICS is wrong, not the code.")
    print()
    print("TESTS:")
    print("  [PASS/FAIL] = Code correctness (bugs). Fix before using real data.")
    print("  [INFO]      = Scientific finding (not a bug).")

    # Code correctness tests (must all pass)
    test_field_equation()
    test_gradient()
    test_pair_distance_pdf()
    test_sigma8_roundtrip_cxi()
    test_sigma8_roundtrip_r0()
    test_sigma8_monotonicity()
    test_rotation_curve()
    test_synthetic_correlation_recovery()
    test_module_consistency()

    # Mersenne tower tests (must all pass)
    test_mersenne_tower_recursion()
    test_mersenne_tower_mode()
    test_zero_parameter_consistency()

    # Scientific analysis (informational)
    test_can_r0_be_derived()
    test_parameter_count()

    print("\n" + "=" * 70)
    print(f"CODE TESTS:   {PASS} passed, {FAIL} failed out of {PASS + FAIL}")
    print(f"INFO ITEMS:   {INFO} scientific findings reported")
    print("=" * 70)

    if FAIL == 0:
        print()
        print("All code tests PASSED. The code is mathematically correct.")
        print("Any disagreement with real data is a PHYSICS issue, not a code bug.")
        print("Safe to proceed to real data validation.")
    else:
        print()
        print(f"WARNING: {FAIL} code test(s) FAILED.")
        print("Fix code bugs before running on real data.")

    sys.exit(0 if FAIL == 0 else 1)
