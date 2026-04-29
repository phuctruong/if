"""Tests for the Mersenne Tower Theorem (C_XI = 62 derivation).

Run with: pytest tests/test_mersenne_tower.py

These tests do NOT hardcode pi(127) = 31. They compute everything from primitives
to fail loudly if the algebra drifts.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _is_prime(n: int) -> bool:
    """Trial division. Adequate for n up to ~1e8."""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    d = 3
    while d * d <= n:
        if n % d == 0:
            return False
        d += 2
    return True


def _primepi_native(n: int) -> int:
    """pi(n) by enumeration — no sympy dependency."""
    return sum(1 for k in range(2, n + 1) if _is_prime(k))


# Known Mersenne prime exponents (primes p such that 2^p - 1 is prime).
# Source: GIMPS-verified list as of 2024. The first 11 are listed for tests.
KNOWN_MERSENNE_EXPONENTS = [2, 3, 5, 7, 13, 17, 19, 31, 61, 89, 107, 127, 521, 607]


def test_lemma_L1_pi_127():
    """L1: pi(127) = 31, computed from scratch (no sympy, no hardcode)."""
    assert _primepi_native(127) == 31


def test_lemma_L2_31_is_M5():
    """L2: 31 = 2^5 - 1 = M_5 with both 5 and 31 prime."""
    assert _is_prime(5)
    assert _is_prime(31)
    assert 2**5 - 1 == 31


def test_lemma_L3_M7_uniquely_tower_closed_in_small_range():
    """L3 (key): M_7 = 127 is the unique tower-closed Mersenne prime among
    the first several known Mersenne primes.

    A Mersenne prime M_p = 2^p - 1 is "tower-closed" iff pi(M_p) is itself
    a Mersenne prime. This test verifies uniqueness for all p in
    KNOWN_MERSENNE_EXPONENTS where exact computation is tractable
    (M_p < 1e8).
    """
    KNOWN_MERSENNE_PRIMES = {2**p - 1 for p in KNOWN_MERSENNE_EXPONENTS}
    tower_closed = []
    for p in KNOWN_MERSENNE_EXPONENTS:
        Mp = 2**p - 1
        if Mp >= 1e8:
            # exact pi(Mp) is computationally infeasible here; covered by
            # the asymptotic argument in mersenne_tower_theorem.py main script
            continue
        pi_Mp = _primepi_native(Mp)
        if pi_Mp in KNOWN_MERSENNE_PRIMES:
            tower_closed.append((p, Mp, pi_Mp))
    # Exactly one match: M_7 = 127, pi(127) = 31 = M_5
    assert len(tower_closed) == 1, f"Expected 1 tower-closed Mersenne, got {tower_closed}"
    assert tower_closed[0] == (7, 127, 31)


def test_C_XI_equals_62():
    """C_XI = 2 * pi(127) = 62 — the headline number."""
    pi_127 = _primepi_native(127)
    assert 2 * pi_127 == 62


def test_C_XI_matches_canonical_constant():
    """The C_XI we compute matches prime_field_util.C_XI_CANONICAL."""
    from prime_field_util import C_XI_CANONICAL
    assert C_XI_CANONICAL == 2 * _primepi_native(127)


def test_lemma_L3_module_reports_one():
    """The mersenne_tower_theorem module's verify_lemma_3 internal assert that
    exactly one tower-closed Mersenne prime exists must hold."""
    # This re-runs the project's own assertion; it should pass.
    from mersenne_tower_theorem import verify_lemma_3
    results = verify_lemma_3()
    # results is a list of (p, Mp, pi_Mp, is_tower_closed, exact); count tower-closed
    tower_closed = [r for r in results if r[3]]
    assert len(tower_closed) == 1, f"Expected 1 tower-closed, got {tower_closed}"
    assert tower_closed[0][0] == 7  # p = 7
