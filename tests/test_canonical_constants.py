"""Tests for the canonical Prime Field Theory constants.

Run with: pytest tests/test_canonical_constants.py
"""

import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np


def test_pi_127_equals_31_from_scratch():
    """pi(127) = 31 computed by Eratosthenes-style enumeration, no sympy."""
    pi127 = sum(
        1 for n in range(2, 128)
        if all(n % d != 0 for d in range(2, int(n**0.5) + 1))
    )
    assert pi127 == 31, f"pi(127) = {pi127}, expected 31"


def test_31_is_M5():
    """31 = 2^5 - 1 = M_5, and 5 is prime, and 31 is prime — so 31 is a Mersenne prime."""
    assert 2**5 - 1 == 31
    # 5 prime
    assert all(5 % d != 0 for d in range(2, 3))
    # 31 prime
    assert all(31 % d != 0 for d in range(2, 6))


def test_C_XI_canonical_equals_62():
    """C_XI = 2 * pi(M_7) = 2 * 31 = 62."""
    from prime_field_util import C_XI_CANONICAL, PI_M7
    assert PI_M7 == 31
    assert C_XI_CANONICAL == 62
    assert C_XI_CANONICAL == 2 * PI_M7


def test_R0_kpc_canonical_close_to_documented():
    """R0_KPC_CANONICAL within 0.1% of the documented zero-parameter value 0.6595 kpc."""
    from prime_field_util import R0_KPC_CANONICAL
    documented = 0.6595
    assert abs(R0_KPC_CANONICAL - documented) / documented < 1e-3


def test_R0_kpc_matches_prime_field_theory():
    """The canonical r_0 in prime_field_util matches prime_field_theory.PrimeFieldTheory zero-param mode."""
    import logging
    logging.basicConfig(level=logging.WARNING)
    from prime_field_util import R0_KPC_CANONICAL
    from prime_field_theory import PrimeFieldTheory
    pft = PrimeFieldTheory(use_mersenne_tower=True)
    assert abs(pft.r0_kpc - R0_KPC_CANONICAL) < 1e-9, \
        f"prime_field_theory r0_kpc = {pft.r0_kpc}, util R0_KPC_CANONICAL = {R0_KPC_CANONICAL}"


def test_correlation_model_at_canonical_r0():
    """xi(r=r_0) = 1/log(2)^2 when amplitude=bias=r0_factor=1."""
    from prime_field_util import R0_KPC_CANONICAL, prime_field_correlation_model
    r_mpc = R0_KPC_CANONICAL / 1000.0  # convert kpc to Mpc
    xi = prime_field_correlation_model(np.array([r_mpc]))
    expected = 1.0 / np.log(2.0) ** 2
    assert abs(xi[0] - expected) < 1e-10, \
        f"xi(r0) = {xi[0]}, expected {expected}"


def test_correlation_model_default_kwargs_are_canonical():
    """Default amplitude=1, bias=1, r0_factor=1 gives the canonical (zero-deviation) prediction."""
    from prime_field_util import prime_field_correlation_model
    # At r = 1 Mpc, xi(r) should be 1/log(1000/0.6595 + 1)^2 ≈ very small but well-defined
    xi = prime_field_correlation_model(np.array([1.0]))
    assert xi[0] > 0  # field is positive
    assert xi[0] < 1  # at 1 Mpc we're way past r_0 in kpc, so field is small
