from __future__ import annotations

import numpy as np
import pytest

import prime_field_util
from prime_field_util import PairCounter


@pytest.mark.skipif(not prime_field_util.NUMBA_AVAILABLE, reason="Numba is not installed")
def test_numba_auto_pair_counter_matches_standard_counting() -> None:
    positions = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [2, 0, 0],
            [0, 1, 0],
            [0, 2, 0],
        ],
        dtype=float,
    )
    bins = np.array([0.5, 1.5, 2.5])
    bins_squared = bins * bins
    expected = PairCounter.count_pairs_auto(positions, bins, use_numba=False)

    actual = prime_field_util.numba_count_pairs_auto(positions, bins_squared, len(bins) - 1)

    assert np.array_equal(actual, expected)


@pytest.mark.skipif(not prime_field_util.NUMBA_AVAILABLE, reason="Numba is not installed")
def test_numba_auto_pair_counter_is_deterministic() -> None:
    rng = np.random.default_rng(65537)
    positions = rng.random((160, 3))
    bins = np.array([0.0, 0.2, 0.4, 0.6, 0.8])
    expected = PairCounter.count_pairs_auto(positions, bins, use_numba=False)

    for _ in range(5):
        actual = PairCounter.count_pairs_auto(positions, bins, use_numba=True)
        assert np.array_equal(actual, expected)
