"""Carryover selection and marginal occupancies. These run without fulqrum installed."""

from __future__ import annotations

import numpy as np
import pytest
from qcsc_prefect_fulqrum.carryover import (
    marginal_weights,
    occupancies_from_marginals,
    select_carryover,
)

# Alpha strings 0b0011, 0b0101, 0b1100 (rows) x beta strings 0b0011, 0b1010 (columns).
STRS_A = np.array([0b0011, 0b0101, 0b1100], dtype=np.int64)
STRS_B = np.array([0b0011, 0b1010], dtype=np.int64)
PROBS = np.array([[0.40, 0.10], [0.05, 0.05], [0.30, 0.10]])


def test_marginal_weights_accept_matrix_and_fulqrum_basis_order():
    w_a, w_b = marginal_weights(PROBS, 3, 2)
    np.testing.assert_allclose(w_a, [0.50, 0.10, 0.40])
    np.testing.assert_allclose(w_b, [0.75, 0.25])
    # fulqrum's basis index is ib * na + ia.
    flat = PROBS.T.reshape(-1)
    for got, want in zip(marginal_weights(flat, 3, 2), (w_a, w_b), strict=True):
        np.testing.assert_allclose(got, want)


def test_marginal_weights_reject_other_shapes():
    with pytest.raises(ValueError, match="shape"):
        marginal_weights(np.ones(5), 3, 2)


def test_select_carryover_keeps_the_heaviest_fraction_per_spin():
    w_a, w_b = marginal_weights(PROBS, 3, 2)
    ca, cb = select_carryover(w_a, w_b, STRS_A, STRS_B, 0.7)
    # int(0.7 * 3) = 2 alpha strings, int(0.7 * 2) = 1 beta string, heaviest first.
    np.testing.assert_array_equal(ca, [0b0011, 0b1100])
    np.testing.assert_array_equal(cb, [0b0011])
    assert ca.dtype == STRS_A.dtype


@pytest.mark.parametrize("fraction, sizes", [(0.0, (0, 0)), (1.0, (3, 2)), (0.2, (0, 0))])
def test_select_carryover_bounds(fraction, sizes):
    w_a, w_b = marginal_weights(PROBS, 3, 2)
    ca, cb = select_carryover(w_a, w_b, STRS_A, STRS_B, fraction)
    assert (len(ca), len(cb)) == sizes


def test_select_carryover_breaks_ties_in_subspace_order():
    strs = np.array([1, 2, 4, 8])
    ca, _ = select_carryover(np.full(4, 0.25), np.ones(1), strs, np.array([1]), 0.5)
    np.testing.assert_array_equal(ca, [1, 2])


def test_select_carryover_keeps_wide_strings():
    big = (1 << 69) | 1
    strs = np.array([3, big], dtype=object)
    ca, _ = select_carryover(np.array([0.1, 0.9]), np.ones(1), strs, np.array([3]), 0.5)
    assert ca.dtype == object and [int(x) for x in ca] == [big]


def test_select_carryover_rejects_bad_input():
    w_a, w_b = marginal_weights(PROBS, 3, 2)
    with pytest.raises(ValueError, match="fraction"):
        select_carryover(w_a, w_b, STRS_A, STRS_B, 1.5)
    with pytest.raises(ValueError, match="length"):
        select_carryover(w_a[:2], w_b, STRS_A, STRS_B, 0.5)


def test_occupancies_from_marginals_match_the_full_state():
    w_a, w_b = marginal_weights(PROBS, 3, 2)
    occ_a, occ_b = occupancies_from_marginals(w_a, w_b, STRS_A, STRS_B, 4)
    # Orbital i is occupied in the strings with bit i set.
    np.testing.assert_allclose(occ_a, [0.60, 0.50, 0.50, 0.40])
    np.testing.assert_allclose(occ_b, [0.75, 1.00, 0.00, 0.25])
    assert occ_a.sum() == pytest.approx(2.0)
    assert occ_b.sum() == pytest.approx(2.0)
