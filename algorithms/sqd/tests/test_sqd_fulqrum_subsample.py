from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("qiskit_addon_sqd")

from qiskit_addon_sqd.fermion import bitstring_matrix_to_ci_strs  # noqa: E402
from sqd_fulqrum.subsample import (  # noqa: E402
    bitstrings_to_ci_strs,
    postselect,
    subsample,
)


def _random_bitstrings(rng, num, norb, n_alpha, n_beta, invalid=0):
    rows = []
    for _ in range(num):
        a = np.zeros(norb, dtype=bool)
        b = np.zeros(norb, dtype=bool)
        a[rng.choice(norb, n_alpha, replace=False)] = True
        b[rng.choice(norb, n_beta, replace=False)] = True
        # Row layout: beta half (left) then alpha half (right), orbital 0 rightmost.
        rows.append(np.concatenate((b[::-1], a[::-1])))
    for _ in range(invalid):
        rows.append(rng.integers(0, 2, size=2 * norb).astype(bool))
    return np.array(rows, dtype=bool)


def test_ci_strs_match_qiskit_addon_sqd():
    rng = np.random.default_rng(1)
    bs = _random_bitstrings(rng, 50, 6, 3, 2)
    ci_a, ci_b = bitstrings_to_ci_strs(bs)
    ref_a, ref_b = bitstring_matrix_to_ci_strs(bs, open_shell=True)
    assert ci_a.dtype == np.int64 and len(ci_a) == len(bs)
    np.testing.assert_array_equal(np.unique(ci_a), ref_a)
    np.testing.assert_array_equal(np.unique(ci_b), ref_b)
    assert all(int(x).bit_count() == 3 for x in ci_a)
    assert all(int(x).bit_count() == 2 for x in ci_b)


def test_ci_strs_keep_python_ints_for_large_norb():
    norb = 70
    row = np.zeros(2 * norb, dtype=bool)
    row[norb] = True  # alpha orbital norb-1
    row[2 * norb - 1] = True  # alpha orbital 0
    row[0] = True  # beta orbital norb-1
    ci_a, ci_b = bitstrings_to_ci_strs(row[None, :])
    assert ci_a.dtype == object
    assert ci_a[0] == (1 << (norb - 1)) | 1
    assert ci_b[0] == 1 << (norb - 1)


def test_postselect_filters_and_normalizes():
    rng = np.random.default_rng(2)
    bs = _random_bitstrings(rng, 20, 5, 2, 2, invalid=10)
    probs = rng.random(len(bs))
    bs_post, probs_post = postselect(bs, probs, hamming_right=2, hamming_left=2)
    assert len(bs_post) >= 20
    assert np.all(bs_post[:, 5:].sum(axis=1) == 2)
    assert np.all(bs_post[:, :5].sum(axis=1) == 2)
    assert probs_post.sum() == pytest.approx(1.0)


def test_closed_shell_small_set_yields_one_full_batch():
    rng = np.random.default_rng(3)
    bs = _random_bitstrings(rng, 30, 5, 2, 2)
    probs = np.full(len(bs), 1 / len(bs))
    batches = list(subsample(bs, probs, subspace_dim=10_000, num_batches=4, rng=rng))
    assert len(batches) == 1
    (a, b), *_ = batches
    ci_a, ci_b = bitstrings_to_ci_strs(bs)
    np.testing.assert_array_equal(a, np.union1d(ci_a, ci_b))
    np.testing.assert_array_equal(a, b)


def test_closed_shell_subsamples_sqrt_dim_without_replacement():
    rng = np.random.default_rng(4)
    bs = _random_bitstrings(rng, 200, 8, 3, 3)
    probs = np.full(len(bs), 1 / len(bs))
    batches = list(subsample(bs, probs, subspace_dim=100, num_batches=3, rng=rng))
    assert len(batches) == 3
    for a, b in batches:
        assert a is b
        assert len(a) == 10 and len(np.unique(a)) == 10


def test_open_shell_keeps_spins_separate_and_nonempty():
    rng = np.random.default_rng(5)
    bs = _random_bitstrings(rng, 300, 8, 4, 2)
    probs = rng.random(len(bs))
    probs /= probs.sum()
    batches = list(subsample(bs, probs, subspace_dim=60, num_batches=2, rng=rng, open_shell=True))
    assert len(batches) == 2
    for a, b in batches:
        assert all(int(x).bit_count() == 4 for x in a)
        assert all(int(x).bit_count() == 2 for x in b)
        assert len(a) * len(b) <= 60
    # A tiny target still leaves at least one string per spin.
    ((a, b),) = subsample(bs, probs, subspace_dim=1, num_batches=1, rng=rng, open_shell=True)
    assert len(a) == 1 and len(b) == 1


def test_subsample_is_deterministic_for_a_seed():
    bs = _random_bitstrings(np.random.default_rng(6), 200, 8, 3, 3)
    probs = np.full(len(bs), 1 / len(bs))
    first = list(subsample(bs, probs, 64, 2, np.random.default_rng(9)))
    second = list(subsample(bs, probs, 64, 2, np.random.default_rng(9)))
    for (a1, _), (a2, _) in zip(first, second, strict=True):
        np.testing.assert_array_equal(a1, a2)
