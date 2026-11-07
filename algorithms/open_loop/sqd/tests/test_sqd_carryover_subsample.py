"""Subsampling with carryover CI strings (no Prefect, no solver)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("qiskit_addon_sqd")

from sqd_common.subsample import (  # noqa: E402
    _draw_with_forced,
    _unique_and_accumulate_probs,
    bitstrings_to_ci_strs,
    subsample,
)


def _random_bitstrings(rng, num, norb, n_alpha, n_beta):
    rows = []
    for _ in range(num):
        a = np.zeros(norb, dtype=bool)
        b = np.zeros(norb, dtype=bool)
        a[rng.choice(norb, n_alpha, replace=False)] = True
        b[rng.choice(norb, n_beta, replace=False)] = True
        # Row layout: beta half (left) then alpha half (right), orbital 0 rightmost.
        rows.append(np.concatenate((b[::-1], a[::-1])))
    return np.array(rows, dtype=bool)


def _samples(seed=0, num=300, norb=8, nelec=(3, 3)):
    rng = np.random.default_rng(seed)
    bs = _random_bitstrings(rng, num, norb, *nelec)
    probs = rng.random(len(bs))
    return bs, probs / probs.sum()


def _plain_subsample(bitstring_matrix, probabilities, subspace_dim, num_batches, rng, open_shell):
    """The subsampling without carryover, as a reference for the RNG parity."""
    ci_strs_a, ci_strs_b = bitstrings_to_ci_strs(bitstring_matrix)
    if not open_shell:
        strs, probs = _unique_and_accumulate_probs(
            np.concatenate((ci_strs_a, ci_strs_b)), np.tile(probabilities, 2) / 2.0
        )
        if len(strs) ** 2 <= subspace_dim:
            return [(strs, strs)]
        out = []
        for _ in range(num_batches):
            s = rng.choice(strs, size=max(1, int(np.sqrt(subspace_dim))), replace=False, p=probs)
            out.append((s, s))
        return out
    sa, pa = _unique_and_accumulate_probs(ci_strs_a, probabilities)
    sb, pb = _unique_and_accumulate_probs(ci_strs_b, probabilities)
    net_dim = len(sa) * len(sb)
    if net_dim <= subspace_dim:
        return [(sa, sb)]
    rate = np.sqrt(subspace_dim / net_dim)
    return [
        (
            rng.choice(sa, size=max(1, int(len(sa) * rate)), replace=False, p=pa),
            rng.choice(sb, size=max(1, int(len(sb) * rate)), replace=False, p=pb),
        )
        for _ in range(num_batches)
    ]


@pytest.mark.parametrize("open_shell", [False, True])
@pytest.mark.parametrize(
    "carryover", [None, (np.zeros(0, np.int64), np.zeros(0, np.int64))], ids=["none", "empty"]
)
@pytest.mark.parametrize("subspace_dim", [49, 10**6])
def test_no_carryover_draws_the_same_batches_and_rng_stream(open_shell, carryover, subspace_dim):
    nelec = (4, 2) if open_shell else (3, 3)
    bs, probs = _samples(nelec=nelec)
    rng, ref_rng = np.random.default_rng(7), np.random.default_rng(7)
    got = list(subsample(bs, probs, subspace_dim, 3, rng, open_shell, carryover=carryover))
    ref = _plain_subsample(bs, probs, subspace_dim, 3, ref_rng, open_shell)
    assert len(got) == len(ref)
    for (a, b), (ra, rb) in zip(got, ref, strict=True):
        assert a.dtype == ra.dtype and b.dtype == rb.dtype
        np.testing.assert_array_equal(a, ra)
        np.testing.assert_array_equal(b, rb)
    assert rng.bit_generator.state == ref_rng.bit_generator.state


def _ci(*orbitals):
    return sum(1 << o for o in orbitals)


def _absent(strings, norb, weight):
    """A CI string of ``weight`` electrons that ``strings`` lacks."""
    present = {int(x) for x in strings}
    return next(x for x in range(1 << norb) if x.bit_count() == weight and x not in present)


def test_closed_shell_carryover_comes_first_and_keeps_the_size():
    bs, probs = _samples(num=40, norb=10)
    ci_a, ci_b = bitstrings_to_ci_strs(bs)
    pool = set(np.union1d(ci_a, ci_b).tolist())
    # A carryover string that no sample has is kept too.
    absent = _absent(pool, 10, 3)
    sampled = sorted(pool)[:2]
    carry_a = np.array([sampled[0], absent, sampled[0]], dtype=np.int64)
    carry_b = np.array([absent, sampled[1]], dtype=np.int64)

    batches = list(
        subsample(bs, probs, 36, 4, np.random.default_rng(3), carryover=(carry_a, carry_b))
    )
    assert len(batches) == 4
    for a, b in batches:
        assert a is b
        # Alpha then beta strings, without repeats, then sampled ones.
        assert a[:3].tolist() == [sampled[0], absent, sampled[1]]
        assert len(a) == 6 and len(np.unique(a)) == 6
        assert set(a[3:].tolist()) <= pool - {sampled[0], sampled[1]}
        assert a.dtype == np.int64


def test_carryover_with_a_wrong_electron_number_is_dropped():
    bs, probs = _samples(nelec=(4, 2))
    good_a, good_b, bad = _ci(0, 1, 2, 7), _ci(3, 6), _ci(0, 1, 2)
    batches = list(
        subsample(
            bs,
            probs,
            60,
            2,
            np.random.default_rng(1),
            open_shell=True,
            carryover=(np.array([bad, good_a]), np.array([good_b, good_a, bad])),
        )
    )
    for a, b in batches:
        assert a[0] == good_a and bad not in a.tolist()
        assert b[0] == good_b and good_a not in b.tolist() and bad not in b.tolist()
        assert all(int(x).bit_count() == 4 for x in a)
        assert all(int(x).bit_count() == 2 for x in b)


def test_nelec_defaults_to_the_sample_weights():
    bs, probs = _samples(nelec=(4, 2))
    carry = (np.array([_ci(0, 1)]), np.array([_ci(0, 1)]))
    explicit = list(
        subsample(bs, probs, 60, 2, np.random.default_rng(1), True, carry, nelec=(4, 2))
    )
    implied = list(subsample(bs, probs, 60, 2, np.random.default_rng(1), True, carry))
    for (a, b), (ia, ib) in zip(explicit, implied, strict=True):
        np.testing.assert_array_equal(a, ia)
        np.testing.assert_array_equal(b, ib)
        assert _ci(0, 1) not in a.tolist() and b[0] == _ci(0, 1)


def test_carryover_is_truncated_to_the_batch_size():
    bs, probs = _samples()
    ci_a, ci_b = bitstrings_to_ci_strs(bs)
    heavy = np.union1d(ci_a, ci_b)[::-1][:12]
    rng = np.random.default_rng(5)
    before = rng.bit_generator.state
    batches = list(subsample(bs, probs, 25, 2, rng, carryover=(heavy, heavy)))
    for a, _ in batches:
        np.testing.assert_array_equal(a, heavy[:5])
    # A batch filled by carryover draws nothing.
    assert rng.bit_generator.state == before


def test_open_shell_carryover_counts_toward_each_spin():
    bs, probs = _samples(num=40, norb=10, nelec=(4, 2))
    ci_a, ci_b = bitstrings_to_ci_strs(bs)
    absent_b = _absent(ci_b, 10, 2)
    carry = (np.unique(ci_a)[:3], np.array([absent_b]))
    rng = np.random.default_rng(2)
    plain = list(subsample(bs, probs, 60, 3, np.random.default_rng(2), True))
    batches = list(subsample(bs, probs, 60, 3, rng, True, carryover=carry))
    for (a, b), (pa, pb) in zip(batches, plain, strict=True):
        np.testing.assert_array_equal(a[:3], carry[0])
        assert b[0] == absent_b
        assert len(np.unique(a)) == len(a) and len(np.unique(b)) == len(b)
        assert len(a) * len(b) <= 60
        # One more beta string in the pool can change the per-spin sizes by one at most.
        assert abs(len(a) - len(pa)) <= 1 and abs(len(b) - len(pb)) <= 1


def test_pool_smaller_than_the_remainder_is_taken_whole():
    pool = np.array([3, 5, 6], dtype=np.int64)
    probs = np.array([0.2, 0.3, 0.5])
    rng = np.random.default_rng(0)
    before = rng.bit_generator.state
    got = _draw_with_forced(pool, probs, np.array([12, 5]), size=6, rng=rng)
    assert got.tolist() == [12, 5, 3, 6]
    assert rng.bit_generator.state == before


def test_remainder_is_drawn_with_renormalized_probabilities():
    pool = np.array([3, 5, 6, 9], dtype=np.int64)
    # All weight left outside the forced string is on 9 and 6.
    probs = np.array([0.0, 0.7, 0.1, 0.2])
    for seed in range(20):
        got = _draw_with_forced(pool, probs, np.array([5]), size=3, rng=np.random.default_rng(seed))
        assert got[0] == 5 and sorted(got[1:].tolist()) == [6, 9]


def test_whole_set_with_carryover_is_one_union_batch():
    bs, probs = _samples(num=20)
    ci_a, ci_b = bitstrings_to_ci_strs(bs)
    pool = np.union1d(ci_a, ci_b)
    absent = _absent(pool, 8, 3)
    carry = (np.array([absent, pool[0]]), np.zeros(0, np.int64))
    batches = list(subsample(bs, probs, 10**6, 4, np.random.default_rng(0), carryover=carry))
    assert len(batches) == 1
    (a, b), *_ = batches
    np.testing.assert_array_equal(a, np.union1d(pool, [absent]))
    np.testing.assert_array_equal(b, a)

    batches = list(subsample(bs, probs, 10**6, 4, np.random.default_rng(0), True, carry))
    ((a, b),) = batches
    np.testing.assert_array_equal(a, np.union1d(ci_a, [absent, pool[0]]))
    np.testing.assert_array_equal(b, np.unique(ci_b))


def test_carryover_can_push_the_set_over_the_budget():
    bs, probs = _samples(num=20)
    ci_a, ci_b = bitstrings_to_ci_strs(bs)
    pool = np.union1d(ci_a, ci_b)
    present = set(pool.tolist())
    extra = [x for x in range(1 << 8) if x.bit_count() == 3 and x not in present][:3]
    dim = len(pool) ** 2
    batches = list(
        subsample(bs, probs, dim, 2, np.random.default_rng(0), carryover=(np.array(extra), None))
    )
    assert len(batches) == 2
    for a, _ in batches:
        assert a[:3].tolist() == extra and len(a) == len(pool)


def test_large_norb_carryover_keeps_python_ints():
    norb = 70
    rng = np.random.default_rng(0)
    bs = _random_bitstrings(rng, 40, norb, 2, 2)
    probs = np.full(len(bs), 1 / len(bs))
    big = (1 << 69) | (1 << 68)
    batches = list(
        subsample(bs, probs, 16, 2, rng, carryover=(np.array([big], dtype=object), None))
    )
    for a, _ in batches:
        assert a.dtype == object and a[0] == big and len(a) == 4
        assert all(isinstance(x, int) for x in a)
