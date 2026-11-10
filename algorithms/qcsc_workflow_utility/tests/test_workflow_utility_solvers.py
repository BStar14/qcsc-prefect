"""Solver selection by slug and the open-loop conversion to the SBD input form."""

from __future__ import annotations

import numpy as np
import pytest
from qcsc_workflow_utility import (
    DICE_SLUG,
    SBD_SLUG,
    parse_block_ref,
    solver_class,
    to_sbd_closed_shell,
)
from qcsc_workflow_utility.solvers import to_sbd_subspace


class Dice:
    pass


class Sbd:
    pass


CLASSES = {DICE_SLUG: Dice, SBD_SLUG: Sbd}


@pytest.mark.parametrize(
    "ref,expected",
    [
        ("dice_shci_solver_job/sqd-solver", (Dice, "sqd-solver")),
        ("sbd_solver_job/davidson-solver", (Sbd, "davidson-solver")),
        # Prefect writes slugs with hyphens.
        ("sbd-solver-job/davidson-solver", (Sbd, "davidson-solver")),
        ("dice-shci-solver-job/a/b", (Dice, "a/b")),
    ],
)
def test_slug_selects_the_solver_class(ref, expected):
    assert solver_class(ref, CLASSES) == expected


@pytest.mark.parametrize("ref", ["other_solver/x", "sbd_solver_job", "/x", "sbd_solver_job/"])
def test_unknown_or_malformed_refs_are_rejected(ref):
    with pytest.raises(ValueError):
        solver_class(ref, CLASSES)


def test_parse_block_ref():
    assert parse_block_ref("sbd_solver_job/davidson-solver") == (
        "sbd_solver_job",
        "davidson-solver",
    )


HF = 0b0011  # Two electrons per spin.


def convert(batches, subspace_dim, seed=0, probabilities=None, keep=None):
    return list(
        to_sbd_closed_shell(
            batches,
            subspace_dim=subspace_dim,
            num_elec_a=2,
            probabilities=probabilities,
            rng=np.random.default_rng(seed),
            keep=keep,
        )
    )


@pytest.mark.parametrize("subspace_dim", [1, 4, 9, 10, 16, 25, 1000])
@pytest.mark.parametrize("with_hf", [False, True])
def test_converted_subspace_keeps_the_limit_and_puts_hf_first(subspace_dim, with_hf):
    determinants = np.array([0b0101, 0b0110, 0b1001, 0b1010, 0b1100] + ([HF] if with_hf else []))
    np.random.default_rng(1).shuffle(determinants)

    for alpha, beta in convert([(determinants, determinants)] * 3, subspace_dim):
        assert np.array_equal(alpha, beta)
        assert alpha[0] == HF
        assert list(alpha).count(HF) == 1
        # One slot for HF, int(sqrt(subspace_dim)) - 1 for the others.
        assert len(alpha) <= max(1, int(np.sqrt(subspace_dim)))
        assert len(alpha) ** 2 <= max(1, subspace_dim)
        assert set(alpha[1:]) <= set(determinants) - {HF}
        assert len(set(alpha)) == len(alpha)


def test_small_subspace_keeps_every_determinant():
    determinants = np.array([0b0101, HF, 0b0110])
    ((alpha, beta),) = convert([(determinants, determinants)], subspace_dim=100)
    assert alpha.tolist() == [HF, 0b0101, 0b0110]


def test_subsampling_follows_the_probabilities():
    determinants = np.array([0b0101, 0b0110, 0b1001, 0b1010])
    probabilities = {0b0101: 1.0, 0b0110: 0.0, 0b1001: 0.0, 0b1010: 0.0}
    ((alpha, _),) = convert([(determinants, determinants)], 4, probabilities=probabilities)
    assert alpha.tolist() == [HF, 0b0101]


def test_open_shell_batches_pass_through_unchanged():
    alpha, beta = np.array([0b0101, 0b0011]), np.array([0b0001, 0b0010])
    ((out_a, out_b),) = convert([(alpha, beta)], subspace_dim=4)
    assert out_a.tolist() == alpha.tolist()
    assert out_b.tolist() == beta.tolist()


@pytest.mark.parametrize("seed", range(20))
def test_kept_strings_survive_the_hartree_fock_slot(seed):
    # A carryover-first batch of 3 strings without HF; subspace_dim 9 leaves 2 other slots.
    determinants = np.array([0b1100, 0b0101, 0b1010])
    ((alpha, beta),) = convert(
        [(determinants, determinants)], subspace_dim=9, seed=seed, keep=[0b1100, 0b0101]
    )
    assert alpha.tolist() == [HF, 0b1100, 0b0101]
    np.testing.assert_array_equal(alpha, beta)


def test_kept_strings_are_truncated_to_the_limit_in_batch_order():
    determinants = np.array([0b1100, 0b0101, 0b1010, 0b0110])
    ((alpha, _),) = convert([(determinants, determinants)], subspace_dim=9, keep=determinants)
    assert alpha.tolist() == [HF, 0b1100, 0b0101]


def test_without_kept_strings_the_draw_is_unchanged():
    determinants = np.array([0b0101, 0b0110, 0b1001, 0b1010, 0b1100])
    for seed in range(5):
        assert [a.tolist() for a, _ in convert([(determinants, determinants)], 9, seed)] == [
            a.tolist() for a, _ in convert([(determinants, determinants)], 9, seed, keep=[])
        ]


# ------------------------------------------------------------------ to_sbd_subspace

HF_A, HF_B = 0b00111, 0b00011  # ROHF: three alpha and two beta electrons in 5 orbitals.
ALPHA = np.array([0b01011, 0b01101, 0b01110, 0b10011, 0b10101, 0b11001, 0b11100])
BETA = np.array([0b00101, 0b00110, 0b01001, 0b01010, 0b10001, 0b11000])


def convert_open(batches, subspace_dim, seed=0, keep=None, probabilities=None):
    return list(
        to_sbd_subspace(
            batches,
            subspace_dim=subspace_dim,
            num_elec_a=3,
            num_elec_b=2,
            probabilities=probabilities,
            rng=np.random.default_rng(seed),
            keep=keep,
        )
    )


@pytest.mark.parametrize("subspace_dim", [1, 2, 4, 9, 10, 16, 25, 1000])
@pytest.mark.parametrize("seed", range(3))
def test_closed_shell_batches_match_the_closed_shell_conversion(subspace_dim, seed):
    determinants = np.array([0b0101, 0b0110, 0b1001, 0b1010, 0b1100, HF])
    for keep in (None, [0b1100], (np.array([0b1010]), np.array([0b0110]))):
        flat = keep if not isinstance(keep, tuple) else [int(k) for ks in keep for k in ks]
        old = convert([(determinants, determinants)] * 2, subspace_dim, seed, keep=flat)
        new = list(
            to_sbd_subspace(
                [(determinants, determinants)] * 2,
                subspace_dim=subspace_dim,
                num_elec_a=2,
                num_elec_b=2,
                rng=np.random.default_rng(seed),
                keep=keep,
            )
        )
        assert [(a.tolist(), b.tolist()) for a, b in new] == [
            (a.tolist(), b.tolist()) for a, b in old
        ]


@pytest.mark.parametrize("subspace_dim", [1, 2, 3, 6, 10, 20, 49, 1000])
@pytest.mark.parametrize("seed", range(3))
def test_open_shell_lists_put_each_spins_hf_first_within_the_size(subspace_dim, seed):
    for alpha, beta in convert_open([(ALPHA, BETA)] * 3, subspace_dim, seed):
        assert alpha[0] == HF_A and beta[0] == HF_B
        assert list(alpha).count(HF_A) == 1 and list(beta).count(HF_B) == 1
        assert len(alpha) * len(beta) <= subspace_dim
        assert len(set(alpha)) == len(alpha) and len(set(beta)) == len(beta)
        assert set(alpha[1:]) <= set(ALPHA) and set(beta[1:]) <= set(BETA)


def test_open_shell_lists_that_fit_are_kept_whole():
    ((alpha, beta),) = convert_open([(ALPHA, BETA)], subspace_dim=(7 + 1) * (6 + 1))
    assert alpha.tolist() == [HF_A, *sorted(ALPHA)]
    assert beta.tolist() == [HF_B, *sorted(BETA)]


def test_open_shell_lists_are_cut_by_the_same_factor():
    # 8 x 7 with Hartree-Fock; subspace_dim 14 gives r = 0.5, so 4 x 3.
    ((alpha, beta),) = convert_open([(ALPHA, BETA)], subspace_dim=14)
    assert (len(alpha), len(beta)) == (4, 3)


def test_a_list_held_at_one_string_cuts_the_other_to_fit():
    one_beta = np.array([HF_B])
    ((alpha, beta),) = convert_open([(ALPHA, one_beta)], subspace_dim=3)
    assert beta.tolist() == [HF_B]
    assert len(alpha) == 3 and alpha[0] == HF_A


def test_hf_is_added_when_the_batch_lacks_it():
    ((alpha, beta),) = convert_open([(ALPHA[:2], BETA[:2])], subspace_dim=100)
    assert alpha.tolist() == [HF_A, *ALPHA[:2]]
    assert beta.tolist() == [HF_B, *BETA[:2]]


@pytest.mark.parametrize("seed", range(10))
def test_open_shell_keeps_each_spins_carryover_first(seed):
    keep = (np.array([0b11100, 0b01011]), np.array([0b11000]))
    batch_a = np.array([0b11100, 0b01011, *ALPHA[1:5]])
    batch_b = np.array([0b11000, *BETA[:4]])
    ((alpha, beta),) = convert_open([(batch_a, batch_b)], subspace_dim=12, seed=seed, keep=keep)
    assert alpha[:3].tolist() == [HF_A, 0b11100, 0b01011]
    assert beta[:2].tolist() == [HF_B, 0b11000]
    assert len(alpha) * len(beta) <= 12


def test_a_flat_keep_applies_to_both_spins():
    batch_a, batch_b = np.array([0b11100, *ALPHA[:5]]), np.array([0b11000, *BETA[:4]])
    ((alpha, beta),) = convert_open([(batch_a, batch_b)], subspace_dim=9, keep=[0b11100, 0b11000])
    assert alpha[:2].tolist() == [HF_A, 0b11100]
    assert beta[:2].tolist() == [HF_B, 0b11000]


def test_open_shell_draw_follows_the_probabilities():
    probabilities = {int(ci): 0.0 for ci in (*ALPHA, *BETA)}
    probabilities.update({0b11100: 1.0, 0b11000: 1.0})
    # 8 x 7 strings with Hartree-Fock; subspace_dim 6 cuts them to 2 x 2.
    ((alpha, beta),) = convert_open([(ALPHA, BETA)], subspace_dim=6, probabilities=probabilities)
    assert alpha.tolist() == [HF_A, 0b11100] and beta.tolist() == [HF_B, 0b11000]


def test_same_lists_with_different_electron_numbers_are_open_shell():
    strings = np.array([0b0101, 0b0110])
    ((alpha, beta),) = list(
        to_sbd_subspace([(strings, strings)], subspace_dim=100, num_elec_a=2, num_elec_b=1)
    )
    assert alpha[0] == 0b11 and beta[0] == 0b1


def test_subsampling_without_rng_is_rejected():
    with pytest.raises(ValueError, match="rng is required"):
        list(to_sbd_subspace([(ALPHA, BETA)], subspace_dim=4, num_elec_a=3, num_elec_b=2))
    with pytest.raises(ValueError, match="subspace_dim must be at least 1"):
        list(to_sbd_subspace([(ALPHA, BETA)], subspace_dim=0, num_elec_a=3, num_elec_b=2))
