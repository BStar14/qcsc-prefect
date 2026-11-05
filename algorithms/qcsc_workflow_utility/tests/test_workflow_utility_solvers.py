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


def convert(batches, subspace_dim, seed=0, probabilities=None):
    return list(
        to_sbd_closed_shell(
            batches,
            subspace_dim=subspace_dim,
            num_elec_a=2,
            probabilities=probabilities,
            rng=np.random.default_rng(seed),
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
