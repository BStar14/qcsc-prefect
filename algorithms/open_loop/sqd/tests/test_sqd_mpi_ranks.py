"""``sqd_common.mpi_ranks``: the launcher's rank count must match ``num_nodes x mpiprocs``."""

from __future__ import annotations

import pytest
from sqd_common.mpi_ranks import check_rank_count, max_proc_per_node, mpi_rank_count


@pytest.mark.parametrize(
    ("options", "count"),
    [
        (None, None),
        ([], None),
        (["--mpi=pmix"], None),
        (["-n", "8"], 8),
        (["-np", "4"], 4),
        (["-np 4"], 4),
        (["--np=6"], 6),
        (["--ntasks", "2"], 2),
        (["-genv", "I_MPI_DEBUG", "5", "-np", "12"], 12),
    ],
)
def test_mpi_rank_count_reads_the_count_flags(options, count):
    assert mpi_rank_count(options) == count


@pytest.mark.parametrize(
    ("options", "message"),
    [(["-np"], "ends with -np"), (["-n", "all"], "-n all, which is not a number")],
)
def test_mpi_rank_count_rejects_a_missing_or_bad_count(options, message):
    with pytest.raises(RuntimeError, match=message):
        mpi_rank_count(options)


@pytest.mark.parametrize(
    ("options", "per_node"),
    [
        (None, None),
        (["max-proc-per-node=4"], 4),
        (["proc-core=unbind,max-proc-per-node=12"], 12),
        (["rank-map-bynode"], None),
    ],
)
def test_max_proc_per_node_reads_the_pjm_option(options, per_node):
    assert max_proc_per_node(options) == per_node


def _kwargs(**overrides):
    kwargs = {
        "hpc_target": "miyabi",
        "launcher": "mpiexec.hydra",
        "num_nodes": 2,
        "mpiprocs": 4,
        "mpi_options": ["-np", "8"],
        "mpi_options_for_pjm": None,
    }
    return {**kwargs, **overrides}


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"mpi_options": None},
        {"mpi_options": ["--mpi=pmix"], "hpc_target": "slurm", "launcher": "srun"},
        {"launcher": "single", "mpi_options": ["-np", "1"]},
        {
            "hpc_target": "fugaku",
            "launcher": "mpiexec",
            "mpi_options": ["-n", "8"],
            "mpi_options_for_pjm": ["max-proc-per-node=4"],
        },
    ],
)
def test_matching_counts_pass(overrides):
    check_rank_count(_kwargs(**overrides))


def test_a_count_that_differs_from_the_blocks_fails():
    with pytest.raises(RuntimeError, match=r"starts 4 MPI ranks.*2 x 4 = 8.*to 8"):
        check_rank_count(_kwargs(mpi_options=["-np", "4"]))


def test_a_fugaku_placement_that_differs_from_mpiprocs_fails():
    kwargs = _kwargs(
        hpc_target="fugaku",
        launcher="mpiexec",
        mpi_options=["-n", "8"],
        mpi_options_for_pjm=["max-proc-per-node=1"],
    )
    with pytest.raises(RuntimeError, match=r"max-proc-per-node=1 ranks.*mpiprocs = 4"):
        check_rank_count(kwargs)
