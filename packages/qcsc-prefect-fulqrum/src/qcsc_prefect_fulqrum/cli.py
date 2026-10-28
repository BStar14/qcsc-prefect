"""``qcsc-fulqrum-solve``: run one fulqrum subspace solve in a job directory.

This is the executable a scheduler job runs on the compute node. It needs only
numpy, scipy, fulqrum, and this package (no Prefect), and it reads and writes
the files described in :mod:`qcsc_prefect_fulqrum.io_utils`.

Under a multi-process launcher (``srun``, ``mpirun``) every process runs this
program. Every process reads the input and calls :func:`solve_fulqrum`, because a
distributed solve is collective: all of them have to enter it. Only the control
process (:func:`is_control_process`) touches the job directory or writes to
stdout, so the processes cannot race on the result files. The process index
comes from the launcher's environment, so no MPI library is needed to read it
and the single-process path is unchanged.

The ``process_index`` / ``is_control_process`` names follow
``qiskit_addon_sqd.processes`` (qiskit-addon-sqd 0.13 and later), but the
detection does not: see :func:`process_index`.

Exit codes: ``0`` success, ``1`` the solve failed (details in ``error.json``),
``2`` usage error. A failing process returns ``1`` whether or not it is the
control process, because the scheduler reports the job as failed only if the
process it samples exited non-zero. A failure that only some processes hit is
not handled here: one process failing inside a collective leaves the others
waiting in it, which needs ``MPI_Abort`` or a communicator-wide check.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from . import io_utils
from .solver import solve_fulqrum

logger = logging.getLogger("qcsc_prefect_fulqrum")


# Global rank variables set by the common launchers, in priority order.
#
# Only job-wide ranks belong here. A node-local rank such as MPI_LOCALRANKID
# (Intel MPI, MPICH Hydra) restarts at 0 on every node, so a multi-node run
# would elect one control process per node and race on the result files.
#
# SLURM_PROCID is last: srun sets it, but so does sbatch for the batch step
# itself, where it is 0 and agrees with the single-process default.
_PROCESS_INDEX_ENV_VARS = (
    "OMPI_COMM_WORLD_RANK",  # Open MPI, IBM Spectrum MPI
    "PMI_RANK",  # Intel MPI, MPICH
    "PMIX_RANK",  # PMIx-based launchers
    "SLURM_PROCID",  # srun
)


def process_index() -> int:
    """Return the index of this process, 0 for the control process.

    ``qiskit_addon_sqd.processes`` uses the launcher's ``*_SIZE`` variables
    only to decide whether it runs under MPI, and then asks ``mpi4py``
    (``COMM_WORLD.Get_rank()``) for the index. Here the index is read from the
    launcher's rank variables directly, so that the compute node keeps its
    minimal dependency set of numpy, scipy and fulqrum. The first
    variable holding an integer wins; a malformed value is skipped. Launched as
    a single process, or under a launcher none of these variables identify, the
    index is 0.
    """

    for name in _PROCESS_INDEX_ENV_VARS:
        value = os.environ.get(name)
        if value is None:
            continue
        try:
            return int(value)
        except ValueError:
            logger.warning("Ignoring non-integer %s=%r.", name, value)
    return 0


def is_control_process() -> bool:
    """Return whether this process owns the job directory and stdout.

    True for a single process and for index 0 of a multi-process launch. The
    other processes take part in the solve but leave the result files and the
    summary line to this one.
    """

    return process_index() == 0


def _default_omp_threads() -> None:
    # Slurm does not export OMP_NUM_THREADS for --cpus-per-task. Set it before
    # fulqrum (and its OpenMP runtime) is imported inside solve_fulqrum.
    if "OMP_NUM_THREADS" not in os.environ and os.environ.get("SLURM_CPUS_PER_TASK"):
        os.environ["OMP_NUM_THREADS"] = os.environ["SLURM_CPUS_PER_TASK"]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qcsc-fulqrum-solve",
        description="Solve the fulqrum subspace problem stored in a job directory.",
    )
    parser.add_argument(
        "work_dir",
        nargs="?",
        default=".",
        type=Path,
        help="Job directory with input.npz and solver.json (default: current directory).",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    work_dir = args.work_dir.resolve()
    if not work_dir.is_dir():
        print(f"qcsc-fulqrum-solve: not a directory: {work_dir}", file=sys.stderr)
        return 2

    # Resolved once: every branch below must agree on who is in control.
    control = is_control_process()

    # A stale result must never be read as the outcome of this run. Only the
    # control process deletes, and only output files, which nothing reads.
    if control:
        for name in (io_utils.OUTPUT_RESULT, io_utils.OUTPUT_ARRAYS, io_utils.OUTPUT_ERROR):
            (work_dir / name).unlink(missing_ok=True)

    _default_omp_threads()
    try:
        problem = io_utils.read_solver_input(work_dir)
        solution = solve_fulqrum(**problem)
        if control:
            io_utils.write_solver_output(work_dir, solution)
    except Exception as exc:
        logger.exception("fulqrum solve failed in %s", work_dir)
        if control:
            io_utils.write_solver_error(work_dir, exc)
        # Every process fails the job: the scheduler may sample any one of them.
        return 1

    if not control:
        return 0

    meta = solution.metadata
    print(
        f"qcsc-fulqrum-solve: energy={solution.energy:.12f} "
        f"dim={meta.get('subspace_dim')} solver={meta.get('solver_path')} "
        f"omp_num_threads={meta.get('omp_num_threads')}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
