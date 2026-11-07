"""``qcsc-fulqrum-solve``: run one fulqrum subspace solve in a job directory.

This is the executable a scheduler job runs on the compute node. It needs only
numpy, scipy, fulqrum, and this package (no Prefect), and it reads and writes
the files described in :mod:`qcsc_prefect_fulqrum.io_utils`.

Under a multi-process launcher (``srun``, ``mpirun``) every process runs this
program. Every process reads the input and calls :func:`solve_fulqrum`, because a
distributed solve is collective: all of them have to enter it. Only the control
process (index 0) writes the results or to stdout, so the processes cannot race
on the result files.

When the launcher starts more than one process and mpi4py is installed, the
processes use ``MPI.COMM_WORLD``, whatever the solver backend: the process index
is its rank, and every process learns whether all of them read the input before
any of them solves. Otherwise the index comes from the launcher's environment
(:func:`process_index`), so no MPI library is needed and the single-process path
is unchanged. The ``process_index`` / ``is_control_process`` names follow
``qiskit_addon_sqd.processes`` (qiskit-addon-sqd 0.13 and later), but the
detection does not: see :func:`process_index`.

Exit codes: ``0`` success, ``1`` the solve failed, ``2`` usage error. A failing
process writes its own error file, ``error.json`` for index 0 and
``error.rank<k>.json`` otherwise (:func:`io_utils.read_solver_error` reads them
all), and returns ``1`` whether or not it is the control process, because the
scheduler reports the job as failed only if the process it samples exited
non-zero. Any error file fails the solve, so a failure of one process fails it
even when the control process wrote a result. If a process cannot read the
input, every process returns ``1`` together. If one fails inside the solve, it
calls ``MPI.COMM_WORLD.Abort(1)``: its peers may be waiting for it inside a
collective and would otherwise hang until the job's walltime.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
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
# Process counts set by the same launchers. More than one means a multi-process launch.
_PROCESS_COUNT_ENV_VARS = (
    "OMPI_COMM_WORLD_SIZE",  # Open MPI, IBM Spectrum MPI
    "PMI_SIZE",  # Intel MPI, MPICH
    "PMIX_SIZE",  # PMIx-based launchers
    "SLURM_NTASKS",  # srun
)
# Without a communicator, the control process clears only rank error files that are
# this much older than its start, so it never deletes one a peer writes in this run.
_STALE_ERROR_AGE_SECONDS = 60.0


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


def _launched_processes() -> int:
    """The process count of the launch, from the launcher's environment; 1 if none is set."""

    for name in _PROCESS_COUNT_ENV_VARS:
        try:
            return max(int(os.environ[name]), 1)
        except (KeyError, ValueError):
            continue
    return 1


def _mpi_world():
    """``MPI.COMM_WORLD`` for a multi-process launch with mpi4py, else ``None``.

    The decision depends only on the environment, which every process of a launch
    shares, so the processes never disagree on whether they use the communicator.
    """

    if _launched_processes() < 2:
        return None
    try:
        from mpi4py import MPI
    except ImportError as exc:
        logger.debug("No mpi4py (%s); using the launcher's process index.", exc)
        return None
    comm = MPI.COMM_WORLD
    return comm if comm.Get_size() > 1 else None


def _abort(comm) -> None:
    # Peers blocked in a collective never return, so end the whole MPI job.
    logger.error("Aborting MPI.COMM_WORLD so that the other processes do not hang.")
    for stream in (sys.stdout, sys.stderr):
        stream.flush()
    comm.Abort(1)


def _remove_stale_error(path: Path, started: float | None) -> None:
    """Delete a stale error file; with ``started``, only one older than this run."""

    try:
        if started is None or path.stat().st_mtime < started - _STALE_ERROR_AGE_SECONDS:
            path.unlink()
    except FileNotFoundError:
        pass


def _write_error(work_dir: Path, exc: BaseException, index: int) -> None:
    try:
        io_utils.write_solver_error(work_dir, exc, rank=index)
    except OSError:
        logger.exception("could not write the error file in %s", work_dir)


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

    started = time.time()
    # Resolved once: every branch below must agree on who is in control.
    comm = _mpi_world()
    index = comm.Get_rank() if comm is not None else process_index()
    control = index == 0

    # A stale result must never be read as the outcome of this run. Each process
    # deletes its own error file, and the control process the outputs, which
    # nothing reads yet, and the other stale error files. With a communicator the
    # barrier keeps it from deleting a file that a peer writes in this run;
    # without one, it deletes only files older than this run.
    (work_dir / io_utils.error_file_name(index)).unlink(missing_ok=True)
    if control:
        for name in (io_utils.OUTPUT_RESULT, io_utils.OUTPUT_ARRAYS):
            (work_dir / name).unlink(missing_ok=True)
        for path in io_utils.solver_error_files(work_dir):
            _remove_stale_error(path, None if comm is not None else started)
    if comm is not None:
        comm.Barrier()

    _default_omp_threads()
    try:
        problem = io_utils.read_solver_input(work_dir)
    except Exception as exc:
        problem, failure = None, exc
    else:
        failure = None
    if comm is not None and not all(comm.allgather(failure is None)):
        # Every process stops here together, so none waits for a peer in the solve.
        if failure is None:
            failure = RuntimeError("another process could not read the solver input")
    if failure is not None:
        logger.error("fulqrum input not read in %s (process %d): %s", work_dir, index, failure)
        _write_error(work_dir, failure, index)
        return 1

    try:
        solution = solve_fulqrum(**problem)
        if control:
            io_utils.write_solver_output(work_dir, solution)
    except Exception as exc:
        logger.exception("fulqrum solve failed in %s (process %d)", work_dir, index)
        _write_error(work_dir, exc, index)
        if comm is not None:
            _abort(comm)
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
