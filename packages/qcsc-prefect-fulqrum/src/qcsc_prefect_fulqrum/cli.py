"""``qcsc-fulqrum-solve``: run one fulqrum subspace solve in a job directory.

This is the executable a scheduler job runs on the compute node. It needs only
numpy, scipy, fulqrum, and this package (no Prefect), and it reads and writes
the files described in :mod:`qcsc_prefect_fulqrum.io_utils`.

Under a multi-process launcher (``srun``, ``mpirun``) every process runs this
program. Every process reads the input and calls :func:`solve_fulqrum`, because a
distributed solve is collective: all of them have to enter it. Only the control
process (index 0) writes the results or to stdout, so the processes cannot race
on the result files.

The process index depends on the solver backend in ``solver.json``. Under
``gpu-mpi`` it is ``MPI.COMM_WORLD`` rank from mpi4py, the communicator the
solve runs on. Otherwise it comes from the launcher's environment
(:func:`process_index`), so no MPI library is needed and the single-process path
is unchanged. The ``process_index`` / ``is_control_process`` names follow
``qiskit_addon_sqd.processes`` (qiskit-addon-sqd 0.13 and later), but the
detection does not: see :func:`process_index`.

Exit codes: ``0`` success, ``1`` the solve failed, ``2`` usage error. A failing
process writes its own error file, ``error.json`` for index 0 and
``error.rank<k>.json`` otherwise (:func:`io_utils.read_solver_error` reads them
all), and returns ``1`` whether or not it is the control process, because the
scheduler reports the job as failed only if the process it samples exited
non-zero. Under ``gpu-mpi`` a failing process then calls
``MPI.COMM_WORLD.Abort(1)``: its peers may be waiting for it inside a collective
and would otherwise hang until the job's walltime.
"""

from __future__ import annotations

import argparse
import json
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


def _requested_backend(work_dir: Path) -> str | None:
    """The solver backend in ``solver.json``, or ``None`` if it cannot be read.

    An unreadable spec is reported by :func:`io_utils.read_solver_input` inside
    the solve, where every process records its failure.
    """

    try:
        spec = json.loads((work_dir / io_utils.INPUT_SPEC).read_text(encoding="utf-8"))
        return spec["options"].get("backend")
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


def _mpi_world():
    """``MPI.COMM_WORLD``, or ``None`` if mpi4py is missing.

    Without mpi4py the ``gpu-mpi`` solve fails with an ImportError on every
    process, so the index from the environment is good enough for the error files.
    """

    try:
        from mpi4py import MPI
    except ImportError as exc:
        logger.warning("Backend gpu-mpi without mpi4py (%s); using the launcher's rank.", exc)
        return None
    return MPI.COMM_WORLD


def _abort(comm) -> None:
    # Peers blocked in a collective never return, so end the whole MPI job.
    logger.error("Aborting MPI.COMM_WORLD so that the other processes do not hang.")
    for stream in (sys.stdout, sys.stderr):
        stream.flush()
    comm.Abort(1)


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
    comm = _mpi_world() if _requested_backend(work_dir) == "gpu-mpi" else None
    index = comm.Get_rank() if comm is not None else process_index()
    control = index == 0

    # A stale result must never be read as the outcome of this run. Each
    # process deletes its own error file, and the control process the outputs,
    # which nothing reads yet. With a communicator the control process also
    # deletes every stale error file, and the barrier keeps it from deleting a
    # file that a peer writes in this run.
    (work_dir / io_utils.error_file_name(index)).unlink(missing_ok=True)
    if control:
        stale = [io_utils.OUTPUT_RESULT, io_utils.OUTPUT_ARRAYS]
        if comm is not None:
            stale += [path.name for path in io_utils.solver_error_files(work_dir)]
        for name in stale:
            (work_dir / name).unlink(missing_ok=True)
    if comm is not None:
        comm.Barrier()

    _default_omp_threads()
    try:
        problem = io_utils.read_solver_input(work_dir)
        solution = solve_fulqrum(**problem)
        if control:
            io_utils.write_solver_output(work_dir, solution)
    except Exception as exc:
        logger.exception("fulqrum solve failed in %s (process %d)", work_dir, index)
        try:
            io_utils.write_solver_error(work_dir, exc, rank=index)
        except OSError:
            logger.exception("could not write the error file in %s", work_dir)
        if comm is not None and comm.Get_size() > 1:
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
