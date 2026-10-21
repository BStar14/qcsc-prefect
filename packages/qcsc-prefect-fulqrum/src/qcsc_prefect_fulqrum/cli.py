"""``qcsc-fulqrum-solve``: run one fulqrum subspace solve in a job directory.

This is the executable a scheduler job runs on the compute node. It needs only
numpy, scipy, fulqrum, and this package (no Prefect), and it reads and writes
the files described in :mod:`qcsc_prefect_fulqrum.io_utils`.

Exit codes: ``0`` success, ``1`` the solve failed (details in ``error.json``),
``2`` usage error.
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

    # A stale result must never be read as the outcome of this run.
    for name in (io_utils.OUTPUT_RESULT, io_utils.OUTPUT_ARRAYS, io_utils.OUTPUT_ERROR):
        (work_dir / name).unlink(missing_ok=True)

    _default_omp_threads()
    try:
        problem = io_utils.read_solver_input(work_dir)
        solution = solve_fulqrum(**problem)
        io_utils.write_solver_output(work_dir, solution)
    except Exception as exc:
        logger.exception("fulqrum solve failed in %s", work_dir)
        io_utils.write_solver_error(work_dir, exc)
        return 1

    meta = solution.metadata
    print(
        f"qcsc-fulqrum-solve: energy={solution.energy:.12f} "
        f"dim={meta.get('subspace_dim')} solver={meta.get('solver_path')} "
        f"omp_num_threads={meta.get('omp_num_threads')}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
