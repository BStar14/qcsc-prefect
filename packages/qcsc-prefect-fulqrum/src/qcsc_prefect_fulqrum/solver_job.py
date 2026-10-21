"""Fulqrum subspace solver block backed by qcsc-prefect block execution."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import numpy as np
from prefect import get_run_logger, task
from prefect.blocks.core import Block
from pydantic import Field
from qcsc_prefect_executor.from_blocks import run_job_from_blocks
from qiskit_addon_sqd.fermion import SCIResult, SCIState

from .io_utils import (
    FulqrumSolverError,
    make_job_work_dir,
    read_solver_output,
    write_solver_input,
)
from .solver import FulqrumSolution, FulqrumSolverOptions


def _logger():
    try:
        return get_run_logger()
    except Exception:
        return logging.getLogger(__name__)


class FulqrumSolverJob(Block):
    """Prefect block facade for fulqrum execution through qcsc-prefect blocks.

    ``run()`` has the same signature and return type as ``DiceSHCISolverJob.run``,
    so SQD flows can switch solvers by changing only the block they load.
    """

    _block_type_name = "Fulqrum Solver Job"
    _block_type_slug = "fulqrum_solver_job"

    root_dir: str = Field(
        title="Root Directory",
        description="Root directory where per-job work directories are created.",
    )
    command_block_name: str = Field(
        default="cmd-fulqrum-solver",
        title="Command Block Name",
        description="Prefect CommandBlock document name.",
    )
    execution_profile_block_name: str = Field(
        default="exec-fulqrum-slurm",
        title="Execution Profile Block Name",
        description="Prefect ExecutionProfileBlock document name.",
    )
    hpc_profile_block_name: str = Field(
        default="hpc-slurm-fulqrum",
        title="HPC Profile Block Name",
        description="Prefect HPCProfileBlock document name.",
    )
    script_filename: str = Field(
        default="fulqrum_solver.slurm",
        title="Script Filename",
        description="Scheduler script filename. The suffix is normalized per target.",
    )
    metrics_artifact_key: str = Field(default="fulqrum-metrics", title="Metrics Artifact Key")
    timeout_seconds: float = Field(default=7200.0, title="Timeout Seconds")

    eigensolver: Literal["scipy", "primme"] = Field(
        default="scipy",
        title="Eigensolver",
        description="'scipy' (ARPACK) or 'primme' (needs primme on the compute node).",
    )
    hamiltonian_format: Literal["csr", "matrix_free"] = Field(
        default="csr",
        title="Hamiltonian Format",
        description="'csr' builds a sparse matrix (faster); 'matrix_free' saves memory.",
    )
    tol: float = Field(
        default=1e-8,
        ge=0,
        title="Eigensolver Tolerance",
        description="Eigensolver convergence tolerance. 0 means machine precision.",
    )
    max_iter: int | None = Field(
        default=None,
        gt=0,
        title="Maximum Iterations",
        description="Maximum eigensolver iterations. Empty uses the solver default.",
    )
    dense_max_dim: int = Field(
        default=64,
        ge=0,
        title="Dense Max Dimension",
        description="Subspaces up to this dimension are diagonalized densely.",
    )
    return_sci_state: bool = Field(
        default=True,
        title="SCI State",
        description="Return the ground-state amplitudes as an SCIState.",
    )

    def solver_options(self) -> FulqrumSolverOptions:
        """Return the numerical controls passed to the compute-node solver."""

        return FulqrumSolverOptions(
            eigensolver=self.eigensolver,
            hamiltonian_format=self.hamiltonian_format,
            tol=self.tol,
            max_iter=self.max_iter,
            dense_max_dim=self.dense_max_dim,
            return_amplitudes=self.return_sci_state,
        )

    async def run(
        self,
        ci_strings: tuple[np.ndarray, np.ndarray],
        one_body_tensor: np.ndarray,
        two_body_tensor: np.ndarray,
        norb: int,
        nelec: tuple[int, int],
        spin_sq: float | None = None,
    ) -> SCIResult:
        """Run fulqrum and return the subspace ground state.

        ``spin_sq`` is accepted for compatibility with the DICE solver block and
        is ignored: fulqrum does not apply a spin penalty.
        """

        return await _run_fulqrum_inner(
            solver=self,
            ci_strings=ci_strings,
            one_body_tensor=one_body_tensor,
            two_body_tensor=two_body_tensor,
            norb=norb,
            nelec=nelec,
            spin_sq=spin_sq,
        )


def solution_to_sci_result(
    solution: FulqrumSolution, *, norb: int, nelec: tuple[int, int]
) -> SCIResult:
    """Convert a :class:`FulqrumSolution` into a qiskit-addon-sqd ``SCIResult``.

    The energy is electronic only, as with the DICE solver. Without amplitudes the
    ``sci_state`` is ``None``.
    """

    sci_state = None
    if solution.amplitudes is not None:
        sci_state = SCIState(
            amplitudes=solution.amplitudes,
            ci_strs_a=solution.ci_strs_a,
            ci_strs_b=solution.ci_strs_b,
            norb=norb,
            nelec=tuple(nelec),
        )
    return SCIResult(
        energy=float(solution.energy),
        sci_state=sci_state,
        orbital_occupancies=(
            np.asarray(solution.orbital_occupancies[0]),
            np.asarray(solution.orbital_occupancies[1]),
        ),
    )


@task(name="solve_eigenstate")
async def _run_fulqrum_inner(
    *,
    solver: FulqrumSolverJob,
    ci_strings: tuple[np.ndarray, np.ndarray],
    one_body_tensor: np.ndarray,
    two_body_tensor: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    spin_sq: float | None = None,
) -> SCIResult:
    logger = _logger()
    if spin_sq is not None:
        logger.warning("FulqrumSolverJob ignores spin_sq=%s (no spin penalty in fulqrum).", spin_sq)

    base_work_dir = Path(solver.root_dir).expanduser().resolve()
    job_work_dir = make_job_work_dir(base_work_dir)

    write_solver_input(
        work_dir=job_work_dir,
        ci_strings=ci_strings,
        one_body_tensor=one_body_tensor,
        two_body_tensor=two_body_tensor,
        norb=norb,
        nelec=nelec,
        options=solver.solver_options(),
    )

    result = await run_job_from_blocks(
        command_block_name=solver.command_block_name,
        execution_profile_block_name=solver.execution_profile_block_name,
        hpc_profile_block_name=solver.hpc_profile_block_name,
        work_dir=job_work_dir,
        script_filename=solver.script_filename,
        user_args=[str(job_work_dir)],
        watch_poll_interval=5.0,
        timeout_seconds=solver.timeout_seconds,
        metrics_artifact_key=solver.metrics_artifact_key,
    )

    try:
        solution = read_solver_output(job_work_dir, norb)
    except FulqrumSolverError as exc:
        raise RuntimeError(
            f"{exc} (exit_status={result.exit_status}, work_dir={str(job_work_dir)!r})"
        ) from exc
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"fulqrum output file not found: {exc.filename!r}. "
            f"exit_status={result.exit_status}, "
            f"work_dir={str(job_work_dir)!r}. "
            "Check output.err in the work directory."
        ) from exc

    if result.exit_status != 0:
        logger.warning(
            "fulqrum job returned nonzero exit status %s but wrote complete outputs.",
            result.exit_status,
        )
    meta = solution.metadata
    logger.info(
        "fulqrum energy %.12f, dim=%s, solver=%s, timings=%s, work_dir=%s",
        solution.energy,
        meta.get("subspace_dim"),
        meta.get("solver_path"),
        meta.get("timings"),
        job_work_dir,
    )
    return solution_to_sci_result(solution, norb=norb, nelec=nelec)
