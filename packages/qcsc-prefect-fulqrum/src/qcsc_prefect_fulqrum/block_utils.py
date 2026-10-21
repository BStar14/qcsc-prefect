"""Helpers for registering and creating fulqrum-related Prefect blocks."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from qcsc_prefect_blocks.common.blocks import CommandBlock, ExecutionProfileBlock, HPCProfileBlock

from .solver_job import FulqrumSolverJob


def register_fulqrum_block_types() -> None:
    """Register common block schemas used by fulqrum integrations."""

    for block_cls in (
        CommandBlock,
        ExecutionProfileBlock,
        HPCProfileBlock,
        FulqrumSolverJob,
    ):
        register = getattr(block_cls, "register_type_and_schema", None)
        if callable(register):
            register()


def create_fulqrum_blocks(
    *,
    hpc_target: Literal["local", "slurm", "miyabi", "fugaku"],
    root_dir: str,
    fulqrum_executable: str,
    project: str = "",
    queue: str = "",
    command_block_name: str = "cmd-fulqrum-solver",
    execution_profile_block_name: str | None = None,
    hpc_profile_block_name: str | None = None,
    solver_block_name: str = "fulqrum-solver",
    command_name: str = "fulqrum",
    executable_key: str = "fulqrum_solver",
    profile_name: str = "fulqrum-omp",
    launcher: Literal["single", "srun", "mpirun", "mpiexec", "mpiexec.hydra"] = "single",
    num_nodes: int = 1,
    mpiprocs: int = 1,
    ompthreads: int | None = None,
    walltime: str = "01:00:00",
    resource_class: Literal["cpu", "gpu"] = "cpu",
    modules: list[str] | None = None,
    mpi_options: list[str] | None = None,
    pre_commands: list[str] | None = None,
    environments: dict[str, str] | None = None,
    script_filename: str | None = None,
    metrics_artifact_key: str | None = None,
    timeout_seconds: float = 7200.0,
    eigensolver: Literal["scipy", "primme"] = "scipy",
    hamiltonian_format: Literal["csr", "matrix_free"] = "csr",
    tol: float = 1e-8,
    max_iter: int | None = None,
    dense_max_dim: int = 64,
    return_sci_state: bool = True,
    slurm_memory: str | None = None,
    gfscache: str | None = None,
    spack_modules: list[str] | None = None,
    mpi_options_for_pjm: list[str] | None = None,
    pjm_resources: list[str] | None = None,
) -> dict[str, str]:
    """Create the standard block set required to run fulqrum on qcsc-prefect.

    ``fulqrum_executable`` is the ``qcsc-fulqrum-solve`` entry point of the
    compute-node environment, for example
    ``/opt/venvs/fulqrum/bin/qcsc-fulqrum-solve``.

    Fulqrum currently runs as one OpenMP process, so the defaults are
    ``launcher="single"``, ``num_nodes=1`` and ``mpiprocs=1``. When ``ompthreads``
    is set and ``environments`` has no ``OMP_NUM_THREADS``, it is exported with
    the same value, because not every scheduler template sets it.

    This helper keeps a typed, explicit API on purpose. Algorithm-specific
    wrappers can translate TOML or CLI config dictionaries into these arguments
    without leaking config-shape coupling into the shared package.
    """

    if hpc_target == "local" and (modules or pre_commands):
        unsupported = []
        if modules:
            unsupported.append("modules")
        if pre_commands:
            unsupported.append("pre_commands")
        raise ValueError("Local execution does not support " + " or ".join(unsupported) + ".")

    register_fulqrum_block_types()

    resolved_exec_block_name = execution_profile_block_name or f"exec-fulqrum-{hpc_target}"
    resolved_hpc_block_name = hpc_profile_block_name or f"hpc-{hpc_target}-fulqrum"
    default_script_filename = {
        "local": "fulqrum_solver",
        "slurm": "fulqrum_solver.slurm",
        "miyabi": "fulqrum_solver.pbs",
        "fugaku": "fulqrum_solver.pjm",
    }[hpc_target]
    resolved_script_filename = script_filename or default_script_filename
    resolved_metrics_key = metrics_artifact_key or f"{hpc_target}-fulqrum-metrics"
    resolved_environments = dict(environments or {})
    if ompthreads is not None and "OMP_NUM_THREADS" not in resolved_environments:
        resolved_environments["OMP_NUM_THREADS"] = str(ompthreads)

    CommandBlock(
        command_name=command_name,
        executable_key=executable_key,
        description="Fulqrum subspace solver (qcsc-fulqrum-solve <work_dir>)",
        default_args=[],
    ).save(command_block_name, overwrite=True)

    ExecutionProfileBlock(
        profile_name=profile_name,
        command_name=command_name,
        resource_class=resource_class,
        num_nodes=num_nodes,
        mpiprocs=mpiprocs,
        ompthreads=ompthreads,
        walltime=walltime,
        launcher=launcher,
        mpi_options=list(mpi_options or []),
        modules=list(modules or []),
        pre_commands=list(pre_commands or []),
        environments=resolved_environments,
    ).save(resolved_exec_block_name, overwrite=True)

    HPCProfileBlock(
        hpc_target=hpc_target,
        queue_cpu=queue,
        queue_gpu=queue,
        project_cpu=project,
        project_gpu=project,
        executable_map={executable_key: str(fulqrum_executable)},
        slurm_memory=slurm_memory,
        gfscache=gfscache,
        spack_modules=list(spack_modules or []),
        mpi_options_for_pjm=list(mpi_options_for_pjm or []),
        pjm_resources=list(pjm_resources or []),
    ).save(resolved_hpc_block_name, overwrite=True)

    FulqrumSolverJob(
        root_dir=str(Path(root_dir).expanduser().resolve()),
        command_block_name=command_block_name,
        execution_profile_block_name=resolved_exec_block_name,
        hpc_profile_block_name=resolved_hpc_block_name,
        script_filename=resolved_script_filename,
        metrics_artifact_key=resolved_metrics_key,
        timeout_seconds=timeout_seconds,
        eigensolver=eigensolver,
        hamiltonian_format=hamiltonian_format,
        tol=tol,
        max_iter=max_iter,
        dense_max_dim=dense_max_dim,
        return_sci_state=return_sci_state,
    ).save(solver_block_name, overwrite=True)

    return {
        "command_block_name": command_block_name,
        "execution_profile_block_name": resolved_exec_block_name,
        "hpc_profile_block_name": resolved_hpc_block_name,
        "solver_block_name": solver_block_name,
    }
