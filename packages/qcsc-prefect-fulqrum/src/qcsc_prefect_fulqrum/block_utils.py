"""Helpers for registering and creating fulqrum-related Prefect blocks."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from qcsc_prefect_blocks.common.blocks import CommandBlock, ExecutionProfileBlock, HPCProfileBlock
from qcsc_prefect_core.models.gpu_request import validate_gpu_request

from .solver import FulqrumSolverOptions
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
    mpiprocs: int | None = None,
    ompthreads: int | None = None,
    walltime: str = "12:00:00",
    resource_class: Literal["cpu", "gpu"] = "cpu",
    gpus_per_node: int | None = None,
    modules: list[str] | None = None,
    mpi_options: list[str] | None = None,
    pre_commands: list[str] | None = None,
    environments: dict[str, str] | None = None,
    script_filename: str | None = None,
    metrics_artifact_key: str | None = None,
    timeout_seconds: float = 46800.0,
    eigensolver: Literal["scipy", "primme"] = "scipy",
    hamiltonian_format: Literal["csr", "matrix_free"] = "csr",
    tol: float = 1e-8,
    max_iter: int | None = None,
    dense_max_dim: int = 64,
    return_sci_state: bool = True,
    backend: Literal["cpu", "gpu", "gpu-mpi"] = "cpu",
    max_basis_size: int | None = None,
    block_size: int | None = None,
    min_restart_size: int | None = None,
    primme_method: str | None = None,
    convergence: Literal["abs", "rel", "norm", "energy"] = "abs",
    energy_tol: float = 1e-6,
    process_grid: tuple[int, int] | None = None,
    comm_backend: Literal["auto", "nccl", "host_mpi", "cuda_mpi"] = "auto",
    slurm_memory: str | None = None,
    gfscache: str | None = None,
    spack_modules: list[str] | None = None,
    mpi_options_for_pjm: list[str] | None = None,
    pjm_resources: list[str] | None = None,
) -> dict[str, str]:
    """Create the standard block set required to run fulqrum on qcsc-prefect.

    ``fulqrum_executable`` is the ``qcsc-fulqrum-solve`` entry point of the
    compute-node environment that ``native/build_fulqrum.sh`` builds, for example
    ``/path/to/qcsc-prefect/packages/qcsc-prefect-fulqrum/native/fulqrum/bin/qcsc-fulqrum-solve``.

    ``backend`` selects the solver path (see :mod:`qcsc_prefect_fulqrum.solver`):

    - ``"cpu"`` and ``"gpu"`` run one process, so the defaults are
      ``launcher="single"``, ``num_nodes=1`` and ``mpiprocs=1``.
    - ``"gpu-mpi"`` runs one MPI rank per GPU. It needs an MPI ``launcher``
      (``"srun"``, ``"mpirun"``, ...). With ``gpus_per_node`` set, ``mpiprocs``
      is ``gpus_per_node``, one rank per GPU, and another value raises; without
      it, ``mpiprocs`` defaults to 1.

    ``gpus_per_node`` is the scheduler GPU request on the execution profile. It
    needs ``resource_class="gpu"``, which selects the GPU queue; Fugaku and local
    execution reject it, and Miyabi-G accepts only 1. These are checked here, before
    any block is saved. The solver block's
    ``carryover_fraction`` is left at 0; the SQD flow sets it for each run.

    The eigensolver tuning (``max_basis_size``, ``block_size``,
    ``min_restart_size``, ``primme_method``, and for ``gpu-mpi`` ``convergence``,
    ``energy_tol``, ``process_grid`` and ``comm_backend``) is checked against the
    eigensolver and backend before any block is saved, so an option the solver would
    ignore fails here.

    When ``ompthreads`` is set and ``environments`` has no ``OMP_NUM_THREADS``,
    it is exported with the same value, because not every scheduler template
    sets it.

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
    if backend == "gpu-mpi" and launcher == "single":
        raise ValueError(
            "backend='gpu-mpi' runs one MPI rank per GPU; set launcher to an MPI launcher "
            "such as 'srun' or 'mpirun'."
        )
    validate_gpu_request(
        hpc_target=hpc_target, resource_class=resource_class, gpus_per_node=gpus_per_node
    )
    if backend == "gpu-mpi" and gpus_per_node is not None:
        # One rank per GPU: a different count would leave GPUs idle or share them.
        if mpiprocs is not None and mpiprocs != gpus_per_node:
            raise ValueError(
                f"backend='gpu-mpi' runs one MPI rank per GPU, but mpiprocs={mpiprocs} and "
                f"gpus_per_node={gpus_per_node}; set them to the same value."
            )
        mpiprocs = gpus_per_node
    if mpiprocs is None:
        mpiprocs = 1
    if backend == "gpu-mpi" and process_grid is not None:
        ranks = num_nodes * mpiprocs
        grid = tuple(int(p) for p in process_grid)
        if len(grid) == 2 and grid[0] * grid[1] != ranks:
            # The job would fail on every rank only after waiting in the queue.
            raise ValueError(
                f"process_grid {list(grid)} needs {grid[0] * grid[1]} MPI ranks, but "
                f"num_nodes={num_nodes} x mpiprocs={mpiprocs} gives {ranks}."
            )

    # Raises for tuning the eigensolver or backend would ignore, before any block is saved.
    FulqrumSolverOptions(
        eigensolver=eigensolver,
        hamiltonian_format=hamiltonian_format,
        tol=tol,
        max_iter=max_iter,
        dense_max_dim=dense_max_dim,
        backend=backend,
        max_basis_size=max_basis_size,
        block_size=block_size,
        min_restart_size=min_restart_size,
        primme_method=primme_method,
        convergence=convergence,
        energy_tol=energy_tol,
        process_grid=process_grid,
        comm_backend=comm_backend,
    )

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
        gpus_per_node=gpus_per_node,
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
        backend=backend,
        max_basis_size=max_basis_size,
        block_size=block_size,
        min_restart_size=min_restart_size,
        primme_method=primme_method,
        convergence=convergence,
        energy_tol=energy_tol,
        process_grid=process_grid,
        comm_backend=comm_backend,
    ).save(solver_block_name, overwrite=True)

    return {
        "command_block_name": command_block_name,
        "execution_profile_block_name": resolved_exec_block_name,
        "hpc_profile_block_name": resolved_hpc_block_name,
        "solver_block_name": solver_block_name,
    }
