"""Helpers for registering and creating SBD-related Prefect blocks."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from qcsc_prefect_blocks.common.blocks import CommandBlock, ExecutionProfileBlock, HPCProfileBlock
from qcsc_prefect_core.models.gpu_request import validate_gpu_request

from .solver_job import SBDSolverJob


def register_sbd_block_types() -> None:
    """Register common block schemas used by SBD integrations."""

    for block_cls in (
        CommandBlock,
        ExecutionProfileBlock,
        HPCProfileBlock,
        SBDSolverJob,
    ):
        register = getattr(block_cls, "register_type_and_schema", None)
        if callable(register):
            register()


def create_sbd_blocks(
    *,
    hpc_target: Literal["local", "slurm", "miyabi", "fugaku"],
    root_dir: str,
    sbd_executable: str,
    project: str = "",
    queue: str = "",
    command_block_name: str = "cmd-sbd-solver",
    execution_profile_block_name: str | None = None,
    hpc_profile_block_name: str | None = None,
    solver_block_name: str = "sbd-solver",
    command_name: str = "sbd",
    executable_key: str = "sbd_solver",
    profile_name: str = "sbd-mpi",
    launcher: Literal["single", "srun", "mpirun", "mpiexec", "mpiexec.hydra"] | None = None,
    num_nodes: int = 1,
    mpiprocs: int = 4,
    ompthreads: int | None = None,
    walltime: str = "01:00:00",
    resource_class: Literal["cpu", "gpu"] = "cpu",
    gpus_per_node: int | None = None,
    modules: list[str] | None = None,
    mpi_options: list[str] | None = None,
    pre_commands: list[str] | None = None,
    environments: dict[str, str] | None = None,
    script_filename: str | None = None,
    metrics_artifact_key: str | None = None,
    timeout_seconds: float = 7200.0,
    task_comm_size: int = 1,
    adet_comm_size: int = 1,
    bdet_comm_size: int = 1,
    block: int = 10,
    iteration: int = 2,
    tolerance: float = 1e-4,
    carryover_ratio: float = 0.5,
    carryover_type: int = 1,
    solver_mode: Literal["cpu", "gpu"] = "cpu",
    slurm_memory: str | None = None,
    gfscache: str | None = None,
    spack_modules: list[str] | None = None,
    mpi_options_for_pjm: list[str] | None = None,
    pjm_resources: list[str] | None = None,
) -> dict[str, str]:
    """Create the standard block set required to run SBD on qcsc-prefect.

    ``sbd_executable`` is the SBD driver built from ``native/main.cc`` for the
    target, for example ``native/diag`` or ``native/diag-gpu``.

    The launcher defaults to ``"single"`` for ``local``, ``"srun"`` for ``slurm``,
    ``"mpiexec.hydra"`` for ``miyabi`` and ``"mpiexec"`` for ``fugaku``.

    These checks run before any block is saved:

    - Local execution rejects ``modules`` and ``pre_commands``.
    - ``gpus_per_node`` must fit the target and needs ``resource_class="gpu"``
      (see ``validate_gpu_request``).
    - ``solver_mode="gpu"`` needs ``resource_class="gpu"``, which selects the GPU
      queue.
    - SBD splits the MPI world into ``task_comm_size x adet_comm_size x
      bdet_comm_size`` groups, and the ranks left over form the Hamiltonian
      communicator, so the product must divide the rank count, ``num_nodes x
      mpiprocs`` (``1`` with the ``"single"`` launcher).

    When ``ompthreads`` is set and ``environments`` has no ``OMP_NUM_THREADS``,
    it is exported with the same value, because not every scheduler template
    sets it. On Fugaku the directory of ``sbd_executable`` is prepended to
    ``LD_LIBRARY_PATH`` unless ``environments`` sets it.

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
    validate_gpu_request(
        hpc_target=hpc_target, resource_class=resource_class, gpus_per_node=gpus_per_node
    )
    if solver_mode == "gpu" and resource_class != "gpu":
        raise ValueError(
            "solver_mode='gpu' runs the GPU build of SBD and needs resource_class='gpu', "
            f"which selects the GPU queue; got resource_class={resource_class!r}."
        )

    default_launcher = {
        "local": "single",
        "slurm": "srun",
        "miyabi": "mpiexec.hydra",
        "fugaku": "mpiexec",
    }[hpc_target]
    resolved_launcher = launcher or default_launcher
    ranks = 1 if resolved_launcher == "single" else num_nodes * mpiprocs
    comm_product = task_comm_size * adet_comm_size * bdet_comm_size
    if ranks % comm_product != 0:
        # The job would fail on every rank only after waiting in the queue.
        raise ValueError(
            f"task_comm_size x adet_comm_size x bdet_comm_size = {task_comm_size} x "
            f"{adet_comm_size} x {bdet_comm_size} = {comm_product} must divide the "
            f"{ranks} MPI rank(s) of launcher={resolved_launcher!r}, num_nodes={num_nodes}, "
            f"mpiprocs={mpiprocs}."
        )

    register_sbd_block_types()

    resolved_exec_block_name = execution_profile_block_name or f"exec-sbd-{hpc_target}"
    resolved_hpc_block_name = hpc_profile_block_name or f"hpc-{hpc_target}-sbd"
    default_script_filename = {
        "local": "sbd_solver",
        "slurm": "sbd_solver.slurm",
        "miyabi": "sbd_solver.pbs",
        "fugaku": "sbd_solver.pjm",
    }[hpc_target]
    resolved_script_filename = script_filename or default_script_filename
    resolved_metrics_key = metrics_artifact_key or f"{hpc_target}-sbd-metrics"
    resolved_sbd_executable = str(sbd_executable)
    resolved_environments = dict(environments or {})
    if ompthreads is not None and "OMP_NUM_THREADS" not in resolved_environments:
        resolved_environments["OMP_NUM_THREADS"] = str(ompthreads)
    if hpc_target == "fugaku" and "LD_LIBRARY_PATH" not in resolved_environments:
        sbd_bin_dir = str(Path(resolved_sbd_executable).parent)
        if sbd_bin_dir != ".":
            resolved_environments["LD_LIBRARY_PATH"] = f"{sbd_bin_dir}:$LD_LIBRARY_PATH"

    CommandBlock(
        command_name=command_name,
        executable_key=executable_key,
        description="SBD selected basis diagonalization driver",
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
        launcher=resolved_launcher,
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
        executable_map={executable_key: resolved_sbd_executable},
        slurm_memory=slurm_memory,
        gfscache=gfscache,
        spack_modules=list(spack_modules or []),
        mpi_options_for_pjm=list(mpi_options_for_pjm or []),
        pjm_resources=list(pjm_resources or []),
    ).save(resolved_hpc_block_name, overwrite=True)

    SBDSolverJob(
        root_dir=str(Path(root_dir).expanduser().resolve()),
        command_block_name=command_block_name,
        execution_profile_block_name=resolved_exec_block_name,
        hpc_profile_block_name=resolved_hpc_block_name,
        script_filename=resolved_script_filename,
        metrics_artifact_key=resolved_metrics_key,
        timeout_seconds=timeout_seconds,
        task_comm_size=task_comm_size,
        adet_comm_size=adet_comm_size,
        bdet_comm_size=bdet_comm_size,
        block=block,
        iteration=iteration,
        tolerance=tolerance,
        carryover_ratio=carryover_ratio,
        carryover_type=carryover_type,
        solver_mode=solver_mode,
    ).save(solver_block_name, overwrite=True)

    return {
        "command_block_name": command_block_name,
        "execution_profile_block_name": resolved_exec_block_name,
        "hpc_profile_block_name": resolved_hpc_block_name,
        "solver_block_name": solver_block_name,
    }
