"""Create the Prefect blocks of the SQD workflow with the fulqrum solver.

Sampling goes through a ``QrmiRuntimeBlock`` (``sampling.runtime_block_name`` of
each run, default ``qrmi-runtime``). It is saved from one of two optional tables:

- ``[qrmi]`` (Slurm only): the three blocks of the QRMI sampler job, plus a
  ``scheduler="slurm"`` runtime block that names them.
- ``[runtime]``: any ``QrmiRuntimeBlock`` fields, for example ``scheduler="iqp"``
  with ``[runtime.qrmi]``; the API key comes from the environment.

Shots are a flow parameter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from qcsc_prefect_core.models.gpu_request import validate_gpu_request
from qcsc_prefect_fulqrum.solver import FulqrumSolverOptions
from sqd_common.advanced_blocks import advanced_block_name, advanced_settings, save_advanced_block
from sqd_common.block_config import load_block_config
from sqd_common.block_names import solver_block_names
from sqd_common.block_names import solver_ref as solver_ref_of
from sqd_common.qrmi_blocks import (
    add_qrmi_arguments,
    create_qrmi_blocks,
    resolve_qrmi_settings,
    resolve_runtime_settings,
)
from sqd_common.runtime_block import build_runtime_block, describe, runtime_block_name

HPC_TARGETS = ("slurm", "local", "miyabi", "fugaku")
BACKENDS = ("cpu", "gpu", "gpu-mpi")
# Run settings that belong to the flow parameters, not to the blocks.
FLOW_PARAMETER_KEYS = {
    "shots": "sampling.shots",
    "sampler_options_json": "sampling.shots",
    "options_variable_name": "sampling.shots",
    "carryover_fraction": "sqd.carryover_fraction",
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Prefect blocks for the SQD fulqrum workflow "
        "(Slurm, local, Miyabi or Fugaku)."
    )
    parser.add_argument("--config", type=Path, help="Path to TOML or JSON config file.")

    parser.add_argument("--hpc-target", choices=HPC_TARGETS)
    parser.add_argument("--project")
    parser.add_argument("--group")
    parser.add_argument("--queue")
    parser.add_argument("--work-dir")
    parser.add_argument("--fulqrum-executable")

    parser.add_argument("--launcher")
    parser.add_argument("--walltime")
    parser.add_argument("--num-nodes", type=int)
    parser.add_argument("--mpiprocs", type=int)
    parser.add_argument("--ompthreads", type=int)
    parser.add_argument("--resource-class", choices=["cpu", "gpu"])
    parser.add_argument("--gpus-per-node", type=int)
    parser.add_argument("--modules", nargs="+")
    parser.add_argument("--mpi-options", nargs="*")
    parser.add_argument("--pre-commands", nargs="*")
    parser.add_argument("--slurm-memory")

    parser.add_argument("--fugaku-gfscache")
    parser.add_argument("--fugaku-spack-modules", nargs="+")
    parser.add_argument("--fugaku-mpi-options-for-pjm", nargs="*")
    parser.add_argument("--fugaku-pjm-resources", nargs="*")

    parser.add_argument("--script-filename")
    parser.add_argument("--metrics-artifact-key")
    parser.add_argument("--timeout-seconds", type=float)

    parser.add_argument("--command-block-name")
    parser.add_argument("--execution-profile-block-name")
    parser.add_argument("--hpc-profile-block-name")
    parser.add_argument("--solver-block-name")

    parser.add_argument("--backend", choices=BACKENDS)
    parser.add_argument("--eigensolver", choices=["scipy", "primme"])
    parser.add_argument("--hamiltonian-format", choices=["csr", "matrix_free"])
    parser.add_argument("--tol", type=float)
    parser.add_argument("--max-iter", type=int)
    parser.add_argument("--dense-max-dim", type=int)
    parser.add_argument("--max-basis-size", type=int)
    parser.add_argument("--block-size", type=int)
    parser.add_argument("--min-restart-size", type=int)
    parser.add_argument("--primme-method")
    parser.add_argument("--convergence", choices=["abs", "rel", "norm", "energy"])
    parser.add_argument("--energy-tol", type=float)
    parser.add_argument(
        "--process-grid",
        type=int,
        nargs=2,
        metavar=("P_ALPHA", "P_BETA"),
        help="gpu-mpi: ranks over the alpha and beta strings; their product is the rank count.",
    )
    parser.add_argument("--comm-backend", choices=["auto", "nccl", "host_mpi", "cuda_mpi"])
    parser.add_argument(
        "--return-sci-state",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Whether to return the ground-state amplitudes as an SCIState.",
    )
    add_qrmi_arguments(parser)
    parser.add_argument(
        "--runtime-block-name",
        help="Name of the QrmiRuntimeBlock to save (default: qrmi-runtime).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved settings as JSON without saving blocks.",
    )
    return parser.parse_args(argv)


def _load_config_file(path: Path | None) -> dict[str, Any]:
    """The config file with its profile and ``[advanced]`` merged in."""
    return load_block_config(path)


def _pick_value(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _normalize_str_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    raise ValueError(f"Expected list[str] or CSV string, got: {type(value)}")


def _normalize_str_dict(value: Any) -> dict[str, str] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        normalized = {
            str(key).strip(): str(val).strip() for key, val in value.items() if str(key).strip()
        }
        return normalized or None
    raise ValueError(f"Expected dict[str, str], got: {type(value)}")


def _require(name: str, value: Any) -> Any:
    if value is None or (isinstance(value, str) and not value.strip()):
        raise RuntimeError(f"Missing required setting: {name}")
    return value


def _optional_str(value: Any) -> str | None:
    if value is None or not str(value).strip():
        return None
    return str(value).strip()


# create_fulqrum_blocks keywords that are FulqrumSolverOptions fields too.
_SOLVER_OPTION_KEYS = (
    "eigensolver",
    "hamiltonian_format",
    "tol",
    "max_iter",
    "dense_max_dim",
    "backend",
    "max_basis_size",
    "block_size",
    "min_restart_size",
    "primme_method",
    "convergence",
    "energy_tol",
    "process_grid",
    "comm_backend",
)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _tuning_settings(pick) -> dict[str, Any]:
    """The eigensolver tuning keys; unset ones keep the solver block defaults."""
    grid = pick("process_grid")
    if grid is not None:
        grid = tuple(int(p) for p in grid)
        if len(grid) != 2:
            raise RuntimeError(f"'process_grid' must be [alpha ranks, beta ranks]; got {grid}.")
    method = pick("primme_method")
    return dict(
        max_basis_size=_optional_int(pick("max_basis_size")),
        block_size=_optional_int(pick("block_size")),
        min_restart_size=_optional_int(pick("min_restart_size")),
        primme_method=None if method is None else str(method).strip(),
        convergence=str(pick("convergence", "abs")).strip(),
        energy_tol=float(pick("energy_tol", 1e-6)),
        process_grid=grid,
        comm_backend=str(pick("comm_backend", "auto")).strip(),
    )


def resolve_settings(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    """Merge CLI arguments over the config file.

    Returns:
        The keyword arguments of ``create_fulqrum_blocks``.
    """
    for key, parameter in FLOW_PARAMETER_KEYS.items():
        if key in config:
            raise RuntimeError(
                f"'{key}' is not a block setting. Set {parameter} in the flow "
                "parameters of each run instead."
            )

    def pick(name: str, default: Any = None) -> Any:
        return _pick_value(getattr(args, name, None), config.get(name), default)

    hpc_target = str(pick("hpc_target", "slurm")).strip().lower()
    if hpc_target not in HPC_TARGETS:
        raise RuntimeError(f"'hpc_target' must be one of {', '.join(HPC_TARGETS)}.")
    defaults = solver_block_names("fulqrum", hpc_target)

    project = _pick_value(args.project, config.get("project"), args.group, config.get("group"))
    if hpc_target in {"miyabi", "fugaku"}:
        project = _require("project/group", project)
    queue = pick("queue")
    if hpc_target != "local":
        queue = _require("queue", queue)
    work_dir = _require("work_dir", pick("work_dir"))
    fulqrum_executable = _require("fulqrum_executable", pick("fulqrum_executable"))

    max_iter = pick("max_iter")
    ompthreads = pick("ompthreads")
    gpus_per_node = pick("gpus_per_node")
    gpus_per_node = None if gpus_per_node is None else int(gpus_per_node)
    backend = str(pick("backend", "cpu")).strip().lower()
    if backend not in BACKENDS:
        raise RuntimeError(f"'backend' must be one of {', '.join(BACKENDS)}; got {backend!r}.")
    launcher = str(pick("launcher", "single")).strip()
    if backend == "gpu-mpi" and launcher == "single":
        raise RuntimeError(
            "backend = 'gpu-mpi' runs one MPI rank per GPU; set launcher to an MPI launcher "
            "such as 'srun' or 'mpirun'."
        )
    mpiprocs = pick("mpiprocs", 1)
    if backend == "gpu-mpi" and gpus_per_node is not None:
        # One rank per GPU. A profile's mpiprocs (1 for the CPU profiles) gives way to
        # gpus_per_node; only an explicit --mpiprocs that differs is an error.
        if args.mpiprocs is not None and int(args.mpiprocs) != gpus_per_node:
            raise RuntimeError(
                f"backend = 'gpu-mpi' runs one MPI rank per GPU, but --mpiprocs "
                f"{args.mpiprocs} differs from gpus_per_node = {gpus_per_node}."
            )
        mpiprocs = gpus_per_node
    block_kwargs: dict[str, Any] = dict(
        hpc_target=hpc_target,
        project=str(project or "").strip(),
        queue=str(queue or "").strip(),
        root_dir=str(Path(str(work_dir)).expanduser()),
        # Not resolved: the path must be valid on the compute node, and a bare
        # command name is looked up on the compute node's PATH.
        fulqrum_executable=str(Path(str(fulqrum_executable).strip()).expanduser()),
        command_block_name=str(pick("command_block_name", defaults["command_block_name"])).strip(),
        execution_profile_block_name=str(
            pick("execution_profile_block_name", defaults["execution_profile_block_name"])
        ).strip(),
        hpc_profile_block_name=str(
            pick("hpc_profile_block_name", defaults["hpc_profile_block_name"])
        ).strip(),
        solver_block_name=str(pick("solver_block_name", defaults["solver_block_name"])).strip(),
        command_name="sqd-fulqrum",
        executable_key="sqd_fulqrum_solver",
        profile_name="sqd-fulqrum-omp",
        launcher=launcher,
        num_nodes=int(pick("num_nodes", 1)),
        mpiprocs=int(mpiprocs),
        ompthreads=None if ompthreads is None else int(ompthreads),
        walltime=str(pick("walltime", "01:00:00")).strip(),
        resource_class=str(pick("resource_class", "cpu")).strip(),
        gpus_per_node=gpus_per_node,
        modules=_normalize_str_list(pick("modules")),
        mpi_options=_normalize_str_list(pick("mpi_options")),
        pre_commands=_normalize_str_list(pick("pre_commands")),
        environments=_normalize_str_dict(config.get("environments")),
        script_filename=str(pick("script_filename", defaults["script_filename"])).strip(),
        metrics_artifact_key=str(
            pick("metrics_artifact_key", defaults["metrics_artifact_key"])
        ).strip(),
        timeout_seconds=float(pick("timeout_seconds", 7200.0)),
        eigensolver=str(pick("eigensolver", "scipy")).strip(),
        hamiltonian_format=str(pick("hamiltonian_format", "csr")).strip(),
        tol=float(pick("tol", 1e-8)),
        max_iter=None if max_iter is None else int(max_iter),
        dense_max_dim=int(pick("dense_max_dim", 64)),
        return_sci_state=bool(pick("return_sci_state", False)),
        **_tuning_settings(pick),
        backend=backend,
        slurm_memory=_optional_str(pick("slurm_memory")),
        gfscache=_optional_str(_pick_value(args.fugaku_gfscache, config.get("fugaku_gfscache"))),
        spack_modules=_normalize_str_list(
            _pick_value(args.fugaku_spack_modules, config.get("fugaku_spack_modules"))
        ),
        mpi_options_for_pjm=_normalize_str_list(
            _pick_value(args.fugaku_mpi_options_for_pjm, config.get("fugaku_mpi_options_for_pjm"))
        ),
        pjm_resources=_normalize_str_list(
            _pick_value(args.fugaku_pjm_resources, config.get("fugaku_pjm_resources"))
        ),
    )
    try:
        validate_gpu_request(
            hpc_target=hpc_target,
            resource_class=block_kwargs["resource_class"],
            gpus_per_node=gpus_per_node,
        )
        # The eigensolver tuning must fit the eigensolver and backend; --dry-run checks it too.
        FulqrumSolverOptions(**{k: block_kwargs[k] for k in _SOLVER_OPTION_KEYS})
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    return block_kwargs


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    config = _load_config_file(args.config)
    block_kwargs = resolve_settings(args, config)
    qrmi = resolve_qrmi_settings(args, config, block_kwargs)
    runtime = resolve_runtime_settings(config, qrmi)
    runtime_name = runtime_block_name(args.runtime_block_name, config)
    solver_ref = solver_ref_of("fulqrum", block_kwargs["solver_block_name"])
    advanced = advanced_settings(config, solver_ref)
    advanced_name = advanced_block_name(config)

    if args.dry_run:
        resolved = {"blocks": block_kwargs}
        resolved["advanced_block"] = {"name": advanced_name, **advanced.model_dump()}
        if qrmi is not None:
            resolved["qrmi_blocks"] = qrmi
        if runtime is not None:
            resolved["runtime_block"] = {"name": runtime_name, **describe(runtime)}
        print(json.dumps(resolved, indent=2, default=str))
        return

    from qcsc_prefect_fulqrum import create_fulqrum_blocks

    block_names = create_fulqrum_blocks(**block_kwargs)
    if qrmi is not None:
        create_qrmi_blocks(qrmi)
    if runtime is not None:
        build_runtime_block(runtime).save(runtime_name, overwrite=True)
    save_advanced_block(advanced, advanced_name)

    print(f"Saved blocks for SQD fulqrum workflow (target={block_kwargs['hpc_target']})")
    print(f"  SQD solver block: {block_names['solver_block_name']}")
    print(f"  Command block: {block_names['command_block_name']}")
    print(f"  Execution profile block: {block_names['execution_profile_block_name']}")
    print(f"  HPC profile block: {block_names['hpc_profile_block_name']}")
    print(f"  Work directory: {block_kwargs['root_dir']}")
    print(f"  Fulqrum executable: {block_kwargs['fulqrum_executable']}")
    if qrmi is not None:
        print(f"  QRMI command block: {qrmi['command_block_name']}")
        print(f"  QRMI execution profile block: {qrmi['execution_profile_block_name']}")
        print(f"  QRMI HPC profile block: {qrmi['hpc_profile_block_name']}")
        print(f"  QRMI resource (slurm_qpu): {qrmi['resource']}")
    if runtime is not None:
        print(f"  QRMI runtime block: {runtime_name} (scheduler={runtime['scheduler']})")
        if runtime_name != "qrmi-runtime":
            print(f"Set sampling.runtime_block_name of each run to {runtime_name!r}.")
    print(f"  Advanced SQD Settings block: {advanced_name} (Solver Block {solver_ref})")
    if advanced_name != "sqd-advanced-settings":
        print(f"Set advanced_block_name of each run to {advanced_name!r}.")


if __name__ == "__main__":
    main()
