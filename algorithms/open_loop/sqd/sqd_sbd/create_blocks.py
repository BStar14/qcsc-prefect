"""Create the Prefect blocks of the SQD workflow with the SBD Davidson solver.

Each batch is one MPI job of the SBD ``diag`` executable (``diag-gpu`` with
``solver_mode = "gpu"``). The MPI launcher and ranks come from the profile's
``[sbd]`` table; the SBD tuning comes from the ``[sbd]`` table of your file.

Sampling goes through a ``QrmiRuntimeBlock`` (``sampling.runtime_block_name`` of
each run, default ``qrmi-runtime``). It is saved from one of two optional tables:

- ``[qrmi]`` (Slurm only): the three blocks of the QRMI sampler job, plus a
  ``scheduler="slurm"`` runtime block that names them.
- ``[runtime]``: any ``QrmiRuntimeBlock`` fields, for example ``scheduler="iqp"``
  with ``[runtime.qrmi]``; the API key comes from the environment.

Shots and the carryover fraction are flow parameters; the flow sets the SBD
block's ``carryover_ratio`` from ``sqd.carryover_fraction`` on each run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from qcsc_prefect_core.models.gpu_request import validate_gpu_request
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
from sqd_common.runtime_block import describe, runtime_block_name, save_runtime_block

HPC_TARGETS = ("slurm", "local", "miyabi", "fugaku")
SOLVER_MODES = ("cpu", "gpu")
# The MPI launcher of each target when neither the profile nor the file sets one.
DEFAULT_LAUNCHERS = {
    "local": "single",
    "slurm": "srun",
    "miyabi": "mpiexec.hydra",
    "fugaku": "mpiexec",
}
# Run settings that belong to the flow parameters, not to the blocks.
FLOW_PARAMETER_KEYS = {
    "shots": "sampling.shots",
    "sampler_options_json": "sampling.shots",
    "options_variable_name": "sampling.shots",
    "carryover_fraction": "sqd.carryover_fraction",
    "carryover_ratio": "sqd.carryover_fraction",
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Prefect blocks for the SQD SBD workflow "
        "(Slurm, local, Miyabi or Fugaku)."
    )
    parser.add_argument("--config", type=Path, help="Path to TOML or JSON config file.")

    parser.add_argument("--hpc-target", choices=HPC_TARGETS)
    parser.add_argument("--project")
    parser.add_argument("--group")
    parser.add_argument("--queue")
    parser.add_argument("--work-dir")
    parser.add_argument("--sbd-executable")

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

    parser.add_argument("--task-comm-size", type=int)
    parser.add_argument("--adet-comm-size", type=int)
    parser.add_argument("--bdet-comm-size", type=int)
    parser.add_argument("--block", type=int, help="Maximum Davidson subspace size.")
    parser.add_argument("--iteration", type=int, help="Number of Davidson restarts.")
    parser.add_argument("--tolerance", type=float, help="Davidson residual norm threshold.")
    parser.add_argument("--carryover-type", type=int)
    parser.add_argument("--solver-mode", choices=SOLVER_MODES)
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
    """The config file with its profile, ``[advanced]`` and ``[sbd]`` merged in."""
    return load_block_config(path, "sbd")


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


def _positive_int(name: str, value: Any) -> int:
    number = int(value)
    if number < 1:
        raise RuntimeError(f"'{name}' must be at least 1; got {number}.")
    return number


def _check_comm_sizes(block_kwargs: dict[str, Any]) -> None:
    """The SBD communicators split the MPI ranks, so their product must divide them."""
    product = (
        block_kwargs["task_comm_size"]
        * block_kwargs["adet_comm_size"]
        * block_kwargs["bdet_comm_size"]
    )
    ranks = block_kwargs["num_nodes"] * block_kwargs["mpiprocs"]
    if ranks % product:
        raise RuntimeError(
            f"task_comm_size x adet_comm_size x bdet_comm_size = {product} must divide the "
            f"{ranks} MPI ranks (num_nodes x mpiprocs); change the comm sizes or mpiprocs."
        )


def resolve_settings(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    """Merge CLI arguments over the config file.

    Only the keys the SBD blocks use are read, so the fulqrum keys of a profile
    (``backend``, the eigensolver tuning) do not reach them.

    Returns:
        The keyword arguments of ``create_sbd_blocks``.
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
    defaults = solver_block_names("sbd", hpc_target)

    project = _pick_value(args.project, config.get("project"), args.group, config.get("group"))
    if hpc_target in {"miyabi", "fugaku"}:
        project = _require("project/group", project)
    queue = pick("queue")
    if hpc_target != "local":
        queue = _require("queue", queue)
    work_dir = _require("work_dir", pick("work_dir"))
    sbd_executable = _require("sbd_executable", pick("sbd_executable"))

    solver_mode = str(pick("solver_mode", "cpu")).strip().lower()
    if solver_mode not in SOLVER_MODES:
        raise RuntimeError(
            f"'solver_mode' must be one of {', '.join(SOLVER_MODES)}; got {solver_mode!r}."
        )
    resource_class = str(pick("resource_class", "gpu" if solver_mode == "gpu" else "cpu")).strip()
    if solver_mode == "gpu" and resource_class != "gpu":
        raise RuntimeError(
            "solver_mode = 'gpu' runs on the GPU queue; set resource_class = 'gpu' "
            "or use the slurm-gpu or miyabi-gpu profile."
        )
    gpus_per_node = pick("gpus_per_node")
    gpus_per_node = None if gpus_per_node is None else int(gpus_per_node)
    launcher = str(pick("launcher", DEFAULT_LAUNCHERS[hpc_target])).strip()
    mpiprocs = int(pick("mpiprocs", 1 if launcher == "single" else 4))
    if solver_mode == "gpu" and gpus_per_node is not None:
        # One rank per GPU. A profile's mpiprocs gives way to gpus_per_node; only an
        # explicit --mpiprocs that differs is an error.
        if args.mpiprocs is not None and int(args.mpiprocs) != gpus_per_node:
            raise RuntimeError(
                f"solver_mode = 'gpu' runs one MPI rank per GPU, but --mpiprocs "
                f"{args.mpiprocs} differs from gpus_per_node = {gpus_per_node}."
            )
        mpiprocs = gpus_per_node
    ompthreads = pick("ompthreads")
    block_kwargs: dict[str, Any] = dict(
        hpc_target=hpc_target,
        project=str(project or "").strip(),
        queue=str(queue or "").strip(),
        root_dir=str(Path(str(work_dir)).expanduser()),
        # Not resolved: the path must be valid on the compute node, and a bare
        # command name is looked up on the compute node's PATH.
        sbd_executable=str(Path(str(sbd_executable).strip()).expanduser()),
        command_block_name=str(pick("command_block_name", defaults["command_block_name"])).strip(),
        execution_profile_block_name=str(
            pick("execution_profile_block_name", defaults["execution_profile_block_name"])
        ).strip(),
        hpc_profile_block_name=str(
            pick("hpc_profile_block_name", defaults["hpc_profile_block_name"])
        ).strip(),
        solver_block_name=str(pick("solver_block_name", defaults["solver_block_name"])).strip(),
        command_name="sqd-sbd",
        executable_key="sqd_sbd_solver",
        profile_name="sqd-sbd-mpi",
        launcher=launcher,
        num_nodes=_positive_int("num_nodes", pick("num_nodes", 1)),
        mpiprocs=_positive_int("mpiprocs", mpiprocs),
        ompthreads=None if ompthreads is None else int(ompthreads),
        walltime=str(pick("walltime", "01:00:00")).strip(),
        resource_class=resource_class,
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
        task_comm_size=_positive_int("task_comm_size", pick("task_comm_size", 1)),
        adet_comm_size=_positive_int("adet_comm_size", pick("adet_comm_size", 1)),
        bdet_comm_size=_positive_int("bdet_comm_size", pick("bdet_comm_size", 1)),
        block=_positive_int("block", pick("block", 10)),
        iteration=_positive_int("iteration", pick("iteration", 2)),
        tolerance=float(pick("tolerance", 1e-4)),
        carryover_type=int(pick("carryover_type", 1)),
        solver_mode=solver_mode,
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
    if block_kwargs["tolerance"] <= 0:
        raise RuntimeError(f"'tolerance' must be above 0; got {block_kwargs['tolerance']}.")
    _check_comm_sizes(block_kwargs)
    try:
        validate_gpu_request(
            hpc_target=hpc_target,
            resource_class=resource_class,
            gpus_per_node=gpus_per_node,
        )
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
    solver_ref = solver_ref_of("sbd", block_kwargs["solver_block_name"])
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

    try:
        from qcsc_prefect_sbd import create_sbd_blocks
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f'The SBD solver needs qcsc-prefect-sbd: pip install "sqd[sbd]" ({exc})'
        ) from exc

    try:
        block_names = create_sbd_blocks(**block_kwargs)
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    if qrmi is not None:
        create_qrmi_blocks(qrmi)
    changes = save_runtime_block(runtime, runtime_name) if runtime is not None else []
    save_advanced_block(advanced, advanced_name)

    print(f"Saved blocks for SQD SBD workflow (target={block_kwargs['hpc_target']})")
    print(f"  SQD solver block: {block_names['solver_block_name']}")
    print(f"  Command block: {block_names['command_block_name']}")
    print(f"  Execution profile block: {block_names['execution_profile_block_name']}")
    print(f"  HPC profile block: {block_names['hpc_profile_block_name']}")
    print(f"  Work directory: {block_kwargs['root_dir']}")
    print(f"  SBD executable: {block_kwargs['sbd_executable']}")
    if qrmi is not None:
        print(f"  QRMI command block: {qrmi['command_block_name']}")
        print(f"  QRMI execution profile block: {qrmi['execution_profile_block_name']}")
        print(f"  QRMI HPC profile block: {qrmi['hpc_profile_block_name']}")
    if runtime is not None:
        print(f"  QRMI runtime block: {runtime_name} (scheduler={runtime['scheduler']})")
        for change in changes:
            print(f"    {change}")
        if runtime_name != "qrmi-runtime":
            print(f"Set sampling.runtime_block_name of each run to {runtime_name!r}.")
    print(f"  Advanced SQD Settings block: {advanced_name} (Solver Block {solver_ref})")
    if advanced_name != "sqd-advanced-settings":
        print(f"Set advanced_block_name of each run to {advanced_name!r}.")


if __name__ == "__main__":
    main()
