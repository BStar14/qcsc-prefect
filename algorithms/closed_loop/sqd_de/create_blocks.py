from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any


def _import_sbd_solver_block():
    # Supports direct execution (python algorithms/closed_loop/sqd_de/create_blocks.py ...):
    # the script directory is put on sys.path so that the sbd package can be imported below.
    script_dir = str(Path(__file__).resolve().parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    from qcsc_prefect_sbd import SBDSolverJob

    return SBDSolverJob


def _register_block_types(*custom_block_classes) -> None:
    from qcsc_prefect_blocks.common.blocks import (
        CommandBlock,
        ExecutionProfileBlock,
        HPCProfileBlock,
    )

    block_types = [
        CommandBlock,
        ExecutionProfileBlock,
        HPCProfileBlock,
        *custom_block_classes,
    ]
    for block_cls in block_types:
        register = getattr(block_cls, "register_type_and_schema", None)
        if callable(register):
            register()


# QRMI settings that are secrets: read from the environment, never from the config file.
SECRET_RESOURCE_KEYS = ("QRMI_IBM_QCS_IAM_APIKEY",)
RUNTIME_BLOCK_NAME = "qrmi-runtime"


def _require(name: str, value: Any) -> Any:
    if value is None or (isinstance(value, str) and not value.strip()):
        raise RuntimeError(f"Missing required setting: {name}")
    return value


def _runtime_block(runtime: dict[str, Any]):
    """Build the QrmiRuntimeBlock described by the ``[runtime]`` config table.

    Other keys of the table are QrmiRuntimeBlock fields. The SBD flow sets the
    prepare hook, the transpiler and its configuration itself, so they are not accepted.
    """
    from qcsc_prefect.integrations.qiskit.qrmi import QrmiRuntimeBlock

    settings = {"scheduler": "iqp", **runtime}
    qrmi = {str(key): str(value) for key, value in settings.pop("qrmi", {}).items()}
    unsupported = sorted(settings.keys() & {"transpiler", "transpile_config", "resource_config"})
    if unsupported:
        raise RuntimeError(
            f"[runtime] must not set {', '.join(unsupported)}; the SBD flow sets the "
            "transpiler, and resource_config is built from [runtime.qrmi] and the environment"
        )
    _require("runtime.resource", settings.get("resource"))
    if settings["scheduler"] == "iqp":
        secrets = sorted(qrmi.keys() & set(SECRET_RESOURCE_KEYS))
        if secrets:
            raise RuntimeError(
                f"Do not write {', '.join(secrets)} in the config file; set it in the environment"
            )
        for key in SECRET_RESOURCE_KEYS:
            qrmi[key] = _require(f"environment variable {key}", os.environ.get(key))
        settings["resource_config"] = qrmi
    elif qrmi:
        raise RuntimeError(
            '[runtime.qrmi] is only used by scheduler="iqp"; Slurm compute nodes read QRMI '
            "settings from their own environment"
        )
    block = QrmiRuntimeBlock(**settings)
    block.get_runtime()  # Validate the settings before saving.
    return block


def _set_variable(variable_name: str, value: Any) -> None:
    subprocess.run(
        ["prefect", "variable", "set", variable_name, json.dumps(value), "--overwrite"],
        check=True,
    )


def _quantum_source_block(settings: dict[str, Any]):
    """Validate source deployment settings before any Blocks are registered or saved."""
    from qcsc_prefect.integrations.qiskit.sources import QuantumSourceBlock

    values = dict(settings)
    name = values.pop("name", "quantum-source")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("source.name must be a nonempty Block document name")
    block = QuantumSourceBlock.model_validate(values)
    return name, block


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Prefect blocks for SBD workflow (Miyabi or Fugaku or Slurm)."
    )
    parser.add_argument("--config", type=Path, help="Path to TOML/JSON config file.")

    parser.add_argument("--hpc-target", choices=["miyabi", "fugaku", "slurm"])
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
    parser.add_argument("--modules", nargs="+")
    parser.add_argument("--mpi-options", nargs="*")
    parser.add_argument("--pre-commands", nargs="*")

    parser.add_argument("--fugaku-gfscache")
    parser.add_argument("--fugaku-spack-modules", nargs="+")
    parser.add_argument("--fugaku-mpi-options-for-pjm", nargs="*")

    parser.add_argument("--script-filename")
    parser.add_argument("--metrics-artifact-key")

    parser.add_argument("--command-block-name")
    parser.add_argument("--execution-profile-block-name")
    parser.add_argument("--hpc-profile-block-name")
    parser.add_argument("--solver-block-name")
    parser.add_argument("--options-variable-name")

    parser.add_argument("--task-comm-size", type=int)
    parser.add_argument("--adet-comm-size", type=int)
    parser.add_argument("--bdet-comm-size", type=int)
    parser.add_argument("--block", type=int)
    parser.add_argument("--iteration", type=int)
    parser.add_argument("--tolerance", type=float)
    parser.add_argument("--carryover-ratio", type=float)
    parser.add_argument("--solver-mode", choices=["cpu", "gpu"])
    parser.add_argument("--shots", type=int)
    parser.add_argument(
        "--sqd-options-json",
        help='Raw JSON for Prefect variable value (e.g. \'{"params": {"shots": 50000}}\').',
    )
    return parser.parse_args()


def _load_config_file(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Config file was not found: {path}")
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    with path.open("rb") as f:
        return tomllib.load(f)


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


def _default_block_names(*, hpc_target: str, solver_mode: str) -> dict[str, str]:
    is_gpu = solver_mode == "gpu"
    if hpc_target == "miyabi":
        return {
            "profile_name": "sbd-gpu" if is_gpu else "sbd-mpi",
            "execution_profile_block_name": "exec-sbd-gpu" if is_gpu else "exec-sbd-mpi",
            "hpc_profile_block_name": "hpc-miyabi-sbd-gpu" if is_gpu else "hpc-miyabi-sbd",
            "solver_block_name": "davidson-solver-gpu" if is_gpu else "davidson-solver",
        }
    if hpc_target == "fugaku":
        return {
            "profile_name": "sbd-gpu" if is_gpu else "sbd-mpi",
            "execution_profile_block_name": "exec-sbd-fugaku-gpu" if is_gpu else "exec-sbd-fugaku",
            "hpc_profile_block_name": "hpc-fugaku-sbd-gpu" if is_gpu else "hpc-fugaku-sbd",
            "solver_block_name": "davidson-solver-gpu" if is_gpu else "davidson-solver",
        }
    return {
        "profile_name": "sbd-gpu" if is_gpu else "sbd-mpi",
        "execution_profile_block_name": "exec-sbd-slurm-gpu" if is_gpu else "exec-sbd-slurm",
        "hpc_profile_block_name": "hpc-slurm-sbd-gpu" if is_gpu else "hpc-slurm-sbd",
        "solver_block_name": "davidson-solver-gpu" if is_gpu else "davidson-solver-slurm",
    }


def _normalize_modules_for_target(
    *, is_miyabi: bool, solver_mode: str, modules: list[str] | None
) -> list[str]:
    if is_miyabi and solver_mode == "gpu":
        return []
    return list(modules or [])


def _env_values() -> dict[str, Any]:
    def env_int(name: str) -> int | None:
        raw = os.getenv(name, "").strip()
        if not raw:
            return None
        return int(raw)

    def env_float(name: str) -> float | None:
        raw = os.getenv(name, "").strip()
        if not raw:
            return None
        return float(raw)

    def env_csv(name: str) -> list[str] | None:
        raw = os.getenv(name, "").strip()
        if not raw:
            return None
        return [item.strip() for item in raw.split(",") if item.strip()]

    def env_first_str(*names: str) -> str | None:
        for name in names:
            raw = os.getenv(name, "").strip()
            if raw:
                return raw
        return None

    def env_first_int(*names: str) -> int | None:
        for name in names:
            value = env_int(name)
            if value is not None:
                return value
        return None

    def env_first_csv(*names: str) -> list[str] | None:
        for name in names:
            value = env_csv(name)
            if value:
                return value
        return None

    return {
        "hpc_target": env_first_str("SBD_HPC_TARGET"),
        "project": env_first_str("SBD_PROJECT", "MIYABI_PBS_PROJECT", "FUGAKU_PROJECT"),
        "group": env_first_str("SBD_GROUP", "FUGAKU_GROUP", "FUGAKU_PROJECT"),
        "queue": env_first_str("SBD_QUEUE", "MIYABI_PBS_QUEUE", "FUGAKU_RSCGRP"),
        "work_dir": env_first_str("SBD_WORK_DIR"),
        "sbd_executable": env_first_str("SBD_EXECUTABLE"),
        "launcher": env_first_str("SBD_LAUNCHER", "MIYABI_LAUNCHER", "FUGAKU_LAUNCHER"),
        "walltime": env_first_str("SBD_WALLTIME", "MIYABI_WALLTIME", "FUGAKU_WALLTIME"),
        "num_nodes": env_first_int("SBD_NUM_NODES", "MIYABI_NUM_NODES", "FUGAKU_NUM_NODES"),
        "mpiprocs": env_first_int("SBD_MPIPROCS", "MIYABI_MPIPROCS", "FUGAKU_MPIPROCS"),
        "ompthreads": env_first_int("SBD_OMPTHREADS", "MIYABI_OMPTHREADS", "FUGAKU_OMPTHREADS"),
        "modules": env_first_csv("SBD_MODULES", "MIYABI_MODULES"),
        "mpi_options": env_first_csv("SBD_MPI_OPTIONS", "MIYABI_MPI_OPTIONS", "FUGAKU_MPI_OPTIONS"),
        "pre_commands": env_first_csv("SBD_PRE_COMMANDS"),
        "fugaku_gfscache": env_first_str("FUGAKU_GFSCACHE"),
        "fugaku_spack_modules": env_first_csv("FUGAKU_SPACK_MODULES"),
        "fugaku_mpi_options_for_pjm": env_first_csv("FUGAKU_MPI_OPTIONS_FOR_PJM"),
        "script_filename": env_first_str("SBD_SCRIPT_FILENAME"),
        "metrics_artifact_key": env_first_str("SBD_METRICS_ARTIFACT_KEY"),
        "command_block_name": env_first_str("SBD_CMD_BLOCK_NAME"),
        "execution_profile_block_name": env_first_str("SBD_EXEC_PROFILE_BLOCK_NAME"),
        "hpc_profile_block_name": env_first_str("SBD_HPC_PROFILE_BLOCK_NAME"),
        "solver_block_name": env_first_str("SBD_SOLVER_BLOCK_NAME"),
        "options_variable_name": env_first_str("SBD_OPTIONS_VARIABLE"),
        "task_comm_size": env_int("SBD_TASK_COMM_SIZE"),
        "adet_comm_size": env_int("SBD_ADET_COMM_SIZE"),
        "bdet_comm_size": env_int("SBD_BDET_COMM_SIZE"),
        "block": env_int("SBD_BLOCK"),
        "iteration": env_int("SBD_ITERATION"),
        "tolerance": env_float("SBD_TOLERANCE"),
        "carryover_ratio": env_float("SBD_CARRYOVER_RATIO"),
        "solver_mode": env_first_str("SBD_SOLVER_MODE"),
        "shots": env_int("SBD_SHOTS"),
        "sqd_options_json": env_first_str("SBD_OPTIONS_JSON"),
    }


def main() -> None:
    args = _parse_args()
    config = _load_config_file(args.config)
    env = _env_values()
    source = _quantum_source_block(config["source"]) if "source" in config else None

    from qcsc_prefect.integrations.qiskit.sources import QuantumSourceBlock
    from qcsc_prefect_blocks.common.blocks import (
        CommandBlock,
        ExecutionProfileBlock,
        HPCProfileBlock,
    )

    sbd_solver_cls = _import_sbd_solver_block()
    _register_block_types(sbd_solver_cls, QuantumSourceBlock)

    hpc_target = (
        str(_pick_value(args.hpc_target, config.get("hpc_target"), env.get("hpc_target"), "miyabi"))
        .strip()
        .lower()
    )
    if hpc_target not in {"miyabi", "fugaku", "slurm"}:
        raise RuntimeError("'hpc_target' must be either 'miyabi' or 'fugaku' or 'slurm'.")
    is_miyabi = hpc_target == "miyabi"
    is_fugaku = hpc_target == "fugaku"

    if is_miyabi:
        project = _pick_value(
            args.project,
            config.get("project"),
            env.get("project"),
            args.group,
            config.get("group"),
            env.get("group"),
        )
    elif is_fugaku:
        project = _pick_value(
            args.group,
            config.get("group"),
            env.get("group"),
            args.project,
            config.get("project"),
            env.get("project"),
        )
    else:
        project = _pick_value(
            args.project,
            config.get("project"),
            env.get("project"),
        )
    if not project:
        if is_miyabi:
            raise RuntimeError(
                "Set 'project' in --config/--project or SBD_PROJECT/MIYABI_PBS_PROJECT."
            )
        raise RuntimeError(
            "Set 'group' in --config/--group (preferred) or 'project' for compatibility, "
            "or use SBD_GROUP/FUGAKU_GROUP/FUGAKU_PROJECT."
        )

    queue = _pick_value(args.queue, config.get("queue"), env.get("queue"))
    if not queue:
        raise RuntimeError(
            "Set 'queue' in --config/--queue or SBD_QUEUE/MIYABI_PBS_QUEUE/FUGAKU_RSCGRP."
        )

    work_dir = _pick_value(args.work_dir, config.get("work_dir"), env.get("work_dir"))
    if not work_dir:
        raise RuntimeError("Set 'work_dir' in --config, --work-dir, or SBD_WORK_DIR.")

    sbd_executable = _pick_value(
        args.sbd_executable, config.get("sbd_executable"), env.get("sbd_executable")
    )
    if not sbd_executable:
        raise RuntimeError("Set 'sbd_executable' in --config, --sbd-executable, or SBD_EXECUTABLE.")

    # launcher_default = "mpiexec.hydra" if is_miyabi else "mpiexec"
    if is_miyabi:
        launcher_default = "mpiexec.hydra"
    elif is_fugaku:
        launcher_default = "mpiexec"
    else:
        launcher_default = "srun"

    launcher = str(
        _pick_value(args.launcher, config.get("launcher"), env.get("launcher"), launcher_default)
    ).strip()
    walltime = str(
        _pick_value(args.walltime, config.get("walltime"), env.get("walltime"), "02:00:00")
    ).strip()

    num_nodes = int(_pick_value(args.num_nodes, config.get("num_nodes"), env.get("num_nodes"), 1))
    mpiprocs = int(_pick_value(args.mpiprocs, config.get("mpiprocs"), env.get("mpiprocs"), 4))
    ompthreads_raw = _pick_value(args.ompthreads, config.get("ompthreads"), env.get("ompthreads"))
    ompthreads = int(ompthreads_raw) if ompthreads_raw is not None else None

    modules_default = ["intel/2023.2.0", "impi/2021.10.0"] if is_miyabi else []
    modules = _normalize_str_list(
        _pick_value(args.modules, config.get("modules"), env.get("modules"), modules_default)
    )
    mpi_options = _normalize_str_list(
        _pick_value(args.mpi_options, config.get("mpi_options"), env.get("mpi_options"), [])
    )
    pre_commands = (
        _normalize_str_list(
            _pick_value(args.pre_commands, config.get("pre_commands"), env.get("pre_commands"), [])
        )
        or []
    )
    environments = _normalize_str_dict(config.get("environments")) or {}

    fugaku_gfscache = str(
        _pick_value(
            args.fugaku_gfscache,
            config.get("fugaku_gfscache"),
            env.get("fugaku_gfscache"),
            "/vol0002",
        )
    ).strip()
    fugaku_spack_modules = _normalize_str_list(
        _pick_value(
            args.fugaku_spack_modules,
            config.get("fugaku_spack_modules"),
            env.get("fugaku_spack_modules"),
            [],
        )
    )
    fugaku_mpi_options_for_pjm = _normalize_str_list(
        _pick_value(
            args.fugaku_mpi_options_for_pjm,
            config.get("fugaku_mpi_options_for_pjm"),
            env.get("fugaku_mpi_options_for_pjm"),
            [],
        )
    )

    task_comm_size = int(
        _pick_value(args.task_comm_size, config.get("task_comm_size"), env.get("task_comm_size"), 1)
    )
    adet_comm_size = int(
        _pick_value(args.adet_comm_size, config.get("adet_comm_size"), env.get("adet_comm_size"), 1)
    )
    bdet_comm_size = int(
        _pick_value(args.bdet_comm_size, config.get("bdet_comm_size"), env.get("bdet_comm_size"), 1)
    )
    block = int(_pick_value(args.block, config.get("block"), env.get("block"), 4))
    iteration = int(_pick_value(args.iteration, config.get("iteration"), env.get("iteration"), 1))
    tolerance = float(
        _pick_value(args.tolerance, config.get("tolerance"), env.get("tolerance"), 1e-2)
    )
    carryover_ratio = float(
        _pick_value(
            args.carryover_ratio, config.get("carryover_ratio"), env.get("carryover_ratio"), 0.1
        )
    )
    solver_mode = str(
        _pick_value(args.solver_mode, config.get("solver_mode"), env.get("solver_mode"), "cpu")
    ).strip()
    resource_class = "gpu" if solver_mode == "gpu" else "cpu"
    user_args = _normalize_str_list(config.get("user_args")) or []
    if is_miyabi and solver_mode == "gpu":
        if "unset OMPI_MCA_mca_base_env_list" not in pre_commands:
            pre_commands.insert(0, "unset OMPI_MCA_mca_base_env_list")
        environments.setdefault("MIYABI", "G")
    modules = _normalize_modules_for_target(
        is_miyabi=is_miyabi, solver_mode=solver_mode, modules=modules
    )

    default_names = _default_block_names(hpc_target=hpc_target, solver_mode=solver_mode)

    command_block_name = str(
        _pick_value(
            args.command_block_name,
            config.get("command_block_name"),
            env.get("command_block_name"),
            "cmd-sbd-diag",
        )
    ).strip()

    execution_profile_block_name = str(
        _pick_value(
            args.execution_profile_block_name,
            config.get("execution_profile_block_name"),
            env.get("execution_profile_block_name"),
            default_names["execution_profile_block_name"],
        )
    ).strip()

    hpc_profile_block_name = str(
        _pick_value(
            args.hpc_profile_block_name,
            config.get("hpc_profile_block_name"),
            env.get("hpc_profile_block_name"),
            default_names["hpc_profile_block_name"],
        )
    ).strip()

    solver_block_name = str(
        _pick_value(
            args.solver_block_name,
            config.get("solver_block_name"),
            env.get("solver_block_name"),
            default_names["solver_block_name"],
        )
    ).strip()

    options_variable_name = str(
        _pick_value(
            args.options_variable_name,
            config.get("options_variable_name"),
            env.get("options_variable_name"),
            "sqd_options",
        )
    ).strip()

    if is_miyabi:
        script_filename_default = "sbd_solver.pbs"
    elif is_fugaku:
        script_filename_default = "sbd_solver.pjm"
    else:
        script_filename_default = "sbd_solver.slurm"
    script_filename = str(
        _pick_value(
            args.script_filename,
            config.get("script_filename"),
            env.get("script_filename"),
            script_filename_default,
        )
    ).strip()

    if is_miyabi:
        metrics_artifact_key_default = "miyabi-sbd-metrics"
    elif is_fugaku:
        metrics_artifact_key_default = "fugaku-sbd-metrics"
    else:
        metrics_artifact_key_default = "slurm-sbd-metrics"
    metrics_artifact_key = str(
        _pick_value(
            args.metrics_artifact_key,
            config.get("metrics_artifact_key"),
            env.get("metrics_artifact_key"),
            metrics_artifact_key_default,
        )
    ).strip()

    shots_default = 500000 if is_miyabi else 50000
    shots = int(_pick_value(args.shots, config.get("shots"), env.get("shots"), shots_default))
    sqd_options_json = _pick_value(
        args.sqd_options_json, config.get("sqd_options_json"), env.get("sqd_options_json")
    )

    CommandBlock(
        command_name="sbd-diag",
        executable_key="sbd_diag",
        description="SBD diagonalization executable",
        default_args=[],
    ).save(command_block_name, overwrite=True)

    ExecutionProfileBlock(
        profile_name=default_names["profile_name"],
        command_name="sbd-diag",
        resource_class=resource_class,
        num_nodes=num_nodes,
        mpiprocs=mpiprocs,
        ompthreads=ompthreads,
        walltime=walltime,
        launcher=launcher,
        mpi_options=mpi_options or [],
        modules=modules or [],
        pre_commands=pre_commands,
        environments=environments,
    ).save(execution_profile_block_name, overwrite=True)

    executable_path = str(Path(str(sbd_executable)).expanduser().resolve())

    if is_miyabi:
        HPCProfileBlock(
            hpc_target="miyabi",
            queue_cpu=str(queue),
            queue_gpu="regular-g",
            project_cpu=str(project),
            project_gpu=str(project),
            executable_map={"sbd_diag": executable_path},
        ).save(hpc_profile_block_name, overwrite=True)
    elif is_fugaku:
        HPCProfileBlock(
            hpc_target="fugaku",
            queue_cpu=str(queue),
            queue_gpu=str(queue),
            project_cpu=str(project),
            project_gpu=str(project),
            executable_map={"sbd_diag": executable_path},
            gfscache=fugaku_gfscache or None,
            spack_modules=fugaku_spack_modules or [],
            mpi_options_for_pjm=fugaku_mpi_options_for_pjm or [],
        ).save(hpc_profile_block_name, overwrite=True)
    else:
        HPCProfileBlock(
            hpc_target="slurm",
            queue_cpu=str(queue),
            queue_gpu=str(queue),
            project_cpu=str(project),
            project_gpu=str(project),
            executable_map={"sbd_diag": executable_path},
        ).save(hpc_profile_block_name, overwrite=True)

    sbd_solver_cls(
        root_dir=str(Path(str(work_dir)).expanduser().resolve()),
        command_block_name=command_block_name,
        execution_profile_block_name=execution_profile_block_name,
        hpc_profile_block_name=hpc_profile_block_name,
        script_filename=script_filename,
        metrics_artifact_key=metrics_artifact_key,
        task_comm_size=task_comm_size,
        adet_comm_size=adet_comm_size,
        bdet_comm_size=bdet_comm_size,
        block=block,
        iteration=iteration,
        tolerance=tolerance,
        carryover_ratio=carryover_ratio,
        solver_mode=solver_mode,
        user_args=user_args,
    ).save(solver_block_name, overwrite=True)

    if sqd_options_json:
        options_value = json.loads(str(sqd_options_json))
    else:
        options_value = {"params": {"shots": shots}}
    _set_variable(options_variable_name, options_value)

    runtime_saved = "runtime" in config
    if runtime_saved:
        _runtime_block(config["runtime"]).save(RUNTIME_BLOCK_NAME, overwrite=True)

    source_name = None
    if source is not None:
        source_name, source_block = source
        source_block.save(source_name, overwrite=True)

    print(f"Saved blocks and variable for SBD workflow (target={hpc_target})")
    print(f"  SBD solver block: {solver_block_name}")
    print(f"  Command block: {command_block_name}")
    print(f"  Execution profile block: {execution_profile_block_name}")
    print(f"  HPC profile block: {hpc_profile_block_name}")
    print(f"  Options variable: {options_variable_name}")
    print(f"  Work directory: {Path(str(work_dir)).expanduser().resolve()}")
    print(f"  SBD executable: {Path(str(sbd_executable)).expanduser().resolve()}")
    print(f"  Script filename: {script_filename}")
    print(f"  Metrics artifact key: {metrics_artifact_key}")
    if runtime_saved:
        print(f"  QRMI runtime block: {RUNTIME_BLOCK_NAME}")
    if source_name:
        print(f"  Quantum source block: {source_name}")


if __name__ == "__main__":
    main()
