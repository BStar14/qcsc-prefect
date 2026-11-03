"""Create Prefect blocks and variables required for the SQD workflow."""

from __future__ import annotations

import argparse
import json
import os
import tomllib
from pathlib import Path
from typing import Any

from prefect.variables import Variable
from qcsc_prefect.integrations.qiskit.qrmi import QrmiRuntimeBlock
from qcsc_prefect_dice import create_dice_blocks

from sqd_dice.main import DEFAULT_RUNTIME_BLOCK_NAME

# QRMI settings that are secrets: read from the environment, never from the config file.
SECRET_RESOURCE_KEYS = ("QRMI_IBM_QCS_IAM_APIKEY",)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Prefect blocks for the SQD workflow (Miyabi or Fugaku)."
    )
    parser.add_argument("--config", type=Path, help="Path to TOML or JSON config file.")

    parser.add_argument("--hpc-target", choices=["miyabi", "fugaku"])
    parser.add_argument("--project")
    parser.add_argument("--group")
    parser.add_argument("--queue")
    parser.add_argument("--work-dir")
    parser.add_argument("--dice-executable")

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
    parser.add_argument("--fugaku-pjm-resources", nargs="*")

    parser.add_argument("--script-filename")
    parser.add_argument("--metrics-artifact-key")

    parser.add_argument("--command-block-name")
    parser.add_argument("--execution-profile-block-name")
    parser.add_argument("--hpc-profile-block-name")
    parser.add_argument("--solver-block-name")
    parser.add_argument("--options-variable-name")
    parser.add_argument("--runtime-block-name")

    parser.add_argument("--select-cutoff", type=float)
    parser.add_argument("--davidson-tol", type=float)
    parser.add_argument("--energy-tol", type=float)
    parser.add_argument("--max-iter", type=int)
    parser.add_argument(
        "--return-sci-state",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Whether to reconstruct the SCIState from dets.bin.",
    )
    parser.add_argument("--shots", type=int)
    parser.add_argument(
        "--sampler-options-json",
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


def _require(name: str, value: Any) -> Any:
    if value is None or (isinstance(value, str) and not value.strip()):
        raise RuntimeError(f"Missing required setting: {name}")
    return value


def _default_names(*, hpc_target: str) -> dict[str, str]:
    if hpc_target == "miyabi":
        return {
            "command_block_name": "cmd-sqd-dice-solver",
            "execution_profile_block_name": "exec-sqd-dice-mpi",
            "hpc_profile_block_name": "hpc-miyabi-sqd-dice",
            "solver_block_name": "sqd-solver",
            "script_filename": "dice_solver.pbs",
            "metrics_artifact_key": "miyabi-sqd-dice-metrics",
        }
    return {
        "command_block_name": "cmd-sqd-dice-solver",
        "execution_profile_block_name": "exec-sqd-dice-fugaku",
        "hpc_profile_block_name": "hpc-fugaku-sqd-dice",
        "solver_block_name": "sqd-solver",
        "script_filename": "dice_solver.pjm",
        "metrics_artifact_key": "fugaku-sqd-dice-metrics",
    }


def _default_mpi_options(*, hpc_target: str, mpiprocs: int) -> list[str]:
    if hpc_target == "miyabi":
        return ["-np", str(mpiprocs)]
    return ["-n", str(mpiprocs)]


def _set_variable(variable_name: str, value: Any) -> None:
    Variable.set(variable_name, value, overwrite=True)


def _runtime_block(runtime: dict[str, Any]) -> QrmiRuntimeBlock:
    """Build the QrmiRuntimeBlock described by the ``[runtime]`` config table.

    Other keys of the table are QrmiRuntimeBlock fields. The SQD flow sets the
    transpiler and its configuration itself, so they are not accepted here.
    """
    settings = {"scheduler": "iqp", **runtime}
    qrmi = {str(key): str(value) for key, value in settings.pop("qrmi", {}).items()}
    unsupported = sorted(settings.keys() & {"transpiler", "transpile_config", "resource_config"})
    if unsupported:
        raise RuntimeError(
            f"[runtime] must not set {', '.join(unsupported)}; the SQD flow sets the "
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


def main() -> None:
    args = _parse_args()
    config = _load_config_file(args.config)

    hpc_target = (
        str(_pick_value(args.hpc_target, config.get("hpc_target"), "miyabi")).strip().lower()
    )
    if hpc_target not in {"miyabi", "fugaku"}:
        raise RuntimeError("'hpc_target' must be either 'miyabi' or 'fugaku'.")
    defaults = _default_names(hpc_target=hpc_target)

    project = _require(
        "project/group",
        _pick_value(args.project, config.get("project"), args.group, config.get("group")),
    )
    queue = _require("queue", _pick_value(args.queue, config.get("queue")))
    work_dir = _require("work_dir", _pick_value(args.work_dir, config.get("work_dir")))
    dice_executable = _require(
        "dice_executable",
        _pick_value(args.dice_executable, config.get("dice_executable")),
    )

    mpiprocs = int(
        _pick_value(args.mpiprocs, config.get("mpiprocs"), 4 if hpc_target == "miyabi" else 2)
    )
    options_variable_name = str(
        _pick_value(
            args.options_variable_name, config.get("options_variable_name"), "sampler_options"
        )
    ).strip()

    block_names = create_dice_blocks(
        hpc_target=hpc_target,
        project=str(project).strip(),
        queue=str(queue).strip(),
        root_dir=str(Path(str(work_dir)).expanduser().resolve()),
        dice_executable=str(Path(str(dice_executable)).expanduser().resolve()),
        command_block_name=str(
            _pick_value(
                args.command_block_name,
                config.get("command_block_name"),
                defaults["command_block_name"],
            )
        ).strip(),
        execution_profile_block_name=str(
            _pick_value(
                args.execution_profile_block_name,
                config.get("execution_profile_block_name"),
                defaults["execution_profile_block_name"],
            )
        ).strip(),
        hpc_profile_block_name=str(
            _pick_value(
                args.hpc_profile_block_name,
                config.get("hpc_profile_block_name"),
                defaults["hpc_profile_block_name"],
            )
        ).strip(),
        solver_block_name=str(
            _pick_value(
                args.solver_block_name,
                config.get("solver_block_name"),
                defaults["solver_block_name"],
            )
        ).strip(),
        command_name="sqd-dice",
        executable_key="sqd_dice_solver",
        profile_name="sqd-dice-mpi",
        launcher=_pick_value(args.launcher, config.get("launcher")),
        num_nodes=int(_pick_value(args.num_nodes, config.get("num_nodes"), 1)),
        mpiprocs=mpiprocs,
        ompthreads=_pick_value(args.ompthreads, config.get("ompthreads")),
        walltime=str(_pick_value(args.walltime, config.get("walltime"), "01:00:00")).strip(),
        modules=_normalize_str_list(_pick_value(args.modules, config.get("modules"))),
        mpi_options=_normalize_str_list(
            _pick_value(
                args.mpi_options,
                config.get("mpi_options"),
                _default_mpi_options(hpc_target=hpc_target, mpiprocs=mpiprocs),
            )
        ),
        pre_commands=_normalize_str_list(
            _pick_value(args.pre_commands, config.get("pre_commands"))
        ),
        environments=_normalize_str_dict(config.get("environments")),
        script_filename=str(
            _pick_value(
                args.script_filename, config.get("script_filename"), defaults["script_filename"]
            )
        ).strip(),
        metrics_artifact_key=str(
            _pick_value(
                args.metrics_artifact_key,
                config.get("metrics_artifact_key"),
                defaults["metrics_artifact_key"],
            )
        ).strip(),
        select_cutoff=float(_pick_value(args.select_cutoff, config.get("select_cutoff"), 5e-4)),
        davidson_tol=float(_pick_value(args.davidson_tol, config.get("davidson_tol"), 1e-5)),
        energy_tol=float(_pick_value(args.energy_tol, config.get("energy_tol"), 1e-10)),
        max_iter=int(_pick_value(args.max_iter, config.get("max_iter"), 10)),
        return_sci_state=bool(
            _pick_value(args.return_sci_state, config.get("return_sci_state"), False)
        ),
        gfscache=_pick_value(args.fugaku_gfscache, config.get("fugaku_gfscache")),
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

    sampler_options_json = _pick_value(
        args.sampler_options_json, config.get("sampler_options_json")
    )
    if sampler_options_json:
        sampler_options = json.loads(str(sampler_options_json))
    else:
        shots = int(_pick_value(args.shots, config.get("shots"), 100000))
        sampler_options = {"params": {"shots": shots}}
    _set_variable(options_variable_name, sampler_options)

    runtime_block_name = None
    if "runtime" in config:
        runtime_block_name = str(
            _pick_value(
                args.runtime_block_name,
                config.get("runtime_block_name"),
                DEFAULT_RUNTIME_BLOCK_NAME,
            )
        ).strip()
        _runtime_block(config["runtime"]).save(runtime_block_name, overwrite=True)

    print(f"Saved blocks and variable for SQD workflow (target={hpc_target})")
    print(f"  SQD solver block: {block_names['solver_block_name']}")
    print(f"  Command block: {block_names['command_block_name']}")
    print(f"  Execution profile block: {block_names['execution_profile_block_name']}")
    print(f"  HPC profile block: {block_names['hpc_profile_block_name']}")
    print(f"  Options variable: {options_variable_name}")
    if runtime_block_name:
        print(f"  QRMI runtime block: {runtime_block_name}")
    print(f"  Work directory: {Path(str(work_dir)).expanduser().resolve()}")
    print(f"  DICE executable: {Path(str(dice_executable)).expanduser().resolve()}")


if __name__ == "__main__":
    main()
