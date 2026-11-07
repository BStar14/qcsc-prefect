"""Create the Prefect blocks of the SQD workflow with the DICE solver.

An optional ``[runtime]`` table also saves the ``QrmiRuntimeBlock`` that samples
(``sampling.runtime_block_name`` of each run, default ``qrmi-runtime``). The DICE
blocks target Miyabi or Fugaku, so ``[qrmi]``, a QRMI job in Slurm, is not used
with them. Shots are a flow parameter, not a block setting.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from qcsc_prefect_dice import create_dice_blocks
from sqd_common.advanced_blocks import advanced_block_name, advanced_settings, save_advanced_block
from sqd_common.block_config import load_block_config
from sqd_common.block_names import solver_block_names, solver_ref
from sqd_common.runtime_block import (
    build_runtime_block,
    describe,
    runtime_block_name,
    runtime_block_settings,
)


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
    parser.add_argument("--runtime-block-name")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved settings as JSON without saving blocks.",
    )

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
    return parser.parse_args()


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


def _default_names(*, hpc_target: str) -> dict[str, str]:
    names = solver_block_names("dice", hpc_target)
    if hpc_target == "miyabi":
        # Miyabi runs DICE with mpiexec.hydra; keep the profile name of that launcher.
        names["execution_profile_block_name"] = "exec-sqd-dice-miyabi"
    return names


def _default_mpi_options(*, hpc_target: str, mpiprocs: int) -> list[str]:
    if hpc_target == "miyabi":
        return ["-np", str(mpiprocs)]
    return ["-n", str(mpiprocs)]


# Run settings that belong to the flow parameters, not to the blocks.
FLOW_PARAMETER_KEYS = {
    "shots": "sampling.shots",
    "sampler_options_json": "sampling.shots",
    "options_variable_name": "sampling.shots",
}


def main() -> None:
    args = _parse_args()
    config = _load_config_file(args.config)
    for key, parameter in FLOW_PARAMETER_KEYS.items():
        if key in config:
            raise RuntimeError(
                f"'{key}' is not a block setting. Set {parameter} in the flow "
                "parameters of each run instead."
            )
    # Validate [runtime] and [sqd_advanced] before saving any block.
    runtime = runtime_block_settings(config["runtime"]) if "runtime" in config else None
    runtime_name = runtime_block_name(args.runtime_block_name, config)
    advanced_name = advanced_block_name(config)
    advanced_settings(config, "dice_shci_solver_job/check")
    if "qrmi" in config:
        raise RuntimeError(
            "[qrmi] samples through a QRMI job in Slurm, and the DICE blocks target Miyabi "
            "or Fugaku. Use [runtime] (IBM Quantum from the workflow host) with DICE."
        )

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

    block_kwargs = dict(
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
    ref = solver_ref("dice", block_kwargs["solver_block_name"])
    advanced = advanced_settings(config, ref)
    if args.dry_run:
        resolved = {
            "blocks": block_kwargs,
            "advanced_block": {"name": advanced_name, **advanced.model_dump()},
        }
        if runtime is not None:
            resolved["runtime_block"] = {"name": runtime_name, **describe(runtime)}
        print(json.dumps(resolved, indent=2, default=str))
        return

    block_names = create_dice_blocks(**block_kwargs)
    print(f"Saved blocks for SQD workflow (target={hpc_target})")
    print(f"  SQD solver block: {block_names['solver_block_name']}")
    print(f"  Command block: {block_names['command_block_name']}")
    print(f"  Execution profile block: {block_names['execution_profile_block_name']}")
    print(f"  HPC profile block: {block_names['hpc_profile_block_name']}")
    print(f"  Work directory: {Path(str(work_dir)).expanduser().resolve()}")
    print(f"  DICE executable: {Path(str(dice_executable)).expanduser().resolve()}")
    if runtime is not None:
        build_runtime_block(runtime).save(runtime_name, overwrite=True)
        print(f"  QRMI runtime block: {runtime_name} (scheduler={runtime['scheduler']})")
    save_advanced_block(advanced, advanced_name)
    print(f"  Advanced SQD Settings block: {advanced_name} (Solver Block {ref})")


if __name__ == "__main__":
    main()
