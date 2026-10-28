"""Create Prefect blocks and variables required for the SQD fulqrum workflow.

An optional ``[qrmi]`` table (Slurm only) also creates the three blocks of the
QRMI sampler job and prints the ``qrmi_runtime_config`` to pass on each run.
"""

from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path
from typing import Any

HPC_TARGETS = ("slurm", "local", "miyabi", "fugaku")
QRMI_WORKER_ARGS = ["-m", "qcsc_prefect.integrations.qiskit.qrmi.worker"]
QRMI_KEYS = frozenset(
    {
        "resource",
        "python_executable",
        "work_dir",
        "queue",
        "project",
        "walltime",
        "ompthreads",
        "slurm_memory",
        "modules",
        "pre_commands",
        "environments",
        "extra_directives",
        "timeout_seconds",
        "watch_poll_interval",
        "command_block_name",
        "execution_profile_block_name",
        "hpc_profile_block_name",
    }
)


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
    parser.add_argument("--options-variable-name")

    parser.add_argument("--eigensolver", choices=["scipy", "primme"])
    parser.add_argument("--hamiltonian-format", choices=["csr", "matrix_free"])
    parser.add_argument("--tol", type=float)
    parser.add_argument("--max-iter", type=int)
    parser.add_argument("--dense-max-dim", type=int)
    parser.add_argument(
        "--return-sci-state",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Whether to return the ground-state amplitudes as an SCIState.",
    )
    parser.add_argument("--shots", type=int)
    parser.add_argument(
        "--sampler-options-json",
        help='Raw JSON for Prefect variable value (e.g. \'{"params": {"shots": 50000}}\').',
    )
    parser.add_argument(
        "--qrmi-resource",
        help="QRMI resource ID; creates the QRMI sampler blocks (Slurm only).",
    )
    parser.add_argument(
        "--qrmi-python-executable",
        help="Compute-node Python that runs the QRMI worker.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved settings as JSON without saving blocks or variables.",
    )
    return parser.parse_args(argv)


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


def _optional_str(value: Any) -> str | None:
    if value is None or not str(value).strip():
        return None
    return str(value).strip()


def _default_names(*, hpc_target: str) -> dict[str, str]:
    suffix = {"local": "", "slurm": ".slurm", "miyabi": ".pbs", "fugaku": ".pjm"}[hpc_target]
    return {
        "command_block_name": "cmd-sqd-fulqrum-solver",
        "execution_profile_block_name": f"exec-sqd-fulqrum-{hpc_target}",
        "hpc_profile_block_name": f"hpc-{hpc_target}-sqd-fulqrum",
        "solver_block_name": "sqd-fulqrum-solver",
        "script_filename": f"fulqrum_solver{suffix}",
        "metrics_artifact_key": f"{hpc_target}-sqd-fulqrum-metrics",
    }


def resolve_settings(
    args: argparse.Namespace, config: dict[str, Any]
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    """Merge CLI arguments over the config file.

    Returns:
        ``(block_kwargs, options_variable_name, sampler_options)``, where
        ``block_kwargs`` are the keyword arguments of ``create_fulqrum_blocks``.
    """

    def pick(name: str, default: Any = None) -> Any:
        return _pick_value(getattr(args, name, None), config.get(name), default)

    hpc_target = str(pick("hpc_target", "slurm")).strip().lower()
    if hpc_target not in HPC_TARGETS:
        raise RuntimeError(f"'hpc_target' must be one of {', '.join(HPC_TARGETS)}.")
    defaults = _default_names(hpc_target=hpc_target)

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
        launcher=str(pick("launcher", "single")).strip(),
        num_nodes=int(pick("num_nodes", 1)),
        mpiprocs=int(pick("mpiprocs", 1)),
        ompthreads=None if ompthreads is None else int(ompthreads),
        walltime=str(pick("walltime", "01:00:00")).strip(),
        resource_class=str(pick("resource_class", "cpu")).strip(),
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

    options_variable_name = str(pick("options_variable_name", "sampler_options")).strip()
    sampler_options_json = pick("sampler_options_json")
    if sampler_options_json:
        sampler_options = json.loads(str(sampler_options_json))
    else:
        sampler_options = {"params": {"shots": int(pick("shots", 100000))}}
    return block_kwargs, options_variable_name, sampler_options


def resolve_qrmi_settings(
    args: argparse.Namespace, config: dict[str, Any], block_kwargs: dict[str, Any]
) -> dict[str, Any] | None:
    """Resolve the optional ``[qrmi]`` table, or None when QRMI is not configured.

    Unset queue, project, work_dir, modules, pre_commands and environments
    inherit the solver settings. The QRMI work directory defaults to a ``qrmi``
    subdirectory of the solver's, and the job holds the QPU while it compiles,
    so ``ompthreads`` defaults to the solver's thread count.
    """
    table = config.get("qrmi")
    if table is None:
        table = {}
    if not isinstance(table, dict):
        raise RuntimeError("'qrmi' must be a table.")
    unknown = set(table) - QRMI_KEYS
    if unknown:
        raise RuntimeError(f"Unknown [qrmi] settings: {', '.join(sorted(unknown))}")
    resource = _optional_str(_pick_value(args.qrmi_resource, table.get("resource")))
    if resource is None:
        if table or args.qrmi_python_executable:
            raise RuntimeError("Missing required setting: qrmi.resource")
        return None
    if any(char.isspace() for char in resource):
        raise RuntimeError("'qrmi.resource' must not contain whitespace.")
    if block_kwargs["hpc_target"] != "slurm":
        raise RuntimeError("QRMI sampling needs hpc_target = 'slurm'.")
    python_executable = _require(
        "qrmi.python_executable",
        _pick_value(args.qrmi_python_executable, table.get("python_executable")),
    )

    def pick(name: str, default: Any = None) -> Any:
        return _pick_value(table.get(name), default)

    ompthreads = pick("ompthreads", block_kwargs["ompthreads"])
    environments = _normalize_str_dict(table.get("environments"))
    if environments is None:
        environments = block_kwargs["environments"]
    timeout_seconds = pick("timeout_seconds")
    watch_poll_interval = pick("watch_poll_interval")
    return dict(
        resource=resource,
        # Not resolved: the path must be valid on the compute node.
        python_executable=str(Path(str(python_executable).strip()).expanduser()),
        root_dir=str(
            Path(str(pick("work_dir", Path(block_kwargs["root_dir"]) / "qrmi"))).expanduser()
        ),
        queue=str(pick("queue", block_kwargs["queue"])).strip(),
        project=str(pick("project", block_kwargs["project"])).strip(),
        walltime=str(pick("walltime", "00:30:00")).strip(),
        ompthreads=None if ompthreads is None else int(ompthreads),
        slurm_memory=_optional_str(pick("slurm_memory", block_kwargs["slurm_memory"])),
        modules=_normalize_str_list(pick("modules", block_kwargs["modules"])),
        pre_commands=_normalize_str_list(pick("pre_commands", block_kwargs["pre_commands"])),
        environments=environments,
        extra_directives=_normalize_str_list(pick("extra_directives")),
        timeout_seconds=None if timeout_seconds is None else float(timeout_seconds),
        watch_poll_interval=None if watch_poll_interval is None else float(watch_poll_interval),
        command_block_name=str(pick("command_block_name", "cmd-sqd-fulqrum-qrmi")).strip(),
        execution_profile_block_name=str(
            pick("execution_profile_block_name", "exec-sqd-fulqrum-qrmi")
        ).strip(),
        hpc_profile_block_name=str(
            pick("hpc_profile_block_name", "hpc-slurm-sqd-fulqrum-qrmi")
        ).strip(),
    )


def qrmi_runtime_config(qrmi: dict[str, Any]) -> dict[str, Any]:
    """The ``qrmi_runtime_config`` flow argument that selects these blocks."""
    runtime = {
        "root_dir": qrmi["root_dir"],
        "command_block_name": qrmi["command_block_name"],
        "execution_profile_block_name": qrmi["execution_profile_block_name"],
        "hpc_profile_block_name": qrmi["hpc_profile_block_name"],
        "resource": qrmi["resource"],
    }
    for name in ("timeout_seconds", "watch_poll_interval"):
        if qrmi[name] is not None:
            runtime[name] = qrmi[name]
    return runtime


def create_qrmi_blocks(qrmi: dict[str, Any]) -> None:
    """Save the QRMI sampler job's Command, ExecutionProfile and HPCProfile blocks.

    These blocks are separate from the solver's: the HPC profile requests the
    QPU (``slurm_qpu``), which a fulqrum solve must not hold.
    """
    from qcsc_prefect_blocks.common.blocks import (
        CommandBlock,
        ExecutionProfileBlock,
        HPCProfileBlock,
    )

    environments = dict(qrmi["environments"] or {})
    if qrmi["ompthreads"] is not None and "OMP_NUM_THREADS" not in environments:
        environments["OMP_NUM_THREADS"] = str(qrmi["ompthreads"])
    CommandBlock(
        command_name="sqd-fulqrum-qrmi",
        executable_key="qrmi_python",
        description="QRMI sampler worker (python -m qcsc_prefect.integrations.qiskit.qrmi.worker)",
        default_args=list(QRMI_WORKER_ARGS),
    ).save(qrmi["command_block_name"], overwrite=True)
    ExecutionProfileBlock(
        profile_name="sqd-fulqrum-qrmi",
        command_name="sqd-fulqrum-qrmi",
        num_nodes=1,
        mpiprocs=1,
        ompthreads=qrmi["ompthreads"],
        walltime=qrmi["walltime"],
        launcher="single",
        modules=list(qrmi["modules"] or []),
        pre_commands=list(qrmi["pre_commands"] or []),
        environments=environments,
    ).save(qrmi["execution_profile_block_name"], overwrite=True)
    HPCProfileBlock(
        hpc_target="slurm",
        queue_cpu=qrmi["queue"],
        queue_gpu=qrmi["queue"],
        project_cpu=qrmi["project"],
        project_gpu=qrmi["project"],
        executable_map={"qrmi_python": qrmi["python_executable"]},
        slurm_qpu=qrmi["resource"],
        slurm_ntasks=1,
        slurm_memory=qrmi["slurm_memory"],
        slurm_extra_directives=list(qrmi["extra_directives"] or []),
    ).save(qrmi["hpc_profile_block_name"], overwrite=True)


def _set_variable(variable_name: str, value: Any) -> None:
    from prefect.variables import Variable

    Variable.set(variable_name, value, overwrite=True)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    config = _load_config_file(args.config)
    block_kwargs, options_variable_name, sampler_options = resolve_settings(args, config)
    qrmi = resolve_qrmi_settings(args, config, block_kwargs)

    if args.dry_run:
        resolved = {
            "blocks": block_kwargs,
            "options_variable_name": options_variable_name,
            "sampler_options": sampler_options,
        }
        if qrmi is not None:
            resolved["qrmi_blocks"] = qrmi
            resolved["qrmi_runtime_config"] = qrmi_runtime_config(qrmi)
        print(json.dumps(resolved, indent=2))
        return

    from qcsc_prefect_fulqrum import create_fulqrum_blocks

    block_names = create_fulqrum_blocks(**block_kwargs)
    if qrmi is not None:
        create_qrmi_blocks(qrmi)
    _set_variable(options_variable_name, sampler_options)

    print(
        f"Saved blocks and variable for SQD fulqrum workflow (target={block_kwargs['hpc_target']})"
    )
    print(f"  SQD solver block: {block_names['solver_block_name']}")
    print(f"  Command block: {block_names['command_block_name']}")
    print(f"  Execution profile block: {block_names['execution_profile_block_name']}")
    print(f"  HPC profile block: {block_names['hpc_profile_block_name']}")
    print(f"  Options variable: {options_variable_name}")
    print(f"  Work directory: {block_kwargs['root_dir']}")
    print(f"  Fulqrum executable: {block_kwargs['fulqrum_executable']}")
    if qrmi is not None:
        print(f"  QRMI command block: {qrmi['command_block_name']}")
        print(f"  QRMI execution profile block: {qrmi['execution_profile_block_name']}")
        print(f"  QRMI HPC profile block: {qrmi['hpc_profile_block_name']}")
        print(f"  QRMI resource (slurm_qpu): {qrmi['resource']}")
        print("Pass this as the flow argument qrmi_runtime_config to sample through QRMI:")
        print(json.dumps(qrmi_runtime_config(qrmi), indent=2))


if __name__ == "__main__":
    main()
