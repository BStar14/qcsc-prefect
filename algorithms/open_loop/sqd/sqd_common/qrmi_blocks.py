"""The QRMI sampler blocks of the SQD workflow, shared by the solvers.

``sqd-create-blocks`` saves them from the ``[qrmi]`` table for a solver on Slurm
(fulqrum and SBD): a command, an execution profile and an HPC profile for the QRMI
job in Slurm, and the ``qrmi-runtime`` ``QrmiRuntimeBlock`` that uses them. The DICE
blocks target Miyabi or Fugaku, so DICE takes ``[runtime]`` only. ``[runtime]``
saves another ``QrmiRuntimeBlock`` instead, for example for IBM Quantum (IQP).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from sqd_common.block_names import QRMI_BLOCK_NAMES
from sqd_common.runtime_block import runtime_block_settings

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
        "instance",
        "service_crn",
        "endpoint",
        "iam_endpoint",
    }
)
# The QRMI job's settings that its QrmiRuntimeBlock holds, not its job blocks.
QRMI_RUNTIME_KEYS = ("instance", "service_crn", "endpoint", "iam_endpoint")


def add_qrmi_arguments(parser: argparse.ArgumentParser) -> None:
    """The ``--qrmi-*`` options of a ``create_blocks`` script."""
    parser.add_argument(
        "--qrmi-resource",
        help="QPU backend; default: the one qcsc-qrmi login chose (Slurm only).",
    )
    parser.add_argument(
        "--qrmi-python-executable",
        help="Compute-node Python that runs the QRMI worker.",
    )


def _pick_value(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _optional_str(value: Any) -> str | None:
    if value is None or not str(value).strip():
        return None
    return str(value).strip()


def _str_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",") if item.strip()]
    elif isinstance(value, list):
        items = [str(item).strip() for item in value if str(item).strip()]
    else:
        raise ValueError(f"Expected list[str] or CSV string, got: {type(value)}")
    return items or None


def _str_dict(value: Any) -> dict[str, str] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"Expected dict[str, str], got: {type(value)}")
    items = {str(k).strip(): str(v).strip() for k, v in value.items() if str(k).strip()}
    return items or None


def _require(name: str, value: Any) -> Any:
    if value is None or (isinstance(value, str) and not value.strip()):
        raise RuntimeError(f"Missing required setting: {name}")
    return value


def resolve_qrmi_settings(
    args: argparse.Namespace, config: dict[str, Any], solver: dict[str, Any]
) -> dict[str, Any] | None:
    """Resolve the optional ``[qrmi]`` table, or None when QRMI is not configured.

    ``solver`` holds the solver blocks' settings: ``hpc_target``, ``queue``,
    ``project``, ``root_dir``, ``ompthreads``, ``modules``, ``pre_commands``,
    ``environments`` and, optionally, ``slurm_memory``. Unset queue, project,
    work_dir, modules, pre_commands and environments inherit them. The QRMI work
    directory defaults to a ``qrmi`` subdirectory of the solver's, and the job holds
    the QPU while it compiles, so ``ompthreads`` defaults to the solver's.
    """
    table = config.get("qrmi")
    if table is None:
        table = {}
    if not isinstance(table, dict):
        raise RuntimeError("'qrmi' must be a table.")
    if "api_key" in table:
        raise RuntimeError(
            "Do not write api_key in [qrmi]: Slurm compute nodes read the API key from "
            "qrmi_config.json, and the job environment would store it."
        )
    unknown = set(table) - QRMI_KEYS
    if unknown:
        raise RuntimeError(f"Unknown [qrmi] settings: {', '.join(sorted(unknown))}")
    resource = _optional_str(
        _pick_value(getattr(args, "qrmi_resource", None), table.get("resource"))
    )
    if resource is None and not table and not getattr(args, "qrmi_python_executable", None):
        return None
    # Without resource, the backend that qcsc-qrmi login saved in the block is kept.
    if resource is not None and any(char.isspace() for char in resource):
        raise RuntimeError("'qrmi.resource' must not contain whitespace.")
    if solver["hpc_target"] != "slurm":
        raise RuntimeError("QRMI sampling needs hpc_target = 'slurm'.")
    python_executable = _require(
        "qrmi.python_executable",
        _pick_value(getattr(args, "qrmi_python_executable", None), table.get("python_executable")),
    )

    def pick(name: str, default: Any = None) -> Any:
        return _pick_value(table.get(name), default)

    ompthreads = pick("ompthreads", solver["ompthreads"])
    environments = _str_dict(table.get("environments"))
    if environments is None:
        environments = solver["environments"]
    timeout_seconds = pick("timeout_seconds")
    watch_poll_interval = pick("watch_poll_interval")
    return dict(
        resource=resource,
        # Not resolved: the path must be valid on the compute node.
        python_executable=str(Path(str(python_executable).strip()).expanduser()),
        root_dir=str(Path(str(pick("work_dir", Path(solver["root_dir"]) / "qrmi"))).expanduser()),
        queue=str(pick("queue", solver["queue"])).strip(),
        project=str(pick("project", solver["project"])).strip(),
        walltime=str(pick("walltime", "00:30:00")).strip(),
        ompthreads=None if ompthreads is None else int(ompthreads),
        slurm_memory=_optional_str(pick("slurm_memory", solver.get("slurm_memory"))),
        modules=_str_list(pick("modules", solver["modules"])),
        pre_commands=_str_list(pick("pre_commands", solver["pre_commands"])),
        environments=environments,
        extra_directives=_str_list(pick("extra_directives")),
        timeout_seconds=None if timeout_seconds is None else float(timeout_seconds),
        watch_poll_interval=None if watch_poll_interval is None else float(watch_poll_interval),
        **{name: str(pick(name, default)).strip() for name, default in QRMI_BLOCK_NAMES.items()},
        **{name: _optional_str(table.get(name)) for name in QRMI_RUNTIME_KEYS},
    )


def qrmi_runtime_settings(qrmi: dict[str, Any]) -> dict[str, Any]:
    """QrmiRuntimeBlock fields of a ``scheduler="slurm"`` runtime that uses these blocks.

    ``resource`` and ``instance`` are included only when ``[qrmi]`` sets them, so the
    choices ``qcsc-qrmi login`` saved in the block are kept otherwise.
    """
    runtime = {
        "scheduler": "slurm",
        "root_dir": qrmi["root_dir"],
        "command_block_name": qrmi["command_block_name"],
        "execution_profile_block_name": qrmi["execution_profile_block_name"],
        "hpc_profile_block_name": qrmi["hpc_profile_block_name"],
    }
    for name in ("resource", "timeout_seconds", "watch_poll_interval", *QRMI_RUNTIME_KEYS):
        if qrmi.get(name) is not None:
            runtime[name] = qrmi[name]
    return runtime


def create_qrmi_blocks(qrmi: dict[str, Any]) -> None:
    """Save the QRMI sampler job's Command, ExecutionProfile and HPCProfile blocks.

    They are shared by every solver and separate from the solver's blocks: the QRMI
    job holds the QPU, which a solver job must not. The job requests the backend of
    the QRMI Runtime block as ``--qpu``, so the HPC profile names none.
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
        command_name="sqd-qrmi",
        executable_key="qrmi_python",
        description="QRMI sampler worker (python -m qcsc_prefect.integrations.qiskit.qrmi.worker)",
        default_args=list(QRMI_WORKER_ARGS),
    ).save(qrmi["command_block_name"], overwrite=True)
    ExecutionProfileBlock(
        profile_name="sqd-qrmi",
        command_name="sqd-qrmi",
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
        slurm_ntasks=1,
        slurm_memory=qrmi["slurm_memory"],
        slurm_extra_directives=list(qrmi["extra_directives"] or []),
    ).save(qrmi["hpc_profile_block_name"], overwrite=True)


def resolve_runtime_settings(
    config: dict[str, Any], qrmi: dict[str, Any] | None
) -> dict[str, Any] | None:
    """QrmiRuntimeBlock fields from ``[qrmi]`` or ``[runtime]``, or None without either."""
    if qrmi is not None and "runtime" in config:
        raise RuntimeError(
            "Set either [qrmi] (a Slurm QRMI job) or [runtime] (any QrmiRuntimeBlock), not both."
        )
    if qrmi is not None:
        return qrmi_runtime_settings(qrmi)
    if "runtime" in config:
        return runtime_block_settings(config["runtime"])
    return None
