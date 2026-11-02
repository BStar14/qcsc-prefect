"""Workflow-side sampler facade: Slurm through the block executor, or IQP directly."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
from uuid import uuid4

from prefect import task
from prefect.cache_policies import NO_CACHE
from prefect.logging import get_run_logger

from qiskit import QuantumCircuit, qpy
from qiskit.primitives import PrimitiveResult

from .contract import SCHEMA_VERSION, circuit_registers, decode_result, resolve_shots
from .transpilation import preset_pass_manager_options

DEFAULT_TRANSPILER = "qcsc_prefect.integrations.qiskit.qrmi.worker:generic_transpile"
SLURM_FIELDS = ("root_dir", "command_block_name", "execution_profile_block_name")
SLURM_FIELDS += ("hpc_profile_block_name",)
# Without acquire(), QRMI submits jobs outside a session; these keys would attach one.
SESSION_KEYS = frozenset({"QRMI_IBM_QCS_SESSION_ID", "QRMI_JOB_ACQUISITION_TOKEN"})


async def run_job_from_blocks(**kwargs):
    # Optional dependency: native Qiskit users need not install the HPC executor.
    from qcsc_prefect_executor.from_blocks import run_job_from_blocks as execute

    return await execute(**kwargs)


@dataclass
class PreparedCircuit:
    """Algorithm-owned QPY-safe circuit and JSON config for its node transpiler."""

    circuit: QuantumCircuit
    config: dict = field(default_factory=dict)


@dataclass
class QrmiRuntime:
    """Sample one or more bound, measured circuits through QRMI in one job.

    Both schedulers connect to IBM Quantum through QRMI; they differ in where
    the code runs:

    - ``scheduler="slurm"`` (default) submits a Slurm job. The compute node
      acquires the Target, transpiles and runs ``task_runner`` inside the
      Batch that SPANK acquired. Needs ``root_dir`` and the three block names.
    - ``scheduler="iqp"`` runs in the workflow process. It builds a QRMI
      ``ibm-quantum-compute-service`` resource from ``resource_config`` (QRMI
      configuration keys without the ``<resource>_`` prefix), transpiles
      against its Target and submits one job without a session.

    ``sampler`` accepts ``[(circuit,), (circuit,), ...]``; result ``i``
    corresponds to PUB ``i``. All PUBs share the same shots.

    ``prepare`` runs on a copy of each circuit before compilation.
    ``transpiler`` is an importable ``module:function(circuit, target, config)``
    called once per circuit, on the compute node for Slurm and in the workflow
    process for IQP. For Slurm, both hosts must have that module installed and
    no target is acquired on the workflow host. This facade is not a drop-in
    ``prefect_qiskit.QuantumRuntime`` Block; see ``QrmiRuntimeBlock``.

    ``transpile_config`` is the base configuration for every PUB, and each
    ``PreparedCircuit.config`` overrides matching keys for its own circuit.
    The default transpiler accepts JSON-safe preset pass manager options,
    including ``initial_layout`` (physical qubit indices), ``optimization_level``
    (default 1), and ``seed_transpiler``. Each PUB's final configuration is
    validated against its prepared circuit. Custom pass managers can be
    constructed and run inside a ``transpiler`` hook.
    """

    root_dir: Path | None = None
    command_block_name: str | None = None
    execution_profile_block_name: str | None = None
    hpc_profile_block_name: str | None = None
    resource: str = ""
    prepare: Callable[[QuantumCircuit], PreparedCircuit] | None = None
    transpiler: str = DEFAULT_TRANSPILER
    transpile_config: dict = field(default_factory=dict)
    watch_poll_interval: float = 2.0
    timeout_seconds: float | None = None
    scheduler: Literal["slurm", "iqp"] = field(default="slurm", kw_only=True)
    # Credentials: kept out of repr so that logs and task parameters do not show them.
    resource_config: Mapping[str, str] | None = field(default=None, kw_only=True, repr=False)

    def __post_init__(self) -> None:
        self._validate_scheduler()

    def _validate_scheduler(self) -> None:
        if self.scheduler == "slurm":
            missing = [name for name in SLURM_FIELDS if not getattr(self, name)]
            if missing:
                raise ValueError(f'scheduler="slurm" requires {", ".join(missing)}')
            if self.resource_config is not None:
                raise ValueError(
                    'resource_config is only used by scheduler="iqp"; Slurm compute nodes '
                    "read QRMI settings from their own environment"
                )
        elif self.scheduler == "iqp":
            extra = [name for name in SLURM_FIELDS if getattr(self, name)]
            if extra:
                raise ValueError(f'scheduler="iqp" does not use {", ".join(extra)}')
            if not isinstance(self.resource_config, Mapping) or not self.resource_config:
                raise ValueError('scheduler="iqp" requires a nonempty resource_config mapping')
            if any(not isinstance(v, str) for v in self.resource_config.values()):
                raise ValueError("resource_config values must be strings")
            session_keys = sorted(SESSION_KEYS & set(self.resource_config))
            if session_keys:
                raise ValueError(
                    f"resource_config must not contain {', '.join(session_keys)}; "
                    'scheduler="iqp" submits jobs without a session'
                )
        else:
            raise ValueError(
                f"Unsupported scheduler: {self.scheduler!r}; expected 'slurm' or 'iqp'"
            )

    async def sampler(
        self, sampler_pubs, *, shots: int | None = None, options: Mapping | None = None
    ) -> PrimitiveResult:
        self._validate_scheduler()
        shots = resolve_shots(shots, options)
        pubs = list(sampler_pubs)
        if not pubs or any(not isinstance(pub, (tuple, list)) or len(pub) != 1 for pub in pubs):
            raise ValueError(
                "QrmiRuntime accepts only [(circuit,), ...]; bind parameters first and "
                "use the common shots argument"
            )
        circuits = [pub[0] for pub in pubs]
        registers = [circuit_registers(circuit) for circuit in circuits]
        if not self.resource or any(char.isspace() for char in self.resource):
            raise ValueError("resource must be a nonempty QRMI resource ID without whitespace")
        if self.transpiler.count(":") != 1:
            raise ValueError("transpiler must be an importable module:function")
        prepared_circuits = []
        configs = []
        # Validate every PUB before creating a job directory or starting any task.
        for index, (circuit, pub_registers) in enumerate(zip(circuits, registers)):
            prepared = (
                self.prepare(circuit.copy()) if self.prepare else PreparedCircuit(circuit.copy())
            )
            if circuit_registers(prepared.circuit) != pub_registers:
                raise ValueError("prepare must preserve classical register names and widths")
            config = {**self.transpile_config, **prepared.config}
            json.dumps(config, allow_nan=False)
            if self.transpiler == DEFAULT_TRANSPILER:
                try:
                    preset_pass_manager_options(config, num_qubits=prepared.circuit.num_qubits)
                except ValueError as exc:
                    raise ValueError(f"PUB {index}: {exc}") from exc
            prepared_circuits.append(prepared.circuit)
            configs.append(config)
        if self.scheduler == "iqp":
            from .iqp import sample_on_iqp

            return await sample_on_iqp(prepared_circuits, shots, registers, configs, self)
        return await _sample_on_slurm(prepared_circuits, shots, registers, configs, self)


@task(name="quantum-sampling-qrmi-slurm", retries=0, cache_policy=NO_CACHE, persist_result=False)
async def _sample_on_slurm(circuits, shots, registers, configs, runtime) -> PrimitiveResult:
    work_dir = Path(runtime.root_dir).expanduser().resolve() / uuid4().hex
    work_dir.mkdir(parents=True, exist_ok=False)
    request = {
        "schema_version": SCHEMA_VERSION,
        "resource": runtime.resource,
        "shots": shots,
        "registers": registers,
        "transpiler": runtime.transpiler,
    }
    (work_dir / "sampler_request.json").write_text(json.dumps(request, indent=2), encoding="utf-8")
    (work_dir / "transpile_config.json").write_text(json.dumps(configs, indent=2), encoding="utf-8")
    with (work_dir / "circuit.qpy").open("wb") as stream:
        qpy.dump(circuits, stream)
    job = await run_job_from_blocks(
        command_block_name=runtime.command_block_name,
        execution_profile_block_name=runtime.execution_profile_block_name,
        hpc_profile_block_name=runtime.hpc_profile_block_name,
        work_dir=work_dir,
        script_filename="sampler.slurm",
        user_args=["--request", "sampler_request.json"],
        watch_poll_interval=runtime.watch_poll_interval,
        timeout_seconds=runtime.timeout_seconds,
        metrics_artifact_key="qrmi-slurm-job",
    )
    record = {
        "job_id": job.job_id,
        "state": job.state,
        "exit_status": job.exit_status,
        "work_dir": str(work_dir),
        "resource": runtime.resource,
        "mode": "batch",
        "transpiler": runtime.transpiler,
    }
    (work_dir / "slurm_job.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    if job.state != "COMPLETED" or job.exit_status != 0:
        raise RuntimeError(f"QRMI Slurm job {job.job_id} failed; see {work_dir}")
    result_path = work_dir / "result.json"
    if not result_path.is_file():
        raise RuntimeError(f"QRMI produced no result; see {work_dir}/output.err")
    result = decode_result(result_path, shots, registers)
    result.metadata["qcsc_slurm"] = record
    get_run_logger().info("Quantum Slurm job %s completed: %s", job.job_id, work_dir)
    return result
