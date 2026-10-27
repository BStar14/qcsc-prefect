"""Workflow-side sampler facade using the existing block-driven executor."""

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
    """Sample one bound, measured circuit through QRMI.

    Only ``scheduler="slurm"`` is currently supported. Direct QRMI execution
    without Slurm (``scheduler="iqp"``) is not implemented.

    ``prepare`` runs before QPY serialization. ``transpiler`` is an importable
    ``module:function(circuit, target, config)`` on the compute node. Both hosts
    must have that module installed. No target is acquired on the workflow host;
    this facade is not a drop-in ``prefect_qiskit.QuantumRuntime`` Block.

    The default transpiler accepts JSON-safe preset pass manager options in
    ``transpile_config``, including ``initial_layout`` (physical qubit indices),
    ``optimization_level`` (default 1), and ``seed_transpiler``. Custom pass
    managers can be constructed and run inside a ``transpiler`` hook.
    """

    root_dir: Path
    command_block_name: str
    execution_profile_block_name: str
    hpc_profile_block_name: str
    resource: str
    prepare: Callable[[QuantumCircuit], PreparedCircuit] | None = None
    transpiler: str = DEFAULT_TRANSPILER
    transpile_config: dict = field(default_factory=dict)
    watch_poll_interval: float = 2.0
    timeout_seconds: float | None = None
    scheduler: Literal["slurm"] = field(default="slurm", kw_only=True)

    def __post_init__(self) -> None:
        self._validate_scheduler()

    def _validate_scheduler(self) -> None:
        if self.scheduler == "iqp":
            raise NotImplementedError('scheduler="iqp" is not implemented; use scheduler="slurm"')
        if self.scheduler != "slurm":
            raise ValueError(f"Unsupported scheduler: {self.scheduler!r}; expected 'slurm'")

    async def sampler(
        self, sampler_pubs, *, shots: int | None = None, options: Mapping | None = None
    ) -> PrimitiveResult:
        self._validate_scheduler()
        shots = resolve_shots(shots, options)
        pubs = list(sampler_pubs)
        if len(pubs) != 1 or not isinstance(pubs[0], (tuple, list)) or len(pubs[0]) != 1:
            raise ValueError("QRMI Slurm accepts only [(circuit,)]; bind parameters first")
        circuit = pubs[0][0]
        registers = circuit_registers(circuit)
        if not self.resource or any(char.isspace() for char in self.resource):
            raise ValueError("resource must be a nonempty QRMI resource ID without whitespace")
        if self.transpiler.count(":") != 1:
            raise ValueError("transpiler must be an importable module:function")
        prepared = self.prepare(circuit.copy()) if self.prepare else PreparedCircuit(circuit.copy())
        if circuit_registers(prepared.circuit) != registers:
            raise ValueError("prepare must preserve classical register names and widths")
        config = {**self.transpile_config, **prepared.config}
        # Validate JSON before creating a job directory or starting any task.
        json.dumps(config, allow_nan=False)
        if self.transpiler == DEFAULT_TRANSPILER:
            preset_pass_manager_options(config, num_qubits=prepared.circuit.num_qubits)
        return await _sample_on_slurm(prepared.circuit, shots, registers, config, self)


@task(name="quantum-sampling-qrmi-slurm", retries=0, cache_policy=NO_CACHE, persist_result=False)
async def _sample_on_slurm(circuit, shots, registers, config, runtime) -> PrimitiveResult:
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
    (work_dir / "transpile_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    with (work_dir / "circuit.qpy").open("wb") as stream:
        qpy.dump(circuit, stream)
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
