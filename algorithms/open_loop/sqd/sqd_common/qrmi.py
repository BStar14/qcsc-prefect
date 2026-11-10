"""SQD-owned QPY preparation and the SQD hooks of a QrmiRuntime.

These helpers run on the workflow host. The compilation hook is
``sqd_common.transpile:transpile_for_target``. It runs in the workflow process
for ``scheduler="iqp"`` and on the compute node for ``scheduler="slurm"``, and
imports neither ffsim nor Prefect.
"""

from __future__ import annotations

import dataclasses
import re
from functools import partial
from pathlib import Path

from ffsim.qiskit import PRE_INIT
from qcsc_prefect.integrations.qiskit.qrmi import PreparedCircuit, QrmiRuntime
from qiskit import qpy
from qiskit.primitives import PrimitiveResult

from sqd_common.transpile import isa_gate_depth, two_qubit_depth

TRANSPILER = "sqd_common.transpile:transpile_for_target"
# task_runner prints the IBM Quantum job ID in the QRMI job's output.out.
_TASK_ID = re.compile(r"Task ID: (\S+)")


def prepare_sampler_circuit(circuit, *, transpile_config: dict) -> PreparedCircuit:
    """Run ffSIM passes while Python gate attributes exist, then expand for QPY."""
    circuit = PRE_INIT.run(circuit)
    for _ in range(10):
        gates = {
            inst.operation.name
            for inst in circuit.data
            if (type(inst.operation).__module__ or "").startswith("ffsim.")
        }
        if not gates:
            return PreparedCircuit(circuit, dict(transpile_config))
        circuit = circuit.decompose(gates_to_decompose=sorted(gates))
    raise RuntimeError("ffsim gates remain after ten decomposition rounds")


def make_qrmi_runtime(block_runtime: QrmiRuntime, transpile_config: dict) -> QrmiRuntime:
    """Replace the hooks of a ``QrmiRuntimeBlock`` runtime with the SQD ones.

    The block holds where to sample (scheduler, instance, backend, Slurm blocks or
    IQP settings). The preparation, the compilation hook and its configuration come
    from the workflow's settings, so the block's ``transpiler`` and
    ``transpile_config`` are replaced.
    """
    return dataclasses.replace(
        block_runtime,
        prepare=partial(prepare_sampler_circuit, transpile_config=transpile_config),
        transpiler=TRANSPILER,
        transpile_config={},
    )


def compiled_circuit_summary(result: PrimitiveResult) -> str | None:
    """Describe the circuit a QRMI Slurm job compiled on the compute node.

    Reads ``isa.qpy`` from the job directory recorded in the result metadata.
    Returns None for IQP, which compiles in this process and records its report
    in the metadata, or when the file is not readable from this host.
    """
    work_dir = (result.metadata.get("qcsc_slurm") or {}).get("work_dir")
    path = Path(work_dir) / "isa.qpy" if work_dir else None
    if path is None or not path.is_file():
        return None
    with path.open("rb") as stream:
        (isa_circuit,) = qpy.load(stream)
    return (
        f"Circuit depth = {isa_gate_depth(isa_circuit)}\n"
        f"Two-qubit depth = {two_qubit_depth(isa_circuit)}\n"
        f"Instruction counts = {dict(isa_circuit.count_ops())}\n"
        f"Compiled on the compute node: {path}"
    )


def _slurm_task_id(work_dir: str | None) -> str | None:
    """The IBM Quantum job ID that task_runner printed, if the job output is readable."""
    path = Path(work_dir) / "output.out" if work_dir else None
    if path is None or not path.is_file():
        return None
    task_ids = _TASK_ID.findall(path.read_text(encoding="utf-8", errors="replace"))
    return task_ids[-1] if task_ids else None


def sampling_summary(result: PrimitiveResult, resource: str, shots: int) -> str:
    """One log line on a finished sampling job: where it ran, its IDs, shots and QPU time.

    The QPU time is the result's execution spans: the window in which the QPU took the
    data. It excludes the time the job waited in the IBM Quantum queue, and may differ
    from the usage that IBM Quantum bills.
    """
    slurm = result.metadata.get("qcsc_slurm")
    if slurm is not None:
        ids = [f"Slurm job {slurm.get('job_id')}"]
        task_id = _slurm_task_id(slurm.get("work_dir"))
    else:
        ids = []
        task_id = (result.metadata.get("qcsc_qrmi") or {}).get("job_id")
    if task_id:
        ids.append(f"IBM Quantum job {task_id}")
    where = f"{resource} ({', '.join(ids)})" if ids else resource
    lines = [f"Quantum sampling completed on {where}.", f"Shots = {shots}"]
    spans = (result.metadata.get("execution") or {}).get("execution_spans")
    if spans:
        lines.append(
            f"QPU execution = {spans.duration:.0f} s, {spans.start:%Y-%m-%d %H:%M:%S} to "
            f"{spans.stop:%H:%M:%S} UTC (execution spans; excludes the queue time and may "
            "differ from the billed usage)"
        )
    return "\n".join(lines)
