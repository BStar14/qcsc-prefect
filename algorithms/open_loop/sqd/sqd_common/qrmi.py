"""SQD-owned QPY preparation and the SQD hooks of a QrmiRuntime.

These helpers run on the workflow host. The compilation hook is
``sqd_common.transpile:transpile_for_target``. It runs in the workflow process
for ``scheduler="iqp"`` and on the compute node for ``scheduler="slurm"``, and
imports neither ffsim nor Prefect.
"""

from __future__ import annotations

import dataclasses
from functools import partial
from pathlib import Path

from ffsim.qiskit import PRE_INIT
from qcsc_prefect.integrations.qiskit.qrmi import PreparedCircuit, QrmiRuntime
from qiskit import qpy
from qiskit.primitives import PrimitiveResult

from sqd_common.transpile import isa_gate_depth

TRANSPILER = "sqd_common.transpile:transpile_for_target"


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
        f"Instruction counts = {dict(isa_circuit.count_ops())}\n"
        f"Compiled on the compute node: {path}"
    )
