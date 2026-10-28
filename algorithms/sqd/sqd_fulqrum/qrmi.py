"""SQD-owned QPY preparation and QrmiRuntime construction for QRMI sampling.

These helpers run on the workflow host. The compute-node compilation hook is
``sqd_fulqrum.transpile:transpile_for_target``, which does not import ffsim.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import Any

from ffsim.qiskit import PRE_INIT
from qcsc_prefect.integrations.qiskit.qrmi import PreparedCircuit, QrmiRuntime
from qiskit import qpy
from qiskit.primitives import PrimitiveResult

from sqd_fulqrum.transpile import isa_gate_depth

TRANSPILER = "sqd_fulqrum.transpile:transpile_for_target"
RESERVED_KEYS = frozenset({"prepare", "transpiler", "transpile_config"})


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


def make_qrmi_runtime(
    qrmi_runtime_config: Mapping[str, Any], transpile_config: dict
) -> QrmiRuntime:
    """Build a QrmiRuntime with the SQD preparation and compilation hooks.

    ``qrmi_runtime_config`` holds the QrmiRuntime fields chosen per run (job
    root, block names, resource, monitoring). The hooks and their configuration
    come from the workflow's circuit parameters, so they cannot be set here.
    """
    reserved = RESERVED_KEYS & set(qrmi_runtime_config)
    if reserved:
        raise ValueError(
            f"qrmi_runtime_config must not set {sorted(reserved)}: sqd_fulqrum supplies "
            "its own compilation hooks, configured by parameters.circuit."
        )
    return QrmiRuntime(
        **qrmi_runtime_config,
        prepare=partial(prepare_sampler_circuit, transpile_config=transpile_config),
        transpiler=TRANSPILER,
    )


def compiled_circuit_summary(result: PrimitiveResult) -> str | None:
    """Describe the circuit the compute node compiled, as the host path logs it.

    Reads ``isa.qpy`` from the QRMI job directory recorded in the result
    metadata. Returns None when the file is not readable from this host.
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
