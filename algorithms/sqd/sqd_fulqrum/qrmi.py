"""SQD-owned QPY preparation and compute-node compilation hooks for QRMI."""

from __future__ import annotations

import numpy as np
from ffsim.qiskit import PRE_INIT
from prefect.logging.loggers import disable_run_logger
from qcsc_prefect.integrations.qiskit.qrmi import PreparedCircuit


def prepare_sampler_circuit(circuit, *, circuit_parameters: dict) -> PreparedCircuit:
    """Run ffSIM passes while Python gate attributes exist, then expand for QPY."""
    circuit = PRE_INIT.run(circuit)
    for _ in range(10):
        gates = {
            inst.operation.name
            for inst in circuit.data
            if (type(inst.operation).__module__ or "").startswith("ffsim.")
        }
        if not gates:
            return PreparedCircuit(circuit, {"circuit_parameters": circuit_parameters, "seed": 24})
        circuit = circuit.decompose(gates_to_decompose=sorted(gates))
    raise RuntimeError("ffsim gates remain after ten decomposition rounds")


def transpile_for_target(circuit, target, config: dict):
    """Reuse the workflow's custom transpiler with the compute-node Target."""
    from sqd_fulqrum.main import CircuitParameters, transpile_circuit

    # The QRMI worker runs outside a Prefect flow/task context. Execute the
    # existing transpiler directly without creating a nested task or server.
    with disable_run_logger():
        return transpile_circuit.fn(
            circuit=circuit,
            target=target,
            circuit_params=CircuitParameters(**config["circuit_parameters"]),
            seed=np.random.default_rng(config["seed"]),
        )
