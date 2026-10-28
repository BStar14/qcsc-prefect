"""LUCJ compilation shared by the workflow host and the QRMI compute-node hook.

This module imports only Qiskit and qiskit-ibm-runtime. The QRMI worker loads
``transpile_for_target`` on the compute node, which therefore needs neither
ffsim nor Prefect-Qiskit nor the rest of the workflow.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.passmanager import ConditionalController
from qiskit.transpiler import (
    PassManager,
    StagedPassManager,
    Target,
    generate_preset_pass_manager,
)
from qiskit.transpiler.passes import (
    ApplyLayout,
    BarrierBeforeFinalMeasurements,
    EnlargeWithAncilla,
    FullAncillaAllocation,
    Optimize1qGatesDecomposition,
    RemoveIdentityEquivalent,
    SabreLayout,
)
from qiskit_ibm_runtime.transpiler.passes import FoldRzzAngle

TRANSPILE_OPTIONS = frozenset(
    {
        "optimization_level",
        "seed_transpiler",
        "sabre_max_iterations",
        "sabre_layout_trials",
        "sabre_swap_trials",
    }
)


def derive_seed(seed: int | np.random.Generator) -> int:
    """Draw the transpiler seed from a generator, or pass an integer through."""
    if isinstance(seed, np.random.Generator):
        return seed.integers(np.iinfo(int).max, size=1).item()
    return seed


def build_pass_manager(
    target: Target,
    *,
    optimization_level: int,
    seed_transpiler: int,
    sabre_max_iterations: int,
    sabre_layout_trials: int,
    sabre_swap_trials: int,
    pre_init: PassManager | None = None,
) -> StagedPassManager:
    """Preset pass manager with a massive SABRE layout search.

    ``pre_init`` replaces the preset init stage, for example with ffsim's
    ``PRE_INIT`` when the circuit still holds ffsim gates.
    """
    coupling_map = target.build_coupling_map()
    passmanager = generate_preset_pass_manager(
        optimization_level=optimization_level,
        seed_transpiler=seed_transpiler,
        target=target,
    )
    if pre_init is not None:
        passmanager.pre_init = pre_init
    passmanager.layout = PassManager(
        [
            BarrierBeforeFinalMeasurements(
                label="qiskit.transpiler.internal.routing.protection.barrier",
            ),
            SabreLayout(
                coupling_map=coupling_map,
                seed=seed_transpiler,
                max_iterations=sabre_max_iterations,
                layout_trials=sabre_layout_trials,
                swap_trials=sabre_swap_trials,
            ),
            ConditionalController(
                tasks=[
                    FullAncillaAllocation(coupling_map=coupling_map),
                    EnlargeWithAncilla(),
                    ApplyLayout(),
                ],
                condition=lambda propset: propset["final_layout"] is None,
            ),
        ]
    )
    if "rzz" in target.operation_names:
        passmanager.post_optimization = PassManager(
            [
                FoldRzzAngle(),
                Optimize1qGatesDecomposition(target=target),  # Cancel added local gates
                RemoveIdentityEquivalent(target=target),  # Remove GlobalPhaseGate
            ]
        )
    return passmanager


def isa_gate_depth(circuit: QuantumCircuit) -> int:
    """Depth over gates other than ``rz``, barriers and measurements."""
    return circuit.depth(lambda inst: inst.operation.name not in ("rz", "barrier", "measure"))


def transpile_for_target(
    circuit: QuantumCircuit, target: Target, config: Mapping[str, Any]
) -> QuantumCircuit:
    """QRMI compute-node hook: compile the prepared LUCJ circuit for ``target``.

    ``config`` holds exactly the ``TRANSPILE_OPTIONS`` keys. The circuit must
    already be free of ffsim gates (see ``sqd_fulqrum.qrmi.prepare_sampler_circuit``).
    """
    unknown = set(config) - TRANSPILE_OPTIONS
    missing = TRANSPILE_OPTIONS - set(config)
    if unknown or missing:
        raise ValueError(
            f"Unexpected sqd_fulqrum transpile config: unknown={sorted(unknown)}, "
            f"missing={sorted(missing)}"
        )
    return build_pass_manager(target, **config).run(circuit)
