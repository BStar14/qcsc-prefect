"""The SKQD transpiler hook reproduces the workflow-host transpilation it replaced."""

from __future__ import annotations

import io
import json
from types import SimpleNamespace

import ffsim
import numpy as np
import pytest
from ffsim.qiskit import PRE_INIT
from qiskit import qpy
from qiskit.passmanager import ConditionalController
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.transpiler import CouplingMap, generate_preset_pass_manager
from qiskit.transpiler.passes import (
    ApplyLayout,
    BarrierBeforeFinalMeasurements,
    EnlargeWithAncilla,
    FullAncillaAllocation,
    Optimize1qGatesDecomposition,
    RemoveIdentityEquivalent,
    SabreLayout,
)
from qiskit.transpiler.passmanager import PassManager
from qiskit_ibm_runtime.transpiler.passes import FoldRzzAngle
from skqd_dice import main
from skqd_dice import transpilation as hook

NORB = 3
NUM_QUBITS = 2 * NORB + 4
SABRE = {"sabre_max_iterations": 2, "sabre_swap_trials": 4, "sabre_layout_trials": 20}


def line_target(basis=("cz", "rz", "sx", "x")):
    # GenericBackendV2 is fully connected by default, where every layout is equally good
    # and the SABRE settings and seed have no effect. A line makes them matter.
    return GenericBackendV2(
        num_qubits=NUM_QUBITS,
        basis_gates=list(basis),
        coupling_map=CouplingMap.from_line(NUM_QUBITS),
        seed=7,
    ).target


def reference_transpile(circuits, target, circuit_params, seed):
    """``transpile_circuits`` as it was before the QrmiRuntime migration (bf820ee)."""
    if isinstance(seed, np.random.Generator):
        seed = seed.integers(np.iinfo(int).max, size=1).item()
    coupling_map = target.build_coupling_map()
    passmanager = generate_preset_pass_manager(
        optimization_level=circuit_params.optimization_level,
        seed_transpiler=seed,
        target=target,
    )
    passmanager.pre_init = PRE_INIT
    passmanager.layout = PassManager(
        [
            BarrierBeforeFinalMeasurements(
                label="qiskit.transpiler.internal.routing.protection.barrier",
            ),
            SabreLayout(
                coupling_map=coupling_map,
                seed=seed,
                max_iterations=circuit_params.sabre_max_iterations,
                layout_trials=circuit_params.sabre_layout_trials,
                swap_trials=circuit_params.sabre_swap_trials,
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
                Optimize1qGatesDecomposition(target=target),
                RemoveIdentityEquivalent(target=target),
            ]
        )
    return [passmanager.run(circuit) for circuit in circuits]


@pytest.fixture(autouse=True)
def no_run_logger(monkeypatch):
    monkeypatch.setattr(main, "get_run_logger", lambda: SimpleNamespace(info=lambda *a: None))


def electronic_properties(norb=NORB):
    hamiltonian = ffsim.random.random_molecular_hamiltonian(norb, seed=1, dtype=float)
    return SimpleNamespace(
        num_orbitals=norb,
        one_body_tensor=hamiltonian.one_body_tensor,
        two_body_tensor=hamiltonian.two_body_tensor,
        num_electrons=(1, 1),
    )


def trotter_circuits(krylov_dim=3):
    skqd = main.SKQDParameters(
        krylov_dim=krylov_dim, subspace_dim=4, num_batches=1, max_iterations=1
    )
    return main.create_trotter_circuits.fn(skqd, electronic_properties())


def qpy_round_trip(circuits):
    # The Slurm path sends the circuits to the compute node as QPY.
    stream = io.BytesIO()
    qpy.dump(circuits, stream)
    stream.seek(0)
    return qpy.load(stream)


def flow_config(circuit_params, seed):
    return hook.transpile_config(
        optimization_level=circuit_params.optimization_level,
        seed_transpiler=seed,
        sabre_max_iterations=circuit_params.sabre_max_iterations,
        sabre_swap_trials=circuit_params.sabre_swap_trials,
        sabre_layout_trials=circuit_params.sabre_layout_trials,
    )


@pytest.mark.parametrize("basis", [["cz", "rz", "sx", "x"], ["cz", "rz", "sx", "x", "rzz"]])
@pytest.mark.parametrize("transport", ["iqp", "slurm"])
def test_hook_gives_the_same_isa_as_the_old_transpile_circuits(basis, transport):
    target = line_target(basis)
    circuits = trotter_circuits()
    circuit_params = main.CircuitParameters(**SABRE)
    reference = reference_transpile(circuits, target, circuit_params, np.random.default_rng(24))

    # The flow draws the seed from the same generator state as the old task did.
    seed = np.random.default_rng(24).integers(np.iinfo(int).max, size=1).item()
    config = json.loads(json.dumps(flow_config(circuit_params, seed)))
    sent = qpy_round_trip(circuits) if transport == "slurm" else circuits
    isa = [hook.transpile(circuit, target, config) for circuit in sent]

    for expected, actual in zip(reference, isa, strict=True):
        assert dict(actual.count_ops()) == dict(expected.count_ops())
        assert actual.layout.final_index_layout() == expected.layout.final_index_layout()
        assert actual == expected
    if "rzz" in basis:
        assert any("rzz" in circuit.count_ops() for circuit in isa)


def test_seed_decides_the_isa():
    target = line_target()
    circuit = trotter_circuits(krylov_dim=2)[1]
    circuit_params = main.CircuitParameters(**SABRE)
    first = hook.transpile(circuit, target, flow_config(circuit_params, seed=11))
    assert first == hook.transpile(circuit, target, flow_config(circuit_params, seed=11))
    others = [
        hook.transpile(circuit, target, flow_config(circuit_params, seed=s)) for s in range(4)
    ]
    assert any(other != first for other in others)


def test_pass_manager_uses_the_ffsim_pre_init_and_sabre_settings():
    # For these circuits, other pre_init stages can give the same ISA, so check the stages.
    target = line_target()
    config = flow_config(main.CircuitParameters(**SABRE), seed=11)
    passmanager = hook.build_pass_manager(target, config)
    assert passmanager.pre_init is PRE_INIT
    sabre = [
        task
        for controller in passmanager.layout.to_flow_controller().tasks
        for task in getattr(controller, "tasks", [controller])
        if isinstance(task, SabreLayout)
    ]
    assert len(sabre) == 1
    assert sabre[0].seed == 11
    assert sabre[0].max_iterations == SABRE["sabre_max_iterations"]
    assert sabre[0].swap_trials == SABRE["sabre_swap_trials"]
    assert sabre[0].layout_trials == SABRE["sabre_layout_trials"]


def test_hook_preserves_the_measurement_register():
    target = line_target()
    circuit = trotter_circuits(krylov_dim=2)[1]
    isa = hook.transpile(circuit, target, flow_config(main.CircuitParameters(**SABRE), seed=3))
    assert [(reg.name, len(reg)) for reg in isa.cregs] == [("meas", 2 * NORB)]


@pytest.mark.parametrize(
    "config,message",
    [
        ({"initial_layout": [0, 1]}, "Unsupported SKQD transpile options"),
        ({"optimization_level": 4}, "optimization_level"),
        ({"seed_transpiler": -1}, "seed_transpiler"),
        ({"seed_transpiler": 1.5}, "seed_transpiler"),
        ({"sabre_layout_trials": 0}, "sabre_layout_trials"),
        ({"sabre_swap_trials": True}, "sabre_swap_trials"),
    ],
)
def test_invalid_config_is_rejected(config, message):
    target = GenericBackendV2(num_qubits=4, seed=7).target
    with pytest.raises(ValueError, match=message):
        hook.build_pass_manager(target, config)
