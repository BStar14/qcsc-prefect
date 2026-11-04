"""The SQD hooks reproduce the workflow-host transpile_circuit they replaced."""

from __future__ import annotations

import io
import json
from types import SimpleNamespace

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
from sqd_dice import main
from sqd_dice import transpilation as hook

NORB = 4
NUM_QUBITS = 2 * NORB + 6
SABRE = {"sabre_max_iterations": 2, "sabre_swap_trials": 4, "sabre_layout_trials": 20}


def reference_transpile(circuit, target, circuit_params, seed):
    """``transpile_circuit`` as it was before the QrmiRuntime migration."""
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
    return passmanager.run(circuit)


@pytest.fixture(autouse=True)
def no_run_logger(monkeypatch):
    monkeypatch.setattr(main, "get_run_logger", lambda: SimpleNamespace(info=lambda *a: None))


def line_target(basis=("cz", "rz", "sx", "x")):
    # GenericBackendV2 is fully connected by default, where every layout is equally good
    # and the SABRE settings and seed have no effect. A line makes them matter.
    return GenericBackendV2(
        num_qubits=NUM_QUBITS,
        basis_gates=list(basis),
        coupling_map=CouplingMap.from_line(NUM_QUBITS),
        seed=7,
    ).target


def electronic_properties():
    t2 = np.random.default_rng(5).normal(scale=0.1, size=(2, 2, 2, 2))
    return SimpleNamespace(
        num_orbitals=NORB,
        num_electrons=(2, 2),
        open_shell=False,
        t2=(t2 + t2.transpose(1, 0, 3, 2)) / 2,  # Pair-symmetric, as from CCSD.
    )


def lucj_circuit(use_reset_mitigation):
    circuit_params = main.CircuitParameters(use_reset_mitigation=use_reset_mitigation, **SABRE)
    return main.create_ansatz_circuits.fn(circuit_params, electronic_properties())


def qpy_round_trip(circuit):
    # The Slurm path sends the prepared circuit to the compute node as QPY.
    stream = io.BytesIO()
    qpy.dump(circuit, stream)
    stream.seek(0)
    return qpy.load(stream)[0]


def flow_config(circuit_params, seed):
    return hook.transpile_config(
        optimization_level=circuit_params.optimization_level,
        seed_transpiler=seed,
        sabre_max_iterations=circuit_params.sabre_max_iterations,
        sabre_swap_trials=circuit_params.sabre_swap_trials,
        sabre_layout_trials=circuit_params.sabre_layout_trials,
    )


@pytest.mark.parametrize("use_reset_mitigation", [False, True])
@pytest.mark.parametrize("basis", [["cz", "rz", "sx", "x"], ["cz", "rz", "sx", "x", "rzz"]])
@pytest.mark.parametrize("transport", ["iqp", "slurm"])
def test_hooks_give_the_same_isa_as_the_old_transpile_circuit(
    basis, transport, use_reset_mitigation
):
    target = line_target(basis)
    circuit = lucj_circuit(use_reset_mitigation)
    circuit_params = main.CircuitParameters(**SABRE)
    expected = reference_transpile(circuit, target, circuit_params, np.random.default_rng(24))

    # The flow draws the seed from the same generator state as the old task did.
    seed = np.random.default_rng(24).integers(np.iinfo(int).max, size=1).item()
    config = json.loads(json.dumps(flow_config(circuit_params, seed)))
    prepared = hook.prepare(circuit.copy()).circuit
    sent = qpy_round_trip(prepared) if transport == "slurm" else prepared
    actual = hook.transpile(sent, target, config)

    assert dict(actual.count_ops()) == dict(expected.count_ops())
    assert actual.layout.initial_index_layout() == expected.layout.initial_index_layout()
    assert actual.layout.final_index_layout() == expected.layout.final_index_layout()
    assert actual == expected
    if "rzz" in basis:
        assert "rzz" in actual.count_ops()


def test_pre_init_cannot_run_after_qpy_so_prepare_runs_it_first():
    # QPY turns the ffsim gates into plain gates, which the ffsim pre-init stage rejects.
    circuit = lucj_circuit(use_reset_mitigation=True)
    with pytest.raises(AttributeError):
        PRE_INIT.run(qpy_round_trip(circuit))
    prepared = hook.prepare(circuit).circuit
    assert "ucj_balanced_jw" in circuit.count_ops()
    assert "ucj_balanced_jw" not in prepared.count_ops()
    assert [(reg.name, len(reg)) for reg in prepared.cregs] == [("meas", 8), ("test", 8)]


def test_seed_reaches_the_preset_pass_manager(monkeypatch):
    # For this small circuit every seed finds the same layout, so the ISA alone cannot
    # show that the seed is used. Record the preset arguments instead; the SabreLayout
    # seed is checked below.
    seen = []
    preset = hook.generate_preset_pass_manager

    def recording_preset(**kwargs):
        seen.append(kwargs)
        return preset(**kwargs)

    monkeypatch.setattr(hook, "generate_preset_pass_manager", recording_preset)
    hook.build_pass_manager(line_target(), flow_config(main.CircuitParameters(**SABRE), seed=11))
    assert [(kw["optimization_level"], kw["seed_transpiler"]) for kw in seen] == [(3, 11)]


def test_pass_manager_has_no_pre_init_and_uses_the_sabre_settings():
    passmanager = hook.build_pass_manager(
        line_target(), flow_config(main.CircuitParameters(**SABRE), seed=11)
    )
    assert passmanager.pre_init is None
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


@pytest.mark.parametrize(
    "config,message",
    [
        ({"initial_layout": [0, 1]}, "Unsupported SQD transpile options"),
        ({"optimization_level": 4}, "optimization_level"),
        ({"seed_transpiler": -1}, "seed_transpiler"),
        ({"sabre_layout_trials": 0}, "sabre_layout_trials"),
        ({"sabre_swap_trials": True}, "sabre_swap_trials"),
    ],
)
def test_invalid_config_is_rejected(config, message):
    with pytest.raises(ValueError, match=message):
        hook.build_pass_manager(line_target(), config)
