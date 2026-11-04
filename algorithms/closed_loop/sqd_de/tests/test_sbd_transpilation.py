"""The SBD hooks reproduce find_optimal_layout + transpile_circuit on the workflow host."""

from __future__ import annotations

import io
import json

import pytest
from ffsim.qiskit import PRE_INIT
from qiskit import qpy
from sbd import transpilation as hook
from sbd.flow_params import CircuitParameters
from sbd.transpile_core import _find_optimal_layout, _transpile_circuit
from sbd_helpers import line_target

CIRCUIT_PARAMS = CircuitParameters(
    optimization_level=3, sabre_max_iterations=2, sabre_swap_trials=4, sabre_layout_trials=20
)


def reference(circuits, target):
    """Former real-device path: one layout search (cached per flow run), then transpile."""
    layout, _ = _find_optimal_layout(
        test_circuit=circuits[0],
        target=target,
        optimization_level=CIRCUIT_PARAMS.optimization_level,
        max_iterations=CIRCUIT_PARAMS.sabre_max_iterations,
        swap_trials=CIRCUIT_PARAMS.sabre_swap_trials,
        layout_trials=CIRCUIT_PARAMS.sabre_layout_trials,
    )
    return [
        _transpile_circuit(
            circuit=circuit,
            target=target,
            layout=layout,
            optimization_level=CIRCUIT_PARAMS.optimization_level,
        )
        for circuit in circuits
    ]


def qpy_round_trip(circuit):
    # The Slurm path sends the prepared circuits to the compute node as QPY.
    stream = io.BytesIO()
    qpy.dump(circuit, stream)
    stream.seek(0)
    return qpy.load(stream)[0]


@pytest.mark.parametrize("basis", [["cz", "rz", "sx", "x"], ["cz", "rz", "sx", "x", "rzz"]])
@pytest.mark.parametrize("transport", ["iqp", "slurm"])
def test_hooks_give_the_same_isa_as_the_old_layout_and_transpile(circuits, basis, transport):
    target = line_target(basis)
    expected = reference(circuits, target)

    config = json.loads(json.dumps(hook.transpile_config(CIRCUIT_PARAMS)))
    actual = []
    for circuit in circuits:
        prepared = hook.prepare(circuit.copy()).circuit
        sent = qpy_round_trip(prepared) if transport == "slurm" else prepared
        actual.append(hook.transpile(sent, target, config))

    for old, new in zip(expected, actual, strict=True):
        assert dict(new.count_ops()) == dict(old.count_ops())
        assert new.layout.initial_index_layout() == old.layout.initial_index_layout()
        assert new.layout.final_index_layout() == old.layout.final_index_layout()
        assert new == old
    if "rzz" in basis:
        assert any("rzz" in circuit.count_ops() for circuit in actual)


def test_pre_init_cannot_run_after_qpy_so_prepare_runs_it_first(circuits):
    # QPY turns the ffsim gates into plain gates, which the ffsim pre-init stage rejects.
    # This is why prepare runs it on the workflow host for both schedulers.
    with pytest.raises(AttributeError):
        PRE_INIT.run(qpy_round_trip(circuits[0]))
    prepared = hook.prepare(circuits[0]).circuit
    assert "ucj_balanced_jw" in circuits[0].count_ops()
    assert "ucj_balanced_jw" not in prepared.count_ops()
    assert "hartree_fock_jw" not in prepared.count_ops()


def test_prepare_preserves_both_classical_registers(circuits):
    prepared = hook.prepare(circuits[0]).circuit
    assert [(reg.name, len(reg)) for reg in prepared.cregs] == [("meas", 8), ("test", 8)]


def test_config_reaches_the_layout_search_and_transpile(circuits, monkeypatch):
    # For these small circuits every SABRE setting finds the same layout, so the ISA
    # alone cannot show that the settings are used. Record the calls instead.
    calls = []

    def spy(name, function):
        def wrapper(**kwargs):
            calls.append((name, {k: v for k, v in kwargs.items() if k != "circuit"}))
            return function(**kwargs)

        return wrapper

    monkeypatch.setattr(hook, "_find_optimal_layout", spy("layout", hook._find_optimal_layout))
    monkeypatch.setattr(hook, "_transpile_circuit", spy("transpile", hook._transpile_circuit))
    config = {
        "optimization_level": 2,
        "sabre_max_iterations": 3,
        "sabre_swap_trials": 5,
        "sabre_layout_trials": 7,
    }
    prepared = hook.prepare(circuits[0]).circuit
    hook.transpile(prepared, line_target(), config)

    (layout_name, layout_args), (transpile_name, transpile_args) = calls
    assert layout_name == "layout"
    assert {k: layout_args[k] for k in ("optimization_level", "use_pre_init")} == {
        "optimization_level": 2,
        "use_pre_init": False,
    }
    assert (layout_args["max_iterations"], layout_args["swap_trials"]) == (3, 5)
    assert layout_args["layout_trials"] == 7
    assert transpile_name == "transpile"
    assert transpile_args["optimization_level"] == 2
    assert transpile_args["use_pre_init"] is False


def test_config_comes_from_circuit_parameters():
    assert hook.transpile_config(CIRCUIT_PARAMS) == {
        "optimization_level": 3,
        "sabre_max_iterations": 2,
        "sabre_swap_trials": 4,
        "sabre_layout_trials": 20,
    }


@pytest.mark.parametrize(
    "change,message",
    [
        ({"initial_layout": [0]}, "exactly the keys"),
        ({"optimization_level": 4}, "optimization_level"),
        ({"sabre_layout_trials": 0}, "sabre_layout_trials"),
        ({"sabre_swap_trials": True}, "sabre_swap_trials"),
    ],
)
def test_invalid_config_is_rejected(circuits, change, message):
    config = {**hook.transpile_config(CIRCUIT_PARAMS), **change}
    with pytest.raises(ValueError, match=message):
        hook.transpile(circuits[0], line_target(), config)
