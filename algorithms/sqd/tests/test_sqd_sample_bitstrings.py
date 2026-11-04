"""SQD sampling through QrmiRuntime, with QRMI replaced at the IQP sampling task."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest
from qcsc_prefect.integrations.qiskit.qrmi import QrmiRuntime, QrmiRuntimeBlock
from qcsc_prefect.integrations.qiskit.qrmi import iqp as iqp_mod
from qcsc_prefect.integrations.qiskit.qrmi.compile import compile_pubs
from qiskit.primitives import PrimitiveResult, SamplerPubResult
from qiskit.primitives.containers import BitArray, DataBin
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.transpiler import CouplingMap
from sqd_dice import main
from sqd_dice.transpilation import TRANSPILER, prepare

NORB = 4
WIDTH = 2 * NORB
SHOTS = 8


class FakeVariable:
    value = {"params": {"shots": SHOTS}}

    @classmethod
    async def get(cls, name, default=None):
        return cls.value


@pytest.fixture(autouse=True)
def no_prefect(monkeypatch):
    monkeypatch.setattr(main, "get_run_logger", lambda: SimpleNamespace(info=lambda *a: None))
    monkeypatch.setattr(main, "create_ansatz_circuits", main.create_ansatz_circuits.fn)
    monkeypatch.setattr(FakeVariable, "value", {"params": {"shots": SHOTS}})
    monkeypatch.setattr(main, "Variable", FakeVariable)


@pytest.fixture
def calls(monkeypatch):
    """Replace QRMI: compile with the runtime's hook on a fixed Target, return fixed bits.

    ``meas`` is all ones. ``test`` is all zeros except for the last two shots, so reset
    mitigation keeps ``shots - 2`` of them.
    """
    num_qubits = WIDTH + 6
    target = GenericBackendV2(
        num_qubits=num_qubits, coupling_map=CouplingMap.from_line(num_qubits), seed=7
    ).target
    calls = []

    async def fake_sample_on_iqp(circuits, shots, registers, configs, runtime):
        isa, _ = compile_pubs(circuits, configs, registers, target, runtime.transpiler)
        calls.append({"circuits": circuits, "configs": configs, "runtime": runtime, "isa": isa})
        data = {"meas": BitArray.from_samples(["1" * WIDTH] * shots, WIDTH)}
        if "test" in registers[0]:
            test = ["0" * WIDTH] * (shots - 2) + ["1" + "0" * (WIDTH - 1)] * 2
            data["test"] = BitArray.from_samples(test, WIDTH)
        return PrimitiveResult([SamplerPubResult(DataBin(**data), metadata={})], metadata={})

    monkeypatch.setattr(iqp_mod, "sample_on_iqp", fake_sample_on_iqp)
    return calls


@pytest.fixture
def saved_runtime(monkeypatch):
    loaded = []

    async def aload_runtime(name, prepare=None):
        loaded.append(name)
        return QrmiRuntime(
            scheduler="iqp",
            resource="ibm_test",
            resource_config={"QRMI_IBM_QCS_IAM_APIKEY": "secret"},
            transpile_config={"optimization_level": 0},
        )

    monkeypatch.setattr(QrmiRuntimeBlock, "aload_runtime", aload_runtime)
    return loaded


def electronic_properties():
    t2 = np.random.default_rng(5).normal(scale=0.1, size=(2, 2, 2, 2))
    return SimpleNamespace(
        num_orbitals=NORB,
        num_electrons=(2, 2),
        open_shell=False,
        t2=(t2 + t2.transpose(1, 0, 3, 2)) / 2,
    )


def circuit_params(use_reset_mitigation=True):
    return main.CircuitParameters(
        use_reset_mitigation=use_reset_mitigation,
        sabre_max_iterations=2,
        sabre_swap_trials=4,
        sabre_layout_trials=20,
    )


def sample(quantum_source, params=None, runner_name="my-runtime", random_seed=24):
    return asyncio.run(
        main.sample_bitstrings.fn(
            circuit_params=params or circuit_params(),
            elec_props=electronic_properties(),
            runner_name=runner_name,
            option_name="sampler_options",
            quantum_source=quantum_source,
            random_seed=random_seed,
        )
    )


def test_real_device_sends_the_lucj_circuit_untranspiled_once(calls, saved_runtime):
    bits = sample("real-device")

    assert saved_runtime == ["my-runtime"]
    (call,) = calls
    runtime = call["runtime"]
    assert runtime.transpiler == TRANSPILER
    assert runtime.prepare is prepare
    # prepare ran the ffsim pre-init stage; the circuit is otherwise untranspiled.
    (sent,) = call["circuits"]
    assert "ucj_balanced_jw" not in sent.count_ops()
    assert {"hartree_fock_jw", "measure"} <= set(sent.count_ops()) | {"hartree_fock_jw"}
    assert {reg.name for reg in call["isa"][0].cregs} == {"meas", "test"}
    # Reset mitigation keeps the shots whose test register is all zeros.
    assert bits.num_shots == SHOTS - 2
    assert bits.get_bitstrings() == ["1" * WIDTH] * (SHOTS - 2)


def test_real_device_without_reset_mitigation(calls, saved_runtime):
    bits = sample("real-device", circuit_params(use_reset_mitigation=False))
    assert [reg.name for reg in calls[0]["isa"][0].cregs] == ["meas"]
    assert bits.num_shots == SHOTS


def test_circuit_parameters_reach_the_hook_as_json(calls, saved_runtime):
    params = circuit_params()
    sample("real-device", params)

    seed = np.random.default_rng(24).integers(np.iinfo(int).max, size=1).item()
    # The flow's configuration replaces the block's transpile_config.
    assert calls[0]["configs"] == [
        {
            "optimization_level": params.optimization_level,
            "seed_transpiler": seed,
            "sabre_max_iterations": 2,
            "sabre_swap_trials": 4,
            "sabre_layout_trials": 20,
        }
    ]


def test_sampler_options_are_passed_to_qrmi_runtime(calls, saved_runtime, monkeypatch):
    # Options are passed unchanged; QrmiRuntime rejects anything but positive integer shots.
    monkeypatch.setattr(FakeVariable, "value", {"params": {"shots": 5}})
    assert sample("real-device", circuit_params(use_reset_mitigation=False)).num_shots == 5

    for options in ({"params": {"shots": 4.9}}, {"params": {"shots": SHOTS}, "twirling": {}}):
        monkeypatch.setattr(FakeVariable, "value", options)
        with pytest.raises(ValueError):
            sample("real-device")
    assert len(calls) == 1


def test_random_mode_returns_before_any_circuit_is_built(monkeypatch, calls):
    async def aload_runtime(name, prepare=None):
        raise AssertionError("random mode must not load a runtime")

    def create_ansatz_circuits(**kwargs):
        raise AssertionError("random mode must not build a circuit")

    monkeypatch.setattr(QrmiRuntimeBlock, "aload_runtime", aload_runtime)
    monkeypatch.setattr(main, "create_ansatz_circuits", create_ansatz_circuits)
    bits = sample("random")

    assert calls == []
    assert (bits.num_shots, bits.num_bits) == (SHOTS, WIDTH)
    assert bits.get_bitstrings() == sample("random").get_bitstrings()
    assert bits.get_bitstrings() != sample("random", random_seed=25).get_bitstrings()


def test_missing_runtime_block_points_to_random_mode(monkeypatch, calls):
    async def aload_runtime(name, prepare=None):
        raise ValueError(f"Unable to find block document named {name}")

    monkeypatch.setattr(QrmiRuntimeBlock, "aload_runtime", aload_runtime)
    with pytest.raises(RuntimeError, match="quantum_source='random'"):
        sample("real-device")
    assert calls == []
