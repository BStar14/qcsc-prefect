"""SKQD sampling through QrmiRuntime, with QRMI replaced at the IQP sampling task."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import ffsim
import numpy as np
import pytest
from qcsc_prefect.integrations.qiskit.qrmi import QrmiRuntime, QrmiRuntimeBlock
from qcsc_prefect.integrations.qiskit.qrmi import iqp as iqp_mod
from qcsc_prefect.integrations.qiskit.qrmi.compile import compile_pubs
from qiskit.primitives import PrimitiveResult, SamplerPubResult
from qiskit.primitives.containers import BitArray, DataBin
from qiskit.providers.fake_provider import GenericBackendV2
from skqd_dice import main
from skqd_dice.transpilation import TRANSPILER

NORB = 3
SHOTS = 8


class FakeVariable:
    def __init__(self, value):
        self.value = value

    async def get(self, name, default=None):
        return self.value if self.value is not None else default


@pytest.fixture(autouse=True)
def no_prefect(monkeypatch):
    monkeypatch.setattr(main, "get_run_logger", lambda: SimpleNamespace(info=lambda *a: None))
    monkeypatch.setattr(main, "create_trotter_circuits", main.create_trotter_circuits.fn)
    monkeypatch.setattr(main, "Variable", FakeVariable({"params": {"shots": SHOTS}}))


def trotter_steps(circuit):
    return circuit.count_ops().get("PauliEvolution", 0)


@pytest.fixture
def calls(monkeypatch):
    """Replace QRMI: compile with the runtime's hook on a fixed Target, return fixed bits.

    Every shot of a PUB is the bitstring of ``k + 1``, where the circuit applies the
    Trotter step ``k`` times. The results therefore follow the circuits that were sent,
    and their order can be checked after concatenation.
    """
    target = GenericBackendV2(num_qubits=2 * NORB + 4, seed=7).target
    calls = []

    async def fake_sample_on_iqp(circuits, shots, registers, configs, runtime):
        isa_circuits, reports = compile_pubs(
            circuits, configs, registers, target, runtime.transpiler
        )
        calls.append(
            {"circuits": circuits, "configs": configs, "runtime": runtime, "isa": isa_circuits}
        )
        width = registers[0]["meas"]
        results = [
            SamplerPubResult(
                DataBin(meas=BitArray.from_samples([format(i + 1, f"0{width}b")] * shots, width)),
                metadata={},
            )
            for i in map(trotter_steps, circuits)
        ]
        return PrimitiveResult(results, metadata={"qcsc_qrmi": {"pubs": reports}})

    monkeypatch.setattr(iqp_mod, "sample_on_iqp", fake_sample_on_iqp)
    return calls


@pytest.fixture
def saved_runtime(monkeypatch):
    runtime = QrmiRuntime(
        scheduler="iqp",
        resource="ibm_test",
        resource_config={"QRMI_IBM_QCS_IAM_APIKEY": "secret"},
        transpile_config={"optimization_level": 0},
    )
    loaded = []

    async def aload_runtime(name, prepare=None):
        loaded.append(name)
        return runtime

    monkeypatch.setattr(QrmiRuntimeBlock, "aload_runtime", aload_runtime)
    return loaded


def electronic_properties():
    hamiltonian = ffsim.random.random_molecular_hamiltonian(NORB, seed=1, dtype=float)
    return SimpleNamespace(
        num_orbitals=NORB,
        one_body_tensor=hamiltonian.one_body_tensor,
        two_body_tensor=hamiltonian.two_body_tensor,
        num_electrons=(1, 1),
    )


def parameters(quantum_source, krylov_dim=3):
    return main.Parameters(
        molecule=main.FCIDumpFile(),
        circuit=main.CircuitParameters(
            sabre_max_iterations=2, sabre_swap_trials=4, sabre_layout_trials=20
        ),
        skqd=main.SKQDParameters(
            krylov_dim=krylov_dim, subspace_dim=4, num_batches=1, max_iterations=1
        ),
        quantum_source=quantum_source,
    )


def sample(params, runner_name="my-runtime"):
    return asyncio.run(
        main.sample_bitstrings.fn(
            parameters=params,
            elec_props=electronic_properties(),
            runner_name=runner_name,
            option_name="sampler_options",
            quantum_source=params.quantum_source,
            random_seed=params.random_seed,
        )
    )


def test_krylov_circuits_are_sent_in_one_call_and_results_kept_in_order(calls, saved_runtime):
    bits = sample(parameters("real-device", krylov_dim=3))

    assert saved_runtime == ["my-runtime"]
    assert len(calls) == 1
    call = calls[0]
    assert [trotter_steps(circuit) for circuit in call["circuits"]] == [0, 1, 2]
    # The circuits are sent untranspiled; the runtime compiles them with the SKQD hook.
    assert call["circuits"][0].count_ops().get("hartree_fock_jw") == 1
    assert call["runtime"].transpiler == TRANSPILER
    assert all(len(isa.cregs) == 1 for isa in call["isa"])

    width = 2 * NORB
    expected = [format(i + 1, f"0{width}b") for i in range(3) for _ in range(SHOTS)]
    assert bits.num_shots == 3 * SHOTS
    assert bits.get_bitstrings() == expected


def test_circuit_parameters_reach_the_hook_as_json(calls, saved_runtime):
    params = parameters("real-device", krylov_dim=2)
    sample(params)

    seed = np.random.default_rng(24).integers(np.iinfo(int).max, size=1).item()
    expected = {
        "optimization_level": params.circuit.optimization_level,
        "seed_transpiler": seed,
        "sabre_max_iterations": 2,
        "sabre_swap_trials": 4,
        "sabre_layout_trials": 20,
    }
    # The flow's configuration replaces the block's transpile_config for every PUB.
    assert calls[0]["configs"] == [expected, expected]


def test_random_mode_does_not_load_a_runtime(monkeypatch, calls):
    async def aload_runtime(name, prepare=None):
        raise AssertionError("random mode must not load a runtime")

    monkeypatch.setattr(QrmiRuntimeBlock, "aload_runtime", aload_runtime)
    params = parameters("random", krylov_dim=3)
    bits = sample(params)

    assert calls == []
    assert bits.num_shots == SHOTS * 3
    assert bits.num_bits == 2 * NORB
    again = sample(params)
    assert bits.get_bitstrings() == again.get_bitstrings()


def test_missing_runtime_block_points_to_random_mode(monkeypatch, calls):
    async def aload_runtime(name, prepare=None):
        raise ValueError(f"Unable to find block document named {name}")

    monkeypatch.setattr(QrmiRuntimeBlock, "aload_runtime", aload_runtime)
    with pytest.raises(RuntimeError, match="quantum_source='random'"):
        sample(parameters("real-device"))
    assert calls == []
