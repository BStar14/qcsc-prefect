"""Offline checks of the actual SQD workflow's optional QRMI boundary."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import socket
from types import SimpleNamespace
from unittest.mock import patch

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("ffsim")
pytest.importorskip("pydantic_numpy")
pytest.importorskip("prefect_qiskit")
pytest.importorskip("qcsc_prefect.integrations.qiskit.qrmi")

from pyscf import lib  # noqa: E402
from qcsc_prefect.integrations.qiskit.qrmi import runtime as runtime_mod  # noqa: E402
from qcsc_prefect.integrations.qiskit.qrmi.worker import validate_isa  # noqa: E402
from qcsc_workflow_utility import chem  # noqa: E402
from qiskit import qpy  # noqa: E402
from qiskit.primitives import PrimitiveResult, SamplerPubResult  # noqa: E402
from qiskit.primitives.containers import BitArray, DataBin  # noqa: E402
from qiskit.providers.fake_provider import GenericBackendV2  # noqa: E402
from qiskit.quantum_info import Statevector  # noqa: E402
from qiskit_ibm_runtime import RuntimeEncoder  # noqa: E402
from sqd_fulqrum import main as workflow  # noqa: E402
from sqd_fulqrum.qrmi import prepare_sampler_circuit, transpile_for_target  # noqa: E402

LOGGER = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def reject(*args, **kwargs):
        pytest.fail("SQD QRMI tests must not open network connections")

    monkeypatch.setattr(socket.socket, "connect", reject)
    monkeypatch.setattr(socket.socket, "connect_ex", reject)
    monkeypatch.setattr(socket, "create_connection", reject)


@pytest.fixture(scope="module")
def h2():
    with lib.with_omp_threads(1), patch.object(chem, "get_run_logger", return_value=LOGGER):
        return chem.compute_molecular_integrals_from_geometry.fn(
            atom="H 0 0 0; H 0 0 0.74", basis="sto-3g"
        )


def circuit_parameters(reset=False):
    return workflow.CircuitParameters(
        use_reset_mitigation=reset,
        sabre_layout_trials=4,
        sabre_swap_trials=2,
        sabre_max_iterations=2,
    )


def make_circuit(params, h2):
    with patch.object(workflow, "get_run_logger", return_value=LOGGER):
        return workflow.create_ansatz_circuits.fn(params, h2)


def target(with_rzz=False):
    gates = ["rz", "sx", "x", "cz"] + (["rzz"] if with_rzz else [])
    edges = [(i, i + 1) for i in range(5)] + [(i + 1, i) for i in range(5)]
    return GenericBackendV2(num_qubits=6, basis_gates=gates, coupling_map=edges, seed=42).target


def readout_probabilities(circuit):
    """Account for physical placement and logical readout order after routing."""
    meas = next(reg for reg in circuit.cregs if reg.name == "meas")
    unmeasured = circuit.copy_empty_like()
    mapping = []
    for inst in circuit.data:
        qubits = [circuit.find_bit(bit).index for bit in inst.qubits]
        if inst.operation.name == "measure":
            if inst.clbits[0] in meas:
                mapping.append((qubits[0], list(meas).index(inst.clbits[0])))
        else:
            unmeasured.append(inst.operation, qubits, inst.clbits)
    assert sorted(bit for _, bit in mapping) == list(range(len(meas)))
    probabilities = np.zeros(1 << len(meas))
    for basis, probability in enumerate(Statevector.from_instruction(unmeasured).probabilities()):
        outcome = sum(((basis >> qubit) & 1) << bit for qubit, bit in mapping)
        probabilities[outcome] += probability
    return probabilities


@pytest.mark.parametrize("reset", [False, True])
@pytest.mark.parametrize("with_rzz", [False, True])
def test_lucj_hooks_preserve_readout_through_qpy_without_prefect_context(h2, reset, with_rzz):
    params = circuit_parameters(reset)
    original = make_circuit(params, h2)
    prepared = prepare_sampler_circuit(original.copy(), circuit_parameters=params.model_dump())
    stream = io.BytesIO()
    qpy.dump(prepared.circuit, stream)
    stream.seek(0)
    (restored,) = qpy.load(stream)
    actual_target = target(with_rzz)
    isa = transpile_for_target(restored, actual_target, prepared.config)
    validate_isa(isa, actual_target)
    assert isa.num_qubits == actual_target.num_qubits
    assert isa.num_parameters == 0
    assert [(reg.name, len(reg)) for reg in isa.cregs] == [
        (reg.name, len(reg)) for reg in original.cregs
    ]
    np.testing.assert_allclose(
        readout_probabilities(isa), readout_probabilities(original), atol=1e-10, rtol=0
    )


def sample_result(reset):
    data = {"meas": BitArray.from_samples(["0101", "0110", "1001", "1010"], num_bits=4)}
    if reset:
        data["test"] = BitArray.from_samples(["0000", "0001", "0000", "0010"], num_bits=4)
    return PrimitiveResult([SamplerPubResult(DataBin(**data), metadata={})], metadata={})


@pytest.mark.parametrize("reset", [False, True])
def test_sampler_qrmi_qpy_handoff_and_reset_postselection(tmp_path, monkeypatch, h2, reset):
    params = circuit_parameters(reset)
    config = {
        "root_dir": str(tmp_path / "quantum"),
        "command_block_name": "qrmi-command",
        "execution_profile_block_name": "qrmi-execution",
        "hpc_profile_block_name": "qrmi-hpc",
        "resource": "test_heron",
    }
    calls = []

    async def submit(**kwargs):
        calls.append(kwargs)
        work_dir = kwargs["work_dir"]
        request = json.loads((work_dir / "sampler_request.json").read_text())
        assert request["transpiler"] == "sqd_fulqrum.qrmi:transpile_for_target"
        assert request["resource"] == "test_heron"
        assert request["shots"] == 4
        hook_config = json.loads((work_dir / "transpile_config.json").read_text())
        assert hook_config["circuit_parameters"] == params.model_dump()
        assert hook_config["seed"] == 24
        with (work_dir / "circuit.qpy").open("rb") as stream:
            (circuit,) = qpy.load(stream)
        actual_target = target()
        isa = transpile_for_target(circuit, actual_target, hook_config)
        validate_isa(isa, actual_target)
        (work_dir / "result.json").write_text(json.dumps(sample_result(reset), cls=RuntimeEncoder))
        return SimpleNamespace(job_id="73", state="COMPLETED", exit_status=0)

    async def options(*args, **kwargs):
        return {"params": {"shots": 4}}

    async def reject_ibm(*args, **kwargs):
        pytest.fail("QRMI sampling must not load an IBM Runtime block")

    monkeypatch.setattr(runtime_mod, "_sample_on_slurm", runtime_mod._sample_on_slurm.fn)
    monkeypatch.setattr(runtime_mod, "get_run_logger", lambda: LOGGER)
    monkeypatch.setattr(runtime_mod, "run_job_from_blocks", submit)
    monkeypatch.setattr(workflow, "get_run_logger", lambda: LOGGER)
    monkeypatch.setattr(workflow, "create_ansatz_circuits", workflow.create_ansatz_circuits.fn)
    monkeypatch.setattr(workflow.Variable, "get", options)
    monkeypatch.setattr(workflow.QuantumRuntime, "load", reject_ibm)
    result = asyncio.run(
        workflow.sample_bitstrings.fn(
            params,
            h2,
            "unused-block",
            "options",
            "real-device",
            24,
            qrmi_runtime_config=config,
        )
    )
    assert result.get_bitstrings() == (
        ["0101", "1001"] if reset else ["0101", "0110", "1001", "1010"]
    )
    assert len(calls) == 1
    assert calls[0]["command_block_name"] == "qrmi-command"
    assert "prepare" not in config and "transpiler" not in config

    # Exercise the original flow body and recovery loop. Only scheduler/native
    # execution and Prefect orchestration boundaries are replaced.
    solver_calls = []
    artifacts = []

    async def solve(**kwargs):
        assert kwargs["one_body_tensor"] is h2.one_body_tensor
        assert kwargs["two_body_tensor"] is h2.two_body_tensor
        assert kwargs["nelec"] == h2.num_electrons
        solver_calls.append(kwargs)
        return SimpleNamespace(energy=-1.5, orbital_occupancies=h2.initial_occupancy)

    async def load_solver(name):
        assert name == "sqd-fulqrum-solver"
        return SimpleNamespace(run=solve)

    async def artifact(**kwargs):
        artifacts.append(kwargs)

    class DirectTask:
        def __init__(self, fn):
            self.fn = fn

        def with_options(self, **kwargs):
            return self.fn

    monkeypatch.setattr(
        workflow, "compute_molecular_integrals_from_geometry", DirectTask(lambda **kwargs: h2)
    )
    monkeypatch.setattr(workflow, "sample_bitstrings", DirectTask(workflow.sample_bitstrings.fn))
    monkeypatch.setattr(workflow.FulqrumSolverJob, "load", load_solver)
    monkeypatch.setattr(workflow, "create_table_artifact", artifact)
    parameters = workflow.Parameters(
        molecule={"atom": "H 0 0 0; H 0 0 0.74", "basis": "sto-3g"},
        circuit=params,
        sqd={"subspace_dim": 1, "num_batches": 1, "max_iterations": 1},
    )
    energy = asyncio.run(workflow.sqd_2405_05068_fulqrum.fn(parameters, qrmi_runtime_config=config))
    assert energy == pytest.approx(-1.5 + h2.nuclear_repulsion_energy)
    assert len(calls) == 2
    assert len(solver_calls) == 1
    assert artifacts[0]["key"] == workflow.ARTIFACT_KEY
    assert artifacts[0]["table"][0]["dimension"] == 1


def test_ibm_runtime_path_is_preserved(monkeypatch, h2):
    params = circuit_parameters()
    circuit = make_circuit(params, h2)
    isa = circuit.copy()
    calls = []

    async def load(name):
        calls.append(name)
        return SimpleNamespace(get_target=get_target, sampler=sample)

    async def get_target():
        return "ibm-target"

    def transpile(**kwargs):
        assert kwargs["target"] == "ibm-target"
        assert kwargs["circuit"] is circuit
        return isa

    async def sample(*, sampler_pubs, options):
        assert sampler_pubs == [(isa,)]
        assert options == {"params": {"shots": 4}}
        return sample_result(False)

    async def options(*args, **kwargs):
        return {"params": {"shots": 4}}

    monkeypatch.setattr(workflow, "get_run_logger", lambda: LOGGER)
    monkeypatch.setattr(workflow, "create_ansatz_circuits", lambda **kwargs: circuit)
    monkeypatch.setattr(workflow, "transpile_circuit", transpile)
    monkeypatch.setattr(workflow.QuantumRuntime, "load", load)
    monkeypatch.setattr(workflow.Variable, "get", options)
    result = asyncio.run(
        workflow.sample_bitstrings.fn(params, h2, "ibm", "options", "real-device", 24)
    )
    assert calls == ["ibm"]
    assert result.get_bitstrings() == sample_result(False)[0].data.meas.get_bitstrings()


def test_random_sampling_bypasses_qrmi_and_ibm(monkeypatch):
    async def options(*args, **kwargs):
        return {"params": {"shots": 4}}

    async def reject(*args, **kwargs):
        pytest.fail("Random sampling must not load a Runtime block")

    monkeypatch.setattr(workflow, "get_run_logger", lambda: LOGGER)
    monkeypatch.setattr(workflow.Variable, "get", options)
    monkeypatch.setattr(workflow.QuantumRuntime, "load", reject)
    samples = [
        asyncio.run(
            workflow.sample_bitstrings.fn(
                circuit_parameters(),
                SimpleNamespace(num_orbitals=2),
                "unused",
                "options",
                "random",
                24,
                qrmi_runtime_config={},
            )
        )
        for _ in range(2)
    ]
    assert samples[0].get_bitstrings() == samples[1].get_bitstrings()
    assert samples[0].num_bits == samples[0].num_shots == 4
