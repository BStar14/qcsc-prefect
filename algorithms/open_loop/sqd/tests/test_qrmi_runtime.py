"""Offline checks of the actual SQD workflow's optional QRMI boundary."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import socket
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from qrmi_stubs import QrmiBlockStub

np = pytest.importorskip("numpy")
pytest.importorskip("ffsim")
pytest.importorskip("pydantic_numpy")
pytest.importorskip("prefect_qiskit")
pytest.importorskip("qcsc_prefect.integrations.qiskit.qrmi")

from pyscf import lib  # noqa: E402
from qcsc_prefect.integrations.qiskit.qrmi import runtime as runtime_mod  # noqa: E402
from qcsc_prefect.integrations.qiskit.qrmi.worker import validate_isa  # noqa: E402
from qcsc_prefect_executor import from_blocks  # noqa: E402
from qcsc_workflow_utility import chem  # noqa: E402
from qiskit import QuantumCircuit, qpy  # noqa: E402
from qiskit.primitives import PrimitiveResult, SamplerPubResult  # noqa: E402
from qiskit.primitives.containers import BitArray, DataBin  # noqa: E402
from qiskit.providers.fake_provider import GenericBackendV2  # noqa: E402
from qiskit.quantum_info import Statevector  # noqa: E402
from qiskit_ibm_runtime import RuntimeEncoder  # noqa: E402
from sqd_common import flow, lucj, postprocess, sampling, topology  # noqa: E402
from sqd_common.advanced import AdvancedSettings  # noqa: E402
from sqd_common.flow_params import (  # noqa: E402
    CircuitParameters,
    FlowParameters,
    SamplingParameters,
)
from sqd_common.main import ARTIFACT_KEY, sqd  # noqa: E402
from sqd_common.qrmi import TRANSPILER, make_qrmi_runtime, prepare_sampler_circuit  # noqa: E402
from sqd_common.transpile import transpile_for_target, two_qubit_depth  # noqa: E402

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
    """Advanced SQD Settings with reset mitigation and a small SABRE search."""
    return AdvancedSettings(
        use_reset_mitigation=reset,
        sabre_layout_trials=4,
        sabre_swap_trials=2,
        sabre_max_iterations=2,
    )


def lucj_parameters(params, h2):
    with patch.object(lucj, "get_run_logger", return_value=LOGGER):
        return lucj.get_lucj_params.fn(CircuitParameters(), h2, "heavy-hex")


def make_circuit(params, h2):
    with patch.object(lucj, "get_run_logger", return_value=LOGGER):
        return lucj.build_lucj_circuit.fn(
            lucj_parameters(params, h2), use_reset_mitigation=params.use_reset_mitigation
        )


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
    config = lucj.circuit_transpile_options(params, np.random.default_rng(24))
    prepared = prepare_sampler_circuit(original.copy(), transpile_config=config)
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


def test_node_hook_imports_neither_ffsim_nor_prefect():
    """The compute node needs only Qiskit and qiskit-ibm-runtime for the hook."""
    sqd_root = Path(lucj.__file__).resolve().parents[1]
    code = (
        "import sys; import sqd_common.transpile; "
        "heavy = {'ffsim', 'prefect', 'prefect_qiskit', 'qcsc_prefect_fulqrum', "
        "'qcsc_workflow_utility', 'pyscf', 'qiskit_addon_sqd'}; "
        "print(sorted(heavy & {m.split('.')[0] for m in sys.modules}))"
    )
    # -B: do not write __pycache__ into the source tree.
    out = subprocess.run(
        [sys.executable, "-B", "-c", code],
        cwd=sqd_root,
        env={"PYTHONPATH": str(sqd_root)},
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == "[]"


@pytest.mark.parametrize("reset", [False, True])
@pytest.mark.parametrize("with_rzz", [False, True])
def test_node_hook_matches_host_transpile(h2, reset, with_rzz):
    """The QRMI hook compiles exactly as the in-process ``transpile_lucj_circuit`` does."""
    params = circuit_parameters(reset)
    original = make_circuit(params, h2)
    actual_target = target(with_rzz)
    with patch.object(lucj, "get_run_logger", return_value=LOGGER):
        host = lucj.transpile_lucj_circuit.fn(
            circuit=original.copy(),
            target=actual_target,
            advanced=params,
            seed=np.random.default_rng(24),
        )
    config = lucj.circuit_transpile_options(params, np.random.default_rng(24))
    prepared = prepare_sampler_circuit(original.copy(), transpile_config=config)
    stream = io.BytesIO()
    qpy.dump(prepared.circuit, stream)
    stream.seek(0)
    (restored,) = qpy.load(stream)
    node = transpile_for_target(restored, actual_target, prepared.config)
    assert dict(node.count_ops()) == dict(host.count_ops())
    assert node.depth() == host.depth()
    assert node.layout.final_index_layout() == host.layout.final_index_layout()


def test_two_qubit_depth_counts_only_two_qubit_gates():
    circuit = QuantumCircuit(3, 3)
    circuit.h(0)
    circuit.cz(0, 1)
    circuit.sx(1)
    circuit.rz(0.1, 2)
    circuit.cz(1, 2)
    circuit.cz(0, 2)
    circuit.barrier()
    circuit.measure(range(3), range(3))
    # The single-qubit gates, the 3-qubit barrier and the measurements do not count.
    assert (two_qubit_depth(circuit), circuit.depth()) == (3, 6)


def test_node_hook_rejects_unexpected_config():
    with pytest.raises(ValueError, match="unknown=\\['circuit_parameters'\\]"):
        transpile_for_target(None, target(), {"circuit_parameters": {}})


def slurm_runtime(tmp_path, **changes):
    """The QrmiRuntime a ``scheduler="slurm"`` QrmiRuntimeBlock builds."""
    settings = dict(
        root_dir=tmp_path / "quantum",
        command_block_name="qrmi-command",
        execution_profile_block_name="qrmi-execution",
        hpc_profile_block_name="qrmi-hpc",
        resource="test_heron",
        transpile_config={"optimization_level": 1},
    )
    return runtime_mod.QrmiRuntime(**{**settings, **changes})


def test_runtime_uses_sqd_hooks_and_keeps_the_block_settings(tmp_path):
    block_runtime = slurm_runtime(tmp_path, watch_poll_interval=7.0)
    runtime = make_qrmi_runtime(block_runtime, transpile_config={"seed_transpiler": 1})
    assert runtime.transpiler == TRANSPILER
    assert runtime.transpile_config == {}
    assert runtime.prepare.keywords == {"transpile_config": {"seed_transpiler": 1}}
    assert (runtime.resource, runtime.watch_poll_interval) == ("test_heron", 7.0)
    assert block_runtime.transpile_config == {"optimization_level": 1}  # not modified


def sample_result(reset):
    data = {"meas": BitArray.from_samples(["0101", "0110", "1001", "1010"], num_bits=4)}
    if reset:
        data["test"] = BitArray.from_samples(["0000", "0001", "0000", "0010"], num_bits=4)
    return PrimitiveResult([SamplerPubResult(DataBin(**data), metadata={})], metadata={})


@pytest.mark.parametrize("reset", [False, True])
def test_sampler_qrmi_qpy_handoff_and_reset_postselection(tmp_path, monkeypatch, caplog, h2, reset):
    params = circuit_parameters(reset)
    calls = []
    compiled = []

    async def submit(**kwargs):
        calls.append(kwargs)
        work_dir = kwargs["work_dir"]
        request = json.loads((work_dir / "sampler_request.json").read_text())
        assert request["transpiler"] == "sqd_common.transpile:transpile_for_target"
        assert request["resource"] == "test_heron"
        assert request["shots"] == 4
        (hook_config,) = json.loads((work_dir / "transpile_config.json").read_text())
        assert hook_config == lucj.circuit_transpile_options(params, np.random.default_rng(24))
        with (work_dir / "circuit.qpy").open("rb") as stream:
            (circuit,) = qpy.load(stream)
        actual_target = target()
        isa = transpile_for_target(circuit, actual_target, hook_config)
        validate_isa(isa, actual_target)
        compiled.append(isa)
        with (work_dir / "isa.qpy").open("wb") as stream:
            qpy.dump(isa, stream)
        (work_dir / "result.json").write_text(json.dumps(sample_result(reset), cls=RuntimeEncoder))
        return SimpleNamespace(job_id="73", state="COMPLETED", exit_status=0)

    async def reject_ibm(*args, **kwargs):
        pytest.fail("QRMI sampling must not load an IBM Runtime block")

    async def load_block(name):
        assert name == "qrmi-runtime"
        return QrmiBlockStub(slurm_runtime(tmp_path))

    artifacts = []

    async def artifact(**kwargs):
        artifacts.append(kwargs)

    monkeypatch.setattr(runtime_mod, "_sample_on_slurm", runtime_mod._sample_on_slurm.fn)
    monkeypatch.setattr(runtime_mod, "get_run_logger", lambda: LOGGER)
    monkeypatch.setattr(runtime_mod, "run_job_from_blocks", submit)

    async def load_profile(block_cls, name):
        assert name == "qrmi-execution"
        return SimpleNamespace(walltime="12:00:00")

    monkeypatch.setattr(from_blocks, "_load_block", load_profile)
    for module in (lucj, sampling, postprocess, flow, topology):
        monkeypatch.setattr(module, "get_run_logger", lambda: LOGGER)
    monkeypatch.setattr(sampling, "build_lucj_circuit", lucj.build_lucj_circuit.fn)
    monkeypatch.setattr(sampling, "create_table_artifact", artifact)
    monkeypatch.setattr(postprocess, "create_table_artifact", artifact)
    monkeypatch.setattr(sampling.QuantumRuntime, "load", reject_ibm)
    monkeypatch.setattr(sampling, "load_qrmi_block", load_block)
    monkeypatch.setattr(postprocess.flow_run, "get_id", lambda: "run")
    sampling_params = SamplingParameters(shots=4)
    with caplog.at_level(logging.INFO, logger=LOGGER.name):
        result = asyncio.run(
            sampling.quantum_sampling.fn(lucj_parameters(params, h2), params, sampling_params)
        )
    assert BitArray.from_bool_array(result).get_bitstrings() == (
        ["0101", "1001"] if reset else ["0101", "0110", "1001", "1010"]
    )
    assert len(calls) == 1
    assert calls[0]["command_block_name"] == "qrmi-command"
    assert f"Instruction counts = {dict(compiled[0].count_ops())}" in caplog.text
    assert f"Two-qubit depth = {two_qubit_depth(compiled[0])}\n" in caplog.text
    assert "Compiled on the compute node:" in caplog.text
    assert "shots=4, walltime=12:00:00)." in caplog.text
    assert "Quantum sampling completed on test_heron (Slurm job 73)." in caplog.text
    ((source),) = artifacts
    assert source["key"] == sampling.SAMPLING_ARTIFACT_KEY
    row = source["table"][0]
    assert row["source"] == "qrmi-slurm" and row["resource"] == "test_heron"
    assert row["qrmi_result_dir"] == str(calls[0]["work_dir"]) and row["slurm_job_id"] == "73"
    assert (row["shots"], row["shots_kept"]) == (4, len(result))

    # The saved job directory reloads to the same samples, without a new job.
    reloaded = asyncio.run(
        sampling.load_qrmi_result.fn(row["qrmi_result_dir"], params.use_reset_mitigation)
    )
    np.testing.assert_array_equal(reloaded, result)
    assert len(calls) == 1

    # Exercise the flow body and recovery loop. Only scheduler/native
    # execution and Prefect orchestration boundaries are replaced.
    solver_calls = []

    async def solve(**kwargs):
        assert kwargs["one_body_tensor"] is h2.one_body_tensor
        assert kwargs["two_body_tensor"] is h2.two_body_tensor
        assert kwargs["nelec"] == h2.num_electrons
        solver_calls.append(kwargs)
        return SimpleNamespace(energy=-1.5, orbital_occupancies=h2.initial_occupancy)

    async def load_solver(name):
        assert name == "sqd-fulqrum-solver"
        return SimpleNamespace(run=solve, root_dir=str(tmp_path))

    class DirectTask:
        def __init__(self, fn):
            self.fn = fn

        def with_options(self, **kwargs):
            return self.fn

        def __call__(self, **kwargs):
            return self.fn(**kwargs)

    monkeypatch.setattr(
        flow, "compute_molecular_integrals_from_geometry", DirectTask(lambda **kwargs: h2)
    )
    for name in ("get_lucj_params", "quantum_sampling", "sqd_loop"):
        monkeypatch.setattr(flow, name, DirectTask(getattr(flow, name).fn))
    monkeypatch.setattr(
        flow, "solver_class", lambda slug: SimpleNamespace(load=load_solver, __name__=slug)
    )

    async def advanced_block(name, logger):
        return params

    monkeypatch.setattr(flow, "load_advanced_settings", advanced_block)
    artifacts.clear()
    parameters = FlowParameters(
        molecule={"atom": "H 0 0 0; H 0 0 0.74", "basis": "sto-3g"},
        sampling=sampling_params,
        sqd={"subspace_dim": 1, "num_batches": 1, "max_iterations": 1},
    )
    energy = asyncio.run(sqd.fn(parameters))
    assert energy == pytest.approx(-1.5 + h2.nuclear_repulsion_energy)
    assert len(calls) == 2
    assert len(solver_calls) == 1
    keys = [a["key"] for a in artifacts]
    assert keys == [
        sampling.SAMPLING_ARTIFACT_KEY,
        ARTIFACT_KEY,
        postprocess.CHECKPOINT_ARTIFACT_KEY,
    ]
    assert artifacts[1]["table"][0]["dimension"] == 1


def test_pre_init_cannot_run_after_qpy_so_prepare_runs_it_first(h2):
    # QPY turns the ffsim gates into plain gates, which the ffsim pre-init stage rejects.
    from ffsim.qiskit import PRE_INIT

    params = circuit_parameters(reset=True)
    circuit = make_circuit(params, h2)
    stream = io.BytesIO()
    qpy.dump(circuit, stream)
    stream.seek(0)
    (restored,) = qpy.load(stream)
    with pytest.raises(AttributeError):
        PRE_INIT.run(restored)
    prepared = prepare_sampler_circuit(circuit, transpile_config={}).circuit
    assert "ucj_balanced_jw" in circuit.count_ops()
    assert not any(name.startswith("ucj") for name in prepared.count_ops())
    assert [(reg.name, len(reg)) for reg in prepared.cregs] == [
        (reg.name, len(reg)) for reg in circuit.cregs
    ]


def test_the_run_form_has_no_qpu_backend():
    assert "qpu_resource" not in SamplingParameters.model_fields
    # A run form saved before keeps the field hidden; empty is ignored, a value fails.
    assert SamplingParameters(qpu_resource="").shots == 100_000
    assert SamplingParameters(qpu_resource=None).shots == 100_000
    with pytest.raises(ValueError, match="set QPU Backend in the QRMI Runtime block"):
        SamplingParameters(qpu_resource="ibm_kingston")


def test_runtime_table_rejects_a_secret_conflicts_and_typos(monkeypatch):
    from sqd_common import runtime_block

    monkeypatch.setenv("QRMI_IBM_QCS_IAM_APIKEY", "k")
    base = {"scheduler": "iqp", "resource": "ibm_test"}
    with pytest.raises(RuntimeError, match="Do not write QRMI_IBM_QCS_IAM_APIKEY"):
        runtime_block.runtime_block_settings({**base, "QRMI_IBM_QCS_IAM_APIKEY": "x"})
    with pytest.raises(RuntimeError, match="differ; set only service_crn"):
        runtime_block.runtime_block_settings(
            {**base, "service_crn": "a", "qrmi": {"QRMI_IBM_QCS_SERVICE_CRN": "b"}}
        )
    with pytest.raises(RuntimeError, match="Unknown \\[runtime\\] settings: service_cr"):
        runtime_block.build_runtime_block({**base, "service_cr": "a"})


LOGIN_BLOCK = dict(
    scheduler="slurm",
    resource="ibm_kingston",
    instance="open-instance (open)",
    instances={"main-instance (premium)": "crn:v1:main", "open-instance (open)": "crn:v1:open"},
    instance_backends={
        "main-instance (premium)": ["ibm_fez", "ibm_kingston"],
        "open-instance (open)": ["ibm_kingston"],
    },
    root_dir="/data/sqd/qrmi",
    command_block_name="cmd-sqd-qrmi",
    execution_profile_block_name="exec-sqd-qrmi",
    hpc_profile_block_name="hpc-slurm-sqd-qrmi",
)


def _saved_runtime(monkeypatch, saved_block):
    from sqd_common import runtime_block

    saved = {}
    monkeypatch.setattr(runtime_block, "_saved_block", lambda name: saved_block)

    def capture(self, name, overwrite=False):
        saved[name] = self

    from qcsc_prefect.integrations.qiskit.qrmi import QrmiRuntimeBlock

    monkeypatch.setattr(QrmiRuntimeBlock, "save", capture)
    return runtime_block, saved


def test_create_blocks_keeps_the_login_choice_and_dropdowns(monkeypatch):
    from qcsc_prefect.integrations.qiskit.qrmi import QrmiRuntimeBlock

    runtime_block, saved = _saved_runtime(monkeypatch, QrmiRuntimeBlock(**LOGIN_BLOCK))
    # A [qrmi] table that sets no backend or instance, with a new work directory.
    settings = {
        "scheduler": "slurm",
        "root_dir": "/data/new/qrmi",
        "command_block_name": "cmd-sqd-qrmi",
        "execution_profile_block_name": "exec-sqd-qrmi",
        "hpc_profile_block_name": "hpc-slurm-sqd-qrmi",
    }
    assert runtime_block.save_runtime_block(settings, "qrmi-runtime") == []
    block = saved["qrmi-runtime"]
    assert (block.instance, block.resource, block.root_dir) == (
        "open-instance (open)",
        "ibm_kingston",
        "/data/new/qrmi",
    )
    schema = type(block).model_json_schema()["properties"]
    assert schema["resource"]["enum"] == ["ibm_fez", "ibm_kingston"]
    assert schema["instance"]["enum"] == list(LOGIN_BLOCK["instances"])


def test_create_blocks_prints_a_changed_backend_or_instance(monkeypatch):
    from qcsc_prefect.integrations.qiskit.qrmi import QrmiRuntimeBlock

    runtime_block, saved = _saved_runtime(monkeypatch, QrmiRuntimeBlock(**LOGIN_BLOCK))
    changes = runtime_block.save_runtime_block(
        {"scheduler": "slurm", "resource": "ibm_fez", "instance": "main-instance"}, "qrmi-runtime"
    )
    assert changes == [
        "QPU Backend: ibm_kingston -> ibm_fez",
        "IBM Quantum Instance: open-instance (open) -> main-instance",
    ]
    # The bare instance name selects its 'name (plan)' label.
    assert saved["qrmi-runtime"].instance == "main-instance (premium)"
    with pytest.raises(RuntimeError, match="is not one of"):
        runtime_block.save_runtime_block({"instance": "elsewhere"}, "qrmi-runtime")


def test_a_first_block_needs_a_backend(monkeypatch):
    runtime_block, _ = _saved_runtime(monkeypatch, None)
    with pytest.raises(RuntimeError, match="Run qcsc-qrmi login, or set resource"):
        runtime_block.save_runtime_block({"scheduler": "slurm", "root_dir": "/x"}, "qrmi-runtime")
