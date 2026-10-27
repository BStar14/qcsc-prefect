"""Library contracts: no Slurm, QRMI service, Prefect server or QPU calls."""

from __future__ import annotations

import asyncio
import json
import socket
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from qcsc_prefect.integrations.qiskit.qrmi import PreparedCircuit, QrmiRuntime, worker
from qcsc_prefect.integrations.qiskit.qrmi import runtime as runtime_mod
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, qpy
from qiskit.circuit import Parameter
from qiskit.primitives import PrimitiveResult, SamplerPubResult
from qiskit.primitives.containers import BitArray, DataBin
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit_ibm_runtime import RuntimeEncoder


def measured_circuit():
    circuit = QuantumCircuit(2)
    circuit.add_register(ClassicalRegister(2, "readout"), ClassicalRegister(1, "test"))
    circuit.measure(0, circuit.cregs[1][0])
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure([0, 1], circuit.cregs[0])
    return circuit


def make_result(shots=4, registers=None):
    if registers is None:
        registers = {"readout": 2, "test": 1}
    return PrimitiveResult(
        [
            SamplerPubResult(
                DataBin(
                    **{
                        name: BitArray.from_samples(["0" * width] * shots, num_bits=width)
                        for name, width in registers.items()
                    }
                ),
                metadata={},
            )
        ],
        metadata={},
    )


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def reject_network(*args, **kwargs):
        pytest.fail("QRMI library tests must not open network connections")

    monkeypatch.setattr(socket.socket, "connect", reject_network)
    monkeypatch.setattr(socket.socket, "connect_ex", reject_network)
    monkeypatch.setattr(socket, "create_connection", reject_network)
    monkeypatch.setattr(runtime_mod, "_sample_on_slurm", runtime_mod._sample_on_slurm.fn)
    monkeypatch.setattr(
        runtime_mod, "get_run_logger", lambda: SimpleNamespace(info=lambda *a: None)
    )


@pytest.fixture
def runtime(tmp_path):
    return QrmiRuntime(
        root_dir=tmp_path / "jobs",
        command_block_name="sampler-command",
        execution_profile_block_name="quantum-execution",
        hpc_profile_block_name="quantum-hpc",
        resource="custom_resource",
        timeout_seconds=120,
    )


@pytest.fixture
def submitted(monkeypatch):
    calls = []

    async def submit(**kwargs):
        calls.append(kwargs)
        request = json.loads((kwargs["work_dir"] / "sampler_request.json").read_text())
        (kwargs["work_dir"] / "result.json").write_text(
            json.dumps(make_result(shots=request["shots"]), cls=RuntimeEncoder)
        )
        return SimpleNamespace(job_id="73", state="COMPLETED", exit_status=0)

    monkeypatch.setattr(runtime_mod, "run_job_from_blocks", submit)
    return calls


def test_explicit_slurm_scheduler_submits_and_decodes_result(runtime, submitted):
    assert runtime.scheduler == "slurm"
    explicit = replace(runtime, scheduler="slurm")
    result = asyncio.run(explicit.sampler([(measured_circuit(),)], shots=4))
    assert len(submitted) == 1
    assert submitted[0]["script_filename"] == "sampler.slurm"
    assert result[0].data.readout.num_shots == 4


@pytest.mark.parametrize(
    "scheduler,exception,message",
    [
        ("iqp", NotImplementedError, 'scheduler="iqp" is not implemented'),
        ("unknown", ValueError, "Unsupported scheduler"),
        (None, ValueError, "Unsupported scheduler"),
    ],
)
def test_unsupported_scheduler_fails_before_submission(
    runtime, submitted, scheduler, exception, message
):
    with pytest.raises(exception, match=message):
        replace(runtime, scheduler=scheduler)
    # The mutable runtime must also reject a scheduler changed after construction.
    runtime.scheduler = scheduler
    with pytest.raises(exception, match=message):
        asyncio.run(runtime.sampler([(measured_circuit(),)], shots=4))
    assert submitted == []
    assert not runtime.root_dir.exists()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"shots": 4},
        {"options": {"params": {"shots": 4}}},
        {"shots": 4, "options": {"params": {"shots": 4}}},
    ],
)
def test_sampler_forwards_resource_blocks_and_timeout(runtime, submitted, kwargs):
    result = asyncio.run(runtime.sampler(sampler_pubs=[(measured_circuit(),)], **kwargs))
    assert result[0].data.readout.num_shots == 4
    assert result[0].data.test.num_bits == 1
    request = json.loads((submitted[0]["work_dir"] / "sampler_request.json").read_text())
    assert request["shots"] == 4
    assert request["resource"] == "custom_resource"
    assert submitted[0]["timeout_seconds"] == 120
    assert submitted[0]["command_block_name"] == "sampler-command"
    assert submitted[0]["execution_profile_block_name"] == "quantum-execution"
    assert submitted[0]["hpc_profile_block_name"] == "quantum-hpc"


def test_sampler_defaults_to_100_shots(runtime, submitted):
    result = asyncio.run(runtime.sampler([(measured_circuit(),)]))
    assert result[0].data.readout.num_shots == 100
    assert result[0].data.test.num_shots == 100


@pytest.mark.parametrize(
    "kwargs",
    [
        {"shots": 0},
        {"shots": -1},
        {"shots": True},
        {"shots": 1.5},
        {"shots": "4"},
        {"options": []},
        {"options": {"params": []}},
        {"options": {"params": {"shots": False}}},
        {"options": {"params": {"shots": 0}}},
        {"options": {"params": {"shots": 4}, "resilience_level": 1}},
        {"options": {"params": {"shots": 4, "other": True}}},
        {"shots": 5, "options": {"params": {"shots": 4}}},
    ],
)
def test_bad_options_fail_before_files_or_submission(runtime, submitted, kwargs):
    with pytest.raises((TypeError, ValueError)):
        asyncio.run(runtime.sampler([(measured_circuit(),)], **kwargs))
    assert submitted == []
    assert not runtime.root_dir.exists()


@pytest.mark.parametrize("kind", ["empty", "multiple", "params", "unbound", "unmeasured"])
def test_unsupported_pubs_fail_before_submission(runtime, submitted, kind):
    circuit = measured_circuit()
    pubs = [(circuit,)]
    if kind == "empty":
        pubs = []
    elif kind == "multiple":
        pubs *= 2
    elif kind == "params":
        pubs = [(circuit, [1])]
    elif kind == "unbound":
        circuit.rx(Parameter("theta"), 0)
    elif kind == "unmeasured":
        pubs = [(QuantumCircuit(1, 1),)]
    with pytest.raises(ValueError):
        asyncio.run(runtime.sampler(pubs, shots=4))
    assert submitted == []


def test_prepare_config_and_qpy_handoff_does_not_mutate_input(runtime, submitted):
    original = measured_circuit()

    def prepare(circuit):
        circuit.x(1)
        return PreparedCircuit(circuit, {"seed_transpiler": 24})

    runtime.prepare = prepare
    runtime.transpile_config = {"seed_transpiler": 1, "initial_layout": [2, 0]}
    result = asyncio.run(runtime.sampler([(original,)], shots=4))
    work_dir = Path(result.metadata["qcsc_slurm"]["work_dir"])
    with (work_dir / "circuit.qpy").open("rb") as stream:
        (restored,) = qpy.load(stream)
    assert "x" not in original.count_ops()
    assert restored.count_ops()["x"] == 1
    assert json.loads((work_dir / "transpile_config.json").read_text()) == {
        "seed_transpiler": 24,
        "initial_layout": [2, 0],
    }
    assert result.metadata["qcsc_slurm"]["job_id"] == "73"


@pytest.mark.parametrize("layout", [[2, 0], [2, 0, 1]])
def test_prepared_layout_overrides_runtime_config_and_matches_prepared_circuit(
    runtime, submitted, layout
):
    def prepare(circuit):
        if len(layout) == 3:
            circuit.add_register(QuantumRegister(1, "extra"))
        return PreparedCircuit(circuit, {"initial_layout": layout})

    runtime.prepare = prepare
    runtime.transpile_config = {"initial_layout": [0]}
    original = measured_circuit()
    asyncio.run(runtime.sampler([(original,)], shots=4))
    work_dir = submitted[0]["work_dir"]
    assert json.loads((work_dir / "transpile_config.json").read_text()) == {
        "initial_layout": layout
    }
    with (work_dir / "circuit.qpy").open("rb") as stream:
        (restored,) = qpy.load(stream)
    assert restored.num_qubits == len(layout)
    assert original.num_qubits == 2


@pytest.mark.parametrize("layout,num_qubits", [([0], 2), ([0, 1, 2], 2), ([0, 1], 3)])
def test_invalid_prepared_layout_fails_before_submission(runtime, submitted, layout, num_qubits):
    def prepare(circuit):
        if num_qubits == 3:
            circuit.add_register(QuantumRegister(1, "extra"))
        return PreparedCircuit(circuit, {"initial_layout": layout})

    runtime.prepare = prepare
    runtime.transpile_config = {"initial_layout": [0, 1]}
    with pytest.raises(ValueError, match=f"expected {num_qubits}, got {len(layout)}"):
        asyncio.run(runtime.sampler([(measured_circuit(),)], shots=4))
    assert submitted == []
    assert not runtime.root_dir.exists()


@pytest.mark.parametrize(
    "config",
    [
        {"initial_layout": [0, 0]},
        {"initial_layout": [True, 1]},
        {"initial_layout": [-1, 1]},
        {"initial_layout": {"0": 1}},
        {"initial_layout": []},
        {"initial_layout": [0]},
        {"initial_layout": [0, 1, 2]},
        {"optimization_level": True},
        {"optimization_level": 4},
        {"optimization_level": 1.0},
        {"seed_transpiler": -1},
        {"seed_transpiler": True},
        {"seed_transpiler": 1.5},
        {"initial_layot": [0, 1]},
        {"target": None},
        {"backend": None},
        {"coupling_map": [[0, 1]]},
        {"basis_gates": ["cx"]},
        {"approximation_degree": -0.1},
        {"approximation_degree": 1.1},
        {"approximation_degree": True},
        {"approximation_degree": "0.5"},
        {"approximation_degree": []},
        {"approximation_degree": float("nan")},
        {"approximation_degree": float("inf")},
        {"qubits_initially_zero": 0},
        {"qubits_initially_zero": 1},
        {"qubits_initially_zero": None},
        {"qubits_initially_zero": "true"},
        {"unitary_synthesis_plugin_config": []},
        {"unitary_synthesis_plugin_config": "config"},
        {"unitary_synthesis_plugin_config": False},
        *[
            {method: value}
            for method in (
                "layout_method",
                "routing_method",
                "translation_method",
                "scheduling_method",
                "init_method",
                "optimization_method",
                "unitary_synthesis_method",
            )
            for value in (1, [], {})
        ],
    ],
)
def test_invalid_preset_options_fail_before_submission(runtime, submitted, config):
    runtime.transpile_config = config
    with pytest.raises(ValueError):
        asyncio.run(runtime.sampler([(measured_circuit(),)], shots=4))
    assert submitted == []
    assert not runtime.root_dir.exists()
    # The compute node also validates files independently of the workflow host.
    with pytest.raises(ValueError):
        worker.generic_transpile(measured_circuit(), None, config)


@pytest.mark.parametrize(
    "config",
    [
        {"initial_layout": None},
        {"approximation_degree": None},
        {"approximation_degree": 0},
        {"approximation_degree": 1},
        {"approximation_degree": 0.5},
        {"qubits_initially_zero": True},
        {"qubits_initially_zero": False},
        {"unitary_synthesis_plugin_config": None},
        {"unitary_synthesis_plugin_config": {"plugin_option": [1, "value"]}},
        *[
            {method: value}
            for method in (
                "layout_method",
                "routing_method",
                "translation_method",
                "scheduling_method",
                "init_method",
                "optimization_method",
                "unitary_synthesis_method",
            )
            for value in (None, "node_plugin")
        ],
    ],
)
def test_valid_preset_types_and_boundaries_are_forwarded(runtime, submitted, config):
    # Plugin existence and plugin-specific configuration remain compute-node checks.
    runtime.transpile_config = config
    asyncio.run(runtime.sampler([(measured_circuit(),)], shots=4))
    assert json.loads((submitted[0]["work_dir"] / "transpile_config.json").read_text()) == config


def test_generic_transpile_applies_physical_layout_and_preserves_registers():
    target = GenericBackendV2(num_qubits=3, basis_gates=["rz", "sx", "x", "cx"]).target
    circuit = measured_circuit()
    config = {"initial_layout": [2, 0], "optimization_level": 0, "seed_transpiler": 24}
    isa = worker.generic_transpile(circuit, target, config)
    worker.validate_isa(isa, target)
    assert isa.layout.initial_index_layout(filter_ancillas=True) == [2, 0]
    assert [(reg.name, len(reg)) for reg in isa.cregs] == [("readout", 2), ("test", 1)]
    measurements = {
        isa.find_bit(inst.clbits[0]).index: isa.find_bit(inst.qubits[0]).index
        for inst in isa.data
        if inst.name == "measure"
    }
    assert measurements == {0: 2, 1: 0, 2: 2}
    assert circuit.num_qubits == 2
    assert config == {"initial_layout": [2, 0], "optimization_level": 0, "seed_transpiler": 24}


@pytest.mark.parametrize("custom", [False, True])
def test_configured_compilation_through_worker(runtime, monkeypatch, custom):
    target = GenericBackendV2(num_qubits=3, basis_gates=["rz", "sx", "x", "cx"]).target
    resource = SimpleNamespace(resource_id=lambda: runtime.resource)
    qrmi = ModuleType("qrmi")
    qrmi.QRMIService = lambda: SimpleNamespace(resources=lambda: [resource])
    ibm = ModuleType("qrmi.primitives.ibm")
    ibm.get_target = lambda acquired: target if acquired is resource else None
    monkeypatch.setitem(sys.modules, "qrmi", qrmi)
    monkeypatch.setitem(sys.modules, "qrmi.primitives", ModuleType("qrmi.primitives"))
    monkeypatch.setitem(sys.modules, "qrmi.primitives.ibm", ibm)
    monkeypatch.setenv(f"{runtime.resource}_QRMI_IBM_QCS_SESSION_MODE", "batch")
    monkeypatch.setenv(f"{runtime.resource}_QRMI_JOB_ACQUISITION_TOKEN", "test-token")
    runtime.transpile_config = {"initial_layout": [2, 0], "seed_transpiler": 24}
    if custom:
        hook = ModuleType("custom_compilation")

        def transpile(circuit, actual_target, config):
            from qiskit.transpiler import PassManager
            from qiskit.transpiler.passes import RemoveBarriers

            assert actual_target is target
            config = dict(config)
            assert config.pop("algorithm_option") == "custom"
            pm = worker.generate_preset_pass_manager(target=actual_target, **config)
            pm.pre_init = PassManager([RemoveBarriers()])
            return pm.run(circuit)

        hook.transpile = transpile
        monkeypatch.setitem(sys.modules, "custom_compilation", hook)
        runtime.transpiler = "custom_compilation:transpile"
        runtime.transpile_config["algorithm_option"] = "custom"

    def execute(isa, shots, executable, resource_id):
        assert resource_id == runtime.resource
        assert isa.layout.initial_index_layout(filter_ancillas=True) == [2, 0]
        if custom:
            assert "barrier" not in isa.count_ops()
        Path("result.json").write_text(json.dumps(make_result(shots), cls=RuntimeEncoder))

    async def submit(**kwargs):
        monkeypatch.chdir(kwargs["work_dir"])
        worker.main(kwargs["user_args"])
        return SimpleNamespace(job_id="73", state="COMPLETED", exit_status=0)

    monkeypatch.setattr(worker, "execute_task_runner", execute)
    monkeypatch.setattr(runtime_mod, "run_job_from_blocks", submit)
    circuit = measured_circuit()
    circuit.barrier()
    result = asyncio.run(runtime.sampler([(circuit,)], shots=4))
    assert result[0].data.readout.num_shots == 4
    work_dir = Path(result.metadata["qcsc_slurm"]["work_dir"])
    report = json.loads((work_dir / "transpile_report.json").read_text())
    assert report["initial_layout"] == [2, 0]
    assert report["transpiler"] == runtime.transpiler
    if custom:
        assert report["algorithm_option"] == "custom"


def test_prepare_cannot_change_registers(runtime, submitted):
    runtime.prepare = lambda _: PreparedCircuit(QuantumCircuit(1))
    with pytest.raises(ValueError):
        asyncio.run(runtime.sampler([(measured_circuit(),)], shots=4))
    assert submitted == []


@pytest.mark.parametrize(
    "registers,shots,message",
    [
        ({"readout": 2}, 4, "registers"),
        ({"readout": 2, "test": 2}, 4, "1 bits for test"),
        ({"readout": 2, "test": 1}, 3, "3 shots; expected 4"),
    ],
)
def test_host_rejects_invalid_results_and_retains_evidence(
    runtime, monkeypatch, registers, shots, message
):
    async def submit(**kwargs):
        (kwargs["work_dir"] / "result.json").write_text(
            json.dumps(make_result(shots, registers), cls=RuntimeEncoder)
        )
        return SimpleNamespace(job_id="73", state="COMPLETED", exit_status=0)

    monkeypatch.setattr(runtime_mod, "run_job_from_blocks", submit)
    with pytest.raises(RuntimeError, match=message):
        asyncio.run(runtime.sampler([(measured_circuit(),)], shots=4))
    (work_dir,) = runtime.root_dir.iterdir()
    assert (work_dir / "result.json").is_file()
    assert json.loads((work_dir / "slurm_job.json").read_text())["job_id"] == "73"


def test_worker_validates_every_register_even_when_cli_exits_zero(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def zero_exit_with_missing_test(*args, **kwargs):
        Path("result.json").write_text(
            json.dumps(make_result(registers={"readout": 2}), cls=RuntimeEncoder)
        )

    monkeypatch.setattr(worker.subprocess, "run", zero_exit_with_missing_test)
    with pytest.raises(RuntimeError, match="registers"):
        worker.execute_task_runner(measured_circuit(), 4, Path("task_runner"), "custom_resource")


@pytest.mark.parametrize(
    "state,exit_status,has_result",
    [
        ("FAILED", 1, True),
        ("CANCELLED", 0, True),
        ("COMPLETED", 1, True),
        ("COMPLETED", 0, False),
    ],
)
def test_failed_or_missing_results_preserve_job_evidence(
    runtime, monkeypatch, state, exit_status, has_result
):
    async def submit(**kwargs):
        if has_result:
            (kwargs["work_dir"] / "result.json").write_text(
                json.dumps(make_result(), cls=RuntimeEncoder)
            )
        return SimpleNamespace(job_id="73", state=state, exit_status=exit_status)

    monkeypatch.setattr(runtime_mod, "run_job_from_blocks", submit)
    with pytest.raises(RuntimeError):
        asyncio.run(runtime.sampler([(measured_circuit(),)], shots=4))
    (job_dir,) = runtime.root_dir.iterdir()
    record = json.loads((job_dir / "slurm_job.json").read_text())
    assert record["job_id"] == "73"
    assert record["state"] == state
    assert (job_dir / "result.json").exists() == has_result


@pytest.mark.parametrize("mode", ["", "dedicated"])
def test_worker_requires_batch_mode_for_selected_resource(monkeypatch, mode):
    monkeypatch.setenv("custom_resource_QRMI_IBM_QCS_SESSION_MODE", mode)
    with pytest.raises(RuntimeError, match="custom_resource.*SESSION_MODE"):
        worker.require_batch_environment("custom_resource")


def test_worker_requires_acquisition_token_for_selected_resource(monkeypatch):
    monkeypatch.setenv("custom_resource_QRMI_IBM_QCS_SESSION_MODE", "batch")
    monkeypatch.delenv("custom_resource_QRMI_JOB_ACQUISITION_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="acquisition token"):
        worker.require_batch_environment("custom_resource")
    monkeypatch.setenv("custom_resource_QRMI_JOB_ACQUISITION_TOKEN", "offline-test-token")
    worker.require_batch_environment("custom_resource")


def test_generic_handoff_with_custom_resource_and_registers(runtime, tmp_path, monkeypatch):
    pytest.importorskip(
        "qiskit_qasm3_import", reason="Install qiskit-qasm3-import to test QASM import"
    )
    resource = SimpleNamespace(resource_id=lambda: "custom_resource")
    qrmi = ModuleType("qrmi")
    qrmi.QRMIService = lambda: SimpleNamespace(resources=lambda: [resource])
    ibm = ModuleType("qrmi.primitives.ibm")
    target = GenericBackendV2(num_qubits=3, basis_gates=["rz", "sx", "x", "cx"]).target
    ibm.get_target = lambda _: target
    monkeypatch.setitem(sys.modules, "qrmi", qrmi)
    monkeypatch.setitem(sys.modules, "qrmi.primitives", ModuleType("qrmi.primitives"))
    monkeypatch.setitem(sys.modules, "qrmi.primitives.ibm", ibm)
    monkeypatch.setenv("custom_resource_QRMI_IBM_QCS_SESSION_MODE", "batch")
    monkeypatch.setenv("custom_resource_QRMI_JOB_ACQUISITION_TOKEN", "test-token")
    runner = tmp_path / "fake_task_runner"
    runner.write_text(
        f"#!{sys.executable}\n"
        """import json, socket, sys
def reject(*args, **kwargs):
    raise RuntimeError('No network in the fake CLI')
socket.socket.connect = reject
socket.socket.connect_ex = reject
socket.create_connection = reject
from pathlib import Path
from qiskit import qasm3
from qiskit.primitives import PrimitiveResult, SamplerPubResult
from qiskit.primitives.containers import BitArray, DataBin
from qiskit_ibm_runtime import RuntimeEncoder
resource, source, destination = sys.argv[1:]
assert resource == 'custom_resource'
parameters = json.loads(Path(source).read_text())['parameters']
assert parameters['support_qiskit'] is True
qasm, values, shots = parameters['pubs'][0]
assert values is None and shots == 4
circuit = qasm3.loads(qasm)
assert {r.name: len(r) for r in circuit.cregs} == {'readout': 2, 'test': 1}
data = {r.name: BitArray.from_samples(['0' * len(r)] * shots, num_bits=len(r))
        for r in circuit.cregs}
result = PrimitiveResult([SamplerPubResult(DataBin(**data), metadata={})], metadata={})
Path(destination).write_text(json.dumps(result, cls=RuntimeEncoder))
"""
    )
    runner.chmod(0o700)

    async def submit(**kwargs):
        monkeypatch.chdir(kwargs["work_dir"])
        worker.main([*kwargs["user_args"], "--task-runner", str(runner)])
        return SimpleNamespace(job_id="73", state="COMPLETED", exit_status=0)

    monkeypatch.setattr(runtime_mod, "run_job_from_blocks", submit)
    result = asyncio.run(runtime.sampler([(measured_circuit(),)], shots=4))
    work_dir = Path(result.metadata["qcsc_slurm"]["work_dir"])
    report = json.loads((work_dir / "transpile_report.json").read_text())
    assert report["resource"] == "custom_resource"
    assert report["registers"] == {"readout": 2, "test": 1}
    assert report["target_num_qubits"] == 3
    assert result[0].data.readout.get_counts() == {"00": 4}


def test_worker_rejects_unknown_schema_before_qrmi_import(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("sampler_request.json").write_text('{"schema_version": 2}')
    with pytest.raises(ValueError, match="schema_version"):
        worker.main([])


def test_repeated_calls_create_distinct_jobs(runtime, submitted):
    asyncio.run(runtime.sampler([(measured_circuit(),)], shots=4))
    asyncio.run(runtime.sampler([(measured_circuit(),)], shots=4))
    assert len(submitted) == 2
    assert submitted[0]["work_dir"] != submitted[1]["work_dir"]


def test_library_imports_without_compute_node_or_executor_dependencies():
    import subprocess
    import sys

    program = """
import importlib.abc, sys
class Reject(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'examples', 'qrmi', 'qcsc_prefect_executor'}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, Reject())
from qcsc_prefect.integrations.qiskit.qrmi import QrmiRuntime
from qcsc_prefect.integrations.qiskit.qrmi.worker import main
"""
    # Test source packages in a fresh interpreter, without inherited imports.
    import os

    src = Path(__file__).parents[1] / "src"
    env = {**os.environ, "PYTHONPATH": str(src)}
    subprocess.run([sys.executable, "-c", program], env=env, check=True, capture_output=True)
