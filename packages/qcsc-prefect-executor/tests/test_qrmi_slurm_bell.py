"""Exercise the Bell example through QPY, the worker and a fake CLI, offline."""

from __future__ import annotations

import asyncio
import json
import socket
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

pytest.importorskip("qiskit_ibm_runtime")

from qcsc_prefect.integrations.qiskit.qrmi import runtime as runtime_mod  # noqa: E402
from qcsc_prefect.integrations.qiskit.qrmi import worker  # noqa: E402
from qiskit.providers.fake_provider import GenericBackendV2  # noqa: E402

from examples.qrmi_slurm_bell.flow import bell_flow  # noqa: E402


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def reject(*args, **kwargs):
        pytest.fail("Bell example tests must not open network connections")

    monkeypatch.setattr(socket.socket, "connect", reject)
    monkeypatch.setattr(socket.socket, "connect_ex", reject)
    monkeypatch.setattr(socket, "create_connection", reject)
    monkeypatch.setattr(runtime_mod, "_sample_on_slurm", runtime_mod._sample_on_slurm.fn)
    monkeypatch.setattr(
        runtime_mod, "get_run_logger", lambda: SimpleNamespace(info=lambda *args: None)
    )


@pytest.mark.parametrize("shots", [5, 100])
def test_bell_example_round_trip(tmp_path, monkeypatch, shots):
    resource = SimpleNamespace(resource_id=lambda: "bell_resource")
    qrmi = ModuleType("qrmi")
    qrmi.QRMIService = lambda: SimpleNamespace(resources=lambda: [resource])
    ibm = ModuleType("qrmi.primitives.ibm")
    target = GenericBackendV2(num_qubits=2, basis_gates=["rz", "sx", "x", "cx"], seed=42).target
    ibm.get_target = lambda _: target
    monkeypatch.setitem(sys.modules, "qrmi", qrmi)
    monkeypatch.setitem(sys.modules, "qrmi.primitives", ModuleType("qrmi.primitives"))
    monkeypatch.setitem(sys.modules, "qrmi.primitives.ibm", ibm)
    monkeypatch.setenv("bell_resource_QRMI_IBM_QCS_SESSION_MODE", "batch")
    monkeypatch.setenv("bell_resource_QRMI_JOB_ACQUISITION_TOKEN", "offline-test-token")

    executable = tmp_path / "fake_task_runner"
    executable.write_text(
        f"#!{sys.executable}\n"
        """import json, socket, sys
def reject(*args, **kwargs):
    raise RuntimeError('No network in the fake CLI')
socket.socket.connect = reject
socket.socket.connect_ex = reject
socket.create_connection = reject
from pathlib import Path
from qiskit import qpy
from qiskit.quantum_info import Statevector
from qiskit.primitives import PrimitiveResult, SamplerPubResult
from qiskit.primitives.containers import BitArray, DataBin
from qiskit_ibm_runtime import RuntimeEncoder
resource, source, destination = sys.argv[1:]
assert resource == 'bell_resource'
payload = json.loads(Path(source).read_text())
assert payload['program_id'] == 'sampler'
params = payload['parameters']
assert params['version'] == 2 and params['support_qiskit'] is True
qasm, values, shots = params['pubs'][0]
assert qasm.startswith('OPENQASM 3.0;')
assert values is None and shots == params['shots']
with Path('isa.qpy').open('rb') as stream:
    circuit, = qpy.load(stream)
assert {r.name: len(r) for r in circuit.cregs} == {'meas': 2}
probs = Statevector.from_instruction(
    circuit.remove_final_measurements(inplace=False)
).probabilities_dict()
assert abs(probs.get('00', 0) - 0.5) < 1e-12
assert abs(probs.get('11', 0) - 0.5) < 1e-12
samples = ['00'] * (shots // 2) + ['11'] * (shots - shots // 2)
result = PrimitiveResult([
    SamplerPubResult(DataBin(meas=BitArray.from_samples(samples, num_bits=2)), metadata={})
], metadata={})
Path(destination).write_text(json.dumps(result, cls=RuntimeEncoder))
""",
        encoding="utf-8",
    )
    executable.chmod(0o700)
    calls = []

    async def submit(**kwargs):
        calls.append(kwargs)
        monkeypatch.chdir(kwargs["work_dir"])
        worker.main([*kwargs["user_args"], "--task-runner", str(executable)])
        return SimpleNamespace(job_id="123", state="COMPLETED", exit_status=0)

    monkeypatch.setattr(runtime_mod, "run_job_from_blocks", submit)
    summary = asyncio.run(
        bell_flow.fn(
            shots=shots,
            work_dir=str(tmp_path / "jobs"),
            resource="bell_resource",
            command_block_name="bell-command",
            execution_profile_block_name="bell-execution",
            hpc_profile_block_name="bell-hpc",
        )
    )
    assert summary["num_bits"] == 2
    assert summary["num_shots"] == shots
    assert summary["counts"] == {"00": shots // 2, "11": shots - shots // 2}
    assert summary["slurm"]["resource"] == "bell_resource"
    assert summary["slurm"]["state"] == "COMPLETED"
    assert summary["slurm"]["exit_status"] == 0
    assert len(calls) == 1
    assert calls[0]["command_block_name"] == "bell-command"
    assert calls[0]["execution_profile_block_name"] == "bell-execution"
    assert calls[0]["hpc_profile_block_name"] == "bell-hpc"
    job_dir = Path(summary["slurm"]["work_dir"])
    report = json.loads((job_dir / "transpile_report.json").read_text())
    assert report["registers"] == {"meas": 2}
    assert report["transpiler"] == runtime_mod.DEFAULT_TRANSPILER


@pytest.mark.parametrize("shots", [0, True])
def test_bell_example_rejects_invalid_shots_before_submission(tmp_path, monkeypatch, shots):
    async def reject(**kwargs):
        pytest.fail("Invalid shots must not submit a workload")

    monkeypatch.setattr(runtime_mod, "run_job_from_blocks", reject)
    with pytest.raises(ValueError, match="positive integer"):
        asyncio.run(bell_flow.fn(shots=shots, work_dir=str(tmp_path / "jobs")))
    assert not (tmp_path / "jobs").exists()
