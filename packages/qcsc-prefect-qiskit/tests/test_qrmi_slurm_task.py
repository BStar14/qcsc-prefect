"""Run the real Prefect task against a temporary server and a fake Slurm boundary."""

from __future__ import annotations

import asyncio
import ipaddress
import json
import socket
from types import SimpleNamespace

from prefect import flow
from prefect.client.orchestration import get_client
from prefect.context import TaskRunContext
from prefect.testing.utilities import prefect_test_harness
from qcsc_prefect.integrations.qiskit.qrmi import PreparedCircuit, QrmiSlurmRuntime
from qcsc_prefect.integrations.qiskit.qrmi import runtime as runtime_mod
from qiskit import QuantumCircuit, qpy
from qiskit.primitives import PrimitiveResult, SamplerPubResult
from qiskit.primitives.containers import BitArray, DataBin
from qiskit_ibm_runtime import RuntimeEncoder


def test_sampler_runs_as_prefect_task_with_prepare_callable(tmp_path, monkeypatch):
    # The harness overrides the user's API URL and uses an isolated SQLite database.
    monkeypatch.setenv("PREFECT_HOME", str(tmp_path / "prefect"))
    monkeypatch.setenv("PREFECT_SERVER_ANALYTICS_ENABLED", "false")
    monkeypatch.setenv("PREFECT_API_KEY", "")
    connect = socket.socket.connect
    connect_ex = socket.socket.connect_ex

    def local_only(sock, address):
        if sock.family != socket.AF_UNIX:
            host = address[0]
            assert host == "localhost" or ipaddress.ip_address(host).is_loopback, (
                f"Prefect task tests must not connect to external hosts: {host}"
            )

    def guarded_connect(sock, address):
        local_only(sock, address)
        return connect(sock, address)

    def guarded_connect_ex(sock, address):
        local_only(sock, address)
        return connect_ex(sock, address)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)
    prepared = []
    task_run_ids = []

    def prepare(circuit):
        prepared.append(circuit)
        circuit.metadata = {"prepared": True}
        return PreparedCircuit(circuit, {"seed": 24})

    runtime = QrmiSlurmRuntime(
        root_dir=tmp_path / "jobs",
        command_block_name="test-command",
        execution_profile_block_name="test-execution",
        hpc_profile_block_name="test-hpc",
        resource="test-resource",
        prepare=prepare,
    )
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure_all()

    async def submit(**kwargs):
        context = TaskRunContext.get()
        assert context is not None
        assert context.task.name == "quantum-sampling-qrmi-slurm"
        assert context.parameters["runtime"].prepare is prepare
        assert isinstance(context.parameters["circuit"], QuantumCircuit)
        task_run_ids.append(context.task_run.id)
        work_dir = kwargs["work_dir"]
        with (work_dir / "circuit.qpy").open("rb") as stream:
            (restored,) = qpy.load(stream)
        assert restored.metadata == {"prepared": True}
        assert json.loads((work_dir / "transpile_config.json").read_text()) == {"seed": 24}
        request = json.loads((work_dir / "sampler_request.json").read_text())
        assert request["shots"] == 100
        assert request["registers"] == {"meas": 2}
        result = PrimitiveResult(
            [
                SamplerPubResult(
                    DataBin(meas=BitArray.from_samples(["00", "11"] * 50, num_bits=2)),
                    metadata={},
                )
            ],
            metadata={},
        )
        (work_dir / "result.json").write_text(json.dumps(result, cls=RuntimeEncoder))
        return SimpleNamespace(job_id="73", state="COMPLETED", exit_status=0)

    monkeypatch.setattr(runtime_mod, "run_job_from_blocks", submit)

    @flow(name="qrmi-slurm-task-test", retries=0, persist_result=False)
    async def sample():
        return await runtime.sampler([(circuit,)], shots=100)

    async def exercise():
        result = await sample()
        assert len(task_run_ids) == 1
        # Prefect records task state events asynchronously on the server.
        async with get_client() as client:
            async with asyncio.timeout(10):
                while True:
                    task_run = await client.read_task_run(task_run_ids[0])
                    if task_run.state is not None and task_run.state.is_final():
                        break
                    await asyncio.sleep(0.1)
        assert task_run.state.is_completed()
        return result

    with prefect_test_harness():
        result = asyncio.run(exercise())

    assert len(prepared) == 1
    assert prepared[0] is not circuit
    assert circuit.metadata == {}
    samples = result[0].data.meas
    assert isinstance(samples, BitArray)
    assert samples.num_bits == 2
    assert samples.num_shots == 100
    assert samples.get_counts() == {"00": 50, "11": 50}
    assert result.metadata["qcsc_slurm"]["job_id"] == "73"
