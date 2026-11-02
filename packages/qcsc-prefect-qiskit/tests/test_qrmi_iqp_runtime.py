"""scheduler="iqp" contracts with a fake qrmi module: no network, IQP or Prefect server."""

from __future__ import annotations

import socket
import sys
import threading
import time
from enum import Enum
from types import ModuleType, SimpleNamespace

import anyio
import pytest
from qcsc_prefect.integrations.qiskit.qrmi import (
    PreparedCircuit,
    QrmiRuntime,
    QrmiRuntimeBlock,
)
from qcsc_prefect.integrations.qiskit.qrmi import iqp as iqp_mod
from qcsc_prefect.integrations.qiskit.qrmi import runtime as runtime_mod
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.primitives import PrimitiveResult, SamplerPubResult
from qiskit.primitives.containers import BitArray, DataBin
from qiskit.providers.fake_provider import GenericBackendV2

CONFIG = {
    "QRMI_IBM_QCS_ENDPOINT": "https://quantum.example/api/v1",
    "QRMI_IBM_QCS_IAM_ENDPOINT": "https://iam.example",
    "QRMI_IBM_QCS_IAM_APIKEY": "secret-api-key",
    "QRMI_IBM_QCS_SERVICE_CRN": "crn:v1:example",
}


def bell(register="meas", extra_qubits=0):
    circuit = QuantumCircuit(2 + extra_qubits)
    circuit.add_register(ClassicalRegister(2, register))
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure([0, 1], circuit.cregs[0])
    return circuit


class Status(Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"
    CANCELLED = "cancelled"


class FakeQrmi:
    """Records QRMI calls and fails the test if one resource is used concurrently."""

    def __init__(self, target, *, final=Status.DONE, polls_before_final=1):
        self.target = target
        self.final = final
        self.polls_before_final = polls_before_final
        self.calls = []
        self.resources = []
        self.jobs = []
        self.submitted = []
        self.overlap = []
        self.fail_on = {}  # call name -> exception
        self.block_on = {}  # call name -> threading.Event to wait for
        self.entered = {}  # call name -> threading.Event set when the call starts
        self.bits = None
        self.result_override = None  # a PrimitiveResult returned instead of a matching one
        self.lock = threading.Lock()

    def operation(self, resource, name):
        fake = self

        class Busy:
            def __enter__(self):
                with fake.lock:
                    if resource.busy:
                        fake.overlap.append(name)
                    resource.busy = True
                    fake.calls.append(name)
                fake.entered.setdefault(name, threading.Event()).set()
                if name in fake.block_on:
                    fake.block_on[name].wait(5)
                # Widen the window so that concurrent use would be detected.
                time.sleep(0.005)
                if name in fake.fail_on:
                    raise fake.fail_on[name]

            def __exit__(self, *exc):
                resource.busy = False

        return Busy()

    def module(self):
        fake = self
        qrmi = ModuleType("qrmi")

        class ResourceType(Enum):
            IBMQuantumComputeService = "ibm-quantum-compute-service"

        class QuantumResource:
            def __init__(self, resource_id, config):
                self.id = resource_id
                self.config = config
                self.busy = False

            @staticmethod
            def from_config(resource_id, resource_type, config):
                assert resource_type is ResourceType.IBMQuantumComputeService
                resource = QuantumResource(resource_id, config)
                with fake.operation(resource, "from_config"):
                    fake.resources.append(resource)
                return resource

            def acquire(self):
                fake.calls.append("acquire")

            def release(self, token):
                fake.calls.append("release")

        class Job:
            def __init__(self, resource, circuits, shots):
                self.resource = resource
                self.circuits = circuits
                self.shots = shots
                self.polls = 0
                self.cancelled = False

            def job_id(self):
                return f"job-{fake.jobs.index(self)}"

            def status(self):
                with fake.operation(self.resource, "status"):
                    if self.cancelled:
                        return Status.CANCELLED
                    self.polls += 1
                    if self.polls > fake.polls_before_final:
                        return fake.final
                    return Status.QUEUED

            def result(self):
                with fake.operation(self.resource, "result"):
                    if fake.result_override is not None:
                        return fake.result_override
                    bits = fake.bits or ["0"] * len(self.circuits)
                    return PrimitiveResult(
                        [
                            SamplerPubResult(
                                DataBin(
                                    **{
                                        reg.name: BitArray.from_samples(
                                            [bit * len(reg)] * self.shots, num_bits=len(reg)
                                        )
                                        for reg in circuit.cregs
                                    }
                                ),
                                metadata={},
                            )
                            for circuit, bit in zip(self.circuits, bits)
                        ],
                        metadata={},
                    )

            def cancel(self):
                with fake.operation(self.resource, "cancel"):
                    self.cancelled = True

        class SamplerV2:
            def __init__(self, resource, *, options):
                self.resource = resource
                self.options = options

            def run(self, circuits):
                with fake.operation(self.resource, "run"):
                    job = Job(self.resource, list(circuits), self.options["default_shots"])
                    fake.jobs.append(job)
                    fake.submitted.append(job)
                    return job

        ibm = ModuleType("qrmi.primitives.ibm")
        ibm.SamplerV2 = SamplerV2

        def get_backend(resource):
            with fake.operation(resource, "get_backend"):
                return SimpleNamespace(target=fake.target)

        ibm.get_backend = get_backend
        qrmi.QuantumResource = QuantumResource
        qrmi.ResourceType = ResourceType
        return qrmi, ibm


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def reject_network(*args, **kwargs):
        pytest.fail("QRMI IQP tests must not open network connections")

    monkeypatch.setattr(socket.socket, "connect", reject_network)
    monkeypatch.setattr(socket.socket, "connect_ex", reject_network)
    monkeypatch.setattr(socket, "create_connection", reject_network)
    monkeypatch.setattr(iqp_mod, "sample_on_iqp", iqp_mod.sample_on_iqp.fn)
    logs = SimpleNamespace(records=[])

    def log(level):
        return lambda message, *args: logs.records.append((level, message % args))

    logger = SimpleNamespace(info=log("info"), error=log("error"), exception=log("exception"))
    monkeypatch.setattr(iqp_mod, "get_run_logger", lambda: logger)
    return logs


@pytest.fixture
def fake(monkeypatch):
    target = GenericBackendV2(num_qubits=5, basis_gates=["rz", "sx", "x", "cx"]).target
    fake = FakeQrmi(target)
    qrmi, ibm = fake.module()
    monkeypatch.setitem(sys.modules, "qrmi", qrmi)
    monkeypatch.setitem(sys.modules, "qrmi.primitives", ModuleType("qrmi.primitives"))
    monkeypatch.setitem(sys.modules, "qrmi.primitives.ibm", ibm)
    return fake


@pytest.fixture
def runtime():
    return QrmiRuntime(
        scheduler="iqp",
        resource="ibm_test",
        resource_config=dict(CONFIG),
        watch_poll_interval=0.01,
    )


def sample(runtime, pubs, shots=4):
    return anyio.run(lambda: runtime.sampler(pubs, shots=shots))


def test_success_compiles_against_resource_target_and_returns_ordered_results(runtime, fake):
    def prepare(circuit):
        if circuit.cregs[0].name == "wide":
            circuit.add_register(QuantumRegister(1, "extra"))
            return PreparedCircuit(circuit, {"initial_layout": [4, 2, 0]})
        return PreparedCircuit(circuit)

    runtime.prepare = prepare
    runtime.transpile_config = {"initial_layout": [1, 3], "seed_transpiler": 7}
    fake.bits = ["0", "1"]
    result = sample(runtime, [(bell("narrow"),), (bell("wide"),)])

    assert result[0].data.narrow.get_counts() == {"00": 4}
    assert result[1].data.wide.get_counts() == {"11": 4}
    (resource,) = fake.resources
    assert resource.id == "ibm_test" and resource.config == CONFIG
    (job,) = fake.submitted
    layouts = [isa.layout.initial_index_layout(filter_ancillas=True) for isa in job.circuits]
    assert layouts == [[1, 3], [4, 2, 0]]
    assert job.shots == 4
    record = result.metadata["qcsc_qrmi"]
    assert record["job_id"] == "job-0"
    assert record["resource"] == "ibm_test"
    assert record["mode"] == "job"
    assert record["status"] == "DONE"
    assert [pub["initial_layout"] for pub in record["pubs"]] == [[1, 3], [4, 2, 0]]


def test_no_session_is_acquired_or_released(runtime, fake):
    sample(runtime, [(bell(),)])
    assert "acquire" not in fake.calls
    assert "release" not in fake.calls


@pytest.mark.parametrize("key", ["QRMI_IBM_QCS_SESSION_ID", "QRMI_JOB_ACQUISITION_TOKEN"])
def test_session_settings_are_rejected(fake, key):
    config = {**CONFIG, key: "existing-session"}
    with pytest.raises(ValueError, match=key):
        QrmiRuntime(scheduler="iqp", resource="ibm_test", resource_config=config)
    assert fake.calls == []


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({}, "requires a nonempty resource_config"),
        ({"resource_config": {}}, "requires a nonempty resource_config"),
        ({"resource_config": {"QRMI_IBM_QCS_IAM_APIKEY": 1}}, "values must be strings"),
        ({"resource_config": CONFIG, "root_dir": "/shared"}, "does not use root_dir"),
        (
            {"resource_config": CONFIG, "hpc_profile_block_name": "hpc"},
            "does not use hpc_profile_block_name",
        ),
    ],
)
def test_invalid_iqp_settings_fail_at_construction(fake, kwargs, message):
    with pytest.raises(ValueError, match=message):
        QrmiRuntime(scheduler="iqp", resource="ibm_test", **kwargs)
    assert fake.calls == []


def test_slurm_rejects_resource_config_and_requires_its_blocks(tmp_path):
    slurm = {
        "root_dir": tmp_path,
        "command_block_name": "c",
        "execution_profile_block_name": "e",
        "hpc_profile_block_name": "h",
        "resource": "test_heron",
    }
    with pytest.raises(ValueError, match="only used by scheduler"):
        QrmiRuntime(**slurm, resource_config=CONFIG)
    with pytest.raises(ValueError, match="requires root_dir"):
        QrmiRuntime(resource="test_heron")


def test_resource_config_is_not_in_repr(runtime):
    assert "secret-api-key" not in repr(runtime)


def test_invalid_pub_fails_before_creating_a_resource(runtime, fake):
    runtime.transpile_config = {"initial_layout": [0, 1, 2]}
    with pytest.raises(ValueError, match="PUB 0: .*expected 2, got 3"):
        sample(runtime, [(bell(),)])
    assert fake.calls == []


def test_job_failure_raises_and_does_not_cancel_a_final_job(runtime, fake):
    fake.final = Status.ERROR
    with pytest.raises(iqp_mod.IqpJobFailed, match="job-0 on ibm_test ended as ERROR"):
        sample(runtime, [(bell(),)])
    assert len(fake.submitted) == 1
    assert "cancel" not in fake.calls
    assert "result" not in fake.calls


@pytest.mark.parametrize("failing", ["status", "result"])
def test_error_after_submission_cancels_once_and_does_not_resubmit(runtime, fake, failing):
    fake.fail_on[failing] = ConnectionError("network down")
    with pytest.raises(ConnectionError):
        sample(runtime, [(bell(),)])
    assert len(fake.submitted) == 1
    if failing == "status":
        assert fake.calls.count("cancel") == 1
    else:
        # The job reached DONE; only fetching its result failed. Nothing to cancel.
        assert "cancel" not in fake.calls


def test_cancellation_while_waiting_cancels_job_once(runtime, fake):
    fake.polls_before_final = 10_000

    async def main():
        with anyio.move_on_after(0.2):
            await runtime.sampler([(bell(),)], shots=4)

    anyio.run(main)
    assert len(fake.submitted) == 1
    assert fake.calls.count("cancel") == 1
    assert fake.submitted[0].cancelled


def test_cancellation_during_submission_still_cancels_the_returned_job(runtime, fake):
    # The caller is cancelled while run() is in its thread. The thread finishes and
    # returns the job, so its ID is known and the job is cancelled at the next poll.
    fake.block_on["run"] = threading.Event()
    fake.polls_before_final = 10_000

    async def main():
        with anyio.CancelScope() as scope:
            async with anyio.create_task_group() as group:
                group.start_soon(runtime.sampler, [(bell(),)])

                async def cancel_when_submitting():
                    await anyio.to_thread.run_sync(
                        fake.entered.setdefault("run", threading.Event()).wait, 5
                    )
                    scope.cancel()
                    fake.block_on["run"].set()

                group.start_soon(cancel_when_submitting)

    anyio.run(main)
    assert len(fake.submitted) == 1
    assert fake.calls.count("cancel") == 1


def test_unknown_submission_is_not_resubmitted_and_is_recorded(runtime, fake, offline):
    fake.fail_on["run"] = TimeoutError("no response")
    with pytest.raises(iqp_mod.IqpSubmissionUnknown) as raised:
        sample(runtime, [(bell(),)])
    message = str(raised.value)
    assert "ibm_test" in message and "not resubmitted" in message
    assert fake.calls.count("run") == 1
    assert "cancel" not in fake.calls
    errors = [text for level, text in offline.records if level == "error"]
    assert any("ibm_test" in text and "did not return a job" in text for text in errors)


def test_failed_cancel_is_logged_and_not_retried(runtime, fake, offline):
    fake.fail_on["status"] = ConnectionError("network down")
    fake.fail_on["cancel"] = ConnectionError("still down")
    with pytest.raises(ConnectionError):
        sample(runtime, [(bell(),)])
    assert fake.calls.count("cancel") == 1
    assert any("Cancelling IQP job job-0 failed" in text for _, text in offline.records)


def test_one_resource_is_never_used_from_two_threads(runtime, fake):
    async def main():
        async with anyio.create_task_group() as group:
            for _ in range(4):
                group.start_soon(runtime.sampler, [(bell(),), (bell("b"),)])

    anyio.run(main)
    assert len(fake.resources) == 4
    assert len({id(r) for r in fake.resources}) == 4
    assert fake.overlap == []


@pytest.mark.parametrize(
    "returned,message",
    [
        ("missing_pub", "returned 1 PUBs; expected 2"),
        ("wrong_register", r"expected \{'b': 2\} \(PUB 1\); IQP job job-0"),
        ("wrong_shots", r"3 shots; expected 4 \(PUB 0\)"),
    ],
)
def test_mismatched_iqp_result_is_rejected(runtime, fake, returned, message):
    registers, shots = ["a", "b"], 4
    if returned == "missing_pub":
        registers = ["a"]
    elif returned == "wrong_register":
        registers = ["a", "c"]
    else:
        shots = 3
    fake.result_override = PrimitiveResult(
        [
            SamplerPubResult(
                DataBin(**{name: BitArray.from_samples(["00"] * shots, num_bits=2)}),
                metadata={},
            )
            for name in registers
        ],
        metadata={},
    )
    with pytest.raises(RuntimeError, match=message):
        sample(runtime, [(bell("a"),), (bell("b"),)])


def test_runtime_block_round_trip_keeps_settings_and_secret():
    block = QrmiRuntimeBlock(
        scheduler="iqp",
        resource="ibm_test",
        resource_config=dict(CONFIG),
        transpile_config={"optimization_level": 3},
        watch_poll_interval=0.5,
    )
    assert "secret-api-key" not in repr(block)
    doc = block._to_block_document(
        name="test", block_schema_id=uuid(), block_type_id=uuid(), include_secrets=True
    )
    restored = QrmiRuntimeBlock._from_block_document(doc)
    runtime = restored.get_runtime()
    assert runtime.scheduler == "iqp"
    assert runtime.resource == "ibm_test"
    assert runtime.resource_config == CONFIG
    assert runtime.transpile_config == {"optimization_level": 3}
    assert runtime.watch_poll_interval == 0.5
    assert runtime.root_dir is None
    masked = block._to_block_document(name="test", block_schema_id=uuid(), block_type_id=uuid())
    assert "secret-api-key" not in str(masked.data)


def test_runtime_block_builds_slurm_runtime_and_passes_prepare(tmp_path):
    def prepare(circuit):
        return PreparedCircuit(circuit)

    block = QrmiRuntimeBlock(
        resource="test_heron",
        root_dir=str(tmp_path),
        command_block_name="c",
        execution_profile_block_name="e",
        hpc_profile_block_name="h",
        timeout_seconds=120,
    )
    runtime = block.get_runtime(prepare)
    assert runtime.scheduler == "slurm"
    assert runtime.root_dir == tmp_path
    assert runtime.prepare is prepare
    assert runtime.timeout_seconds == 120
    assert runtime.transpiler == runtime_mod.DEFAULT_TRANSPILER


def test_runtime_block_rejects_invalid_settings_when_building_runtime():
    block = QrmiRuntimeBlock(scheduler="iqp", resource="ibm_test")
    with pytest.raises(ValueError, match="requires a nonempty resource_config"):
        block.get_runtime()


def uuid():
    import uuid as uuid_module

    return uuid_module.uuid4()
