"""scheduler="iqp" contracts with a fake qrmi module: no network, IQP or Prefect server."""

from __future__ import annotations

import asyncio
import gc
import json
import re
import socket
import sys
import threading
import time
from enum import Enum
from types import ModuleType, SimpleNamespace

import anyio
import prefect.blocks.core
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
from qiskit.providers import JobStatus
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.providers.jobstatus import JOB_FINAL_STATES
from qiskit_ibm_runtime import RuntimeEncoder

try:
    from qiskit_ibm_runtime.decoders.result_decoder import ResultDecoder
except ModuleNotFoundError:
    from qiskit_ibm_runtime.utils.result_decoder import ResultDecoder

REGISTER = re.compile(r"^bit\[(\d+)\] (\w+);$", re.MULTILINE)
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


class TaskStatus(Enum):
    # Same members as qrmi.TaskStatus.
    Queued = 1
    Running = 2
    Completed = 3
    Failed = 4
    Cancelled = 5


class FakeQrmi:
    """Records QRMI calls and fails the test if one resource is used concurrently.

    The fake follows QRMI 0.26 (8284bb2): QuantumResource.task_start/task_status/
    task_result/task_stop, and qrmi.primitives.RuntimeJobV2 whose status() and
    result() call the resource and whose __del__ calls task_stop when delete_job=True.
    """

    def __init__(self, target, *, final=TaskStatus.Completed, polls_before_final=1):
        self.target = target
        self.final = final
        self.polls_before_final = polls_before_final
        self.calls = []
        self.call_threads = {}  # call name -> thread names
        self.resources = []
        self.jobs = []  # submitted job records, in order
        self.overlap = []
        self.fail_on = {}  # call name -> exception
        self.block_on = {}  # call name -> threading.Event to wait for
        self.entered = {}  # call name -> threading.Event set when the call starts
        self.bits = None
        self.result_override = None  # a PrimitiveResult returned instead of a matching one
        self.lock = threading.Lock()

    def entered_event(self, name):
        return self.entered.setdefault(name, threading.Event())

    def operation(self, resource, name):
        fake = self

        class Busy:
            def __enter__(self):
                with fake.lock:
                    if resource.busy:
                        fake.overlap.append(name)
                    resource.busy = True
                    fake.calls.append(name)
                    thread = threading.current_thread().name
                    fake.call_threads.setdefault(name, []).append(thread)
                fake.entered_event(name).set()
                if name in fake.block_on:
                    fake.block_on[name].wait(5)
                # Widen the window so that concurrent use would be detected.
                time.sleep(0.005)
                if name in fake.fail_on:
                    raise fake.fail_on[name]

            def __exit__(self, *exc):
                resource.busy = False

        return Busy()

    def make_result(self, job):
        if self.result_override is not None:
            return self.result_override
        bits = self.bits or ["0"] * len(job["registers"])
        return PrimitiveResult(
            [
                SamplerPubResult(
                    DataBin(
                        **{
                            name: BitArray.from_samples(
                                [bit * width] * job["shots"], num_bits=width
                            )
                            for name, width in registers.items()
                        }
                    ),
                    metadata={},
                )
                for registers, bit in zip(job["registers"], bits)
            ],
            metadata={},
        )

    def module(self):
        fake = self
        qrmi = ModuleType("qrmi")

        class ResourceType(Enum):
            IBMQuantumComputeService = "ibm-quantum-compute-service"

        class Payload:
            @staticmethod
            def QiskitPrimitive(*, input, program_id):  # noqa: N802 (QRMI's name)
                return SimpleNamespace(input=input, program_id=program_id)

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

            def task_start(self, payload):
                with fake.operation(self, "task_start"):
                    assert payload.program_id == "sampler"
                    parameters = json.loads(payload.input)
                    # Read classical registers from the QASM without qiskit_qasm3_import.
                    registers = [
                        {name: int(width) for width, name in REGISTER.findall(pub[0])}
                        for pub in parameters["pubs"]
                    ]
                    job = {
                        "id": f"job-{len(fake.jobs)}",
                        "parameters": parameters,
                        "registers": registers,
                        "shots": parameters["shots"],
                        "polls": 0,
                        "stopped": False,
                    }
                    fake.jobs.append(job)
                    return job["id"]

            def _job(self, task_id):
                return next(job for job in fake.jobs if job["id"] == task_id)

            def task_status(self, task_id):
                with fake.operation(self, "task_status"):
                    job = self._job(task_id)
                    if job["stopped"]:
                        return TaskStatus.Cancelled
                    job["polls"] += 1
                    if job["polls"] > fake.polls_before_final:
                        return fake.final
                    return TaskStatus.Queued

            def task_result(self, task_id):
                with fake.operation(self, "task_result"):
                    result = fake.make_result(self._job(task_id))
                    return SimpleNamespace(value=json.dumps(result, cls=RuntimeEncoder))

            def task_stop(self, task_id):
                with fake.operation(self, "task_stop"):
                    self._job(task_id)["stopped"] = True

        status_map = {
            TaskStatus.Queued: JobStatus.QUEUED,
            TaskStatus.Running: JobStatus.RUNNING,
            TaskStatus.Completed: JobStatus.DONE,
            TaskStatus.Failed: JobStatus.ERROR,
            TaskStatus.Cancelled: JobStatus.CANCELLED,
        }

        class RuntimeJobV2:
            # Mirrors qrmi.primitives.RuntimeJobV2 at 8284bb2, including __del__.
            def __init__(self, qrmi, job_id, *, delete_job=False):
                self._qrmi = qrmi
                self._job_id = job_id
                self._delete_job = delete_job
                self._last_status = None

            def __del__(self):
                if self._delete_job is True:
                    self._qrmi.task_stop(self._job_id)

            def job_id(self):
                return self._job_id

            def cancel(self):
                self._qrmi.task_stop(self._job_id)

            def status(self):
                if self._last_status is None or self._last_status not in JOB_FINAL_STATES:
                    self._last_status = self._qrmi.task_status(self._job_id)
                return status_map.get(self._last_status)

            def result(self):
                return ResultDecoder.decode(self._qrmi.task_result(self._job_id).value)

        primitives = ModuleType("qrmi.primitives")
        primitives.RuntimeJobV2 = RuntimeJobV2
        ibm = ModuleType("qrmi.primitives.ibm")

        def get_backend(resource):
            with fake.operation(resource, "get_backend"):
                return SimpleNamespace(target=fake.target)

        ibm.get_backend = get_backend
        qrmi.QuantumResource = QuantumResource
        qrmi.ResourceType = ResourceType
        qrmi.Payload = Payload
        qrmi.TaskStatus = TaskStatus
        return qrmi, primitives, ibm


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
    qrmi, primitives, ibm = fake.module()
    monkeypatch.setitem(sys.modules, "qrmi", qrmi)
    monkeypatch.setitem(sys.modules, "qrmi.primitives", primitives)
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
    (job,) = fake.jobs
    assert job["registers"] == [{"narrow": 2}, {"wide": 2}]
    assert job["shots"] == 4
    assert [pub[1:] for pub in job["parameters"]["pubs"]] == [[None, 4], [None, 4]]
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


def test_job_failure_raises_and_does_not_stop_a_final_job(runtime, fake):
    fake.final = TaskStatus.Failed
    with pytest.raises(iqp_mod.IqpJobFailed, match="job-0 on ibm_test ended as ERROR"):
        sample(runtime, [(bell(),)])
    gc.collect()
    assert len(fake.jobs) == 1
    assert "task_stop" not in fake.calls
    assert "task_result" not in fake.calls


@pytest.mark.parametrize("failing", ["task_status", "task_result"])
def test_error_after_submission_stops_once_and_does_not_resubmit(runtime, fake, failing):
    fake.fail_on[failing] = ConnectionError("network down")
    with pytest.raises(ConnectionError):
        sample(runtime, [(bell(),)])
    gc.collect()
    assert fake.calls.count("task_start") == 1
    if failing == "task_status":
        assert fake.calls.count("task_stop") == 1
    else:
        # The job reached DONE; only fetching its result failed. Nothing to stop.
        assert "task_stop" not in fake.calls


def test_successful_job_is_never_stopped_even_after_collection(runtime, fake):
    sample(runtime, [(bell(),)])
    gc.collect()
    assert "task_stop" not in fake.calls


def test_stop_requests_never_run_on_the_event_loop_thread(runtime, fake):
    fake.fail_on["task_status"] = ConnectionError("network down")
    loop_thread = []

    async def main():
        loop_thread.append(threading.current_thread().name)
        with pytest.raises(ConnectionError):
            await runtime.sampler([(bell(),)], shots=4)

    asyncio.run(main())
    gc.collect()
    assert fake.call_threads["task_stop"] and loop_thread[0] not in fake.call_threads["task_stop"]
    for name, threads in fake.call_threads.items():
        assert loop_thread[0] not in threads, f"{name} ran on the event loop thread"


def test_anyio_cancellation_while_waiting_stops_job_once(runtime, fake):
    fake.polls_before_final = 10_000

    async def main():
        with anyio.move_on_after(0.2):
            await runtime.sampler([(bell(),)], shots=4)

    anyio.run(main)
    gc.collect()
    assert fake.calls.count("task_start") == 1
    assert fake.calls.count("task_stop") == 1
    assert fake.jobs[0]["stopped"]
    assert fake.overlap == []


@pytest.mark.parametrize("how", ["task_cancel", "task_cancel_twice", "asyncio_timeout"])
def test_asyncio_cancellation_during_status_waits_before_stopping(runtime, fake, how):
    # Native asyncio cancellation does not wait for the thread running task_status.
    # The stop request must still start only after that call has returned.
    release = threading.Event()
    fake.block_on["task_status"] = release
    fake.polls_before_final = 10_000

    async def main():
        task = asyncio.create_task(runtime.sampler([(bell(),)], shots=4))
        await asyncio.to_thread(fake.entered_event("task_status").wait, 5)
        if how == "asyncio_timeout":
            threading.Timer(0.2, release.set).start()
            with pytest.raises(TimeoutError):
                async with asyncio.timeout(0.05):
                    await asyncio.shield(task)
            task.cancel()
        else:
            task.cancel()
            await asyncio.sleep(0.05)
            if how == "task_cancel_twice":
                task.cancel()
            await asyncio.sleep(0.05)
            assert "task_stop" not in fake.calls  # still waiting for task_status
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(main())
    gc.collect()
    assert fake.overlap == []
    assert fake.calls.count("task_stop") == 1
    assert fake.calls.index("task_stop") > fake.calls.index("task_status")


def test_asyncio_timeout_around_sampler_does_not_overlap_calls(runtime, fake):
    release = threading.Event()
    fake.block_on["task_status"] = release
    fake.polls_before_final = 10_000
    threading.Timer(0.3, release.set).start()

    async def main():
        with pytest.raises(TimeoutError):
            async with asyncio.timeout(0.1):
                await runtime.sampler([(bell(),)], shots=4)

    asyncio.run(main())
    gc.collect()
    assert fake.overlap == []
    assert fake.calls.count("task_stop") == 1


def test_cancellation_during_submission_still_stops_the_returned_job(runtime, fake):
    # The caller is cancelled while task_start is in its thread. The thread finishes and
    # returns the job ID, so the job is stopped once the submission call has returned.
    release = threading.Event()
    fake.block_on["task_start"] = release
    fake.polls_before_final = 10_000

    async def main():
        task = asyncio.create_task(runtime.sampler([(bell(),)], shots=4))
        await asyncio.to_thread(fake.entered_event("task_start").wait, 5)
        task.cancel()
        await asyncio.sleep(0.05)
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(main())
    gc.collect()
    assert fake.calls.count("task_start") == 1
    assert fake.calls.count("task_stop") == 1
    assert fake.overlap == []


def test_submission_error_during_cancellation_is_kept(runtime, fake, offline):
    # Cancelled while task_start runs, and task_start then fails. The cancellation still
    # propagates, but the reason the submission result is unknown is not lost.
    release = threading.Event()
    fake.block_on["task_start"] = release
    fake.fail_on["task_start"] = TimeoutError("no response")

    async def main():
        task = asyncio.create_task(runtime.sampler([(bell(),)], shots=4))
        await asyncio.to_thread(fake.entered_event("task_start").wait, 5)
        task.cancel()
        await asyncio.sleep(0.05)
        release.set()
        with pytest.raises(asyncio.CancelledError) as raised:
            await task
        return raised.value

    cancelled = asyncio.run(main())
    unknown = cancelled.__cause__
    assert isinstance(unknown, iqp_mod.IqpSubmissionUnknown)
    assert "no response" in str(unknown) and "not resubmitted" in str(unknown)
    assert isinstance(unknown.__cause__, TimeoutError)
    assert fake.calls.count("task_start") == 1
    assert "task_stop" not in fake.calls
    errors = [text for level, text in offline.records if level == "error"]
    assert any("did not return a job" in text and "no response" in text for text in errors)


def test_unknown_submission_is_not_resubmitted_and_is_recorded(runtime, fake, offline):
    fake.fail_on["task_start"] = TimeoutError("no response")
    with pytest.raises(iqp_mod.IqpSubmissionUnknown) as raised:
        sample(runtime, [(bell(),)])
    message = str(raised.value)
    assert "ibm_test" in message and "not resubmitted" in message
    assert fake.calls.count("task_start") == 1
    assert "task_stop" not in fake.calls
    assert isinstance(raised.value.__cause__, TimeoutError)
    errors = [text for level, text in offline.records if level == "error"]
    assert any(
        "ibm_test" in text and "did not return a job" in text and "no response" in text
        for text in errors
    )


def test_failed_stop_is_logged_and_not_retried(runtime, fake, offline):
    fake.fail_on["task_status"] = ConnectionError("network down")
    fake.fail_on["task_stop"] = ConnectionError("still down")
    with pytest.raises(ConnectionError):
        sample(runtime, [(bell(),)])
    gc.collect()
    assert fake.calls.count("task_stop") == 1
    assert any("Cancelling IQP job job-0 failed" in text for _, text in offline.records)


def cancel_messages(logs):
    return [(level, text) for level, text in logs.records if "ancel" in text]


@pytest.mark.parametrize("how", ["anyio", "task_cancel", "task_cancel_during_stop"])
def test_failed_stop_during_cancellation_is_logged_as_failure(runtime, fake, offline, how):
    # The stop request fails. Its failure must be reported, not hidden behind a
    # re-raised cancellation. anyio cancels the stop request itself (the scope stays
    # cancelled); "task_cancel_during_stop" does the same with a second Task.cancel().
    fake.polls_before_final = 10_000
    fake.fail_on["task_stop"] = ConnectionError("still down")

    if how == "anyio":

        async def main():
            with anyio.move_on_after(0.1):
                await runtime.sampler([(bell(),)], shots=4)

        anyio.run(main)
    else:
        release = threading.Event()
        if how == "task_cancel_during_stop":
            fake.block_on["task_stop"] = release

        async def main():
            task = asyncio.create_task(runtime.sampler([(bell(),)], shots=4))
            await asyncio.sleep(0.1)
            task.cancel()
            if how == "task_cancel_during_stop":
                await asyncio.to_thread(fake.entered_event("task_stop").wait, 5)
                task.cancel()
                await asyncio.sleep(0.05)
                release.set()
            with pytest.raises(asyncio.CancelledError):
                await task

        asyncio.run(main())
    assert fake.calls.count("task_stop") == 1
    messages = cancel_messages(offline)
    assert len(messages) == 1
    level, text = messages[0]
    assert level == "error"
    assert "Cancelling IQP job job-0 failed" in text and "still down" in text


def test_successful_stop_during_cancellation_is_logged_once(runtime, fake, offline):
    fake.polls_before_final = 10_000

    async def main():
        with anyio.move_on_after(0.1):
            await runtime.sampler([(bell(),)], shots=4)

    anyio.run(main)
    assert cancel_messages(offline) == [("info", "Requested cancellation of IQP job job-0")]


@pytest.mark.parametrize(
    "final,stopped", [(TaskStatus.Completed, False), (TaskStatus.Failed, False), (None, True)]
)
def test_status_returned_during_cancellation_decides_whether_to_stop(runtime, fake, final, stopped):
    # Cancelled while task_status runs. If that call reports a final status, the job
    # has already ended and must not be stopped; otherwise it is stopped once.
    release = threading.Event()
    fake.block_on["task_status"] = release
    if final is None:
        fake.polls_before_final = 10_000
    else:
        fake.polls_before_final = 0
        fake.final = final

    async def main():
        task = asyncio.create_task(runtime.sampler([(bell(),)], shots=4))
        await asyncio.to_thread(fake.entered_event("task_status").wait, 5)
        task.cancel()
        await asyncio.sleep(0.05)
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(main())
    gc.collect()
    assert fake.calls.count("task_status") == 1
    assert fake.calls.count("task_stop") == (1 if stopped else 0)
    assert fake.overlap == []


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


def test_runtime_block_round_trip_keeps_settings_and_secret(monkeypatch):
    # Restoring a block emits a Prefect "loaded" event, which starts a temporary
    # Prefect server. Record the event instead; this test checks only the settings.
    events = []
    monkeypatch.setattr(prefect.blocks.core, "emit_event", lambda **event: events.append(event))
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
    assert [event["event"] for event in events] == ["prefect.block.qrmi-runtime.loaded"]
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
    block = QrmiRuntimeBlock(scheduler="iqp", resource="ibm_test", service_crn="crn:v1:x")
    with pytest.raises(ValueError, match="needs IBM Cloud API Key"):
        block.get_runtime()


API_KEY = "AbCdEfGhIjKlMnOpQrStUvWxYz0123456789_-QWERT"


@pytest.mark.parametrize(
    "settings",
    [
        dict(scheduler="slurm", resource="r", root_dir="/x", api_key=API_KEY),
        dict(scheduler="iqp", service_crn="c", api_key=API_KEY),
        dict(api_key=API_KEY, scheduler="nope", resource="r"),
    ],
)
def test_a_validation_error_never_shows_the_api_key(settings):
    with pytest.raises(ValueError) as error:
        QrmiRuntimeBlock(**settings)
    for part in (API_KEY, API_KEY[:6], API_KEY[-6:]):
        assert part not in str(error.value)


def test_block_fields_are_stripped_and_blank_means_unset():
    block = QrmiRuntimeBlock(
        scheduler="iqp",
        resource=" ibm_test ",
        service_crn=" crn:v1:x\n",
        endpoint="",
        api_key="  ",
    )
    assert (block.resource, block.service_crn, block.endpoint, block.api_key) == (
        "ibm_test",
        "crn:v1:x",
        None,
        None,
    )
    with pytest.raises(ValueError, match="without spaces"):
        QrmiRuntimeBlock(scheduler="iqp", resource="ibm_test", service_crn="crn v1")
    with pytest.raises(ValueError, match="QRMI Resource is required"):
        QrmiRuntimeBlock(scheduler="iqp", resource=" ", api_key="k")
    # A blank key does not replace an older block's key in resource_config.
    old = QrmiRuntimeBlock(
        scheduler="iqp",
        resource="ibm_test",
        resource_config={"QRMI_IBM_QCS_IAM_APIKEY": "old-key"},
        api_key="",
    )
    assert old.get_runtime().resource_config["QRMI_IBM_QCS_IAM_APIKEY"] == "old-key"


def test_a_slurm_block_rejects_the_resource_config(tmp_path):
    with pytest.raises(ValueError, match="only used by scheduler 'iqp'"):
        QrmiRuntimeBlock(
            scheduler="slurm",
            resource="test_heron",
            root_dir=str(tmp_path),
            resource_config={"QRMI_IBM_QCS_SERVICE_CRN": "crn"},
        )


def uuid():
    import uuid as uuid_module

    return uuid_module.uuid4()


def test_iqp_block_fields_reach_qrmi_and_only_the_key_is_secret():
    block = QrmiRuntimeBlock(
        scheduler="iqp",
        resource="ibm_test",
        service_crn="crn:v1:visible",
        api_key="secret-api-key",
    )
    schema = QrmiRuntimeBlock.model_json_schema()["properties"]
    # Prefect Cloud masks a password-format, write-only field.
    (secret,) = [item for item in schema["api_key"]["anyOf"] if item.get("type") == "string"]
    assert secret["format"] == "password" and secret["writeOnly"] is True
    plain = schema["service_crn"]["anyOf"][0]
    assert "format" not in plain and "writeOnly" not in plain
    assert "secret-api-key" not in repr(block)
    assert "crn:v1:visible" in repr(block)  # the CRN is plain, by design
    masked = block._to_block_document(name="t", block_schema_id=uuid(), block_type_id=uuid())
    assert "secret-api-key" not in str(masked.data)
    assert "crn:v1:visible" in str(masked.data)
    runtime = block.get_runtime()
    assert runtime.resource_config == {
        "QRMI_IBM_QCS_SERVICE_CRN": "crn:v1:visible",
        "QRMI_IBM_QCS_IAM_APIKEY": "secret-api-key",
        "QRMI_IBM_QCS_ENDPOINT": "https://quantum.cloud.ibm.com/api/v1",
        "QRMI_IBM_QCS_IAM_ENDPOINT": "https://iam.cloud.ibm.com",
    }
    assert "secret-api-key" not in repr(runtime)


def test_block_fields_win_over_the_resource_config():
    block = QrmiRuntimeBlock(
        scheduler="iqp",
        resource="ibm_test",
        endpoint="https://new.example/api/v1",
        resource_config={"QRMI_IBM_QCS_ENDPOINT": "https://old.example", "OTHER": "kept"},
        api_key="k",
    )
    config = block.get_runtime().resource_config
    assert config["QRMI_IBM_QCS_ENDPOINT"] == "https://new.example/api/v1"
    assert config["OTHER"] == "kept"


def test_slurm_block_passes_plain_settings_and_rejects_an_api_key(tmp_path):
    slurm = dict(
        resource="test_heron",
        root_dir=str(tmp_path),
        command_block_name="c",
        execution_profile_block_name="e",
        hpc_profile_block_name="h",
    )
    runtime = QrmiRuntimeBlock(**slurm, service_crn="crn:v1:x", session_mode="batch").get_runtime()
    assert runtime.job_settings == {
        "QRMI_IBM_QCS_SERVICE_CRN": "crn:v1:x",
        "QRMI_IBM_QCS_SESSION_MODE": "batch",
    }
    assert runtime.resource_config is None
    with pytest.raises(ValueError, match="leave\\s+IBM Cloud API Key empty"):
        QrmiRuntimeBlock(**slurm, api_key="secret")
    with pytest.raises(ValueError, match="job_settings must not contain"):
        runtime_mod.QrmiRuntime(
            **{**slurm, "root_dir": tmp_path}, job_settings={"QRMI_IBM_QCS_IAM_APIKEY": "x"}
        )
