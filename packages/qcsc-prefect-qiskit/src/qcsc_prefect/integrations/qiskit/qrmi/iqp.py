"""Direct QRMI execution on IBM Quantum Compute Service, without Slurm or a session."""

from __future__ import annotations

import asyncio
import json
import socket
from datetime import datetime, timezone

from prefect import task
from prefect.cache_policies import NO_CACHE
from prefect.logging import get_run_logger

from qiskit.primitives import PrimitiveResult

from .compile import compile_pubs, make_sampler_parameters
from .contract import validate_result

FINAL_STATES = frozenset({"DONE", "ERROR", "CANCELLED"})


class IqpJobFailed(RuntimeError):
    """The IQP job ended without a usable result."""


class IqpSubmissionUnknown(RuntimeError):
    """Submission raised, so it is unknown whether an IQP job exists. Not resubmitted."""


def _status_name(status) -> str:
    return getattr(status, "name", str(status)).upper()


async def _call(fn):
    """Run one QRMI call in a worker thread and return only after that thread is done.

    QuantumResource methods take ``&mut self`` and must not overlap. A cancelled
    ``await`` does not stop the thread, so on cancellation this keeps waiting for
    the thread (through repeated cancellation) before re-raising. The re-raised
    ``CancelledError`` carries the finished call as ``qrmi_future`` so that its
    value or error (a job ID, a final status, a failed stop) is not lost; read it
    with ``_finished``. The wait follows QRMI's own network timeouts.
    """
    # Raises RuntimeError outside asyncio (for example under trio).
    future = asyncio.get_running_loop().run_in_executor(None, fn)
    try:
        return await asyncio.shield(future)
    except asyncio.CancelledError as exc:
        while not future.done():
            try:
                await asyncio.shield(future)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        exc.qrmi_future = future
        raise


def _finished(exc):
    """Return ``(value, error)`` of the QRMI call finished during cancellation ``exc``."""
    future = getattr(exc, "qrmi_future", None)
    if future is None or future.cancelled():
        return None, None
    error = future.exception()
    return (None, error) if error is not None else (future.result(), None)


@task(name="quantum-sampling-qrmi-iqp", retries=0, cache_policy=NO_CACHE, persist_result=False)
async def sample_on_iqp(circuits, shots, registers, configs, runtime) -> PrimitiveResult:
    # QRMI reads its settings when the resource is created; import only when sampling.
    from qrmi import Payload, QuantumResource, ResourceType
    from qrmi.primitives import RuntimeJobV2
    from qrmi.primitives.ibm import get_backend

    logger = get_run_logger()

    # The QRMI resource is created per call and never shared. Every QRMI call below goes
    # through _call and is complete before the next one starts.
    resource = await _call(
        lambda: QuantumResource.from_config(
            runtime.resource, ResourceType.IBMQuantumComputeService, dict(runtime.resource_config)
        )
    )
    target = await _call(lambda: get_backend(resource).target)
    isa_circuits, pub_reports = await _call(
        lambda: compile_pubs(circuits, configs, registers, target, runtime.transpiler)
    )
    parameters = make_sampler_parameters(isa_circuits, shots)
    payload = Payload.QiskitPrimitive(input=json.dumps(parameters), program_id="sampler")
    record = {
        "resource": runtime.resource,
        "mode": "job",
        "transpiler": runtime.transpiler,
        "target_num_qubits": target.num_qubits,
        "host": socket.gethostname(),
        "pubs": pub_reports,
    }

    job = None
    submitted_at = datetime.now(timezone.utc).isoformat()

    def submitted(job_id):
        nonlocal job
        # delete_job=False: the job object must not stop the job when it is collected.
        # This function decides when to cancel, and does it once in a worker thread.
        job = RuntimeJobV2(resource, job_id, delete_job=False)
        record["job_id"] = job_id
        record["submitted_at"] = submitted_at
        logger.info("Submitted IQP job %s to %s", job_id, runtime.resource)

    def polled(status):
        status = _status_name(status)
        if status in FINAL_STATES:
            record["status"] = status
        return status

    try:
        try:
            job_id = await _call(lambda: resource.task_start(payload))
        except asyncio.CancelledError as exc:
            # Cancelled while task_start ran: still record the job ID it returned.
            job_id, _ = _finished(exc)
            if job_id is not None:
                submitted(job_id)
            raise
        except Exception as exc:
            raise IqpSubmissionUnknown(
                f"Submitting to {runtime.resource} at {submitted_at} failed: {exc!r}. "
                "The job may still exist; it is not resubmitted. "
                "Check the IBM Quantum dashboard before retrying."
            ) from exc
        submitted(job_id)

        # Cancellation is accepted between polls; a running status call finishes first.
        while True:
            try:
                status = polled(await _call(job.status))
            except asyncio.CancelledError as exc:
                # Cancelled while task_status ran: keep a final status it returned, so
                # that a job that already ended is not stopped.
                value, _ = _finished(exc)
                if value is not None:
                    polled(value)
                raise
            if status in FINAL_STATES:
                break
            await asyncio.sleep(runtime.watch_poll_interval)
        if status != "DONE":
            raise IqpJobFailed(
                f"IQP job {record['job_id']} on {runtime.resource} ended as {status}"
            )
        # The job is final, but fetching the result is still a network call.
        result = await _call(job.result)
    except BaseException:
        if job is not None and record.get("status") is None:
            await _cancel_once(job, record, logger)
        if job is None:
            logger.error(
                "IQP submission to %s at %s did not return a job; nothing to cancel",
                runtime.resource,
                submitted_at,
            )
        raise
    validate_result(result, shots, registers, where=f"IQP job {record['job_id']}")
    result.metadata["qcsc_qrmi"] = record
    logger.info("IQP job %s completed on %s", record["job_id"], runtime.resource)
    return result


async def _cancel_once(job, record, logger) -> None:
    """Try job.cancel() once, after any running QRMI call, even while being cancelled."""
    try:
        await _call(job.cancel)
    except asyncio.CancelledError as exc:
        # _call already waited for the stop request; report how it ended, then propagate.
        _, error = _finished(exc)
        _log_cancel(record, logger, error)
        raise
    except Exception as error:
        _log_cancel(record, logger, error)
        return
    _log_cancel(record, logger, None)


def _log_cancel(record, logger, error) -> None:
    if error is None:
        logger.info("Requested cancellation of IQP job %s", record["job_id"])
    else:
        logger.error(
            "Cancelling IQP job %s failed; not retried; the job may still run: %r",
            record["job_id"],
            error,
        )
