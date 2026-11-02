"""Direct QRMI execution on IBM Quantum Compute Service, without Slurm or a session."""

from __future__ import annotations

import socket
from datetime import datetime, timezone

import anyio
from prefect import task
from prefect.cache_policies import NO_CACHE
from prefect.logging import get_run_logger

from qiskit.primitives import PrimitiveResult

from .compile import compile_pubs
from .contract import validate_result

# Seconds to wait for QRMI's cancel request after the caller was cancelled or failed.
CANCEL_TIMEOUT_SECONDS = 30.0
FINAL_STATES = frozenset({"DONE", "ERROR", "CANCELLED"})


class IqpJobFailed(RuntimeError):
    """The IQP job ended without a usable result."""


class IqpSubmissionUnknown(RuntimeError):
    """Submission raised, so it is unknown whether an IQP job exists. Not resubmitted."""


def _status_name(status) -> str:
    return getattr(status, "name", str(status)).upper()


@task(name="quantum-sampling-qrmi-iqp", retries=0, cache_policy=NO_CACHE, persist_result=False)
async def sample_on_iqp(circuits, shots, registers, configs, runtime) -> PrimitiveResult:
    # QRMI reads its settings when the resource is created; import only when sampling.
    from qrmi import QuantumResource, ResourceType
    from qrmi.primitives.ibm import SamplerV2, get_backend

    logger = get_run_logger()

    # The QRMI resource is created per call and never shared. Every call below runs in
    # one worker thread at a time and is awaited before the next one starts, because
    # QuantumResource methods take &mut self and must not run concurrently.
    def create():
        return QuantumResource.from_config(
            runtime.resource, ResourceType.IBMQuantumComputeService, dict(runtime.resource_config)
        )

    resource = await anyio.to_thread.run_sync(create)
    target = await anyio.to_thread.run_sync(lambda: get_backend(resource).target)
    isa_circuits, pub_reports = await anyio.to_thread.run_sync(
        lambda: compile_pubs(circuits, configs, registers, target, runtime.transpiler)
    )
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
    try:
        sampler = SamplerV2(resource, options={"default_shots": shots})
        try:
            # Keep the reference for the whole call: RuntimeJobV2 stops its job when
            # it is garbage collected.
            job = await anyio.to_thread.run_sync(lambda: sampler.run(isa_circuits))
        except anyio.get_cancelled_exc_class():
            raise
        except Exception as exc:
            raise IqpSubmissionUnknown(
                f"Submitting to {runtime.resource} at {submitted_at} failed: {exc!r}. "
                "The job may still exist; it is not resubmitted. "
                "Check the IBM Quantum dashboard before retrying."
            ) from exc
        record["job_id"] = job.job_id()
        record["submitted_at"] = submitted_at
        logger.info("Submitted IQP job %s to %s", record["job_id"], runtime.resource)

        # Cancellation is accepted only here, between polls. A status call that is
        # running in its thread finishes first; its wait follows QRMI's own timeouts.
        while True:
            status = _status_name(await anyio.to_thread.run_sync(job.status))
            if status in FINAL_STATES:
                break
            await anyio.sleep(runtime.watch_poll_interval)
        record["status"] = status
        if status != "DONE":
            raise IqpJobFailed(
                f"IQP job {record['job_id']} on {runtime.resource} ended as {status}"
            )
        # The job is final, but fetching the result is still a network call.
        result = await anyio.to_thread.run_sync(job.result)
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
    """Try job.cancel() once, even while this task is being cancelled."""
    with anyio.CancelScope(shield=True), anyio.move_on_after(CANCEL_TIMEOUT_SECONDS) as scope:
        try:
            await anyio.to_thread.run_sync(job.cancel)
        except Exception:
            logger.exception("Cancelling IQP job %s failed; not retried", record["job_id"])
            return
    if scope.cancelled_caught:
        logger.error(
            "Cancelling IQP job %s did not return in %ss; the request may still be in flight",
            record["job_id"],
            CANCEL_TIMEOUT_SECONDS,
        )
    else:
        logger.info("Requested cancellation of IQP job %s", record["job_id"])
