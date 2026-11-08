from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path

from prefect.artifacts import create_table_artifact
from prefect.logging import get_run_logger
from qcsc_prefect_adapters.local.runtime import (
    LocalJobRequest,
    LocalRuntime,
    build_local_command,
)
from qcsc_prefect_core.models.execution_profile import ExecutionProfile

from qcsc_prefect_executor.job_logs import log_job_stderr

MAX_LOG_SIZE = 10_000


def _truncate_log(text: str) -> str:
    if len(text) > MAX_LOG_SIZE:
        return text[:MAX_LOG_SIZE] + f"... (truncated {len(text) - MAX_LOG_SIZE} chars)"
    return text


@dataclass(frozen=True)
class LocalRunResult:
    """Result returned after a command exits on the local Prefect worker."""

    job_id: str
    exit_status: int
    stdout: str
    stderr: str


async def run_local_job(
    *,
    work_dir: Path,
    exec_profile: ExecutionProfile,
    req: LocalJobRequest,
    timeout_seconds: float | None = None,
    metrics_artifact_key: str = "local-job-metrics",
) -> LocalRunResult:
    """Execute a command directly without generating or submitting a job script."""

    command = build_local_command(exec_profile=exec_profile, executable=req.executable)
    work_dir.mkdir(parents=True, exist_ok=True)

    logger = get_run_logger()
    process_result = await LocalRuntime().execute(
        command,
        cwd=work_dir,
        environments=exec_profile.environments,
        timeout_seconds=timeout_seconds,
    )

    if process_result.stdout:
        logger.info(_truncate_log(process_result.stdout))
    log_job_stderr(
        logger,
        process_result.stderr,
        failed=process_result.exit_status != 0,
        truncate=_truncate_log,
    )

    job_id = f"local-{process_result.pid}"
    artifact = {
        "job_id": job_id,
        "exit_status": process_result.exit_status,
        "command": shlex.join(command),
        "work_dir": str(work_dir),
    }
    await create_table_artifact(
        table=[list(artifact.keys()), list(artifact.values())],
        key=metrics_artifact_key,
    )

    return LocalRunResult(
        job_id=job_id,
        exit_status=process_result.exit_status,
        stdout=process_result.stdout,
        stderr=process_result.stderr,
    )
