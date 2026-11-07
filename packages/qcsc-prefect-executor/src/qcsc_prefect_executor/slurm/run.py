from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from prefect.artifacts import create_table_artifact
from prefect.logging import get_run_logger
from qcsc_prefect_adapters.slurm.builder import SlurmJobRequest, render_script, write_script_file
from qcsc_prefect_adapters.slurm.runtime import SlurmRuntime
from qcsc_prefect_core.models.execution_profile import ExecutionProfile

MAX_LOG_SIZE = 10_000


def truncate_log(text: str) -> str:
    """Truncate large log text to the configured maximum length."""

    if len(text) > MAX_LOG_SIZE:
        return text[:MAX_LOG_SIZE] + f"... (truncated {len(text) - MAX_LOG_SIZE} chars)"
    return text


def _read_text_if_exists(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(errors="replace")


def _create_job_artifact(
    *,
    job_id: str,
    job_status: dict[str, Any],
    stdout_file: Path,
    stderr_file: Path,
) -> dict[str, Any]:
    exit_code_text = str(job_status.get("ExitCode", "-1:0"))
    exit_code, _, signal = exit_code_text.partition(":")
    return {
        "job_id": job_id,
        "state": job_status.get("State"),
        "exit_code": exit_code,
        "signal": signal,
        "elapsed_time": job_status.get("Elapsed"),
        "allocated_cpus": job_status.get("AllocCPUS"),
        "node_list": job_status.get("NodeList"),
        "stdout_file": str(stdout_file) if stdout_file.exists() else None,
        "stderr_file": str(stderr_file) if stderr_file.exists() else None,
    }


@dataclass(frozen=True)
class SlurmRunResult:
    """Normalized result returned by `run_slurm_job`.

    Attributes:
        job_id: Slurm job id returned by ``sbatch``.
        exit_status: Integer process exit code parsed from Slurm ``ExitCode``.
        state: Final Slurm job state.
        job_status: Parsed terminal ``sacct`` status dictionary.
    """

    job_id: str
    exit_status: int
    state: str
    job_status: dict[str, Any]


async def run_slurm_job(
    *,
    work_dir: Path,
    script_filename: str,
    exec_profile: ExecutionProfile,
    req: SlurmJobRequest,
    watch_poll_interval: float = 10.0,
    timeout_seconds: float | None = None,
    metrics_artifact_key: str = "slurm-job-metrics",
    submit_env: Mapping[str, str | None] | None = None,
) -> SlurmRunResult:
    """Execute a Slurm job end-to-end from runtime models.

    This high-level executor renders the Slurm script, submits it with
    ``sbatch``, waits for terminal ``sacct`` status, captures stdout/stderr
    files, and publishes a Prefect table artifact with scheduler metrics.

    Args:
        work_dir: Working directory where scripts and job outputs are written.
        script_filename: Job script filename to create in ``work_dir``.
        exec_profile: Scheduler-independent execution profile.
        req: Slurm-specific scheduler request fields.
        watch_poll_interval: Poll interval in seconds for job status checks.
        timeout_seconds: Optional timeout for waiting final status.
        metrics_artifact_key: Prefect artifact key for job metrics table.
        submit_env: Extra environment variables for ``sbatch`` only; the job and
            its SPANK plugins see them, and the rendered script does not hold them.

    Returns:
        `SlurmRunResult` containing job id, exit status, state, and
        final scheduler status payload.
    """

    logger = get_run_logger()

    script_text = render_script(work_dir=work_dir, exec_profile=exec_profile, req=req)
    script_path = write_script_file(work_dir=work_dir, filename=script_filename, text=script_text)

    runtime = SlurmRuntime()
    extra = {"env": dict(submit_env)} if submit_env else {}
    submit = await runtime.submit(script_path, cwd=work_dir, **extra)
    final_status = await runtime.wait_final_status(
        submit.job_id,
        watch_poll_interval=watch_poll_interval,
        timeout_seconds=timeout_seconds,
    )

    stdout_file = work_dir / "output.out"
    stderr_file = work_dir / "output.err"

    exit_code_text = str(final_status.get("ExitCode", "-1:0")).split(":", 1)[0]
    exit_status = int(exit_code_text) if exit_code_text.isdigit() else -1

    if logs := _read_text_if_exists(stdout_file):
        logger.info(truncate_log(logs))
    if logs := _read_text_if_exists(stderr_file):
        # Programs also write progress and warnings to stderr; only a failed job's is an error.
        log_stderr = logger.warning if exit_status == 0 else logger.error
        log_stderr(truncate_log(logs))

    if final_status:
        artifact = _create_job_artifact(
            job_id=submit.job_id,
            job_status=final_status,
            stdout_file=stdout_file,
            stderr_file=stderr_file,
        )
        await create_table_artifact(
            table=[list(artifact.keys()), list(artifact.values())],
            key=metrics_artifact_key,
        )

    return SlurmRunResult(
        job_id=submit.job_id,
        exit_status=exit_status,
        state=str(final_status.get("State", "")),
        job_status=final_status,
    )
