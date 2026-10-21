from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class SubmitError(RuntimeError):
    """Raised when job submission fails."""


class WaitTimeout(RuntimeError):
    """Raised when waiting for final job status times out."""


class CancelError(RuntimeError):
    """Raised when job cancellation fails."""


async def run_command(*args: str, cwd: Path | None = None) -> str:
    """Run a Slurm command asynchronously and return decoded stdout.

    Args:
        *args: Command and arguments passed to
            `asyncio.create_subprocess_exec`.
        cwd: Optional working directory for the subprocess.

    Returns:
        Standard output decoded with replacement for invalid bytes.

    Raises:
        RuntimeError: If the command exits with a non-zero return code. The
            error message includes decoded stdout and stderr for diagnostics.
    """

    proc = await asyncio.create_subprocess_exec(
        *args,
        cwd=str(cwd) if cwd else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out_b, err_b = await proc.communicate()
    out = (out_b or b"").decode(errors="replace")
    err = (err_b or b"").decode(errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(args)} rc={proc.returncode}\nstdout:\n{out}\nstderr:\n{err}"
        )
    return out


@dataclass(frozen=True)
class SubmitResult:
    """Submission result returned after a scheduler accepts a batch script.

    Attributes:
        job_id: Scheduler job id parsed from the submission command output.
        raw_output: Raw, stripped stdout emitted by the submission command.
    """

    job_id: str
    raw_output: str


def _is_terminal_state(state: str) -> bool:
    normalized = state.strip().split()[0].rstrip("+")
    return normalized in {
        "BOOT_FAIL",
        "CANCELLED",
        "COMPLETED",
        "DEADLINE",
        "FAILED",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "PREEMPTED",
        "TIMEOUT",
    }


class SlurmRuntime:
    """Async runtime wrapper for Slurm scheduler commands.

    This class is the low-level boundary around ``sbatch``, ``sacct``, and
    ``scancel``. Workflow code usually calls
    `qcsc_prefect_executor.slurm.run.run_slurm_job` or
    `qcsc_prefect_executor.from_blocks.run_job_from_blocks` instead.
    """

    def __init__(self):
        # Initialize the logger attribute for the class
        self.logger = logging.getLogger(self.__class__.__name__)

    async def submit(self, script_path: Path, *, cwd: Path | None = None) -> SubmitResult:
        """Submit a Slurm batch script with ``sbatch --parsable``.

        Args:
            script_path: Path to the generated Slurm script.
            cwd: Optional working directory for ``sbatch``.

        Returns:
            Parsed submission payload containing the Slurm job id.

        Raises:
            SubmitError: If ``sbatch`` fails or the job id cannot be parsed.
        """

        try:
            stdout = await run_command("sbatch", "--parsable", str(script_path), cwd=cwd)
        except Exception as e:
            raise SubmitError(f"sbatch failed for {script_path}") from e

        out = stdout.strip()
        if not out:
            raise SubmitError("sbatch returned empty stdout; cannot parse job id.")
        job_id = out.split(";", 1)[0].strip()
        if not job_id:
            raise SubmitError(f"Failed to parse Slurm job id from sbatch output: {out}")
        return SubmitResult(job_id=job_id, raw_output=out)

    async def wait_final_status(
        self,
        job_id: str,
        *,
        watch_poll_interval: float = 10.0,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Poll ``sacct`` until the job reaches a terminal state.

        The returned dictionary is normalized to the fields requested from
        ``sacct``: ``JobID``, ``State``, ``ExitCode``, ``Elapsed``,
        ``AllocCPUS``, and ``NodeList``.

        Args:
            job_id: Slurm job id to watch.
            watch_poll_interval: Seconds to wait between ``sacct`` calls.
            timeout_seconds: Optional maximum wait time.

        Returns:
            Parsed terminal ``sacct`` row for the batch job.

        Raises:
            WaitTimeout: If ``timeout_seconds`` elapses before a terminal state.
            RuntimeError: If an underlying ``sacct`` command fails.
        """

        start = asyncio.get_running_loop().time()
        try:
            while True:
                try:
                    stdout = await run_command(
                        "scontrol", "show", "job", "-o", job_id
                    )

                    if stdout.strip():
                        # scontrol -o gives a single line of space-separated key=value pairs per job
                        for line in stdout.strip().split('\n'):
                            # Parse key-value pairs (handling potential spaces inside values if any,
                            # though -o usually formats tightly)
                            job_data = {}
                            for token in line.split():
                                if '=' in token:
                                    k, v = token.split('=', 1)
                                    job_data[k] = v

                            # Map scontrol fields to your required dictionary keys
                            # Note: scontrol uses 'JobId', 'JobState', 'ExitCode', 'RunTime', 'NumCPUs', 'NodeList'
                            out = {
                                'JobID': job_data.get('JobId', job_id),
                                'State': job_data.get('JobState', 'UNKNOWN'),
                                'ExitCode': job_data.get('ExitCode', '0:0'),
                                'Elapsed': job_data.get('RunTime', '0:00'),
                                'AllocCPUS': job_data.get('NumCPUs', job_data.get('AllocCPUS', '0')),
                                'NodeList': job_data.get('NodeList', ''),
                            }

                            # Check if job finished
                            final_states = ['BOOT_FAIL', 'COMPLETED', 'FAILED', 'CANCELLED', 'DEADLINE', 'TIMEOUT', 'NODE_FAIL', 'OUT_OF_MEMORY', 'PREEMPTED']
                            if out['State'] in final_states:
                                exit_code = out['ExitCode'].split(':')[0]
                                self.logger.info(f"Job {job_id} finished with state {out['State']} and exit code {exit_code}")
                                return out
                except Exception as e:
                    self.logger.warning(f"scontrol command failed for job {job_id}: {e}")

                try:
                    stdout = await run_command(
                        "sacct", "-j", job_id,
                        "--format=JobID,State,ExitCode,Elapsed,AllocCPUS,NodeList",
                        "--parsable2", "--noheader"
                    )

                    if stdout.strip():
                        for line in stdout.strip().split('\n'):
                            fields = line.split('|')
                            job_id_field = fields[0]

                            # Skip job steps (.batch, .extern)
                            if '.' in job_id_field or '+' in job_id_field:
                                continue

                            out = {
                                'JobID': fields[0],
                                'State': fields[1],
                                'ExitCode': fields[2],
                                'Elapsed': fields[3],
                                'AllocCPUS': fields[4],
                                'NodeList': fields[5],
                            }

                            # Check if job finished
                            final_states = ['COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'NODE_FAIL']
                            if out['State'] in final_states:
                                exit_code = out['ExitCode'].split(':')[0]
                                self.logger.info(f"Job {job_id} finished with state {out['State']} and exit code {exit_code}")
                                return out
                except Exception as e:
                    self.logger.warning(f"sacct command failed for job {job_id}: {e}")

                await asyncio.sleep(watch_poll_interval)

        except asyncio.CancelledError:
            await run_command("scancel", job_id)
            return {}

    async def cancel(self, job_id: str) -> None:
        """Cancel a Slurm job using ``scancel``.

        Args:
            job_id: Target Slurm job id.

        Raises:
            CancelError: If ``scancel`` exits with an error.
        """

        try:
            await run_command("scancel", job_id)
        except Exception as e:
            raise CancelError(f"scancel failed for job_id={job_id}") from e
