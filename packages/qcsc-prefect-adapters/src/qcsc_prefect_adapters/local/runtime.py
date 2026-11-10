from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path

from qcsc_prefect_core.models.execution_profile import ExecutionProfile


@dataclass(frozen=True)
class LocalJobRequest:
    """Executable resolved for a local process invocation."""

    executable: str


@dataclass(frozen=True)
class LocalProcessResult:
    """Captured result of a local process invocation."""

    pid: int
    exit_status: int
    stdout: str
    stderr: str


def build_local_command(*, exec_profile: ExecutionProfile, executable: str) -> list[str]:
    """Build a shell-free command for local execution.

    Local execution intentionally does not interpret shell setup. Modules and
    pre-commands must be applied before the Prefect worker is started. GPU
    requests are scheduler directives, so ``gpus_per_node`` is rejected too:
    the process sees whatever GPUs the worker host exposes.
    """

    if exec_profile.gpus_per_node is not None:
        raise ValueError(
            "Local execution does not support gpus_per_node. "
            "The process uses the GPUs visible to the Prefect worker."
        )

    unsupported: list[str] = []
    if exec_profile.modules:
        unsupported.append("modules")
    if exec_profile.pre_commands:
        unsupported.append("pre_commands")
    if unsupported:
        raise ValueError(
            "Local execution does not support "
            + " or ".join(unsupported)
            + ". Configure the local worker environment before execution."
        )

    if not executable.strip():
        raise ValueError("Local executable path must not be empty.")

    command: list[str] = []
    if exec_profile.launcher != "single":
        command.append(exec_profile.launcher)
        command.extend(exec_profile.mpi_options)
    command.append(executable)
    command.extend(exec_profile.arguments)
    return command


class LocalRuntime:
    """Execute a command directly on the Prefect worker without a job script."""

    async def execute(
        self,
        command: list[str],
        *,
        cwd: Path,
        environments: dict[str, str] | None = None,
        timeout_seconds: float | None = None,
    ) -> LocalProcessResult:
        if not command:
            raise ValueError("Local command must not be empty.")

        environment = os.environ.copy()
        if environments:
            environment.update({str(key): str(value) for key, value in environments.items()})

        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=str(cwd),
            env=environment,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout_seconds,
            )
        except TimeoutError:
            process.kill()
            await process.communicate()
            raise TimeoutError(
                f"Local command timed out after {timeout_seconds} seconds: {command[0]}"
            ) from None

        return LocalProcessResult(
            pid=process.pid,
            exit_status=int(process.returncode or 0),
            stdout=(stdout_bytes or b"").decode(errors="replace"),
            stderr=(stderr_bytes or b"").decode(errors="replace"),
        )
