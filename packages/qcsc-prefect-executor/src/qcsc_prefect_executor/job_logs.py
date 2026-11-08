"""Relay a finished job's stdout and stderr to the Prefect run log."""

from __future__ import annotations

import re
from typing import Any

# A stderr line that reports a problem, as Python logging, warnings and tracebacks print it.
_PROBLEM = re.compile(
    r"\b(WARNING|WARN|ERROR|CRITICAL|FATAL|Traceback|Exception|\w+Warning|\w+Error)\b"
)


def problem_lines(stderr: str) -> list[str]:
    """The lines of ``stderr`` that report a warning or an error."""
    return [line for line in stderr.splitlines() if _PROBLEM.search(line)]


def log_job_stderr(logger: Any, stderr: str, *, failed: bool, truncate) -> None:
    """Log a job's stderr: an error if the job failed, else progress at info level.

    Programs write progress to stderr too (for example Python logging at INFO), so a
    successful job's stderr is logged at info level. Its lines that report a warning
    or an error are also logged once at warning level, so they stand out.
    """
    if not stderr:
        return
    if failed:
        logger.error(truncate(stderr))
        return
    logger.info(truncate(stderr))
    problems = problem_lines(stderr)
    if problems:
        logger.warning(truncate("Warnings in the job's stderr:\n" + "\n".join(problems)))
