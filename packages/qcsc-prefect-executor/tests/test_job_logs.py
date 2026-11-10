"""A finished job's stderr: progress at info level, warnings and errors called out."""

from __future__ import annotations

from qcsc_prefect_executor.job_logs import log_job_stderr, problem_lines

FULQRUM_STDERR = """\
2026-11-08 04:50:30,154 INFO fulqrum.core.fermi_operator: Extended JW time: 7.789 ms
2026-11-08 04:50:30,155 INFO fulqrum.core.subspace: Subspace size: 1000000
2026-11-08 04:50:37,342 INFO fulqrum.core.spmv: CSR fast-mode total build time: 6967.383 ms
"""


class Logger:
    def __init__(self):
        self.lines = []

    def info(self, message):
        self.lines.append(("info", message))

    def warning(self, message):
        self.lines.append(("warning", message))

    def error(self, message):
        self.lines.append(("error", message))


def test_solver_progress_is_info_only():
    logger = Logger()
    log_job_stderr(logger, FULQRUM_STDERR, failed=False, truncate=str)
    assert logger.lines == [("info", FULQRUM_STDERR)]


def test_warnings_and_tracebacks_stand_out():
    stderr = (
        FULQRUM_STDERR
        + "/x/annotation.py:61: ComplexWarning: Casting complex values to real\n"
        + "2026-11-08 04:50:40,000 WARNING qcsc_prefect_fulqrum.solver: slow convergence\n"
    )
    assert problem_lines(stderr) == stderr.splitlines()[-2:]
    logger = Logger()
    log_job_stderr(logger, stderr, failed=False, truncate=str)
    assert [level for level, _ in logger.lines] == ["info", "warning"]
    assert "slow convergence" in logger.lines[1][1] and "Subspace size" not in logger.lines[1][1]


def test_a_failed_job_logs_its_stderr_as_one_error():
    logger = Logger()
    log_job_stderr(logger, "Traceback: boom", failed=True, truncate=str)
    assert logger.lines == [("error", "Traceback: boom")]
    log_job_stderr(logger, "", failed=False, truncate=str)
    assert len(logger.lines) == 1
