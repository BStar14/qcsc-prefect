from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from qcsc_prefect_adapters.local.runtime import LocalJobRequest, LocalProcessResult
from qcsc_prefect_core.models.execution_profile import ExecutionProfile
from qcsc_prefect_executor.local import run as run_mod


class _LoggerStub:
    def __init__(self) -> None:
        self.info_lines: list[str] = []
        self.warning_lines: list[str] = []
        self.error_lines: list[str] = []

    def info(self, message: str) -> None:
        self.info_lines.append(message)

    def warning(self, message: str) -> None:
        self.warning_lines.append(message)

    def error(self, message: str) -> None:
        self.error_lines.append(message)


@pytest.mark.parametrize("exit_status, level", [(0, "warning"), (1, "error")])
def test_run_local_job_logs_stderr_as_error_only_on_failure(
    tmp_path: Path, monkeypatch, exit_status, level
):
    logger = _LoggerStub()

    class _RuntimeStub:
        async def execute(self, command, **kwargs) -> LocalProcessResult:
            return LocalProcessResult(
                pid=42, exit_status=exit_status, stdout="out", stderr="INFO progress"
            )

    async def fake_create_table_artifact(*, table: list[list[Any]], key: str) -> None:
        pass

    monkeypatch.setattr(run_mod, "LocalRuntime", _RuntimeStub)
    monkeypatch.setattr(run_mod, "get_run_logger", lambda: logger)
    monkeypatch.setattr(run_mod, "create_table_artifact", fake_create_table_artifact)

    result = asyncio.run(
        run_mod.run_local_job(
            work_dir=tmp_path,
            exec_profile=ExecutionProfile(command_key="hello", num_nodes=1, launcher="single"),
            req=LocalJobRequest(executable="/bin/echo"),
        )
    )

    assert result.exit_status == exit_status
    assert logger.info_lines == ["out"]
    assert getattr(logger, f"{level}_lines") == ["INFO progress"]
    other = "error" if level == "warning" else "warning"
    assert getattr(logger, f"{other}_lines") == []
