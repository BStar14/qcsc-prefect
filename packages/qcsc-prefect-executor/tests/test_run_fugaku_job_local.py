from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from qcsc_prefect_adapters.fugaku.builder import FugakuJobRequest
from qcsc_prefect_adapters.fugaku.runtime import SubmitResult
from qcsc_prefect_core.models.execution_profile import ExecutionProfile
from qcsc_prefect_executor.fugaku import run as run_mod


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


class _RuntimeStub:
    def __init__(
        self, final_status: dict[str, Any], calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]]
    ) -> None:
        self._final_status = final_status
        self._calls = calls

    async def submit(self, script_path: Path, *, cwd: Path | None = None) -> SubmitResult:
        self._calls.append(("submit", (script_path,), {"cwd": cwd}))
        return SubmitResult(job_id="43607196", raw_output="Job 43607196 submitted.")

    async def wait_final_status(
        self,
        job_id: str,
        *,
        watch_poll_interval: float = 10.0,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        self._calls.append(
            (
                "wait_final_status",
                (job_id,),
                {"watch_poll_interval": watch_poll_interval, "timeout_seconds": timeout_seconds},
            )
        )
        return self._final_status


def test_run_fugaku_job_local_mock(tmp_path: Path, monkeypatch):
    script_name = "batch.pjm"
    job_name = "pytest-job"
    out_file = tmp_path / f"{script_name}.{job_name}.out"
    err_file = tmp_path / f"{script_name}.{job_name}.err"
    stats_file = tmp_path / f"{script_name}.{job_name}.stats"
    out_file.write_text("hello from fugaku stdout")
    err_file.write_text("hello from fugaku stderr")
    stats_file.write_text(
        "Job Statistical Information\n"
        " JOB ID                    : 43607196\n"
        " HOST NAME                 : fn02sv05\n"
    )

    final_status = {"ST": "EXT", "EC": "0"}
    runtime_calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
    artifact_calls: list[dict[str, Any]] = []
    logger = _LoggerStub()

    def runtime_factory() -> _RuntimeStub:
        return _RuntimeStub(final_status=final_status, calls=runtime_calls)

    async def fake_create_table_artifact(*, table: list[list[Any]], key: str) -> None:
        artifact_calls.append({"table": table, "key": key})

    monkeypatch.setattr(run_mod, "FugakuPJMRuntime", runtime_factory)
    monkeypatch.setattr(run_mod, "get_run_logger", lambda: logger)
    monkeypatch.setattr(run_mod, "create_table_artifact", fake_create_table_artifact)

    profile = ExecutionProfile(
        command_key="hello",
        num_nodes=1,
        launcher="single",
        walltime="00:05:00",
    )
    req = FugakuJobRequest(
        queue_name="small",
        project="ra000000",
        executable="/bin/echo",
        job_name=job_name,
    )

    result = asyncio.run(
        run_mod.run_fugaku_job(
            work_dir=tmp_path,
            script_filename=script_name,
            exec_profile=profile,
            req=req,
            watch_poll_interval=0.01,
            timeout_seconds=5,
            metrics_artifact_key="fugaku-metrics-test",
        )
    )

    assert result.job_id == "43607196"
    assert result.exit_status == 0
    assert result.state == "EXT"
    assert runtime_calls[0][0] == "submit"
    assert runtime_calls[1] == (
        "wait_final_status",
        ("43607196",),
        {"watch_poll_interval": 0.01, "timeout_seconds": 5},
    )
    assert logger.info_lines == ["hello from fugaku stdout"]
    # A successful job's stderr is progress or warnings, not an error.
    assert logger.warning_lines == ["hello from fugaku stderr"]
    assert logger.error_lines == []
    assert len(artifact_calls) == 1
    assert artifact_calls[0]["key"] == "fugaku-metrics-test"
    table_rows = artifact_calls[0]["table"]
    assert "stats.job_id" in table_rows[0]
    assert "stats.host_name" in table_rows[0]


def test_run_fugaku_job_logs_stderr_of_a_failed_job_as_error(tmp_path: Path, monkeypatch):
    job_name = "pytest-job"
    (tmp_path / f"batch.pjm.{job_name}.err").write_text("Traceback: boom")
    final_status = {"ST": "EXT", "EC": "1"}
    logger = _LoggerStub()

    async def fake_create_table_artifact(*, table: list[list[Any]], key: str) -> None:
        pass

    monkeypatch.setattr(run_mod, "FugakuPJMRuntime", lambda: _RuntimeStub(final_status, []))
    monkeypatch.setattr(run_mod, "get_run_logger", lambda: logger)
    monkeypatch.setattr(run_mod, "create_table_artifact", fake_create_table_artifact)

    result = asyncio.run(
        run_mod.run_fugaku_job(
            work_dir=tmp_path,
            script_filename="batch.pjm",
            exec_profile=ExecutionProfile(command_key="hello", num_nodes=1, launcher="single"),
            req=FugakuJobRequest(
                queue_name="small", project="ra000000", executable="/bin/false", job_name=job_name
            ),
            watch_poll_interval=0.01,
        )
    )

    assert result.exit_status == 1
    assert logger.error_lines == ["Traceback: boom"]
    assert logger.warning_lines == []
