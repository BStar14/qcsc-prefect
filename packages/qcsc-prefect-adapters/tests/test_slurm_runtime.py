from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from qcsc_prefect_adapters.slurm import runtime as runtime_mod


def test_submit_parses_job_id(tmp_path: Path, monkeypatch):
    calls: list[tuple[tuple[str, ...], Path | None]] = []

    async def fake_run_command(*args: str, cwd: Path | None = None) -> str:
        calls.append((args, cwd))
        return "12345;cluster-a\n"

    monkeypatch.setattr(runtime_mod, "run_command", fake_run_command)
    rt = runtime_mod.SlurmRuntime()

    result = asyncio.run(rt.submit(tmp_path / "job.slurm", cwd=tmp_path))

    assert result.job_id == "12345"
    assert result.raw_output == "12345;cluster-a"
    assert calls == [(("sbatch", "--parsable", str(tmp_path / "job.slurm")), tmp_path)]


def test_wait_final_status_parses_sacct_output(monkeypatch):
    async def fake_run_command(*args: str, cwd: Path | None = None) -> str:
        if args[0] == "scontrol":
            return ""
        return (
            "12345|COMPLETED|0:0|00:00:12|32|node001\n"
            "12345.batch|COMPLETED|0:0|00:00:12|32|node001\n"
        )

    monkeypatch.setattr(runtime_mod, "run_command", fake_run_command)
    rt = runtime_mod.SlurmRuntime()

    status = asyncio.run(rt.wait_final_status("12345", watch_poll_interval=0.01, timeout_seconds=3))

    assert status["JobID"] == "12345"
    assert status["State"] == "COMPLETED"
    assert status["ExitCode"] == "0:0"
    assert status["Elapsed"] == "00:00:12"
    assert status["AllocCPUS"] == "32"
    assert status["NodeList"] == "node001"


@pytest.mark.parametrize(
    ("terminal_state", "exit_code"),
    [("COMPLETED", "0:0"), ("FAILED", "1:0"), ("CANCELLED", "0:15")],
)
def test_wait_final_status_without_accounting(monkeypatch, terminal_state, exit_code):
    states = iter(["PENDING", "RUNNING", terminal_state])
    calls: list[tuple[str, ...]] = []

    async def fake_run_command(*args: str, cwd: Path | None = None) -> str:
        calls.append(args)
        if args[0] == "sacct":
            raise RuntimeError("Slurm accounting storage is disabled")
        return (
            f"JobId=12345 JobState={next(states)} ExitCode={exit_code} "
            "RunTime=00:00:12 NumCPUs=1 NodeList=slurm-vm1\n"
        )

    monkeypatch.setattr(runtime_mod, "run_command", fake_run_command)
    status = asyncio.run(
        runtime_mod.SlurmRuntime().wait_final_status(
            "12345", watch_poll_interval=0, timeout_seconds=3
        )
    )

    assert status == {
        "JobID": "12345",
        "State": terminal_state,
        "ExitCode": exit_code,
        "Elapsed": "00:00:12",
        "AllocCPUS": "1",
        "NodeList": "slurm-vm1",
    }
    assert calls == [("scontrol", "show", "job", "-o", "12345")] * 3


def test_wait_final_status_times_out_without_accounting(monkeypatch):
    async def fake_run_command(*args: str, cwd: Path | None = None) -> str:
        assert args[0] == "scontrol"
        return "JobId=12345 JobState=PENDING ExitCode=0:0\n"

    monkeypatch.setattr(runtime_mod, "run_command", fake_run_command)
    with pytest.raises(runtime_mod.WaitTimeout, match="job_id=12345"):
        asyncio.run(
            runtime_mod.SlurmRuntime().wait_final_status(
                "12345", watch_poll_interval=0.01, timeout_seconds=0.02
            )
        )


def test_cancel_invokes_scancel(monkeypatch):
    calls: list[tuple[str, ...]] = []

    async def fake_run_command(*args: str, cwd: Path | None = None) -> str:
        calls.append(args)
        return ""

    monkeypatch.setattr(runtime_mod, "run_command", fake_run_command)
    rt = runtime_mod.SlurmRuntime()

    asyncio.run(rt.cancel("12345"))

    assert calls == [("scancel", "12345")]


def test_scontrol_failure_falls_back_to_accounting(monkeypatch):
    async def fake_run_command(*args, **kwargs):
        if args[0] == "scontrol":
            raise RuntimeError("Job no longer retained by the controller")
        return "12345|COMPLETED|0:0|00:00:01|1|node001\n"

    monkeypatch.setattr(runtime_mod, "run_command", fake_run_command)
    status = asyncio.run(runtime_mod.SlurmRuntime().wait_final_status("12345", timeout_seconds=1))
    assert status["State"] == "COMPLETED"


def test_status_queries_ignore_other_jobs_and_steps(monkeypatch):
    async def fake_run_command(*args, **kwargs):
        if args[0] == "scontrol":
            return "JobId=99999 JobState=COMPLETED ExitCode=0:0\n"
        return (
            "99999|COMPLETED|0:0|00:00:01|1|node001\n"
            "12345.batch|COMPLETED|0:0|00:00:01|1|node001\n"
            "12345|CANCELLED by 1000|0:15|00:00:01|1|node001\n"
        )

    monkeypatch.setattr(runtime_mod, "run_command", fake_run_command)
    status = asyncio.run(runtime_mod.SlurmRuntime().wait_final_status("12345", timeout_seconds=1))
    assert status["JobID"] == "12345"
    assert status["State"] == "CANCELLED by 1000"


def test_transient_scheduler_failures_are_retried(monkeypatch):
    calls = []

    async def fake_run_command(*args, **kwargs):
        calls.append(args[0])
        if len(calls) <= 2:
            raise RuntimeError("Temporary scheduler failure")
        return "JobId=12345 JobState=COMPLETED ExitCode=0:0\n"

    monkeypatch.setattr(runtime_mod, "run_command", fake_run_command)
    status = asyncio.run(
        runtime_mod.SlurmRuntime().wait_final_status(
            "12345",
            watch_poll_interval=0,
            timeout_seconds=1,
        )
    )
    assert status["State"] == "COMPLETED"
    assert calls == ["scontrol", "sacct", "scontrol"]
