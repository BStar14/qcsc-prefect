from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest
from qcsc_prefect_adapters.local.runtime import LocalRuntime, build_local_command
from qcsc_prefect_core.models.execution_profile import ExecutionProfile


def _profile(**overrides) -> ExecutionProfile:
    values = {
        "command_key": "test",
        "num_nodes": 1,
        "launcher": "single",
    }
    values.update(overrides)
    return ExecutionProfile(**values)


def test_execute_runs_directly_with_cwd_environment_and_captured_logs(tmp_path: Path):
    command = [
        sys.executable,
        "-c",
        (
            "import os, pathlib, sys; "
            "print(pathlib.Path.cwd().name); "
            "print(os.environ['LOCAL_TEST_VALUE']); "
            "print('local stderr', file=sys.stderr); "
            "raise SystemExit(7)"
        ),
    ]

    result = asyncio.run(
        LocalRuntime().execute(
            command,
            cwd=tmp_path,
            environments={"LOCAL_TEST_VALUE": "from-profile"},
        )
    )

    assert result.exit_status == 7
    assert result.stdout.splitlines() == [tmp_path.name, "from-profile"]
    assert result.stderr.strip() == "local stderr"


def test_execute_times_out_and_stops_process(tmp_path: Path):
    with pytest.raises(TimeoutError, match="Local command timed out"):
        asyncio.run(
            LocalRuntime().execute(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                cwd=tmp_path,
                timeout_seconds=0.01,
            )
        )


def test_build_local_command_uses_launcher_without_a_shell():
    profile = _profile(
        launcher="mpiexec",
        mpi_options=["-n", "4"],
        arguments=["--input", "value with spaces"],
    )

    command = build_local_command(exec_profile=profile, executable="/opt/bin/solver")

    assert command == [
        "mpiexec",
        "-n",
        "4",
        "/opt/bin/solver",
        "--input",
        "value with spaces",
    ]


def test_build_local_command_single_invokes_only_executable_and_arguments():
    profile = _profile(
        launcher="single",
        mpi_options=["--ignored"],
        arguments=["--input", "data.bin"],
    )

    command = build_local_command(exec_profile=profile, executable="/opt/bin/solver")

    assert command == ["/opt/bin/solver", "--input", "data.bin"]


@pytest.mark.parametrize(
    ("profile_values", "setting_name"),
    [
        ({"modules": ["openmpi"]}, "modules"),
        ({"pre_commands": ["echo setup"]}, "pre_commands"),
    ],
)
def test_build_local_command_rejects_shell_setup(profile_values, setting_name):
    profile = _profile(**profile_values)

    with pytest.raises(ValueError, match=setting_name):
        build_local_command(exec_profile=profile, executable="/opt/bin/solver")


def test_build_local_command_rejects_gpus_per_node():
    with pytest.raises(ValueError, match="gpus_per_node"):
        build_local_command(exec_profile=_profile(gpus_per_node=1), executable="/opt/bin/solver")
