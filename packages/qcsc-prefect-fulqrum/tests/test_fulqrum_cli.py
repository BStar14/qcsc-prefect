from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from qcsc_prefect_fulqrum import cli, io_utils
from qcsc_prefect_fulqrum.solver import FulqrumSolution


def _write_input(work_dir: Path, **kwargs) -> None:
    io_utils.write_solver_input(
        work_dir=work_dir,
        ci_strings=(np.array([0b01, 0b10]), np.array([0b01])),
        one_body_tensor=np.eye(2),
        two_body_tensor=np.zeros((2, 2, 2, 2)),
        norb=2,
        nelec=(1, 1),
        **kwargs,
    )


def _fake_solve(**problem) -> FulqrumSolution:
    strs_a, strs_b = problem["ci_strings"]
    return FulqrumSolution(
        energy=-0.75,
        orbital_occupancies=(np.array([0.5, 0.5]), np.array([1.0, 0.0])),
        ci_strs_a=strs_a,
        ci_strs_b=strs_b,
        amplitudes=np.full((len(strs_a), len(strs_b)), 2**-0.5),
        metadata={"subspace_dim": 2, "solver_path": "fake", "omp_num_threads": "3"},
    )


def test_cli_solves_and_writes_outputs(monkeypatch, tmp_path: Path, capsys):
    _write_input(tmp_path)
    seen = {}

    def fake_solve(**problem):
        seen.update(problem)
        return _fake_solve(**problem)

    monkeypatch.setattr(cli, "solve_fulqrum", fake_solve)
    assert cli.main([str(tmp_path)]) == 0

    assert seen["norb"] == 2 and seen["nelec"] == (1, 1)
    assert "energy=-0.750000000000" in capsys.readouterr().out
    solution = io_utils.read_solver_output(tmp_path, norb=2)
    assert solution.energy == -0.75
    assert solution.metadata["solver_path"] == "fake"
    assert not (tmp_path / "error.json").exists()


def test_cli_failure_writes_error_json_and_exits_1(monkeypatch, tmp_path: Path):
    _write_input(tmp_path)

    def failing_solve(**problem):
        raise MemoryError("subspace too large")

    monkeypatch.setattr(cli, "solve_fulqrum", failing_solve)
    assert cli.main([str(tmp_path)]) == 1

    error = io_utils.read_solver_error(tmp_path)
    assert error["type"] == "MemoryError"
    assert error["message"] == "subspace too large"
    assert not (tmp_path / "result.json").exists()


def test_cli_missing_input_writes_error_json(tmp_path: Path):
    assert cli.main([str(tmp_path)]) == 1
    assert io_utils.read_solver_error(tmp_path)["type"] == "FileNotFoundError"


def test_cli_removes_stale_outputs(monkeypatch, tmp_path: Path):
    _write_input(tmp_path)
    monkeypatch.setattr(cli, "solve_fulqrum", _fake_solve)
    assert cli.main([str(tmp_path)]) == 0
    try:
        raise RuntimeError("old failure")
    except RuntimeError as exc:
        io_utils.write_solver_error(tmp_path, exc)

    assert cli.main([str(tmp_path)]) == 0
    assert io_utils.read_solver_error(tmp_path) is None


def test_cli_rejects_missing_directory(tmp_path: Path):
    assert cli.main([str(tmp_path / "nope")]) == 2


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({"SLURM_CPUS_PER_TASK": "6"}, "6"),
        ({"SLURM_CPUS_PER_TASK": "6", "OMP_NUM_THREADS": "2"}, "2"),
        ({}, None),
    ],
)
def test_cli_defaults_omp_threads_from_slurm(monkeypatch, env, expected):
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    cli._default_omp_threads()
    assert os.environ.get("OMP_NUM_THREADS") == expected


def test_compute_side_modules_do_not_import_prefect():
    code = (
        "import sys, qcsc_prefect_fulqrum, qcsc_prefect_fulqrum.cli; "
        "bad = sorted(m for m in sys.modules "
        "if m.split('.')[0] in {'prefect', 'qiskit_addon_sqd', 'qcsc_prefect_executor'}); "
        "print(bad); sys.exit(1 if bad else 0)"
    )
    src = Path(cli.__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(src)},
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


@pytest.fixture
def _clear_rank_env(monkeypatch):
    for var in cli._PROCESS_INDEX_ENV_VARS + cli._PROCESS_COUNT_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({}, 0),
        ({"OMPI_COMM_WORLD_RANK": "0"}, 0),
        ({"OMPI_COMM_WORLD_RANK": "3"}, 3),
        ({"PMI_RANK": "2"}, 2),
        ({"PMIX_RANK": "5"}, 5),
        ({"SLURM_PROCID": "7"}, 7),
        # An MPI launcher's own variable wins over srun's.
        ({"OMPI_COMM_WORLD_RANK": "1", "SLURM_PROCID": "4"}, 1),
        # Unrecognizable values must not silence the process.
        ({"PMI_RANK": "not-a-rank"}, 0),
    ],
)
def test_process_index_reads_launcher_environment(_clear_rank_env, monkeypatch, env, expected):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    assert cli.process_index() == expected


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({}, True),
        ({"OMPI_COMM_WORLD_RANK": "0"}, True),
        ({"SLURM_PROCID": "0"}, True),
        ({"OMPI_COMM_WORLD_RANK": "1"}, False),
        ({"PMIX_RANK": "9"}, False),
    ],
)
def test_is_control_process(_clear_rank_env, monkeypatch, env, expected):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    assert cli.is_control_process() is expected


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({}, 1),
        ({"OMPI_COMM_WORLD_SIZE": "4"}, 4),
        ({"PMI_SIZE": "8"}, 8),
        ({"MPI_LOCALNRANKS": "2"}, 2),
        ({"SLURM_NTASKS": "16"}, 16),
        ({"PMI_SIZE": "bogus"}, 1),
    ],
)
def test_process_count_reads_launcher_environment(_clear_rank_env, monkeypatch, env, expected):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    assert cli.process_count() == expected


def test_process_index_needs_no_mpi_library(_clear_rank_env, monkeypatch):
    monkeypatch.setitem(sys.modules, "mpi4py", None)
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "2")
    assert cli.process_index() == 2


def test_noncontrol_process_solves_but_writes_nothing(
    _clear_rank_env, monkeypatch, tmp_path, capsys
):
    _write_input(tmp_path)
    calls = []
    monkeypatch.setattr(cli, "solve_fulqrum", lambda **p: calls.append(p) or _fake_solve(**p))
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "1")

    assert cli.main([str(tmp_path)]) == 0
    # The solve is collective: a non-control process still has to enter it.
    assert len(calls) == 1
    assert not (tmp_path / io_utils.OUTPUT_RESULT).exists()
    assert not (tmp_path / io_utils.OUTPUT_ARRAYS).exists()
    assert capsys.readouterr().out == ""


def test_noncontrol_process_does_not_delete_control_outputs(_clear_rank_env, monkeypatch, tmp_path):
    _write_input(tmp_path)
    monkeypatch.setattr(cli, "solve_fulqrum", _fake_solve)
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "0")
    assert cli.main([str(tmp_path)]) == 0
    before = (tmp_path / io_utils.OUTPUT_RESULT).read_bytes()

    # A late-starting peer must leave the control process's result in place.
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "2")
    assert cli.main([str(tmp_path)]) == 0
    assert (tmp_path / io_utils.OUTPUT_RESULT).read_bytes() == before


def test_every_process_reports_failure(_clear_rank_env, monkeypatch, tmp_path):
    _write_input(tmp_path)

    def _boom(**_problem):
        raise RuntimeError("solve exploded")

    monkeypatch.setattr(cli, "solve_fulqrum", _boom)

    # The scheduler may sample any process, so all of them must exit non-zero.
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "1")
    assert cli.main([str(tmp_path)]) == 1
    assert io_utils.read_solver_error(tmp_path) is None

    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "0")
    assert cli.main([str(tmp_path)]) == 1
    error = io_utils.read_solver_error(tmp_path)
    assert error is not None and "solve exploded" in error["message"]


def test_simulated_two_process_run_writes_one_result(
    _clear_rank_env, monkeypatch, tmp_path, capsys
):
    _write_input(tmp_path)
    monkeypatch.setattr(cli, "solve_fulqrum", _fake_solve)

    for index in ("0", "1"):
        monkeypatch.setenv("OMPI_COMM_WORLD_RANK", index)
        assert cli.main([str(tmp_path)]) == 0

    solution = io_utils.read_solver_output(tmp_path, norb=2)
    assert solution.energy == pytest.approx(-0.75)
    # One summary line, from the control process only.
    assert capsys.readouterr().out.count("qcsc-fulqrum-solve: energy=") == 1
