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
