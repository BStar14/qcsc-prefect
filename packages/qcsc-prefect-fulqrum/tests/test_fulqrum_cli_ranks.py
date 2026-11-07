"""Per-process error files and MPI_Abort in ``qcsc-fulqrum-solve``, with a fake mpi4py."""

from __future__ import annotations

import os
import sys
import time
import types
from pathlib import Path

import numpy as np
import pytest
from qcsc_prefect_fulqrum import cli, io_utils
from qcsc_prefect_fulqrum.solver import FulqrumSolution, FulqrumSolverOptions


@pytest.fixture(autouse=True)
def _clear_rank_env(monkeypatch):
    for var in (*cli._PROCESS_INDEX_ENV_VARS, *cli._PROCESS_COUNT_ENV_VARS):
        monkeypatch.delenv(var, raising=False)


class FakeComm:
    """One process's view of COMM_WORLD; ``peers_ok`` is what the others report."""

    def __init__(self, rank, size, peers_ok=True):
        self.rank, self.size, self.peers_ok = rank, size, peers_ok
        self.events = []

    def Get_rank(self):
        return self.rank

    def Get_size(self):
        return self.size

    def Barrier(self):
        self.events.append("barrier")

    def allgather(self, value):
        self.events.append("allgather")
        return [value if r == self.rank else self.peers_ok for r in range(self.size)]

    def Abort(self, code):
        self.events.append(("abort", code))


@pytest.fixture
def fake_mpi(monkeypatch):
    def install(rank, size, peers_ok=True):
        monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", str(size))
        comm = FakeComm(rank, size, peers_ok)
        mpi = types.SimpleNamespace(COMM_WORLD=comm)
        monkeypatch.setitem(sys.modules, "mpi4py", types.SimpleNamespace(MPI=mpi))
        monkeypatch.setitem(sys.modules, "mpi4py.MPI", mpi)
        return comm

    return install


def _write_input(work_dir: Path, backend: str) -> None:
    io_utils.write_solver_input(
        work_dir=work_dir,
        ci_strings=(np.array([0b01, 0b10]), np.array([0b01])),
        one_body_tensor=np.eye(2),
        two_body_tensor=np.zeros((2, 2, 2, 2)),
        norb=2,
        nelec=(1, 1),
        options=FulqrumSolverOptions(backend=backend),
    )


def _fake_solve(**problem) -> FulqrumSolution:
    strs_a, strs_b = problem["ci_strings"]
    return FulqrumSolution(
        energy=-0.5,
        orbital_occupancies=(np.array([0.5, 0.5]), np.array([1.0, 0.0])),
        ci_strs_a=strs_a,
        ci_strs_b=strs_b,
        amplitudes=None,
        metadata={"subspace_dim": 2, "solver_path": "fake"},
    )


def _boom(**_problem):
    raise RuntimeError("rank exploded")


def _raise(exc: BaseException):
    try:
        raise exc
    except type(exc) as caught:
        return caught


@pytest.mark.parametrize("backend", ["cpu", "gpu", "gpu-mpi"])
def test_a_multi_process_launch_uses_the_comm_world_rank(
    monkeypatch, tmp_path, fake_mpi, capsys, backend
):
    _write_input(tmp_path, backend)
    monkeypatch.setattr(cli, "solve_fulqrum", _fake_solve)
    # The launcher says rank 0, COMM_WORLD says rank 1: COMM_WORLD wins.
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "0")
    comm = fake_mpi(rank=1, size=2)
    assert cli.main([str(tmp_path)]) == 0
    assert not (tmp_path / io_utils.OUTPUT_RESULT).exists()
    assert capsys.readouterr().out == ""
    assert comm.events == ["barrier", "allgather"]

    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "1")
    fake_mpi(rank=0, size=2)
    assert cli.main([str(tmp_path)]) == 0
    assert io_utils.read_solver_output(tmp_path, norb=2).energy == -0.5


def test_a_single_process_launch_needs_no_mpi(monkeypatch, tmp_path, fake_mpi):
    _write_input(tmp_path, "gpu-mpi")
    monkeypatch.setattr(cli, "solve_fulqrum", _fake_solve)
    comm = fake_mpi(rank=0, size=4)
    monkeypatch.delenv("OMPI_COMM_WORLD_SIZE")
    assert cli.main([str(tmp_path)]) == 0
    assert comm.events == []


@pytest.mark.parametrize("backend", ["cpu", "gpu", "gpu-mpi"])
def test_a_failing_rank_writes_its_file_and_aborts(monkeypatch, tmp_path, fake_mpi, backend):
    _write_input(tmp_path, backend)
    monkeypatch.setattr(cli, "solve_fulqrum", _boom)
    comm = fake_mpi(rank=1, size=4)

    assert cli.main([str(tmp_path)]) == 1
    assert comm.events == ["barrier", "allgather", ("abort", 1)]
    assert not (tmp_path / io_utils.OUTPUT_ERROR).exists()
    error = io_utils.read_solver_error(tmp_path)
    assert (error["rank"], error["type"], error["message"]) == (1, "RuntimeError", "rank exploded")

    # The submit side sees the failure although rank 0 wrote nothing.
    with pytest.raises(io_utils.FulqrumSolverError, match="on rank 1: RuntimeError") as info:
        io_utils.read_solver_output(tmp_path, norb=2)
    assert info.value.error["rank"] == 1


def test_a_failing_rank_fails_the_solve_even_with_a_result(monkeypatch, tmp_path, fake_mpi):
    _write_input(tmp_path, "cpu")
    monkeypatch.setattr(cli, "solve_fulqrum", _fake_solve)
    fake_mpi(rank=0, size=2)
    assert cli.main([str(tmp_path)]) == 0
    io_utils.write_solver_error(tmp_path, _raise(MemoryError("rank 1 out of memory")), rank=1)
    with pytest.raises(io_utils.FulqrumSolverError, match="on rank 1: MemoryError"):
        io_utils.read_solver_output(tmp_path, norb=2)


def test_rank_0_failure_writes_error_json_and_aborts(monkeypatch, tmp_path, fake_mpi):
    _write_input(tmp_path, "gpu-mpi")
    monkeypatch.setattr(cli, "solve_fulqrum", _boom)
    comm = fake_mpi(rank=0, size=2)
    assert cli.main([str(tmp_path)]) == 1
    assert comm.events[-1] == ("abort", 1)
    assert io_utils.read_solver_error(tmp_path)["rank"] == 0
    assert (tmp_path / io_utils.OUTPUT_ERROR).exists()


def test_an_input_read_failure_on_any_rank_stops_every_rank(monkeypatch, tmp_path, fake_mpi):
    """A rank that cannot read the input must not leave its peers waiting in the solve."""
    _write_input(tmp_path, "gpu-mpi")
    solved = []
    monkeypatch.setattr(cli, "solve_fulqrum", lambda **p: solved.append(p) or _fake_solve(**p))

    # This rank reads the input; a peer does not.
    comm = fake_mpi(rank=0, size=2, peers_ok=False)
    assert cli.main([str(tmp_path)]) == 1
    assert solved == []
    assert comm.events == ["barrier", "allgather"]
    assert "another process" in io_utils.read_solver_error(tmp_path)["message"]

    # This rank cannot read it: it reports its own error, and nobody solves.
    (tmp_path / io_utils.INPUT_SPEC).write_text("{not json", encoding="utf-8")
    comm = fake_mpi(rank=1, size=2)
    assert cli.main([str(tmp_path)]) == 1
    assert solved == []
    assert comm.events == ["barrier", "allgather"]
    error = io_utils.read_solver_error(tmp_path)
    assert error["type"] == "RuntimeError"  # rank 0's error.json from the first call
    assert (tmp_path / "error.rank1.json").exists()


def test_without_mpi4py_the_environment_index_is_used(monkeypatch, tmp_path):
    _write_input(tmp_path, "gpu-mpi")
    monkeypatch.setitem(sys.modules, "mpi4py", None)
    monkeypatch.setattr(cli, "solve_fulqrum", _boom)
    monkeypatch.setenv("PMIX_SIZE", "4")
    monkeypatch.setenv("PMIX_RANK", "3")
    assert cli.main([str(tmp_path)]) == 1
    assert io_utils.read_solver_error(tmp_path)["rank"] == 3


def test_control_process_clears_stale_rank_errors(monkeypatch, tmp_path, fake_mpi):
    _write_input(tmp_path, "gpu-mpi")
    io_utils.write_solver_error(tmp_path, _raise(RuntimeError("old")), rank=3)
    io_utils.write_solver_error(tmp_path, _raise(RuntimeError("old")), rank=0)
    monkeypatch.setattr(cli, "solve_fulqrum", _fake_solve)
    fake_mpi(rank=0, size=4)
    assert cli.main([str(tmp_path)]) == 0
    assert io_utils.read_solver_error(tmp_path) is None
    assert io_utils.read_solver_output(tmp_path, norb=2).energy == -0.5


def test_without_a_communicator_only_old_rank_errors_are_cleared(monkeypatch, tmp_path):
    _write_input(tmp_path, "cpu")
    for rank in (1, 2):
        io_utils.write_solver_error(tmp_path, _raise(RuntimeError("old")), rank=rank)
    old = time.time() - 3600
    os.utime(tmp_path / "error.rank1.json", (old, old))
    monkeypatch.setattr(cli, "solve_fulqrum", _fake_solve)
    assert cli.main([str(tmp_path)]) == 0
    # The old file is from an earlier run; the fresh one may be a peer's of this run.
    assert not (tmp_path / "error.rank1.json").exists()
    assert (tmp_path / "error.rank2.json").exists()


def test_each_process_clears_its_own_stale_error(monkeypatch, tmp_path):
    _write_input(tmp_path, "cpu")
    for rank in (1, 2):
        io_utils.write_solver_error(tmp_path, _raise(RuntimeError("old")), rank=rank)
    monkeypatch.setattr(cli, "solve_fulqrum", _fake_solve)
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "1")
    assert cli.main([str(tmp_path)]) == 0
    assert not (tmp_path / "error.rank1.json").exists()
    assert (tmp_path / "error.rank2.json").exists()


def test_unreadable_spec_fails_with_its_own_error(tmp_path):
    (tmp_path / io_utils.INPUT_SPEC).write_text("{not json", encoding="utf-8")
    assert cli.main([str(tmp_path)]) == 1
    assert io_utils.read_solver_error(tmp_path)["type"] == "JSONDecodeError"


# --- io_utils ---------------------------------------------------------------


def test_error_file_names():
    assert io_utils.error_file_name(0) == "error.json"
    assert io_utils.error_file_name(7) == "error.rank7.json"


def test_read_solver_output_sees_error_rank1(tmp_path):
    io_utils.write_solver_error(tmp_path, _raise(ValueError("tile mismatch")), rank=1)
    with pytest.raises(io_utils.FulqrumSolverError, match="on rank 1: ValueError: tile mismatch"):
        io_utils.read_solver_output(tmp_path, norb=2)


def test_error_json_comes_before_rank_files(tmp_path):
    io_utils.write_solver_error(tmp_path, _raise(ValueError("rank 2")), rank=2)
    io_utils.write_solver_error(tmp_path, _raise(ValueError("rank 10")), rank=10)
    assert io_utils.read_solver_error(tmp_path)["message"] == "rank 2"
    io_utils.write_solver_error(tmp_path, _raise(ValueError("rank 0")))
    assert io_utils.read_solver_error(tmp_path)["message"] == "rank 0"
    names = [p.name for p in io_utils.solver_error_files(tmp_path)]
    assert names == ["error.json", "error.rank2.json", "error.rank10.json"]


def test_unrelated_files_are_not_error_files(tmp_path):
    (tmp_path / "error.rankX.json").write_text("{}", encoding="utf-8")
    (tmp_path / "error.rank1.json.tmp").write_text("{}", encoding="utf-8")
    assert io_utils.read_solver_error(tmp_path) is None


def test_unreadable_rank_error_is_reported(tmp_path):
    (tmp_path / "error.rank4.json").write_text("{", encoding="utf-8")
    error = io_utils.read_solver_error(tmp_path)
    assert "unreadable error.rank4.json" in error["message"]
