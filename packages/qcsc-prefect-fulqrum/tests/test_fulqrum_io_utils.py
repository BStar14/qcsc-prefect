from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from qcsc_prefect_fulqrum import io_utils
from qcsc_prefect_fulqrum.solver import FulqrumSolution, FulqrumSolverOptions


def _problem(norb: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed)
    h1 = rng.normal(size=(norb, norb))
    eri = rng.normal(size=(norb,) * 4)
    # Two electrons per spin in four orbitals; unsorted with duplicates.
    strs_a = np.array([0b1100, 0b0011, 0b0101, 0b0011])
    strs_b = np.array([0b1010, 0b0011])
    return (strs_a, strs_b), h1, eri


def test_make_job_work_dir_is_unique(tmp_path: Path):
    first = io_utils.make_job_work_dir(tmp_path / "root")
    second = io_utils.make_job_work_dir(tmp_path / "root")
    assert first != second
    assert first.parent == second.parent == tmp_path / "root"
    assert first.name.startswith("job_")


def test_solver_input_round_trip(tmp_path: Path):
    ci_strings, h1, eri = _problem()
    options = FulqrumSolverOptions(eigensolver="primme", tol=1e-10, max_iter=50)
    io_utils.write_solver_input(
        work_dir=tmp_path,
        ci_strings=ci_strings,
        one_body_tensor=h1,
        two_body_tensor=eri,
        norb=4,
        nelec=(2, 2),
        options=options,
    )
    assert sorted(p.name for p in tmp_path.iterdir()) == ["input.npz", "solver.json"]

    problem = io_utils.read_solver_input(tmp_path)
    strs_a, strs_b = problem["ci_strings"]
    np.testing.assert_array_equal(strs_a, [0b0011, 0b0101, 0b1100])
    np.testing.assert_array_equal(strs_b, [0b0011, 0b1010])
    assert strs_a.dtype == np.int64
    np.testing.assert_array_equal(problem["one_body_tensor"], h1)
    np.testing.assert_array_equal(problem["two_body_tensor"], eri)
    assert problem["norb"] == 4
    assert problem["nelec"] == (2, 2)
    assert problem["options"] == options


def test_write_solver_input_validates_before_writing(tmp_path: Path):
    ci_strings, h1, eri = _problem()
    with pytest.raises(ValueError, match="electrons"):
        io_utils.write_solver_input(
            work_dir=tmp_path,
            ci_strings=ci_strings,
            one_body_tensor=h1,
            two_body_tensor=eri,
            norb=4,
            nelec=(1, 2),
        )
    assert list(tmp_path.iterdir()) == []


def test_read_solver_input_rejects_unknown_format(tmp_path: Path):
    ci_strings, h1, eri = _problem()
    io_utils.write_solver_input(
        work_dir=tmp_path,
        ci_strings=ci_strings,
        one_body_tensor=h1,
        two_body_tensor=eri,
        norb=4,
        nelec=(2, 2),
    )
    spec = json.loads((tmp_path / "solver.json").read_text())
    spec["format_version"] = 99
    (tmp_path / "solver.json").write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="format_version"):
        io_utils.read_solver_input(tmp_path)


@pytest.mark.parametrize("with_amplitudes", [True, False])
def test_solver_output_round_trip(tmp_path: Path, with_amplitudes: bool):
    amplitudes = np.arange(6, dtype=float).reshape(3, 2) if with_amplitudes else None
    solution = FulqrumSolution(
        energy=-1.25,
        orbital_occupancies=(np.array([1.0, 0.5, 0.5, 0.0]), np.array([1.0, 1.0, 0.0, 0.0])),
        ci_strs_a=np.array([3, 5, 12], dtype=np.int64),
        ci_strs_b=np.array([3, 10], dtype=np.int64),
        amplitudes=amplitudes,
        metadata={"subspace_dim": 6, "nnz": np.int64(20), "timings": {"eigensolve": 0.1}},
    )
    io_utils.write_solver_output(tmp_path, solution)

    loaded = io_utils.read_solver_output(tmp_path, norb=4)
    assert loaded.energy == -1.25
    np.testing.assert_array_equal(loaded.orbital_occupancies[0], solution.orbital_occupancies[0])
    np.testing.assert_array_equal(loaded.orbital_occupancies[1], solution.orbital_occupancies[1])
    np.testing.assert_array_equal(loaded.ci_strs_a, solution.ci_strs_a)
    np.testing.assert_array_equal(loaded.ci_strs_b, solution.ci_strs_b)
    if with_amplitudes:
        np.testing.assert_array_equal(loaded.amplitudes, amplitudes)
    else:
        assert loaded.amplitudes is None
    assert loaded.metadata == {"subspace_dim": 6, "nnz": 20, "timings": {"eigensolve": 0.1}}
    assert not list(tmp_path.glob("*.tmp*"))


def test_wide_ci_strings_round_trip_without_pickle(tmp_path: Path):
    norb = 70
    big = (1 << 69) | 1
    solution = FulqrumSolution(
        energy=0.0,
        orbital_occupancies=(np.zeros(norb), np.zeros(norb)),
        ci_strs_a=np.array([3, big], dtype=object),
        ci_strs_b=np.array([big], dtype=object),
        amplitudes=np.ones((2, 1)),
    )
    io_utils.write_solver_output(tmp_path, solution)
    loaded = io_utils.read_solver_output(tmp_path, norb=norb)
    assert loaded.ci_strs_a.dtype == object
    assert [int(x) for x in loaded.ci_strs_a] == [3, big]
    assert [int(x) for x in loaded.ci_strs_b] == [big]


def test_error_json_takes_precedence(tmp_path: Path):
    try:
        raise ValueError("boom")
    except ValueError as exc:
        io_utils.write_solver_error(tmp_path, exc)

    error = io_utils.read_solver_error(tmp_path)
    assert error["type"] == "ValueError"
    assert error["message"] == "boom"
    assert "Traceback" in error["traceback"]
    with pytest.raises(io_utils.FulqrumSolverError, match="ValueError: boom") as info:
        io_utils.read_solver_output(tmp_path, norb=4)
    assert info.value.error["type"] == "ValueError"


def test_missing_outputs_raise_file_not_found(tmp_path: Path):
    assert io_utils.read_solver_error(tmp_path) is None
    with pytest.raises(FileNotFoundError):
        io_utils.read_solver_output(tmp_path, norb=4)
