from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

# Prefect-side modules; the compute-node venv has no Prefect.
pytest.importorskip("prefect")
pytest.importorskip("qiskit_addon_sqd")

from qcsc_prefect_fulqrum import io_utils  # noqa: E402
from qcsc_prefect_fulqrum import solver_job as mod  # noqa: E402
from qcsc_prefect_fulqrum.solver import FulqrumSolution, FulqrumSolverOptions  # noqa: E402
from qcsc_prefect_fulqrum.solver_job import FulqrumSolverJob  # noqa: E402
from qiskit_addon_sqd.fermion import SCIResult, SCIState  # noqa: E402

CI_STRINGS = (np.array([0b01, 0b10]), np.array([0b01]))
H1 = np.diag([-1.0, -0.5])
ERI = np.zeros((2, 2, 2, 2))


def _solution(amplitudes: bool = True) -> FulqrumSolution:
    return FulqrumSolution(
        energy=-1.5,
        orbital_occupancies=(np.array([0.8, 0.2]), np.array([1.0, 0.0])),
        ci_strs_a=np.array([1, 2], dtype=np.int64),
        ci_strs_b=np.array([1], dtype=np.int64),
        amplitudes=np.array([[0.8**0.5], [0.2**0.5]]) if amplitudes else None,
        metadata={"subspace_dim": 2, "solver_path": "dense"},
    )


def _run(solver: FulqrumSolverJob, **kwargs):
    return asyncio.run(
        mod._run_fulqrum_inner.fn(
            solver=solver,
            ci_strings=CI_STRINGS,
            one_body_tensor=H1,
            two_body_tensor=ERI,
            norb=2,
            nelec=(1, 1),
            **kwargs,
        )
    )


def _fake_scheduler(monkeypatch, *, exit_status: int = 0, on_run=None):
    executed: dict[str, object] = {}

    async def fake_run_job_from_blocks(**kwargs):
        executed.update(kwargs)
        if on_run is not None:
            on_run(Path(kwargs["work_dir"]))
        return SimpleNamespace(exit_status=exit_status)

    monkeypatch.setattr(mod, "run_job_from_blocks", fake_run_job_from_blocks)
    return executed


def test_run_fulqrum_inner_runs_through_block_executor(monkeypatch, tmp_path: Path):
    root = tmp_path / "fulqrum_jobs"
    seen_inputs = {}

    def on_run(work_dir: Path):
        seen_inputs.update(io_utils.read_solver_input(work_dir))
        io_utils.write_solver_output(work_dir, _solution())

    executed = _fake_scheduler(monkeypatch, on_run=on_run)
    solver = FulqrumSolverJob(root_dir=str(root), eigensolver="primme", tol=1e-9)

    result = _run(solver)

    assert executed["command_block_name"] == "cmd-fulqrum-solver"
    assert executed["execution_profile_block_name"] == "exec-fulqrum-slurm"
    assert executed["hpc_profile_block_name"] == "hpc-slurm-fulqrum"
    assert executed["script_filename"] == "fulqrum_solver.slurm"
    work_dir = Path(executed["work_dir"])
    assert work_dir.parent == root.resolve()
    assert executed["user_args"] == [str(work_dir)]
    assert seen_inputs["options"] == FulqrumSolverOptions(eigensolver="primme", tol=1e-9)

    assert isinstance(result, SCIResult)
    assert result.energy == -1.5
    assert isinstance(result.sci_state, SCIState)
    np.testing.assert_array_equal(result.sci_state.ci_strs_a, [1, 2])
    np.testing.assert_allclose(result.sci_state.amplitudes[:, 0] ** 2, [0.8, 0.2])
    np.testing.assert_allclose(result.orbital_occupancies[0], [0.8, 0.2])


def test_return_sci_state_false_skips_amplitudes(monkeypatch, tmp_path: Path):
    def on_run(work_dir: Path):
        assert io_utils.read_solver_input(work_dir)["options"].return_amplitudes is False
        io_utils.write_solver_output(work_dir, _solution(amplitudes=False))

    _fake_scheduler(monkeypatch, on_run=on_run)
    result = _run(FulqrumSolverJob(root_dir=str(tmp_path), return_sci_state=False))
    assert result.sci_state is None
    assert result.energy == -1.5


def test_spin_sq_is_ignored_with_warning(monkeypatch, tmp_path: Path, caplog):
    _fake_scheduler(monkeypatch, on_run=lambda wd: io_utils.write_solver_output(wd, _solution()))
    with caplog.at_level(logging.WARNING):
        result = _run(FulqrumSolverJob(root_dir=str(tmp_path)), spin_sq=0.0)
    assert result.energy == -1.5
    assert "ignores spin_sq" in caplog.text


def test_solver_error_is_reported(monkeypatch, tmp_path: Path):
    def on_run(work_dir: Path):
        try:
            raise MemoryError("subspace too large")
        except MemoryError as exc:
            io_utils.write_solver_error(work_dir, exc)

    _fake_scheduler(monkeypatch, exit_status=1, on_run=on_run)
    with pytest.raises(RuntimeError, match="MemoryError: subspace too large.*exit_status=1"):
        _run(FulqrumSolverJob(root_dir=str(tmp_path)))


def test_missing_outputs_are_reported(monkeypatch, tmp_path: Path):
    _fake_scheduler(monkeypatch, exit_status=127)
    with pytest.raises(RuntimeError, match="output file not found.*exit_status=127"):
        _run(FulqrumSolverJob(root_dir=str(tmp_path)))


def test_invalid_problem_fails_before_submission(monkeypatch, tmp_path: Path):
    executed = _fake_scheduler(monkeypatch)
    with pytest.raises(ValueError, match="electrons"):
        asyncio.run(
            mod._run_fulqrum_inner.fn(
                solver=FulqrumSolverJob(root_dir=str(tmp_path)),
                ci_strings=(np.array([0b11]), np.array([0b01])),
                one_body_tensor=H1,
                two_body_tensor=ERI,
                norb=2,
                nelec=(1, 1),
            )
        )
    assert executed == {}


def test_block_validates_solver_fields(tmp_path: Path):
    with pytest.raises(ValueError):
        FulqrumSolverJob(root_dir=str(tmp_path), eigensolver="lanczos")
    with pytest.raises(ValueError):
        FulqrumSolverJob(root_dir=str(tmp_path), tol=-1.0)
    options = FulqrumSolverJob(
        root_dir=str(tmp_path), hamiltonian_format="matrix_free", max_iter=10, dense_max_dim=0
    ).solver_options()
    assert options == FulqrumSolverOptions(
        hamiltonian_format="matrix_free", max_iter=10, dense_max_dim=0
    )


def test_package_exports_prefect_side_lazily():
    import qcsc_prefect_fulqrum as pkg

    assert pkg.FulqrumSolverJob is FulqrumSolverJob
    assert callable(pkg.create_fulqrum_blocks)
    with pytest.raises(AttributeError):
        pkg.not_a_name  # noqa: B018
