"""Real Slurm round trip: block helpers -> run_job_from_blocks -> sbatch -> CLI.

No Prefect server is used. Blocks built by ``create_fulqrum_blocks`` are kept in
memory and served to ``run_job_from_blocks``; the Prefect logger and artifact
calls are stubbed. Run it from the repository root on a Slurm submit host whose
job root is shared with the compute nodes:

    SLURM_INTEGRATION=1 python -m pytest \
        packages/qcsc-prefect-fulqrum/tests/test_fulqrum_slurm_integration.py

Environment:
    SLURM_INTEGRATION=1        enable the tests
    FULQRUM_SLURM_ROOT         shared job root (default /data/fulqrum-integration)
    FULQRUM_SLURM_EXECUTABLE   default /opt/venvs/fulqrum/bin/qcsc-fulqrum-solve
    FULQRUM_SLURM_PARTITION    default normal
    FULQRUM_SLURM_OMPTHREADS   default 4
    FULQRUM_SLURM_TIMEOUT      seconds, default 600
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytestmark = pytest.mark.slurm_integration


def _env_enabled(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


if not _env_enabled("SLURM_INTEGRATION"):
    pytest.skip("Set SLURM_INTEGRATION=1 to run the Slurm round trip.", allow_module_level=True)
if shutil.which("sbatch") is None:
    pytest.skip("sbatch not found. Run on a Slurm login node.", allow_module_level=True)
pytest.importorskip("pyscf")

from pyscf import ao2mo, fci, gto, scf  # noqa: E402
from pyscf.fci import cistring  # noqa: E402
from qcsc_prefect_executor import from_blocks  # noqa: E402
from qcsc_prefect_executor.slurm import run as slurm_run  # noqa: E402
from qcsc_prefect_fulqrum import block_utils, io_utils, solver_job  # noqa: E402
from qiskit_addon_sqd.fermion import solve_fermion  # noqa: E402

ROOT = Path(os.getenv("FULQRUM_SLURM_ROOT", "/data/fulqrum-integration"))
EXECUTABLE = os.getenv("FULQRUM_SLURM_EXECUTABLE", "/opt/venvs/fulqrum/bin/qcsc-fulqrum-solve")
PARTITION = os.getenv("FULQRUM_SLURM_PARTITION", "normal")
OMPTHREADS = int(os.getenv("FULQRUM_SLURM_OMPTHREADS", "4"))
TIMEOUT = float(os.getenv("FULQRUM_SLURM_TIMEOUT", "600"))


class _LoggerStub:
    def __init__(self):
        self.lines: list[str] = []

    def info(self, msg, *args):
        self.lines.append(str(msg) % args if args else str(msg))

    warning = error = debug = info


@pytest.fixture
def blocks(monkeypatch, request):
    """Create the fulqrum block set in memory and serve it to run_job_from_blocks."""

    store: dict[str, Any] = {}

    def fake_save(self, name, overwrite=False):
        store[name] = self
        return self

    async def fake_artifact(*, table, key):
        store.setdefault("_artifacts", []).append((key, table))

    def loader(name):
        async def _load(block_name):
            return store[block_name]

        return type(name, (), {"load": staticmethod(_load)})

    for cls in (
        block_utils.CommandBlock,
        block_utils.ExecutionProfileBlock,
        block_utils.HPCProfileBlock,
        block_utils.FulqrumSolverJob,
    ):
        monkeypatch.setattr(cls, "save", fake_save, raising=False)
    monkeypatch.setattr(block_utils, "register_fulqrum_block_types", lambda: None)
    monkeypatch.setattr(from_blocks, "CommandBlock", loader("CommandBlock"))
    monkeypatch.setattr(from_blocks, "ExecutionProfileBlock", loader("ExecutionProfileBlock"))
    monkeypatch.setattr(from_blocks, "HPCProfileBlock", loader("HPCProfileBlock"))
    logger = _LoggerStub()
    monkeypatch.setattr(slurm_run, "get_run_logger", lambda: logger)
    monkeypatch.setattr(slurm_run, "create_table_artifact", fake_artifact)

    root = ROOT / request.node.name.replace("[", "_").replace("]", "")
    shutil.rmtree(root, ignore_errors=True)
    names = block_utils.create_fulqrum_blocks(
        hpc_target="slurm",
        queue=PARTITION,
        root_dir=str(root),
        fulqrum_executable=EXECUTABLE,
        ompthreads=OMPTHREADS,
        walltime="00:10:00",
        timeout_seconds=TIMEOUT,
        dense_max_dim=0,
        tol=1e-12,
    )
    store["_names"] = names
    store["_root"] = root
    store["_logger"] = logger
    return store


def _lih():
    mol = gto.M(atom="Li 0 0 0; H 0 0 1.6", basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run()
    norb = mf.mo_coeff.shape[1]
    h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    eri = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), norb)
    return np.ascontiguousarray(h1), np.ascontiguousarray(eri), norb, (2, 2)


def _run_solver(store, ci_strings, h1, eri, norb, nelec):
    solver = store[store["_names"]["solver_block_name"]]
    result = asyncio.run(
        solver_job._run_fulqrum_inner.fn(
            solver=solver,
            ci_strings=ci_strings,
            one_body_tensor=h1,
            two_body_tensor=eri,
            norb=norb,
            nelec=nelec,
        )
    )
    (work_dir,) = sorted(store["_root"].glob("job_*"))
    return result, work_dir


def test_full_space_round_trip_matches_fci(blocks):
    h1, eri, norb, nelec = _lih()
    strs = np.asarray(cistring.make_strings(range(norb), 2), dtype=np.int64)
    e_fci, civec = fci.direct_spin1.kernel(h1, eri, norb, nelec, conv_tol=1e-12)

    result, work_dir = _run_solver(blocks, (strs, strs), h1, eri, norb, nelec)

    assert result.energy == pytest.approx(e_fci, abs=1e-9)
    amps = result.sci_state.amplitudes
    sign = np.sign(np.vdot(amps, civec)) or 1.0
    np.testing.assert_allclose(sign * amps, civec, atol=1e-5)

    script = (work_dir / "fulqrum_solver.slurm").read_text()
    assert f"#SBATCH --partition={PARTITION}" in script
    assert f"#SBATCH --cpus-per-task={OMPTHREADS}" in script
    assert f'export OMP_NUM_THREADS="{OMPTHREADS}"' in script
    assert f'"${{QCSC_PREFECT_EXECUTABLE}}" {work_dir}' in script

    meta = json.loads((work_dir / "result.json").read_text())["metadata"]
    assert meta["solver_path"] == "scipy-csr"
    assert meta["omp_num_threads"] == str(OMPTHREADS)
    assert meta["subspace_dim"] == len(strs) ** 2
    assert "qcsc-fulqrum-solve: energy=" in (work_dir / "output.out").read_text()
    (key, table), *_ = blocks["_artifacts"]
    assert key == "slurm-fulqrum-metrics"
    row = dict(zip(*table, strict=True))
    assert row["state"] == "COMPLETED" and row["exit_code"] == "0"


def test_subspace_round_trip_matches_solve_fermion(blocks):
    h1, eri, norb, nelec = _lih()
    strs = np.asarray(cistring.make_strings(range(norb), 2), dtype=np.int64)
    rng = np.random.default_rng(7)
    strs_a = rng.choice(strs, size=9, replace=False)
    strs_b = rng.choice(strs, size=6, replace=False)

    result, _ = _run_solver(blocks, (strs_a, strs_b), h1, eri, norb, nelec)
    e_ref, state_ref, occ_ref, _ = solve_fermion((strs_a, strs_b), h1, eri, open_shell=True)

    assert result.energy == pytest.approx(e_ref, abs=1e-9)
    np.testing.assert_array_equal(result.sci_state.ci_strs_a, state_ref.ci_strs_a)
    np.testing.assert_array_equal(result.sci_state.ci_strs_b, state_ref.ci_strs_b)
    np.testing.assert_allclose(result.orbital_occupancies[0], occ_ref[0], atol=1e-8)
    np.testing.assert_allclose(result.orbital_occupancies[1], occ_ref[1], atol=1e-8)
    np.testing.assert_allclose(
        result.sci_state.orbital_occupancies()[0], result.orbital_occupancies[0], atol=1e-8
    )


def test_failed_solve_propagates_exit_status_and_error_json(blocks):
    h1, eri, norb, nelec = _lih()
    strs = np.asarray(cistring.make_strings(range(norb), 2), dtype=np.int64)
    work_dir = io_utils.make_job_work_dir(blocks["_root"])
    io_utils.write_solver_input(
        work_dir=work_dir,
        ci_strings=(strs, strs),
        one_body_tensor=h1,
        two_body_tensor=eri,
        norb=norb,
        nelec=nelec,
    )
    spec = json.loads((work_dir / "solver.json").read_text())
    spec["options"]["eigensolver"] = "not-a-solver"
    (work_dir / "solver.json").write_text(json.dumps(spec))

    names = blocks["_names"]
    result = asyncio.run(
        from_blocks.run_job_from_blocks(
            command_block_name=names["command_block_name"],
            execution_profile_block_name=names["execution_profile_block_name"],
            hpc_profile_block_name=names["hpc_profile_block_name"],
            work_dir=work_dir,
            script_filename="fulqrum_solver",
            user_args=[str(work_dir)],
            watch_poll_interval=2.0,
            timeout_seconds=TIMEOUT,
        )
    )

    assert result.exit_status == 1
    assert result.state.startswith("FAILED")
    with pytest.raises(io_utils.FulqrumSolverError, match="Unknown eigensolver"):
        io_utils.read_solver_output(work_dir, norb)
