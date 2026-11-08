"""Real Slurm round trip: block helpers -> run_job_from_blocks -> sbatch -> SBD driver.

No Prefect server is used. Blocks built by ``create_sbd_blocks`` are kept in memory
and served to ``run_job_from_blocks``; the Prefect logger and artifact calls are
stubbed. Run it from the repository root on a Slurm submit host whose job root is
shared with the compute nodes, after building the driver with
``native/build_sbd_slurm.sh``:

    SLURM_INTEGRATION=1 python -m pytest \
        packages/qcsc-prefect-sbd/tests/test_sbd_slurm_integration.py

Environment:
    SLURM_INTEGRATION=1     enable the tests
    SBD_SLURM_EXECUTABLE    default packages/qcsc-prefect-sbd/native/diag of this checkout
    SBD_SLURM_ROOT          shared job root (default /data/sbd-integration)
    SBD_SLURM_PARTITION     default normal
    SBD_SLURM_NODES         default 2
    SBD_SLURM_MPIPROCS      MPI ranks per node, default 2
    SBD_SLURM_TIMEOUT       seconds, default 600
"""

from __future__ import annotations

import asyncio
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

from pyscf import ao2mo, fci, gto, mcscf, scf  # noqa: E402
from pyscf.fci import cistring  # noqa: E402
from qcsc_prefect_executor import from_blocks  # noqa: E402
from qcsc_prefect_executor.slurm import run as slurm_run  # noqa: E402
from qcsc_prefect_sbd import block_utils, solver_job  # noqa: E402

DEFAULT_EXECUTABLE = Path(__file__).resolve().parents[1] / "native" / "diag"
EXECUTABLE = os.getenv("SBD_SLURM_EXECUTABLE", str(DEFAULT_EXECUTABLE))
ROOT = Path(os.getenv("SBD_SLURM_ROOT", "/data/sbd-integration"))
PARTITION = os.getenv("SBD_SLURM_PARTITION", "normal")
NODES = int(os.getenv("SBD_SLURM_NODES", "2"))
MPIPROCS = int(os.getenv("SBD_SLURM_MPIPROCS", "2"))
TIMEOUT = float(os.getenv("SBD_SLURM_TIMEOUT", "600"))


class _LoggerStub:
    def __init__(self):
        self.lines: list[str] = []

    def info(self, msg, *args):
        self.lines.append(str(msg) % args if args else str(msg))

    warning = error = debug = info


@pytest.fixture
def blocks(monkeypatch, request):
    """Create the SBD block set in memory and serve it to run_job_from_blocks."""

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
        block_utils.SBDSolverJob,
    ):
        monkeypatch.setattr(cls, "save", fake_save, raising=False)
    monkeypatch.setattr(block_utils, "register_sbd_block_types", lambda: None)
    monkeypatch.setattr(from_blocks, "CommandBlock", loader("CommandBlock"))
    monkeypatch.setattr(from_blocks, "ExecutionProfileBlock", loader("ExecutionProfileBlock"))
    monkeypatch.setattr(from_blocks, "HPCProfileBlock", loader("HPCProfileBlock"))
    logger = _LoggerStub()
    monkeypatch.setattr(slurm_run, "get_run_logger", lambda: logger)
    monkeypatch.setattr(slurm_run, "create_table_artifact", fake_artifact)
    monkeypatch.setattr(solver_job, "get_run_logger", lambda: logger)

    root = ROOT / request.node.name.replace("[", "_").replace("]", "")
    shutil.rmtree(root, ignore_errors=True)
    names = block_utils.create_sbd_blocks(
        hpc_target="slurm",
        queue=PARTITION,
        root_dir=str(root),
        sbd_executable=EXECUTABLE,
        launcher="srun",
        mpi_options=["--mpi=pmix"],
        num_nodes=NODES,
        mpiprocs=MPIPROCS,
        ompthreads=1,
        # The job starts in its work directory (the sbatch submit directory).
        pre_commands=["srun hostname > hosts.txt"],
        walltime="00:10:00",
        timeout_seconds=TIMEOUT,
        block=20,
        iteration=10,
        tolerance=1e-8,
    )
    store["_names"] = names
    store["_root"] = root
    store["_logger"] = logger
    return store


def _active_space(mf, ncas, nelecas):
    # Frozen-core active-space integrals. The core energy is left out of the FCIDUMP
    # (ECORE = 0), so SBD and the FCI reference both return the active-space electronic
    # energy.
    cas = mcscf.CASCI(mf, ncas, nelecas)
    h1, _ = cas.get_h1eff()
    eri = ao2mo.restore(1, cas.get_h2eff(), ncas)
    return np.ascontiguousarray(h1), np.ascontiguousarray(eri)


def _water():
    mol = gto.M(atom="O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587", basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run()
    norb, nelec = 6, (4, 4)
    return (*_active_space(mf, norb, nelec), norb, nelec)


def _triplet_methylene():
    mol = gto.M(atom="C 0 0 0; H 0 0.99 0.62; H 0 -0.99 0.62", basis="sto-3g", spin=2, verbose=0)
    mf = scf.ROHF(mol).run()
    norb, nelec = 6, (4, 2)
    return (*_active_space(mf, norb, nelec), norb, nelec)


def _full_space(norb, nelec):
    # cistring orders the strings ascending, so the Hartree-Fock string comes first.
    strs_a = np.asarray(cistring.make_strings(range(norb), nelec[0]), dtype=np.int64)
    strs_b = np.asarray(cistring.make_strings(range(norb), nelec[1]), dtype=np.int64)
    assert strs_a[0] == (1 << nelec[0]) - 1 and strs_b[0] == (1 << nelec[1]) - 1
    return strs_a, strs_b


def _run_solver(store, ci_strings, h1, eri, norb, nelec):
    solver = store[store["_names"]["solver_block_name"]]
    solver_job.check_sbd_input(ci_strings=ci_strings, nelec=nelec, spin_sq=None)
    result = asyncio.run(
        solver_job._run_sbd_inner.fn(
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


def _assert_ran_on_every_node(store, work_dir):
    hosts = (work_dir / "hosts.txt").read_text().split()
    assert len(hosts) == NODES * MPIPROCS
    assert len(set(hosts)) == NODES
    (key, table), *_ = store["_artifacts"]
    assert key == "slurm-sbd-metrics"
    row = dict(zip(*table, strict=True))
    assert row["state"] == "COMPLETED" and row["exit_code"] == "0"


def test_closed_shell_full_space_matches_fci(blocks):
    h1, eri, norb, nelec = _water()
    strs, _ = _full_space(norb, nelec)
    e_fci, _ = fci.direct_spin1.kernel(h1, eri, norb, nelec, conv_tol=1e-12)

    result, work_dir = _run_solver(blocks, (strs, strs), h1, eri, norb, nelec)

    assert abs(result.energy - e_fci) < 1e-6
    assert result.carryover_bitstrings_b is None
    np.testing.assert_allclose(sum(result.orbital_occupancies[0]), nelec[0], atol=1e-6)
    np.testing.assert_allclose(sum(result.orbital_occupancies[1]), nelec[1], atol=1e-6)
    assert not (work_dir / "BetaDets.bin").exists()

    script = (work_dir / "sbd_solver.slurm").read_text()
    assert f"#SBATCH --partition={PARTITION}" in script
    assert f"#SBATCH --nodes={NODES}" in script
    assert f"#SBATCH --ntasks-per-node={MPIPROCS}" in script
    assert 'srun --mpi=pmix "${QCSC_PREFECT_EXECUTABLE}"' in script
    assert "--bdetfile" not in script
    _assert_ran_on_every_node(blocks, work_dir)


def test_rohf_open_shell_full_space_matches_fci(blocks):
    h1, eri, norb, nelec = _triplet_methylene()
    strs_a, strs_b = _full_space(norb, nelec)
    e_fci, _ = fci.direct_spin1.kernel(h1, eri, norb, nelec, conv_tol=1e-12)

    result, work_dir = _run_solver(blocks, (strs_a, strs_b), h1, eri, norb, nelec)

    assert abs(result.energy - e_fci) < 1e-6
    np.testing.assert_allclose(sum(result.orbital_occupancies[0]), nelec[0], atol=1e-6)
    np.testing.assert_allclose(sum(result.orbital_occupancies[1]), nelec[1], atol=1e-6)
    assert (work_dir / "BetaDets.bin").is_file()
    assert "MS2=2" in (work_dir / "fcidump.txt").read_text().splitlines()[0]
    # Carryover rows stay in their own electron-number sector.
    assert result.carryover_bitstrings.shape[0] > 0
    assert result.carryover_bitstrings_b.shape[0] > 0
    assert np.all(result.carryover_bitstrings.sum(axis=1) == nelec[0])
    assert np.all(result.carryover_bitstrings_b.sum(axis=1) == nelec[1])

    script = (work_dir / "sbd_solver.slurm").read_text()
    assert "--adetfile AlphaDets.bin --bdetfile BetaDets.bin" in script
    _assert_ran_on_every_node(blocks, work_dir)
