"""SQD recovery loop with every subspace solved by fulqrum through real Slurm jobs.

No Prefect server is used. The blocks resolved from the example TOML are built
by ``create_fulqrum_blocks`` in memory, and the solver task body is called
directly. Run it from the repository root on a Slurm submit host whose job root
is shared with the compute nodes:

    SLURM_INTEGRATION=1 python -m pytest algorithms/sqd/tests/test_sqd_fulqrum_slurm_integration.py

Environment:
    SLURM_INTEGRATION=1        enable the test
    SQD_FULQRUM_SLURM_ROOT     shared job root (default /data/sqd-fulqrum-integration)
    FULQRUM_SLURM_EXECUTABLE   default from the example TOML
    FULQRUM_SLURM_PARTITION    default from the example TOML
    FULQRUM_SLURM_TIMEOUT      seconds, default 600
"""

from __future__ import annotations

import asyncio
import os
import shutil
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytestmark = pytest.mark.slurm_integration

if os.getenv("SLURM_INTEGRATION", "").strip().lower() not in {"1", "true", "yes", "on"}:
    pytest.skip("Set SLURM_INTEGRATION=1 to run the Slurm round trip.", allow_module_level=True)
if shutil.which("sbatch") is None:
    pytest.skip("sbatch not found. Run on a Slurm login node.", allow_module_level=True)
pytest.importorskip("pyscf")

from pyscf import ao2mo, gto, scf  # noqa: E402
from qcsc_prefect_executor import from_blocks  # noqa: E402
from qcsc_prefect_executor.slurm import run as slurm_run  # noqa: E402
from qcsc_prefect_fulqrum import block_utils, solver_job  # noqa: E402
from qiskit_addon_sqd.counts import bit_array_to_arrays, generate_bit_array_uniform  # noqa: E402
from qiskit_addon_sqd.fermion import SCIResult, solve_fermion  # noqa: E402
from sqd_fulqrum import create_blocks, recovery  # noqa: E402

EXAMPLE = Path(__file__).resolve().parents[1] / "sqd_fulqrum_blocks.slurm.example.toml"
ROOT = Path(os.getenv("SQD_FULQRUM_SLURM_ROOT", "/data/sqd-fulqrum-integration"))
TIMEOUT = float(os.getenv("FULQRUM_SLURM_TIMEOUT", "600"))


class _LoggerStub:
    def info(self, msg, *args):
        pass

    warning = error = debug = info


@pytest.fixture
def store(monkeypatch):
    blocks: dict[str, Any] = {}

    def fake_save(self, name, overwrite=False):
        blocks[name] = self
        return self

    async def fake_artifact(*, table, key):
        blocks.setdefault("_artifacts", []).append((key, table))

    def loader(name):
        async def _load(block_name):
            return blocks[block_name]

        return type(name, (), {"load": staticmethod(_load)})

    for cls in (
        block_utils.CommandBlock,
        block_utils.ExecutionProfileBlock,
        block_utils.HPCProfileBlock,
        block_utils.FulqrumSolverJob,
    ):
        monkeypatch.setattr(cls, "save", fake_save, raising=False)
    monkeypatch.setattr(block_utils, "register_fulqrum_block_types", lambda: None)
    for name in ("CommandBlock", "ExecutionProfileBlock", "HPCProfileBlock"):
        monkeypatch.setattr(from_blocks, name, loader(name))
    monkeypatch.setattr(slurm_run, "get_run_logger", lambda: _LoggerStub())
    monkeypatch.setattr(slurm_run, "create_table_artifact", fake_artifact)
    monkeypatch.setattr(solver_job, "_logger", lambda: _LoggerStub())

    with EXAMPLE.open("rb") as f:
        config = tomllib.load(f)
    config.update(
        work_dir=str(ROOT),
        walltime="00:10:00",
        timeout_seconds=TIMEOUT,
        tol=1e-12,
        dense_max_dim=0,
    )
    if os.getenv("FULQRUM_SLURM_EXECUTABLE"):
        config["fulqrum_executable"] = os.environ["FULQRUM_SLURM_EXECUTABLE"]
    if os.getenv("FULQRUM_SLURM_PARTITION"):
        config["queue"] = os.environ["FULQRUM_SLURM_PARTITION"]
    kwargs, _, _ = create_blocks.resolve_settings(create_blocks._parse_args([]), config)

    shutil.rmtree(ROOT, ignore_errors=True)
    names = block_utils.create_fulqrum_blocks(**kwargs)
    blocks["_solver"] = blocks[names["solver_block_name"]]
    return blocks


def test_recovery_loop_over_slurm_matches_solve_fermion(store):
    mol = gto.M(atom="Li 0 0 0; H 0 0 1.6", basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run()
    norb = mf.mo_coeff.shape[1]
    h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    eri = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), norb)
    nelec = (2, 2)
    raw = bit_array_to_arrays(generate_bit_array_uniform(2000, 2 * norb, rand_seed=11))
    occ = np.array([0.95, 0.95, 0.05, 0.05, 0.05, 0.05])
    occ0 = (occ, occ)

    async def slurm_solve(ci_strings):
        return await solver_job._run_fulqrum_inner.fn(
            solver=store["_solver"],
            ci_strings=ci_strings,
            one_body_tensor=h1,
            two_body_tensor=eri,
            norb=norb,
            nelec=nelec,
        )

    async def reference_solve(ci_strings):
        energy, state, occ, _ = solve_fermion(ci_strings, h1, eri, open_shell=True)
        return SCIResult(energy=energy, sci_state=state, orbital_occupancies=occ)

    def run(solve):
        return asyncio.run(
            recovery.run_configuration_recovery(
                *raw,
                solve=solve,
                initial_occupancies=occ0,
                nelec=nelec,
                open_shell=False,
                subspace_dim=64,
                num_batches=2,
                max_iterations=2,
                rng=np.random.default_rng(24),
                nuclear_repulsion_energy=mol.energy_nuc(),
            )
        )

    got = run(slurm_solve)
    ref = run(reference_solve)

    np.testing.assert_allclose(
        [r["energy"] for r in got.records], [r["energy"] for r in ref.records], atol=1e-8
    )
    assert got.best_energy == pytest.approx(ref.best_energy, abs=1e-8)

    job_dirs = sorted(ROOT.glob("job_*"))
    assert len(job_dirs) == 4
    for job_dir in job_dirs:
        assert (job_dir / "result.json").exists()
        assert (job_dir / "fulqrum_solver.slurm").exists()
    states = [dict(zip(*table, strict=True))["state"] for _, table in store["_artifacts"]]
    assert states == ["COMPLETED"] * 4
