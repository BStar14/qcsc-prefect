"""Run the compiled SBD driver with real MPI and check the Python file contract.

Set SBD_NATIVE_BINARY to a built diag executable to enable these tests. CI builds
the pinned upstream revision first; normal Python-only installations can skip them.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from qcsc_prefect_sbd import solver_job

NATIVE_BINARY = os.environ.get("SBD_NATIVE_BINARY")
pytestmark = pytest.mark.skipif(
    not NATIVE_BINARY, reason="Set SBD_NATIVE_BINARY to test real SBD/MPI"
)


@pytest.mark.parametrize("ranks", [1, 2])
@pytest.mark.parametrize("norb", [2, 10, 30])
def test_native_solver_energy_occupancies_and_carryover(tmp_path, monkeypatch, ranks, norb):
    # Two active orbitals in a closed-shell two-electron problem. The complete
    # selected subspace is 2 x 2, so its energy is twice the lowest one-body eigenvalue.
    # norb=10 checks non-byte-aligned input/output; norb=30 spans multiple SBD words.
    h1 = np.eye(norb)
    h1[0, 0], h1[-1, -1] = -1.0, -0.5
    h1[0, -1] = h1[-1, 0] = 0.1
    h2 = np.zeros((norb,) * 4)
    determinants = np.array([1, 1 << (norb - 1)], dtype=np.int64)
    monkeypatch.setattr(
        solver_job, "get_run_logger", lambda: SimpleNamespace(debug=lambda *args: None)
    )
    solver_job._prep_files(
        work_dir=tmp_path,
        ci_strings=(determinants, determinants),
        one_body_tensor=h1,
        two_body_tensor=h2,
        norb=norb,
        nelec=(1, 1),
    )
    launcher = shlex.split(os.environ.get("SBD_NATIVE_LAUNCHER", "mpiexec"))
    solver = solver_job.SBDSolverJob(
        root_dir=str(tmp_path), block=4, iteration=4, tolerance=1e-10, carryover_ratio=0.5
    )
    command = launcher + ["-n", str(ranks), str(Path(NATIVE_BINARY).resolve())]
    command += solver_job._build_solver_args(solver)
    environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    run = subprocess.run(
        command, cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=90
    )
    assert run.returncode == 0, run.stdout + run.stderr
    result = solver_job._read_files(work_dir=tmp_path, norb=norb)
    values, vectors = np.linalg.eigh(h1[np.ix_([0, norb - 1], [0, norb - 1])])
    assert result.energy == pytest.approx(2 * values[0], abs=1e-7)
    expected_occupancy = np.zeros(norb)
    expected_occupancy[[0, norb - 1]] = vectors[:, 0] ** 2
    for occupancy in result.orbital_occupancies:
        np.testing.assert_allclose(occupancy, expected_occupancy, atol=1e-7)
    carryover = result.carryover_bitstrings
    assert carryover.shape[0] > 0
    assert carryover.shape[1] == norb
    assert np.all(carryover.sum(axis=1) == 1)
    decoded = [int("".join("1" if bit else "0" for bit in row), 2) for row in carryover]
    assert set(decoded) <= set(determinants)
    # Python consumes MSB-first rows with trailing byte padding in carryover.bin.
    assert (tmp_path / "carryover.bin").read_bytes() == np.packbits(carryover, axis=1).tobytes()
