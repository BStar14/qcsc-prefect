"""Real-GPU checks of the ``gpu`` and ``gpu-mpi`` backends against the CPU backend.

They need a GPU build of fulqrum (``FULQRUM_WITH_GPU=1 native/build_fulqrum.sh``),
PySCF and a CUDA device, and are skipped unless ``FULQRUM_GPU=1``. Run the file
under an MPI launcher, one rank per GPU, from a directory every rank can write
(the CLI test puts its job directory there):

    FULQRUM_GPU=1 mpirun -n 2 python -m pytest -p no:cacheprovider \
        packages/qcsc-prefect-fulqrum/tests/test_fulqrum_gpu_integration.py

``srun --mpi=pmix -n 2 --gpus-per-task=1`` works the same way, and a plain
``python -m pytest`` runs it on one rank. Every rank runs every test in the same
order, because the ``gpu-mpi`` solves are collective. The ``gpu`` tests run on
rank 0 only and use the GPUs that rank sees.

Checks: the energy of both backends matches the CPU backend to 1e-8, the
occupancies match the diagonals of fulqrum's own one-body RDM
(``compute_rdms(rank=1)``), and the carryover strings match those of the CPU
backend.
"""

from __future__ import annotations

import os
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.fulqrum_gpu_integration

if os.getenv("FULQRUM_GPU", "").strip().lower() not in {"1", "true", "yes", "on"}:
    pytest.skip("Set FULQRUM_GPU=1 to run the real-GPU tests.", allow_module_level=True)
pytest.importorskip("fulqrum.gpu.distributed.kernel_launch")
pytest.importorskip("cupy")
pytest.importorskip("pyscf")

import cupy  # noqa: E402
from pyscf import ao2mo, gto, mcscf, scf  # noqa: E402
from pyscf.fci import cistring  # noqa: E402
from qcsc_prefect_fulqrum import cli, io_utils, solver  # noqa: E402
from qcsc_prefect_fulqrum.solver import FulqrumSolverOptions, solve_fulqrum  # noqa: E402

if cupy.cuda.runtime.getDeviceCount() < 1:
    pytest.skip("No CUDA device is visible.", allow_module_level=True)

try:
    from mpi4py import MPI

    COMM = MPI.COMM_WORLD
except ImportError:  # the gpu tests still run; the gpu-mpi tests skip
    MPI = COMM = None
RANK = COMM.Get_rank() if COMM is not None else 0

ENERGY_ATOL = 1e-8
OCC_ATOL = 1e-6
TOL = 1e-10
MOLECULES = {
    # name: (atom, basis, spin, frozen core orbitals); at least two electrons
    # per spin, which the GPU kernels need.
    "lih": ("Li 0 0 0; H 0 0 1.6", "sto-3g", 0, 0),
    "n2": ("N 0 0 0; N 0 0 1.1", "sto-3g", 0, 2),
    # Open shell: unequal alpha and beta string counts.
    "o2": ("O 0 0 0; O 0 0 1.21", "sto-3g", 2, 2),
}

needs_mpi = pytest.mark.skipif(COMM is None, reason="gpu-mpi needs mpi4py")
rank0_only = pytest.mark.skipif(RANK != 0, reason="single-process backend; runs on rank 0")


@lru_cache(maxsize=None)
def _problem(name: str):
    """Integrals and a random subspace with unequal, unsorted alpha and beta strings."""

    atom, basis, spin, ncore = MOLECULES[name]
    mol = gto.M(atom=atom, basis=basis, spin=spin, verbose=0)
    mf = (scf.ROHF if spin else scf.RHF)(mol).run()
    norb = mol.nao_nr() - ncore
    nel = mol.nelectron - 2 * ncore
    nelec = ((nel + spin) // 2, (nel - spin) // 2)
    cas = mcscf.CASCI(mf, norb, nelec)
    h1, _ = cas.get_h1cas()
    eri = ao2mo.restore(1, cas.get_h2cas(), norb)
    rng = np.random.default_rng(11)
    strs = []
    for n, share in zip(nelec, (0.8, 0.6)):
        full = np.asarray(cistring.make_strings(range(norb), n), dtype=np.int64)
        strs.append(rng.choice(full, size=max(3, int(share * len(full))), replace=False))
    return (strs[0], strs[1]), np.ascontiguousarray(h1), np.ascontiguousarray(eri), norb, nelec


def _solve(name: str, **options):
    ci_strings, h1, eri, norb, nelec = _problem(name)
    opts = FulqrumSolverOptions(tol=TOL, dense_max_dim=0, carryover_fraction=0.5, **options)
    return solve_fulqrum(ci_strings, h1, eri, norb, nelec, opts)


@lru_cache(maxsize=None)
def _cpu(name: str):
    return _solve(name, backend="cpu")


def _cpu_weights(name: str):
    probs = np.abs(_cpu(name).amplitudes) ** 2
    return probs.sum(axis=1), probs.sum(axis=0)


def _assert_same_carryover(name: str, sol) -> None:
    # Strings of equal weight may come in either order, so compare the CPU
    # weights of the selected strings, heaviest first.
    ref = _cpu(name)
    for spin, weights in enumerate(_cpu_weights(name)):
        strs = (ref.ci_strs_a, ref.ci_strs_b)[spin]
        index = {int(s): i for i, s in enumerate(strs)}
        ref_carry = (ref.carryover_a, ref.carryover_b)[spin]
        carry = (sol.carryover_a, sol.carryover_b)[spin]
        assert len(carry) == len(ref_carry)
        np.testing.assert_allclose(
            [weights[index[int(s)]] for s in carry],
            [weights[index[int(s)]] for s in ref_carry],
            atol=1e-8,
        )


def _hsub(name: str):
    ci_strings, h1, eri, norb, nelec = _problem(name)
    strs_a, strs_b, h1, eri = solver.normalize_problem(ci_strings, h1, eri, norb, nelec)
    _, _, _, hsub = solver._build_hamiltonian(h1, eri, strs_a, strs_b, norb, {})
    return hsub


@rank0_only
@pytest.mark.parametrize("eigensolver", ["scipy", "primme"])
@pytest.mark.parametrize("name", list(MOLECULES))
def test_gpu_matches_cpu(name, eigensolver):
    if eigensolver == "primme":
        pytest.importorskip("primme")
    import fulqrum.gpu as fq_gpu

    cpu = _cpu(name)
    sol = _solve(name, backend="gpu", eigensolver=eigensolver)

    assert sol.metadata["solver_path"] == f"gpu-{eigensolver}"
    assert sol.energy == pytest.approx(cpu.energy, abs=ENERGY_ATOL)
    _assert_same_carryover(name, sol)

    # Occupancies against fulqrum's GPU one-body RDM of the solver's state.
    amplitudes = sol.amplitudes
    vec = amplitudes.T.reshape(-1)  # back to fulqrum's basis order ib * na + ia
    op = fq_gpu.GPUHamiltonian(_hsub(name))
    gamma_a, gamma_b = op.compute_rdms(vec / np.linalg.norm(vec), rank=1)
    np.testing.assert_allclose(sol.orbital_occupancies[0], np.diag(gamma_a), atol=OCC_ATOL)
    np.testing.assert_allclose(sol.orbital_occupancies[1], np.diag(gamma_b), atol=OCC_ATOL)


@needs_mpi
@pytest.mark.parametrize("name", list(MOLECULES))
def test_gpu_mpi_matches_cpu(name):
    import fulqrum.gpu as fq_gpu

    cpu = _cpu(name)
    sol = _solve(name, backend="gpu-mpi")

    assert sol.metadata["solver_path"] == "gpu-mpi-primme"
    assert sol.amplitudes is None
    assert sol.energy == pytest.approx(cpu.energy, abs=ENERGY_ATOL)
    np.testing.assert_allclose(
        sol.orbital_occupancies[0], cpu.orbital_occupancies[0], atol=OCC_ATOL
    )
    np.testing.assert_allclose(
        sol.orbital_occupancies[1], cpu.orbital_occupancies[1], atol=OCC_ATOL
    )
    _assert_same_carryover(name, sol)
    # The result is the same on every rank.
    energies = COMM.allgather(sol.energy)
    assert max(energies) - min(energies) == 0.0
    carry = COMM.allgather((sol.carryover_a.tolist(), sol.carryover_b.tolist()))
    assert all(c == carry[0] for c in carry)

    # Occupancies against the distributed one-body RDM of an independent solve.
    hsub = _hsub(name)
    device = solver._local_rank() % cupy.cuda.runtime.getDeviceCount()
    H = fq_gpu.DistributedMPIGPUHamiltonian(hsub, comm=COMM, gpu_ids=(device,))
    diag = np.asarray(H.diag_local, dtype=np.float64)
    _, evecs = fq_gpu.distributed_primme_eigsh(
        H,
        k=1,
        which="SA",
        tol=TOL,
        v0_local=solver._distributed_v0(diag, COMM, MPI),
        precond_diag_local=diag,
    )
    local = np.asarray(evecs)[:, 0]
    norm = np.sqrt(COMM.allreduce(float(local @ local), op=MPI.SUM))
    gamma_a, gamma_b = H.compute_rdms(local / norm, rank=1)
    np.testing.assert_allclose(sol.orbital_occupancies[0], np.diag(gamma_a), atol=OCC_ATOL)
    np.testing.assert_allclose(sol.orbital_occupancies[1], np.diag(gamma_b), atol=OCC_ATOL)


@needs_mpi
def test_cli_gpu_mpi_round_trip():
    name = "n2"
    ci_strings, h1, eri, norb, nelec = _problem(name)
    work_dir = None
    if RANK == 0:
        work_dir = Path(tempfile.mkdtemp(prefix="fulqrum-gpu-cli-", dir=Path.cwd()))
        io_utils.write_solver_input(
            work_dir=work_dir,
            ci_strings=ci_strings,
            one_body_tensor=h1,
            two_body_tensor=eri,
            norb=norb,
            nelec=nelec,
            options=FulqrumSolverOptions(
                tol=TOL, dense_max_dim=0, backend="gpu-mpi", carryover_fraction=0.5
            ),
        )
    work_dir = COMM.bcast(work_dir, root=0)

    assert cli.main([str(work_dir)]) == 0
    COMM.Barrier()
    if RANK == 0:
        sol = io_utils.read_solver_output(work_dir, norb=norb)
        assert sol.energy == pytest.approx(_cpu(name).energy, abs=ENERGY_ATOL)
        assert sol.metadata["solver_path"] == "gpu-mpi-primme"
        _assert_same_carryover(name, sol)
        assert io_utils.solver_error_files(work_dir)[1:] == []
