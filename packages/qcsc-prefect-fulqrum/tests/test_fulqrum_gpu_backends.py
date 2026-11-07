"""The GPU backends of the solver, against fake fulqrum, CuPy and mpi4py modules.

No GPU, CUDA, MPI or fulqrum is needed. The fakes model the interfaces of
fulqrum 1.0 (``GPUHamiltonian``, ``DistributedMPIGPUHamiltonian``,
``distributed_primme_eigsh``), with the tile layout of
``fulqrum.gpu.distributed.dist_matvec_tile_resident``. Ranks of a distributed
solve run as threads that share one fake communicator.
"""

from __future__ import annotations

import logging
import sys
import threading
import types
from itertools import combinations

import numpy as np
import pytest
import scipy.sparse as sp
from qcsc_prefect_fulqrum import solver
from qcsc_prefect_fulqrum.carryover import marginal_weights, occupancies_from_marginals
from qcsc_prefect_fulqrum.solver import FulqrumSolverOptions, solve_fulqrum
from scipy.sparse.linalg import LinearOperator, aslinearoperator

# Six orbitals give 15 alpha and 20 beta strings at nelec=(2, 3): unequal sizes
# catch a swapped alpha/beta axis in the tile fold.
NORB = 6


def _strs(norb: int, nelec: int) -> np.ndarray:
    return np.array(
        sorted(sum(1 << i for i in occ) for occ in combinations(range(norb), nelec)),
        dtype=np.int64,
    )


class FakeSubspace:
    def __init__(self, strs_a, strs_b, norb):
        self.strs_a, self.strs_b, self.norb = strs_a, strs_b, norb
        self.alpha_strs = [format(int(x), f"0{norb}b") for x in strs_a]
        self.beta_strs = [format(int(x), f"0{norb}b") for x in strs_b]

    def get_orbital_occupancies(self, probs, norb):
        w_a, w_b = marginal_weights(probs, len(self.strs_a), len(self.strs_b))
        return occupancies_from_marginals(w_a, w_b, self.strs_a, self.strs_b, norb)


class FakeHsub(LinearOperator):
    """A random real symmetric matrix in fulqrum's basis order (``ib * na + ia``)."""

    def __init__(self, strs_a, strs_b, norb=NORB, seed=3):
        na, nb = len(strs_a), len(strs_b)
        rng = np.random.default_rng(seed)
        off = rng.normal(scale=0.3, size=(na * nb, na * nb))
        self.matrix = (off + off.T) / 2 + np.diag(rng.uniform(-3.0, 1.0, na * nb))
        super().__init__(shape=self.matrix.shape, dtype=np.float64)
        self.spmv = types.SimpleNamespace(subspace=FakeSubspace(strs_a, strs_b, norb))

    def _matvec(self, x):
        return self.matrix @ np.asarray(x).reshape(-1)

    def diagonal_vector(self):
        return np.diag(self.matrix).astype(complex)

    def to_csr_linearoperator_fast(self):
        op = aslinearoperator(sp.csr_matrix(self.matrix))
        op.matrix = sp.csr_matrix(self.matrix)
        return op

    def ground_state(self):
        evals, evecs = np.linalg.eigh(self.matrix)
        return float(evals[0]), evecs[:, 0]


class FakeGPUHamiltonian(LinearOperator):
    instances: list = []

    def __init__(self, hsub, gpu_ids=None, threads_per_block=256, verbose=False):
        super().__init__(shape=hsub.shape, dtype=np.float64)
        self.hsub, self.gpu_ids = hsub, tuple(gpu_ids)
        FakeGPUHamiltonian.instances.append(self)

    def _matvec(self, x):
        return self.hsub @ x


class FakeComm:
    """``MPI.COMM_WORLD`` for ranks that run as threads; the rank is thread-local."""

    def __init__(self, size):
        self.size = size
        self.local = threading.local()
        self.barrier = threading.Barrier(size)
        self.slots = [None] * size
        self.aborted = []

    def Get_rank(self):
        return getattr(self.local, "rank", 0)

    def Get_size(self):
        return self.size

    def _gather(self, value):
        self.slots[self.Get_rank()] = value
        self.barrier.wait()
        values = list(self.slots)
        self.barrier.wait()
        return values

    def allreduce(self, value, op):
        assert op == "MINLOC"
        return min(self._gather(value))

    def Allreduce(self, sendbuf, recvbuf, op):
        assert sendbuf == "IN_PLACE" and op == "SUM"
        recvbuf[...] = np.sum(self._gather(recvbuf.copy()), axis=0)

    def bcast(self, value, root=0):
        return self._gather(value)[root]

    def Abort(self, code):
        self.aborted.append(code)


def _fake_mpi4py(comm):
    mpi = types.SimpleNamespace(COMM_WORLD=comm, MINLOC="MINLOC", SUM="SUM", IN_PLACE="IN_PLACE")
    return types.SimpleNamespace(MPI=mpi), mpi


class FakeTileHamiltonian:
    """The public surface of ``TileGPUResidentHamiltonian`` that the solver uses."""

    def __init__(self, hsub, comm, gpu_ids, process_grid):
        self.hsub, self.comm, self.gpu_ids = hsub, comm, tuple(gpu_ids)
        na, nb = len(hsub.spmv.subspace.alpha_strs), len(hsub.spmv.subspace.beta_strs)
        p_alpha, p_beta = process_grid
        rank = comm.Get_rank()
        self.process_grid = (p_alpha, p_beta)
        # The construction of dist_matvec_tile_resident.py:450-494 and grid.py:39-51.
        self.coord_alpha, self.coord_beta = rank // p_beta, rank % p_beta
        self._a0, self._a1 = solver._split_even(na, p_alpha)[self.coord_alpha]
        self._b0, self._b1 = solver._split_even(nb, p_beta)[self.coord_beta]
        self.n_a_local, self.n_b_local = self._a1 - self._a0, self._b1 - self._b0
        self.n_local = self.n_a_local * self.n_b_local
        self.na, self.nb = na, nb

    def local_tile(self, full):
        # Beta-outer, alpha-inner: local index b * n_a_local + a.
        grid = np.asarray(full).reshape(self.nb, self.na)
        return np.ascontiguousarray(grid[self._b0 : self._b1, self._a0 : self._a1]).ravel()

    @property
    def diag_local(self):
        return self.local_tile(np.diag(self.hsub.matrix))


class FakeGPU:
    """``fulqrum.gpu`` with a configurable process grid and recorded calls."""

    def __init__(self, process_grid=(1, 1), primme_mpi=True, scale=-2.5):
        self.process_grid, self.primme_mpi, self.scale = process_grid, primme_mpi, scale
        self.calls = []
        self.lock = threading.Lock()
        self.GPUHamiltonian = FakeGPUHamiltonian

    def is_primme_mpi_available(self):
        return self.primme_mpi

    def DistributedMPIGPUHamiltonian(self, hsub, layout="tile_resident", **kwargs):
        return FakeTileHamiltonian(hsub, kwargs["comm"], kwargs["gpu_ids"], self.process_grid)

    def distributed_primme_eigsh(self, H, k, **kwargs):
        energy, vec = H.hsub.ground_state()
        with self.lock:
            self.calls.append({"rank": H.comm.Get_rank(), "H": H, "k": k, **kwargs})
        # Unnormalized and with an arbitrary sign: the solver must not rely on either.
        return np.array([energy]), (self.scale * H.local_tile(vec)).reshape(-1, 1)


@pytest.fixture
def fake_gpu(monkeypatch):
    """Install fake fulqrum (1.0.0), fulqrum.gpu and CuPy modules with 2 devices."""

    gpu = FakeGPU()
    fulqrum = types.ModuleType("fulqrum")
    fulqrum.__version__ = "1.0.0"
    fulqrum.gpu = gpu
    runtime = types.SimpleNamespace(getDeviceCount=lambda: gpu.device_count)
    gpu.device_count = 2
    cupy = types.ModuleType("cupy")
    cupy.cuda = types.SimpleNamespace(runtime=runtime)
    # SciPy's array-API layer checks isinstance(x, cupy.ndarray) once cupy is imported.
    cupy.ndarray = type("ndarray", (), {})
    monkeypatch.setitem(sys.modules, "fulqrum", fulqrum)
    monkeypatch.setitem(sys.modules, "fulqrum.gpu", gpu)
    monkeypatch.setitem(sys.modules, "fulqrum.gpu.distributed", types.ModuleType("d"))
    monkeypatch.setitem(sys.modules, "fulqrum.gpu.distributed.kernel_launch", types.ModuleType("k"))
    monkeypatch.setitem(sys.modules, "cupy", cupy)
    for name in solver._LOCAL_RANK_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    FakeGPUHamiltonian.instances.clear()
    gpu.fulqrum = fulqrum
    return gpu


@pytest.fixture
def fake_mpi(monkeypatch):
    def install(size):
        comm = FakeComm(size)
        module, mpi = _fake_mpi4py(comm)
        monkeypatch.setitem(sys.modules, "mpi4py", module)
        monkeypatch.setitem(sys.modules, "mpi4py.MPI", mpi)
        return comm

    return install


def _run_ranks(comm, fn):
    """Run ``fn()`` on every rank of ``comm`` in threads; return the results by rank."""

    results, errors = [None] * comm.size, []

    def target(rank):
        comm.local.rank = rank
        try:
            results[rank] = fn()
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
            comm.barrier.abort()

    threads = [threading.Thread(target=target, args=(r,)) for r in range(comm.size)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    if errors:
        raise errors[0]
    return results


def _problem(nelec=(2, 3), norb=NORB, seed=3):
    strs_a, strs_b = _strs(norb, nelec[0]), _strs(norb, nelec[1])
    return strs_a, strs_b, FakeHsub(strs_a, strs_b, norb, seed)


def _expected_weights(hsub, na, nb):
    _, vec = hsub.ground_state()
    probs = np.abs(vec) ** 2
    return marginal_weights(probs / probs.sum(), na, nb)


# --- gpu --------------------------------------------------------------------


@pytest.mark.parametrize("eigensolver", ["scipy", "primme"])
def test_gpu_uses_every_visible_device_and_the_cpu_start_vector(fake_gpu, monkeypatch, eigensolver):
    strs_a, strs_b, hsub = _problem()
    seen = {}

    def fake_eigsh(op, dim, v0, options):
        seen.update(op=op, v0=v0, options=options)
        return hsub.ground_state()

    monkeypatch.setattr(solver, "_eigsh_lowest", fake_eigsh)
    options = FulqrumSolverOptions(eigensolver=eigensolver, dense_max_dim=0)
    timings = {}
    energy, vec, path, nnz = solver._solve_gpu(hsub, hsub.shape[0], options, timings)

    assert path == f"gpu-{eigensolver}" and nnz is None
    assert energy == pytest.approx(hsub.ground_state()[0])
    assert isinstance(seen["op"], FakeGPUHamiltonian)
    assert seen["op"].gpu_ids == (0, 1)
    # The same start vector as the CPU path: weight on the lowest diagonal entries.
    np.testing.assert_array_equal(
        seen["v0"], solver._initial_vector(hsub, hsub.shape[0], np.float64)
    )
    assert {"projection", "eigensolve"} <= timings.keys()


def test_gpu_eigsh_runs_on_the_gpu_operator(fake_gpu):
    strs_a, strs_b, hsub = _problem()
    options = FulqrumSolverOptions(dense_max_dim=0, tol=1e-12)
    energy, vec, path, _ = solver._solve_gpu(hsub, hsub.shape[0], options, {})
    ref_energy, ref_vec = hsub.ground_state()
    assert path == "gpu-scipy"
    assert energy == pytest.approx(ref_energy, abs=1e-10)
    assert abs(np.dot(vec, ref_vec)) == pytest.approx(1.0, abs=1e-8)


def test_solve_fulqrum_gpu_end_to_end(fake_gpu, monkeypatch):
    strs_a, strs_b, hsub = _problem()
    monkeypatch.setattr(
        solver,
        "_build_hamiltonian",
        lambda h1, eri, a, b, norb, t: (fake_gpu.fulqrum, None, hsub.spmv.subspace, hsub),
    )
    options = FulqrumSolverOptions(
        backend="gpu", dense_max_dim=0, tol=1e-12, carryover_fraction=0.5
    )
    sol = solve_fulqrum(
        (strs_a, strs_b), np.zeros((NORB, NORB)), np.zeros((NORB,) * 4), NORB, (2, 3), options
    )
    w_a, w_b = _expected_weights(hsub, len(strs_a), len(strs_b))

    assert sol.metadata["backend"] == "gpu"
    assert sol.metadata["solver_path"] == "gpu-scipy"
    assert sol.energy == pytest.approx(hsub.ground_state()[0], abs=1e-10)
    occ = occupancies_from_marginals(w_a, w_b, strs_a, strs_b, NORB)
    np.testing.assert_allclose(sol.orbital_occupancies[0], occ[0], atol=1e-8)
    np.testing.assert_allclose(sol.orbital_occupancies[1], occ[1], atol=1e-8)
    assert sol.amplitudes.shape == (len(strs_a), len(strs_b))
    assert len(sol.carryover_a) == len(strs_a) // 2
    assert sol.carryover_a[0] == strs_a[np.argmax(w_a)]


def test_gpu_falls_back_to_cpu_below_two_electrons_per_spin(fake_gpu, caplog):
    strs_a, strs_b, hsub = _problem(nelec=(1, 2))
    options = FulqrumSolverOptions(dense_max_dim=0, tol=1e-12)
    with caplog.at_level(logging.WARNING, logger=solver.logger.name):
        energy, _, path, nnz = solver._solve_gpu(hsub, hsub.shape[0], options, {})
    assert path == "scipy-csr" and nnz is not None
    assert energy == pytest.approx(hsub.ground_state()[0], abs=1e-10)
    assert not FakeGPUHamiltonian.instances
    assert "solving on the CPU" in caplog.text


def test_small_subspaces_stay_on_the_cpu(fake_gpu, monkeypatch):
    strs_a, strs_b = _strs(3, 2), _strs(3, 2)
    hsub = FakeHsub(strs_a, strs_b, 3)
    monkeypatch.setattr(
        solver,
        "_build_hamiltonian",
        lambda *args: (fake_gpu.fulqrum, None, hsub.spmv.subspace, hsub),
    )
    for backend in ("gpu", "gpu-mpi"):
        sol = solve_fulqrum(
            (strs_a, strs_b),
            np.zeros((3, 3)),
            np.zeros((3,) * 4),
            3,
            (2, 2),
            FulqrumSolverOptions(backend=backend),
        )
        assert sol.metadata["backend"] == "cpu" and sol.metadata["solver_path"] == "dense"


# --- errors -----------------------------------------------------------------


def test_gpu_needs_fulqrum_gpu(fake_gpu, monkeypatch):
    monkeypatch.setitem(sys.modules, "fulqrum.gpu", None)
    _, _, hsub = _problem()
    with pytest.raises(ImportError, match="FQ_BUILD_GPU=1"):
        solver._solve_gpu(hsub, hsub.shape[0], FulqrumSolverOptions(), {})


def test_gpu_needs_the_compiled_kernels(fake_gpu, monkeypatch):
    # A CPU-only fulqrum 1.0 build imports fulqrum.gpu but has no kernel launcher.
    monkeypatch.setitem(sys.modules, "fulqrum.gpu.distributed.kernel_launch", None)
    _, _, hsub = _problem()
    with pytest.raises(ImportError, match="FQ_BUILD_GPU=1"):
        solver._solve_gpu(hsub, hsub.shape[0], FulqrumSolverOptions(), {})


def test_gpu_needs_cupy(fake_gpu, monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", None)
    _, _, hsub = _problem()
    with pytest.raises(ImportError, match="CuPy"):
        solver._solve_gpu(hsub, hsub.shape[0], FulqrumSolverOptions(), {})


@pytest.mark.parametrize("version", ["0.5.0", "0.9.9rc1"])
def test_gpu_backends_need_fulqrum_1(fake_gpu, fake_mpi, version):
    fake_gpu.fulqrum.__version__ = version
    fake_mpi(1)
    strs_a, strs_b, hsub = _problem()
    with pytest.raises(ImportError, match=r"fulqrum >= 1\.0"):
        solver._solve_gpu(hsub, hsub.shape[0], FulqrumSolverOptions(), {})
    with pytest.raises(ImportError, match=r"fulqrum >= 1\.0"):
        solver._solve_gpu_mpi(hsub, None, strs_a, strs_b, FulqrumSolverOptions(), {})


def test_gpu_backends_reject_complex_integrals(fake_gpu, fake_mpi):
    fake_mpi(1)
    strs_a, strs_b, hsub = _problem()
    hsub.dtype = np.dtype(complex)
    with pytest.raises(NotImplementedError, match="real integrals"):
        solver._solve_gpu(hsub, hsub.shape[0], FulqrumSolverOptions(), {})
    with pytest.raises(NotImplementedError, match="real integrals"):
        solver._solve_gpu_mpi(hsub, None, strs_a, strs_b, FulqrumSolverOptions(), {})


def test_gpu_needs_a_visible_device(fake_gpu):
    fake_gpu.device_count = 0
    _, _, hsub = _problem()
    with pytest.raises(RuntimeError, match="no CUDA device"):
        solver._solve_gpu(hsub, hsub.shape[0], FulqrumSolverOptions(), {})


def test_gpu_mpi_needs_mpi4py(fake_gpu, monkeypatch):
    monkeypatch.setitem(sys.modules, "mpi4py", None)
    strs_a, strs_b, hsub = _problem()
    with pytest.raises(ImportError, match="mpi4py"):
        solver._solve_gpu_mpi(hsub, None, strs_a, strs_b, FulqrumSolverOptions(), {})


def test_gpu_mpi_needs_distributed_primme(fake_gpu, fake_mpi):
    fake_mpi(1)
    fake_gpu.primme_mpi = False
    strs_a, strs_b, hsub = _problem()
    with pytest.raises(ImportError, match="distributed PRIMME"):
        solver._solve_gpu_mpi(hsub, None, strs_a, strs_b, FulqrumSolverOptions(), {})


def test_gpu_mpi_rejects_an_unexpected_tile_layout(fake_gpu, fake_mpi):
    fake_mpi(1)
    strs_a, strs_b, hsub = _problem()
    original = fake_gpu.DistributedMPIGPUHamiltonian

    def shifted(hsub, **kwargs):
        H = original(hsub, **kwargs)
        H._a0 += 1
        return H

    fake_gpu.DistributedMPIGPUHamiltonian = shifted
    with pytest.raises(RuntimeError, match="unexpected fulqrum tile layout"):
        solver._solve_gpu_mpi(hsub, None, strs_a, strs_b, FulqrumSolverOptions(), {})


# --- gpu-mpi ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("process_grid", "nelec"),
    [((2, 1), (2, 3)), ((1, 2), (2, 3)), ((2, 2), (2, 3)), ((3, 1), (3, 2)), ((1, 1), (2, 3))],
    ids=["2x1", "1x2", "2x2", "3x1-uneven", "1x1"],
)
def test_gpu_mpi_folds_tiles_into_global_marginal_weights(fake_gpu, fake_mpi, process_grid, nelec):
    strs_a, strs_b, hsub = _problem(nelec=nelec)
    na, nb = len(strs_a), len(strs_b)
    fake_gpu.process_grid = process_grid
    comm = fake_mpi(process_grid[0] * process_grid[1])
    options = FulqrumSolverOptions(backend="gpu-mpi", tol=1e-9, max_iter=500)

    results = _run_ranks(
        comm, lambda: solver._solve_gpu_mpi(hsub, None, strs_a, strs_b, options, {})
    )

    w_a_ref, w_b_ref = _expected_weights(hsub, na, nb)
    for energy, (w_a, w_b), path in results:
        assert path == "gpu-mpi-primme"
        assert energy == results[0][0] == pytest.approx(hsub.ground_state()[0])
        np.testing.assert_allclose(w_a, w_a_ref, atol=1e-12)
        np.testing.assert_allclose(w_b, w_b_ref, atol=1e-12)
        assert w_a.sum() == pytest.approx(1.0) and w_b.sum() == pytest.approx(1.0)

    # Every rank asks for the same solve, preconditioned with its local diagonal.
    calls = sorted(fake_gpu.calls, key=lambda c: c["rank"])
    assert [c["rank"] for c in calls] == list(range(comm.size))
    for call in calls:
        assert (call["k"], call["which"], call["gpu"]) == (1, "SA", "auto")
        assert (call["tol"], call["maxiter"]) == (1e-9, 500)
        np.testing.assert_array_equal(call["precond_diag_local"], call["H"].diag_local)
    # One unit start vector, at the global minimum of the diagonal.
    full_v0 = np.zeros(na * nb)
    for call in calls:
        H = call["H"]
        assert H.gpu_ids == (0,) and call["v0_local"].shape == (H.n_local,)
        tile = call["v0_local"].reshape(H.n_b_local, H.n_a_local)
        full_v0.reshape(nb, na)[H._b0 : H._b1, H._a0 : H._a1] += tile
    assert full_v0.sum() == 1.0
    assert int(np.argmax(full_v0)) == int(np.argmin(np.diag(hsub.matrix)))


def test_gpu_mpi_tol_zero_uses_a_floor(fake_gpu, fake_mpi):
    fake_mpi(1)
    strs_a, strs_b, hsub = _problem()
    solver._solve_gpu_mpi(hsub, None, strs_a, strs_b, FulqrumSolverOptions(tol=0.0), {})
    assert fake_gpu.calls[0]["tol"] == solver._GPU_MPI_MIN_TOL


def test_gpu_mpi_device_is_local_rank_modulo_device_count(fake_gpu, fake_mpi, monkeypatch):
    fake_mpi(1)
    fake_gpu.device_count = 4
    monkeypatch.setenv("OMPI_COMM_WORLD_LOCAL_RANK", "6")
    strs_a, strs_b, hsub = _problem()
    solver._solve_gpu_mpi(hsub, None, strs_a, strs_b, FulqrumSolverOptions(), {})
    assert fake_gpu.calls[0]["H"].gpu_ids == (2,)


def test_solve_fulqrum_gpu_mpi_end_to_end(fake_gpu, fake_mpi, monkeypatch):
    strs_a, strs_b, hsub = _problem()
    fake_gpu.process_grid = (1, 2)
    comm = fake_mpi(2)
    monkeypatch.setattr(
        solver, "_build_hamiltonian", lambda *args: (fake_gpu.fulqrum, None, None, hsub)
    )
    options = FulqrumSolverOptions(backend="gpu-mpi", dense_max_dim=0, carryover_fraction=0.4)
    args = ((strs_a, strs_b), np.zeros((NORB, NORB)), np.zeros((NORB,) * 4), NORB, (2, 3))
    sols = _run_ranks(comm, lambda: solve_fulqrum(*args, options))

    w_a, w_b = _expected_weights(hsub, len(strs_a), len(strs_b))
    occ = occupancies_from_marginals(w_a, w_b, strs_a, strs_b, NORB)
    for sol in sols:
        assert sol.metadata["backend"] == "gpu-mpi"
        assert sol.metadata["solver_path"] == "gpu-mpi-primme"
        assert sol.amplitudes is None
        np.testing.assert_allclose(sol.orbital_occupancies[0], occ[0], atol=1e-12)
        np.testing.assert_allclose(sol.orbital_occupancies[1], occ[1], atol=1e-12)
        np.testing.assert_allclose(sum(sol.orbital_occupancies[1]), 3.0)
        np.testing.assert_array_equal(sol.carryover_a, sols[0].carryover_a)
        np.testing.assert_array_equal(sol.carryover_b, sols[0].carryover_b)
    assert len(sols[0].carryover_a) == int(0.4 * len(strs_a))
    assert sols[0].carryover_b[0] == strs_b[np.argmax(w_b)]


def test_gpu_mpi_falls_back_to_rank_0_cpu_below_two_electrons(fake_gpu, fake_mpi, caplog):
    strs_a, strs_b, hsub = _problem(nelec=(2, 1))
    comm = fake_mpi(2)
    options = FulqrumSolverOptions(dense_max_dim=0, tol=1e-12)
    with caplog.at_level(logging.WARNING, logger=solver.logger.name):
        results = _run_ranks(
            comm, lambda: solver._solve_gpu_mpi(hsub, None, strs_a, strs_b, options, {})
        )

    w_a_ref, w_b_ref = _expected_weights(hsub, len(strs_a), len(strs_b))
    for energy, (w_a, w_b), path in results:
        assert path == "scipy-csr"
        assert energy == pytest.approx(hsub.ground_state()[0], abs=1e-10)
        np.testing.assert_allclose(w_a, w_a_ref, atol=1e-8)
        np.testing.assert_allclose(w_b, w_b_ref, atol=1e-8)
    assert not fake_gpu.calls
    assert caplog.text.count("solving on the CPU") == 1


# --- helpers ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({}, 0),
        ({"OMPI_COMM_WORLD_LOCAL_RANK": "3"}, 3),
        ({"MPI_LOCALRANKID": "2"}, 2),
        ({"SLURM_LOCALID": "1"}, 1),
        ({"OMPI_COMM_WORLD_LOCAL_RANK": "1", "MPI_LOCALRANKID": "2", "SLURM_LOCALID": "3"}, 1),
        ({"MPI_LOCALRANKID": "2", "SLURM_LOCALID": "3"}, 2),
        ({"OMPI_COMM_WORLD_LOCAL_RANK": "x", "SLURM_LOCALID": "3"}, 3),
        ({"SLURM_LOCALID": ""}, 0),
        # Job-wide ranks are not node-local and are not consulted.
        ({"OMPI_COMM_WORLD_RANK": "5", "SLURM_PROCID": "5"}, 0),
    ],
)
def test_local_rank(monkeypatch, env, expected):
    for name in (*solver._LOCAL_RANK_ENV_VARS, "OMPI_COMM_WORLD_RANK", "SLURM_PROCID"):
        monkeypatch.delenv(name, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    assert solver._local_rank() == expected


@pytest.mark.parametrize(("n", "parts"), [(5, 2), (6, 2), (7, 3), (2, 4), (10, 1)])
def test_split_even_matches_fulqrum(n, parts):
    ranges = solver._split_even(n, parts)
    sizes = [b - a for a, b in ranges]
    assert ranges[0][0] == 0 and ranges[-1][1] == n
    assert all(ranges[i][1] == ranges[i + 1][0] for i in range(parts - 1))
    # The first n % parts ranges are one longer.
    assert sizes == [n // parts + (i < n % parts) for i in range(parts)]


def test_fold_tile_weights_is_beta_outer_alpha_inner():
    na, nb = 3, 4
    probs = np.arange(1.0, na * nb + 1)
    full = np.sqrt(probs)
    total_a, total_b = np.zeros(na), np.zeros(nb)
    for ranges in [(0, 2, 0, 1), (2, 3, 0, 1), (0, 2, 1, 4), (2, 3, 1, 4)]:
        a0, a1, b0, b1 = ranges
        tile = full.reshape(nb, na)[b0:b1, a0:a1].ravel()
        w_a, w_b = solver._fold_tile_weights(tile, na, nb, ranges)
        total_a += w_a
        total_b += w_b
    ref_a, ref_b = marginal_weights(probs, na, nb)
    np.testing.assert_allclose(total_a, ref_a)
    np.testing.assert_allclose(total_b, ref_b)
