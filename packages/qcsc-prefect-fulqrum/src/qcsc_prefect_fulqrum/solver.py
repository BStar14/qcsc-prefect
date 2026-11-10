"""Fulqrum subspace eigensolver core.

This module has no Prefect or qiskit-addon-sqd dependency so it can run in a
minimal compute-node environment (numpy, scipy, fulqrum). ``fulqrum`` and
``scipy`` are imported lazily inside :func:`solve_fulqrum`.

Conventions:

- CI strings are integers where bit ``i`` is the occupation of spatial orbital
  ``i`` (the same convention as ``qiskit_addon_sqd`` and PySCF).
- Fulqrum half strings are fixed-width binary strings with orbital 0 rightmost,
  so ``format(ci_str, f"0{norb}b")`` maps one to the other. A full fulqrum basis
  state is ``beta_half + alpha_half``.
- ``fulqrum.Subspace([alpha, beta])`` sorts both lists and stores the Cartesian
  product with basis index ``ib * len(alpha) + ia``. The returned amplitude matrix
  is reshaped to ``amplitudes[ia, ib]``, matching ``SCIState``.
- The returned energy is electronic only. The caller adds the core/nuclear
  constant, as with the DICE solver.

Backends (:attr:`FulqrumSolverOptions.backend`):

- ``"cpu"``: one OpenMP process.
- ``"gpu"``: one process using the GPUs of its node (``fulqrum.gpu.GPUHamiltonian``).
- ``"gpu-mpi"``: one MPI rank per GPU, across nodes
  (``fulqrum.gpu.DistributedMPIGPUHamiltonian``). The state stays distributed, so
  this backend returns marginal weights instead of a vector and no amplitudes.

Subspaces small enough for a dense solve use the CPU path with every backend.
"""

from __future__ import annotations

import importlib
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .carryover import marginal_weights, occupancies_from_marginals, select_carryover

logger = logging.getLogger(__name__)

Eigensolver = Literal["scipy", "primme"]
HamiltonianFormat = Literal["csr", "matrix_free"]
Backend = Literal["cpu", "gpu", "gpu-mpi"]
BACKENDS = ("cpu", "gpu", "gpu-mpi")
Convergence = Literal["abs", "rel", "norm", "energy"]
CONVERGENCE_TESTS = ("abs", "rel", "norm", "energy")
CommBackend = Literal["auto", "nccl", "host_mpi", "cuda_mpi"]
COMM_BACKENDS = ("auto", "nccl", "host_mpi", "cuda_mpi")
# Distributed PRIMME's default method in fulqrum 1.0 (primme_distributed.py at 2e2d66e).
GPU_MPI_DEFAULT_METHOD = "PRIMME_GD_Olsen_plusK"


@dataclass(frozen=True)
class FulqrumSolverOptions:
    """Numerical controls for one fulqrum subspace solve.

    Attributes:
        eigensolver: ``"scipy"`` (ARPACK ``eigsh``) or ``"primme"`` (requires the
            optional ``primme`` package).
        hamiltonian_format: ``"csr"`` builds an explicit sparse matrix (faster,
            more memory). ``"matrix_free"`` applies the projected Hamiltonian on
            the fly (slower, minimal memory).
        tol: Eigensolver convergence tolerance. ``0`` means machine precision.
        max_iter: Maximum eigensolver iterations. ``None`` uses the solver default.
        dense_max_dim: Subspaces up to this dimension are diagonalized densely
            with ``numpy.linalg.eigh``. Dimensions below 3 are always dense
            because ARPACK cannot handle them.
        return_amplitudes: Return the ground-state amplitude matrix. Ignored by
            the ``"gpu-mpi"`` backend, whose state stays distributed.
        backend: ``"cpu"``, ``"gpu"`` or ``"gpu-mpi"``; see the module docstring.
        carryover_fraction: Share of each spin's CI strings returned as carryover,
            heaviest first, in ``[0, 1]``. ``0`` returns none.
        max_basis_size: Largest search subspace of the iterative eigensolver
            (``ncv`` of scipy, PRIMME and distributed PRIMME). ``None`` uses the
            solver default.
        block_size: PRIMME block size (``maxBlockSize``), the number of vectors
            expanded per step. ``primme`` and ``gpu-mpi`` only.
        min_restart_size: Vectors PRIMME keeps at a restart (``minRestartSize``).
            ``primme`` and ``gpu-mpi`` only.
        primme_method: PRIMME preset (``method``). ``None`` keeps PRIMME's choice
            for ``primme`` and ``PRIMME_GD_Olsen_plusK`` for ``gpu-mpi``.
        convergence: Stopping test of distributed PRIMME (``conv``): ``abs``,
            ``rel``, ``norm`` or ``energy``. ``gpu-mpi`` only.
        energy_tol: Ritz-value stability tolerance of the ``energy`` test.
            ``gpu-mpi`` only.
        process_grid: ``(p_alpha, p_beta)`` ranks over the alpha and beta strings,
            whose product is the number of ranks. ``None`` lets fulqrum choose.
            ``gpu-mpi`` only.
        comm_backend: Transport between ranks: ``auto``, ``nccl``, ``host_mpi``
            or ``cuda_mpi``. ``gpu-mpi`` only.
    """

    eigensolver: Eigensolver = "scipy"
    hamiltonian_format: HamiltonianFormat = "csr"
    tol: float = 1e-8
    max_iter: int | None = None
    dense_max_dim: int = 64
    return_amplitudes: bool = True
    backend: Backend = "cpu"
    carryover_fraction: float = 0.0
    max_basis_size: int | None = None
    block_size: int | None = None
    min_restart_size: int | None = None
    primme_method: str | None = None
    convergence: Convergence = "abs"
    energy_tol: float = 1e-6
    process_grid: tuple[int, int] | None = None
    comm_backend: CommBackend = "auto"

    def __post_init__(self) -> None:
        if self.eigensolver not in ("scipy", "primme"):
            raise ValueError(f"Unknown eigensolver: {self.eigensolver!r}")
        if self.hamiltonian_format not in ("csr", "matrix_free"):
            raise ValueError(f"Unknown hamiltonian_format: {self.hamiltonian_format!r}")
        if self.tol < 0:
            raise ValueError(f"tol must be >= 0, got {self.tol}")
        if self.max_iter is not None and self.max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {self.max_iter}")
        if self.backend not in BACKENDS:
            raise ValueError(f"Unknown backend: {self.backend!r}")
        if not 0.0 <= self.carryover_fraction <= 1.0:
            raise ValueError(f"carryover_fraction must be in [0, 1], got {self.carryover_fraction}")
        for name in ("max_basis_size", "block_size", "min_restart_size"):
            value = getattr(self, name)
            if value is not None and value < 1:
                raise ValueError(f"{name} must be >= 1, got {value}")
        if self.max_basis_size is not None:
            # One eigenpair needs a search subspace of at least 2 vectors (ARPACK: k < ncv).
            if self.max_basis_size < 2:
                raise ValueError(f"max_basis_size must be >= 2, got {self.max_basis_size}")
            for name in ("block_size", "min_restart_size"):
                value = getattr(self, name)
                if value is not None and value >= self.max_basis_size:
                    raise ValueError(
                        f"{name} ({value}) must be smaller than max_basis_size "
                        f"({self.max_basis_size})"
                    )
        if self.convergence not in CONVERGENCE_TESTS:
            raise ValueError(f"Unknown convergence: {self.convergence!r}")
        if self.energy_tol <= 0:
            raise ValueError(f"energy_tol must be > 0, got {self.energy_tol}")
        if self.comm_backend not in COMM_BACKENDS:
            raise ValueError(f"Unknown comm_backend: {self.comm_backend!r}")
        if self.process_grid is not None:
            grid = tuple(int(p) for p in self.process_grid)
            if len(grid) != 2 or min(grid) < 1:
                raise ValueError(f"process_grid must be two positive integers, got {grid}")
            # JSON gives a list back; keep a tuple so options compare equal.
            object.__setattr__(self, "process_grid", grid)
        # Options the selected eigensolver cannot use fail here, not on the node.
        primme_only = ("block_size", "min_restart_size", "primme_method")
        if self.backend != "gpu-mpi" and self.eigensolver == "scipy":
            unused = [name for name in primme_only if getattr(self, name) is not None]
            if unused:
                raise ValueError(
                    f"{', '.join(unused)} apply to PRIMME only; set eigensolver='primme' or "
                    "backend='gpu-mpi', or leave them unset"
                )
        if self.backend != "gpu-mpi":
            unused = [
                name
                for name, default in (
                    ("convergence", "abs"),
                    ("energy_tol", 1e-6),
                    ("process_grid", None),
                    ("comm_backend", "auto"),
                )
                if getattr(self, name) != default
            ]
            if unused:
                raise ValueError(
                    f"{', '.join(unused)} apply to backend='gpu-mpi' only; leave them unset"
                )


@dataclass
class FulqrumSolution:
    """Ground state of the Hamiltonian projected onto a CI subspace.

    Attributes:
        energy: Electronic ground-state energy (no nuclear/core constant).
        orbital_occupancies: Average ``(alpha, beta)`` occupancy per spatial orbital.
        ci_strs_a: Sorted unique alpha CI strings defining the subspace rows.
        ci_strs_b: Sorted unique beta CI strings defining the subspace columns.
        amplitudes: ``(len(ci_strs_a), len(ci_strs_b))`` amplitudes, or ``None``.
            The global sign is fixed so the largest-magnitude amplitude is positive.
        metadata: Subspace dimension, solver path, and timings in seconds.
        carryover_a: Alpha carryover strings, heaviest first (empty when off).
        carryover_b: Beta carryover strings, heaviest first (empty when off).
    """

    energy: float
    orbital_occupancies: tuple[np.ndarray, np.ndarray]
    ci_strs_a: np.ndarray
    ci_strs_b: np.ndarray
    amplitudes: np.ndarray | None
    metadata: dict = field(default_factory=dict)
    carryover_a: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    carryover_b: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))


def _unique_ci_strs(strs: np.ndarray, norb: int, nelec: int, spin: str) -> np.ndarray:
    values = sorted({int(x) for x in np.asarray(strs).reshape(-1)})
    if not values:
        raise ValueError(f"No {spin} CI strings given")
    if values[0] < 0 or values[-1] >= 1 << norb:
        raise ValueError(f"{spin} CI strings must be in [0, 2**norb) with norb={norb}")
    bad = [x for x in values if x.bit_count() != nelec]
    if bad:
        raise ValueError(
            f"{len(bad)} {spin} CI string(s) do not have {nelec} electrons, e.g. {bad[0]:#b}"
        )
    # Keep Python ints for norb > 63; int64 otherwise for compatibility with SCIState.
    return np.array(values, dtype=np.int64 if norb < 64 else object)


def normalize_problem(
    ci_strings: tuple[np.ndarray, np.ndarray],
    one_body_tensor: np.ndarray,
    two_body_tensor: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate a subspace problem and return it in canonical form.

    Returns:
        ``(strs_a, strs_b, h1, eri)``: sorted unique CI strings and C-contiguous
        integrals with a common dtype.

    Raises:
        ValueError: If shapes, electron counts, or CI string ranges are invalid.
    """
    h1 = np.ascontiguousarray(one_body_tensor)
    eri = np.ascontiguousarray(two_body_tensor)
    if h1.shape != (norb, norb):
        raise ValueError(f"one_body_tensor shape {h1.shape} != ({norb}, {norb})")
    if eri.shape != (norb,) * 4:
        raise ValueError(f"two_body_tensor shape {eri.shape} != {(norb,) * 4}")
    if np.iscomplexobj(h1) != np.iscomplexobj(eri):
        dtype = np.result_type(h1, eri)
        h1, eri = h1.astype(dtype), eri.astype(dtype)
    strs_a = _unique_ci_strs(ci_strings[0], norb, nelec[0], "alpha")
    strs_b = _unique_ci_strs(ci_strings[1], norb, nelec[1], "beta")
    return strs_a, strs_b, h1, eri


def _half_strings(ci_strs: np.ndarray, norb: int) -> list[str]:
    return [format(int(x), f"0{norb}b") for x in ci_strs]


def _initial_vector(hsub, dim: int, dtype) -> np.ndarray:
    """Unit weight on the lowest diagonal entries (fulqrum's recommended guess)."""
    diag = np.asarray(hsub.diagonal_vector()).real
    v0 = np.zeros(dim, dtype=dtype)
    v0[diag == diag.min()] = 1.0
    return v0


def _eigsh_lowest(op, dim: int, v0: np.ndarray, options: FulqrumSolverOptions):
    if options.eigensolver == "primme":
        import primme

        evals, evecs = primme.eigsh(
            op,
            k=1,
            which="SA",
            v0=v0.reshape(dim, 1),
            tol=options.tol,
            maxiter=options.max_iter,
            **_primme_options(options),
        )
    else:
        import scipy.sparse.linalg as spla

        extra = {}
        if options.max_basis_size is not None:
            # ARPACK needs k < ncv <= dim.
            extra["ncv"] = min(options.max_basis_size, dim)
        evals, evecs = spla.eigsh(
            op, k=1, which="SA", v0=v0, tol=options.tol, maxiter=options.max_iter, **extra
        )
    return float(np.real(evals[0])), np.asarray(evecs).reshape(dim)


def _primme_options(options: FulqrumSolverOptions) -> dict:
    """The set PRIMME options, as ``primme.eigsh`` and distributed PRIMME name them."""
    names = {
        "max_basis_size": "ncv",
        "block_size": "maxBlockSize",
        "min_restart_size": "minRestartSize",
        "primme_method": "method",
    }
    return {
        kwarg: getattr(options, field)
        for field, kwarg in names.items()
        if getattr(options, field) is not None
    }


def _build_hamiltonian(h1, eri, strs_a, strs_b, norb: int, timings: dict):
    """The subspace and its projected Hamiltonian; the same for every backend."""
    import fulqrum as fq
    from fulqrum.convert.integrals import integrals_to_fq_fermionic_op

    t0 = time.perf_counter()
    qop = integrals_to_fq_fermionic_op(
        one_body_integrals=h1, two_body_integrals=eri
    ).extended_jw_transformation()
    timings["operator"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    subspace = fq.Subspace([_half_strings(strs_a, norb), _half_strings(strs_b, norb)])
    hsub = fq.SubspaceHamiltonian(qop, subspace)
    timings["subspace"] = time.perf_counter() - t0
    return fq, qop, subspace, hsub


def _solve_cpu(hsub, dim: int, options: FulqrumSolverOptions, timings: dict):
    """Lowest eigenpair on the CPU: dense for small subspaces, else ARPACK or PRIMME.

    Returns:
        ``(energy, vector, solver_path, nnz)``; the vector is in fulqrum's basis order.
    """
    dense = dim <= max(options.dense_max_dim, 2)
    t0 = time.perf_counter()
    nnz = None
    if dense or options.hamiltonian_format == "csr":
        csr_op = hsub.to_csr_linearoperator_fast()
        nnz = int(csr_op.matrix.nnz)
        op = csr_op
    else:
        op = hsub
    timings["projection"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    if dense:
        evals, evecs = np.linalg.eigh(csr_op.matrix.toarray())
        energy, vec = float(evals[0]), evecs[:, 0]
        solver_path = "dense"
    else:
        v0 = _initial_vector(hsub, dim, hsub.dtype)
        energy, vec = _eigsh_lowest(op, dim, v0, options)
        solver_path = f"{options.eigensolver}-{options.hamiltonian_format}"
    timings["eigensolve"] = time.perf_counter() - t0
    return energy, vec, solver_path, nnz


# Node-local rank variables, in priority order, for choosing a rank's GPU.
_LOCAL_RANK_ENV_VARS = (
    "OMPI_COMM_WORLD_LOCAL_RANK",  # Open MPI
    "MPI_LOCALRANKID",  # Intel MPI, MPICH Hydra
    "SLURM_LOCALID",  # srun
)
# The GPU kernels are verified only with two or more electrons per half string.
_GPU_MIN_HALF_WEIGHT = 2
# Distributed PRIMME tests ``residual < tol``, so tol=0 would never converge.
_GPU_MPI_MIN_TOL = 1e-12


def _version_tuple(version: str | None) -> tuple[int, int] | None:
    match = re.match(r"\s*(\d+)\.(\d+)", version or "")
    return (int(match[1]), int(match[2])) if match else None


def _require_fulqrum_gpu(backend: str, hsub):
    """Import ``fulqrum.gpu`` and CuPy, or fail with what the backend needs."""
    if np.dtype(hsub.dtype).kind == "c":
        raise NotImplementedError(f"the {backend!r} backend supports real integrals only")
    try:
        import fulqrum
    except ImportError as exc:
        raise ImportError(f"the {backend!r} backend needs fulqrum: {exc}") from exc
    version = getattr(fulqrum, "__version__", None)
    parsed = _version_tuple(version)
    if parsed is not None and parsed < (1, 0):
        raise ImportError(
            f"the {backend!r} backend needs fulqrum >= 1.0 with GPU support, found {version}"
        )
    try:
        import cupy
        import fulqrum.gpu as fq_gpu

        # The compiled CUDA launcher exists only in an FQ_BUILD_GPU=1 build;
        # fulqrum.gpu itself imports in a CPU-only build too.
        importlib.import_module("fulqrum.gpu.distributed.kernel_launch")
    except ImportError as exc:
        raise ImportError(
            f"the {backend!r} backend needs fulqrum built with FQ_BUILD_GPU=1 and CuPy: {exc}"
        ) from exc
    return fq_gpu, cupy


def _device_count(cupy) -> int:
    # Counts the devices visible to this process, so CUDA_VISIBLE_DEVICES applies.
    count = int(cupy.cuda.runtime.getDeviceCount())
    if count < 1:
        raise RuntimeError("no CUDA device is visible to this process")
    return count


def _min_half_weight(strs_a, strs_b) -> int:
    # Every string of a spin has the same electron count (see normalize_problem).
    return min(int(strs_a[0]).bit_count(), int(strs_b[0]).bit_count())


def _warn_cpu_fallback(backend: str, weight: int) -> None:
    logger.warning(
        "Fulqrum %r backend: %d electron(s) in a spin is below the %d the GPU kernels "
        "support; solving on the CPU instead.",
        backend,
        weight,
        _GPU_MIN_HALF_WEIGHT,
    )


def _solve_gpu(hsub, dim: int, options: FulqrumSolverOptions, timings: dict):
    """Lowest eigenpair with ``fulqrum.gpu.GPUHamiltonian`` on this node's GPUs.

    The operator spans every visible device. It is a SciPy ``LinearOperator``
    over the full subspace, so ``scipy`` and ``primme`` run as on the CPU, from the
    same starting vector. ``_build_hamiltonian`` already gives what the GPU
    operator requires: a half-string ``Subspace`` and a type-2 operator
    (``extended_jw_transformation`` sets type 2). A spin with fewer than two
    electrons is outside the verified GPU support and is solved on the CPU.

    Returns:
        ``(energy, vector, solver_path, nnz)``, as :func:`_solve_cpu`.

    Raises:
        NotImplementedError: If the integrals are complex.
        ImportError: If fulqrum >= 1.0 with GPU support or CuPy is missing.
        RuntimeError: If no CUDA device is visible.
    """
    fq_gpu, cupy = _require_fulqrum_gpu("gpu", hsub)
    subspace = hsub.spmv.subspace
    weight = min(subspace.alpha_strs[0].count("1"), subspace.beta_strs[0].count("1"))
    if weight < _GPU_MIN_HALF_WEIGHT:
        _warn_cpu_fallback("gpu", weight)
        return _solve_cpu(hsub, dim, options, timings)

    gpu_ids = tuple(range(_device_count(cupy)))
    t0 = time.perf_counter()
    op = fq_gpu.GPUHamiltonian(hsub, gpu_ids=gpu_ids)
    timings["projection"] = time.perf_counter() - t0
    logger.info("Fulqrum GPU operator on devices %s", gpu_ids)

    t0 = time.perf_counter()
    v0 = _initial_vector(hsub, dim, np.float64)
    energy, vec = _eigsh_lowest(op, dim, v0, options)
    timings["eigensolve"] = time.perf_counter() - t0
    return energy, vec, f"gpu-{options.eigensolver}", None


def _local_rank() -> int:
    """This process's rank on its node, from the launcher; 0 if none is set."""
    for name in _LOCAL_RANK_ENV_VARS:
        value = os.environ.get(name)
        if value is None:
            continue
        try:
            return int(value)
        except ValueError:
            logger.warning("Ignoring non-integer %s=%r.", name, value)
    return 0


def _split_even(n: int, parts: int) -> list[tuple[int, int]]:
    # Same split as fulqrum.gpu.distributed.dist_matvec._split_ranges_even
    # (dist_matvec.py:62-71 at 2e2d66e): the first n % parts ranges get one extra.
    base, rem = divmod(int(n), int(parts))
    ranges, start = [], 0
    for part in range(int(parts)):
        stop = start + base + (1 if part < rem else 0)
        ranges.append((start, stop))
        start = stop
    return ranges


def _tile_ranges(H, na: int, nb: int) -> tuple[int, int, int, int]:
    """``(a0, a1, b0, b1)``: the alpha and beta index ranges of this rank's tile.

    In fulqrum 2e2d66e (fulqrum/gpu/distributed/):

    - dist_matvec_tile_resident.py:450-452: ``coord_alpha = rank // p_beta`` and
      ``coord_beta = rank % p_beta`` (also ``TileGrid.coords``, grid.py:58-60).
    - dist_matvec_tile_resident.py:489-494 and grid.py:39-51: the ranges are
      ``_split_ranges_even(n_a, p_alpha)[coord_alpha]`` and
      ``_split_ranges_even(n_b, p_beta)[coord_beta]`` over the sorted strings
      (``_resolve_uniq_arrays``, :1307-1316), stored as the private ``_a0, _a1,
      _b0, _b1``. ``GPUHamiltonian`` reads those (gpu_hamiltonian.py:147).

    Only ``process_grid``, ``coord_alpha``, ``coord_beta``, ``n_a_local`` and
    ``n_b_local`` are public, so the ranges are rebuilt from them and checked
    against the local sizes, and against the private fields where present.

    Raises:
        RuntimeError: If the rebuilt ranges disagree with the operator.
    """
    p_alpha, p_beta = (int(p) for p in H.process_grid)
    a0, a1 = _split_even(na, p_alpha)[int(H.coord_alpha)]
    b0, b1 = _split_even(nb, p_beta)[int(H.coord_beta)]
    expected = {"n_a_local": a1 - a0, "n_b_local": b1 - b0}
    actual = {name: int(getattr(H, name)) for name in expected}
    for name, value in {"_a0": a0, "_a1": a1, "_b0": b0, "_b1": b1}.items():
        if hasattr(H, name):
            expected[name], actual[name] = value, int(getattr(H, name))
    if actual != expected:
        raise RuntimeError(
            f"unexpected fulqrum tile layout: expected {expected}, the operator has {actual}"
        )
    return a0, a1, b0, b1


def _fold_tile_weights(evec_local, na: int, nb: int, ranges) -> tuple[np.ndarray, np.ndarray]:
    """This rank's share of the marginal weights, as global-length vectors.

    The local tile is beta-outer, alpha-inner: local index ``b * n_a_local + a``
    (dist_matvec_tile_resident.py:265-268 at 2e2d66e).
    """
    a0, a1, b0, b1 = ranges
    tile = np.abs(np.asarray(evec_local, dtype=np.float64).reshape(b1 - b0, a1 - a0)) ** 2
    w_a = np.zeros(na, dtype=np.float64)
    w_b = np.zeros(nb, dtype=np.float64)
    w_a[a0:a1] = tile.sum(axis=0)
    w_b[b0:b1] = tile.sum(axis=1)
    return w_a, w_b


def _distributed_v0(diag_local: np.ndarray, comm, MPI) -> np.ndarray:
    """Unit vector at the lowest diagonal entry over all ranks (lowest rank on ties)."""
    local_min = float(diag_local.min()) if diag_local.size else float("inf")
    owner = comm.allreduce((local_min, comm.Get_rank()), op=MPI.MINLOC)[1]
    v0 = np.zeros(diag_local.size, dtype=np.float64)
    if comm.Get_rank() == owner:
        v0[int(np.argmin(diag_local))] = 1.0
    return v0


def _solve_gpu_mpi_on_cpu(hsub, strs_a, strs_b, options, timings, comm):
    # Rank 0 solves and broadcasts, so every rank returns the same values.
    na, nb = len(strs_a), len(strs_b)
    result = None
    if comm.Get_rank() == 0:
        energy, vec, solver_path, _ = _solve_cpu(hsub, na * nb, options, timings)
        probs = np.abs(vec) ** 2
        w_a, w_b = marginal_weights(probs / probs.sum(), na, nb)
        result = (energy, w_a, w_b, solver_path)
    energy, w_a, w_b, solver_path = comm.bcast(result, root=0)
    return energy, (w_a, w_b), solver_path


def _solve_gpu_mpi(hsub, subspace, strs_a, strs_b, options: FulqrumSolverOptions, timings: dict):
    """Lowest eigenpair with one MPI rank per GPU; collective on ``MPI.COMM_WORLD``.

    The eigenvector is never gathered: every rank folds its tile into the
    marginal weights, which are then summed over the ranks.

    Each rank uses device ``local_rank % device_count``, with the node-local
    rank from :data:`_LOCAL_RANK_ENV_VARS`. The solver is fulqrum's distributed
    PRIMME, started from the lowest diagonal entry and preconditioned with the
    diagonal; ``eigensolver`` and ``hamiltonian_format`` do not apply. Its
    convergence test is an absolute residual below ``tol`` (``tol=0`` uses
    ``1e-12``), and ``max_iter`` caps its matrix-vector products. A spin with
    fewer than two electrons is solved on the CPU of rank 0 and broadcast.

    Returns:
        ``(energy, (w_a, w_b), solver_path)``, identical on every rank. The weights
        follow the order of ``strs_a`` and ``strs_b`` and sum to 1.

    Raises:
        NotImplementedError: If the integrals are complex.
        ImportError: If mpi4py, fulqrum >= 1.0 with GPU support, CuPy or
            fulqrum's distributed PRIMME is missing.
        RuntimeError: If no CUDA device is visible or the tile layout is not
            the expected one.
    """
    try:
        from mpi4py import MPI
    except ImportError as exc:
        raise ImportError(f"the 'gpu-mpi' backend needs mpi4py: {exc}") from exc
    fq_gpu, cupy = _require_fulqrum_gpu("gpu-mpi", hsub)
    if not fq_gpu.is_primme_mpi_available():
        raise ImportError(
            "the 'gpu-mpi' backend needs fulqrum's distributed PRIMME "
            "(fulqrum.gpu.eigensolver._primme_mpi), built with FQ_BUILD_GPU=1"
        )
    comm = MPI.COMM_WORLD
    weight = _min_half_weight(strs_a, strs_b)
    if weight < _GPU_MIN_HALF_WEIGHT:
        if comm.Get_rank() == 0:
            _warn_cpu_fallback("gpu-mpi", weight)
        return _solve_gpu_mpi_on_cpu(hsub, strs_a, strs_b, options, timings, comm)

    na, nb = len(strs_a), len(strs_b)
    device = _local_rank() % _device_count(cupy)
    t0 = time.perf_counter()
    grid_kwargs = {} if options.process_grid is None else {"process_grid": options.process_grid}
    if options.process_grid is not None and np.prod(options.process_grid) != comm.Get_size():
        raise ValueError(
            f"process_grid {options.process_grid} needs {int(np.prod(options.process_grid))} "
            f"ranks; the job runs {comm.Get_size()}"
        )
    H = fq_gpu.DistributedMPIGPUHamiltonian(
        hsub, comm=comm, gpu_ids=(device,), comm_backend=options.comm_backend, **grid_kwargs
    )
    ranges = _tile_ranges(H, na, nb)
    timings["projection"] = time.perf_counter() - t0
    if comm.Get_rank() == 0:
        logger.info(
            "Fulqrum distributed GPU operator: %d ranks, process grid %s",
            comm.Get_size(),
            tuple(H.process_grid),
        )

    t0 = time.perf_counter()
    diag = np.asarray(H.diag_local, dtype=np.float64)
    evals, evecs = fq_gpu.distributed_primme_eigsh(
        H,
        k=1,
        which="SA",
        tol=options.tol if options.tol > 0 else _GPU_MPI_MIN_TOL,
        maxiter=options.max_iter,
        v0_local=_distributed_v0(diag, comm, MPI),
        precond_diag_local=diag,
        gpu="auto",
        conv=options.convergence,
        energy_tol=options.energy_tol,
        **{"method": GPU_MPI_DEFAULT_METHOD, **_primme_options(options)},
    )
    energy = float(np.real(evals[0]))
    timings["eigensolve"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    w_a, w_b = _fold_tile_weights(np.asarray(evecs)[:, 0], na, nb, ranges)
    weights = np.concatenate([w_a, w_b])
    comm.Allreduce(MPI.IN_PLACE, weights, op=MPI.SUM)
    # The vector norm is global, so normalize after the sum.
    w_a, w_b = weights[:na], weights[na:]
    total = float(w_a.sum())
    if not total > 0:
        raise RuntimeError("distributed PRIMME returned a zero eigenvector")
    timings["weights"] = time.perf_counter() - t0
    return energy, (w_a / total, w_b / total), "gpu-mpi-primme"


def solve_fulqrum(
    ci_strings: tuple[np.ndarray, np.ndarray],
    one_body_tensor: np.ndarray,
    two_body_tensor: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    options: FulqrumSolverOptions | None = None,
) -> FulqrumSolution:
    """Diagonalize the molecular Hamiltonian in the subspace spanned by CI strings.

    Args:
        ci_strings: ``(alpha, beta)`` integer CI strings. Duplicates are removed
            and both are sorted; the subspace is their Cartesian product.
        one_body_tensor: ``(norb, norb)`` one-electron integrals.
        two_body_tensor: ``(norb, norb, norb, norb)`` two-electron integrals in
            PySCF chemist notation (full 4-index, e.g. ``ao2mo.restore(1, ...)``).
        norb: Number of spatial orbitals.
        nelec: Number of ``(alpha, beta)`` electrons. Every CI string must match.
        options: Solver controls. Defaults to :class:`FulqrumSolverOptions`.

    Returns:
        The ground-state energy, occupancies, carryover and (optionally) amplitudes.

    Raises:
        ValueError: If inputs are inconsistent (shapes, electron counts, range).
    """
    options = options or FulqrumSolverOptions()
    strs_a, strs_b, h1, eri = normalize_problem(
        ci_strings, one_body_tensor, two_body_tensor, norb, nelec
    )
    na, nb = len(strs_a), len(strs_b)
    dim = na * nb

    timings: dict[str, float] = {}
    fq, _, subspace, hsub = _build_hamiltonian(h1, eri, strs_a, strs_b, norb, timings)
    logger.info("Fulqrum subspace %d x %d = %d (norb=%d, nelec=%s)", na, nb, dim, norb, nelec)

    backend = "cpu" if dim <= max(options.dense_max_dim, 2) else options.backend
    nnz = None
    amplitudes = None
    if backend == "gpu-mpi":
        energy, (w_a, w_b), solver_path = _solve_gpu_mpi(
            hsub, subspace, strs_a, strs_b, options, timings
        )
        occ_a, occ_b = occupancies_from_marginals(w_a, w_b, strs_a, strs_b, norb)
    else:
        solve = _solve_gpu if backend == "gpu" else _solve_cpu
        energy, vec, solver_path, nnz = solve(hsub, dim, options, timings)

        vec = vec / np.linalg.norm(vec)
        pivot = vec[np.argmax(np.abs(vec))]
        vec = vec * (np.abs(pivot) / pivot)
        if not np.iscomplexobj(h1):
            vec = vec.real

        probs = np.ascontiguousarray(np.abs(vec) ** 2, dtype=np.float64)
        occ_a, occ_b = subspace.get_orbital_occupancies(probs, norb)
        w_a, w_b = marginal_weights(probs, na, nb)
        # Basis index is ib * na + ia; transpose to amplitudes[ia, ib].
        if options.return_amplitudes:
            amplitudes = vec.reshape(nb, na).T.copy()

    carryover_a, carryover_b = select_carryover(
        w_a, w_b, strs_a, strs_b, options.carryover_fraction
    )

    metadata = {
        "subspace_dim": dim,
        "dim_a": na,
        "dim_b": nb,
        "nnz": nnz,
        "solver_path": solver_path,
        "backend": backend,
        "num_carryover_a": len(carryover_a),
        "num_carryover_b": len(carryover_b),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "fulqrum_version": getattr(fq, "__version__", None),
        "timings": timings,
    }
    logger.info("Fulqrum energy %.12f via %s (%s)", energy, solver_path, timings)
    return FulqrumSolution(
        energy=energy,
        orbital_occupancies=(np.asarray(occ_a), np.asarray(occ_b)),
        ci_strs_a=strs_a,
        ci_strs_b=strs_b,
        amplitudes=amplitudes,
        metadata=metadata,
        carryover_a=carryover_a,
        carryover_b=carryover_b,
    )
