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

import logging
import os
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
    """

    eigensolver: Eigensolver = "scipy"
    hamiltonian_format: HamiltonianFormat = "csr"
    tol: float = 1e-8
    max_iter: int | None = None
    dense_max_dim: int = 64
    return_amplitudes: bool = True
    backend: Backend = "cpu"
    carryover_fraction: float = 0.0

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
        )
    else:
        import scipy.sparse.linalg as spla

        evals, evecs = spla.eigsh(
            op, k=1, which="SA", v0=v0, tol=options.tol, maxiter=options.max_iter
        )
    return float(np.real(evals[0])), np.asarray(evecs).reshape(dim)


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


def _solve_gpu(hsub, dim: int, options: FulqrumSolverOptions, timings: dict):
    """Lowest eigenpair with ``fulqrum.gpu.GPUHamiltonian`` on this node's GPUs.

    Returns:
        ``(energy, vector, solver_path, nnz)``, as :func:`_solve_cpu`.
    """
    raise NotImplementedError("the 'gpu' fulqrum backend is not implemented yet")


def _solve_gpu_mpi(hsub, subspace, strs_a, strs_b, options: FulqrumSolverOptions, timings: dict):
    """Lowest eigenpair with one MPI rank per GPU; collective on ``MPI.COMM_WORLD``.

    The eigenvector is never gathered: every rank folds its tile into the
    marginal weights, which are then summed over the ranks.

    Returns:
        ``(energy, (w_a, w_b), solver_path)``, identical on every rank. The weights
        follow the order of ``strs_a`` and ``strs_b`` and sum to 1.
    """
    raise NotImplementedError("the 'gpu-mpi' fulqrum backend is not implemented yet")


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
