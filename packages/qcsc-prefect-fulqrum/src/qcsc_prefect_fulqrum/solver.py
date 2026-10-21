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
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)

Eigensolver = Literal["scipy", "primme"]
HamiltonianFormat = Literal["csr", "matrix_free"]


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
        return_amplitudes: Return the ground-state amplitude matrix.
    """

    eigensolver: Eigensolver = "scipy"
    hamiltonian_format: HamiltonianFormat = "csr"
    tol: float = 1e-8
    max_iter: int | None = None
    dense_max_dim: int = 64
    return_amplitudes: bool = True

    def __post_init__(self) -> None:
        if self.eigensolver not in ("scipy", "primme"):
            raise ValueError(f"Unknown eigensolver: {self.eigensolver!r}")
        if self.hamiltonian_format not in ("csr", "matrix_free"):
            raise ValueError(f"Unknown hamiltonian_format: {self.hamiltonian_format!r}")
        if self.tol < 0:
            raise ValueError(f"tol must be >= 0, got {self.tol}")
        if self.max_iter is not None and self.max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {self.max_iter}")


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
    """

    energy: float
    orbital_occupancies: tuple[np.ndarray, np.ndarray]
    ci_strs_a: np.ndarray
    ci_strs_b: np.ndarray
    amplitudes: np.ndarray | None
    metadata: dict = field(default_factory=dict)


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
        The ground-state energy, occupancies, and (optionally) amplitudes.

    Raises:
        ValueError: If inputs are inconsistent (shapes, electron counts, range).
    """
    options = options or FulqrumSolverOptions()
    strs_a, strs_b, h1, eri = normalize_problem(
        ci_strings, one_body_tensor, two_body_tensor, norb, nelec
    )
    na, nb = len(strs_a), len(strs_b)
    dim = na * nb

    import fulqrum as fq
    from fulqrum.convert.integrals import integrals_to_fq_fermionic_op

    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    qop = integrals_to_fq_fermionic_op(
        one_body_integrals=h1, two_body_integrals=eri
    ).extended_jw_transformation()
    timings["operator"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    subspace = fq.Subspace([_half_strings(strs_a, norb), _half_strings(strs_b, norb)])
    hsub = fq.SubspaceHamiltonian(qop, subspace)
    timings["subspace"] = time.perf_counter() - t0
    logger.info("Fulqrum subspace %d x %d = %d (norb=%d, nelec=%s)", na, nb, dim, norb, nelec)

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

    vec = vec / np.linalg.norm(vec)
    pivot = vec[np.argmax(np.abs(vec))]
    vec = vec * (np.abs(pivot) / pivot)
    if not np.iscomplexobj(h1):
        vec = vec.real

    probs = np.ascontiguousarray(np.abs(vec) ** 2, dtype=np.float64)
    occ_a, occ_b = subspace.get_orbital_occupancies(probs, norb)
    # Basis index is ib * na + ia; transpose to amplitudes[ia, ib].
    amplitudes = vec.reshape(nb, na).T.copy() if options.return_amplitudes else None

    metadata = {
        "subspace_dim": dim,
        "dim_a": na,
        "dim_b": nb,
        "nnz": nnz,
        "solver_path": solver_path,
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
    )
