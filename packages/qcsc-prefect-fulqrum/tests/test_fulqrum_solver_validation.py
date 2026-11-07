"""Input validation for the solver core. These run without fulqrum installed."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest
from qcsc_prefect_fulqrum import solver
from qcsc_prefect_fulqrum.solver import FulqrumSolverOptions, _unique_ci_strs, solve_fulqrum


def test_options_reject_unknown_values():
    with pytest.raises(ValueError, match="eigensolver"):
        FulqrumSolverOptions(eigensolver="lapack")
    with pytest.raises(ValueError, match="hamiltonian_format"):
        FulqrumSolverOptions(hamiltonian_format="dense")
    with pytest.raises(ValueError, match="tol"):
        FulqrumSolverOptions(tol=-1.0)
    with pytest.raises(ValueError, match="max_iter"):
        FulqrumSolverOptions(max_iter=0)
    with pytest.raises(ValueError, match="backend"):
        FulqrumSolverOptions(backend="tpu")
    with pytest.raises(ValueError, match="carryover_fraction"):
        FulqrumSolverOptions(carryover_fraction=1.5)


def test_unique_ci_strs_sorts_and_deduplicates():
    out = _unique_ci_strs(np.array([0b110, 0b011, 0b110, 0b101]), norb=3, nelec=2, spin="alpha")
    assert out.tolist() == [0b011, 0b101, 0b110]
    assert out.dtype == np.int64


def test_unique_ci_strs_rejects_wrong_electron_count():
    with pytest.raises(ValueError, match="do not have 2 electrons"):
        _unique_ci_strs(np.array([0b011, 0b111]), norb=3, nelec=2, spin="alpha")


def test_unique_ci_strs_rejects_out_of_range_and_empty():
    with pytest.raises(ValueError, match="2\\*\\*norb"):
        _unique_ci_strs(np.array([0b1001]), norb=3, nelec=2, spin="beta")
    with pytest.raises(ValueError, match="No beta"):
        _unique_ci_strs(np.array([], dtype=np.int64), norb=3, nelec=2, spin="beta")


def test_solve_rejects_bad_tensor_shapes():
    strs = (np.array([0b01]), np.array([0b01]))
    with pytest.raises(ValueError, match="one_body_tensor"):
        solve_fulqrum(strs, np.zeros((3, 3)), np.zeros((2,) * 4), norb=2, nelec=(1, 1))
    with pytest.raises(ValueError, match="two_body_tensor"):
        solve_fulqrum(strs, np.zeros((2, 2)), np.zeros((2, 2)), norb=2, nelec=(1, 1))


def test_tuning_bounds_are_checked():
    for name in ("max_basis_size", "block_size", "min_restart_size"):
        with pytest.raises(ValueError, match=name):
            FulqrumSolverOptions(eigensolver="primme", **{name: 0})
    with pytest.raises(ValueError, match="convergence"):
        FulqrumSolverOptions(backend="gpu-mpi", convergence="relative")
    with pytest.raises(ValueError, match="energy_tol"):
        FulqrumSolverOptions(backend="gpu-mpi", energy_tol=0.0)
    with pytest.raises(ValueError, match="comm_backend"):
        FulqrumSolverOptions(backend="gpu-mpi", comm_backend="gloo")
    with pytest.raises(ValueError, match="process_grid"):
        FulqrumSolverOptions(backend="gpu-mpi", process_grid=(0, 2))


def test_options_the_eigensolver_ignores_are_rejected():
    """An option the selected solver would not use fails here, not on the node."""
    with pytest.raises(ValueError, match="block_size apply to PRIMME only"):
        FulqrumSolverOptions(block_size=4)
    with pytest.raises(ValueError, match="process_grid apply to backend='gpu-mpi' only"):
        FulqrumSolverOptions(eigensolver="primme", process_grid=(1, 2))
    # max_basis_size is scipy's ncv too.
    assert FulqrumSolverOptions(max_basis_size=30).max_basis_size == 30
    assert FulqrumSolverOptions(eigensolver="primme", block_size=4, primme_method="PRIMME_DYNAMIC")


def test_a_json_process_grid_compares_equal_to_the_tuple():
    assert FulqrumSolverOptions(backend="gpu-mpi", process_grid=[2, 1]).process_grid == (2, 1)


def test_scipy_and_primme_get_their_tuning(monkeypatch):
    import scipy.sparse.linalg as spla

    seen = {}

    def fake_eigsh(op, **kwargs):
        seen.update(kwargs)
        return np.array([-1.0]), np.ones((4, 1))

    monkeypatch.setattr(spla, "eigsh", fake_eigsh)
    solver._eigsh_lowest(None, 4, np.ones(4), FulqrumSolverOptions(max_basis_size=40))
    # ARPACK needs ncv <= dim.
    assert seen["ncv"] == 4

    primme = types.ModuleType("primme")
    primme.eigsh = lambda op, **kwargs: (
        (seen.clear(), seen.update(kwargs))[0]
        or (
            np.array([-1.0]),
            np.ones((4, 1)),
        )
    )
    monkeypatch.setitem(sys.modules, "primme", primme)
    options = FulqrumSolverOptions(
        eigensolver="primme",
        max_basis_size=16,
        block_size=2,
        min_restart_size=4,
        primme_method="PRIMME_DYNAMIC",
    )
    solver._eigsh_lowest(None, 4, np.ones(4), options)
    assert (seen["ncv"], seen["maxBlockSize"], seen["minRestartSize"], seen["method"]) == (
        16,
        2,
        4,
        "PRIMME_DYNAMIC",
    )
