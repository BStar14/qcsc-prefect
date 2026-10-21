"""Input validation for the solver core. These run without fulqrum installed."""

from __future__ import annotations

import numpy as np
import pytest
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
