"""Numerical checks of the fulqrum solver against PySCF FCI and qiskit-addon-sqd.

These need fulqrum and PySCF, so they skip in CI. Run them with the Python of the
environment where fulqrum is built:

    python -m pytest packages/qcsc-prefect-fulqrum/tests
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pytest

pytest.importorskip("fulqrum")
pytest.importorskip("pyscf")

from pyscf import ao2mo, fci, gto, mcscf, scf  # noqa: E402
from pyscf.fci import cistring  # noqa: E402
from qcsc_prefect_fulqrum.solver import FulqrumSolverOptions, solve_fulqrum  # noqa: E402

MOLECULES = {
    # name: (atom, basis, spin, number of frozen core orbitals)
    "h2": ("H 0 0 0; H 0 0 0.74", "sto-3g", 0, 0),
    "lih": ("Li 0 0 0; H 0 0 1.6", "sto-3g", 0, 0),
    "n2": ("N 0 0 0; N 0 0 1.1", "sto-3g", 0, 2),
    # Open shell (triplet O2) exercises nelec[0] != nelec[1].
    "o2": ("O 0 0 0; O 0 0 1.21", "sto-3g", 2, 2),
}


@lru_cache(maxsize=None)
def _problem(name: str):
    atom, basis, spin, ncore = MOLECULES[name]
    mol = gto.M(atom=atom, basis=basis, spin=spin, verbose=0)
    mf = (scf.ROHF if spin else scf.RHF)(mol).run()
    norb = mol.nao_nr() - ncore
    nel = mol.nelectron - 2 * ncore
    nelec = ((nel + spin) // 2, (nel - spin) // 2)
    cas = mcscf.CASCI(mf, norb, nelec)
    h1, ecore = cas.get_h1cas()
    eri = ao2mo.restore(1, cas.get_h2cas(), norb)
    return np.ascontiguousarray(h1), np.ascontiguousarray(eri), norb, nelec, ecore


def _all_strs(norb: int, n: int) -> np.ndarray:
    return np.asarray(cistring.make_strings(range(norb), n), dtype=np.int64)


def _assert_same_up_to_sign(a: np.ndarray, b: np.ndarray, atol: float) -> None:
    sign = np.sign(np.vdot(a.ravel(), b.ravel()).real) or 1.0
    np.testing.assert_allclose(sign * a, b, atol=atol)


@pytest.mark.parametrize("name", list(MOLECULES))
def test_full_space_matches_pyscf_fci(name):
    h1, eri, norb, nelec, _ = _problem(name)
    e_fci, civec = fci.direct_spin1.kernel(h1, eri, norb, nelec, conv_tol=1e-12)
    strs_a, strs_b = _all_strs(norb, nelec[0]), _all_strs(norb, nelec[1])

    sol = solve_fulqrum((strs_a, strs_b), h1, eri, norb, nelec, FulqrumSolverOptions(tol=1e-12))

    assert sol.energy == pytest.approx(e_fci, abs=1e-9)
    # PySCF orders the FCI vector by ascending CI string, as does the solver.
    np.testing.assert_array_equal(sol.ci_strs_a, strs_a)
    np.testing.assert_array_equal(sol.ci_strs_b, strs_b)
    _assert_same_up_to_sign(sol.amplitudes, civec, atol=1e-5)

    dm1a, dm1b = fci.direct_spin1.make_rdm1s(civec, norb, nelec)
    np.testing.assert_allclose(sol.orbital_occupancies[0], np.diag(dm1a), atol=1e-8)
    np.testing.assert_allclose(sol.orbital_occupancies[1], np.diag(dm1b), atol=1e-8)


@pytest.mark.parametrize("name", ["lih", "n2", "o2"])
@pytest.mark.parametrize(
    "options",
    [
        FulqrumSolverOptions(tol=1e-12, dense_max_dim=0),
        FulqrumSolverOptions(tol=1e-12, dense_max_dim=0, hamiltonian_format="matrix_free"),
        FulqrumSolverOptions(),
    ],
    ids=["scipy-csr", "scipy-matrix_free", "default"],
)
def test_random_subspace_matches_solve_fermion(name, options):
    from qiskit_addon_sqd.fermion import SCIState, solve_fermion

    h1, eri, norb, nelec, _ = _problem(name)
    rng = np.random.default_rng(7)
    full_a, full_b = _all_strs(norb, nelec[0]), _all_strs(norb, nelec[1])
    # Asymmetric, unsorted, with duplicates: the solver must sort and deduplicate.
    pick_a = rng.choice(full_a, size=max(3, (2 * len(full_a)) // 3), replace=False)
    pick_b = rng.choice(full_b, size=max(3, len(full_b) // 2), replace=False)
    pick_a = np.concatenate([pick_a, pick_a[:2]])
    sub_a, sub_b = np.unique(pick_a), np.unique(pick_b)

    e_ref, state_ref, occ_ref, _ = solve_fermion((sub_a, sub_b), h1, eri, open_shell=True)
    sol = solve_fulqrum((pick_a, pick_b), h1, eri, norb, nelec, options)

    assert sol.metadata["subspace_dim"] == len(sub_a) * len(sub_b)
    assert sol.energy == pytest.approx(e_ref, abs=1e-7)
    np.testing.assert_array_equal(sol.ci_strs_a, state_ref.ci_strs_a)
    np.testing.assert_array_equal(sol.ci_strs_b, state_ref.ci_strs_b)
    _assert_same_up_to_sign(sol.amplitudes, state_ref.amplitudes, atol=1e-4)
    np.testing.assert_allclose(sol.orbital_occupancies[0], occ_ref[0], atol=1e-6)
    np.testing.assert_allclose(sol.orbital_occupancies[1], occ_ref[1], atol=1e-6)

    # The amplitudes drop straight into SCIState and reproduce the occupancies.
    state = SCIState(sol.amplitudes, sol.ci_strs_a, sol.ci_strs_b, norb, nelec)
    occ_a, occ_b = state.orbital_occupancies()
    np.testing.assert_allclose(occ_a, sol.orbital_occupancies[0], atol=1e-10)
    np.testing.assert_allclose(occ_b, sol.orbital_occupancies[1], atol=1e-10)


def test_hartree_fock_only_subspace_gives_hf_energy():
    h1, eri, norb, nelec, ecore = _problem("n2")
    hf_a = np.array([(1 << nelec[0]) - 1])
    hf_b = np.array([(1 << nelec[1]) - 1])
    sol = solve_fulqrum((hf_a, hf_b), h1, eri, norb, nelec)

    occ_a = np.zeros(norb)
    occ_a[: nelec[0]] = 1
    e_hf = (
        2 * np.einsum("ii", h1[: nelec[0], : nelec[0]])
        + 2 * np.einsum("iijj", eri[: nelec[0], : nelec[0], : nelec[0], : nelec[0]])
        - np.einsum("ijji", eri[: nelec[0], : nelec[0], : nelec[0], : nelec[0]])
    )
    assert sol.metadata["subspace_dim"] == 1
    assert sol.metadata["solver_path"] == "dense"
    assert sol.energy == pytest.approx(e_hf, abs=1e-10)
    np.testing.assert_allclose(sol.orbital_occupancies[0], occ_a)
    np.testing.assert_allclose(sol.amplitudes, [[1.0]])


def test_return_amplitudes_false():
    h1, eri, norb, nelec, _ = _problem("h2")
    strs = _all_strs(norb, 1)
    sol = solve_fulqrum(
        (strs, strs), h1, eri, norb, nelec, FulqrumSolverOptions(return_amplitudes=False)
    )
    assert sol.amplitudes is None
    assert sol.orbital_occupancies[0].shape == (norb,)


@pytest.mark.parametrize("hamiltonian_format", ["csr", "matrix_free"])
@pytest.mark.parametrize("name", ["lih", "o2"])
def test_primme_eigensolver_if_available(name, hamiltonian_format):
    pytest.importorskip("primme")
    h1, eri, norb, nelec, _ = _problem(name)
    e_fci, civec = fci.direct_spin1.kernel(h1, eri, norb, nelec, conv_tol=1e-12)
    strs_a, strs_b = _all_strs(norb, nelec[0]), _all_strs(norb, nelec[1])
    options = FulqrumSolverOptions(
        eigensolver="primme", hamiltonian_format=hamiltonian_format, tol=1e-10, dense_max_dim=0
    )
    sol = solve_fulqrum((strs_a, strs_b), h1, eri, norb, nelec, options)
    assert sol.metadata["solver_path"] == f"primme-{hamiltonian_format}"
    assert sol.energy == pytest.approx(e_fci, abs=1e-8)
    _assert_same_up_to_sign(sol.amplitudes, civec, atol=1e-4)
