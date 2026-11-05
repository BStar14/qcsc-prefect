"""Configuration recovery loop with in-process solvers (no Prefect, no scheduler)."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

pytest.importorskip("qiskit_addon_sqd")

from qiskit_addon_sqd.counts import bit_array_to_arrays, generate_bit_array_uniform  # noqa: E402
from qiskit_addon_sqd.fermion import SCIResult, solve_fermion  # noqa: E402
from sqd_common import recovery  # noqa: E402
from sqd_common import subsample as shared_subsample  # noqa: E402

# Batches that the DICE workflow's original subsample() drew for the inputs of
# test_subsample_matches_sqd_dice_reference. Both workflows now share
# sqd_common.subsample, which must keep drawing the same subspaces.
SQD_DICE_REFERENCE_BATCHES = {
    False: [
        ([3, 9, 34, 20, 5, 17, 33, 10], [3, 9, 34, 20, 5, 17, 33, 10]),
        ([12, 18, 17, 20, 33, 48, 10, 24], [12, 18, 17, 20, 33, 48, 10, 24]),
        ([24, 10, 3, 48, 36, 20, 17, 33], [24, 10, 3, 48, 36, 20, 17, 33]),
    ],
    True: [
        ([13, 21, 44, 37, 26, 28, 14, 42, 11], [12, 18, 17, 20, 34, 48]),
        ([22, 38, 41, 7, 56, 50, 35, 26, 44], [3, 33, 12, 5, 24, 40]),
        ([19, 38, 22, 42, 49, 41, 44, 26, 52], [5, 36, 12, 18, 6, 33]),
    ],
}


def _lih():
    pyscf = pytest.importorskip("pyscf")
    from pyscf import ao2mo, fci, scf

    mol = pyscf.gto.M(atom="Li 0 0 0; H 0 0 1.6", basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run()
    norb = mf.mo_coeff.shape[1]
    h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    eri = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), norb)
    nelec = (2, 2)
    e_fci, _ = fci.direct_spin1.kernel(h1, eri, norb, nelec, conv_tol=1e-12)
    return dict(h1=h1, eri=eri, norb=norb, nelec=nelec, e_fci=e_fci, e_nuc=mol.energy_nuc())


def _hf_occupancies(norb, nelec):
    # Smoothed HF occupancies; exact 0/1 make recover_configurations divide by zero.
    occ_a = np.array([0.95] * nelec[0] + [0.05] * (norb - nelec[0]))
    occ_b = np.array([0.95] * nelec[1] + [0.05] * (norb - nelec[1]))
    return occ_a, occ_b


def _solve_fermion_fn(mol, *, open_shell=True):
    async def solve(ci_strings):
        energy, state, occ, _ = solve_fermion(
            ci_strings, mol["h1"], mol["eri"], open_shell=open_shell
        )
        return SCIResult(energy=energy, sci_state=state, orbital_occupancies=occ)

    return solve


def _raw_samples(norb, shots=2000, seed=11):
    return bit_array_to_arrays(generate_bit_array_uniform(shots, 2 * norb, rand_seed=seed))


def _run(raw, mol, solve, **overrides):
    kwargs = dict(
        solve=solve,
        initial_occupancies=_hf_occupancies(mol["norb"], mol["nelec"]),
        nelec=mol["nelec"],
        open_shell=False,
        subspace_dim=100,
        num_batches=2,
        max_iterations=2,
        rng=np.random.default_rng(24),
        nuclear_repulsion_energy=mol["e_nuc"],
    )
    kwargs.update(overrides)
    return asyncio.run(recovery.run_configuration_recovery(*raw, **kwargs))


def test_full_space_reaches_fci():
    mol = _lih()
    raw = _raw_samples(mol["norb"])
    outcome = _run(raw, mol, _solve_fermion_fn(mol), subspace_dim=10**6, max_iterations=1)
    assert len(outcome.records) == 1
    assert outcome.records[0]["dimension"] == 15 * 15
    assert outcome.best_result.energy == pytest.approx(mol["e_fci"], abs=1e-8)
    assert outcome.best_energy == pytest.approx(mol["e_fci"] + mol["e_nuc"], abs=1e-8)


def test_subsampled_rounds_are_variational_and_recorded():
    mol = _lih()
    raw = _raw_samples(mol["norb"])
    outcome = _run(raw, mol, _solve_fermion_fn(mol), subspace_dim=49, num_batches=3)
    assert len(outcome.records) == 6
    assert [r["round"] for r in outcome.records] == [0, 0, 0, 1, 1, 1]
    assert all(r["dimension"] == 49 for r in outcome.records)
    total_fci = mol["e_fci"] + mol["e_nuc"]
    assert all(r["energy"] >= total_fci - 1e-9 for r in outcome.records)
    assert outcome.best_energy == pytest.approx(min(r["energy"] for r in outcome.records))
    assert outcome.round_energies == pytest.approx(
        [min(r["energy"] for r in outcome.records if r["round"] == k) for k in (0, 1)]
    )


def test_best_occupancies_seed_the_next_round(monkeypatch):
    seen = []
    real_recover = recovery.recover_configurations

    def spy(bitstrings, probs, occupancies, n_alpha, n_beta, rand_seed=None):
        seen.append(occupancies)
        return real_recover(bitstrings, probs, occupancies, n_alpha, n_beta, rand_seed=rand_seed)

    monkeypatch.setattr(recovery, "recover_configurations", spy)
    norb, nelec = 5, (2, 2)
    calls = []

    async def fake_solve(ci_strings):
        idx = len(calls)
        calls.append(ci_strings)
        occ = np.full(norb, 0.1 * (idx + 1))
        return SCIResult(energy=-float(idx % 2), sci_state=None, orbital_occupancies=(occ, occ))

    mol = dict(norb=norb, nelec=nelec, e_nuc=1.5)
    raw = _raw_samples(norb, shots=500)
    outcome = _run(raw, mol, fake_solve, subspace_dim=9, num_batches=2, max_iterations=3)

    assert len(calls) == 6
    # Round 0 solves calls 0 (E=0) and 1 (E=-1): call 1 seeds round 1, and so on.
    np.testing.assert_allclose(seen[1][0], np.full(norb, 0.2))
    np.testing.assert_allclose(seen[2][0], np.full(norb, 0.4))
    assert outcome.round_energies == [0.5, 0.5, 0.5]
    assert outcome.best_energy == 0.5
    np.testing.assert_allclose(outcome.best_result.orbital_occupancies[0], np.full(norb, 0.2))


def _counting_solve(norb, energies):
    calls = []

    async def solve(ci_strings):
        calls.append(ci_strings)
        occ = np.full(norb, 0.1 * len(calls))
        return SCIResult(
            energy=energies[len(calls) - 1], sci_state=None, orbital_occupancies=(occ, occ)
        )

    return solve, calls


def test_resume_runs_only_the_remaining_rounds_and_reports_them():
    norb = 5
    mol = dict(norb=norb, nelec=(2, 2), e_nuc=1.0)
    solve, calls = _counting_solve(norb, [-2.0, -3.0, -1.0, -1.5])
    ends = []
    outcome = _run(
        _raw_samples(norb, shots=500),
        mol,
        solve,
        subspace_dim=9,
        num_batches=2,
        max_iterations=4,
        start_round=2,
        best_energy=-10.0,
        on_round_end=lambda *args: ends.append(args),
    )
    assert len(calls) == 4
    assert [r["round"] for r in outcome.records] == [2, 2, 3, 3]
    assert outcome.round_energies == [-2.0, -0.5]
    # The stored best is a floor; no round of this call beats it.
    assert outcome.best_energy == -10.0 and outcome.best_result is None
    assert [(n, best) for n, _, best in ends] == [(3, -10.0), (4, -10.0)]
    # Call 2 (E=-3) is the best of round 2, so its occupancies seed round 3.
    np.testing.assert_allclose(ends[0][1][0], np.full(norb, 0.2))


def test_resume_reports_a_better_round_than_the_stored_best():
    norb = 5
    mol = dict(norb=norb, nelec=(2, 2), e_nuc=0.0)
    solve, _ = _counting_solve(norb, [-4.0, -1.0])
    outcome = _run(
        _raw_samples(norb, shots=500),
        mol,
        solve,
        subspace_dim=9,
        max_iterations=3,
        start_round=2,
        best_energy=-3.0,
    )
    assert outcome.best_energy == -4.0 and outcome.best_result.energy == -4.0


def test_resume_of_a_finished_run_returns_the_stored_best():
    async def never(_):
        raise AssertionError("solver must not be called")

    mol = dict(norb=4, nelec=(1, 1), e_nuc=0.0)
    outcome = _run(
        _raw_samples(4, shots=50), mol, never, max_iterations=3, start_round=3, best_energy=-7.0
    )
    assert outcome.best_energy == -7.0 and outcome.records == []


def test_no_valid_bitstrings_raises(monkeypatch):
    monkeypatch.setattr(
        recovery,
        "postselect",
        lambda bitstring_matrix, probabilities, **_: (bitstring_matrix[:0], probabilities[:0]),
    )

    async def never(_):
        raise AssertionError("solver must not be called")

    mol = dict(norb=4, nelec=(1, 1), e_nuc=0.0)
    with pytest.raises(RuntimeError, match="did not contain any valid bitstrings"):
        _run(_raw_samples(4, shots=50), mol, never)


@pytest.mark.parametrize("open_shell", [False, True])
def test_subsample_matches_sqd_dice_reference(open_shell):
    """The shared subsample draws the subspaces of the original DICE workflow."""

    raw_bs, raw_probs = _raw_samples(6, shots=3000, seed=5)
    nelec = (3, 2) if open_shell else (2, 2)
    bs, probs = shared_subsample.postselect(
        raw_bs, raw_probs, hamming_right=nelec[0], hamming_left=nelec[1]
    )
    ours = list(shared_subsample.subsample(bs, probs, 64, 3, np.random.default_rng(3), open_shell))
    assert [([int(x) for x in a], [int(x) for x in b]) for a, b in ours] == (
        SQD_DICE_REFERENCE_BATCHES[open_shell]
    )
    assert all(a.dtype == np.int64 and b.dtype == np.int64 for a, b in ours)


def test_fulqrum_loop_matches_solve_fermion_loop():
    """Same seeds and samples: fulqrum and solve_fermion give the same SQD run."""

    pytest.importorskip("fulqrum")
    from qcsc_prefect_fulqrum import FulqrumSolverOptions, solve_fulqrum

    mol = _lih()
    options = FulqrumSolverOptions(tol=1e-12, dense_max_dim=0, return_amplitudes=False)

    async def fulqrum_solve(ci_strings):
        sol = solve_fulqrum(
            ci_strings, mol["h1"], mol["eri"], mol["norb"], mol["nelec"], options=options
        )
        return SCIResult(
            energy=sol.energy, sci_state=None, orbital_occupancies=sol.orbital_occupancies
        )

    raw = _raw_samples(mol["norb"])
    params = dict(subspace_dim=64, num_batches=2, max_iterations=3)
    got = _run(raw, mol, fulqrum_solve, **params)
    ref = _run(raw, mol, _solve_fermion_fn(mol), **params)

    assert [r["dimension"] for r in got.records] == [r["dimension"] for r in ref.records]
    np.testing.assert_allclose(
        [r["energy"] for r in got.records], [r["energy"] for r in ref.records], atol=1e-8
    )
    assert got.best_energy == pytest.approx(ref.best_energy, abs=1e-8)
