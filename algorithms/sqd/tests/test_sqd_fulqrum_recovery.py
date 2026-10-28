"""Configuration recovery loop with in-process solvers (no Prefect, no scheduler)."""

from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("qiskit_addon_sqd")

from qiskit_addon_sqd.counts import bit_array_to_arrays, generate_bit_array_uniform  # noqa: E402
from qiskit_addon_sqd.fermion import SCIResult, solve_fermion  # noqa: E402
from sqd_fulqrum import recovery  # noqa: E402
from sqd_fulqrum import subsample as fulqrum_subsample  # noqa: E402

SQD_DICE_SUBSAMPLE = Path(__file__).resolve().parents[1] / "sqd_dice" / "subsample.py"


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


@pytest.mark.skipif(not SQD_DICE_SUBSAMPLE.exists(), reason="sqd_dice/subsample.py not found")
@pytest.mark.parametrize("open_shell", [False, True])
def test_subsample_matches_sqd_dice_reference(open_shell):
    """The port draws the same subspaces as sqd_dice for the same RNG state."""

    spec = importlib.util.spec_from_file_location("_sqd_dice_subsample_ref", SQD_DICE_SUBSAMPLE)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)

    raw_bs, raw_probs = _raw_samples(6, shots=3000, seed=5)
    nelec = (3, 2) if open_shell else (2, 2)
    bs, probs = fulqrum_subsample.postselect(
        raw_bs, raw_probs, hamming_right=nelec[0], hamming_left=nelec[1]
    )
    ours = list(fulqrum_subsample.subsample(bs, probs, 64, 3, np.random.default_rng(3), open_shell))
    ref = list(reference.subsample(bs, probs, 64, 3, np.random.default_rng(3), open_shell))
    assert len(ours) == len(ref) == 3
    for (a, b), (ra, rb) in zip(ours, ref, strict=True):
        np.testing.assert_array_equal(a, ra)
        np.testing.assert_array_equal(b, rb)


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
