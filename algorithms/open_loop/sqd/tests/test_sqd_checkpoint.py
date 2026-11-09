"""The recovery checkpoint file and its resume key (no Prefect, no solver)."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pytest
from sqd_common import checkpoint

NORB = 4


def _elec_props(seed=0):
    rng = np.random.default_rng(seed)
    h1 = rng.normal(size=(NORB, NORB))
    h2 = rng.normal(size=(NORB,) * 4)
    return SimpleNamespace(
        num_orbitals=NORB,
        num_electrons=(2, 2),
        one_body_tensor=h1 + h1.T,
        two_body_tensor=h2,
        nuclear_repulsion_energy=1.25,
    )


def _sqd(subspace_dim=16, num_batches=2):
    return SimpleNamespace(subspace_dim=subspace_dim, num_batches=num_batches)


def _samples(seed=1):
    return np.random.default_rng(seed).integers(0, 2, size=(50, 2 * NORB)).astype(bool)


def test_round_trip_writes_the_uhf_fields_atomically(tmp_path):
    path = tmp_path / checkpoint.CKPT_NAME
    occ = (np.linspace(0, 1, NORB), np.linspace(1, 0, NORB))
    checkpoint.save_checkpoint(path, occ, next_step=32, best_energy=-108.9)

    assert sorted(p.name for p in tmp_path.iterdir()) == [checkpoint.CKPT_NAME]
    with np.load(path) as raw:
        assert set(raw.files) == {
            "occ_a",
            "occ_b",
            "next_step",
            "best_energy",
            "carryover_a",
            "carryover_b",
        }
        assert raw["carryover_a"].shape == (0,) and raw["carryover_a"].dtype == np.int64
    state = checkpoint.load_checkpoint(path)
    assert state.next_step == 32 and state.best_energy == -108.9
    np.testing.assert_array_equal(state.occupancies[0], occ[0])
    np.testing.assert_array_equal(state.occupancies[1], occ[1])


def test_a_failed_save_is_logged_not_raised(tmp_path, caplog):
    missing = tmp_path / "no" / "such" / checkpoint.CKPT_NAME
    with caplog.at_level(logging.WARNING):
        checkpoint.save_checkpoint(missing, (np.ones(2), np.ones(2)), 1, 0.0)
    assert "checkpoint save failed" in caplog.text


def _key(samples=None, props=None, sqd=None, solver="fulqrum_solver_job"):
    return checkpoint.resume_key(
        _samples() if samples is None else samples,
        props or _elec_props(),
        sqd or _sqd(),
        solver=solver,
    )


def test_key_matches_the_same_hamiltonian_in_another_orbital_basis():
    props = _elec_props()
    key = _key(props=props)
    # Orbital rotation and sign flips change the tensors but not the Hamiltonian.
    q, _ = np.linalg.qr(np.random.default_rng(7).normal(size=(NORB, NORB)))
    rotated = SimpleNamespace(**vars(props))
    rotated.one_body_tensor = q.T @ props.one_body_tensor @ q
    rotated.two_body_tensor = np.einsum("pqrs,pi,qj,rk,sl->ijkl", props.two_body_tensor, q, q, q, q)
    assert checkpoint.key_mismatch(key, _key(props=rotated)) == []


@pytest.mark.parametrize(
    "change, expected",
    [
        (dict(samples=_samples(2)), ["samples"]),
        (dict(props=_elec_props(3)), ["Hamiltonian"]),
        (dict(sqd=_sqd(subspace_dim=25)), ["subspace_dim"]),
        (dict(sqd=_sqd(num_batches=3)), ["num_batches"]),
        (dict(solver="sbd_solver_job"), ["solver"]),
    ],
)
def test_key_rejects_other_samples_hamiltonian_subspace_and_solver(change, expected):
    assert checkpoint.key_mismatch(_key(), _key(**change)) == expected


def test_key_without_a_solver_is_refused():
    # source.json written before the solver joined the resume key.
    old = _key()
    del old["solver"]
    assert checkpoint.key_mismatch(old, _key()) == ["solver"]


def test_source_round_trip(tmp_path):
    assert checkpoint.read_source(tmp_path) is None
    key = _key()
    checkpoint.write_source(tmp_path, key, resumed_from=None)
    source = checkpoint.read_source(tmp_path)
    assert checkpoint.key_mismatch(source, key) == [] and source["resumed_from"] is None


def test_carryover_round_trip_keeps_int64_strings(tmp_path):
    path = tmp_path / checkpoint.CKPT_NAME
    carry = (np.array([12, 3, 5], dtype=np.int64), np.array([9], dtype=np.int64))
    occ = (np.ones(NORB), np.ones(NORB))
    checkpoint.save_checkpoint(path, occ, 2, -1.0, carryover=carry)
    with np.load(path, allow_pickle=False) as raw:
        assert raw["carryover_a"].dtype == np.int64
    state = checkpoint.load_checkpoint(path)
    assert state.carryover[0].tolist() == [12, 3, 5] and state.carryover[1].tolist() == [9]
    assert state.carryover[0].dtype == np.int64


def test_carryover_of_64_or_more_orbitals_is_stored_as_decimal_strings(tmp_path):
    norb = 70
    path = tmp_path / checkpoint.CKPT_NAME
    big = [(1 << 69) | 1, (1 << 64) | (1 << 3)]
    carry = (np.array(big, dtype=object), np.array(big[:1], dtype=object))
    checkpoint.save_checkpoint(path, (np.ones(norb), np.ones(norb)), 1, -2.0, carryover=carry)
    with np.load(path, allow_pickle=False) as raw:
        assert raw["carryover_a"].dtype.kind == "U"
    state = checkpoint.load_checkpoint(path)
    assert state.carryover[0].dtype == object and state.carryover[0].tolist() == big
    assert state.carryover[1].tolist() == big[:1]


@pytest.mark.parametrize("old_key", [True, False], ids=["empty-carryover-key", "no-key"])
def test_checkpoint_without_per_spin_carryover_loads_with_none(tmp_path, old_key):
    path = tmp_path / checkpoint.CKPT_NAME
    arrays = dict(
        occ_a=np.ones(NORB),
        occ_b=np.zeros(NORB),
        next_step=np.int64(3),
        best_energy=np.float64(-1.5),
    )
    if old_key:
        arrays["carryover"] = np.empty((0, 0), dtype=bool)
    np.savez(path, **arrays)
    state = checkpoint.load_checkpoint(path)
    assert state.next_step == 3 and state.best_energy == -1.5
    assert len(state.carryover[0]) == 0 and len(state.carryover[1]) == 0


def test_carryover_fraction_is_not_in_the_resume_key():
    sqd = SimpleNamespace(subspace_dim=16, num_batches=2, carryover_fraction=0.5)
    other = SimpleNamespace(subspace_dim=16, num_batches=2, carryover_fraction=0.0)
    key = checkpoint.resume_key(_samples(), _elec_props(), sqd, solver="fulqrum_solver_job")
    assert "carryover_fraction" not in key
    assert checkpoint.key_mismatch(key, _key(sqd=other)) == []
