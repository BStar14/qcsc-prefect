"""Carryover in the configuration recovery loop (no Prefect, no scheduler)."""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("qiskit_addon_sqd")

from pydantic import BaseModel, Field  # noqa: E402
from qiskit_addon_sqd.counts import bit_array_to_arrays, generate_bit_array_uniform  # noqa: E402
from qiskit_addon_sqd.fermion import SCIResult, SCIState  # noqa: E402
from sqd_common import recovery  # noqa: E402

NORB, NELEC = 6, (2, 2)


def _ci(*orbitals):
    return sum(1 << o for o in orbitals)


def _raw_samples(shots=500, seed=11):
    return bit_array_to_arrays(generate_bit_array_uniform(shots, 2 * NORB, rand_seed=seed))


def _occ():
    occ = np.array([0.95] * NELEC[0] + [0.05] * (NORB - NELEC[0]))
    return occ, occ.copy()


def _carry(*strings):
    values = np.array([_ci(*s) for s in strings], dtype=np.int64)
    return values, values.copy()


class _FakeSolver:
    """Returns energies and carryover per call, and records the subspaces it solved."""

    def __init__(self, energies, carryovers):
        self.energies, self.carryovers, self.calls = energies, carryovers, []

    async def __call__(self, ci_strings):
        idx = len(self.calls)
        self.calls.append(ci_strings)
        return SCIResult(
            energy=self.energies[idx],
            sci_state=None,
            orbital_occupancies=_occ(),
            carryover=self.carryovers[idx],
        )


def _run(solve, **overrides):
    kwargs = dict(
        solve=solve,
        initial_occupancies=_occ(),
        nelec=NELEC,
        open_shell=False,
        subspace_dim=16,
        num_batches=2,
        max_iterations=3,
        rng=np.random.default_rng(24),
        carryover_fraction=0.5,
    )
    kwargs.update(overrides)
    return asyncio.run(recovery.run_configuration_recovery(*_raw_samples(), **kwargs))


def test_best_result_carryover_leads_every_subspace_of_the_next_round():
    co_round0 = _carry((0, 1), (0, 5))  # from call 1, the best of round 0
    co_round1 = _carry((2, 3))  # from call 2, the best of round 1
    solver = _FakeSolver(
        energies=[-1.0, -2.0, -3.0, -2.5, -1.0, -0.5],
        carryovers=[_carry((4, 5)), co_round0, co_round1, _carry((1, 2)), None, None],
    )
    ends = []
    outcome = _run(solver, on_round_end=lambda *args: ends.append(args))

    assert len(solver.calls) == 6
    for a, b in solver.calls[2:4]:
        assert a[:2].tolist() == co_round0[0].tolist() and len(a) == 4
        np.testing.assert_array_equal(a, b)
    for a, _ in solver.calls[4:]:
        assert a[0] == _ci(2, 3) and len(a) == 4
    assert [(r["num_carryover_a"], r["num_carryover_b"]) for r in outcome.records] == [
        (0, 0),
        (0, 0),
        (2, 2),
        (2, 2),
        (1, 1),
        (1, 1),
    ]
    # on_round_end gets the carryover for the next round as its 4th argument.
    np.testing.assert_array_equal(ends[0][3][0], co_round0[0])
    np.testing.assert_array_equal(ends[1][3][0], co_round1[0])
    assert len(ends[2][3][0]) == 0 and len(outcome.final_carryover[0]) == 0
    assert all(r["dimension"] == 16 for r in outcome.records)


def test_batch_transform_gets_the_round_carryover():
    seen = []

    def transform(batches, rng, carryover):
        seen.append(carryover)
        return batches

    co_round0 = _carry((0, 1), (0, 5))
    solver = _FakeSolver([-1.0, -2.0, -3.0, -2.5], [None, co_round0, None, None])
    _run(solver, max_iterations=2, batch_transform=transform)
    assert len(seen[0][0]) == 0
    np.testing.assert_array_equal(seen[1][0], co_round0[0])


def test_initial_carryover_seeds_the_first_round():
    solver = _FakeSolver([-1.0, -2.0], [None, None])
    outcome = _run(
        solver, max_iterations=1, initial_carryover=_carry((1, 4), (2, 4)), start_round=0
    )
    for a, _ in solver.calls:
        assert a[:2].tolist() == [_ci(1, 4), _ci(2, 4)]
    assert outcome.records[0]["num_carryover_a"] == 2


def test_fraction_zero_uses_no_carryover_at_all():
    carry = _carry((0, 1), (0, 5))
    with_carry = _FakeSolver([-1.0, -2.0, -3.0, -2.5], [carry] * 4)
    plain = _FakeSolver([-1.0, -2.0, -3.0, -2.5], [None] * 4)
    ends = []
    got = _run(
        with_carry,
        max_iterations=2,
        carryover_fraction=0.0,
        initial_carryover=carry,
        on_round_end=lambda *args: ends.append(args),
    )
    ref = _run(plain, max_iterations=2, carryover_fraction=0.0)
    for (a, b), (ra, rb) in zip(with_carry.calls, plain.calls, strict=True):
        np.testing.assert_array_equal(a, ra)
        np.testing.assert_array_equal(b, rb)
    assert all(r["num_carryover_a"] == 0 for r in got.records)
    assert all(len(end[3][0]) == 0 for end in ends)
    assert got.records == ref.records


def test_a_solver_without_carryover_warns_once(caplog):
    solver = _FakeSolver([-1.0] * 6, [None] * 6)
    with caplog.at_level(logging.WARNING):
        outcome = _run(solver, logger=logging.getLogger("sqd.test"))
    assert caplog.text.count("returned no carryover") == 1
    assert all(r["num_carryover_a"] == 0 for r in outcome.records)


def test_fulqrum_style_open_shell_carryover_is_per_spin():
    nelec = (3, 1)
    carry = (np.array([_ci(0, 1, 2), _ci(0, 1, 5)]), np.array([_ci(4)]))
    solver = _FakeSolver([-1.0, -2.0, -1.5, -1.0], [carry] * 4)
    _run(
        solver,
        nelec=nelec,
        open_shell=True,
        subspace_dim=12,
        max_iterations=2,
        carryover_fraction=1.0,
    )
    for a, b in solver.calls[2:]:
        assert a[:2].tolist() == carry[0].tolist()
        assert b[0] == _ci(4)
        assert len(a) * len(b) <= 12


# ------------------------------------------------------------------ carryover_of


def test_carryover_of_reads_scir_result_carryover_first():
    carry = (np.array([7, 11]), np.array([13]))
    rows = np.ones((1, NORB), dtype=bool)
    result = SimpleNamespace(carryover=carry, carryover_bitstrings=rows, sci_state=None)
    got = recovery.carryover_of(result, 0.5, NORB)
    assert got[0].tolist() == [7, 11] and got[1].tolist() == [13]


def _pack_sbd_carryover(dets_occ, norb):
    """Bytes as the SBD solver writes carryover.bin: orbital o is bit norb-1-o, big-endian."""
    out = bytearray()
    for occ in dets_occ:
        buf = bytearray((norb + 7) // 8)
        for j in range(norb):
            if norb - 1 - j in occ:
                buf[j // 8] |= 1 << (7 - j % 8)
        out += buf
    return bytes(out)


def _decode_sbd_carryover(data, norb):
    """The decoding of ``qcsc_prefect_sbd.solver_job._read_files``."""
    raw = np.frombuffer(data, dtype=np.uint8)
    bits = np.unpackbits(raw, bitorder="big").reshape(-1, (norb + 7) // 8 * 8)[:, :norb]
    return bits.astype(bool)


@pytest.mark.parametrize("norb", [6, 20, 70])
def test_carryover_of_decodes_sbd_bool_rows_for_both_spins(norb):
    dets = [[0, 1], [1, norb - 1], [2, 3]]
    rows = _decode_sbd_carryover(_pack_sbd_carryover(dets, norb), norb)
    result = SimpleNamespace(carryover_bitstrings=rows)
    got_a, got_b = recovery.carryover_of(result, 0.5, norb)
    assert [int(x) for x in got_a] == [_ci(*occ) for occ in dets]
    np.testing.assert_array_equal(got_a, got_b)
    assert got_a.dtype == (np.int64 if norb < 64 else object)


def test_carryover_of_reads_the_real_sbd_result():
    sbd = pytest.importorskip("qcsc_prefect_sbd.solver_job")
    rows = _decode_sbd_carryover(_pack_sbd_carryover([[0, 2], [1, 3]], NORB), NORB)
    result = sbd.SBDResult(energy=-1.0, orbital_occupancies=_occ(), carryover_bitstrings=rows)
    got_a, got_b = recovery.carryover_of(result, 0.3, NORB)
    assert got_a.tolist() == [_ci(0, 2), _ci(1, 3)] and got_b.tolist() == got_a.tolist()
    assert len(recovery.carryover_of(result, 0.0, NORB)[0]) == 0


def test_carryover_of_selects_the_heaviest_strings_of_a_dice_sci_state():
    strs_a = np.array([_ci(0, 1), _ci(0, 2), _ci(1, 2), _ci(0, 3)])
    strs_b = np.array([_ci(0, 1), _ci(1, 3)])
    amplitudes = np.array([[0.1, 0.0], [0.6, 0.2], [0.05, 0.7], [0.3, 0.1]])
    state = SCIState(
        amplitudes=amplitudes, ci_strs_a=strs_a, ci_strs_b=strs_b, norb=NORB, nelec=NELEC
    )
    result = SCIResult(energy=-1.0, sci_state=state, orbital_occupancies=_occ())
    got_a, got_b = recovery.carryover_of(result, 0.5, NORB)
    # Alpha weights 0.01, 0.40, 0.4925, 0.10; beta weights 0.4625, 0.54.
    assert got_a.tolist() == [_ci(1, 2), _ci(0, 2)]
    assert got_b.tolist() == [_ci(1, 3)]


@pytest.mark.parametrize(
    "result",
    [
        SimpleNamespace(carryover=(np.array([3]), np.array([3]))),
        SimpleNamespace(carryover_bitstrings=np.ones((2, NORB), dtype=bool)),
        SCIResult(
            energy=0.0,
            sci_state=SCIState(
                amplitudes=np.ones((1, 1)),
                ci_strs_a=np.array([3]),
                ci_strs_b=np.array([3]),
                norb=NORB,
                nelec=NELEC,
            ),
            orbital_occupancies=_occ(),
        ),
    ],
    ids=["carryover", "sbd", "sci_state"],
)
def test_carryover_of_is_empty_with_fraction_zero(result):
    got_a, got_b = recovery.carryover_of(result, 0.0, NORB)
    assert len(got_a) == 0 and len(got_b) == 0


def test_carryover_of_without_a_source_is_empty():
    result = SCIResult(energy=0.0, sci_state=None, orbital_occupancies=_occ())
    got_a, got_b = recovery.carryover_of(result, 0.5, NORB)
    assert len(got_a) == 0 and len(got_b) == 0


# ------------------------------------------------------------------ with_carryover_fraction


class _FulqrumLike(BaseModel):
    carryover_fraction: float = Field(default=0.0, ge=0.0, le=1.0)


class _SBDLike(BaseModel):
    carryover_ratio: float = Field(default=0.5, gt=0.0, le=1.0)


class _DiceLike(BaseModel):
    return_sci_state: bool = True


def test_with_carryover_fraction_sets_the_fulqrum_field_on_a_copy():
    block = _FulqrumLike()
    copy = recovery.with_carryover_fraction(block, 0.25)
    assert copy.carryover_fraction == 0.25 and block.carryover_fraction == 0.0
    assert recovery.with_carryover_fraction(block, 0.0) is block


def test_with_carryover_fraction_sets_the_sbd_ratio_only_above_zero():
    block = _SBDLike()
    assert recovery.with_carryover_fraction(block, 0.2).carryover_ratio == 0.2
    assert recovery.with_carryover_fraction(block, 0.0) is block


def test_with_carryover_fraction_leaves_other_blocks_alone():
    block = _DiceLike()
    assert recovery.with_carryover_fraction(block, 0.5) is block
    plain = object()
    assert recovery.with_carryover_fraction(plain, 0.5) is plain


def test_fraction_one_warns_that_no_sample_may_enter(caplog):
    solver = _FakeSolver([-1.0, -2.0], [None, None])
    with caplog.at_level(logging.WARNING):
        _run(solver, max_iterations=1, carryover_fraction=1.0)
    assert "carryover_fraction=1" in caplog.text


def test_the_loop_passes_the_fraction_to_subsampling():
    """A batch of 4 with fraction 0.5 keeps 2 of the 3 carryover strings."""
    co = _carry((0, 1), (0, 5), (1, 5))
    solver = _FakeSolver([-1.0, -2.0, -3.0, -2.5], [co, co, None, None])
    _run(solver, max_iterations=2)
    for a, _ in solver.calls[2:]:
        # The third string is not forced; it may still be drawn as a sample.
        assert a[:2].tolist() == co[0][:2].tolist() and len(np.unique(a)) == 4


def test_a_cancelled_round_cancels_its_solver_jobs():
    cancelled = []

    async def solve(ci_strings):
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.append(len(ci_strings[0]))
            raise

    async def main():
        task = asyncio.ensure_future(
            recovery._solve_round(solve, [(np.array([1]), np.array([1]))] * 3, 0)
        )
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(main())
    assert len(cancelled) == 3


def test_sbd_rows_of_another_width_are_rejected():
    with pytest.raises(ValueError, match=r"shape \(n, 6\)"):
        recovery._bool_rows_to_ints(np.zeros((2, 5), dtype=bool), 6)
    assert recovery._bool_rows_to_ints(np.zeros((0, 6), dtype=bool), 6).dtype == np.int64


def test_each_iteration_logs_its_best_and_the_overall_best(caplog):
    # Iteration bests -1.0, -3.0, -2.0 (two batches each): the last is not a new best.
    solver = _FakeSolver([-1.0, -0.5, -3.0, -2.5, -2.0, -1.5], [None] * 6)
    with caplog.at_level(logging.INFO):
        outcome = _run(solver, logger=logging.getLogger("sqd.test"))
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("SQD iteration")]
    assert lines == [
        "SQD iteration 1/3: iteration best -1.0 Ha; new overall best -1.0 Ha.",
        "SQD iteration 2/3: iteration best -3.0 Ha; new overall best -3.0 Ha.",
        "SQD iteration 3/3: iteration best -2.0 Ha; overall best -3.0 Ha.",
    ]
    assert outcome.best_energy == -3.0


def test_a_resumed_loop_reports_the_overall_best_of_the_earlier_run(caplog):
    solver = _FakeSolver([-1.0, -0.5], [None] * 2)
    with caplog.at_level(logging.INFO):
        _run(solver, logger=logging.getLogger("sqd.test"), start_round=2, best_energy=-4.0)
    assert "SQD iteration 3/3: iteration best -1.0 Ha; overall best -4.0 Ha." in caplog.text
