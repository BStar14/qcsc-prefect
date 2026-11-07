"""sqd_loop with carryover: the solver block's fraction and the checkpoint (Prefect mocked)."""

from __future__ import annotations

import asyncio
import importlib
import logging
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("qiskit_addon_sqd")
pytest.importorskip("prefect")

from pydantic import BaseModel, ConfigDict, Field  # noqa: E402
from qiskit_addon_sqd.fermion import SCIResult  # noqa: E402
from sqd_common import checkpoint, flow_params  # noqa: E402


def _import_postprocess():
    """Import postprocess; without pydantic_numpy, stub the type-only workflow utility."""
    try:
        importlib.import_module("qcsc_workflow_utility")
    except ImportError:
        with pytest.MonkeyPatch.context() as mp:
            stub = types.ModuleType("qcsc_workflow_utility")
            stub.ElectronicProperties = object
            mp.setitem(sys.modules, "qcsc_workflow_utility", stub)
            return importlib.import_module("sqd_common.postprocess")
    return importlib.import_module("sqd_common.postprocess")


postprocess = _import_postprocess()

LOGGER = logging.getLogger("sqd_common.test")
NORB, NELEC = 6, (2, 2)


def _elec_props():
    rng = np.random.default_rng(0)
    h1 = rng.normal(size=(NORB, NORB))
    occ = np.array([0.95] * NELEC[0] + [0.05] * (NORB - NELEC[0]))
    return SimpleNamespace(
        one_body_tensor=h1 + h1.T,
        two_body_tensor=rng.normal(size=(NORB,) * 4),
        initial_occupancy=(occ, occ.copy()),
        nuclear_repulsion_energy=0.5,
        num_orbitals=NORB,
        num_electrons=NELEC,
        open_shell=False,
        spin_sq=0.0,
    )


class _Block(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    root_dir: str
    calls: list = Field(default_factory=list)


class _FulqrumLikeBlock(_Block):
    """A solver block with the fulqrum ``carryover_fraction`` field."""

    _block_type_slug = "fulqrum_solver_job"

    carryover_fraction: float = Field(default=0.0, ge=0.0, le=1.0)

    async def run(self, ci_strings, one_body_tensor, two_body_tensor, norb, nelec, spin_sq=None):
        self.calls.append((self.carryover_fraction, ci_strings))
        occ = np.full(norb, 0.5)
        carryover = None
        if self.carryover_fraction > 0:
            # The two largest strings of the subspace, as a stand-in for the heaviest.
            heavy = np.sort(np.asarray(ci_strings[0]))[::-1][:2]
            carryover = (heavy, heavy.copy())
        return SCIResult(
            energy=-float(len(self.calls)),
            sci_state=None,
            orbital_occupancies=(occ, occ),
            carryover=carryover,
        )


class _SBDLikeBlock(_Block):
    """A solver block with the SBD ``carryover_ratio`` field, which must be above 0."""

    _block_type_slug = "sbd_solver_job"

    carryover_ratio: float = Field(default=0.5, gt=0.0, le=1.0)

    async def run(self, ci_strings, one_body_tensor, two_body_tensor, norb, nelec, spin_sq=None):
        self.calls.append((self.carryover_ratio, ci_strings))
        occ = np.full(norb, 0.5)
        rows = np.zeros((1, norb), dtype=bool)
        rows[0, -2:] = True  # orbitals 0 and 1
        return SimpleNamespace(
            energy=-float(len(self.calls)),
            orbital_occupancies=(occ, occ),
            carryover_bitstrings=rows,
        )


@pytest.fixture
def run_loop(monkeypatch, tmp_path):
    artifacts = []

    async def fake_artifact(**kwargs):
        artifacts.append(kwargs)

    monkeypatch.setattr(postprocess, "get_run_logger", lambda: LOGGER)
    monkeypatch.setattr(postprocess, "create_table_artifact", fake_artifact)
    bool_array = np.random.default_rng(3).integers(0, 2, size=(800, 2 * NORB)).astype(bool)

    def run(run_id, solver, **sqd):
        monkeypatch.setattr(postprocess, "flow_run", SimpleNamespace(get_id=lambda: run_id))
        artifacts.clear()
        params = flow_params.SQDParameters(**{"subspace_dim": 16, "num_batches": 2, **sqd})
        energy = asyncio.run(
            postprocess.sqd_loop.fn(bool_array, _elec_props(), params, solver, "recovery")
        )
        return energy, {a["key"]: a["table"] for a in artifacts}

    run.root = tmp_path / postprocess.CHECKPOINT_SUBDIR
    run.tmp = tmp_path
    return run


def test_sqd_loop_sets_the_fraction_on_a_copy_of_the_block(run_loop):
    block = _FulqrumLikeBlock(root_dir=str(run_loop.tmp))
    _, artifacts = run_loop("run-a", block, max_iterations=2, carryover_fraction=0.4)

    # model_copy is shallow, so the copy records its calls in the same list.
    assert block.carryover_fraction == 0.0
    assert [fraction for fraction, _ in block.calls] == [0.4] * 4
    rows = artifacts["recovery"]
    assert [(r["num_carryover_a"], r["num_carryover_b"]) for r in rows] == [
        (0, 0),
        (0, 0),
        (2, 2),
        (2, 2),
    ]
    # Round 1 starts with the carryover of call 2, the best (lowest) of round 0.
    heavy = np.sort(block.calls[1][1][0])[::-1][:2]
    for _, (a, _) in block.calls[2:]:
        np.testing.assert_array_equal(a[:2], heavy)
    assert all(r["dimension"] == 16 for r in rows)


def test_checkpoint_saves_and_a_resume_reuses_the_carryover(run_loop):
    run_loop("run-a", _FulqrumLikeBlock(root_dir=str(run_loop.tmp)), max_iterations=1)
    saved = checkpoint.load_checkpoint(run_loop.root / "run-a" / checkpoint.CKPT_NAME)
    assert len(saved.carryover[0]) == 2 and saved.carryover[0].dtype == np.int64

    block = _FulqrumLikeBlock(root_dir=str(run_loop.tmp))
    _, artifacts = run_loop(
        "run-b",
        block,
        max_iterations=2,
        resume_checkpoint=str(run_loop.root / "run-a"),
    )
    rows = artifacts["recovery"]
    assert [r["round"] for r in rows] == [1, 1]
    assert all(r["num_carryover_a"] == 2 for r in rows)


def test_a_new_fraction_resumes_the_same_checkpoint(run_loop):
    run_loop("run-a", _FulqrumLikeBlock(root_dir=str(run_loop.tmp)), max_iterations=1)
    _, artifacts = run_loop(
        "run-b",
        _FulqrumLikeBlock(root_dir=str(run_loop.tmp)),
        max_iterations=2,
        carryover_fraction=0.0,
        resume_checkpoint=str(run_loop.root / "run-a"),
    )
    rows = artifacts["recovery"]
    assert [r["round"] for r in rows] == [1, 1]
    # Fraction 0 uses no carryover, not even the checkpointed one.
    assert all(r["num_carryover_a"] == 0 for r in rows)
    saved = checkpoint.load_checkpoint(run_loop.root / "run-b" / checkpoint.CKPT_NAME)
    assert len(saved.carryover[0]) == 0


def test_sbd_block_gets_the_ratio_only_above_zero(run_loop):
    block = _SBDLikeBlock(root_dir=str(run_loop.tmp))
    _, artifacts = run_loop("run-a", block, max_iterations=2, carryover_fraction=0.25)
    rows = artifacts["recovery"]
    assert [(r["num_carryover_a"], r["num_carryover_b"]) for r in rows][2:] == [(1, 1), (1, 1)]
    assert block.carryover_ratio == 0.5
    assert [ratio for ratio, _ in block.calls] == [0.25] * 4
    # The SBD rows decode to orbitals 0 and 1, which lead the next round.
    assert all(a[0] == 0b11 for _, (a, _) in block.calls[2:])

    block = _SBDLikeBlock(root_dir=str(run_loop.tmp))
    _, artifacts = run_loop("run-b", block, max_iterations=2, carryover_fraction=0.0)
    used = [ratio for ratio, _ in block.calls]
    # Fraction 0 leaves the block as it is and ignores the strings it returns.
    assert used == [0.5] * 4
    assert all(r["num_carryover_a"] == 0 for r in artifacts["recovery"])
