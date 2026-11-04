"""The sqd_dice flow with every Prefect-server interaction mocked.

The DICE binary is not needed: the solver block is replaced by an in-process
``solve_fermion`` stand-in, as in the fulqrum flow test.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

for _mod in (
    "prefect",
    "ffsim",
    "prefect_qiskit",
    "pydantic_numpy",
    "pyscf",
    "qiskit_addon_sqd",
    "qcsc_prefect_dice",
):
    pytest.importorskip(_mod)

import qcsc_workflow_utility.chem as chem  # noqa: E402
from qiskit_addon_sqd.fermion import SCIResult, solve_fermion  # noqa: E402
from sqd_common import flow, postprocess, sampling  # noqa: E402
from sqd_dice import main  # noqa: E402

LIH = "Li 0 0 0; H 0 0 1.6"
LOGGER = logging.getLogger("sqd_dice.test")


class _TaskStub:
    def __init__(self, fn):
        self.fn = fn

    def with_options(self, **options):
        return self.fn

    def __call__(self, **kwargs):
        return self.fn(**kwargs)


@pytest.fixture
def offline(monkeypatch):
    for module in (chem, flow, sampling, postprocess):
        monkeypatch.setattr(module, "get_run_logger", lambda: LOGGER)
    artifacts = []

    async def fake_artifact(*, table, key, **kwargs):
        artifacts.append((key, table))

    monkeypatch.setattr(postprocess, "create_table_artifact", fake_artifact)
    for name in ("compute_molecular_integrals_from_geometry", "sqd_loop"):
        monkeypatch.setattr(flow, name, _TaskStub(getattr(flow, name).fn))

    calls, loaded = [], []

    class Solver:
        async def run(self, **kwargs):
            calls.append(kwargs)
            energy, state, occ, _ = solve_fermion(
                kwargs["ci_strings"],
                kwargs["one_body_tensor"],
                kwargs["two_body_tensor"],
                open_shell=True,
            )
            return SCIResult(energy=energy, sci_state=state, orbital_occupancies=occ)

    async def load(name):
        loaded.append(name)
        return Solver()

    monkeypatch.setattr(main.DiceSHCISolverJob, "load", staticmethod(load))
    return dict(artifacts=artifacts, calls=calls, loaded=loaded)


def test_flow_diagonalizes_with_the_dice_block_and_passes_spin_sq(offline):
    params = main.FlowParameters(
        molecule={"atom": LIH, "basis": "sto-3g", "spin_sq": 0.5, "ccsd_maxiter": 1},
        sqd={"subspace_dim": 64, "num_batches": 2, "max_iterations": 2},
        sampling={"quantum_source": "random", "shots": 3000},
    )
    energy = asyncio.run(main.sqd_2405_05068.fn(parameters=params))

    assert offline["loaded"] == ["sqd-solver"]
    assert len(offline["calls"]) == 4
    assert all(call["spin_sq"] == 0.5 for call in offline["calls"])
    ((key, table),) = offline["artifacts"]
    assert key == main.ARTIFACT_KEY == "sqd-configuration-recovery"
    assert energy == pytest.approx(min(row["energy"] for row in table))
    assert -7.8825 < energy < -7.80


def test_flow_keeps_its_registered_name_and_takes_one_argument():
    from prefect.utilities.callables import parameter_schema

    assert main.sqd_2405_05068.name == "sqd-2405-05068"
    schema = parameter_schema(main.sqd_2405_05068.fn).model_dump()
    assert set(schema["properties"]) == {"parameters"}


def test_perf_tuning_flow_takes_dice_flow_parameters():
    import importlib.util
    from pathlib import Path

    path = Path(main.__file__).resolve().parents[1] / "perf" / "tune_dice.py"
    spec = importlib.util.spec_from_file_location("_tune_dice", path)
    tune_dice = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tune_dice)
    hints = tune_dice.main.fn.__annotations__
    assert hints["base_parameters"] is main.FlowParameters
    assert set(hints) == {"base_parameters", "experiments", "return"}
