"""The ``sqd`` flow with every Prefect-server interaction mocked.

The flow body is called through ``.fn``, so no flow run, task run, block load or
artifact reaches a Prefect API. Tasks are replaced by their ``.fn`` and each
solver block class by an in-process ``solve_fermion`` stand-in.
"""

from __future__ import annotations

import asyncio
import importlib.util
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

for _mod in ("prefect", "ffsim", "prefect_qiskit", "pydantic_numpy", "pyscf", "qiskit_addon_sqd"):
    pytest.importorskip(_mod)

import qcsc_workflow_utility.chem as chem  # noqa: E402
from qiskit_addon_sqd.fermion import SCIResult, solve_fermion  # noqa: E402
from sqd_common import flow, main, postprocess, sampling  # noqa: E402
from sqd_common.flow_params import FlowParameters  # noqa: E402

LIH = "Li 0 0 0; H 0 0 1.6"
LOGGER = logging.getLogger("sqd_common.main.test")
SQD = Path(__file__).resolve().parents[1]


class _TaskStub:
    """Stand-in for a Prefect task: ``with_options(...)`` returns the plain function."""

    def __init__(self, fn):
        self.fn = fn

    def with_options(self, **options):
        return self.fn

    def __call__(self, **kwargs):
        return self.fn(**kwargs)


class _FakeSolver:
    """A solver block class: ``load`` records the name, ``run`` solves in process."""

    def __init__(self, kind, calls, loaded):
        self.kind, self.calls, self.loaded = kind, calls, loaded
        self.__name__ = f"Fake{kind.capitalize()}SolverJob"

    async def load(self, name):
        self.loaded.append((self.kind, name))
        return self

    async def run(self, **kwargs):
        self.calls.append(kwargs)
        energy, state, occ, _ = solve_fermion(
            kwargs["ci_strings"],
            kwargs["one_body_tensor"],
            kwargs["two_body_tensor"],
            open_shell=True,
        )
        if self.kind == "sbd":
            from qcsc_prefect_sbd import check_sbd_input

            check_sbd_input(
                ci_strings=kwargs["ci_strings"], nelec=kwargs["nelec"], spin_sq=kwargs["spin_sq"]
            )
            return SimpleNamespace(energy=energy, orbital_occupancies=occ)
        return SCIResult(energy=energy, sci_state=state, orbital_occupancies=occ)


@pytest.fixture
def offline(monkeypatch, tmp_path):
    for module in (chem, flow, sampling, postprocess):
        monkeypatch.setattr(module, "get_run_logger", lambda: LOGGER)
    monkeypatch.setattr(postprocess.flow_run, "get_id", lambda: "run")

    artifacts = []

    async def fake_artifact(*, table, key, **kwargs):
        artifacts.append((key, table))

    monkeypatch.setattr(postprocess, "create_table_artifact", fake_artifact)
    for name in ("compute_molecular_integrals_from_geometry", "sqd_loop"):
        monkeypatch.setattr(flow, name, _TaskStub(getattr(flow, name).fn))

    calls, loaded = [], []

    def solver_class(slug):
        solver = _FakeSolver(slug.split("_")[0], calls, loaded)
        solver.root_dir = str(tmp_path)
        return solver

    monkeypatch.setattr(flow, "solver_class", solver_class)
    return dict(artifacts=artifacts, calls=calls, loaded=loaded)


def _parameters(solver_block_ref=None, spin_sq=0.0, **sqd):
    params = dict(
        molecule={"atom": LIH, "basis": "sto-3g", "spin_sq": spin_sq, "ccsd_maxiter": 1},
        sqd={"subspace_dim": 64, "num_batches": 2, "max_iterations": 2, **sqd},
        sampling={"quantum_source": "random", "shots": 3000},
    )
    if solver_block_ref:
        params["solver_block_ref"] = solver_block_ref
    return FlowParameters(**params)


def _recovery_table(offline):
    return dict(offline["artifacts"])[main.ARTIFACT_KEY]


def test_flow_defaults_to_the_fulqrum_solver_block(offline):
    energy = asyncio.run(main.sqd.fn(parameters=_parameters()))

    assert offline["loaded"] == [("fulqrum", "sqd-fulqrum-solver")]
    calls = offline["calls"]
    assert len(calls) == 4
    for call in calls:
        assert set(call) == {
            "ci_strings",
            "one_body_tensor",
            "two_body_tensor",
            "norb",
            "nelec",
            "spin_sq",
        }
        assert call["norb"] == 6 and call["nelec"] == (2, 2)
    table = _recovery_table(offline)
    assert [row["round"] for row in table] == [1, 1, 2, 2]
    assert energy == pytest.approx(min(row["energy"] for row in table))
    # LiH/STO-3G: SQD is variational, a few mHa above FCI (-7.882 Ha) at most here.
    assert -7.8825 < energy < -7.80


def test_dice_block_gets_spin_sq(offline):
    params = _parameters("dice_shci_solver_job/my-dice", spin_sq=0.5)
    energy = asyncio.run(main.sqd.fn(parameters=params))
    assert offline["loaded"] == [("dice", "my-dice")]
    assert len(offline["calls"]) == 4
    assert all(call["spin_sq"] == 0.5 for call in offline["calls"])
    assert -7.8825 < energy < -7.80


def test_sbd_block_gets_hf_first_closed_shell_subspaces(offline):
    pytest.importorskip("qcsc_prefect_sbd")
    params = _parameters("sbd_solver_job/my-sbd", spin_sq=None)
    energy = asyncio.run(main.sqd.fn(parameters=params))
    assert offline["loaded"] == [("sbd", "my-sbd")]
    limit = int(np.sqrt(64)) - 1
    for call in offline["calls"]:
        alpha, beta = call["ci_strings"]
        assert alpha[0] == 0b11 and len(alpha) <= limit + 1
        np.testing.assert_array_equal(alpha, beta)
        assert call["spin_sq"] is None
    assert np.isfinite(energy)


def test_sbd_with_spin_sq_fails_before_any_solve(offline):
    pytest.importorskip("qcsc_prefect_sbd")
    with pytest.raises(ValueError, match="spin_sq"):
        asyncio.run(main.sqd.fn(parameters=_parameters("sbd_solver_job/my-sbd")))
    assert offline["calls"] == []


def test_flow_parameter_schema_builds_offline():
    from prefect.utilities.callables import parameter_schema

    schema = parameter_schema(main.sqd.fn).model_dump()
    assert set(schema["properties"]) == {"parameters"}
    definitions = schema["definitions"]
    assert definitions["FlowParameters"]["properties"]["solver_block_ref"]["default"] == (
        "fulqrum_solver_job/sqd-fulqrum-solver"
    )
    assert {"MoleculeGeometry", "FCIDumpFile", "SamplingParameters"} <= set(definitions)
    assert main.sqd.name == "sqd"


def test_flow_version_defaults_to_the_installed_version(monkeypatch):
    monkeypatch.setenv("WF_VERSION", "abc")
    assert main.flow_version() == "abc"
    monkeypatch.delenv("WF_VERSION")
    assert main.flow_version()  # installed version, or "unknown" from a bare checkout


def test_perf_tuning_flow_takes_the_flow_parameters():
    path = SQD / "perf" / "tune_dice.py"
    spec = importlib.util.spec_from_file_location("_tune_dice", path)
    tune_dice = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(tune_dice)
    except ModuleNotFoundError as exc:
        pytest.skip(f"perf flow needs sqd[dice]: {exc}")
    hints = tune_dice.main.fn.__annotations__
    assert hints["base_parameters"] is FlowParameters
    assert set(hints) == {"base_parameters", "experiments", "return"}
