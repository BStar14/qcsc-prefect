"""The sqd_fulqrum flow with every Prefect-server interaction mocked.

The flow body is called through ``.fn``, so no flow run, task run, block load or
artifact reaches a Prefect API. Tasks are replaced by their ``.fn`` and the
solver block by an in-process ``solve_fermion`` stand-in.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

for _mod in ("prefect", "ffsim", "prefect_qiskit", "pydantic_numpy", "pyscf", "qiskit_addon_sqd"):
    pytest.importorskip(_mod)

import qcsc_workflow_utility.chem as chem  # noqa: E402
from qiskit_addon_sqd.fermion import SCIResult, solve_fermion  # noqa: E402
from sqd_common import flow, postprocess, sampling  # noqa: E402
from sqd_fulqrum import main  # noqa: E402

LIH = "Li 0 0 0; H 0 0 1.6"
LOGGER = logging.getLogger("sqd_fulqrum.test")


class _TaskStub:
    """Stand-in for a Prefect task: ``with_options(...)`` returns the plain function."""

    def __init__(self, fn):
        self.fn = fn

    def with_options(self, **options):
        return self.fn

    def __call__(self, **kwargs):
        return self.fn(**kwargs)


class _FakeSolver:
    def __init__(self):
        self.calls = []

    async def run(self, **kwargs):
        self.calls.append(kwargs)
        energy, state, occ, _ = solve_fermion(
            kwargs["ci_strings"],
            kwargs["one_body_tensor"],
            kwargs["two_body_tensor"],
            open_shell=True,
        )
        return SCIResult(energy=energy, sci_state=state, orbital_occupancies=occ)


@pytest.fixture
def offline(monkeypatch):
    for module in (main, chem, flow, sampling, postprocess):
        monkeypatch.setattr(module, "get_run_logger", lambda: LOGGER)

    artifacts = []

    async def fake_artifact(*, table, key, **kwargs):
        artifacts.append((key, table))

    monkeypatch.setattr(postprocess, "create_table_artifact", fake_artifact)
    for name in ("compute_molecular_integrals_from_geometry", "sqd_loop"):
        monkeypatch.setattr(flow, name, _TaskStub(getattr(flow, name).fn))

    solver = _FakeSolver()
    loaded = []

    async def load(name):
        loaded.append(name)
        return solver

    monkeypatch.setattr(main.FulqrumSolverJob, "load", staticmethod(load))
    return dict(artifacts=artifacts, solver=solver, loaded=loaded)


def _parameters(**sqd):
    return main.FlowParameters(
        molecule={"atom": LIH, "basis": "sto-3g", "ccsd_maxiter": 1},
        sqd={"subspace_dim": 64, "num_batches": 2, "max_iterations": 2, **sqd},
        sampling={"quantum_source": "random", "shots": 3000},
    )


def test_flow_runs_recovery_with_the_solver_block(offline):
    energy = asyncio.run(main.sqd_2405_05068_fulqrum.fn(parameters=_parameters()))

    assert offline["loaded"] == [main.DEFAULT_SOLVER_BLOCK_NAME]
    calls = offline["solver"].calls
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
    ((key, table),) = offline["artifacts"]
    assert key == main.ARTIFACT_KEY
    assert [row["round"] for row in table] == [0, 0, 1, 1]
    assert energy == pytest.approx(min(row["energy"] for row in table))
    # LiH/STO-3G: SQD is variational, a few mHa above FCI (-7.882 Ha) at most here.
    assert -7.8825 < energy < -7.80


def test_flow_loads_the_named_solver_block(offline):
    params = _parameters(max_iterations=1, num_batches=1)
    params.solver_block_name = "my-solver"
    asyncio.run(main.sqd_2405_05068_fulqrum.fn(parameters=params))
    assert offline["loaded"] == ["my-solver"]


def test_flow_warns_that_spin_sq_is_ignored(offline, caplog):
    params = _parameters(max_iterations=1, num_batches=1)
    params.molecule.spin_sq = 2.0
    with caplog.at_level(logging.WARNING, logger=LOGGER.name):
        asyncio.run(main.sqd_2405_05068_fulqrum.fn(parameters=params))
    assert "spin_sq=2.0 is ignored" in caplog.text


def test_flow_parameters_default_to_the_fulqrum_solver_block():
    params = main.FlowParameters.model_validate(
        {
            "molecule": {"fcidump_file": "/data/n2.fcidump"},
            "sqd": {"subspace_dim": 100, "num_batches": 1, "max_iterations": 1},
        }
    )
    assert params.solver_block_name == "sqd-fulqrum-solver"
    assert params.sampling.quantum_source == "real-device"


def test_flow_parameter_schema_builds_offline():
    from prefect.utilities.callables import parameter_schema

    schema = parameter_schema(main.sqd_2405_05068_fulqrum.fn).model_dump()
    assert set(schema["properties"]) == {"parameters"}
    definitions = schema["definitions"]
    assert definitions["FlowParameters"]["properties"]["solver_block_name"]["default"] == (
        "sqd-fulqrum-solver"
    )
    assert {"MoleculeGeometry", "FCIDumpFile", "SamplingParameters"} <= set(definitions)
    assert main.sqd_2405_05068_fulqrum.name == "sqd_2405_05068_fulqrum"
