"""Flow wiring of sqd_fulqrum.main with every Prefect-server interaction mocked.

The flow body is called through ``.fn``, so no flow run, task run, block load or
variable lookup reaches a Prefect API. Tasks are replaced by their ``.fn`` and
the solver block by an in-process ``solve_fermion`` stand-in.
"""

from __future__ import annotations

import asyncio
import logging

import numpy as np
import pytest

for _mod in ("prefect", "ffsim", "prefect_qiskit", "pydantic_numpy", "pyscf", "qiskit_addon_sqd"):
    pytest.importorskip(_mod)

import qcsc_workflow_utility.chem as chem  # noqa: E402
from pydantic import ValidationError  # noqa: E402
from qiskit_addon_sqd.fermion import SCIResult, solve_fermion  # noqa: E402
from sqd_fulqrum import main  # noqa: E402

LIH = "Li 0 0 0; H 0 0 1.6"


class _Variable:
    values: dict = {}

    @classmethod
    async def get(cls, name, default=None):
        return cls.values.get(name, default)


class _TaskStub:
    """Stand-in for a Prefect task: ``with_options(...)`` returns the plain function."""

    def __init__(self, fn):
        self.fn = fn
        self.options = []

    def with_options(self, **options):
        self.options.append(options)
        return self.fn


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
    logger = logging.getLogger("sqd_fulqrum.test")
    monkeypatch.setattr(main, "get_run_logger", lambda: logger)
    monkeypatch.setattr(chem, "get_run_logger", lambda: logger)
    monkeypatch.setattr(main, "Variable", _Variable)
    _Variable.values = {"sampler_options": {"params": {"shots": 3000}}}

    artifacts = []

    async def fake_artifact(*, table, key):
        artifacts.append((key, table))

    monkeypatch.setattr(main, "create_table_artifact", fake_artifact)

    integrals = _TaskStub(main.compute_molecular_integrals_from_geometry.fn)
    sampler = _TaskStub(main.sample_bitstrings.fn)
    monkeypatch.setattr(main, "compute_molecular_integrals_from_geometry", integrals)
    monkeypatch.setattr(main, "sample_bitstrings", sampler)

    solver = _FakeSolver()
    loaded = []

    async def load(name):
        loaded.append(name)
        return solver

    monkeypatch.setattr(main.FulqrumSolverJob, "load", staticmethod(load))
    return dict(
        artifacts=artifacts, solver=solver, loaded=loaded, integrals=integrals, sampler=sampler
    )


def _parameters(**sqd):
    return main.Parameters(
        molecule=main.MoleculeGeometry(atom=LIH, basis="sto-3g"),
        sqd=main.SQDParameters(
            **{"subspace_dim": 64, "num_batches": 2, "max_iterations": 2, **sqd}
        ),
        quantum_source="random",
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
        }
        assert call["norb"] == 6 and call["nelec"] == (2, 2)
    ((key, table),) = offline["artifacts"]
    assert key == main.ARTIFACT_KEY
    assert [row["round"] for row in table] == [0, 0, 1, 1]
    assert energy == pytest.approx(min(row["energy"] for row in table))
    # LiH/STO-3G: SQD is variational, a few mHa above FCI (-7.882 Ha) at most here.
    assert -7.8825 < energy < -7.80
    assert offline["integrals"].options[0]["cache_policy"] is main.NO_CACHE
    assert offline["sampler"].options[0]["cache_policy"] is main.NO_CACHE


def test_flow_passes_names_and_cache_options(offline):
    asyncio.run(
        main.sqd_2405_05068_fulqrum.fn(
            parameters=_parameters(max_iterations=1, num_batches=1),
            solver_name="my-solver",
            cache_compute_integrals=True,
            cache_sampling=True,
        )
    )
    assert offline["loaded"] == ["my-solver"]
    assert offline["integrals"].options[0]["cache_policy"] is main.INPUTS
    assert offline["sampler"].options[0]["cache_policy"] is main.INPUTS


def test_flow_warns_that_spin_sq_is_ignored(offline, caplog):
    params = _parameters(max_iterations=1, num_batches=1)
    params.molecule.spin_sq = 2.0
    with caplog.at_level(logging.WARNING, logger="sqd_fulqrum.test"):
        asyncio.run(main.sqd_2405_05068_fulqrum.fn(parameters=params))
    assert "spin_sq=2.0 is ignored" in caplog.text


def test_random_sampling_uses_shots_variable(offline):
    elec_props = chem.compute_molecular_integrals_from_geometry.fn(atom=LIH, basis="sto-3g")
    bit_array = asyncio.run(
        main.sample_bitstrings.fn(
            circuit_params=main.CircuitParameters(),
            elec_props=elec_props,
            runner_name="unused",
            option_name="sampler_options",
            quantum_source="random",
            random_seed=7,
        )
    )
    assert bit_array.num_shots == 3000
    assert bit_array.num_bits == 12


def test_real_device_without_runtime_block_explains_random_fallback(offline, monkeypatch):
    async def missing(name):
        raise ValueError(f"Unable to find block document named {name}")

    monkeypatch.setattr(main.QuantumRuntime, "load", staticmethod(missing))
    elec_props = chem.compute_molecular_integrals_from_geometry.fn(atom=LIH, basis="sto-3g")
    with pytest.raises(RuntimeError, match="quantum_source='random'"):
        asyncio.run(
            main.sample_bitstrings.fn(
                circuit_params=main.CircuitParameters(),
                elec_props=elec_props,
                runner_name="ibm-runner",
                option_name="sampler_options",
                quantum_source="real-device",
                random_seed=7,
            )
        )


def test_make_block_solver_binds_integrals():
    solver = _FakeSolver()
    elec_props = type(
        "Props",
        (),
        dict(
            one_body_tensor=np.eye(2),
            two_body_tensor=np.zeros((2, 2, 2, 2)),
            num_orbitals=2,
            num_electrons=(1, 1),
        ),
    )()
    solve = main.make_block_solver(solver, elec_props)
    ci = (np.array([1, 2]), np.array([1, 2]))
    result = asyncio.run(solve(ci))
    assert result.energy == pytest.approx(2.0)
    assert solver.calls[0]["ci_strings"] is ci
    assert "spin_sq" not in solver.calls[0]


def test_parameters_require_molecule_and_sqd():
    with pytest.raises(ValidationError):
        main.Parameters()
    params = main.Parameters.model_validate(
        {
            "molecule": {"fcidump_file": "/data/n2.fcidump"},
            "sqd": {"subspace_dim": 100, "num_batches": 1, "max_iterations": 1},
        }
    )
    assert isinstance(params.molecule, main.FCIDumpFile)
    assert params.quantum_source == "real-device"


def test_flow_parameter_schema_builds_offline():
    from prefect.utilities.callables import parameter_schema

    schema = parameter_schema(main.sqd_2405_05068_fulqrum.fn).model_dump()
    assert {"parameters", "runner_name", "solver_name", "option_name"} <= set(schema["properties"])
    assert schema["properties"]["solver_name"]["default"] == "sqd-fulqrum-solver"
    assert main.sqd_2405_05068_fulqrum.name == "sqd_2405_05068_fulqrum"
