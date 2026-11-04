"""SQD and SKQD run to the end in random mode with DICE or SBD replaced by a tiny fake.

The flows run without a Prefect server: their Prefect calls (variables, artifacts, result
caching) are replaced. The solver blocks are replaced by fakes that record their input.
"""

from __future__ import annotations

import asyncio
import importlib.util
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from qcsc_workflow_utility import chem
from qiskit_addon_sqd.fermion import SCIResult

pytest.importorskip("sqd_dice")
pytest.importorskip("skqd_dice")

ROOT = Path(__file__).resolve().parents[3]
N2 = str(ROOT / "algorithms/closed_loop/sqd_de/data/fcidump_N2_MO.txt")
H2 = str(ROOT / "algorithms/closed_loop/sqd_de/data/fcidump_H2_STO3G.txt")
SHOTS = 2000
FLOWS = {
    "sqd": ("sqd_dice.main", "sqd_2405_05068", "sqd", "sqd-solver"),
    "skqd": ("skqd_dice.main", "skqd_2501_09702", "skqd", "skqd-solver"),
}


class FakeSolver:
    """Records every run; returns a fixed energy and uniform occupancies."""

    calls: list = []

    @classmethod
    async def load(cls, name):
        FakeSolver.calls.append(("load", cls.kind, name))
        return cls()

    async def run(self, **kwargs):
        FakeSolver.calls.append(("run", self.kind, kwargs))
        occupancy = np.full(kwargs["norb"], 0.5)
        return self.result(-1.0 - 0.01 * len(kwargs["ci_strings"][0]), occupancy)


class FakeDice(FakeSolver):
    kind = "DICE"

    @staticmethod
    def result(energy, occupancy):
        return SCIResult(energy=energy, sci_state=None, orbital_occupancies=(occupancy,) * 2)


class FakeSbd(FakeSolver):
    kind = "SBD"

    async def run(self, **kwargs):
        # Keep the real SBD input checks.
        from qcsc_prefect_sbd import check_sbd_input

        check_sbd_input(
            ci_strings=kwargs["ci_strings"], nelec=kwargs["nelec"], spin_sq=kwargs["spin_sq"]
        )
        return await super().run(**kwargs)

    @staticmethod
    def result(energy, occupancy):
        return SimpleNamespace(energy=energy, orbital_occupancies=(occupancy,) * 2)


def prepare(module, monkeypatch, *, real_solver_names=True):
    """Replace the Prefect calls and the solver classes of a flow module."""

    class Variable:
        @staticmethod
        async def get(name, default=None):
            return {"params": {"shots": SHOTS}}

    async def no_artifact(**kwargs):
        return None

    logger = SimpleNamespace(info=lambda *a, **k: None)
    monkeypatch.setattr(module, "Variable", Variable)
    monkeypatch.setattr(module, "create_table_artifact", no_artifact)
    monkeypatch.setattr(module, "get_run_logger", lambda: logger)
    monkeypatch.setattr(chem, "get_run_logger", lambda: logger)
    for name in (
        "compute_molecular_integrals_from_fcidump",
        "compute_molecular_integrals_from_geometry",
        "sample_bitstrings",
    ):
        fn = getattr(module, name).fn
        monkeypatch.setattr(module, name, SimpleNamespace(with_options=lambda fn=fn, **k: fn))
    monkeypatch.setattr(module, "DiceSHCISolverJob", FakeDice)
    if hasattr(module, "SBDSolverJob"):
        monkeypatch.setattr(module, "SBDSolverJob", FakeSbd)
    FakeSolver.calls = []


def parameters(module, params_key, fcidump, spin_sq, subspace_dim=400):
    flow_params = {"subspace_dim": subspace_dim, "num_batches": 2, "max_iterations": 2}
    if params_key == "skqd":
        flow_params["krylov_dim"] = 2
    return module.Parameters.model_validate(
        {
            "molecule": {"fcidump_file": fcidump, "spin_sq": spin_sq},
            params_key: flow_params,
            "quantum_source": "random",
        }
    )


def run_flow(flow_key, monkeypatch, *, fcidump=N2, spin_sq=0.0, **flow_kwargs):
    module_name, flow_name, params_key, _ = FLOWS[flow_key]
    module = importlib.import_module(module_name)
    prepare(module, monkeypatch)
    params = parameters(module, params_key, fcidump, spin_sq)
    energy = asyncio.run(getattr(module, flow_name).fn(params, **flow_kwargs))
    return energy, FakeSolver.calls


def runs(calls):
    return [kwargs for kind, _, kwargs in calls if kind == "run"]


@pytest.mark.parametrize("flow_key", FLOWS)
def test_dice_is_the_default_and_gets_spin_sq(flow_key, monkeypatch):
    energy, calls = run_flow(flow_key, monkeypatch)

    assert calls[0] == ("load", "DICE", FLOWS[flow_key][3])
    assert runs(calls) and all(call["spin_sq"] == 0.0 for call in runs(calls))
    assert np.isfinite(energy)


@pytest.mark.parametrize("flow_key", FLOWS)
def test_sbd_runs_to_the_end_with_hf_first_and_the_size_limit(flow_key, monkeypatch):
    energy, calls = run_flow(
        flow_key, monkeypatch, spin_sq=None, solver_block_ref="sbd_solver_job/davidson"
    )

    assert calls[0] == ("load", "SBD", "davidson")
    hartree_fock = (1 << 7) - 1
    assert len(runs(calls)) == 4  # two batches in each of two iterations
    for call in runs(calls):
        alpha, beta = call["ci_strings"]
        assert np.array_equal(alpha, beta)
        assert alpha[0] == hartree_fock
        assert len(alpha) <= int(np.sqrt(400))
        assert call["spin_sq"] is None
    assert np.isfinite(energy)


@pytest.mark.parametrize("flow_key", FLOWS)
@pytest.mark.parametrize("fcidump,spin_sq,message", [(N2, 0.0, "SBD has no spin_sq target")])
def test_sbd_input_it_cannot_solve_fails_before_sampling(
    flow_key, monkeypatch, fcidump, spin_sq, message
):
    module = importlib.import_module(FLOWS[flow_key][0])
    sampled = []
    original = module.sample_bitstrings.fn

    async def recording_sample(**kwargs):
        sampled.append(kwargs)
        return await original(**kwargs)

    with pytest.raises(ValueError, match=message):
        prepare(module, monkeypatch)
        monkeypatch.setattr(
            module, "sample_bitstrings", SimpleNamespace(with_options=lambda **k: recording_sample)
        )
        params = parameters(module, FLOWS[flow_key][2], fcidump, spin_sq)
        asyncio.run(
            getattr(module, FLOWS[flow_key][1]).fn(params, solver_block_ref="sbd_solver_job/d")
        )
    assert sampled == []
    assert runs(FakeSolver.calls) == []


@pytest.mark.parametrize("flow_key", FLOWS)
def test_unknown_solver_slug_is_rejected(flow_key, monkeypatch):
    with pytest.raises(ValueError, match="solver_block_ref"):
        run_flow(flow_key, monkeypatch, solver_block_ref="other_solver/x")


# The last commit on main before solver selection (the merge of #55), and the flow paths
# at that commit. A fixed revision keeps the comparison meaningful after later changes.
PREVIOUS_REVISION = "a394b846c69c08da272e21db954b0378b517ce8d"
PREVIOUS_PATHS = {
    "sqd": "algorithms/sqd/sqd_dice/main.py",
    "skqd": "algorithms/skqd/skqd_dice/main.py",
}


@pytest.fixture(scope="module")
def previous_flows(tmp_path_factory):
    """The SQD and SKQD flow modules before solver selection, read from git history."""
    directory = tmp_path_factory.mktemp("previous")
    modules = {}
    for flow_key in FLOWS:
        source = subprocess.run(
            ["git", "-C", str(ROOT), "show", f"{PREVIOUS_REVISION}:{PREVIOUS_PATHS[flow_key]}"],
            capture_output=True,
            text=True,
        )
        if source.returncode != 0:
            # For example a shallow clone without this commit.
            pytest.skip(f"commit {PREVIOUS_REVISION[:7]} is not available in this clone")
        assert "solver_block_ref" not in source.stdout
        path = directory / f"previous_{flow_key}.py"
        path.write_text(source.stdout)
        spec = importlib.util.spec_from_file_location(f"previous_{flow_key}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules[flow_key] = module
    return modules


@pytest.mark.parametrize("flow_key", FLOWS)
def test_dice_gets_the_same_input_as_before(flow_key, monkeypatch, previous_flows):
    _, flow_name, params_key, _ = FLOWS[flow_key]

    def solver_inputs(module):
        prepare(module, monkeypatch)
        params = parameters(module, params_key, N2, 0.0)
        asyncio.run(getattr(module, flow_name).fn(params))
        return [
            (
                [np.asarray(s).tolist() for s in call["ci_strings"]],
                call["nelec"],
                call["spin_sq"],
                call["norb"],
            )
            for call in runs(FakeSolver.calls)
        ]

    before = solver_inputs(previous_flows[flow_key])
    after = solver_inputs(importlib.import_module(FLOWS[flow_key][0]))
    assert before and after == before


@pytest.mark.parametrize("flow_key", FLOWS)
def test_open_shell_with_sbd_fails_before_sampling(flow_key, monkeypatch):
    # The molecule helpers build RHF references, so an open-shell system is given here
    # directly as electronic properties with nelec=(4, 3).
    module_name, flow_name, params_key, _ = FLOWS[flow_key]
    module = importlib.import_module(module_name)
    prepare(module, monkeypatch)
    closed = chem.compute_molecular_integrals_from_fcidump.fn(N2, spin_sq=None)
    open_shell = closed.model_copy(update={"num_electrons": (4, 3), "open_shell": True})
    monkeypatch.setattr(
        module,
        "compute_molecular_integrals_from_fcidump",
        SimpleNamespace(with_options=lambda **k: lambda **kw: open_shell),
    )
    sampled = []

    async def recording_sample(**kwargs):
        sampled.append(kwargs)

    monkeypatch.setattr(
        module, "sample_bitstrings", SimpleNamespace(with_options=lambda **k: recording_sample)
    )
    params = parameters(module, params_key, N2, None)

    with pytest.raises(ValueError, match="closed-shell systems only"):
        asyncio.run(getattr(module, flow_name).fn(params, solver_block_ref="sbd_solver_job/d"))
    assert sampled == []
    assert runs(FakeSolver.calls) == []


@pytest.mark.parametrize("flow_key", FLOWS)
def test_existing_positional_arguments_keep_their_positions(flow_key, previous_flows):
    import inspect

    _, flow_name, _, _ = FLOWS[flow_key]
    new = inspect.signature(getattr(importlib.import_module(FLOWS[flow_key][0]), flow_name).fn)
    old = inspect.signature(getattr(previous_flows[flow_key], flow_name).fn)
    old_names = list(old.parameters)
    assert list(new.parameters)[: len(old_names)] == old_names
    assert list(new.parameters)[len(old_names) :] == ["solver_block_ref"]


@pytest.mark.parametrize("flow_key", FLOWS)
def test_positional_call_from_before_still_selects_dice(flow_key, monkeypatch):
    # flow(params, runner_name, solver_name, option_name), as callers wrote before.
    module_name, flow_name, params_key, _ = FLOWS[flow_key]
    module = importlib.import_module(module_name)
    prepare(module, monkeypatch)
    params = parameters(module, params_key, N2, 0.0)

    energy = asyncio.run(
        getattr(module, flow_name).fn(params, "qrmi-runtime", "my-dice", "sampler_options")
    )

    assert FakeSolver.calls[0] == ("load", "DICE", "my-dice")
    assert np.isfinite(energy)
