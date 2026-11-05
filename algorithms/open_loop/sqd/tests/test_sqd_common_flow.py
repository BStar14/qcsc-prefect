"""Data-flow wiring of the shared SQD flow body (``sqd_common.flow``).

Prefect draws an edge between two tasks only when one task's return value is
passed unchanged into the other. These tests replace every task with a stub
that returns a sentinel object, then check which object reaches which stage.
"""

from __future__ import annotations

import asyncio
import logging

import numpy as np
import pytest

for _mod in ("ffsim", "prefect_qiskit", "pydantic_numpy", "pyscf", "qiskit_addon_sqd"):
    pytest.importorskip(_mod)

from sqd_common import flow, sampling  # noqa: E402
from sqd_common.flow_params import FlowParameters  # noqa: E402

LOGGER = logging.getLogger("sqd_common.flow.test")


class _Task:
    """Records calls and returns a fixed object, like a Prefect task's result."""

    def __init__(self, name, result, log):
        self.name, self.result, self.log = name, result, log
        self.options = []

    def with_options(self, **options):
        self.options.append(options)
        return self

    def __call__(self, **kwargs):
        self.log.append((self.name, kwargs))
        return self.result


class _AsyncTask(_Task):
    async def __call__(self, **kwargs):
        return super().__call__(**kwargs)


@pytest.fixture
def stages(monkeypatch):
    log = []
    elec_props = type("Props", (), {"num_orbitals": 3})()
    sentinels = {
        "integrals": elec_props,
        "lucj": object(),
        "samples": np.zeros((2, 6), dtype=bool),
        "energy": -1.25,
    }
    stubs = {
        "compute_molecular_integrals_from_geometry": _Task("map", elec_props, log),
        "compute_molecular_integrals_from_fcidump": _Task("map-fcidump", elec_props, log),
        "get_lucj_params": _Task("optimize", sentinels["lucj"], log),
        "quantum_sampling": _AsyncTask("execute", sentinels["samples"], log),
        "load_runtime_job": _AsyncTask("load_runtime_job", sentinels["samples"], log),
        "load_qrmi_result": _AsyncTask("load_qrmi_result", sentinels["samples"], log),
        "sqd_loop": _AsyncTask("post-process", sentinels["energy"], log),
    }
    for name, stub in stubs.items():
        monkeypatch.setattr(flow, name, stub)
    monkeypatch.setattr(flow, "get_run_logger", lambda: LOGGER)
    monkeypatch.setattr(sampling, "get_run_logger", lambda: LOGGER)

    solver = object()
    loaded = []

    class Solver:
        @staticmethod
        async def load(name):
            loaded.append(name)
            return solver

    return dict(log=log, stubs=stubs, solver=solver, Solver=Solver, loaded=loaded, **sentinels)


def _parameters(molecule=None, **sampling):
    return FlowParameters(
        molecule=molecule or {"atom": "H 0 0 0; H 0 0 0.74", "basis": "sto-3g"},
        sampling=sampling,
        sqd={"subspace_dim": 4, "num_batches": 1, "max_iterations": 1},
        solver_block_name="my-solver",
    )


def _run(stages, parameters):
    return asyncio.run(flow.run_sqd_pattern(parameters, stages["Solver"], "key"))


def test_patterns_stages_hand_results_on_unchanged(stages):
    energy = _run(stages, _parameters())
    assert energy == stages["energy"]
    calls = dict(stages["log"])
    assert [name for name, _ in stages["log"]] == ["map", "optimize", "execute", "post-process"]
    # Map → Optimize
    assert calls["optimize"]["elec_props"] is stages["integrals"]
    # Optimize → Execute
    assert calls["execute"]["lucj_params"] is stages["lucj"]
    # Execute → Post-process, and Map → Post-process
    assert calls["post-process"]["bool_array"] is stages["samples"]
    assert calls["post-process"]["elec_props"] is stages["integrals"]
    assert calls["post-process"]["solver"] is stages["solver"]
    assert calls["post-process"]["artifact_key"] == "key"
    assert stages["stubs"]["compute_molecular_integrals_from_geometry"].options == [
        {"tags": ["map"]}
    ]


def test_molecule_settings_reach_the_map_stage(stages):
    _run(stages, _parameters(molecule={"fcidump_file": "/data/n2.fcidump", "ccsd_maxiter": 2}))
    name, kwargs = stages["log"][0]
    assert name == "map-fcidump"
    assert kwargs == {"fcidump_file": "/data/n2.fcidump", "spin_sq": 0.0, "ccsd_maxiter": 2}


def test_solver_block_is_loaded_before_any_stage(stages):
    class Missing:
        @staticmethod
        async def load(name):
            raise ValueError(f"Unable to find block document named {name}")

    stages["Solver"] = Missing
    with pytest.raises(ValueError, match="my-solver"):
        _run(stages, _parameters())
    assert stages["log"] == []


def test_random_source_skips_optimize_and_execute(stages):
    energy = _run(stages, _parameters(quantum_source="random", shots=10))
    assert energy == stages["energy"]
    assert [name for name, _ in stages["log"]] == ["map", "post-process"]
    samples = stages["log"][1][1]["bool_array"]
    assert samples.shape == (10, 6)


@pytest.mark.parametrize(
    "sampling, stage, expected",
    [
        (
            {"qiskit_runtime_job_id": "d123", "runner_name": "r"},
            "load_runtime_job",
            {"job_id": "d123", "runner_name": "r"},
        ),
        ({"qrmi_result_dir": "/jobs/abc"}, "load_qrmi_result", {"work_dir": "/jobs/abc"}),
    ],
)
def test_reload_replaces_optimize_and_execute(stages, sampling, stage, expected):
    _run(stages, _parameters(**sampling))
    names = [name for name, _ in stages["log"]]
    assert names == ["map", stage, "post-process"]
    kwargs = stages["log"][1][1]
    assert {k: kwargs[k] for k in expected} == expected
    assert stages["log"][2][1]["bool_array"] is stages["samples"]
