"""Data-flow wiring of the shared SQD flow body (``sqd_common.flow``).

Prefect draws an edge between two tasks only when one task's return value is
passed unchanged into the other. These tests replace every task with a stub
that returns a sentinel object, then check which object reaches which stage.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import types

import numpy as np
import pytest
from qrmi_stubs import QrmiBlockStub

for _mod in ("ffsim", "prefect_qiskit", "pydantic_numpy", "pyscf", "qiskit_addon_sqd"):
    pytest.importorskip(_mod)

from sqd_common import flow, sampling, solvers, topology  # noqa: E402
from sqd_common.advanced import AdvancedSettings  # noqa: E402
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
    elec_props = type("Props", (), {"num_orbitals": 3, "num_electrons": (1, 1), "spin_sq": 0.0})()
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
    advanced = {"settings": AdvancedSettings(), "loaded": []}

    async def load_advanced(name, logger):
        advanced["loaded"].append(name)
        return advanced["settings"]

    monkeypatch.setattr(flow, "load_advanced_settings", load_advanced)
    checked = []

    async def check(name, logger):
        log.append(("check", name))
        checked.append(name)
        return QrmiBlockStub(None, resource="test_heron")

    monkeypatch.setattr(flow, "check_qrmi_block", check)
    for module in (flow, sampling, topology):
        monkeypatch.setattr(module, "get_run_logger", lambda: LOGGER)

    solver = object()
    loaded = []

    class Solver:
        @staticmethod
        async def load(name):
            loaded.append(name)
            return solver

    selected = []

    def solver_class(slug):
        selected.append(slug)
        return stages_state["Solver"]

    stages_state = dict(
        log=log,
        stubs=stubs,
        solver=solver,
        Solver=Solver,
        loaded=loaded,
        selected=selected,
        advanced=advanced,
        checked=checked,
    )
    monkeypatch.setattr(flow, "solver_class", solver_class)
    stages_state.update(sentinels)
    return stages_state


def _parameters(molecule=None, solver="fulqrum_solver_job/my-solver", circuit=None, **sampling):
    return FlowParameters(
        molecule=molecule or {"atom": "H 0 0 0; H 0 0 0.74", "basis": "sto-3g"},
        circuit=circuit or {},
        sampling=sampling,
        sqd={"subspace_dim": 4, "num_batches": 1, "max_iterations": 1},
        solver_block_ref=solver,
    )


def _run(stages, parameters):
    return asyncio.run(flow.run_sqd_pattern(parameters, "key"))


def test_patterns_stages_hand_results_on_unchanged(stages):
    energy = _run(stages, _parameters())
    assert energy == stages["energy"]
    calls = dict(stages["log"])
    names = [name for name, _ in stages["log"]]
    assert names == ["check", "map", "optimize", "execute", "post-process"]
    # Map → Optimize
    assert calls["optimize"]["elec_props"] is stages["integrals"]
    # Optimize → Execute
    assert calls["execute"]["lucj_params"] is stages["lucj"]
    # Execute → Post-process, and Map → Post-process
    assert calls["post-process"]["bool_array"] is stages["samples"]
    assert calls["post-process"]["elec_props"] is stages["integrals"]
    assert calls["post-process"]["solver"] is stages["solver"]
    assert calls["post-process"]["artifact_key"] == "key"
    assert calls["post-process"]["batch_transform"] is None
    assert stages["selected"] == ["fulqrum_solver_job"] and stages["loaded"] == ["my-solver"]
    assert stages["stubs"]["compute_molecular_integrals_from_geometry"].options == [
        {"tags": ["map"]}
    ]


def test_the_qpu_backend_is_checked_before_map_and_fails_the_run(stages, monkeypatch):
    _run(stages, _parameters(runtime_block_name="my-qrmi"))
    assert stages["checked"] == ["my-qrmi"]
    # The checked block is the one sampling uses.
    assert dict(stages["log"])["execute"]["qrmi_block"].resource == "test_heron"

    from qcsc_prefect.integrations.qiskit.qrmi.block import BackendAccessError

    async def refuse(name, logger):
        raise BackendAccessError("ibm_fez is not available to open (open)")

    monkeypatch.setattr(flow, "check_qrmi_block", refuse)
    stages["log"].clear()
    with pytest.raises(BackendAccessError, match="not available"):
        _run(stages, _parameters())
    assert stages["log"] == []


@pytest.mark.parametrize(
    "sampling",
    [
        {"quantum_source": "random"},
        {"qrmi_result_dir": "/data/sqd/qrmi/x"},
        {"qiskit_runtime_job_id": "d7"},
    ],
)
def test_no_qpu_check_without_qpu_sampling(stages, sampling):
    _run(stages, _parameters(**sampling))
    assert stages["checked"] == []


@pytest.mark.parametrize("qpu_topology", [None, "heavy-hex", "square"])
def test_optimize_gets_the_qpu_topology(stages, caplog, qpu_topology):
    circuit = {} if qpu_topology is None else {"qpu_topology": qpu_topology}
    with caplog.at_level(logging.INFO, logger=LOGGER.name):
        _run(stages, _parameters(circuit=circuit))
    expected = qpu_topology or "heavy-hex"
    assert dict(stages["log"])["optimize"]["qpu_topology"] == expected
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings == [] and f"QPU topology: {expected}." in caplog.text


def test_molecule_settings_reach_the_map_stage(stages):
    stages["advanced"]["settings"] = AdvancedSettings(ccsd_maxiter=2)
    _run(stages, _parameters(molecule={"fcidump_file": "/data/n2.fcidump"}))
    name, kwargs = stages["log"][1]
    assert name == "map-fcidump"
    assert kwargs == {"fcidump_file": "/data/n2.fcidump", "spin_sq": None, "ccsd_maxiter": 2}


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


def test_sbd_gets_hf_first_batches_and_its_input_is_checked_after_map(stages):
    stages["integrals"].spin_sq = None
    _run(stages, _parameters(solver="sbd_solver_job/my-sbd"))
    assert stages["selected"] == ["sbd_solver_job"] and stages["loaded"] == ["my-sbd"]
    transform = dict(stages["log"])["post-process"]["batch_transform"]
    strings = np.array([1, 2, 4], dtype=np.int64)
    ((alpha, beta),) = list(transform([(strings, strings)], np.random.default_rng(0)))
    # Hartree-Fock (1 << 1) - 1 = 1 first; subspace_dim 4 allows one other determinant.
    assert alpha[0] == 1 and len(alpha) == 2
    np.testing.assert_array_equal(alpha, beta)
    # A carryover string keeps that slot.
    carryover = (np.array([4], dtype=np.int64), np.array([4], dtype=np.int64))
    for seed in range(10):
        ((alpha, _),) = list(
            transform([(strings, strings)], np.random.default_rng(seed), carryover)
        )
        assert alpha.tolist() == [1, 4]


def test_sbd_with_spin_sq_fails_before_sampling(stages):
    pytest.importorskip("qcsc_prefect_sbd")
    stages["integrals"].spin_sq = 0.0
    with pytest.raises(ValueError, match="spin_sq"):
        _run(stages, _parameters(solver="sbd_solver_job/my-sbd"))
    assert [name for name, _ in stages["log"]] == ["check", "map"]


@pytest.fixture
def sbd_checks(monkeypatch):
    """A ``qcsc_prefect_sbd.check_sbd_input`` that records its arguments."""
    calls = []
    module = types.ModuleType("qcsc_prefect_sbd")
    module.check_sbd_input = lambda **kwargs: calls.append(kwargs)
    monkeypatch.setitem(sys.modules, "qcsc_prefect_sbd", module)
    return calls


@pytest.mark.parametrize("nelec", [(1, 1), (2, 1), (3, 1)])
def test_sbd_input_check_gets_each_spins_hartree_fock(stages, sbd_checks, nelec):
    stages["integrals"].spin_sq = None
    stages["integrals"].num_electrons = nelec
    _run(stages, _parameters(solver="sbd_solver_job/my-sbd"))
    (call,) = sbd_checks
    alpha, beta = call["ci_strings"]
    assert alpha.tolist() == [(1 << nelec[0]) - 1] and beta.tolist() == [(1 << nelec[1]) - 1]
    assert call["nelec"] == nelec and call["spin_sq"] is None
    assert [name for name, _ in stages["log"]][-1] == "post-process"


def test_sbd_open_shell_batches_get_hf_first_per_spin(stages, sbd_checks):
    stages["integrals"].spin_sq = None
    stages["integrals"].num_electrons = (2, 1)
    _run(stages, _parameters(solver="sbd_solver_job/my-sbd"))
    transform = dict(stages["log"])["post-process"]["batch_transform"]
    alpha_in = np.array([0b101, 0b110], dtype=np.int64)
    beta_in = np.array([0b010, 0b100], dtype=np.int64)
    # subspace_dim 4: with Hartree-Fock, 3 x 3 strings are cut to 2 x 2.
    ((alpha, beta),) = list(transform([(alpha_in, beta_in)], np.random.default_rng(0)))
    assert (alpha[0], beta[0]) == (0b011, 0b001) and len(alpha) * len(beta) == 4
    assert set(alpha[1:]) <= {0b101, 0b110} and set(beta[1:]) <= {0b010, 0b100}
    # The carryover string of each spin keeps the slot after Hartree-Fock.
    carryover = (np.array([0b110], dtype=np.int64), np.array([0b100], dtype=np.int64))
    for seed in range(10):
        ((alpha, beta),) = list(
            transform([(alpha_in, beta_in)], np.random.default_rng(seed), carryover)
        )
        assert alpha.tolist() == [0b011, 0b110] and beta.tolist() == [0b001, 0b100]


def test_sbd_open_shell_passes_the_new_input_check(stages):
    """Needs the per-spin check_sbd_input of qcsc-prefect-sbd (open shell allowed)."""
    sbd = pytest.importorskip("qcsc_prefect_sbd")
    if not hasattr(sbd, "create_sbd_blocks"):
        pytest.skip("qcsc-prefect-sbd without open-shell support")
    stages["integrals"].spin_sq = None
    stages["integrals"].num_electrons = (2, 1)
    _run(stages, _parameters(solver="sbd_solver_job/my-sbd"))
    assert [name for name, _ in stages["log"]][-1] == "post-process"


def test_fulqrum_warns_that_spin_sq_is_ignored(stages, caplog):
    stages["integrals"].spin_sq = 2.0
    with caplog.at_level(logging.WARNING, logger=LOGGER.name):
        _run(stages, _parameters())
    assert "spin_sq=2.0 is ignored" in caplog.text


@pytest.mark.parametrize(
    "ref, slug",
    [
        ("fulqrum_solver_job/a", "fulqrum_solver_job"),
        ("dice-shci-solver-job/a", "dice_shci_solver_job"),
        ("sbd_solver_job/a", "sbd_solver_job"),
    ],
)
def test_solver_slugs_accept_both_spellings(ref, slug):
    assert solvers.solver_slug(ref) == (slug, "a")


@pytest.mark.parametrize("ref", ["other_solver/x", "no-slash", "fulqrum_solver_job/"])
def test_unknown_or_malformed_solver_ref_is_rejected(stages, ref):
    with pytest.raises(ValueError, match="solver_block_ref"):
        solvers.solver_slug(ref)


def test_missing_solver_package_names_the_extra(monkeypatch):
    missing = ("qcsc_prefect_not_installed", "DiceSHCISolverJob", "dice")
    monkeypatch.setitem(solvers.SOLVERS, "dice_shci_solver_job", missing)
    with pytest.raises(ModuleNotFoundError, match=r"sqd\[dice\]"):
        solvers.solver_class("dice_shci_solver_job")
    assert solvers.solver_class("fulqrum_solver_job").__name__ == "FulqrumSolverJob"


def test_a_bad_gpu_request_fails_when_the_solver_is_loaded(stages, monkeypatch):
    import qcsc_prefect_executor.from_blocks as from_blocks

    class Solver:
        hpc_profile_block_name = "hpc-fugaku"
        execution_profile_block_name = "exec-gpu"

        @classmethod
        async def load(cls, name):
            return cls()

    loaded = {
        "hpc-fugaku": type("H", (), {"hpc_target": "fugaku"})(),
        "exec-gpu": type("E", (), {"resource_class": "gpu", "gpus_per_node": 1})(),
    }

    async def fake_load(block_cls, name):
        return loaded[name]

    monkeypatch.setattr(from_blocks, "_load_block", fake_load)
    stages["Solver"] = Solver
    with pytest.raises(ValueError, match="takes no GPU request"):
        _run(stages, _parameters())
    assert stages["log"] == []


def test_the_solver_comes_from_the_advanced_block(stages):
    stages["advanced"]["settings"] = AdvancedSettings(
        solver_block_ref="fulqrum_solver_job/from-block", use_reset_mitigation=True
    )
    _run(stages, _parameters(solver=None, qrmi_result_dir="/jobs/abc"))
    assert stages["advanced"]["loaded"] == ["sqd-advanced-settings"]
    assert stages["loaded"] == ["from-block"]
    # A reload uses the block's reset mitigation.
    assert dict(stages["log"])["load_qrmi_result"]["use_reset_mitigation"] is True


def test_a_solver_block_override_replaces_the_blocks_solver(stages, caplog):
    stages["advanced"]["settings"] = AdvancedSettings(solver_block_ref="fulqrum_solver_job/x")
    with caplog.at_level(logging.INFO, logger=LOGGER.name):
        _run(stages, _parameters(solver="fulqrum_solver_job/override"))
    assert stages["loaded"] == ["override"]
    assert "Solver Block Override" in caplog.text


def test_the_advanced_transpiler_reaches_execute(stages):
    stages["advanced"]["settings"] = AdvancedSettings(sabre_layout_trials=7)
    _run(stages, _parameters())
    assert dict(stages["log"])["execute"]["advanced"].sabre_layout_trials == 7


def test_a_missing_advanced_block_falls_back_only_for_the_default_name(monkeypatch):
    from sqd_common import advanced

    async def missing(name):
        raise ValueError(f"Unable to find block document named {name}")

    monkeypatch.setattr(advanced.SQDAdvancedSettingsBlock, "aload", missing)
    logger = logging.getLogger(__name__)
    settings = asyncio.run(advanced.load_advanced_settings("sqd-advanced-settings", logger))
    assert settings == AdvancedSettings()
    with pytest.raises(ValueError, match="No Advanced SQD Settings block 'sqd-advanced-setings'"):
        asyncio.run(advanced.load_advanced_settings("sqd-advanced-setings", logger))
