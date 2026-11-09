"""Synchronous QrmiRuntime sampling and the walker_sqd quantum_source paths."""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from prefect.context import TagsContext
from qcsc_prefect.integrations.qiskit.qrmi import QrmiRuntime, QrmiRuntimeBlock
from qcsc_prefect.integrations.qiskit.qrmi import iqp as iqp_mod
from qcsc_prefect.integrations.qiskit.qrmi.compile import compile_pubs
from qcsc_prefect.integrations.qiskit.sources import (
    QFwResource,
    QuantumSourceBlock,
    QuantumSourceConfig,
    RandomSource,
)
from qcsc_prefect.integrations.qiskit.sources import qfw as qfw_mod
from qcsc_prefect_sbd import SBDResult
from qiskit.primitives import PrimitiveResult, SamplerPubResult
from qiskit.primitives.containers import BitArray, DataBin
from qiskit_addon_sqd.counts import generate_bit_array_uniform
from sbd import sampling, sqd
from sbd.flow_params import CircuitParameters
from sbd.transpilation import TRANSPILER
from sbd_helpers import NORB, line_target, lucj_circuit, ucj_parameters

SHOTS = 8
UNSET = object()  # no saved sqd_options Variable
OPTIONS = {"params": {"shots": SHOTS}}
WIDTH = 2 * NORB
CIRCUIT_PARAMS = CircuitParameters(
    optimization_level=1, sabre_max_iterations=1, sabre_swap_trials=1, sabre_layout_trials=2
)
SOURCE_ARGS = dict(
    circuit_params=CIRCUIT_PARAMS, num_orbitals=NORB, random_seed=24, trial_index=0, walker_index=1
)
RUNTIME_BLOCK_NAME = "qrmi-runtime"
QUANTUM_TAG = "res: quantum"


@pytest.fixture(autouse=True)
def source_blocks(monkeypatch):
    """Replace only persisted source settings; algorithm code knows none of these kinds."""
    from pathlib import Path

    blocks = {
        "random": QuantumSourceBlock(config=QuantumSourceConfig("random")),
        "real-device": QuantumSourceBlock(
            config=QuantumSourceConfig("qrmi", block_name=RUNTIME_BLOCK_NAME, tags=(QUANTUM_TAG,))
        ),
        "qfw": QuantumSourceBlock(
            config=QuantumSourceConfig(
                "qfw",
                block_name="qfw-runner",
                sampler=str(Path(sampling.__file__).with_name("qfw_slurm_sqd_sampler.py")),
                sampler_arg_names=(
                    ("optimization_level", "--optimization-level"),
                    ("sabre_max_iterations", "--sabre-max-iterations"),
                    ("sabre_swap_trials", "--sabre-swap-trials"),
                    ("sabre_layout_trials", "--sabre-layout-trials"),
                ),
            )
        ),
    }

    def load(cls, name, **kwargs):
        if name not in blocks:
            raise ValueError(f"Unable to find source block {name}")
        return blocks[name]

    monkeypatch.setattr(QuantumSourceBlock, "load", classmethod(load))
    return blocks


class FakeIqp:
    """Replaces the IQP sampling task: compiles with the runtime's hooks, returns fixed bits.

    ``meas`` is all ones and ``test`` is all zeros except for ``dirty`` shots, so that
    reset mitigation keeps ``SHOTS - dirty`` shots.
    """

    def __init__(self, dirty=0):
        self.target = line_target()
        self.dirty = dirty
        self.calls = []
        self.lock = threading.Lock()

    async def __call__(self, circuits, shots, registers, configs, runtime):
        isa, _ = compile_pubs(circuits, configs, registers, self.target, runtime.transpiler)
        with self.lock:
            self.calls.append(
                {
                    "circuits": circuits,
                    "configs": configs,
                    "runtime": runtime,
                    "isa": isa,
                    "tags": set(TagsContext.get().current_tags),
                    "thread": threading.current_thread().name,
                }
            )
        data = {"meas": BitArray.from_samples(["1" * WIDTH] * shots, WIDTH)}
        if "test" in registers[0]:
            test = ["0" * WIDTH] * (shots - self.dirty) + ["1" + "0" * (WIDTH - 1)] * self.dirty
            data["test"] = BitArray.from_samples(test, WIDTH)
        return PrimitiveResult([SamplerPubResult(DataBin(**data), metadata={})], metadata={})


@pytest.fixture
def fake_iqp(monkeypatch):
    fake = FakeIqp()
    monkeypatch.setattr(iqp_mod, "sample_on_iqp", fake)
    return fake


def iqp_runtime(**changes):
    runtime = QrmiRuntime(
        scheduler="iqp",
        resource="ibm_test",
        resource_config={"QRMI_IBM_QCS_IAM_APIKEY": "secret"},
        transpile_config={"optimization_level": 0},
    )
    return runtime


@pytest.fixture
def saved_runtime(monkeypatch):
    loaded = []

    def load_runtime(name, prepare=None):
        loaded.append(name)
        return iqp_runtime()

    monkeypatch.setattr(QrmiRuntimeBlock, "load_runtime", load_runtime)
    return loaded


def test_source_factory_sets_the_sbd_hooks_and_keeps_the_scheduler(saved_runtime):
    runtime = sampling.load_quantum_source("real-device", **SOURCE_ARGS).runtime
    assert saved_runtime == [RUNTIME_BLOCK_NAME]
    assert runtime.scheduler == "iqp"
    assert runtime.resource == "ibm_test"
    assert runtime.transpiler == TRANSPILER
    assert runtime.prepare is sampling.prepare
    # The block's transpile_config is replaced, not merged.
    assert runtime.transpile_config == {
        "optimization_level": 1,
        "sabre_max_iterations": 1,
        "sabre_swap_trials": 1,
        "sabre_layout_trials": 2,
    }


def qrmi_source():
    return sampling.load_quantum_source("real-device", **SOURCE_ARGS)


def test_real_device_source_runs_without_an_event_loop_and_tags_the_task(saved_runtime, fake_iqp):
    source = sampling.load_quantum_source("real-device", **SOURCE_ARGS)
    circuit = lucj_circuit(ucj_parameters(1)[0])
    with pytest.raises(RuntimeError):
        asyncio.get_running_loop()

    result = source.sample(circuit, shots=SHOTS)

    assert saved_runtime == [RUNTIME_BLOCK_NAME]
    assert source.backend_name == "ibm_test" and source.requires_circuit
    assert result[0].data.meas.num_shots == SHOTS
    (call,) = fake_iqp.calls
    assert QUANTUM_TAG in call["tags"]
    # Sent untranspiled; prepare ran the ffsim pre-init stage before compilation.
    assert "ucj_balanced_jw" in circuit.count_ops()
    assert "ucj_balanced_jw" not in call["circuits"][0].count_ops()
    assert {reg.name for reg in call["isa"][0].cregs} == {"meas", "test"}


def test_qrmi_source_from_threads_without_event_loops(saved_runtime, fake_iqp):
    # Walkers run in worker threads (ConcurrentTaskRunner) or processes. Each call runs
    # its own event loop and gets its own runtime call, so no QRMI object is shared.
    source = qrmi_source()
    circuits = [lucj_circuit(parameter) for parameter in ucj_parameters(4)]
    barrier = threading.Barrier(4)

    def walker(circuit):
        barrier.wait(5)
        return source.sample(circuit, shots=SHOTS)

    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(walker, circuits))

    assert [result[0].data.meas.num_shots for result in results] == [SHOTS] * 4
    assert len(fake_iqp.calls) == 4
    assert len({call["thread"] for call in fake_iqp.calls}) == 4
    assert all(QUANTUM_TAG in call["tags"] for call in fake_iqp.calls)


@pytest.mark.parametrize("shots", [0, -1, 4.9, "8", True, None])
def test_qrmi_source_rejects_invalid_shots_before_sampling(saved_runtime, fake_iqp, shots):
    with pytest.raises(ValueError, match="positive integer"):
        qrmi_source().sample(lucj_circuit(ucj_parameters(1)[0]), shots=shots)
    assert fake_iqp.calls == []


def test_qrmi_source_passes_the_shots(saved_runtime, fake_iqp):
    result = qrmi_source().sample(lucj_circuit(ucj_parameters(1)[0]), shots=5)
    assert result[0].data.meas.num_shots == 5


def test_qrmi_source_refuses_a_running_event_loop(saved_runtime, fake_iqp):
    source = qrmi_source()
    circuit = lucj_circuit(ucj_parameters(1)[0])

    async def inside_loop():
        source.sample(circuit, shots=SHOTS)

    with pytest.raises(RuntimeError):
        asyncio.run(inside_loop())
    assert fake_iqp.calls == []


class FakeSolver:
    def __init__(self):
        self.ci_strings = []

    async def run(self, ci_strings, **kwargs):
        self.ci_strings.append(ci_strings)
        occupancy = np.full(NORB, 0.5)
        return SBDResult(
            energy=-1.0,
            orbital_occupancies=(occupancy, occupancy),
            carryover_bitstrings=np.zeros((0, NORB), dtype=bool),
        )


@pytest.fixture
def walker(monkeypatch, tmp_path):
    """Run walker_sqd with its Prefect calls, solver and storage replaced."""
    solver = FakeSolver()
    monkeypatch.setattr(
        sqd,
        "get_run_logger",
        lambda: SimpleNamespace(
            info=lambda *a: None, debug=lambda *a: None, exception=lambda *a: None
        ),
    )
    monkeypatch.setattr(sqd.SBDSolverJob, "load", classmethod(lambda cls, name: solver))

    # Like Variable.get: the stored value, even None, or default when there is none.
    def get(name, default=None):
        assert name == "sqd_options"
        return default if run.options is UNSET else run.options

    monkeypatch.setattr(sqd.Variable, "get", staticmethod(get))
    monkeypatch.setattr(sqd, "save_ndarray", lambda **arrays: f"file://{tmp_path}/data.npz")
    for name in ("create_lucj_circuit", "recover_configurations", "postselect_bitstrings"):
        monkeypatch.setattr(sqd, name, getattr(sqd, name).fn)
    monkeypatch.setattr(sqd, "subsample_close_shell", sqd.subsample_close_shell.fn)

    props = SimpleNamespace(
        num_orbitals=NORB,
        num_electrons=(2, 2),
        open_shell=False,
        initial_occupancy=(np.full(NORB, 0.5), np.full(NORB, 0.5)),
        one_body_tensor=np.zeros((NORB, NORB)),
        two_body_tensor=np.zeros((NORB,) * 4),
        nuclear_repulsion_energy=0.5,
    )

    def run(quantum_source, *, use_reset_mitigation=True, random_seed=24, walker_index=0):
        return sqd.walker_sqd.fn(
            trial_index=0,
            walker_index=walker_index,
            ucj_parameter=ucj_parameters(1)[0],
            circuit_params=CIRCUIT_PARAMS.model_copy(
                update={"use_reset_mitigation": use_reset_mitigation}
            ),
            elec_props=props,
            aa_indices=[(p, p + 1) for p in range(NORB - 1)],
            ab_indices=[(0, 0)],
            carryover=np.zeros((0, NORB), dtype=bool),
            sqd_dim=16,
            solver_block_ref=run.solver_block_ref,
            quantum_source=quantum_source,
            random_seed=random_seed,
        )

    run.solver = solver
    run.options = OPTIONS
    run.solver_block_ref = "sbd_solver_job/solver"
    return run


def test_real_device_walker_samples_through_qrmi_runtime(walker, saved_runtime, monkeypatch):
    fake = FakeIqp(dirty=2)
    monkeypatch.setattr(iqp_mod, "sample_on_iqp", fake)

    (energy, carryover), telemetry = walker("real-device")

    assert saved_runtime == [RUNTIME_BLOCK_NAME]
    (call,) = fake.calls
    assert call["runtime"].transpiler == TRANSPILER
    assert QUANTUM_TAG in call["tags"]
    # Reset mitigation keeps the shots whose test register is all zeros.
    assert telemetry["shot_retention_rate"] == (SHOTS - 2) / SHOTS
    assert energy == -0.5


def test_real_device_walker_without_reset_mitigation(walker, saved_runtime, fake_iqp):
    _, telemetry = walker("real-device", use_reset_mitigation=False)
    (call,) = fake_iqp.calls
    assert [reg.name for reg in call["isa"][0].cregs] == ["meas"]
    assert telemetry["shot_retention_rate"] == 1.0


def test_missing_runtime_block_reports_the_saved_reference(walker, monkeypatch, fake_iqp):
    def load_runtime(name, prepare=None):
        raise ValueError(f"Unable to find block document named {name}")

    monkeypatch.setattr(QrmiRuntimeBlock, "load_runtime", load_runtime)
    with pytest.raises(RuntimeError, match="QrmiRuntimeBlock 'qrmi-runtime'"):
        walker("real-device")
    assert fake_iqp.calls == []


@pytest.mark.parametrize(
    "options,message",
    [
        ({"params": {"shots": SHOTS}, "twirling": {"enable_gates": True}}, "Only options"),
        ({"params": {"shots": SHOTS, "seed": 1}}, "Only options"),
        ({"params": {"shots": 4.9}}, "positive integer"),
        ({"params": {"shots": "8"}}, "positive integer"),
        ({"params": {"shots": 0}}, "positive integer"),
        ({"params": {"shots": True}}, "positive integer"),
        ({"params": "8"}, "must be a mapping"),
    ],
)
def test_real_device_walker_rejects_options_qrmi_runtime_ignores(
    walker, saved_runtime, fake_iqp, options, message
):
    # These options were accepted before; QrmiRuntime does not support them, so the
    # walker fails instead of silently dropping them or truncating the shots.
    walker.options = options
    with pytest.raises((ValueError, TypeError), match=message):
        walker("real-device")
    assert fake_iqp.calls == []
    assert walker.solver.ci_strings == []


def test_random_walker_does_not_load_a_runtime_or_build_a_circuit(walker, monkeypatch, fake_iqp):
    def load_runtime(name, prepare=None):
        raise AssertionError("random mode must not load a runtime")

    def create_lucj_circuit(**kwargs):
        raise AssertionError("random mode must not build a circuit")

    sampled = []
    sample = RandomSource.sample

    def recording_sample(self, circuit=None, *, shots):
        result = sample(self, circuit, shots=shots)
        sampled.append((self.num_bits, self.seed, shots, result[0].data.meas))
        return result

    monkeypatch.setattr(QrmiRuntimeBlock, "load_runtime", load_runtime)
    monkeypatch.setattr(QFwResource, "load", classmethod(lambda cls, name: load_runtime(name)))
    monkeypatch.setattr(sqd, "create_lucj_circuit", create_lucj_circuit)
    monkeypatch.setattr(RandomSource, "sample", recording_sample)
    _, telemetry = walker("random", walker_index=1)
    walker("random", walker_index=1)

    assert fake_iqp.calls == []
    assert "shot_retention_rate" not in telemetry
    # Same seed rule as before: random_seed plus a pairing of trial and walker index.
    assert sampled[0][:3] == (WIDTH, 24 + 2, SHOTS)
    assert sampled[0][3] == sampled[1][3]
    # Same bits as the former generate_bit_array_uniform call.
    assert sampled[0][3] == generate_bit_array_uniform(
        num_samples=SHOTS, num_bits=WIDTH, rand_seed=24 + 2
    )


def test_qfw_walker_samples_through_qfw_source(walker, monkeypatch, fake_iqp):
    # QFwResource "qfw-runner" and run_sampler_job, with the ffsim pre-init stage on
    # the host and the QFw sampler script on the node, as before.
    submitted = []

    def load(cls, name):
        submitted.append(("load", name))
        return QFwResource(device_id="qfw_device")

    def run_sampler_job(circuit, *, shots, qpu, work_dir, sampler, sampler_args):
        submitted.append(("run", qpu, shots, sampler.rsplit("/", 1)[-1], sampler_args))
        submitted.append(("ops", set(circuit.count_ops())))
        test, meas = "0" * WIDTH, "1" * WIDTH
        return {"counts": {f"{test} {meas}": shots - 1, f"1{test[1:]} {meas}": 1}, "job_id": "j1"}

    def load_runtime(name, prepare=None):
        raise AssertionError("qfw must not load a QrmiRuntimeBlock")

    monkeypatch.setattr(QrmiRuntimeBlock, "load_runtime", load_runtime)
    monkeypatch.setattr(QFwResource, "load", classmethod(load))
    monkeypatch.setattr(qfw_mod, "_launcher", lambda: run_sampler_job)

    _, telemetry = walker("qfw")

    assert submitted[:2] == [
        ("load", "qfw-runner"),
        (
            "run",
            "qfw_device",
            SHOTS,
            "qfw_slurm_sqd_sampler.py",
            [
                "--optimization-level",
                1,
                "--sabre-max-iterations",
                1,
                "--sabre-swap-trials",
                1,
                "--sabre-layout-trials",
                2,
            ],
        ),
    ]
    # The ffsim pre-init stage ran before serialization.
    assert "ucj_balanced_jw" not in submitted[2][1]
    assert telemetry["shot_retention_rate"] == (SHOTS - 1) / SHOTS
    assert fake_iqp.calls == []


def module(name, obj):
    mod = ModuleType(name)
    setattr(mod, obj.__name__, obj)
    return mod


def test_missing_quantum_source_block_is_rejected(walker):
    with pytest.raises(RuntimeError, match="QuantumSourceBlock 'other'"):
        walker("other")


def test_walker_accepts_an_arbitrary_source_block_name(walker, source_blocks):
    source_blocks["configured"] = QuantumSourceBlock(config=QuantumSourceConfig("random"))
    (energy, _), telemetry = walker("configured")
    assert energy == -0.5
    assert "shot_retention_rate" not in telemetry


def test_same_block_name_switches_the_walker_between_all_platforms(
    walker, source_blocks, saved_runtime, fake_iqp, monkeypatch
):
    monkeypatch.setattr(QFwResource, "load", lambda name: QFwResource(device_id="test_qfw"))
    monkeypatch.setattr(
        qfw_mod,
        "_launcher",
        lambda: (
            lambda circuit, **kwargs: {"counts": {f"{'0' * WIDTH} {'1' * WIDTH}": kwargs["shots"]}}
        ),
    )
    for name in ("random", "real-device", "qfw"):
        source_blocks["shared-name"] = source_blocks[name]
        (energy, _), telemetry = walker("shared-name")
        assert energy == -0.5
        assert ("shot_retention_rate" in telemetry) is (name != "random")
    assert len(fake_iqp.calls) == 1
    assert saved_runtime == [RUNTIME_BLOCK_NAME]


def test_walker_records_sbd_with_carryover(walker):
    _, telemetry = walker("random")
    assert (telemetry["solver"], telemetry["carryover"]) == ("SBD", "sbd")


def test_walker_with_dice_returns_an_empty_carryover(walker, monkeypatch):
    from qcsc_prefect_dice import DiceSHCISolverJob
    from qiskit_addon_sqd.fermion import SCIResult

    loaded = []
    inputs = []

    class FakeDice:
        async def run(self, **kwargs):
            inputs.append(kwargs)
            occupancy = np.full(NORB, 0.5)
            return SCIResult(
                energy=-2.0, sci_state=None, orbital_occupancies=(occupancy, occupancy)
            )

    def load(cls, name):
        loaded.append(name)
        return FakeDice()

    monkeypatch.setattr(DiceSHCISolverJob, "load", classmethod(load))
    walker.solver_block_ref = "dice_shci_solver_job/dice"

    (energy, carryover), telemetry = walker("random")

    assert loaded == ["dice"]
    assert walker.solver.ci_strings == []  # the SBD fake is not used
    # DICE gets the same subspace as SBD would: one set, Hartree-Fock first.
    (call,) = inputs
    alpha, beta = call["ci_strings"]
    assert np.array_equal(alpha, beta) and alpha[0] == 0b11
    assert "spin_sq" not in call
    assert energy == -1.5
    assert carryover.shape == (0, NORB) and carryover.dtype == bool
    assert (telemetry["solver"], telemetry["carryover"]) == ("DICE", "none")


def test_walker_rejects_an_unknown_solver_slug(walker):
    walker.solver_block_ref = "other_solver/x"
    with pytest.raises(ValueError, match="solver_block_ref"):
        walker("random")


# --- sqd_options: the same rules for every quantum_source -----------------------------


@pytest.mark.parametrize(
    "options,shots",
    [
        (UNSET, 100_000),
        ({}, 100_000),
        ({"params": {}}, 100_000),
        ({"params": {"shots": 5}}, 5),
        ({"params": {"shots": np.int64(7)}}, 7),
    ],
)
def test_sqd_options_give_the_shots(options, shots):
    value = {} if options is UNSET else options
    assert sqd.resolve_sqd_shots(value) == shots
    assert type(sqd.resolve_sqd_shots(value)) is int


@pytest.mark.parametrize(
    "options,error,message",
    [
        ({"params": {"shots": None}}, ValueError, "positive integer"),
        ({"params": {"shots": 4.9}}, ValueError, "positive integer"),
        ({"params": {"shots": 0}}, ValueError, "positive integer"),
        ({"params": {"shots": -3}}, ValueError, "positive integer"),
        ({"params": {"shots": True}}, ValueError, "positive integer"),
        ({"params": {"shots": "8"}}, ValueError, "positive integer"),
        (None, TypeError, "sqd_options must be a mapping"),
        ([], TypeError, "sqd_options must be a mapping"),
        ("x", TypeError, "sqd_options must be a mapping"),
        ({"params": None}, TypeError, "must be a mapping"),
        ({"params": []}, TypeError, "must be a mapping"),
        ({"params": {"shots": SHOTS, "seed": 1}}, ValueError, "Only options"),
        (
            {"params": {"shots": SHOTS}, "twirling": {"enable_gates": True}},
            ValueError,
            "Only options",
        ),
        ({"twirling": {}}, ValueError, "Only options"),
    ],
)
@pytest.mark.parametrize("quantum_source", ["random", "real-device", "qfw"])
def test_every_source_rejects_bad_sqd_options_before_loading_blocks(
    walker, monkeypatch, fake_iqp, quantum_source, options, error, message
):
    loaded = []
    monkeypatch.setattr(
        sqd.SBDSolverJob, "load", classmethod(lambda cls, name: loaded.append(name))
    )
    monkeypatch.setattr(sqd, "load_quantum_source", lambda *a, **k: loaded.append("source"))
    walker.options = options
    with pytest.raises(error, match=message):
        walker(quantum_source)
    assert loaded == []  # neither the solver block nor the quantum source
    assert fake_iqp.calls == []


@pytest.mark.parametrize("quantum_source", ["random", "real-device", "qfw"])
def test_every_source_samples_the_default_shots_when_omitted(
    walker, monkeypatch, saved_runtime, fake_iqp, quantum_source
):
    requested = []

    class Source:
        backend_name = "fake"
        requires_circuit = False

        def sample(self, circuit=None, *, shots):
            requested.append(shots)
            return RandomSource(num_bits=WIDTH, seed=1).sample(shots=SHOTS)

    monkeypatch.setattr(sqd, "load_quantum_source", lambda *a, **k: Source())
    for options in (UNSET, {}, {"params": {}}):
        walker.options = options
        walker(quantum_source)
    assert requested == [100_000] * 3
