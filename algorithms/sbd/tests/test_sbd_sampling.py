"""Synchronous QrmiRuntime sampling and the walker_sqd quantum_source paths."""

from __future__ import annotations

import asyncio
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from prefect.context import TagsContext
from qcsc_prefect.integrations.qiskit.qrmi import QrmiRuntime, QrmiRuntimeBlock
from qcsc_prefect.integrations.qiskit.qrmi import iqp as iqp_mod
from qcsc_prefect.integrations.qiskit.qrmi.compile import compile_pubs
from qiskit.primitives import PrimitiveResult, SamplerPubResult
from qiskit.primitives.containers import BitArray, DataBin
from sbd import sampling, sqd
from sbd.flow_params import CircuitParameters
from sbd.solver_job import SBDResult
from sbd.transpilation import TRANSPILER
from sbd_helpers import NORB, line_target, lucj_circuit, ucj_parameters

SHOTS = 8
OPTIONS = {"params": {"shots": SHOTS}}
WIDTH = 2 * NORB
CIRCUIT_PARAMS = CircuitParameters(
    optimization_level=1, sabre_max_iterations=1, sabre_swap_trials=1, sabre_layout_trials=2
)


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


def test_load_runtime_sets_the_sbd_hooks_and_keeps_the_scheduler(saved_runtime):
    runtime = sampling.load_runtime("my-runtime", CIRCUIT_PARAMS)
    assert saved_runtime == ["my-runtime"]
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


def test_sample_sync_runs_without_an_event_loop_and_tags_the_task(saved_runtime, fake_iqp):
    runtime = sampling.load_runtime("my-runtime", CIRCUIT_PARAMS)
    circuit = lucj_circuit(ucj_parameters(1)[0])
    with pytest.raises(RuntimeError):
        asyncio.get_running_loop()

    result = sampling.sample_sync(runtime, circuit, options=OPTIONS)

    assert result[0].data.meas.num_shots == SHOTS
    (call,) = fake_iqp.calls
    assert sampling.QUANTUM_TAG in call["tags"]
    # Sent untranspiled; prepare ran the ffsim pre-init stage before compilation.
    assert "ucj_balanced_jw" in circuit.count_ops()
    assert "ucj_balanced_jw" not in call["circuits"][0].count_ops()
    assert {reg.name for reg in call["isa"][0].cregs} == {"meas", "test"}


def test_sample_sync_from_threads_without_event_loops(saved_runtime, fake_iqp):
    # Walkers run in worker threads (ConcurrentTaskRunner) or processes. Each call runs
    # its own event loop and gets its own runtime call, so no QRMI object is shared.
    runtime = sampling.load_runtime("my-runtime", CIRCUIT_PARAMS)
    circuits = [lucj_circuit(parameter) for parameter in ucj_parameters(4)]
    barrier = threading.Barrier(4)

    def walker(circuit):
        barrier.wait(5)
        return sampling.sample_sync(runtime, circuit, options=OPTIONS)

    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(walker, circuits))

    assert [result[0].data.meas.num_shots for result in results] == [SHOTS] * 4
    assert len(fake_iqp.calls) == 4
    assert len({call["thread"] for call in fake_iqp.calls}) == 4
    assert all(sampling.QUANTUM_TAG in call["tags"] for call in fake_iqp.calls)


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
def test_sample_sync_rejects_unsupported_options_before_sampling(
    saved_runtime, fake_iqp, options, message
):
    runtime = sampling.load_runtime("my-runtime", CIRCUIT_PARAMS)
    circuit = lucj_circuit(ucj_parameters(1)[0])
    with pytest.raises((ValueError, TypeError), match=message):
        sampling.sample_sync(runtime, circuit, options=options)
    assert fake_iqp.calls == []


def test_sample_sync_passes_the_options_shots(saved_runtime, fake_iqp):
    runtime = sampling.load_runtime("my-runtime", CIRCUIT_PARAMS)
    circuit = lucj_circuit(ucj_parameters(1)[0])
    result = sampling.sample_sync(runtime, circuit, options={"params": {"shots": 5}})
    assert result[0].data.meas.num_shots == 5


def test_sample_sync_refuses_a_running_event_loop(saved_runtime, fake_iqp):
    runtime = sampling.load_runtime("my-runtime", CIRCUIT_PARAMS)
    circuit = lucj_circuit(ucj_parameters(1)[0])

    async def inside_loop():
        sampling.sample_sync(runtime, circuit, options=OPTIONS)

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
    monkeypatch.setattr(sqd.Variable, "get", staticmethod(lambda name, default=None: run.options))
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
            solver_block_name="solver",
            quantum_source=quantum_source,
            random_seed=random_seed,
        )

    run.solver = solver
    run.options = OPTIONS
    return run


def test_real_device_walker_samples_through_qrmi_runtime(walker, saved_runtime, monkeypatch):
    fake = FakeIqp(dirty=2)
    monkeypatch.setattr(iqp_mod, "sample_on_iqp", fake)

    (energy, carryover), telemetry = walker("real-device")

    assert saved_runtime == [sampling.RUNTIME_BLOCK_NAME]
    (call,) = fake.calls
    assert call["runtime"].transpiler == TRANSPILER
    assert sampling.QUANTUM_TAG in call["tags"]
    # Reset mitigation keeps the shots whose test register is all zeros.
    assert telemetry["shot_retention_rate"] == (SHOTS - 2) / SHOTS
    assert energy == -0.5


def test_real_device_walker_without_reset_mitigation(walker, saved_runtime, fake_iqp):
    _, telemetry = walker("real-device", use_reset_mitigation=False)
    (call,) = fake_iqp.calls
    assert [reg.name for reg in call["isa"][0].cregs] == ["meas"]
    assert telemetry["shot_retention_rate"] == 1.0


def test_missing_runtime_block_points_to_random_mode(walker, monkeypatch, fake_iqp):
    def load_runtime(name, prepare=None):
        raise ValueError(f"Unable to find block document named {name}")

    monkeypatch.setattr(QrmiRuntimeBlock, "load_runtime", load_runtime)
    with pytest.raises(RuntimeError, match="quantum_source='random'"):
        walker("real-device")
    assert fake_iqp.calls == []


@pytest.mark.parametrize(
    "options",
    [
        {"params": {"shots": SHOTS}, "twirling": {"enable_gates": True}},
        {"params": {"shots": 4.9}},
    ],
)
def test_real_device_walker_rejects_options_qrmi_runtime_ignores(
    walker, saved_runtime, fake_iqp, options
):
    # These options were accepted before; QrmiRuntime does not support them, so the
    # walker fails instead of silently dropping them or truncating the shots.
    walker.options = options
    with pytest.raises(ValueError):
        walker("real-device")
    assert fake_iqp.calls == []
    assert walker.solver.ci_strings == []


def test_random_walker_does_not_load_a_runtime(walker, monkeypatch, fake_iqp):
    def load_runtime(name, prepare=None):
        raise AssertionError("random mode must not load a runtime")

    sampled = []
    generate = sqd.generate_bit_array_uniform

    def recording_generate(**kwargs):
        bits = generate(**kwargs)
        sampled.append((kwargs, bits.get_bitstrings()))
        return bits

    monkeypatch.setattr(QrmiRuntimeBlock, "load_runtime", load_runtime)
    monkeypatch.setattr(sqd, "generate_bit_array_uniform", recording_generate)
    _, telemetry = walker("random", walker_index=1)
    walker("random", walker_index=1)

    assert fake_iqp.calls == []
    assert "shot_retention_rate" not in telemetry
    # Same seed rule as before: random_seed plus a pairing of trial and walker index.
    assert sampled[0][0] == {"num_samples": SHOTS, "num_bits": WIDTH, "rand_seed": 24 + 2}
    assert sampled[0][1] == sampled[1][1]


def test_qfw_walker_keeps_its_own_path(walker, monkeypatch, fake_iqp):
    # The qfw path is unchanged: QFwResource "qfw-runner" and run_sampler_job, with the
    # ffsim pre-init stage on the host and the QFw sampler script on the node.
    submitted = []

    class QFwResource:
        @classmethod
        def load(cls, name):
            submitted.append(("load", name))
            return SimpleNamespace(device_id="qfw_device")

    def run_sampler_job(circuit, *, shots, qpu, work_dir, sampler, sampler_args):
        submitted.append(("run", qpu, shots, sampler.rsplit("/", 1)[-1], sampler_args))
        test, meas = "0" * WIDTH, "1" * WIDTH
        return {"counts": {f"{test} {meas}": shots - 1, f"1{test[1:]} {meas}": 1}}

    def load_runtime(name, prepare=None):
        raise AssertionError("qfw must not load a QrmiRuntimeBlock")

    monkeypatch.setattr(QrmiRuntimeBlock, "load_runtime", load_runtime)
    monkeypatch.setitem(sys.modules, "sbd.qfw_resource", module("sbd.qfw_resource", QFwResource))
    monkeypatch.setitem(
        sys.modules, "qfw_slurm_sampler", module("qfw_slurm_sampler", run_sampler_job)
    )

    _, telemetry = walker("qfw")

    assert submitted == [
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
    assert telemetry["shot_retention_rate"] == (SHOTS - 1) / SHOTS
    assert fake_iqp.calls == []


def module(name, obj):
    mod = ModuleType(name)
    setattr(mod, obj.__name__, obj)
    return mod


def test_unknown_quantum_source_is_rejected(walker):
    with pytest.raises(ValueError, match="Unsupported quantum_source"):
        walker("other")
