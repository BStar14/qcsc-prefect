"""Source construction from worker-safe settings, independent of any algorithm."""

import json
from dataclasses import asdict

import pytest
from qcsc_prefect.integrations.qiskit.qrmi import PreparedCircuit, QrmiRuntime, QrmiRuntimeBlock
from qcsc_prefect.integrations.qiskit.sources import (
    QFwResource,
    QuantumSourceConfig,
    SourceLoadError,
    load_quantum_source,
)
from qcsc_prefect.integrations.qiskit.sources import qfw as qfw_mod
from qiskit import QuantumCircuit


def test_random_config_can_cross_a_worker_boundary_without_loading_blocks(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("random source must not load a block or prepare a circuit")

    monkeypatch.setattr(QrmiRuntimeBlock, "load_runtime", forbidden)
    monkeypatch.setattr(QFwResource, "load", forbidden)
    config = QuantumSourceConfig(**json.loads(json.dumps(asdict(QuantumSourceConfig("random")))))
    source = load_quantum_source(config, num_bits=13, seed=42, prepare=forbidden)
    result = source.sample(shots=10)
    assert source.backend_name == "random" and not source.requires_circuit
    assert result[0].data.meas.num_bits == 13
    assert result[0].data.meas == source.sample(shots=10)[0].data.meas


@pytest.fixture(params=["iqp", "slurm"])
def saved_runtime(request, monkeypatch):
    if request.param == "iqp":
        settings = dict(scheduler="iqp", resource_config={"api_key": "secret"})
    else:
        settings = dict(
            scheduler="slurm",
            root_dir="/shared/jobs",
            command_block_name="cmd",
            execution_profile_block_name="exec",
            hpc_profile_block_name="hpc",
        )
    runtime = QrmiRuntime(resource="test_device", **settings)

    def load(name):
        assert name == "application-runtime"
        return runtime

    monkeypatch.setattr(QrmiRuntimeBlock, "load_runtime", load)
    return runtime


def test_qrmi_factory_preserves_policy_unless_the_application_overrides_it(saved_runtime):
    def prepare(circuit):
        return PreparedCircuit(circuit)

    saved_runtime.prepare = prepare
    saved_runtime.transpile_config = {"optimization_level": 0}
    config = QuantumSourceConfig("qrmi", block_name="application-runtime", tags=("app:qpu",))
    source = load_quantum_source(config)
    assert source.runtime == saved_runtime
    assert source.runtime.prepare is prepare
    assert source.tags == ("app:qpu",)

    replacement = {"optimization_level": 2}
    overridden = load_quantum_source(
        config,
        prepare=prepare,
        transpiler="application.compile:transpile",
        transpile_config=replacement,
    )
    replacement["optimization_level"] = 3
    assert overridden.runtime.scheduler == saved_runtime.scheduler
    assert overridden.runtime.resource_config == saved_runtime.resource_config
    assert overridden.runtime.transpiler == "application.compile:transpile"
    assert overridden.runtime.transpile_config == {"optimization_level": 2}
    assert saved_runtime.transpile_config == {"optimization_level": 0}
    assert saved_runtime.transpiler != overridden.runtime.transpiler


def test_qfw_factory_loads_custom_block_and_passes_prepared_circuit_and_cli_policy(monkeypatch):
    calls = []

    def load(cls, name):
        calls.append(("load", name))
        return QFwResource(device_id="custom_device")

    def prepare(circuit):
        calls.append(("prepare", circuit))
        circuit.x(0)
        return PreparedCircuit(circuit)

    def run(circuit, **kwargs):
        calls.append(("run", kwargs))
        assert circuit.count_ops()["x"] == 1
        return {"counts": {"00": kwargs["shots"]}, "job_id": "qfw-job"}

    monkeypatch.setattr(QFwResource, "load", classmethod(load))
    monkeypatch.setattr(QrmiRuntimeBlock, "load_runtime", lambda *a: pytest.fail("wrong block"))
    monkeypatch.setattr(qfw_mod, "_launcher", lambda: run)
    config = QuantumSourceConfig(
        "qfw",
        block_name="application-qfw",
        sampler="/shared/node.py",
        sampler_args=("--verbose",),
        sampler_arg_names=(("level", "--level"), ("trials", "--trials")),
        work_dir="/shared/qfw",
    )
    config = QuantumSourceConfig(**json.loads(json.dumps(asdict(config))))
    source = load_quantum_source(
        config, prepare=prepare, transpile_config={"level": 2, "trials": 5}
    )
    assert calls == [("load", "application-qfw")]  # constructing does not submit
    circuit = QuantumCircuit(2)
    circuit.measure_all()
    result = source.sample(circuit, shots=4)
    assert "x" not in circuit.count_ops()
    assert [call[0] for call in calls] == ["load", "prepare", "run"]
    assert calls[-1][1] == dict(
        shots=4,
        qpu="custom_device",
        work_dir="/shared/qfw",
        sampler="/shared/node.py",
        sampler_args=["--verbose", "--level", 2, "--trials", 5],
    )
    assert result[0].data.meas.get_counts() == {"00": 4}
    assert result.metadata["qcsc_qfw"]["job_id"] == "qfw-job"


@pytest.mark.parametrize("kind,block", [("qrmi", QrmiRuntimeBlock), ("qfw", QFwResource)])
def test_missing_blocks_raise_source_load_error_with_the_original_cause(monkeypatch, kind, block):
    def missing(*args, **kwargs):
        raise ValueError("block was not saved")

    monkeypatch.setattr(block, "load_runtime" if kind == "qrmi" else "load", missing)
    with pytest.raises(SourceLoadError, match="missing-block") as exc:
        load_quantum_source(
            QuantumSourceConfig(kind, block_name="missing-block", sampler="node.py")
        )
    assert str(exc.value.__cause__) == "block was not saved"


@pytest.mark.parametrize(
    "config,message",
    [
        ({"kind": "random"}, "random requires num_bits"),
        ({"kind": "unknown"}, "Input should be"),
        ({"kind": "qrmi"}, "requires a block_name"),
        ({"kind": "qfw", "block_name": "resource"}, "requires a sampler"),
        (
            dict(
                kind="qfw",
                block_name="resource",
                sampler="node.py",
                sampler_arg_names=(("level", "--level"),),
            ),
            "Missing compilation",
        ),
    ],
)
def test_invalid_settings_fail_before_loading_a_block(monkeypatch, config, message):
    monkeypatch.setattr(QrmiRuntimeBlock, "load_runtime", lambda *a: pytest.fail("loaded a block"))
    monkeypatch.setattr(QFwResource, "load", lambda *a: pytest.fail("loaded a block"))
    with pytest.raises(ValueError, match=message):
        load_quantum_source(QuantumSourceConfig(**config))


def test_qfw_rejects_unmapped_compilation_settings_before_loading_any_block(monkeypatch):
    monkeypatch.setattr(QFwResource, "load", lambda *a: pytest.fail("loaded a block"))
    config = QuantumSourceConfig(
        "qfw",
        block_name="resource",
        sampler="node.py",
        sampler_arg_names=(("level", "--level"),),
    )
    with pytest.raises(ValueError, match="no sampler_arg_names mapping: trials"):
        load_quantum_source(config, transpile_config={"level": 2, "trials": 999})


@pytest.mark.parametrize("argument", ["--level", "--level=2"])
def test_config_rejects_duplicate_fixed_and_mapped_flags(argument):
    with pytest.raises(ValueError, match="duplicates mapped CLI flags: --level"):
        QuantumSourceConfig(
            "qfw",
            block_name="resource",
            sampler="node.py",
            sampler_args=(argument,),
            sampler_arg_names=(("level", "--level"),),
        )


@pytest.mark.parametrize(
    "mapping",
    [(("level", "--level"), ("level", "--another")), (("level", "--level"), ("other", "--level"))],
)
def test_config_rejects_duplicate_mapping_entries(mapping):
    with pytest.raises(ValueError, match="must not repeat"):
        QuantumSourceConfig(
            "qfw", block_name="resource", sampler="node.py", sampler_arg_names=mapping
        )


def test_qfw_does_not_silently_discard_per_circuit_compilation_settings(monkeypatch):
    monkeypatch.setattr(QFwResource, "load", lambda *a: QFwResource(device_id="test"))
    monkeypatch.setattr(qfw_mod, "_launcher", lambda: pytest.fail("submitted a job"))
    source = load_quantum_source(
        QuantumSourceConfig("qfw", block_name="resource", sampler="node.py"),
        prepare=lambda circuit: PreparedCircuit(circuit, config={"level": 2}),
    )
    circuit = QuantumCircuit(1)
    circuit.measure_all()
    with pytest.raises(ValueError, match="per-circuit compilation settings"):
        source.sample(circuit, shots=1)
