"""Generic source Blocks work without importing an application or initializing hardware."""

import ipaddress
import socket

import pytest
from prefect.client.schemas.objects import BlockDocument
from prefect.testing.utilities import prefect_test_harness
from qcsc_prefect.integrations.qiskit.qrmi import QrmiRuntime
from qcsc_prefect.integrations.qiskit.qrmi import iqp as iqp_mod
from qcsc_prefect.integrations.qiskit.qrmi.config_file import API_KEY_SETTING
from qcsc_prefect.integrations.qiskit.sources import (
    QFwResource,
    QuantumSourceBlock,
    QuantumSourceConfig,
    factory,
)
from qcsc_prefect.integrations.qiskit.sources import qfw as qfw_mod
from qiskit import QuantumCircuit
from qiskit.primitives import PrimitiveResult, SamplerPubResult
from qiskit.primitives.containers import BitArray, DataBin


@pytest.mark.parametrize("kind", ["random", "qrmi", "qfw"])
def test_saved_source_document_restores_generic_settings(kind):
    settings = {"kind": kind}
    if kind != "random":
        settings["block_name"] = "any-platform"
    if kind == "qfw":
        settings.update(sampler="/shared/node.py", sampler_arg_names=(("level", "--level"),))
    original = QuantumSourceBlock(
        config=QuantumSourceConfig(**settings),
        num_bits=13,
        seed=7,
        transpiler="application.compiler:run" if kind == "qrmi" else None,
        transpile_config={"level": 2},
    )
    block_type = original._to_block_type()
    schema = original._to_block_schema(block_type_id=block_type.id)
    document = BlockDocument(
        name="application-samples",
        data=original.model_dump(mode="json"),
        block_schema_id=schema.id,
        block_schema=schema,
        block_type_id=block_type.id,
        block_type=block_type,
        block_type_name=block_type.name,
    )
    restored = QuantumSourceBlock._from_block_document(document)
    fields = set(QuantumSourceBlock.model_fields)
    assert restored.model_dump(mode="json", include=fields) == original.model_dump(
        mode="json", include=fields
    )
    assert restored.get_block_type_slug() == "quantum-source"


@pytest.mark.parametrize(
    "values,message",
    [
        (
            {
                "config": {
                    "kind": "qfw",
                    "block_name": "device",
                    "sampler": "node.py",
                    "sampler_args_names": [["level", "--level"]],
                }
            },
            "sampler_args_names",
        ),
        ({"config": {"kind": "random"}, "transpile_cfg": {}}, "transpile_cfg"),
        ({"config": {"kind": "qrmi"}}, "requires a block_name"),
        ({"config": {"kind": "qrmi", "block_name": "  "}}, "requires a block_name"),
        ({"config": {"kind": "qfw", "block_name": "device"}}, "requires a sampler"),
        ({"config": {"kind": "qfw", "block_name": "device", "sampler": " "}}, "requires a sampler"),
        (
            {
                "config": {"kind": "qfw", "block_name": "device", "sampler": "node.py"},
                "transpile_config": {"level": 2},
            },
            "no sampler_arg_names mapping: level",
        ),
        (
            {
                "config": {"kind": "qfw", "block_name": "device", "sampler": "node.py"},
                "transpiler": "application.compile:run",
            },
            "do not save a QRMI transpiler hook",
        ),
    ],
)
def test_source_block_rejects_incomplete_and_misspelled_settings(values, message):
    with pytest.raises(ValueError, match=message):
        QuantumSourceBlock.model_validate(values)


def test_missing_source_block_explains_how_to_create_or_select_one(monkeypatch):
    def missing(*args, **kwargs):
        raise ValueError("not saved")

    monkeypatch.setattr(QuantumSourceBlock, "load", missing)
    with pytest.raises(RuntimeError, match="QuantumSourceBlock.*save") as exc:
        QuantumSourceBlock.load_source("old-name")
    assert ".save('old-name')" in str(exc.value)
    assert "same Prefect workspace" in str(exc.value)
    assert str(exc.value.__cause__) == "not saved"


def test_two_applications_can_supply_different_random_policies_to_the_same_block():
    block = QuantumSourceBlock(config=QuantumSourceConfig("random"), num_bits=8, seed=1)
    first_app = block.get_source(num_bits=13, seed=5).sample(shots=7)
    second_app = block.get_source(num_bits=20, seed=9).sample(shots=11)
    defaults = block.get_source().sample(shots=3)
    assert (first_app[0].data.meas.num_bits, first_app[0].data.meas.num_shots) == (13, 7)
    assert (second_app[0].data.meas.num_bits, second_app[0].data.meas.num_shots) == (20, 11)
    assert defaults[0].data.meas.num_bits == 8
    assert (block.num_bits, block.seed) == (8, 1)


def test_source_block_compilation_defaults_can_be_replaced_for_an_application(monkeypatch):
    monkeypatch.setattr(
        factory.QrmiRuntimeBlock,
        "load_runtime",
        lambda name: QrmiRuntime(
            scheduler="iqp", resource="test", resource_config={"key": "value"}
        ),
    )
    block = QuantumSourceBlock(
        config=QuantumSourceConfig("qrmi", block_name="runtime"),
        transpiler="application.compiler:first",
        transpile_config={"level": 2, "trials": 5},
    )
    default = block.get_source()
    override = block.get_source(transpiler="another.compiler:run", transpile_config={"level": 1})
    assert default.runtime.transpiler == "application.compiler:first"
    assert default.runtime.transpile_config == {"level": 2, "trials": 5}
    assert override.runtime.transpiler == "another.compiler:run"
    assert override.runtime.transpile_config == {"level": 1}


def test_real_prefect_source_block_can_switch_platforms_under_the_same_name(tmp_path, monkeypatch):
    """Persist all three kinds through a local server; never contact a quantum backend."""
    monkeypatch.setenv("PREFECT_HOME", str(tmp_path / "prefect"))
    monkeypatch.setenv("PREFECT_SERVER_ANALYTICS_ENABLED", "false")
    monkeypatch.setenv("PREFECT_API_KEY", "")
    monkeypatch.setenv(API_KEY_SETTING, "test-secret")
    connect = socket.socket.connect
    connect_ex = socket.socket.connect_ex

    def local_only(sock, address):
        if sock.family != socket.AF_UNIX:
            host = address[0]
            assert host == "localhost" or ipaddress.ip_address(host).is_loopback, host

    def guarded_connect(sock, address):
        local_only(sock, address)
        return connect(sock, address)

    def guarded_connect_ex(sock, address):
        local_only(sock, address)
        return connect_ex(sock, address)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)

    async def sample_on_iqp(circuits, shots, registers, configs, runtime):
        data = {
            name: BitArray.from_samples(["0" * width] * shots, width)
            for name, width in registers[0].items()
        }
        return PrimitiveResult([SamplerPubResult(DataBin(**data), metadata={})], metadata={})

    monkeypatch.setattr(iqp_mod, "sample_on_iqp", sample_on_iqp)
    monkeypatch.setattr(
        qfw_mod,
        "_launcher",
        lambda: lambda circuit, **kwargs: {"counts": {"0" * 6: kwargs["shots"]}},
    )

    with prefect_test_harness():
        factory.QrmiRuntimeBlock(
            scheduler="iqp",
            resource="test_qrmi",
        ).save("application-runtime")
        QFwResource(device_id="test_qfw").save("application-resource")
        configs = [
            (QuantumSourceConfig("random"), "random", False),
            (QuantumSourceConfig("qrmi", block_name="application-runtime"), "test_qrmi", True),
            (
                QuantumSourceConfig("qfw", block_name="application-resource", sampler="node.py"),
                "test_qfw",
                True,
            ),
        ]
        for config, name, needs_circuit in configs:
            QuantumSourceBlock(config=config, num_bits=6, seed=3).save(
                "application-samples",
                overwrite=True,
            )
            source = QuantumSourceBlock.load_source("application-samples")
            assert source.backend_name == name
            assert source.requires_circuit is needs_circuit
            assert QuantumSourceBlock.load("application-samples").config.kind == config.kind
            circuit = None
            if source.requires_circuit:
                circuit = QuantumCircuit(6)
                circuit.measure_all()
            result = source.sample(circuit, shots=4)
            assert result[0].data.meas.num_bits == 6
            assert result[0].data.meas.num_shots == 4
