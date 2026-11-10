from examples.prefect_bitcount_demo.quantum_sampling import (
    generate_random_bitstrings,
    resolve_shots,
)


def test_resolve_shots_reads_sampler_option_value():
    shots = resolve_shots(
        sampler_options={"params": {"shots": 321}},
        default_shots=100_000,
    )

    assert shots == 321


def test_generate_random_bitstrings_is_reproducible():
    first = generate_random_bitstrings(bitlen=4, shots=5, seed=24)
    second = generate_random_bitstrings(bitlen=4, shots=5, seed=24)

    assert first == second
    assert len(first) == 5
    assert all(len(bits) == 4 for bits in first)


def test_generate_random_bitstrings_accepts_zero_shots():
    assert generate_random_bitstrings(bitlen=4, shots=0, seed=24) == []


def _sample(quantum_source, sampler_options, **kwargs):
    import asyncio

    from examples.prefect_bitcount_demo.quantum_sampling import sample_bitstrings

    return asyncio.run(
        sample_bitstrings(
            quantum_source=quantum_source,
            runtime_block_name="my-runtime",
            sampler_options=sampler_options,
            bitlen=kwargs.get("bitlen", 3),
            default_shots=100_000,
            random_seed=24,
        )
    )


def _fake_qrmi(monkeypatch, *, block_transpiler=None):
    """Replace the runtime block and the IQP sampling task; return the recorded calls."""
    from qcsc_prefect.integrations.qiskit.qrmi import QrmiRuntime, QrmiRuntimeBlock
    from qcsc_prefect.integrations.qiskit.qrmi import iqp as iqp_mod
    from qcsc_prefect.integrations.qiskit.qrmi.compile import compile_pubs
    from qiskit.primitives import PrimitiveResult, SamplerPubResult
    from qiskit.primitives.containers import BitArray, DataBin
    from qiskit.providers.fake_provider import GenericBackendV2

    target = GenericBackendV2(num_qubits=5, seed=7).target
    calls = {"loaded": [], "sampled": []}

    async def aload_runtime(name, prepare=None):
        calls["loaded"].append(name)
        extra = {"transpiler": block_transpiler} if block_transpiler else {}
        return QrmiRuntime(
            scheduler="iqp",
            resource="ibm_test",
            resource_config={"QRMI_IBM_QCS_IAM_APIKEY": "secret"},
            transpile_config={"optimization_level": 0},
            **extra,
        )

    async def sample_on_iqp(circuits, shots, registers, configs, runtime):
        isa, _ = compile_pubs(circuits, configs, registers, target, runtime.transpiler)
        calls["sampled"].append({"circuits": circuits, "configs": configs, "isa": isa})
        width = registers[0]["meas"]
        bits = BitArray.from_samples(["0" * width, "1" * width] * (shots // 2), width)
        return PrimitiveResult([SamplerPubResult(DataBin(meas=bits), metadata={})], metadata={})

    monkeypatch.setattr(QrmiRuntimeBlock, "aload_runtime", aload_runtime)
    monkeypatch.setattr(iqp_mod, "sample_on_iqp", sample_on_iqp)
    calls["target"] = target
    return calls


def test_real_device_samples_the_ghz_circuit_through_qrmi_runtime(monkeypatch):
    from qiskit.transpiler import generate_preset_pass_manager

    calls = _fake_qrmi(monkeypatch, block_transpiler="some.module:transpile")
    bitstrings = _sample("real-device", {"params": {"shots": 4}})

    assert calls["loaded"] == ["my-runtime"]
    (call,) = calls["sampled"]
    # Sent untranspiled; the block's transpiler and transpile_config are replaced, and the
    # ISA matches the former host-side preset pass manager (level 3, seed 123).
    (sent,) = call["circuits"]
    assert dict(sent.count_ops()) == {"h": 1, "cx": 2, "measure": 3, "barrier": 1}
    assert call["configs"] == [{"optimization_level": 3, "seed_transpiler": 123}]
    expected = generate_preset_pass_manager(
        optimization_level=3, target=calls["target"], seed_transpiler=123
    ).run(sent)
    assert call["isa"][0] == expected
    assert bitstrings == ["000", "111", "000", "111"]


def test_real_device_rejects_options_qrmi_runtime_does_not_support(monkeypatch):
    import pytest

    calls = _fake_qrmi(monkeypatch)
    for options in ({"params": {"shots": 4}, "twirling": {}}, {"params": {"shots": 4.9}}):
        with pytest.raises(ValueError):
            _sample("real-device", options)
    assert calls["sampled"] == []


def test_random_mode_does_not_load_a_runtime(monkeypatch):
    from qcsc_prefect.integrations.qiskit.qrmi import QrmiRuntimeBlock

    async def aload_runtime(name, prepare=None):
        raise AssertionError("random mode must not load a runtime")

    monkeypatch.setattr(QrmiRuntimeBlock, "aload_runtime", aload_runtime)
    bitstrings = _sample("random", {"params": {"shots": 5}}, bitlen=4)
    assert bitstrings == generate_random_bitstrings(bitlen=4, shots=5, seed=24)
