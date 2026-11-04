from __future__ import annotations

import dataclasses
import random
from typing import Any, Literal

QuantumSource = Literal["real-device", "random"]


def resolve_shots(*, sampler_options: dict[str, Any], default_shots: int) -> int:
    params = sampler_options.get("params", {})
    if not isinstance(params, dict):
        raise TypeError("'params' in sampler options must be a mapping.")
    return int(params.get("shots", default_shots))


def generate_random_bitstrings(
    *,
    bitlen: int,
    shots: int,
    seed: int,
) -> list[str]:
    if bitlen <= 0:
        raise ValueError("'bitlen' must be positive.")
    if shots < 0:
        raise ValueError("'shots' must be non-negative.")

    rng = random.Random(seed)
    return [format(rng.getrandbits(bitlen), f"0{bitlen}b") for _ in range(shots)]


def _build_ghz_circuit(bitlen: int) -> Any:
    from qiskit import QuantumCircuit

    qc_ghz = QuantumCircuit(bitlen)
    qc_ghz.h(0)
    qc_ghz.cx(0, range(1, bitlen))
    qc_ghz.measure_active()
    return qc_ghz


# Preset pass manager options of the default QrmiRuntime transpiler for the GHZ circuit.
GHZ_TRANSPILE_CONFIG = {"optimization_level": 3, "seed_transpiler": 123}


async def sample_bitstrings(
    *,
    quantum_source: QuantumSource,
    runtime_block_name: str,
    sampler_options: dict[str, Any],
    bitlen: int,
    default_shots: int,
    random_seed: int,
) -> list[str]:
    shots = resolve_shots(sampler_options=sampler_options, default_shots=default_shots)
    if quantum_source == "random":
        return generate_random_bitstrings(
            bitlen=bitlen,
            shots=shots,
            seed=random_seed,
        )

    from qcsc_prefect.integrations.qiskit.qrmi import QrmiRuntimeBlock
    from qcsc_prefect.integrations.qiskit.qrmi.runtime import DEFAULT_TRANSPILER

    # The runtime transpiles the GHZ circuit against the actual Target with its default
    # transpiler, in this process (scheduler="iqp") or on the compute node ("slurm").
    # The block's transpiler and transpile_config are replaced.
    runtime = await QrmiRuntimeBlock.aload_runtime(runtime_block_name)
    runtime = dataclasses.replace(
        runtime, transpiler=DEFAULT_TRANSPILER, transpile_config=dict(GHZ_TRANSPILE_CONFIG)
    )
    results = await runtime.sampler([(_build_ghz_circuit(bitlen),)], options=sampler_options)
    return results[0].data.meas.get_bitstrings()
