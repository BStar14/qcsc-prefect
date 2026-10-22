"""Shared QPY/JSON and result validation contract for a single sampler PUB."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral
from pathlib import Path

from qiskit import QuantumCircuit
from qiskit.primitives import PrimitiveResult, PubResult
from qiskit.primitives.containers import BitArray

try:
    from qiskit_ibm_runtime.decoders.result_decoder import ResultDecoder
except ModuleNotFoundError:
    from qiskit_ibm_runtime.utils.result_decoder import ResultDecoder

SCHEMA_VERSION = 1


def positive_shots(value) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError("shots must be a positive integer")
    return int(value)


def resolve_shots(shots, options: Mapping | None) -> int:
    """Accept the pinned PR's params.shots subset; never ignore other options."""
    if options is None:
        options = {}
    if not isinstance(options, Mapping):
        raise TypeError("options must be a mapping")
    if set(options) - {"params"}:
        raise ValueError("Only options['params']['shots'] is supported")
    params = options.get("params", {})
    if not isinstance(params, Mapping):
        raise TypeError("options['params'] must be a mapping")
    if set(params) - {"shots"}:
        raise ValueError("Only options['params']['shots'] is supported")
    option_shots = positive_shots(params["shots"]) if "shots" in params else None
    explicit = positive_shots(shots) if shots is not None else None
    if explicit is not None and option_shots is not None and explicit != option_shots:
        raise ValueError("shots conflicts with options['params']['shots']")
    return explicit if explicit is not None else option_shots if option_shots is not None else 100


def circuit_registers(circuit: QuantumCircuit) -> dict[str, int]:
    if not isinstance(circuit, QuantumCircuit):
        raise TypeError("sampler requires a QuantumCircuit")
    if circuit.num_parameters:
        raise ValueError("Bind all circuit parameters before sampling")
    registers = {reg.name: len(reg) for reg in circuit.cregs}
    if not registers or sum(registers.values()) != circuit.num_clbits:
        raise ValueError("Use named classical registers covering all classical bits")
    if not any(inst.operation.name == "measure" for inst in circuit.data):
        raise ValueError("Use a measured circuit")
    return registers


def decode_result(path: Path, shots: int, registers: Mapping[str, int]) -> PrimitiveResult:
    result = ResultDecoder.decode(path.read_text(encoding="utf-8"))
    if not isinstance(result, PrimitiveResult):
        raise RuntimeError(
            f"task_runner returned {type(result).__name__}; expected PrimitiveResult; see {path}"
        )
    if len(result) != 1:
        raise RuntimeError(f"task_runner returned {len(result)} PUBs; expected 1; see {path}")
    if not isinstance(result[0], PubResult):
        raise RuntimeError(f"Expected sampler PUB result; see {path}")
    data = result[0].data
    if set(data.keys()) != set(registers):
        raise RuntimeError(f"Unexpected result registers; expected {dict(registers)}; see {path}")
    for name, width in registers.items():
        samples = data[name]
        if not isinstance(samples, BitArray) or samples.shape != ():
            raise RuntimeError(f"Expected scalar BitArray for {name}; see {path}")
        if samples.num_shots != shots:
            raise RuntimeError(
                f"task_runner returned {samples.num_shots} shots; expected {shots}; see {path}"
            )
        if samples.num_bits != width:
            raise RuntimeError(f"Expected {width} bits for {name}; see {path}")
    return result
