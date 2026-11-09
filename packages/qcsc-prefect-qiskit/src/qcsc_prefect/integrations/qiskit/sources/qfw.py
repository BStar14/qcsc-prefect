"""QuantumSource over the QFw-SLURM launcher ``qfw_slurm_sampler.run_sampler_job``."""

from __future__ import annotations

import os
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np

from qiskit import QuantumCircuit
from qiskit.primitives import PrimitiveResult, SamplerPubResult
from qiskit.primitives.containers import BitArray, DataBin

from ..qrmi.contract import circuit_registers, validate_result
from .base import QuantumSource, check_circuit, check_shots

LAUNCHER_MODULE = "qfw_slurm_sampler"
DEFAULT_WORK_DIR = "~/qfw_jobs"
_BINARY = re.compile(r"[01]+")


class QFwSource(QuantumSource):
    """Sample one circuit with the QFw-SLURM launcher ``run_sampler_job``.

    The launcher is called directly, in the calling thread, with
    ``run_sampler_job(circuit, shots=..., qpu=device_id, work_dir=..., sampler=...,
    sampler_args=...)``. ``sampler`` is the node-side sampler script and
    ``sampler_args`` its extra command-line arguments, passed unchanged. The launcher
    is imported on the first ``sample``, from ``QFW_SLURM_BIN`` if that is set.

    ``prepare``, if given, runs once on a copy of the circuit before the launcher
    serializes it, and must keep the classical registers. The counts are returned as
    one PUB with a ``BitArray`` per classical register; the rest of the launcher record
    is in ``result.metadata["qcsc_qfw"]``.
    """

    def __init__(
        self,
        device_id: str,
        *,
        sampler: str | os.PathLike,
        sampler_args: Sequence = (),
        prepare: Callable[[QuantumCircuit], QuantumCircuit] | None = None,
        work_dir: str | os.PathLike = DEFAULT_WORK_DIR,
    ) -> None:
        if not isinstance(device_id, str) or not device_id.strip():
            raise ValueError("device_id must be a nonempty string")
        self.device_id = device_id
        self.sampler = str(sampler)
        self.sampler_args = list(sampler_args)
        self.prepare = prepare
        self.work_dir = work_dir

    @property
    def backend_name(self) -> str:
        return self.device_id

    @property
    def requires_circuit(self) -> bool:
        return True

    def sample(self, circuit: QuantumCircuit | None = None, *, shots: int) -> PrimitiveResult:
        shots = check_shots(shots)
        registers = check_circuit(circuit)
        prepared = circuit
        if self.prepare is not None:
            prepared = self.prepare(circuit.copy())
            if circuit_registers(prepared) != registers:
                raise ValueError("prepare must preserve classical register names and widths")
        run_sampler_job = _launcher()
        record = run_sampler_job(
            prepared,
            shots=shots,
            qpu=self.device_id,
            work_dir=str(Path(self.work_dir).expanduser()),
            sampler=self.sampler,
            sampler_args=self.sampler_args,
        )
        return counts_to_result(record, prepared, shots)


def _launcher():
    """Import ``run_sampler_job``, adding ``QFW_SLURM_BIN`` to ``sys.path`` once."""
    qfw_slurm_bin = os.environ.get("QFW_SLURM_BIN")
    if qfw_slurm_bin and qfw_slurm_bin not in sys.path:
        sys.path.insert(0, qfw_slurm_bin)
    try:
        from qfw_slurm_sampler import run_sampler_job
    except ImportError as exc:
        raise RuntimeError(
            f"QFwSource requires the QFw-SLURM launcher module {LAUNCHER_MODULE!r}; "
            "install QFw-SLURM or set QFW_SLURM_BIN to its directory"
        ) from exc
    return run_sampler_job


def counts_to_result(record: Mapping, circuit: QuantumCircuit, shots: int) -> PrimitiveResult:
    """Convert a QFw record with Qiskit-style ``counts`` into one sampler PUB.

    Count keys list the classical registers in reverse order, separated by spaces or
    joined, in binary. Every register is expanded from the same count entry, so shot
    ``i`` of each register comes from the same measured shot; the order of shots is not
    the original one. The counts must add up to ``shots``.
    """
    if not isinstance(record, Mapping) or not isinstance(record.get("counts"), Mapping):
        raise RuntimeError("QFw returned no counts")
    registers = circuit_registers(circuit)
    names = [reg.name for reg in reversed(circuit.cregs)]
    widths = [registers[name] for name in names]
    keys = []
    repeats = []
    for key, count in record["counts"].items():
        if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 0:
            raise RuntimeError(f"Invalid QFw count {count!r} for {key!r}")
        keys.append(_split_key(key, names, widths))
        repeats.append(int(count))
    total = sum(repeats)
    if total != shots:
        raise RuntimeError(f"QFw returned {total} shots; expected {shots}")
    data = {}
    for index, (name, width) in enumerate(zip(names, widths)):
        samples = np.repeat([parts[index] for parts in keys], repeats)
        data[name] = BitArray.from_samples(list(samples), num_bits=width)
    metadata = {key: value for key, value in record.items() if key != "counts"}
    result = PrimitiveResult([SamplerPubResult(DataBin(**data), metadata={})], metadata={})
    result = validate_result(result, shots, [registers], where="QFw record")
    result.metadata["qcsc_qfw"] = metadata
    return result


def _split_key(key, names, widths) -> list[str]:
    if not isinstance(key, str):
        raise RuntimeError(f"Invalid QFw count key {key!r}")
    parts = key.split()
    if len(parts) == 1 and len(names) > 1:
        joined = parts[0]
        if len(joined) != sum(widths):
            raise RuntimeError(f"QFw count key {key!r} does not have {sum(widths)} bits")
        parts, start = [], 0
        for width in widths:
            parts.append(joined[start : start + width])
            start += width
    if len(parts) != len(names):
        raise RuntimeError(f"QFw count key {key!r} does not have the registers {names}")
    for part, name, width in zip(parts, names, widths):
        if not _BINARY.fullmatch(part):
            raise RuntimeError(f"QFw count key {key!r} is not binary")
        if len(part) != width:
            raise RuntimeError(f"QFw count key {key!r} has {len(part)} bits for {name}")
    return parts
