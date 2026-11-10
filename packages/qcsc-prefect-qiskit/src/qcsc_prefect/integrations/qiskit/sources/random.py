"""Uniform pseudo-random bitstrings, without a circuit or a quantum backend."""

from __future__ import annotations

from numbers import Integral

import numpy as np

from qiskit import QuantumCircuit
from qiskit.primitives import PrimitiveResult, SamplerPubResult
from qiskit.primitives.containers import BitArray, DataBin

from .base import QuantumSource, check_shots


class RandomSource(QuantumSource):
    """Uniform random bits of width ``num_bits`` in the ``meas`` register.

    The same ``seed``, shots and width always give the same bits, on every call. The
    bits are those of ``qiskit_addon_sqd.counts.generate_bit_array_uniform``. This
    is synthetic input, not a simulation: a circuit, if passed, is ignored, and no
    Block or quantum backend is loaded.
    """

    def __init__(self, num_bits: int, seed: int | None = None) -> None:
        if isinstance(num_bits, bool) or not isinstance(num_bits, Integral) or num_bits < 1:
            raise ValueError("num_bits must be a positive integer")
        self.num_bits = int(num_bits)
        self.seed = seed

    @property
    def backend_name(self) -> str:
        return "random"

    @property
    def requires_circuit(self) -> bool:
        return False

    def sample(self, circuit: QuantumCircuit | None = None, *, shots: int) -> PrimitiveResult:
        shots = check_shots(shots)
        rng = np.random.default_rng(self.seed)
        bits = rng.integers(2, size=(shots, self.num_bits), dtype=bool)
        meas = BitArray.from_bool_array(bits)
        return PrimitiveResult([SamplerPubResult(DataBin(meas=meas), metadata={})], metadata={})
