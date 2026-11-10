"""The QuantumSource interface shared by applications that sample bitstrings."""

from __future__ import annotations

from abc import ABC, abstractmethod

from qiskit import QuantumCircuit
from qiskit.primitives import PrimitiveResult

from ..qrmi.contract import circuit_registers, positive_shots


class QuantumSource(ABC):
    """Where an application gets its bitstrings from, whatever platform produces them.

    ``sample`` is synchronous and returns a ``PrimitiveResult`` with one PUB.
    Sources that run circuits return every measured shot, without
    application-specific filtering; the application interprets the registers.

    ``requires_circuit`` only says whether ``sample`` needs an input circuit. It says
    nothing about error mitigation. Invalid shots, and for sources that need one, a
    missing, unbound or unmeasured circuit, are rejected before any work starts.
    """

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Name of the backend, for logs."""

    @property
    @abstractmethod
    def requires_circuit(self) -> bool:
        """Whether ``sample`` needs a circuit."""

    @abstractmethod
    def sample(self, circuit: QuantumCircuit | None = None, *, shots: int) -> PrimitiveResult:
        """Sample ``shots`` shots and return one PUB."""


def check_circuit(circuit) -> dict[str, int]:
    """Return the classical registers of a bound, measured circuit; reject anything else."""
    if circuit is None:
        raise ValueError("This source requires a circuit")
    return circuit_registers(circuit)


def check_shots(shots) -> int:
    return positive_shots(shots)
