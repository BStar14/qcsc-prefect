"""Synchronous QuantumSource over an existing QrmiRuntime (IQP or Slurm)."""

from __future__ import annotations

from collections.abc import Iterable

import anyio
from prefect import tags as prefect_tags

from qiskit import QuantumCircuit
from qiskit.primitives import PrimitiveResult

from ..qrmi.runtime import QrmiRuntime
from .base import QuantumSource, check_circuit, check_shots


class QrmiSource(QuantumSource):
    """Sample one circuit through ``runtime.sampler([(circuit,)], shots=shots)``.

    The runtime is used as given: its scheduler, ``prepare``, ``transpiler`` and
    ``transpile_config`` decide how the circuit is prepared and compiled. Applications
    set their own defaults on the runtime before passing it here.

    Each call runs the async sampler in a new event loop (``anyio.run``) in the calling
    thread, so ``sample`` must not be called from a thread that already runs an event
    loop; it then fails before anything is submitted. Separate calls, in threads or
    processes, share no QRMI object. ``tags`` are applied to the Prefect tasks that the
    runtime starts, for example to use a concurrency limit.
    """

    def __init__(self, runtime: QrmiRuntime, *, tags: Iterable[str] = ()) -> None:
        if not isinstance(runtime, QrmiRuntime):
            raise TypeError("runtime must be a QrmiRuntime")
        # The same check as QrmiRuntime.sampler, made early so that logs show the name.
        resource = runtime.resource
        if not isinstance(resource, str) or not resource or any(c.isspace() for c in resource):
            raise ValueError("resource must be a nonempty QRMI resource ID without whitespace")
        if isinstance(tags, (str, bytes)):
            raise TypeError("tags must be an iterable of tag strings, not a string")
        self.runtime = runtime
        self.tags = tuple(tags)

    @property
    def backend_name(self) -> str:
        return self.runtime.resource

    @property
    def requires_circuit(self) -> bool:
        return True

    def sample(self, circuit: QuantumCircuit | None = None, *, shots: int) -> PrimitiveResult:
        shots = check_shots(shots)
        check_circuit(circuit)

        async def run():
            with prefect_tags(*self.tags):
                return await self.runtime.sampler([(circuit,)], shots=shots)

        return anyio.run(run)
