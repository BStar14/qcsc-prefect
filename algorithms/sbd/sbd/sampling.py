"""Synchronous QrmiRuntime sampling for the real-device path of walker_sqd."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping

import anyio
from prefect import tags
from qcsc_prefect.integrations.qiskit.qrmi import QrmiRuntime, QrmiRuntimeBlock
from qcsc_prefect.integrations.qiskit.qrmi.contract import resolve_shots

from .transpilation import TRANSPILER, prepare, transpile_config

RUNTIME_BLOCK_NAME = "qrmi-runtime"
# Formerly passed to QuantumRuntime.sampler; Prefect concurrency limits can use it.
QUANTUM_TAG = "res: quantum"


def load_runtime(name: str, circuit_params) -> QrmiRuntime:
    """Load the saved QrmiRuntimeBlock and set the SBD compilation hooks.

    The block's scheduler selects IQP or Slurm. Its transpiler and transpile_config are
    replaced, so the SBD compilation cannot be changed by the block.
    """
    runtime = QrmiRuntimeBlock.load_runtime(name)
    return dataclasses.replace(
        runtime,
        prepare=prepare,
        transpiler=TRANSPILER,
        transpile_config=transpile_config(circuit_params),
    )


def sample_sync(runtime: QrmiRuntime, circuit, *, options: Mapping | None):
    """Run ``runtime.sampler([(circuit,)], options=options)`` from synchronous code.

    ``options`` is passed unchanged, so QrmiRuntime's rules apply: only
    ``options["params"]["shots"]`` is accepted, and it must be a positive integer.
    Unsupported options such as twirling are rejected, not ignored, and nothing is
    submitted. They are checked here as well, before an event loop is started.

    Starts a new event loop with ``anyio.run`` in the calling thread, so it must not be
    called from a thread that already runs one. Each call creates its own QRMI resource;
    parallel walkers, in threads or processes, share no QRMI object. The sampling task is
    tagged with ``QUANTUM_TAG``.
    """

    resolve_shots(None, options)

    async def run():
        with tags(QUANTUM_TAG):
            return await runtime.sampler([(circuit,)], options=options)

    return anyio.run(run)
