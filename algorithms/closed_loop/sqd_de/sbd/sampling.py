"""QuantumSource selection for walker_sqd and the SBD settings of each source."""

from __future__ import annotations

import dataclasses
from pathlib import Path

from qcsc_prefect.integrations.qiskit.qrmi import QrmiRuntime, QrmiRuntimeBlock
from qcsc_prefect.integrations.qiskit.sources import (
    QFwSource,
    QrmiSource,
    QuantumSource,
    RandomSource,
)

from .transpilation import TRANSPILER, prepare, transpile_config

RUNTIME_BLOCK_NAME = "qrmi-runtime"
QFW_BLOCK_NAME = "qfw-runner"
# Formerly passed to QuantumRuntime.sampler; Prefect concurrency limits can use it.
QUANTUM_TAG = "res: quantum"
QUANTUM_SOURCES = ("random", "real-device", "qfw")
# The node-side QFw-SLURM sampler; QFwSource passes this path to the launcher.
QFW_SAMPLER = Path(__file__).with_name("qfw_slurm_sqd_sampler.py")


def random_seed_for(random_seed: int, trial_index: int, walker_index: int) -> int:
    """Bitstring seed of one walker: ``random_seed`` plus a pairing of the two indices.

    It does not use the module RNG, so that random and device sampling are controlled
    the same way.
    """
    pair = trial_index + walker_index
    return int(random_seed + (pair * (pair + 1) // 2 + walker_index))


def load_quantum_source(
    quantum_source: str,
    *,
    circuit_params,
    num_orbitals: int,
    random_seed: int,
    trial_index: int,
    walker_index: int,
) -> QuantumSource:
    """Build the walker's QuantumSource; load its block only if it needs one.

    ``random`` uses ``2 * num_orbitals`` bits and the seed of ``random_seed_for``.
    ``real-device`` loads the QrmiRuntimeBlock ``qrmi-runtime`` and ``qfw`` the
    QFwResource ``qfw-runner``; both use the SBD compilation settings.
    """
    if quantum_source == "random":
        return RandomSource(
            num_bits=2 * num_orbitals,
            seed=random_seed_for(random_seed, trial_index, walker_index),
        )
    if quantum_source == "real-device":
        try:
            runtime = load_runtime(RUNTIME_BLOCK_NAME, circuit_params)
        except ValueError as exc:
            raise RuntimeError(
                "Quantum source 'real-device' requested but QrmiRuntimeBlock "
                f"'{RUNTIME_BLOCK_NAME}' could not be loaded. Set quantum_source='random' "
                "to use deterministic random sampling instead."
            ) from exc
        return QrmiSource(runtime, tags=(QUANTUM_TAG,))
    if quantum_source == "qfw":
        return load_qfw_source(QFW_BLOCK_NAME, circuit_params)
    raise ValueError(
        f"Unsupported quantum_source: {quantum_source!r}. "
        "Expected 'random', 'real-device' or 'qfw'."
    )


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


def load_qfw_source(name: str, circuit_params) -> QFwSource:
    """Load the QFwResource ``name`` and sample with the SBD QFw-SLURM sampler script.

    The ffsim pre-init stage runs here, before QPY transport turns the ffsim gates into
    plain gates; the script then searches the SABRE layout on the node with the same
    settings as ``real-device``.
    """
    from qcsc_prefect.integrations.qiskit.sources import QFwResource

    try:
        resource = QFwResource.load(name)
    except ValueError as exc:
        raise RuntimeError(
            f"Quantum source 'qfw' requested but QFwResource block '{name}' is not defined."
        ) from exc
    return QFwSource(
        resource.device_id,
        sampler=QFW_SAMPLER,
        sampler_args=qfw_sampler_args(circuit_params),
        prepare=prepare_for_qfw,
    )


def prepare_for_qfw(circuit):
    return prepare(circuit).circuit


def qfw_sampler_args(circuit_params) -> list:
    """Command-line arguments of the QFw sampler script, as integers in this order."""
    config = transpile_config(circuit_params)
    return [
        "--optimization-level",
        config["optimization_level"],
        "--sabre-max-iterations",
        config["sabre_max_iterations"],
        "--sabre-swap-trials",
        config["sabre_swap_trials"],
        "--sabre-layout-trials",
        config["sabre_layout_trials"],
    ]
