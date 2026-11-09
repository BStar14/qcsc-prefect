"""SQD sampling policy, independent of the platform selected by a saved source Block."""

from __future__ import annotations

from qcsc_prefect.integrations.qiskit.sources import QuantumSource, QuantumSourceBlock

from .transpilation import TRANSPILER, prepare, transpile_config


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
    """Load the named source Block inside the walker and supply SQD policy.

    ``quantum_source`` is an arbitrary Block document name. The Block owns source
    selection and platform settings; SQD supplies its width, walker seed and hooks.
    """
    return QuantumSourceBlock.load_source(
        quantum_source,
        num_bits=2 * num_orbitals,
        seed=random_seed_for(random_seed, trial_index, walker_index),
        prepare=prepare,
        transpiler=TRANSPILER,
        transpile_config=transpile_config(circuit_params),
    )
