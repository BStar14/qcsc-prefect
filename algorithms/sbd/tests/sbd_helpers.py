"""Small LUCJ circuits and a line-connected Target shared by the SBD tests."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.transpiler import CouplingMap
from sbd import lucj

NORB = 4
NELEC = (2, 2)
AA_INDICES = [(p, p + 1) for p in range(NORB - 1)]
AB_INDICES = [(p, p) for p in range(0, NORB, 4)]
N_LAYERS = 2
NUM_QUBITS = 2 * NORB + 6


def line_target(basis=("cz", "rz", "sx", "x")):
    # GenericBackendV2 is fully connected by default, where every layout is equally good
    # and the SABRE settings have no effect. A line makes them matter.
    return GenericBackendV2(
        num_qubits=NUM_QUBITS,
        basis_gates=list(basis),
        coupling_map=CouplingMap.from_line(NUM_QUBITS),
        seed=7,
    ).target


def ucj_parameters(count, seed=3):
    rng = np.random.default_rng(seed)
    size = N_LAYERS * (len(AA_INDICES) + len(AB_INDICES) + NORB**2) + NORB**2
    return [rng.normal(scale=0.3, size=size) for _ in range(count)]


def electronic_properties():
    return SimpleNamespace(num_orbitals=NORB, num_electrons=NELEC, open_shell=False)


def lucj_circuit(ucj_parameter, use_reset_mitigation=True):
    return lucj.create_lucj_circuit.fn(
        ucj_parameter=ucj_parameter,
        elec_props=electronic_properties(),
        aa_indices=AA_INDICES,
        ab_indices=AB_INDICES,
        n_lucj_layers=N_LAYERS,
        use_reset_mitigation=use_reset_mitigation,
    )
