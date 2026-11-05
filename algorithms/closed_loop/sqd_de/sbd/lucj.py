# Workflow for observability demo on Miyabi

import ffsim
import numpy as np
from prefect import task
from qcsc_workflow_utility.chem import (
    ElectronicProperties,
    NpStrict1DArrayF64,
    NpStrict2DArrayF64,
    NpStrict4DArrayF64,
)
from qiskit.circuit import ClassicalRegister, QuantumCircuit, QuantumRegister

MODULE_RNG = np.random.default_rng(seed=4520)


@task
def initialize_ucj_parameters(
    elec_props: ElectronicProperties,
    aa_indices: list[tuple[int, int]],
    ab_indices: list[tuple[int, int]],
    num_walkers: int,
    randomization_factor: float,
    n_lucj_layers: int,
) -> NpStrict2DArrayF64:
    assert not elec_props.open_shell
    global MODULE_RNG

    def _t2_to_ucj_parameters(t2: NpStrict4DArrayF64) -> NpStrict1DArrayF64:
        nonlocal aa_indices
        nonlocal ab_indices
        nonlocal n_lucj_layers

        tmp_operator = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
            t2=t2,
            n_reps=n_lucj_layers + 1,
            interaction_pairs=(aa_indices, ab_indices),
        )
        truncated_ucj_op = ffsim.UCJOpSpinBalanced(
            diag_coulomb_mats=tmp_operator.diag_coulomb_mats[:-1],
            orbital_rotations=tmp_operator.orbital_rotations[:-1],
            final_orbital_rotation=tmp_operator.orbital_rotations[-1],
        )
        return truncated_ucj_op.to_parameters(interaction_pairs=(aa_indices, ab_indices))

    # First walker is the bare CCSD parameters
    initial_params = [_t2_to_ucj_parameters(t2=elec_props.t2)]

    # The rest of walkers are randomized parameters
    for _ in range(num_walkers - 1):
        rand_values = randomization_factor * (MODULE_RNG.random(elec_props.t2.shape) - 0.5)
        drifted_params = _t2_to_ucj_parameters(t2=elec_props.t2 + rand_values)
        initial_params.append(drifted_params)
    return initial_params


@task
def create_lucj_circuit(
    ucj_parameter: NpStrict1DArrayF64,
    elec_props: ElectronicProperties,
    aa_indices: list[tuple[int, int]],
    ab_indices: list[tuple[int, int]],
    n_lucj_layers: int,
    use_reset_mitigation: bool,
) -> QuantumCircuit:
    assert not elec_props.open_shell

    qreg = QuantumRegister(2 * elec_props.num_orbitals, name="q")
    creg_test = ClassicalRegister(2 * elec_props.num_orbitals, name="test")
    creg_meas = ClassicalRegister(2 * elec_props.num_orbitals, name="meas")

    regs = [qreg, creg_meas]
    if use_reset_mitigation:
        regs.append(creg_test)

    circ = QuantumCircuit(*regs)
    if use_reset_mitigation:
        circ.measure(qreg, creg_test)
        circ.barrier()
    circ.append(
        ffsim.qiskit.PrepareHartreeFockJW(
            norb=elec_props.num_orbitals,
            nelec=elec_props.num_electrons,
        ),
        qargs=qreg,
    )
    ucj_op = ffsim.UCJOpSpinBalanced.from_parameters(
        params=ucj_parameter,
        norb=elec_props.num_orbitals,
        n_reps=n_lucj_layers,
        interaction_pairs=(aa_indices, ab_indices),
        with_final_orbital_rotation=True,
    )
    circ.append(
        ffsim.qiskit.UCJOpSpinBalancedJW(
            ucj_op=ucj_op,
        ),
        qargs=qreg,
    )
    circ.measure(qreg, creg_meas)
    return circ
