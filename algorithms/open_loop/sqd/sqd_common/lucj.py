"""LUCJ ansatz: parameters from CCSD (Optimize), circuit and transpilation (Execute)."""

from typing import Annotated

import ffsim
import numpy as np
from ffsim.qiskit import PRE_INIT
from prefect import get_run_logger, task
from prefect.cache_policies import NO_CACHE
from pydantic import BaseModel
from pydantic_numpy.helper.annotation import NpArrayPydanticAnnotation
from qcsc_workflow_utility import ElectronicProperties, NpStrict4DArrayF64
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.transpiler import Target

from sqd_common.flow_params import CircuitParameters
from sqd_common.topology import MAX_AB_PAIRS as MAX_AB_PAIRS
from sqd_common.topology import Topology, interaction_pairs
from sqd_common.transpile import build_pass_manager, derive_seed, isa_gate_depth

NpStrict3DArrayC128 = Annotated[
    np.ndarray[tuple[int, int, int], np.dtype[np.complex128]],
    NpArrayPydanticAnnotation.factory(
        data_type=np.complex128, dimensions=3, strict_data_typing=True
    ),
]


class LUCJParameters(BaseModel):
    """Parameters of a spin-balanced LUCJ operator and the system it acts on."""

    diag_coulomb_mats: NpStrict4DArrayF64
    orbital_rotations: NpStrict3DArrayC128
    norb: int
    nelec: tuple[int, int]
    qpu_topology: Topology

    def to_operator(self) -> ffsim.UCJOpSpinBalanced:
        return ffsim.UCJOpSpinBalanced(
            diag_coulomb_mats=self.diag_coulomb_mats,
            orbital_rotations=self.orbital_rotations,
        )


@task(
    name="get_lucj_params",
    tags=["optimize"],
    persist_result=True,
    result_serializer="compressed/pickle",
)
def get_lucj_params(
    circuit_params: CircuitParameters,
    elec_props: ElectronicProperties,
    qpu_topology: Topology,
) -> LUCJParameters:
    """Initialize the LUCJ parameters from the CCSD t2 amplitudes.

    The interactions are restricted to the pairs that ``qpu_topology`` can
    implement without SWAP gates (see ``interaction_pairs``). With ``compress``,
    the truncated double factorization is optimized to reproduce t2 more closely.

    Args:
        circuit_params: LUCJ layers and compression settings.
        elec_props: Electronic properties with the CCSD t2 amplitudes.
        qpu_topology: QPU topology from ``sqd_common.topology.resolve_topology``.
            It is an input of its own, so the persisted result of one topology is
            not reused for another.

    Returns:
        LUCJ operator parameters, including the system size and the topology.
    """
    logger = get_run_logger()
    pairs = interaction_pairs(elec_props.num_orbitals, qpu_topology)
    logger.info(
        "Getting LUCJ parameters: %d layer(s), %d alpha-alpha and %d alpha-beta pairs "
        "for the %s topology%s.",
        circuit_params.n_lucj_layers,
        len(pairs[0]),
        len(pairs[1]),
        qpu_topology,
        ", compressed" if circuit_params.compress else "",
    )

    compress_kwargs = {}
    if circuit_params.compress:
        iteration = 0

        def callback(intermediate_result):
            nonlocal iteration
            iteration += 1
            logger.info("Compression iteration %d: error = %g", iteration, intermediate_result.fun)

        compress_kwargs = dict(
            optimize=True,
            options=dict(maxiter=circuit_params.compress_maxiter),
            callback=callback,
        )

    ucj_op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
        t2=elec_props.t2,
        n_reps=circuit_params.n_lucj_layers,
        interaction_pairs=pairs,
        **compress_kwargs,
    )
    return LUCJParameters(
        diag_coulomb_mats=ucj_op.diag_coulomb_mats,
        orbital_rotations=ucj_op.orbital_rotations,
        norb=elec_props.num_orbitals,
        nelec=elec_props.num_electrons,
        qpu_topology=qpu_topology,
    )


@task(name="build_lucj_circuit", tags=["execute"], cache_policy=NO_CACHE)
def build_lucj_circuit(
    lucj_params: LUCJParameters,
    circuit_params: CircuitParameters,
) -> QuantumCircuit:
    """Build the measured LUCJ circuit on the Hartree-Fock state.

    With reset mitigation, every qubit is measured into the ``test`` register
    first, so shots that did not start in the ground state can be discarded.
    """
    logger = get_run_logger()
    qreg = QuantumRegister(2 * lucj_params.norb, name="q")
    creg_test = ClassicalRegister(2 * lucj_params.norb, name="test")
    creg_meas = ClassicalRegister(2 * lucj_params.norb, name="meas")

    regs = [qreg, creg_meas]
    if circuit_params.use_reset_mitigation:
        regs.append(creg_test)

    lucj_circ = QuantumCircuit(*regs)
    if circuit_params.use_reset_mitigation:
        lucj_circ.measure(qreg, creg_test)
        lucj_circ.barrier()
    lucj_circ.append(
        ffsim.qiskit.PrepareHartreeFockJW(norb=lucj_params.norb, nelec=lucj_params.nelec),
        qargs=qreg,
    )
    lucj_circ.append(ffsim.qiskit.UCJOpSpinBalancedJW(ucj_op=lucj_params.to_operator()), qargs=qreg)
    lucj_circ.measure(qreg, creg_meas)

    logger.info(
        f"Untranspiled LUCJ circuit:\n"
        f"Number of qubits = {lucj_circ.num_qubits}\n"
        f"Number of classical bits = {lucj_circ.num_clbits}\n"
        f"Circuit depth = {lucj_circ.depth()}\n"
        f"Instruction counts = {dict(lucj_circ.count_ops())}\n"
    )
    return lucj_circ


def circuit_transpile_options(
    circuit_params: CircuitParameters, seed: int | np.random.Generator
) -> dict[str, int]:
    """JSON-safe ``build_pass_manager`` options for the configured circuit.

    The transpiler settings apply only with ``configure_transpiler``; otherwise
    the defaults of ``TranspilerParameters`` are used.
    """
    options = circuit_params.transpile_options
    return {
        "optimization_level": options.optimization_level,
        "seed_transpiler": derive_seed(seed),
        "sabre_max_iterations": options.sabre_max_iterations,
        "sabre_layout_trials": options.sabre_layout_trials,
        "sabre_swap_trials": options.sabre_swap_trials,
    }


@task(name="transpile_lucj_circuit", tags=["execute"], cache_policy=NO_CACHE)
def transpile_lucj_circuit(
    circuit: QuantumCircuit,
    target: Target,
    circuit_params: CircuitParameters,
    seed: int | np.random.Generator,
) -> QuantumCircuit:
    """Compile the LUCJ circuit for ``target`` with a massive SABRE layout search."""
    logger = get_run_logger()
    passmanager = build_pass_manager(
        target,
        **circuit_transpile_options(circuit_params, seed),
        pre_init=PRE_INIT,
    )
    logger.info("Transpiling ansatz circuit...")
    isa_circuit = passmanager.run(circuit)
    logger.info(
        f"Circuit depth = {isa_gate_depth(isa_circuit)}\n"
        f"Instruction counts = {dict(isa_circuit.count_ops())}\n"
    )
    return isa_circuit
