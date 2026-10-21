"""Definition of the SQD workflow with the fulqrum eigensolver.

The quantum stage (LUCJ ansatz, transpilation, sampling) is the same as in
``sqd_dice``. The configuration recovery loop runs each subspace through a
``FulqrumSolverJob`` block instead of the DICE SHCI solver.
"""

import os
import pathlib
from typing import Any, Literal

import ffsim
import numpy as np
from ffsim.qiskit import PRE_INIT
from prefect import flow, get_run_logger, task
from prefect.artifacts import create_table_artifact
from prefect.cache_policies import INPUTS, NO_CACHE
from prefect.variables import Variable
from prefect_qiskit.runtime import QuantumRuntime
from pydantic import BaseModel, Field
from qcsc_prefect_fulqrum import FulqrumSolverJob
from qcsc_workflow_utility import (
    ElectronicProperties,
    compute_molecular_integrals_from_fcidump,
    compute_molecular_integrals_from_geometry,
)
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.passmanager import ConditionalController
from qiskit.primitives.containers import BitArray
from qiskit.transpiler import Target, generate_preset_pass_manager
from qiskit.transpiler.passes import (
    ApplyLayout,
    BarrierBeforeFinalMeasurements,
    EnlargeWithAncilla,
    FullAncillaAllocation,
    Optimize1qGatesDecomposition,
    RemoveIdentityEquivalent,
    SabreLayout,
)
from qiskit.transpiler.passmanager import PassManager
from qiskit_addon_sqd.counts import bit_array_to_arrays, generate_bit_array_uniform
from qiskit_addon_sqd.fermion import SCIResult
from qiskit_ibm_runtime.transpiler.passes import FoldRzzAngle

from sqd_fulqrum.recovery import SolveFn, run_configuration_recovery

DEFAULT_RUNTIME_BLOCK_NAME = "ibm-runner"
DEFAULT_SOLVER_BLOCK_NAME = "sqd-fulqrum-solver"
FLOW_NAME = "sqd_2405_05068_fulqrum"
ARTIFACT_KEY = "sqd-fulqrum-configuration-recovery"


class MoleculeGeometry(BaseModel):
    """Molecule definition by geometory and basis."""

    atom: str = Field(
        description="Definition for molecule structure.",
        title="Atom",
    )
    basis: str = Field(
        default="6-31g",
        description="Name of basis set.",
        title="Basis Set",
    )
    symmetry: str | bool = Field(
        default=False,
        description="Whether to use symmetry, otherwise string of point group name.",
        title="Symmetry",
    )
    spin_sq: float = Field(
        default=0.0,
        description=(
            "Target value for the total spin squared for the ground state. "
            "The fulqrum solver applies no spin penalty and ignores it."
        ),
        title="Total Spin Squared",
    )


class FCIDumpFile(BaseModel):
    """Molecule definition by FCIDump file."""

    fcidump_file: str = Field(
        default="",
        description=(
            "Location of FCIDump file storing 1-electron and 2-electron integrals. "
            "We assume these are precomputed in the MO basis."
        ),
        title="FCIDump File",
    )
    spin_sq: float = Field(
        default=0.0,
        description=(
            "Target value for the total spin squared for the ground state. "
            "The fulqrum solver applies no spin penalty and ignores it."
        ),
        title="Total Spin Squared",
    )


class CircuitParameters(BaseModel):
    """Configuration for circuit."""

    use_reset_mitigation: bool = Field(
        default=False,
        description=(
            "Use reset mitigation scheme that post-selects outcomes with non-ground initial state. "
            "This post-selection reduces the net shot number and its retention rate depends on "
            "the quality of hardware reset instruction."
        ),
        title="Reset Mitigation",
    )
    optimization_level: int = Field(
        default=3,
        description="Transpile: Optimization level of transpiler",
        title="Optimization Level",
        ge=0,
        le=3,
    )
    sabre_max_iterations: int = Field(
        default=8,
        description=(
            "Transpile: The number of forward-backward routing iterations to refine "
            "the layout and reduce routing costs."
        ),
        title="SABRE Max Iteration",
        ge=1,
    )
    sabre_swap_trials: int = Field(
        default=10,
        description=(
            "Transpile: The number of routing trials for each layout, refining gate "
            "placement for better routing."
        ),
        title="SABRE SWAP Trials",
        ge=1,
    )
    sabre_layout_trials: int = Field(
        default=10000,
        description=(
            "Transpile: The number of random initial layouts tested, selecting the "
            "one that minimizes SWAP gates."
        ),
        title="SABRE Layout Trials",
        ge=1,
    )
    n_lucj_layers: int = Field(
        default=1,
        description="Number of repetition of the unit layer of LUCJ ansatz.",
        title="LUCJ Layers",
        ge=1,
    )


class SQDParameters(BaseModel):
    """Configuration for SQD algorithm execution."""

    subspace_dim: int = Field(
        description="Dimension d of subsampled bitstrings for diagonalization.",
        title="Subspace Dimension",
        ge=1,
    )
    num_batches: int = Field(
        description=(
            "Number of batches of configurations used by the different calls to the "
            "eigenstate solver. Each batch is one fulqrum job; they run concurrently."
        ),
        title="Batch Number",
        ge=1,
    )
    max_iterations: int = Field(
        description="Number of self-consistent configuration recovery iterations.",
        title="Max Iteration",
        ge=1,
    )


class Parameters(BaseModel):
    """Workflow parameters to configure the sqd_2405_05068_fulqrum workflow."""

    molecule: MoleculeGeometry | FCIDumpFile = Field(
        description="PySCF definition of the molecule to solve for.",
        title="Molecule",
    )

    circuit: CircuitParameters = Field(
        default_factory=CircuitParameters,
        description="Ansatz circuit definition and transpiler settings.",
        title="Circuit",
    )

    sqd: SQDParameters = Field(
        description="Control of SQD algorithm execution.",
        title="SQD",
    )

    quantum_source: Literal["real-device", "random"] = Field(
        default="real-device",
        description=(
            "Select whether SQD sampling uses IBM Quantum Runtime "
            "or deterministic random bitstrings."
        ),
        title="Quantum Source",
    )

    random_seed: int = Field(
        default=24,
        description="Base RNG seed used when Quantum Source is 'random'.",
        title="Random Seed",
        ge=0,
    )


def _resolve_shots(
    *,
    sampler_options: dict[str, Any],
    default_shots: int = 100_000,
) -> int:
    params = sampler_options.get("params", {})
    if not isinstance(params, dict):
        raise TypeError("'params' in sampler options must be a mapping.")
    return int(params.get("shots", default_shots))


def make_block_solver(
    solver: FulqrumSolverJob,
    elec_props: ElectronicProperties,
) -> SolveFn:
    """Bind the molecular integrals to ``solver.run`` for the recovery loop."""

    async def solve(ci_strings: tuple[np.ndarray, np.ndarray]) -> SCIResult:
        return await solver.run(
            ci_strings=ci_strings,
            one_body_tensor=elec_props.one_body_tensor,
            two_body_tensor=elec_props.two_body_tensor,
            norb=elec_props.num_orbitals,
            nelec=elec_props.num_electrons,
        )

    return solve


@flow(name=FLOW_NAME)
async def sqd_2405_05068_fulqrum(
    parameters: Parameters,
    runner_name: str = DEFAULT_RUNTIME_BLOCK_NAME,
    solver_name: str = DEFAULT_SOLVER_BLOCK_NAME,
    option_name: str = "sampler_options",
    cache_compute_integrals: bool = False,
    cache_sampling: bool = False,
) -> float:
    """SQD Experiment from arXiv2405.05068, diagonalized with fulqrum.

    Args:
        parameters: Workflow parameters.
        runner_name: Name of QuantumRuntime block to load.
        solver_name: Name of FulqrumSolverJob block to load.
        option_name: Name of Variable storing sampler primitive options to load.
        cache_compute_integrals: Set True to cache compute_molecular_integrals task.
        cache_sampling: Set True to cache sample_bitstrings task.

    Returns:
        Minimum ground state energy solved by SQD.
    """
    rng = np.random.default_rng(24)
    logger = get_run_logger()

    # Compute integrals
    if isinstance(parameters.molecule, MoleculeGeometry):
        elec_props = compute_molecular_integrals_from_geometry.with_options(
            cache_policy=INPUTS if cache_compute_integrals else NO_CACHE,
        )(
            atom=parameters.molecule.atom,
            basis=parameters.molecule.basis,
            symmetry=parameters.molecule.symmetry,
            spin_sq=parameters.molecule.spin_sq,
        )
    elif isinstance(parameters.molecule, FCIDumpFile):
        elec_props = compute_molecular_integrals_from_fcidump.with_options(
            cache_policy=INPUTS if cache_compute_integrals else NO_CACHE,
        )(
            fcidump_file=parameters.molecule.fcidump_file,
            spin_sq=parameters.molecule.spin_sq,
        )
    else:
        raise TypeError("Unsupported molecule type")
    if elec_props.spin_sq:
        logger.warning(
            f"spin_sq={elec_props.spin_sq} is ignored: the fulqrum solver applies no spin penalty."
        )

    # Sample bitstrings
    bit_array = await sample_bitstrings.with_options(
        cache_policy=INPUTS if cache_sampling else NO_CACHE,
    )(
        circuit_params=parameters.circuit,
        elec_props=elec_props,
        runner_name=runner_name,
        option_name=option_name,
        quantum_source=parameters.quantum_source,
        random_seed=parameters.random_seed,
    )

    # Convert BitArray into bitstring and probability arrays
    raw_bitstrings, raw_probs = bit_array_to_arrays(bit_array)

    # Run configuration recovery loop
    sci_solver = await FulqrumSolverJob.load(solver_name)
    outcome = await run_configuration_recovery(
        raw_bitstrings,
        raw_probs,
        solve=make_block_solver(sci_solver, elec_props),
        initial_occupancies=elec_props.initial_occupancy,
        nelec=elec_props.num_electrons,
        open_shell=elec_props.open_shell,
        subspace_dim=parameters.sqd.subspace_dim,
        num_batches=parameters.sqd.num_batches,
        max_iterations=parameters.sqd.max_iterations,
        rng=rng,
        nuclear_repulsion_energy=elec_props.nuclear_repulsion_energy,
        logger=logger,
    )

    # Upload artifact
    await create_table_artifact(
        table=outcome.records,
        key=ARTIFACT_KEY,
    )
    return outcome.best_energy


@task
def create_ansatz_circuits(
    circuit_params: CircuitParameters,
    elec_props: ElectronicProperties,
) -> QuantumCircuit:
    """Create LUCJ ansatz circuits for SQD algorithm.

    Args:
        circuit_params: Circuit parameters including n_lucj_layers.
        elec_props: Electronic properties of molecule to solve for.

    Returns:
        LUCJ ansatz circuit for quantum sampling.
    """
    logger = get_run_logger()

    # Create ansatz circuits
    logger.info("Creating ansatz circuit...")
    alpha_alpha_indices = [(p, p + 1) for p in range(elec_props.num_orbitals - 1)]
    alpha_beta_indices = [(p, p) for p in range(0, elec_props.num_orbitals, 4)]
    interaction_pairs = alpha_alpha_indices, alpha_beta_indices

    qreg = QuantumRegister(2 * elec_props.num_orbitals, name="q")
    creg_test = ClassicalRegister(2 * elec_props.num_orbitals, name="test")
    creg_meas = ClassicalRegister(2 * elec_props.num_orbitals, name="meas")

    regs = [qreg, creg_meas]
    if circuit_params.use_reset_mitigation:
        regs.append(creg_test)

    lucj_circ = QuantumCircuit(*regs)
    if circuit_params.use_reset_mitigation:
        lucj_circ.measure(qreg, creg_test)
        lucj_circ.barrier()
    lucj_circ.append(
        ffsim.qiskit.PrepareHartreeFockJW(
            norb=elec_props.num_orbitals,
            nelec=elec_props.num_electrons,
        ),
        qargs=qreg,
    )
    ucj_op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
        t2=elec_props.t2,
        n_reps=circuit_params.n_lucj_layers,
        interaction_pairs=interaction_pairs,
    )
    lucj_circ.append(
        ffsim.qiskit.UCJOpSpinBalancedJW(
            ucj_op=ucj_op,
        ),
        qargs=qreg,
    )
    lucj_circ.measure(qreg, creg_meas)

    logger.info(
        f"Untranspiled LUCJ circuit:\n"
        f"Number of qubits = {lucj_circ.num_qubits}\n"
        f"Number of classical bits = {lucj_circ.num_clbits}\n"
        f"Circuit depth = {lucj_circ.depth()}\n"
        f"Instruction counts = {dict(lucj_circ.count_ops())}\n"
    )

    return lucj_circ


@task
def transpile_circuit(
    circuit: QuantumCircuit,
    target: Target,
    circuit_params: CircuitParameters,
    seed: np.random.Generator,
) -> QuantumCircuit:
    """Custom transpiler with massive layout search."""
    logger = get_run_logger()

    if isinstance(seed, np.random.Generator):
        seed = seed.integers(np.iinfo(int).max, size=1).item()
    coupling_map = target.build_coupling_map()

    # Define a custom pass manager
    passmanager = generate_preset_pass_manager(
        optimization_level=circuit_params.optimization_level,
        seed_transpiler=seed,
        target=target,
    )
    passmanager.pre_init = PRE_INIT
    passmanager.layout = PassManager(
        [
            BarrierBeforeFinalMeasurements(
                label="qiskit.transpiler.internal.routing.protection.barrier",
            ),
            SabreLayout(
                coupling_map=coupling_map,
                seed=seed,
                max_iterations=circuit_params.sabre_max_iterations,
                layout_trials=circuit_params.sabre_layout_trials,
                swap_trials=circuit_params.sabre_swap_trials,
            ),
            ConditionalController(
                tasks=[
                    FullAncillaAllocation(coupling_map=coupling_map),
                    EnlargeWithAncilla(),
                    ApplyLayout(),
                ],
                condition=lambda propset: propset["final_layout"] is None,
            ),
        ]
    )
    if "rzz" in target.operation_names:
        passmanager.post_optimization = PassManager(
            [
                FoldRzzAngle(),
                Optimize1qGatesDecomposition(target=target),  # Cancel added local gates
                RemoveIdentityEquivalent(target=target),  # Remove GlobalPhaseGate
            ]
        )

    # Transpile
    logger.info("Transpiling ansatz circuit...")
    isa_circuit = passmanager.run(circuit)
    gate_depth = isa_circuit.depth(
        lambda inst: inst.operation.name not in ("rz", "barrier", "measure")
    )
    logger.info(
        f"Circuit depth = {gate_depth}\nInstruction counts = {dict(isa_circuit.count_ops())}\n"
    )
    return isa_circuit


@task(
    persist_result=True,
    result_serializer="compressed/pickle",
)
async def sample_bitstrings(
    circuit_params: CircuitParameters,
    elec_props: ElectronicProperties,
    runner_name: str,
    option_name: str,
    quantum_source: Literal["real-device", "random"],
    random_seed: int,
) -> BitArray:
    """Sample bitstring with quantum computer.

    Args:
        circuit_params: Circuit parameters for quantum sampling.
        elec_props: Electronic properties of molecule to solver for.
        runner_name: Block name of QuantumRuntime to execute primitives on.
        option_name: Name of Variable storing sampler primitive options to load.
        quantum_source: ``"real-device"`` or ``"random"``.
        random_seed: Seed for ``"random"`` sampling.

    Returns:
        Sampled bitstrings representing determinants.
    """
    logger = get_run_logger()
    rng = np.random.default_rng(24)

    options = await Variable.get(option_name, default={"params": {"shots": 100_000}})
    shots = _resolve_shots(sampler_options=options)

    if quantum_source == "random":
        logger.info(
            "Quantum source is random. Sampling %s deterministic pseudo-random bitstrings.",
            shots,
        )
        return generate_bit_array_uniform(
            shots,
            elec_props.num_orbitals * 2,
            rand_seed=random_seed,
        )

    try:
        runtime = await QuantumRuntime.load(runner_name)
    except ValueError as exc:
        raise RuntimeError(
            f"Quantum source 'real-device' requested but QuantumRuntime block "
            f"'{runner_name}' is not defined. Set quantum_source='random' to use "
            "deterministic random sampling instead."
        ) from exc

    # Create ansatz circuits
    lucj_circ = create_ansatz_circuits(
        circuit_params=circuit_params,
        elec_props=elec_props,
    )

    # Transpile
    target = await runtime.get_target()
    isa_circuit = transpile_circuit(
        circuit=lucj_circ,
        target=target,
        circuit_params=circuit_params,
        seed=rng,
    )

    # Run primitive
    pub_result = await runtime.sampler(
        sampler_pubs=[(isa_circuit,)],
        options=options,
    )

    # Post-process bitstrings
    meas_bits = pub_result[0].data.meas
    if circuit_params.use_reset_mitigation:
        test_bits = pub_result[0].data.test
        bit_array = meas_bits.get_bitstrings(test_bits.bitcount() == 0)
        bit_array = BitArray.from_samples(bit_array, num_bits=meas_bits.num_bits)
        logger.info(
            "Reset mitigation result:\n"
            f"  Before: {meas_bits.num_shots} bitstrings\n"
            f"  After: {bit_array.num_shots} bitstrings\n"
            f"  Retention rate: {bit_array.num_shots / meas_bits.num_shots}\n"
        )
    else:
        bit_array = meas_bits

    return bit_array


def deploy():
    """Deploy workflow with a local worker."""
    # Prefect deploys with relative path.
    # Workflow is now installed in site-packages.
    os.chdir(pathlib.Path(__file__).parent)

    sqd_2405_05068_fulqrum.with_options(version=os.getenv("WF_VERSION", "unknown")).serve(
        name=FLOW_NAME,
        description="SQD experiment from arXiv2405.05068 with the fulqrum eigensolver.",
    )
