"""Definition of SQD workflow."""

import asyncio
import dataclasses
import os
import pathlib
from typing import Any, Literal

import ffsim
import numpy as np
from prefect import flow, get_run_logger, task
from prefect.artifacts import create_table_artifact
from prefect.cache_policies import INPUTS, NO_CACHE
from prefect.variables import Variable
from pydantic import BaseModel, Field
from qcsc_prefect.integrations.qiskit.qrmi import QrmiRuntimeBlock
from qcsc_prefect_dice import DiceSHCISolverJob
from qcsc_workflow_utility import (
    ElectronicProperties,
    compute_molecular_integrals_from_fcidump,
    compute_molecular_integrals_from_geometry,
)
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.primitives.containers import BitArray
from qiskit_addon_sqd.configuration_recovery import recover_configurations
from qiskit_addon_sqd.counts import bit_array_to_arrays, generate_bit_array_uniform
from qiskit_addon_sqd.fermion import SCIResult

from sqd_dice.subsample import postselect, subsample
from sqd_dice.transpilation import TRANSPILER, prepare, transpile_config

DEFAULT_RUNTIME_BLOCK_NAME = "qrmi-runtime"


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
        description="Target value for the total spin squared for the ground state.",
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
        description="Target value for the total spin squared for the ground state.",
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
            "eigenstate solver."
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
    """Workflow parameters to configure sqd_2405_05068 workflow."""

    molecule: MoleculeGeometry | FCIDumpFile = Field(
        default_factory=MoleculeGeometry,
        description="PySCF definition of the molecule to solve for.",
        title="Molecule",
    )

    circuit: CircuitParameters = Field(
        default_factory=CircuitParameters,
        description="Ansatz circuit definition and transpiler settings.",
        title="Circuit",
    )

    sqd: SQDParameters = Field(
        default_factory=SQDParameters,
        description="Control of SQD algorithm execution and solver parameters.",
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


@flow
async def sqd_2405_05068(
    parameters: Parameters,
    runner_name: str = DEFAULT_RUNTIME_BLOCK_NAME,
    solver_name: str = "sqd-solver",
    option_name: str = "sampler_options",
    cache_compute_integrals: bool = False,
    cache_sampling: bool = False,
) -> float:
    """SQD Experiment from arXiv2405.05068.

    Args:
        parameters: Workflow parameters.
        runner_name: Name of QuantumRuntime block to load.
        solver_name: Name of DiceSHCISolverJob block to load.
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
    n_alpha, n_beta = elec_props.num_electrons

    # Run configuration recovery loop
    sci_solver = await DiceSHCISolverJob.load(solver_name)
    sqd_artifact = []
    current_occupancies = elec_props.initial_occupancy
    best_result = None
    for round_idx in range(parameters.sqd.max_iterations):
        bitstrings, probs = recover_configurations(
            raw_bitstrings,
            raw_probs,
            current_occupancies,
            n_alpha,
            n_beta,
            rand_seed=rng,
        )
        bitstrings_post, probs_post = postselect(
            bitstring_matrix=bitstrings,
            probabilities=probs,
            hamming_right=n_alpha,
            hamming_left=n_beta,
        )
        logger.info(f"Number of post-selected bitstrings: {len(bitstrings_post)}")
        if len(bitstrings_post) == 0:
            raise RuntimeError(
                "The input bit array did not contain any valid bitstrings. "
                "Pass a bit array that contains at least one valid bitstring."
            )
        # Convert bitstrings to CI strings and diagonalize
        coros = []
        subspace_dims = []
        for ci_strings in subsample(
            bitstring_matrix=bitstrings_post,
            probabilities=probs_post,
            subspace_dim=parameters.sqd.subspace_dim,
            num_batches=parameters.sqd.num_batches,
            rng=rng,
            open_shell=elec_props.open_shell,
        ):
            subspace_dims.append(int(len(ci_strings[0]) * len(ci_strings[1])))
            coro = sci_solver.run(
                ci_strings=ci_strings,
                one_body_tensor=elec_props.one_body_tensor,
                two_body_tensor=elec_props.two_body_tensor,
                norb=elec_props.num_orbitals,
                nelec=elec_props.num_electrons,
                spin_sq=elec_props.spin_sq,
            )
            coros.append(coro)
        results: list[SCIResult] = await asyncio.gather(*coros)
        # Get best result from batch
        best_result_in_batch = min(results, key=lambda result: result.energy)
        # Check if the energy is the lowest seen so far
        if best_result is None or best_result_in_batch.energy < best_result.energy:
            best_result = best_result_in_batch
        current_occupancies = best_result_in_batch.orbital_occupancies
        # Update artifact
        for subspace_idx, result in enumerate(results):
            new_record = dict(
                round=int(round_idx),
                subspace=int(subspace_idx),
                energy=float(result.energy + elec_props.nuclear_repulsion_energy),
                dimension=subspace_dims[subspace_idx],
            )
            sqd_artifact.append(new_record)
        logger.info(
            f"Best energy in SQD iteration {round_idx}: "
            f"{best_result_in_batch.energy + elec_props.nuclear_repulsion_energy} Ha."
        )

    # Upload artifact
    await create_table_artifact(
        table=sqd_artifact,
        key="sqd-configuration-recovery",
    )
    return best_result.energy + elec_props.nuclear_repulsion_energy


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
        runner_name: Name of QrmiRuntimeBlock to sample with.
        option_name: Name of Variable storing sampler primitive options to load.

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
        runtime = await QrmiRuntimeBlock.aload_runtime(runner_name)
    except ValueError as exc:
        raise RuntimeError(
            f"Quantum source 'real-device' requested but QrmiRuntimeBlock "
            f"'{runner_name}' could not be loaded. Set quantum_source='random' to use "
            "deterministic random sampling instead."
        ) from exc

    # Create ansatz circuits
    lucj_circ = create_ansatz_circuits(
        circuit_params=circuit_params,
        elec_props=elec_props,
    )

    # The runtime runs the ffsim pre-init stage in this process (prepare), then
    # transpiles against the actual Target with the SQD hook: in this process for
    # scheduler="iqp", on the compute node for "slurm". The block's transpiler and
    # transpile_config are replaced; the compilation report is in the result metadata
    # (iqp) or transpile_report.json (slurm).
    runtime = dataclasses.replace(
        runtime,
        prepare=prepare,
        transpiler=TRANSPILER,
        transpile_config=transpile_config(
            optimization_level=circuit_params.optimization_level,
            seed_transpiler=rng.integers(np.iinfo(int).max, size=1).item(),
            sabre_max_iterations=circuit_params.sabre_max_iterations,
            sabre_swap_trials=circuit_params.sabre_swap_trials,
            sabre_layout_trials=circuit_params.sabre_layout_trials,
        ),
    )

    # Run primitive
    logger.info("Sampling the LUCJ circuit on %s", runtime.resource)
    pub_result = await runtime.sampler(
        sampler_pubs=[(lucj_circ,)],
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

    sqd_2405_05068.with_options(version=os.getenv("WF_VERSION", "unknown")).serve(
        name="sqd_2405_05068",
        description="SQD experiment from arXiv2405.05068.",
    )
