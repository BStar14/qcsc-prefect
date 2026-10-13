# Workflow for observability demo

import asyncio
from datetime import datetime, timezone
from time import perf_counter
from typing import Any

import numpy as np
from prefect import get_run_logger, task
from prefect.variables import Variable
from prefect_qiskit import QuantumRuntime
from qcsc_workflow_utility.chem import ElectronicProperties, NpStrict1DArrayF64
from qiskit.primitives.containers import BitArray
from qiskit_addon_sqd.configuration_recovery import (
    post_select_by_hamming_weight,
)
from qiskit_addon_sqd.configuration_recovery import (
    recover_configurations as _recover_configurations,
)
from qiskit_addon_sqd.counts import bit_array_to_arrays, generate_bit_array_uniform

from .data_io import save_ndarray
from .flow_params import CircuitParameters
from .lucj import create_lucj_circuit
from .np_type_extension import (
    NpStrict1DArrayLL,
    NpStrict2DArrayBool,
)
# from .qfw_resource import QFwResource
from .solver_job import SBDResult, SBDSolverJob
from .transpile_custom import find_optimal_layout, transpile_circuit

# Convert Addon function into Prefect Task
recover_configurations = task(_recover_configurations)

MODULE_RNG = np.random.default_rng(seed=1333)


@task(
    task_run_name="run_sqd_#{trial_index:02d}-{walker_index}",
)
def walker_sqd(
    trial_index: int,
    walker_index: int,
    ucj_parameter: NpStrict1DArrayF64,
    circuit_params: CircuitParameters,
    elec_props: ElectronicProperties,
    aa_indices: list[tuple[int, int]],
    ab_indices: list[tuple[int, int]],
    carryover: NpStrict2DArrayBool,
    sqd_dim: int,
    solver_block_name: str,
    quantum_source: str,
    random_seed: int,
) -> tuple[tuple[float, NpStrict2DArrayBool], dict[str, Any]]:
    logger = get_run_logger()
    davidson_solver = SBDSolverJob.load(solver_block_name)

    telemetry = {
        "trial_index": trial_index,
        "walker_index": walker_index,
    }

    options = Variable.get("sqd_options", default={"params": {"shots": 100_000}})
    shots = int(options.get("params", {}).get("shots") or 100_000)

    runtime = None
    qfw = None
    resource_name = None

    if quantum_source == "real-device":
        try:
            runtime = QuantumRuntime.load("ibm-runner")
        except ValueError as exc:
            raise RuntimeError(
                "Quantum source 'real-device' requested but QuantumRuntime block "
                "'ibm-runner' is not defined. Set quantum_source='random' to use "
                "deterministic random sampling instead."
            ) from exc

        resource_name = runtime.resource_name

    elif quantum_source == "qfw":
        try:
            from .qfw_resource import QFwResource
        except ImportError as exc:
            raise RuntimeError(
                "Quantum source 'qfw' requires the QFw environment and dependencies."
            ) from exc

        try:
            qfw = QFwResource.load("qfw-runner")
        except ValueError as exc:
            raise RuntimeError(
                "Quantum source 'qfw' requested but QFwResource block 'qfw-runner' is not defined."
            ) from exc

        resource_name = qfw.device_id

    if quantum_source in ("real-device", "qfw"):
        logger.info("Preparing quantum sampling on backend %s.", resource_name)

        target_start = perf_counter()
        if runtime is not None:
            target = runtime.get_target()
        else:
            target = qfw.get_target()

        logger.info(
            "Loaded backend target for %s in %.2fs.",
            resource_name,
            perf_counter() - target_start,
        )

        vir_circuit = create_lucj_circuit(
            ucj_parameter=ucj_parameter,
            elec_props=elec_props,
            aa_indices=aa_indices,
            ab_indices=ab_indices,
            n_lucj_layers=circuit_params.n_lucj_layers,
            use_reset_mitigation=circuit_params.use_reset_mitigation,
        )

        logger.info(
            "Searching ISA layout for backend %s "
            "(max_iterations=%s, swap_trials=%s, layout_trials=%s).",
            resource_name,
            circuit_params.sabre_max_iterations,
            circuit_params.sabre_swap_trials,
            circuit_params.sabre_layout_trials,
        )
        layout = find_optimal_layout(
            test_circuit=vir_circuit,
            target=target,
            optimization_level=circuit_params.optimization_level,
            max_iterations=circuit_params.sabre_max_iterations,
            swap_trials=circuit_params.sabre_swap_trials,
            layout_trials=circuit_params.sabre_layout_trials,
        )

        logger.info("Transpiling ISA circuit for backend %s.", resource_name)
        transpile_start = perf_counter()
        isa_circuit = transpile_circuit(
            circuit=vir_circuit,
            target=target,
            layout=layout,
            optimization_level=circuit_params.optimization_level,
        )
        logger.info(
            "Completed ISA transpilation for %s in %.2fs.",
            resource_name,
            perf_counter() - transpile_start,
        )

        logger.info(
            "Submitting sampler workload to %s (shots=%s).",
            resource_name,
            shots,
        )
        sampling_start = perf_counter()

        if runtime is not None:
            try:
                pub_result = runtime.sampler(
                    sampler_pubs=[(isa_circuit,)],
                    options=options,
                    tags=["res: quantum"],
                )
            except Exception:
                logger.exception(
                    "Sampler submission or execution failed for backend %s.",
                    resource_name,
                )
                raise

            meas_bits = pub_result[0].data.meas
            if circuit_params.use_reset_mitigation:
                test_bits = pub_result[0].data.test
                bit_array = meas_bits.get_bitstrings(test_bits.bitcount() == 0)
                bit_array = BitArray.from_samples(
                    bit_array,
                    num_bits=meas_bits.num_bits,
                )
            else:
                bit_array = meas_bits

            shot_retention_rate = float(bit_array.num_shots / meas_bits.num_shots)

        else:
            try:
                bit_array = qfw.run_sampler(
                    isa_circuit,
                    num_shots=shots,
                    use_reset_mitigation=circuit_params.use_reset_mitigation,
                )
            except Exception:
                logger.exception(
                    "QFw sampler submission or execution failed for device %s.",
                    resource_name,
                )
                raise

            shot_retention_rate = float(bit_array.num_shots / shots)

        logger.info(
            "Completed sampler workload on %s in %.2fs.",
            resource_name,
            perf_counter() - sampling_start,
        )

        telemetry.update(
            shot_retention_rate=shot_retention_rate,
        )

    elif quantum_source == "random":
        # Random sampling
        # Isolate bitstring seed from the module seed for equivalent control
        # with quantum-device sampling path.
        seed = int(
            random_seed
            + ((trial_index + walker_index) * (trial_index + walker_index + 1) // 2 + walker_index)
        )
        logger.info("Sampling bitstrings with RNG seed %s", seed)
        bit_array = generate_bit_array_uniform(
            num_samples=shots,
            num_bits=elec_props.num_orbitals * 2,
            rand_seed=seed,
        )

    else:
        raise ValueError(
            f"Unsupported quantum_source: {quantum_source!r}. "
            "Expected 'random', 'real-device' or 'qfw'."
        )

    logger.debug("Starting configuration recovery and diagonalization.")
    raw_bitstrings, raw_probs = bit_array_to_arrays(bit_array)
    bitstrings, probs = recover_configurations(
        bitstring_matrix=raw_bitstrings,
        probabilities=raw_probs,
        avg_occupancies=elec_props.initial_occupancy,
        num_elec_a=elec_props.num_electrons[0],
        num_elec_b=elec_props.num_electrons[1],
        rand_seed=MODULE_RNG,
    )
    bitstrings_post, probs_post = postselect_bitstrings(
        bitstring_matrix=bitstrings,
        probabilities=probs,
        hamming_right=elec_props.num_electrons[0],
        hamming_left=elec_props.num_electrons[1],
    )
    ci_strings = subsample_close_shell(
        bitstring_matrix=bitstrings_post,
        probabilities=probs_post,
        carryover=carryover,
        subspace_dim=sqd_dim,
        norb=elec_props.num_orbitals,
        num_elec_a=elec_props.num_electrons[0],
    )
    # Run SBD immediately
    sbd_result: SBDResult = asyncio.run(
        davidson_solver.run(
            ci_strings=(ci_strings, ci_strings),
            one_body_tensor=elec_props.one_body_tensor,
            two_body_tensor=elec_props.two_body_tensor,
            norb=elec_props.num_orbitals,
            nelec=elec_props.num_electrons,
        )
    )
    logger.debug("Completed diagonalization.")
    energy = sbd_result.energy + elec_props.nuclear_repulsion_energy

    report_s3 = save_ndarray(
        file_prefix="sqd_data",
        ucj_parameter=ucj_parameter,
        raw_bitstrings=raw_bitstrings,
        recovered_bitstrings=bitstrings,
        alphadets=ci_strings,
        avg_occupancy=sbd_result.orbital_occupancies[0],
        carryover=sbd_result.carryover_bitstrings,
    )
    logger.debug(f"Saved SQD data in '{report_s3}'.")

    telemetry.update(
        num_post_determinants=len(bitstrings_post),
        net_subspace_dim=int(len(ci_strings) ** 2),
        energy=float(energy),
        sqd_data=str(report_s3),
        last_updated=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    )

    return ((energy, sbd_result.carryover_bitstrings), telemetry)


@task
def postselect_bitstrings(
    bitstring_matrix: NpStrict2DArrayBool,
    probabilities: NpStrict1DArrayF64,
    *,
    hamming_right: int,
    hamming_left: int,
) -> tuple[NpStrict2DArrayBool, NpStrict1DArrayF64]:
    mask_postsel = post_select_by_hamming_weight(
        bitstring_matrix,
        hamming_right=hamming_right,
        hamming_left=hamming_left,
    )
    bs_mat_postsel = bitstring_matrix[mask_postsel]
    probs_postsel = probabilities[mask_postsel]
    probs_postsel = np.abs(probs_postsel) / np.sum(np.abs(probs_postsel))

    return bs_mat_postsel, probs_postsel


@task
def subsample_close_shell(
    bitstring_matrix: NpStrict2DArrayBool,
    probabilities: NpStrict1DArrayF64,
    carryover: NpStrict2DArrayBool,
    subspace_dim: int,
    norb: int,
    num_elec_a: int,
) -> NpStrict1DArrayLL:
    global MODULE_RNG

    num_configs = bitstring_matrix.shape[0]
    num_carryover = carryover.shape[0]

    # Make sure the Hartree Fock is included at index 0 of determinants.
    # This is requirement of the SBD solver.
    # The Hartree Fock bitstring is something like '0000011111'
    hartreefock = (1 << num_elec_a) - 1

    # Assume longlong is 64 bit integer.
    # Bit at index > 64 overflows.
    assert norb < 64

    ci_strs_a = np.zeros(num_configs, dtype=np.longlong)
    ci_strs_b = np.zeros(num_configs, dtype=np.longlong)
    ci_strs_carryover = np.zeros(num_carryover, dtype=np.longlong)

    # For performance, we accumulate the left and right CI strings together, column-wise,
    # across the two halves of the input bitstring matrix.
    for i in range(norb):
        ci_strs_b[:] += bitstring_matrix[:, i] * 2 ** (norb - 1 - i)
        ci_strs_a[:] += bitstring_matrix[:, norb + i] * 2 ** (norb - 1 - i)
        ci_strs_carryover[:] += carryover[:, i] * 2 ** (norb - 1 - i)
    mixed_ci_strigs = np.concatenate((ci_strs_a, ci_strs_b))

    # Reduce duplicated elements from CI strings and accumurate probabilities.
    ci_strs_unique, ci_probs_unique = _deduplicate_and_accumurate_probs(
        ci_strings=mixed_ci_strigs,
        probabilities=np.tile(probabilities, 2) / 2.0,
    )

    # Remove HF string to make sure it appears at index 0
    non_hf_mask = ci_strs_unique != hartreefock
    ci_strs_carryover = ci_strs_carryover[ci_strs_carryover != hartreefock]

    num_new_samples = int(np.sqrt(subspace_dim)) - len(ci_strs_carryover) - 1
    if len(ci_strs_unique) > num_new_samples:
        # Choose bitstrings not included in carryover bitstrings
        # Subspace dimension must be preserved
        non_co_mask = ~np.isin(ci_strs_unique, ci_strs_carryover)
        mask = non_hf_mask & non_co_mask
        ci_strs_unique = ci_strs_unique[mask]
        ci_probs_unique = ci_probs_unique[mask]
        new_strings = MODULE_RNG.choice(
            ci_strs_unique,
            size=num_new_samples,
            replace=False,
            p=ci_probs_unique / ci_probs_unique.sum(),
        )
    else:
        new_strings = ci_strs_unique[non_hf_mask]

    # Carryover bitstrings are always included
    return np.concatenate(([hartreefock], ci_strs_carryover, new_strings), dtype=np.longlong)


def _deduplicate_and_accumurate_probs(
    ci_strings: NpStrict1DArrayLL,
    probabilities: NpStrict1DArrayF64,
) -> tuple[NpStrict1DArrayLL, NpStrict1DArrayF64]:
    ci_strs_unique, ci_strs_inv = np.unique(
        ci_strings,
        return_inverse=True,
    )
    ci_probs_unique = np.bincount(
        ci_strs_inv,
        weights=probabilities,
        minlength=len(ci_strs_unique),
    )
    return ci_strs_unique, ci_probs_unique
