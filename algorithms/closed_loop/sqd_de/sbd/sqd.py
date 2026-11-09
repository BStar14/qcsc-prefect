# Workflow for observability demo

import asyncio
from collections.abc import Mapping
from datetime import datetime, timezone
from time import perf_counter
from typing import Any

import numpy as np
from prefect import get_run_logger, task
from prefect.variables import Variable
from qcsc_prefect.integrations.qiskit.qrmi.contract import resolve_shots
from qcsc_prefect_dice import DiceSHCISolverJob
from qcsc_prefect_sbd import SBDSolverJob
from qcsc_workflow_utility import DICE_SLUG, SBD_SLUG, solver_class
from qcsc_workflow_utility.chem import ElectronicProperties, NpStrict1DArrayF64
from qiskit.primitives.containers import BitArray
from qiskit_addon_sqd.configuration_recovery import (
    post_select_by_hamming_weight,
)
from qiskit_addon_sqd.configuration_recovery import (
    recover_configurations as _recover_configurations,
)
from qiskit_addon_sqd.counts import bit_array_to_arrays

from .data_io import save_ndarray
from .flow_params import CircuitParameters
from .lucj import create_lucj_circuit
from .np_type_extension import (
    NpStrict1DArrayLL,
    NpStrict2DArrayBool,
)
from .sampling import load_quantum_source

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
    solver_block_ref: str,
    quantum_source: str,
    random_seed: int,
) -> tuple[tuple[float, NpStrict2DArrayBool], dict[str, Any]]:
    logger = get_run_logger()
    # Before any block is loaded, so that a bad Variable fails the walker first.
    shots = resolve_sqd_shots(Variable.get("sqd_options", default={}))

    solver_cls, solver_block_name = load_solver_class(solver_block_ref)
    davidson_solver = solver_cls.load(solver_block_name)

    telemetry = {
        "trial_index": trial_index,
        "walker_index": walker_index,
        **solver_record(solver_cls),
    }

    source = load_quantum_source(
        quantum_source,
        circuit_params=circuit_params,
        num_orbitals=elec_props.num_orbitals,
        random_seed=random_seed,
        trial_index=trial_index,
        walker_index=walker_index,
    )

    circuit = None
    if source.requires_circuit:
        circuit = create_lucj_circuit(
            ucj_parameter=ucj_parameter,
            elec_props=elec_props,
            aa_indices=aa_indices,
            ab_indices=ab_indices,
            n_lucj_layers=circuit_params.n_lucj_layers,
            use_reset_mitigation=circuit_params.use_reset_mitigation,
        )

    logger.info("Sampling from %s (shots=%s).", source.backend_name, shots)
    sampling_start = perf_counter()
    try:
        result = source.sample(circuit, shots=shots)
    except Exception:
        logger.exception("Sampling failed on backend %s.", source.backend_name)
        raise
    logger.info(
        "Completed sampling on %s in %.2fs.",
        source.backend_name,
        perf_counter() - sampling_start,
    )

    bit_array = result[0].data.meas
    if source.requires_circuit:
        # Measured on a device: filter here, the same way for every device source.
        bit_array, shot_retention_rate = mitigate_reset_errors(
            result[0].data, use_reset_mitigation=circuit_params.use_reset_mitigation
        )
        telemetry.update(shot_retention_rate=shot_retention_rate)

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
    # Run the eigensolver immediately. The Hartree-Fock determinant is first, as SBD
    # requires; DICE accepts the same input.
    sbd_result = asyncio.run(
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
    # DICE produces no carryover; the next generation then subsamples without it.
    next_carryover = getattr(
        sbd_result,
        "carryover_bitstrings",
        np.zeros((0, elec_props.num_orbitals), dtype=bool),
    )

    report_s3 = save_ndarray(
        file_prefix="sqd_data",
        ucj_parameter=ucj_parameter,
        raw_bitstrings=raw_bitstrings,
        recovered_bitstrings=bitstrings,
        alphadets=ci_strings,
        avg_occupancy=sbd_result.orbital_occupancies[0],
        carryover=next_carryover,
    )
    logger.debug(f"Saved SQD data in '{report_s3}'.")

    telemetry.update(
        num_post_determinants=len(bitstrings_post),
        net_subspace_dim=int(len(ci_strings) ** 2),
        energy=float(energy),
        sqd_data=str(report_s3),
        last_updated=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    )

    return ((energy, next_carryover), telemetry)


DEFAULT_SHOTS = 100_000


def resolve_sqd_shots(options) -> int:
    """Shots from the ``sqd_options`` Variable, with the same rules for every source.

    Only ``{"params": {"shots": <positive integer>}}`` is accepted; without shots it is
    ``DEFAULT_SHOTS``. ``null`` shots, floats, booleans, strings and other options are
    rejected rather than replaced or ignored.
    """
    if not isinstance(options, Mapping):
        raise TypeError("sqd_options must be a mapping")
    params = options.get("params", {})
    has_shots = isinstance(params, Mapping) and "shots" in params
    return resolve_shots(None if has_shots else DEFAULT_SHOTS, options)


def mitigate_reset_errors(data, *, use_reset_mitigation: bool) -> tuple[BitArray, float]:
    """The ``meas`` bits to use and the fraction of shots kept.

    With reset mitigation, keep the shots whose ``test`` register is all zeros. ``test``
    must be present then, and at least one shot must remain.
    """
    meas_bits = data.meas
    if not use_reset_mitigation:
        return meas_bits, 1.0
    if "test" not in data.keys():
        raise RuntimeError("Reset mitigation needs the 'test' register in the sampler result.")
    samples = meas_bits.get_bitstrings(data.test.bitcount() == 0)
    if not samples:
        raise RuntimeError("Reset mitigation removed all samples.")
    kept = BitArray.from_samples(samples, num_bits=meas_bits.num_bits)
    return kept, float(kept.num_shots / meas_bits.num_shots)


SOLVER_CLASSES = {SBD_SLUG: SBDSolverJob, DICE_SLUG: DiceSHCISolverJob}


def load_solver_class(solver_block_ref: str):
    """Solver block class selected by the slug of ``solver_block_ref``, and its name."""
    return solver_class(solver_block_ref, SOLVER_CLASSES)


def solver_record(solver_cls) -> dict[str, str]:
    """Solver and carryover mode recorded at the flow start and in the telemetry."""
    if solver_cls is DiceSHCISolverJob:
        # A different variant from SBD with carryover; results are not expected to match.
        return {"solver": "DICE", "carryover": "none"}
    return {"solver": "SBD", "carryover": "sbd"}


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
