"""Post-process stage: configuration recovery and subspace diagonalization."""

from typing import Protocol

import numpy as np
from prefect import get_run_logger, task
from prefect.artifacts import create_table_artifact
from prefect.cache_policies import NO_CACHE
from qcsc_workflow_utility import ElectronicProperties
from qiskit.primitives.containers import BitArray
from qiskit_addon_sqd.counts import bit_array_to_arrays
from qiskit_addon_sqd.fermion import SCIResult

from sqd_common.flow_params import SQDParameters
from sqd_common.recovery import SolveFn, run_configuration_recovery

# Seed of the configuration recovery and subsampling RNG.
RECOVERY_SEED = 24


class SolverBlock(Protocol):
    """A solver block (DICE, fulqrum) that diagonalizes one CI subspace per call."""

    async def run(
        self,
        ci_strings: tuple[np.ndarray, np.ndarray],
        one_body_tensor: np.ndarray,
        two_body_tensor: np.ndarray,
        norb: int,
        nelec: tuple[int, int],
        spin_sq: float | None = None,
    ) -> SCIResult: ...


def make_block_solver(solver: SolverBlock, elec_props: ElectronicProperties) -> SolveFn:
    """Bind the molecular integrals to ``solver.run`` for the recovery loop."""

    async def solve(ci_strings: tuple[np.ndarray, np.ndarray]) -> SCIResult:
        return await solver.run(
            ci_strings=ci_strings,
            one_body_tensor=elec_props.one_body_tensor,
            two_body_tensor=elec_props.two_body_tensor,
            norb=elec_props.num_orbitals,
            nelec=elec_props.num_electrons,
            spin_sq=elec_props.spin_sq,
        )

    return solve


@task(name="sqd_loop", tags=["post-process"], cache_policy=NO_CACHE)
async def sqd_loop(
    bool_array: np.ndarray,
    elec_props: ElectronicProperties,
    sqd_params: SQDParameters,
    solver: SolverBlock,
    artifact_key: str,
) -> float:
    """Self-consistent configuration recovery and subspace diagonalization.

    Each round recovers configurations with the current orbital occupancies,
    post-selects on the electron numbers, draws ``num_batches`` subspaces and
    diagonalizes them concurrently with the solver block, which runs each one
    as an HPC job.

    The task is never cached: a retried flow run reuses the persisted samples
    and integrals and repeats only this stage.

    Args:
        bool_array: Sampled bitstrings as a 2D bool array (one row per shot).
        elec_props: Electronic properties of the molecule.
        sqd_params: Subspace dimension, batches and recovery rounds.
        solver: Solver block that diagonalizes one subspace per call.
        artifact_key: Key of the table artifact with one row per solved subspace.

    Returns:
        Lowest energy found, including the nuclear repulsion energy.
    """
    logger = get_run_logger()
    raw_bitstrings, raw_probs = bit_array_to_arrays(BitArray.from_bool_array(bool_array))
    outcome = await run_configuration_recovery(
        raw_bitstrings,
        raw_probs,
        solve=make_block_solver(solver, elec_props),
        initial_occupancies=elec_props.initial_occupancy,
        nelec=elec_props.num_electrons,
        open_shell=elec_props.open_shell,
        subspace_dim=sqd_params.subspace_dim,
        num_batches=sqd_params.num_batches,
        max_iterations=sqd_params.max_iterations,
        rng=np.random.default_rng(RECOVERY_SEED),
        nuclear_repulsion_energy=elec_props.nuclear_repulsion_energy,
        logger=logger,
    )
    await create_table_artifact(table=outcome.records, key=artifact_key)
    return outcome.best_energy
