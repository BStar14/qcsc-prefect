"""Self-consistent configuration recovery loop of SQD.

This module has no Prefect dependency. The eigensolver is passed in as an async
callable, so the flows can use a solver block while tests can use an
in-process solver.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

import numpy as np
from qiskit_addon_sqd.configuration_recovery import recover_configurations
from qiskit_addon_sqd.fermion import SCIResult

from sqd_common.subsample import postselect, subsample

SolveFn = Callable[[tuple[np.ndarray, np.ndarray]], Awaitable[SCIResult]]
# Called after each round with (next_round, occupancies for it, best total energy so far).
RoundEndFn = Callable[[int, tuple[np.ndarray, np.ndarray], float], None]


@dataclass
class RecoveryOutcome:
    """Result of :func:`run_configuration_recovery`.

    Attributes:
        best_result: Lowest-energy result of the rounds run in this call. Its energy is
            electronic only. ``None`` when a resumed run found nothing below ``best_energy``
            it started from, or ran no round.
        best_energy: Lowest total energy, including the nuclear repulsion energy and
            the ``best_energy`` a resumed run started from.
        round_energies: Best total energy of each round run in this call.
        records: One row per solved subspace, for the Prefect table artifact.
    """

    best_result: SCIResult | None
    best_energy: float
    round_energies: list[float] = field(default_factory=list)
    records: list[dict] = field(default_factory=list)


async def run_configuration_recovery(
    raw_bitstrings: np.ndarray,
    raw_probs: np.ndarray,
    *,
    solve: SolveFn,
    initial_occupancies: tuple[np.ndarray, np.ndarray],
    nelec: tuple[int, int],
    open_shell: bool,
    subspace_dim: int,
    num_batches: int,
    max_iterations: int,
    rng: np.random.Generator,
    nuclear_repulsion_energy: float = 0.0,
    logger: logging.Logger | logging.LoggerAdapter | None = None,
    start_round: int = 0,
    best_energy: float | None = None,
    on_round_end: RoundEndFn | None = None,
) -> RecoveryOutcome:
    """Run the SQD configuration recovery loop.

    Each round recovers configurations with the current orbital occupancies,
    post-selects on the electron numbers, draws ``num_batches`` subspaces and
    solves them concurrently with ``solve``. The occupancies of the best result in
    the round seed the next round, as in arXiv:2405.05068.

    A resumed run passes the occupancies of its checkpoint as
    ``initial_occupancies``, the number of completed rounds as ``start_round`` and
    their best total energy as ``best_energy``. It then runs rounds
    ``start_round`` to ``max_iterations - 1`` and never reports a higher energy.

    Args:
        raw_bitstrings: Sampled bitstring matrix (rows are shots, right half alpha).
        raw_probs: Probability of each row.
        solve: Async callable mapping ``(ci_strs_a, ci_strs_b)`` to an ``SCIResult``.
        initial_occupancies: Alpha and beta orbital occupancies for the first round.
        nelec: Number of alpha and beta electrons.
        open_shell: Keep alpha and beta CI strings separate when subsampling.
        subspace_dim: Target subspace dimension per batch.
        num_batches: Number of subspaces solved per round.
        max_iterations: Number of recovery rounds.
        rng: Random generator shared by recovery and subsampling.
        nuclear_repulsion_energy: Added to reported energies only.
        logger: Logger for progress messages.
        start_round: Number of rounds already completed by a checkpointed run.
        best_energy: Best total energy of those rounds.
        on_round_end: Called after each round, for example to write a checkpoint.

    Raises:
        RuntimeError: If no bitstring survives post-selection.
    """
    logger = logger or logging.getLogger(__name__)
    n_alpha, n_beta = nelec
    current_occupancies = initial_occupancies
    best_result: SCIResult | None = None
    outcome_records: list[dict] = []
    round_energies: list[float] = []
    if start_round >= max_iterations and best_energy is not None:
        logger.info(
            f"All {max_iterations} SQD iterations were already completed; "
            f"best energy {best_energy} Ha."
        )

    for round_idx in range(start_round, max_iterations):
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

        batches = list(
            subsample(
                bitstring_matrix=bitstrings_post,
                probabilities=probs_post,
                subspace_dim=subspace_dim,
                num_batches=num_batches,
                rng=rng,
                open_shell=open_shell,
            )
        )
        results: list[SCIResult] = await asyncio.gather(*(solve(ci) for ci in batches))

        best_result_in_batch = min(results, key=lambda result: result.energy)
        round_energy = float(best_result_in_batch.energy + nuclear_repulsion_energy)
        if best_energy is None or round_energy < best_energy:
            best_result = best_result_in_batch
            best_energy = round_energy
        current_occupancies = best_result_in_batch.orbital_occupancies

        for subspace_idx, (ci_strings, result) in enumerate(zip(batches, results, strict=True)):
            outcome_records.append(
                dict(
                    round=int(round_idx),
                    subspace=int(subspace_idx),
                    energy=float(result.energy + nuclear_repulsion_energy),
                    dimension=int(len(ci_strings[0]) * len(ci_strings[1])),
                    num_ci_strs_a=int(len(ci_strings[0])),
                    num_ci_strs_b=int(len(ci_strings[1])),
                )
            )
        round_energies.append(round_energy)
        logger.info(
            f"Best energy in SQD iteration {round_idx + 1}/{max_iterations}: {round_energy} Ha."
        )
        if on_round_end is not None:
            on_round_end(round_idx + 1, current_occupancies, best_energy)

    if best_energy is None:
        raise ValueError("max_iterations must be at least 1.")
    return RecoveryOutcome(
        best_result=best_result,
        best_energy=float(best_energy),
        round_energies=round_energies,
        records=outcome_records,
    )
