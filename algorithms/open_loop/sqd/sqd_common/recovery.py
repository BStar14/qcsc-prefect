"""Self-consistent configuration recovery loop of SQD.

This module has no Prefect dependency. The eigensolver is passed in as an async
callable, so the flows can use a solver block while tests can use an
in-process solver.

The best result of each round seeds the next one twice: its orbital occupancies
drive the configuration recovery, and its carryover CI strings, the heaviest
strings of its ground state, are kept in every subspace of the next round.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field

import numpy as np
from qcsc_prefect_fulqrum.carryover import marginal_weights, select_carryover
from qiskit_addon_sqd.configuration_recovery import recover_configurations
from qiskit_addon_sqd.fermion import SCIResult

from sqd_common.subsample import postselect, subsample

SolveFn = Callable[[tuple[np.ndarray, np.ndarray]], Awaitable[SCIResult]]
# Maps the subsampled batches of a round, and the loop's RNG, to the batches solved.
BatchTransformFn = Callable[
    [Iterable[tuple[np.ndarray, np.ndarray]], np.random.Generator],
    Iterable[tuple[np.ndarray, np.ndarray]],
]
# Called after each round with (next_round, occupancies for it, best total energy so far,
# carryover CI strings for it).
RoundEndFn = Callable[
    [int, tuple[np.ndarray, np.ndarray], float, tuple[np.ndarray, np.ndarray]], None
]


def empty_carryover() -> tuple[np.ndarray, np.ndarray]:
    """No carryover CI strings for either spin."""
    return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)


def carryover_of(result: SCIResult, fraction: float, norb: int) -> tuple[np.ndarray, np.ndarray]:
    """The (alpha, beta) carryover CI strings of a solver result, heaviest first.

    The first source the result has is used:

    1. ``result.carryover``, which the fulqrum solver block sets.
    2. ``carryover_bitstrings`` of an SBD result, one bool row per alpha string
       (column ``p`` is orbital ``norb - 1 - p``), used for both spins.
    3. The heaviest ``fraction`` of each spin's strings in ``result.sci_state``
       (DICE, ``solve_fermion``), by marginal weight.

    Without any of them, or with ``fraction`` 0, the carryover is empty.
    """
    return _carryover_and_source(result, fraction, norb)[0]


def _carryover_and_source(
    result: SCIResult, fraction: float, norb: int
) -> tuple[tuple[np.ndarray, np.ndarray], str | None]:
    if fraction <= 0:
        return empty_carryover(), "off"
    carryover = getattr(result, "carryover", None)
    if carryover is not None:
        return (np.asarray(carryover[0]), np.asarray(carryover[1])), "carryover"
    rows = getattr(result, "carryover_bitstrings", None)
    if rows is not None:
        strings = _bool_rows_to_ints(np.asarray(rows, dtype=bool), norb)
        return (strings, strings.copy()), "carryover_bitstrings"
    try:
        state = getattr(result, "sci_state", None)
    except NotImplementedError:
        state = None
    amplitudes = getattr(state, "amplitudes", None)
    if amplitudes is not None:
        strs_a, strs_b = np.asarray(state.ci_strs_a), np.asarray(state.ci_strs_b)
        probs = np.abs(np.asarray(amplitudes)) ** 2
        w_a, w_b = marginal_weights(probs, len(strs_a), len(strs_b))
        return select_carryover(w_a, w_b, strs_a, strs_b, fraction), "sci_state"
    return empty_carryover(), None


def _bool_rows_to_ints(rows: np.ndarray, norb: int) -> np.ndarray:
    """CI strings of ``(n, norb)`` bool rows whose first column is orbital ``norb - 1``."""
    dtype = np.int64 if norb < 64 else object
    strings = np.zeros(len(rows), dtype=dtype)
    for p in range(min(norb, rows.shape[1] if rows.ndim == 2 else 0)):
        strings += rows[:, p].astype(dtype) * (1 << (norb - 1 - p))
    return strings


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
        final_carryover: Carryover CI strings for the round after the last one run.
    """

    best_result: SCIResult | None
    best_energy: float
    round_energies: list[float] = field(default_factory=list)
    records: list[dict] = field(default_factory=list)
    final_carryover: tuple[np.ndarray, np.ndarray] = field(default_factory=empty_carryover)


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
    batch_transform: BatchTransformFn | None = None,
    initial_carryover: tuple[np.ndarray, np.ndarray] | None = None,
    carryover_fraction: float = 0.0,
) -> RecoveryOutcome:
    """Run the SQD configuration recovery loop.

    Each round recovers configurations with the current orbital occupancies,
    post-selects on the electron numbers, draws ``num_batches`` subspaces and
    solves them concurrently with ``solve``. The occupancies of the best result in
    the round seed the next round, as in arXiv:2405.05068. With a
    ``carryover_fraction`` above 0, the carryover CI strings of that result are
    also kept in every subspace of the next round, within ``subspace_dim``.

    A resumed run passes the occupancies of its checkpoint as
    ``initial_occupancies``, its carryover as ``initial_carryover``, the number of
    completed rounds as ``start_round`` and their best total energy as
    ``best_energy``. It then runs rounds ``start_round`` to ``max_iterations - 1``
    and never reports a higher energy.

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
        batch_transform: Rewrites the subsampled batches of each round before they are
            solved, for a solver that needs another input form (SBD). It draws from
            ``rng`` after the subsampling, so the subspaces of the next round change.
        initial_carryover: Alpha and beta carryover CI strings for the first round.
        carryover_fraction: Share of each spin's strings kept as carryover, read from
            the best result of each round with :func:`carryover_of`. ``0`` uses no
            carryover, not even ``initial_carryover``.

    Raises:
        RuntimeError: If no bitstring survives post-selection.
    """
    logger = logger or logging.getLogger(__name__)
    n_alpha, n_beta = nelec
    current_occupancies = initial_occupancies
    if carryover_fraction <= 0 or initial_carryover is None:
        current_carryover = empty_carryover()
    else:
        current_carryover = (
            np.asarray(initial_carryover[0]),
            np.asarray(initial_carryover[1]),
        )
    norb = raw_bitstrings.shape[1] // 2
    warned_no_carryover = False
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

        batches = subsample(
            bitstring_matrix=bitstrings_post,
            probabilities=probs_post,
            subspace_dim=subspace_dim,
            num_batches=num_batches,
            rng=rng,
            open_shell=open_shell,
            carryover=current_carryover if _has_carryover(current_carryover) else None,
            nelec=nelec,
        )
        if batch_transform is not None:
            batches = batch_transform(batches, rng)
        batches = list(batches)
        results: list[SCIResult] = await asyncio.gather(*(solve(ci) for ci in batches))

        best_result_in_batch = min(results, key=lambda result: result.energy)
        round_energy = float(best_result_in_batch.energy + nuclear_repulsion_energy)
        if best_energy is None or round_energy < best_energy:
            best_result = best_result_in_batch
            best_energy = round_energy
        current_occupancies = best_result_in_batch.orbital_occupancies
        round_carryover = current_carryover
        current_carryover, source = _carryover_and_source(
            best_result_in_batch, carryover_fraction, norb
        )
        if source is None and not warned_no_carryover:
            warned_no_carryover = True
            logger.warning(
                f"carryover_fraction={carryover_fraction}, but the solver returned no "
                "carryover strings or ground state; the next rounds use no carryover."
            )

        for subspace_idx, (ci_strings, result) in enumerate(zip(batches, results, strict=True)):
            outcome_records.append(
                dict(
                    round=int(round_idx),
                    subspace=int(subspace_idx),
                    energy=float(result.energy + nuclear_repulsion_energy),
                    dimension=int(len(ci_strings[0]) * len(ci_strings[1])),
                    num_ci_strs_a=int(len(ci_strings[0])),
                    num_ci_strs_b=int(len(ci_strings[1])),
                    num_carryover_a=int(len(round_carryover[0])),
                    num_carryover_b=int(len(round_carryover[1])),
                )
            )
        round_energies.append(round_energy)
        logger.info(
            f"Best energy in SQD iteration {round_idx + 1}/{max_iterations}: {round_energy} Ha."
        )
        if on_round_end is not None:
            on_round_end(round_idx + 1, current_occupancies, best_energy, current_carryover)

    if best_energy is None:
        raise ValueError("max_iterations must be at least 1.")
    return RecoveryOutcome(
        best_result=best_result,
        best_energy=float(best_energy),
        round_energies=round_energies,
        records=outcome_records,
        final_carryover=current_carryover,
    )


def _has_carryover(carryover: tuple[np.ndarray, np.ndarray]) -> bool:
    return bool(len(carryover[0]) or len(carryover[1]))
