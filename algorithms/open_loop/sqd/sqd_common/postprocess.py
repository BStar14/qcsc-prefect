"""Post-process stage: configuration recovery and subspace diagonalization."""

import time
from pathlib import Path
from typing import Protocol

import numpy as np
from prefect import get_run_logger, task
from prefect.artifacts import create_table_artifact
from prefect.cache_policies import NO_CACHE
from prefect.runtime import flow_run
from qcsc_workflow_utility import ElectronicProperties
from qiskit.primitives.containers import BitArray
from qiskit_addon_sqd.counts import bit_array_to_arrays
from qiskit_addon_sqd.fermion import SCIResult

from sqd_common.checkpoint import (
    CKPT_NAME,
    RecoveryCheckpoint,
    key_mismatch,
    load_checkpoint,
    read_source,
    resume_key,
    save_checkpoint,
    write_source,
)
from sqd_common.flow_params import SQDParameters
from sqd_common.recovery import BatchTransformFn, SolveFn, run_configuration_recovery

# Seed of the configuration recovery and subsampling RNG.
RECOVERY_SEED = 24
CHECKPOINT_ARTIFACT_KEY = "sqd-checkpoint"
# Folder under the solver block's root_dir that holds one checkpoint folder per flow run.
CHECKPOINT_SUBDIR = "sqd_checkpoints"


class SolverBlock(Protocol):
    """A solver block (fulqrum, DICE, SBD) that diagonalizes one CI subspace per call.

    The recovery loop reads only ``energy`` and ``orbital_occupancies`` of the result.
    """

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


def checkpoint_run_dir(solver: SolverBlock) -> Path:
    """Checkpoint folder of this flow run: ``<root_dir>/sqd_checkpoints/<flow run id>``.

    A Retry keeps the flow run ID, so the retried run finds its own checkpoint.
    """
    root = Path(getattr(solver, "root_dir", None) or ".").expanduser().resolve()
    run_id = flow_run.get_id() or time.strftime("local-%Y%m%d-%H%M%S")
    return root / CHECKPOINT_SUBDIR / run_id


def _own_checkpoint(run_dir: Path, key: dict, logger) -> RecoveryCheckpoint | None:
    """The checkpoint a retried run left, if it belongs to the same samples and settings."""
    path = run_dir / CKPT_NAME
    if not path.is_file():
        return None
    diff = key_mismatch(read_source(run_dir) or {}, key)
    if not diff:
        try:
            return load_checkpoint(path)
        except Exception as exc:  # a corrupt checkpoint must never block a fresh start
            diff = [f"unreadable: {type(exc).__name__}"]
    aside = run_dir.with_name(f"{run_dir.name}.prev-{time.strftime('%Y%m%d-%H%M%S')}")
    run_dir.rename(aside)
    logger.warning(f"[ckpt] checkpoint not resumed (other {', '.join(diff)}); moved to {aside}")
    return None


def _given_checkpoint(value: str, key: dict) -> tuple[Path, RecoveryCheckpoint]:
    """The checkpoint named by ``resume_checkpoint``, which must match this run."""
    path = Path(value).expanduser()
    if path.is_dir():
        path = path / CKPT_NAME
    if not path.is_file():
        raise FileNotFoundError(f"resume_checkpoint: no checkpoint file at {path}.")
    source = read_source(path.parent)
    if source is None:
        raise ValueError(
            f"resume_checkpoint: {path.parent} has no source.json, so its samples and "
            "settings cannot be checked."
        )
    diff = key_mismatch(source, key)
    if diff:
        raise ValueError(
            f"resume_checkpoint: {path} belongs to other {', '.join(diff)}. Resume it with "
            "the same samples, molecule, Subspace Dimension and Batch Number; only Max "
            "Iteration may change."
        )
    return path, load_checkpoint(path)


@task(name="sqd_loop", tags=["post-process"], cache_policy=NO_CACHE)
async def sqd_loop(
    bool_array: np.ndarray,
    elec_props: ElectronicProperties,
    sqd_params: SQDParameters,
    solver: SolverBlock,
    artifact_key: str,
    batch_transform: BatchTransformFn | None = None,
) -> float:
    """Self-consistent configuration recovery and subspace diagonalization.

    Each round recovers configurations with the current orbital occupancies,
    post-selects on the electron numbers, draws ``num_batches`` subspaces and
    diagonalizes them concurrently with the solver block, which runs each one
    as an HPC job.

    The task is never cached: a retried flow run reuses the persisted samples
    and integrals and repeats only this stage. With ``sqd_params.checkpoint``,
    the state after every round is saved under the solver's ``root_dir``, and a
    retry continues from the last completed round. ``sqd_params.resume_checkpoint``
    continues the recovery of another run with the same samples and settings.

    Args:
        bool_array: Sampled bitstrings as a 2D bool array (one row per shot).
        elec_props: Electronic properties of the molecule.
        sqd_params: Subspace dimension, batches, recovery rounds and checkpointing.
        solver: Solver block that diagonalizes one subspace per call.
        artifact_key: Key of the table artifact with one row per solved subspace.
        batch_transform: Rewrites each round's subspaces for the solver (SBD).

    Returns:
        Lowest energy found, including the nuclear repulsion energy.

    Raises:
        ValueError: If ``resume_checkpoint`` belongs to other samples or settings.
    """
    logger = get_run_logger()
    key = resume_key(bool_array, elec_props, sqd_params)
    run_dir = checkpoint_run_dir(solver)

    # A retry's own checkpoint first, then the one the user named, else round 1.
    # ``resumed_from`` names another run's checkpoint only; a retry keeps the one its
    # source.json recorded.
    loaded_from, resumed_from = run_dir / CKPT_NAME, None
    state = _own_checkpoint(run_dir, key, logger)
    if state is not None:
        resumed_from = (read_source(run_dir) or {}).get("resumed_from")
    elif sqd_params.resume_checkpoint:
        loaded_from, state = _given_checkpoint(sqd_params.resume_checkpoint, key)
        resumed_from = str(loaded_from)
    if state is None:
        occupancies, start_round, best_energy = elec_props.initial_occupancy, 0, None
        rng = np.random.default_rng(RECOVERY_SEED)
    else:
        logger.info(
            f"[ckpt] RESUME from {loaded_from}: completed {state.next_step} step(s), "
            f"best_energy so far={state.best_energy:.8f}"
        )
        occupancies, start_round, best_energy = (
            state.occupancies,
            state.next_step,
            state.best_energy,
        )
        rng = np.random.default_rng([RECOVERY_SEED, start_round])

    on_round_end = None
    if sqd_params.checkpoint:
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
            write_source(run_dir, key, resumed_from=resumed_from)
        except OSError as exc:  # checkpointing must never break the calculation
            logger.warning(f"[ckpt] checkpointing off: cannot write {run_dir} ({exc})")
        else:

            def on_round_end(next_round, round_occupancies, best):
                save_checkpoint(run_dir / CKPT_NAME, round_occupancies, next_round, best, logger)

            if state is not None and loaded_from != run_dir / CKPT_NAME:
                # Copy the resumed state, so a retry of this run finds it before round k+1 ends.
                on_round_end(start_round, occupancies, best_energy)

    raw_bitstrings, raw_probs = bit_array_to_arrays(BitArray.from_bool_array(bool_array))
    outcome = await run_configuration_recovery(
        raw_bitstrings,
        raw_probs,
        solve=make_block_solver(solver, elec_props),
        initial_occupancies=occupancies,
        nelec=elec_props.num_electrons,
        open_shell=elec_props.open_shell,
        subspace_dim=sqd_params.subspace_dim,
        num_batches=sqd_params.num_batches,
        max_iterations=sqd_params.max_iterations,
        rng=rng,
        nuclear_repulsion_energy=elec_props.nuclear_repulsion_energy,
        logger=logger,
        start_round=start_round,
        best_energy=best_energy,
        on_round_end=on_round_end,
        batch_transform=batch_transform,
    )
    await create_table_artifact(table=outcome.records, key=artifact_key)
    if on_round_end is not None:
        await create_table_artifact(
            table=[
                dict(
                    checkpoint=str(run_dir / CKPT_NAME),
                    completed_iterations=max(start_round, sqd_params.max_iterations),
                    max_iterations=sqd_params.max_iterations,
                    best_energy=outcome.best_energy,
                    resumed_from=resumed_from,
                )
            ],
            key=CHECKPOINT_ARTIFACT_KEY,
            description=(
                "Recovery checkpoint of this run. Paste the checkpoint path into "
                "Resume from Checkpoint of a new run, with a larger Max Iteration, to "
                "continue this recovery."
            ),
        )
    return outcome.best_energy
