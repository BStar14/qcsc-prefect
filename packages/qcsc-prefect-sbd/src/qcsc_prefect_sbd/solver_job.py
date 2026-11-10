"""SBD solver block backed by qcsc-prefect block execution."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import uuid4

import numpy as np
from prefect import get_run_logger, task
from prefect.blocks.core import Block
from pydantic import Field
from pyscf.tools import fcidump
from qcsc_prefect_executor.from_blocks import run_job_from_blocks


@dataclass(frozen=True)
class SBDResult:
    """Result of an SBD calculation."""

    energy: float
    """The SCI energy."""

    orbital_occupancies: tuple[np.ndarray, np.ndarray]
    """The average orbital occupancies."""

    carryover_bitstrings: np.ndarray
    """The 2D array of bool representations of the alpha carryover bitstrings."""

    rdm1: np.ndarray | None = None
    """Spin-summed 1-particle reduced density matrix."""

    rdm2: np.ndarray | None = None
    """Spin-summed 2-particle reduced density matrix."""

    carryover_bitstrings_b: np.ndarray | None = None
    """Beta carryover bitstrings, same layout as ``carryover_bitstrings``.

    Set when the alpha and beta determinant sets differ. None for runs with one shared
    set, where the beta carryover is the alpha carryover.
    """

    @property
    def sci_state(self):
        raise NotImplementedError("SBD Prefect integration doesn't reconstruct sci_state object.")


def _make_job_work_dir(base_work_dir: Path) -> Path:
    base_work_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    job_dir = base_work_dir / f"job_{timestamp}_{uuid4().hex[:8]}"
    job_dir.mkdir(parents=True, exist_ok=False)
    return job_dir


def _beta_differs(ci_strings: tuple[np.ndarray, np.ndarray]) -> bool:
    alpha, beta = (np.asarray(strings, dtype=np.int64).reshape(-1) for strings in ci_strings)
    return not np.array_equal(alpha, beta)


def _build_solver_args(solver: "SBDSolverJob", *, separate_beta: bool = False) -> list[str]:
    args = [
        "--task_comm_size",
        str(solver.task_comm_size),
        "--adet_comm_size",
        str(solver.adet_comm_size),
        "--bdet_comm_size",
        str(solver.bdet_comm_size),
        "--block",
        str(solver.block),
        "--iteration",
        str(solver.iteration),
        "--tolerance",
        str(solver.tolerance),
        "--carryover_ratio",
        str(solver.carryover_ratio),
        "--carryover_type",
        str(solver.carryover_type),
        "--rdm",
        "0",
    ]
    # --dump_matrix_form_wf is not passed: it writes the full CI wavefunction as text, which
    # nothing reads, and at large subspaces the dump (tens of GB) can stall the solver on I/O
    # until the wall time runs out. SBD skips the dump when the flag is absent.
    if solver.solver_mode == "gpu" or separate_beta:
        args.extend(["--adetfile", "AlphaDets.bin"])
    if separate_beta:
        args.extend(["--bdetfile", "BetaDets.bin"])
    if solver.solver_mode == "gpu":
        args.extend(["--carryoverfile", "carryover.txt"])
    if solver.user_args:
        args.extend(list(solver.user_args))
    return args


def check_sbd_input(
    *,
    ci_strings: tuple[np.ndarray, np.ndarray],
    nelec: tuple[int, int],
    spin_sq: float | None,
) -> None:
    """Reject input that the SBD solver would silently treat differently from DICE.

    SBD diagonalizes in the product of the alpha and beta determinant sets with
    restricted (spin-free) integrals, so it solves closed-shell and restricted
    open-shell (ROHF) problems. The sets may differ, and they necessarily differ when
    ``nelec[0] != nelec[1]``. Each set must start with its Hartree-Fock determinant,
    ``(1 << nelec[0]) - 1`` for alpha and ``(1 << nelec[1]) - 1`` for beta. SBD has no
    total spin target, so ``spin_sq`` must be None.
    """
    if spin_sq is not None:
        raise ValueError(
            f"SBD has no spin_sq target; got spin_sq={spin_sq!r}. Set spin_sq to None "
            "(null) in the molecule settings to use SBD."
        )
    for spin, strings, n_elec in zip(("alpha", "beta"), ci_strings, nelec, strict=True):
        strings = np.asarray(strings).reshape(-1)
        hartree_fock = (1 << int(n_elec)) - 1
        if strings.size == 0 or int(strings[0]) != hartree_fock:
            first = int(strings[0]) if strings.size else None
            raise ValueError(
                f"SBD requires the {spin} Hartree-Fock determinant {hartree_fock} first in "
                f"the {spin} ci_strings; got {first}."
            )


def _write_dets_bin(
    path: Path, ci_strings: np.ndarray, norb: int, *, spin: str, num_elec: int | None = None
) -> None:
    """Write a determinant list to ``path`` in the SBD binary format.

    The format is what SBD's ``LoadAlphaDets`` reads for both spins: ``(norb + 7) // 8``
    bytes per determinant, big-endian bit order.

    When ``num_elec`` is given, every determinant must have exactly that many set bits
    and the list must be duplicate-free. SBD checks neither, so a wrong-sector or
    repeated determinant would otherwise be diagonalized silently.
    """
    dets = np.asarray(ci_strings, dtype=np.int64).reshape(-1)
    if np.any(dets < 0):
        raise ValueError(f"{spin} determinants must be non-negative integers.")
    max_ci = 1 << norb
    if np.any(dets >= max_ci):
        raise ValueError(f"{spin} determinant is out of range for norb={norb}.")
    if num_elec is not None and dets.size:
        popcount = np.array([int(d).bit_count() for d in dets.tolist()])
        bad = popcount != num_elec
        if np.any(bad):
            raise ValueError(
                f"{int(bad.sum())} of {dets.size} {spin} determinants do not have {num_elec} "
                f"electrons (popcounts {sorted(set(popcount[bad].tolist()))})."
            )
        n_unique = np.unique(dets).size
        if n_unique != dets.size:
            raise ValueError(
                f"{spin} determinant list has {dets.size - n_unique} duplicate entries."
            )
    bytes_per_config = (norb + 7) // 8
    with path.open("wb") as fp:
        for ci in dets:
            fp.write(int(ci).to_bytes(bytes_per_config, byteorder="big", signed=False))


def _prep_files(
    *,
    work_dir: Path,
    ci_strings: tuple[np.ndarray, np.ndarray],
    one_body_tensor: np.ndarray,
    two_body_tensor: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
) -> None:
    logger = get_run_logger()

    # Write PySCF FCI dump file. A (n_alpha, n_beta) nelec writes MS2 = |n_alpha - n_beta|.
    logger.debug("Writing fcidump.txt file.")
    fcidump.from_integrals(
        str(work_dir / "fcidump.txt"),
        one_body_tensor,
        two_body_tensor,
        norb,
        tuple(int(n) for n in nelec),
    )

    # Write the determinant lists consumed by the SBD binary. BetaDets.bin only when the
    # beta set differs; otherwise SBD uses the alpha set for both spins.
    logger.debug("Writing AlphaDets.bin file.")
    _write_dets_bin(
        work_dir / "AlphaDets.bin", ci_strings[0], norb, spin="Alpha", num_elec=int(nelec[0])
    )
    if _beta_differs(ci_strings):
        logger.debug("Writing BetaDets.bin file.")
        _write_dets_bin(
            work_dir / "BetaDets.bin", ci_strings[1], norb, spin="Beta", num_elec=int(nelec[1])
        )


def _read_carryover_bin(path: Path, norb: int) -> np.ndarray:
    """Decode an SBD carryover file into a ``(n, norb)`` bool array."""
    bytes_per_config = (norb + 7) // 8
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return np.empty((0, norb), dtype=bool)
    if data.size % bytes_per_config != 0:
        raise ValueError(
            f"{path.name} size is not aligned with expected bytes-per-config; "
            f"norb={norb}, bytes_per_config={bytes_per_config}, raw_size={data.size}"
        )
    bits = np.unpackbits(data, bitorder="big").reshape(-1, bytes_per_config * 8)[:, :norb]
    return bits.astype(bool)


def _read_files(
    *,
    work_dir: Path,
    norb: int,
    separate_beta: bool = False,
) -> SBDResult:
    logger = get_run_logger()
    logger.debug("Reading occ_a.txt and occ_b.txt file.")
    occa = np.atleast_1d(np.loadtxt(work_dir / "occ_a.txt", dtype=np.float64))
    occb = np.atleast_1d(np.loadtxt(work_dir / "occ_b.txt", dtype=np.float64))

    logger.debug("Reading carryover.bin file.")
    carryover = _read_carryover_bin(work_dir / "carryover.bin", norb)

    carryover_b = None
    if separate_beta:
        carryover_b_path = work_dir / "carryover_b.bin"
        if not carryover_b_path.is_file():
            # The driver writes it whenever --bdetfile is given; a driver without --bdetfile
            # support would have solved with the alpha set for both spins.
            raise FileNotFoundError(
                f"{carryover_b_path} is missing. The SBD binary does not support separate "
                "beta determinants (--bdetfile); rebuild it from native/main.cc."
            )
        logger.debug("Reading carryover_b.bin file.")
        carryover_b = _read_carryover_bin(carryover_b_path, norb)

    logger.debug("Reading davidson_energy.txt file.")
    energy = float(np.loadtxt(work_dir / "davidson_energy.txt").item())

    return SBDResult(
        energy=energy,
        orbital_occupancies=(occa, occb),
        carryover_bitstrings=carryover,
        carryover_bitstrings_b=carryover_b,
        rdm1=None,
        rdm2=None,
    )


def _dir_size_bytes(path: Path) -> int:
    """Sum the sizes of the files directly under ``path``.

    A file removed between listing and stat is skipped; the number is a diagnostic.
    """
    total = 0
    for entry in path.iterdir():
        try:
            if entry.is_file():
                total += entry.stat().st_size
        except FileNotFoundError:
            continue
    return total


class SBDSolverJob(Block):
    """Prefect block facade for SBD execution through qcsc-prefect blocks."""

    _block_type_name = "SBD Solver Job"
    _block_type_slug = "sbd_solver_job"

    root_dir: str = Field(
        title="Root Directory",
        description="Root directory where per-job work directories are created.",
    )
    command_block_name: str = Field(
        default="cmd-sbd-diag",
        title="Command Block Name",
        description="Prefect CommandBlock document name.",
    )
    execution_profile_block_name: str = Field(
        default="exec-sbd-mpi",
        title="Execution Profile Block Name",
        description="Prefect ExecutionProfileBlock document name.",
    )
    hpc_profile_block_name: str = Field(
        default="hpc-miyabi-sbd",
        title="HPC Profile Block Name",
        description="Prefect HPCProfileBlock document name.",
    )
    script_filename: str = Field(default="sbd_solver.pbs", title="Script Filename")
    metrics_artifact_key: str = Field(default="miyabi-sbd-metrics", title="Metrics Artifact Key")
    timeout_seconds: float = Field(default=7200.0, title="Timeout Seconds")
    user_args: list[str] = Field(default_factory=list, title="Additional User Args")

    task_comm_size: int = Field(
        default=1,
        gt=0,
        title="Task Comm Size",
        description=(
            "Size of task communicator. Controls distribution of Hamiltonian column operations."
        ),
    )
    adet_comm_size: int = Field(
        default=1,
        gt=0,
        title="Adet Comm Size",
        description="Number of alpha-determinant partitions.",
    )
    bdet_comm_size: int = Field(
        default=1,
        gt=0,
        title="Bdet Comm Size",
        description="Number of beta-determinant partitions.",
    )
    block: int = Field(
        default=10,
        gt=0,
        title="Block",
        description="Maximum Davidson subspace size.",
    )
    iteration: int = Field(
        default=2,
        gt=0,
        title="Iteration",
        description="Number of Davidson restarts.",
    )
    tolerance: float = Field(
        default=1e-4,
        gt=0.0,
        title="Tolerance",
        description="Convergence threshold for Davidson residual norm.",
    )
    carryover_ratio: float = Field(
        default=0.5,
        gt=0.0,
        le=1.0,
        title="Carryover Ratio",
        description="Ratio of bitstrings retained as carryover candidates.",
    )
    carryover_type: int = Field(
        default=1,
        ge=0,
        le=3,
        title="Carryover Type",
        description=(
            "Passed to SBD as --carryover_type. 0 writes no carryover determinants; 1 keeps "
            "the carryover_ratio fraction of each spin's determinants with the largest "
            "weight; 2 is type 1 plus the single excitations of the kept set; 3 is the "
            "single excitations of the whole wavefunction above SBD's carryover threshold."
        ),
    )
    solver_mode: Literal["cpu", "gpu"] = Field(
        default="cpu",
        title="Solver Mode",
        description="SBD execution mode.",
    )

    async def run(
        self,
        ci_strings: tuple[np.ndarray, np.ndarray],
        one_body_tensor: np.ndarray,
        two_body_tensor: np.ndarray,
        norb: int,
        nelec: tuple[int, int],
        spin_sq: float | None = None,
    ) -> SBDResult:
        """Run SBD solver job and return parsed outputs.

        The arguments match ``DiceSHCISolverJob.run``. SBD solves closed-shell and
        restricted open-shell problems; input it cannot solve raises ``ValueError``
        instead of silently giving a different calculation; see ``check_sbd_input``.
        """
        check_sbd_input(ci_strings=ci_strings, nelec=nelec, spin_sq=spin_sq)
        return await _run_sbd_inner(
            solver=self,
            ci_strings=ci_strings,
            one_body_tensor=one_body_tensor,
            two_body_tensor=two_body_tensor,
            norb=norb,
            nelec=nelec,
        )


@task(name="solve_eigenstate")
async def _run_sbd_inner(
    *,
    solver: SBDSolverJob,
    ci_strings: tuple[np.ndarray, np.ndarray],
    one_body_tensor: np.ndarray,
    two_body_tensor: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
) -> SBDResult:
    base_work_dir = Path(solver.root_dir).expanduser().resolve()
    job_work_dir = _make_job_work_dir(base_work_dir)
    logger = get_run_logger()
    separate_beta = _beta_differs(ci_strings)

    t_prep0 = time.perf_counter()
    _prep_files(
        work_dir=job_work_dir,
        ci_strings=ci_strings,
        one_body_tensor=one_body_tensor,
        two_body_tensor=two_body_tensor,
        norb=norb,
        nelec=nelec,
    )
    t_prep = time.perf_counter() - t_prep0
    input_bytes = _dir_size_bytes(job_work_dir)
    logger.info(
        "[diag] solve input: n_a=%d n_b=%d separate_beta=%s comm grid task=%d x adet=%d x "
        "bdet=%d input_disk=%.1fMB Tprep=%.2fs",
        len(np.asarray(ci_strings[0]).reshape(-1)),
        len(np.asarray(ci_strings[1]).reshape(-1)),
        separate_beta,
        solver.task_comm_size,
        solver.adet_comm_size,
        solver.bdet_comm_size,
        input_bytes / 1e6,
        t_prep,
    )

    t_solve0 = time.perf_counter()
    result = await run_job_from_blocks(
        command_block_name=solver.command_block_name,
        execution_profile_block_name=solver.execution_profile_block_name,
        hpc_profile_block_name=solver.hpc_profile_block_name,
        work_dir=job_work_dir,
        script_filename=solver.script_filename,
        user_args=_build_solver_args(solver, separate_beta=separate_beta),
        watch_poll_interval=5.0,
        timeout_seconds=solver.timeout_seconds,
        metrics_artifact_key=solver.metrics_artifact_key,
    )
    t_solve = time.perf_counter() - t_solve0
    logger.info(
        "[diag] solve output: exit_status=%s Tsolve=%.1fs output_disk_delta=%.1fMB",
        result.exit_status,
        t_solve,
        (_dir_size_bytes(job_work_dir) - input_bytes) / 1e6,
    )
    if result.exit_status != 0:
        raise RuntimeError(f"SBDSolverJob failed: exit_status={result.exit_status}")

    return _read_files(work_dir=job_work_dir, norb=norb, separate_beta=separate_beta)
