"""Checkpoint of the configuration recovery loop, so a run can resume mid-loop.

After every round, ``sqd_loop`` writes ``ckpt_t00_w0.npz`` with the state that
seeds the next round:

- ``occ_a``, ``occ_b``: alpha and beta orbital occupancies for the next round
- ``next_step``: number of completed rounds, so the first round still to run
- ``best_energy``: lowest total energy so far, including the nuclear repulsion
- ``carryover_a``, ``carryover_b``: alpha and beta carryover CI strings for the
  next round, heaviest first; ``int64``, or decimal strings for 64 or more
  orbitals, so the file loads without pickle. Empty when carryover is off.

A file without ``carryover_a`` and ``carryover_b``, for example one with a single
empty ``carryover`` array, loads with empty carryover.

``source.json`` next to it holds the resume key: the samples, the Hamiltonian,
``subspace_dim``, ``num_batches`` and the solver (its block type) the checkpoint
belongs to. A checkpoint is resumed only by a run with the same key.
``max_iterations`` may change, which is how a run adds rounds, and so may the
solver block's name and job settings, and the carryover fraction.

This module has no Prefect dependency.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

CKPT_NAME = "ckpt_t00_w0.npz"
SOURCE_NAME = "source.json"
# Relative tolerance of the Hamiltonian fingerprint, which a recomputed SCF reproduces.
HAMILTONIAN_RTOL = 1e-6


def _no_carryover() -> tuple[np.ndarray, np.ndarray]:
    return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)


@dataclass(frozen=True)
class RecoveryCheckpoint:
    """State of the recovery loop after ``next_step`` completed rounds."""

    occupancies: tuple[np.ndarray, np.ndarray]
    next_step: int
    best_energy: float
    carryover: tuple[np.ndarray, np.ndarray] = field(default_factory=_no_carryover)


def _encode_ci_strs(strs) -> np.ndarray:
    """``int64`` CI strings as they are; Python ints (``object`` dtype) as decimal strings."""
    strs = np.asarray(strs).reshape(-1)
    if strs.dtype == object:
        return np.array([str(int(x)) for x in strs], dtype=str)
    return strs.astype(np.int64, copy=False)


def _decode_ci_strs(arr: np.ndarray, norb: int) -> np.ndarray:
    if arr.dtype.kind == "U":
        return np.array([int(x) for x in arr], dtype=np.int64 if norb < 64 else object)
    return arr.astype(np.int64, copy=False)


def save_checkpoint(
    path: Path,
    occupancies: tuple[np.ndarray, np.ndarray],
    next_step: int,
    best_energy: float,
    logger: logging.Logger | logging.LoggerAdapter | None = None,
    *,
    carryover: tuple[np.ndarray, np.ndarray] | None = None,
) -> None:
    """Write a checkpoint atomically. A failed write is logged and never raised."""
    logger = logger or logging.getLogger(__name__)
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    carryover = _no_carryover() if carryover is None else carryover
    try:
        # np.savez appends ".npz" to a string path; a file handle keeps the tmp name.
        with open(tmp, "wb") as fh:
            np.savez(
                fh,
                occ_a=np.asarray(occupancies[0], dtype=np.float64),
                occ_b=np.asarray(occupancies[1], dtype=np.float64),
                next_step=np.int64(next_step),
                best_energy=np.float64(best_energy),
                carryover_a=_encode_ci_strs(carryover[0]),
                carryover_b=_encode_ci_strs(carryover[1]),
            )
        os.replace(tmp, path)
        logger.info(
            f"[ckpt] wrote {path} (next_step={next_step}, best={best_energy:.8f}, "
            f"carryover={len(carryover[0])}+{len(carryover[1])})"
        )
    except Exception as exc:  # checkpointing must never break the calculation
        logger.warning(f"[ckpt] checkpoint save failed (non-fatal): {type(exc).__name__}: {exc}")


def load_checkpoint(path: Path) -> RecoveryCheckpoint:
    """Read a checkpoint written by :func:`save_checkpoint`.

    A checkpoint without ``carryover_a`` and ``carryover_b`` has empty carryover.
    """
    with np.load(path, allow_pickle=False) as ck:
        occupancies = (np.array(ck["occ_a"]), np.array(ck["occ_b"]))
        norb = len(occupancies[0])
        carryover = _no_carryover()
        if "carryover_a" in ck.files and "carryover_b" in ck.files:
            carryover = (
                _decode_ci_strs(np.array(ck["carryover_a"]), norb),
                _decode_ci_strs(np.array(ck["carryover_b"]), norb),
            )
        return RecoveryCheckpoint(
            occupancies=occupancies,
            next_step=int(ck["next_step"]),
            best_energy=float(ck["best_energy"]),
            carryover=carryover,
        )


def resume_key(bool_array: np.ndarray, elec_props, sqd_params, *, solver: str) -> dict:
    """What a checkpoint belongs to: the samples, the Hamiltonian, the subspace settings
    and the solver.

    The Hamiltonian enters through quantities that do not change with the sign or
    rotation of the orbitals, so integrals recomputed from the same geometry match.
    ``solver`` is the solver block type (``fulqrum_solver_job``, ...): solvers can give
    different subspaces (SBD puts Hartree-Fock first) and energies, so a recovery
    continues only with the solver that started it.
    """
    bool_array = np.asarray(bool_array, dtype=bool)
    return {
        "samples_sha256": hashlib.sha256(np.packbits(bool_array, axis=None).tobytes()).hexdigest(),
        "samples_shape": list(bool_array.shape),
        "num_orbitals": int(elec_props.num_orbitals),
        "num_electrons": [int(n) for n in elec_props.num_electrons],
        "one_body_eigenvalues": np.linalg.eigvalsh(elec_props.one_body_tensor).tolist(),
        "two_body_norm": float(np.linalg.norm(elec_props.two_body_tensor)),
        "nuclear_repulsion_energy": float(elec_props.nuclear_repulsion_energy),
        "subspace_dim": int(sqd_params.subspace_dim),
        "num_batches": int(sqd_params.num_batches),
        "solver": str(solver),
    }


def key_mismatch(old: dict, new: dict) -> list[str]:
    """Names of what differs between two resume keys; empty if they match."""

    def close(name: str) -> bool:
        a, b = old.get(name), new.get(name)
        if a is None or b is None or np.shape(a) != np.shape(b):
            return False
        return bool(np.allclose(a, b, rtol=HAMILTONIAN_RTOL, atol=1e-10))

    diff = []
    if any(old.get(k) != new.get(k) for k in ("samples_sha256", "samples_shape")):
        diff.append("samples")
    if any(old.get(k) != new.get(k) for k in ("num_orbitals", "num_electrons")) or not all(
        close(k) for k in ("one_body_eigenvalues", "two_body_norm", "nuclear_repulsion_energy")
    ):
        diff.append("Hamiltonian")
    diff += [k for k in ("subspace_dim", "num_batches", "solver") if old.get(k) != new.get(k)]
    return diff


def write_source(directory: Path, key: dict, **meta) -> None:
    """Write ``source.json``: the resume key plus free-form metadata."""
    (Path(directory) / SOURCE_NAME).write_text(json.dumps({**key, **meta}, indent=2))


def read_source(directory: Path) -> dict | None:
    """Read ``source.json``, or ``None`` if the directory has none."""
    path = Path(directory) / SOURCE_NAME
    return json.loads(path.read_text()) if path.is_file() else None
