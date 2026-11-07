"""Work-directory file exchange between the submit side and ``qcsc-fulqrum-solve``.

This module has no Prefect or qiskit-addon-sqd dependency, so the compute-node
CLI can use it.

Layout of one job directory:

- ``input.npz``: ``ci_strs_a``, ``ci_strs_b``, ``one_body_tensor``,
  ``two_body_tensor``.
- ``solver.json``: ``norb``, ``nelec``, and the :class:`FulqrumSolverOptions`.
- ``output.npz``: ``occupancies_a``, ``occupancies_b``, ``ci_strs_a``,
  ``ci_strs_b``, ``carryover_a``, ``carryover_b``, and ``amplitudes`` when
  requested.
- ``result.json``: energy and solver metadata. Written last, so its presence
  means the outputs are complete.
- ``error.json``: written instead of the outputs when the solve fails.
- ``error.rank<k>.json``: written by process ``k > 0`` of a multi-process run
  when it fails. ``error.json`` is from process 0.

CI strings are stored as ``int64`` for ``norb < 64`` and as decimal strings
otherwise, so no file needs pickle.

Version 2 of the format added the solver ``backend`` and ``carryover_fraction``
and the carryover arrays. A file is written as version 1 when it uses none of
them (CPU backend, no carryover), so a workflow host and a compute node of
different package versions keep working for such solves; otherwise it is
version 2. Both versions are read, and a version 1 file has empty carryover. A
compute node with an older package rejects version 2 input instead of ignoring
the new options.
"""

from __future__ import annotations

import json
import logging
import os
import re
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np

from .solver import FulqrumSolution, FulqrumSolverOptions, normalize_problem

logger = logging.getLogger(__name__)

FORMAT_VERSION = 2
# Versions this module reads; a missing key of an older version takes its default.
READABLE_FORMAT_VERSIONS = (1, 2)
# Solver options that version 2 added, with the values a version 1 solve implies.
_V2_OPTION_DEFAULTS = {
    "backend": "cpu",
    "carryover_fraction": 0.0,
    "max_basis_size": None,
    "block_size": None,
    "min_restart_size": None,
    "primme_method": None,
    "convergence": "abs",
    "energy_tol": 1e-6,
    "process_grid": None,
    "comm_backend": "auto",
}
INPUT_ARRAYS = "input.npz"
INPUT_SPEC = "solver.json"
OUTPUT_ARRAYS = "output.npz"
OUTPUT_RESULT = "result.json"
OUTPUT_ERROR = "error.json"
_RANK_ERROR = re.compile(r"error\.rank(\d+)\.json")


def error_file_name(rank: int = 0) -> str:
    """The error file of process ``rank``: ``error.json`` for 0, else ``error.rank<k>.json``."""

    return OUTPUT_ERROR if rank == 0 else f"error.rank{int(rank)}.json"


class FulqrumSolverError(RuntimeError):
    """Raised when a job directory holds an error file from a failed solve."""

    def __init__(self, message: str, *, error: dict | None = None):
        super().__init__(message)
        self.error = error or {}


def make_job_work_dir(base_work_dir: Path) -> Path:
    """Create a unique work directory for one fulqrum execution.

    Args:
        base_work_dir: Parent directory under which per-job directories are
            created.

    Returns:
        Newly created job directory named with a UTC timestamp and short random
        suffix.

    Raises:
        FileExistsError: If the generated unique directory already exists.
        OSError: If the directory cannot be created.
    """

    base_work_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    job_dir = base_work_dir / f"job_{timestamp}_{uuid4().hex[:8]}"
    job_dir.mkdir(parents=True, exist_ok=False)
    return job_dir


def encode_ci_strs(strs) -> np.ndarray:
    """CI strings for a file read without pickle.

    ``int64`` strings are kept; Python ints (``object`` dtype, 64 or more orbitals)
    become decimal strings. :func:`decode_ci_strs` reverses this.
    """
    strs = np.asarray(strs).reshape(-1)
    if strs.dtype == object:
        return np.array([str(int(x)) for x in strs])
    return strs.astype(np.int64, copy=False)


def decode_ci_strs(arr: np.ndarray, norb: int) -> np.ndarray:
    """CI strings written by :func:`encode_ci_strs`: ``int64``, or Python ints from 64 orbitals."""
    if arr.dtype.kind == "U":
        values = [int(x) for x in arr]
        return np.array(values, dtype=np.int64 if norb < 64 else object)
    return arr.astype(np.int64, copy=False)


def _json_default(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _write_json(path: Path, payload: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _write_npz(path: Path, **arrays: np.ndarray) -> None:
    # np.savez appends ".npz" to names without it, so keep the suffix on the temp file.
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def _read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    version = payload.get("format_version")
    if version not in READABLE_FORMAT_VERSIONS:
        raise ValueError(
            f"{path.name}: unsupported format_version {version!r}; this qcsc-prefect-fulqrum "
            f"reads versions {', '.join(map(str, READABLE_FORMAT_VERSIONS))}. Install the "
            "same qcsc-prefect-fulqrum version on the workflow host and the compute node."
        )
    return payload


def _options_payload(options: FulqrumSolverOptions) -> tuple[int, dict]:
    """``(format_version, options)`` for ``solver.json``: version 1 without v2 options."""
    payload = asdict(options)
    if all(payload[key] == value for key, value in _V2_OPTION_DEFAULTS.items()):
        for key in _V2_OPTION_DEFAULTS:
            del payload[key]
        return 1, payload
    return FORMAT_VERSION, payload


def write_solver_input(
    *,
    work_dir: Path,
    ci_strings: tuple[np.ndarray, np.ndarray],
    one_body_tensor: np.ndarray,
    two_body_tensor: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    options: FulqrumSolverOptions | None = None,
) -> None:
    """Validate a subspace problem and write it to ``work_dir``.

    Validation runs here, on the submit side, so a malformed problem fails
    before it waits in a scheduler queue. CI strings are deduplicated and sorted.

    Raises:
        ValueError: If shapes, electron counts, or CI string ranges are invalid.
        OSError: If the files cannot be written.
    """

    options = options or FulqrumSolverOptions()
    strs_a, strs_b, h1, eri = normalize_problem(
        ci_strings, one_body_tensor, two_body_tensor, norb, nelec
    )
    work_dir.mkdir(parents=True, exist_ok=True)
    _write_npz(
        work_dir / INPUT_ARRAYS,
        ci_strs_a=encode_ci_strs(strs_a),
        ci_strs_b=encode_ci_strs(strs_b),
        one_body_tensor=h1,
        two_body_tensor=eri,
    )
    version, options_payload = _options_payload(options)
    _write_json(
        work_dir / INPUT_SPEC,
        {
            "format_version": version,
            "norb": int(norb),
            "nelec": [int(nelec[0]), int(nelec[1])],
            "options": options_payload,
        },
    )
    logger.debug("Wrote fulqrum input (%d x %d) to %s", len(strs_a), len(strs_b), work_dir)


def read_solver_input(work_dir: Path) -> dict:
    """Read a problem written by :func:`write_solver_input`.

    Returns:
        Keyword arguments for :func:`~qcsc_prefect_fulqrum.solver.solve_fulqrum`:
        ``ci_strings``, ``one_body_tensor``, ``two_body_tensor``, ``norb``,
        ``nelec``, and ``options``.

    Raises:
        FileNotFoundError: If an input file is missing.
        ValueError: If ``solver.json`` has an unsupported format or options.
    """

    spec = _read_json(work_dir / INPUT_SPEC)
    norb = int(spec["norb"])
    with np.load(work_dir / INPUT_ARRAYS, allow_pickle=False) as data:
        strs_a = decode_ci_strs(data["ci_strs_a"], norb)
        strs_b = decode_ci_strs(data["ci_strs_b"], norb)
        h1 = data["one_body_tensor"]
        eri = data["two_body_tensor"]
    return {
        "ci_strings": (strs_a, strs_b),
        "one_body_tensor": h1,
        "two_body_tensor": eri,
        "norb": norb,
        "nelec": (int(spec["nelec"][0]), int(spec["nelec"][1])),
        "options": FulqrumSolverOptions(**spec["options"]),
    }


def write_solver_output(work_dir: Path, solution: FulqrumSolution) -> None:
    """Write a solution to ``work_dir``. ``result.json`` is written last."""

    arrays = {
        "occupancies_a": np.asarray(solution.orbital_occupancies[0]),
        "occupancies_b": np.asarray(solution.orbital_occupancies[1]),
        "ci_strs_a": encode_ci_strs(solution.ci_strs_a),
        "ci_strs_b": encode_ci_strs(solution.ci_strs_b),
    }
    has_carryover = len(solution.carryover_a) > 0 or len(solution.carryover_b) > 0
    if has_carryover:
        arrays["carryover_a"] = encode_ci_strs(np.asarray(solution.carryover_a))
        arrays["carryover_b"] = encode_ci_strs(np.asarray(solution.carryover_b))
    if solution.amplitudes is not None:
        arrays["amplitudes"] = solution.amplitudes
    _write_npz(work_dir / OUTPUT_ARRAYS, **arrays)
    _write_json(
        work_dir / OUTPUT_RESULT,
        {
            "format_version": FORMAT_VERSION if has_carryover else 1,
            "energy": float(solution.energy),
            "has_amplitudes": solution.amplitudes is not None,
            "metadata": solution.metadata,
        },
    )


def write_solver_error(work_dir: Path, exc: BaseException, rank: int = 0) -> None:
    """Record a failed solve of process ``rank`` in :func:`error_file_name`."""

    _write_json(
        work_dir / error_file_name(rank),
        {
            # error.json keeps version 1 for older readers; only the rank files are new.
            "format_version": FORMAT_VERSION if rank else 1,
            "rank": int(rank),
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": "".join(traceback.format_exception(exc)),
        },
    )


def solver_error_files(work_dir: Path) -> list[Path]:
    """``error.json`` (present or not), then any ``error.rank<k>.json`` in rank order."""

    ranked = []
    for path in work_dir.glob("error.rank*.json"):
        match = _RANK_ERROR.fullmatch(path.name)
        if match:
            ranked.append((int(match[1]), path))
    return [work_dir / OUTPUT_ERROR] + [path for _, path in sorted(ranked)]


def read_solver_error(work_dir: Path) -> dict | None:
    """Return the first error file of a failed solve, or ``None`` if there is none.

    ``error.json`` is read first, then ``error.rank<k>.json`` in rank order, so
    a failure of any process is seen even when process 0 was aborted before it
    could write its own file.
    """

    for path in solver_error_files(work_dir):
        if not path.exists():
            continue
        try:
            return _read_json(path)
        except (OSError, ValueError) as exc:
            return {"type": type(exc).__name__, "message": f"unreadable {path.name}: {exc}"}
    return None


def read_solver_output(work_dir: Path, norb: int) -> FulqrumSolution:
    """Read a solution written by ``qcsc-fulqrum-solve``.

    Raises:
        FulqrumSolverError: If the solve failed and left an error file.
        FileNotFoundError: If the outputs are missing.
        ValueError: If ``result.json`` has an unsupported format.
    """

    error = read_solver_error(work_dir)
    if error is not None:
        where = f" on rank {error['rank']}" if error.get("rank") else ""
        raise FulqrumSolverError(
            f"fulqrum solve failed{where}: {error.get('type')}: {error.get('message')}",
            error=error,
        )
    result = _read_json(work_dir / OUTPUT_RESULT)
    with np.load(work_dir / OUTPUT_ARRAYS, allow_pickle=False) as data:
        occ_a = data["occupancies_a"]
        occ_b = data["occupancies_b"]
        strs_a = decode_ci_strs(data["ci_strs_a"], norb)
        strs_b = decode_ci_strs(data["ci_strs_b"], norb)
        amplitudes = data["amplitudes"] if "amplitudes" in data.files else None
        empty = np.empty(0, dtype=np.int64)
        carryover = [
            decode_ci_strs(data[key], norb) if key in data.files else empty
            for key in ("carryover_a", "carryover_b")
        ]
    return FulqrumSolution(
        energy=float(result["energy"]),
        orbital_occupancies=(occ_a, occ_b),
        ci_strs_a=strs_a,
        ci_strs_b=strs_b,
        amplitudes=amplitudes,
        metadata=dict(result.get("metadata") or {}),
        carryover_a=carryover[0],
        carryover_b=carryover[1],
    )
