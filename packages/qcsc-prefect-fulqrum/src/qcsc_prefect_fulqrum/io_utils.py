"""Work-directory file exchange between the submit side and ``qcsc-fulqrum-solve``.

This module has no Prefect or qiskit-addon-sqd dependency, so the compute-node
CLI can use it.

Layout of one job directory:

- ``input.npz``: ``ci_strs_a``, ``ci_strs_b``, ``one_body_tensor``,
  ``two_body_tensor``.
- ``solver.json``: ``norb``, ``nelec``, and the :class:`FulqrumSolverOptions`.
- ``output.npz``: ``occupancies_a``, ``occupancies_b``, ``ci_strs_a``,
  ``ci_strs_b``, and ``amplitudes`` when requested.
- ``result.json``: energy and solver metadata. Written last, so its presence
  means the outputs are complete.
- ``error.json``: written instead of the outputs when the solve fails.

CI strings are stored as ``int64`` for ``norb < 64`` and as decimal strings
otherwise, so no file needs pickle.
"""

from __future__ import annotations

import json
import logging
import os
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np

from .solver import FulqrumSolution, FulqrumSolverOptions, normalize_problem

logger = logging.getLogger(__name__)

FORMAT_VERSION = 1
INPUT_ARRAYS = "input.npz"
INPUT_SPEC = "solver.json"
OUTPUT_ARRAYS = "output.npz"
OUTPUT_RESULT = "result.json"
OUTPUT_ERROR = "error.json"


class FulqrumSolverError(RuntimeError):
    """Raised when a job directory holds ``error.json`` from a failed solve."""

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


def _encode_ci_strs(strs: np.ndarray) -> np.ndarray:
    if strs.dtype == object:
        return np.array([str(int(x)) for x in strs])
    return strs


def _decode_ci_strs(arr: np.ndarray, norb: int) -> np.ndarray:
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
    if version != FORMAT_VERSION:
        raise ValueError(f"{path.name}: unsupported format_version {version!r}")
    return payload


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
        ci_strs_a=_encode_ci_strs(strs_a),
        ci_strs_b=_encode_ci_strs(strs_b),
        one_body_tensor=h1,
        two_body_tensor=eri,
    )
    _write_json(
        work_dir / INPUT_SPEC,
        {
            "format_version": FORMAT_VERSION,
            "norb": int(norb),
            "nelec": [int(nelec[0]), int(nelec[1])],
            "options": asdict(options),
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
        strs_a = _decode_ci_strs(data["ci_strs_a"], norb)
        strs_b = _decode_ci_strs(data["ci_strs_b"], norb)
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
        "ci_strs_a": _encode_ci_strs(solution.ci_strs_a),
        "ci_strs_b": _encode_ci_strs(solution.ci_strs_b),
    }
    if solution.amplitudes is not None:
        arrays["amplitudes"] = solution.amplitudes
    _write_npz(work_dir / OUTPUT_ARRAYS, **arrays)
    _write_json(
        work_dir / OUTPUT_RESULT,
        {
            "format_version": FORMAT_VERSION,
            "energy": float(solution.energy),
            "has_amplitudes": solution.amplitudes is not None,
            "metadata": solution.metadata,
        },
    )


def write_solver_error(work_dir: Path, exc: BaseException) -> None:
    """Record a failed solve in ``error.json``."""

    _write_json(
        work_dir / OUTPUT_ERROR,
        {
            "format_version": FORMAT_VERSION,
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": "".join(traceback.format_exception(exc)),
        },
    )


def read_solver_error(work_dir: Path) -> dict | None:
    """Return the contents of ``error.json``, or ``None`` if there is none."""

    path = work_dir / OUTPUT_ERROR
    if not path.exists():
        return None
    try:
        return _read_json(path)
    except (OSError, ValueError) as exc:
        return {"type": type(exc).__name__, "message": f"unreadable {OUTPUT_ERROR}: {exc}"}


def read_solver_output(work_dir: Path, norb: int) -> FulqrumSolution:
    """Read a solution written by ``qcsc-fulqrum-solve``.

    Raises:
        FulqrumSolverError: If the solve failed and left ``error.json``.
        FileNotFoundError: If the outputs are missing.
        ValueError: If ``result.json`` has an unsupported format.
    """

    error = read_solver_error(work_dir)
    if error is not None:
        raise FulqrumSolverError(
            f"fulqrum solve failed: {error.get('type')}: {error.get('message')}", error=error
        )
    result = _read_json(work_dir / OUTPUT_RESULT)
    with np.load(work_dir / OUTPUT_ARRAYS, allow_pickle=False) as data:
        occ_a = data["occupancies_a"]
        occ_b = data["occupancies_b"]
        strs_a = _decode_ci_strs(data["ci_strs_a"], norb)
        strs_b = _decode_ci_strs(data["ci_strs_b"], norb)
        amplitudes = data["amplitudes"] if "amplitudes" in data.files else None
    return FulqrumSolution(
        energy=float(result["energy"]),
        orbital_occupancies=(occ_a, occ_b),
        ci_strs_a=strs_a,
        ci_strs_b=strs_b,
        amplitudes=amplitudes,
        metadata=dict(result.get("metadata") or {}),
    )
