"""Solver block selection by ``solver_block_ref`` (``"<block type slug>/<block name>"``).

fulqrum is part of the base install. DICE and SBD are optional: install
``sqd[dice]`` or ``sqd[sbd]``. A solver package is imported only when a run
selects it, so a missing extra fails with an install hint, before any stage runs.
"""

from __future__ import annotations

import importlib

from qcsc_workflow_utility.solvers import DICE_SLUG, SBD_SLUG, parse_block_ref

FULQRUM_SLUG = "fulqrum_solver_job"
DEFAULT_SOLVER_BLOCK_REF = f"{FULQRUM_SLUG}/sqd-fulqrum-solver"

# Block type slug -> (module, class, install extra or None for the base install).
SOLVERS = {
    FULQRUM_SLUG: ("qcsc_prefect_fulqrum", "FulqrumSolverJob", None),
    DICE_SLUG: ("qcsc_prefect_dice", "DiceSHCISolverJob", "dice"),
    SBD_SLUG: ("qcsc_prefect_sbd", "SBDSolverJob", "sbd"),
}


def solver_slug(ref: str) -> tuple[str, str]:
    """Split ``ref`` into a known block type slug and the block name.

    Prefect also writes slugs with hyphens (``sbd-solver-job``); both spellings are
    accepted.

    Raises:
        ValueError: If ``ref`` is not ``"<slug>/<name>"`` or the slug is unknown.
    """
    slug, name = parse_block_ref(ref)
    slug = slug.replace("-", "_")
    if slug not in SOLVERS:
        accepted = ", ".join(f"'{known}/<name>'" for known in SOLVERS)
        raise ValueError(f"solver_block_ref must be one of {accepted}; got {ref!r}.")
    return slug, name


def solver_class(slug: str) -> type:
    """Import the solver block class of ``slug``.

    Raises:
        ModuleNotFoundError: If the solver's package is not installed.
    """
    module_name, class_name, extra = SOLVERS[slug]
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        hint = f'pip install "sqd[{extra}]"' if extra else "reinstall sqd"
        raise ModuleNotFoundError(
            f"The {slug} solver needs {module_name.replace('_', '-')}: {hint}."
        ) from exc
    return getattr(module, class_name)
