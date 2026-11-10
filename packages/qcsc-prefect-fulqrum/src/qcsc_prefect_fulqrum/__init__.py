"""Fulqrum subspace eigensolver integration for qcsc-prefect workflows.

Only the Prefect-free solver core is imported eagerly, so compute nodes can run
``qcsc-fulqrum-solve`` without Prefect installed. The Prefect-side names
(``FulqrumSolverJob``, ``create_fulqrum_blocks``, ``register_fulqrum_block_types``)
are imported on first access.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

from .solver import FulqrumSolution, FulqrumSolverOptions, solve_fulqrum

if TYPE_CHECKING:
    from .block_utils import create_fulqrum_blocks, register_fulqrum_block_types
    from .solver_job import FulqrumSolverJob

_LAZY = {
    "FulqrumSolverJob": "solver_job",
    "create_fulqrum_blocks": "block_utils",
    "register_fulqrum_block_types": "block_utils",
}

__all__ = [
    "FulqrumSolution",
    "FulqrumSolverJob",
    "FulqrumSolverOptions",
    "create_fulqrum_blocks",
    "register_fulqrum_block_types",
    "solve_fulqrum",
]


def __getattr__(name: str):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{module}", __name__), name)
    globals()[name] = value
    return value
