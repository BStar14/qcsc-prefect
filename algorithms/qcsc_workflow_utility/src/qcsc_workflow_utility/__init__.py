"""Common utilities for quantum algorithms."""

from .chem import (
    ElectronicProperties,
    NpStrict1DArrayF64,
    NpStrict2DArrayF64,
    NpStrict4DArrayF64,
    compute_molecular_integrals_from_fcidump,
    compute_molecular_integrals_from_geometry,
)
from .solvers import (
    DICE_SLUG,
    SBD_SLUG,
    parse_block_ref,
    solver_class,
    to_sbd_closed_shell,
)

__all__ = [
    "ElectronicProperties",
    "compute_molecular_integrals_from_geometry",
    "compute_molecular_integrals_from_fcidump",
    "NpStrict1DArrayF64",
    "NpStrict2DArrayF64",
    "NpStrict4DArrayF64",
    "DICE_SLUG",
    "SBD_SLUG",
    "parse_block_ref",
    "solver_class",
    "to_sbd_closed_shell",
]
