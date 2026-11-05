"""Solver selection by block reference and the open-loop input conversion for SBD.

DICE and SBD take the same ``run`` arguments but solve different problems. These helpers
make the difference explicit instead of swapping one solver for the other silently.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping

import numpy as np

DICE_SLUG = "dice_shci_solver_job"
SBD_SLUG = "sbd_solver_job"


def parse_block_ref(ref: str) -> tuple[str, str]:
    """Split ``"<block_type_slug>/<block_document_name>"``."""
    parts = ref.split("/", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(f"Invalid solver_block_ref: {ref}")
    return parts[0], parts[1]


def solver_class(ref: str, classes: Mapping[str, type]) -> tuple[type, str]:
    """Return the solver block class selected by the slug of ``ref``, and the block name.

    ``classes`` maps the accepted slugs to their block classes. Prefect also writes slugs
    with hyphens (``sbd-solver-job``); both spellings are accepted.
    """
    slug, name = parse_block_ref(ref)
    normalized = slug.replace("-", "_")
    if normalized not in classes:
        accepted = ", ".join(f"'{slug}/<name>'" for slug in classes)
        raise ValueError(f"solver_block_ref must be one of {accepted}; got {ref!r}.")
    return classes[normalized], name


def to_sbd_closed_shell(
    batches: Iterable[tuple[np.ndarray, np.ndarray]],
    *,
    subspace_dim: int,
    num_elec_a: int,
    probabilities: Mapping[int, float] | None = None,
    rng: np.random.Generator | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Convert open-loop closed-shell subspaces into the form the SBD solver requires.

    Each batch ``(s, s)`` becomes ``(t, t)`` where ``t`` starts with the Hartree-Fock
    determinant ``(1 << num_elec_a) - 1``, followed by the other determinants of ``s``.
    The subspace size limit is kept: at most ``int(sqrt(subspace_dim)) - 1`` determinants
    other than Hartree-Fock. When ``s`` has more, they are chosen without replacement,
    weighted by ``probabilities`` (uniform if omitted), with ``rng``.

    This always includes Hartree-Fock, so the subspace differs from the one DICE gets.
    Open-shell batches (different alpha and beta sets) are passed through unchanged; the
    SBD solver rejects them.
    """
    hartree_fock = (1 << num_elec_a) - 1
    limit = int(np.sqrt(subspace_dim)) - 1
    if limit < 0:
        raise ValueError("subspace_dim must be at least 1")
    for alpha, beta in batches:
        alpha = np.asarray(alpha, dtype=np.longlong).reshape(-1)
        beta = np.asarray(beta, dtype=np.longlong).reshape(-1)
        if not np.array_equal(alpha, beta):
            yield alpha, beta
            continue
        others = np.unique(alpha[alpha != hartree_fock])
        if len(others) > limit:
            if rng is None:
                raise ValueError("rng is required to subsample the SBD subspace")
            weights = None
            if probabilities is not None:
                weights = np.array([probabilities.get(int(ci), 0.0) for ci in others])
                weights = weights / weights.sum() if weights.sum() > 0 else None
            others = rng.choice(others, size=limit, replace=False, p=weights)
        strings = np.concatenate(([hartree_fock], others)).astype(np.longlong)
        yield strings, strings
