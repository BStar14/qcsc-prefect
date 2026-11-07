"""Carryover CI strings and occupancies from the marginal weights of a ground state.

This module needs only numpy, so the compute-node solver, the distributed GPU
path and the SQD loop on the workflow host all use the same selection.

A ground state ``c[ia, ib]`` on the subspace ``strs_a x strs_b`` gives each
alpha string the marginal weight ``w_a[ia] = sum_ib |c[ia, ib]|^2``, and each
beta string ``w_b[ib] = sum_ia |c[ia, ib]|^2``. The carryover of a spin is its
``int(fraction * n)`` heaviest strings, as in the ratio mode of the SBD solver.
Each spin is selected on its own, so a closed-shell state gives the same alpha
and beta carryover.
"""

from __future__ import annotations

import numpy as np


def marginal_weights(probs: np.ndarray, na: int, nb: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-string marginal weights of a state.

    Args:
        probs: ``|c|^2`` either as a flat vector in fulqrum's basis order
            (``ib * na + ia``) or as an ``(na, nb)`` matrix ``probs[ia, ib]``.
        na: Number of alpha strings.
        nb: Number of beta strings.

    Returns:
        ``(w_a, w_b)``, of lengths ``na`` and ``nb``.

    Raises:
        ValueError: If ``probs`` has neither shape.
    """
    probs = np.asarray(probs, dtype=np.float64)
    if probs.shape == (na, nb):
        matrix = probs
    elif probs.shape == (na * nb,):
        matrix = probs.reshape(nb, na).T
    else:
        raise ValueError(f"probs shape {probs.shape} is neither ({na * nb},) nor ({na}, {nb})")
    return matrix.sum(axis=1), matrix.sum(axis=0)


def _top_fraction(weights: np.ndarray, strs: np.ndarray, fraction: float) -> np.ndarray:
    kept = int(fraction * len(strs))
    if kept <= 0:
        return np.asarray(strs)[:0]
    # A stable sort on the negated weights keeps equal weights in subspace order.
    order = np.argsort(-np.asarray(weights, dtype=np.float64), kind="stable")
    return np.asarray(strs)[order[:kept]]


def select_carryover(
    w_a: np.ndarray,
    w_b: np.ndarray,
    strs_a: np.ndarray,
    strs_b: np.ndarray,
    fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    """The heaviest ``int(fraction * n)`` strings of each spin, heaviest first.

    Args:
        w_a: Marginal weight of each alpha string.
        w_b: Marginal weight of each beta string.
        strs_a: Alpha CI strings of the subspace, in the order of ``w_a``.
        strs_b: Beta CI strings of the subspace, in the order of ``w_b``.
        fraction: Share of each spin's strings to keep, in ``[0, 1]``. ``0`` keeps none.

    Returns:
        ``(carryover_a, carryover_b)`` with the dtype of ``strs_a`` and ``strs_b``.

    Raises:
        ValueError: If ``fraction`` is outside ``[0, 1]`` or the lengths differ.
    """
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must be in [0, 1], got {fraction}")
    if len(w_a) != len(strs_a) or len(w_b) != len(strs_b):
        raise ValueError("each weight vector must match its CI strings in length")
    return _top_fraction(w_a, strs_a, fraction), _top_fraction(w_b, strs_b, fraction)


def _occupancy(weights: np.ndarray, strs: np.ndarray, norb: int) -> np.ndarray:
    # Bit i of a CI string is the occupation of orbital i.
    bits = np.array([[(int(s) >> i) & 1 for i in range(norb)] for s in strs], dtype=np.float64)
    return np.asarray(weights, dtype=np.float64) @ bits.reshape(len(strs), norb)


def occupancies_from_marginals(
    w_a: np.ndarray,
    w_b: np.ndarray,
    strs_a: np.ndarray,
    strs_b: np.ndarray,
    norb: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Average ``(alpha, beta)`` occupation of each orbital.

    The occupation of orbital ``i`` by spin alpha is the total weight of the
    alpha strings with bit ``i`` set, which is what ``Subspace.get_orbital_occupancies``
    computes from the full state. Normalized weights give occupancies that sum to
    the electron numbers.
    """
    return _occupancy(w_a, strs_a, norb), _occupancy(w_b, strs_b, norb)
