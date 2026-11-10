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


def to_sbd_subspace(
    batches: Iterable[tuple[np.ndarray, np.ndarray]],
    *,
    subspace_dim: int,
    num_elec_a: int,
    num_elec_b: int,
    probabilities: Mapping[int, float] | None = None,
    rng: np.random.Generator | None = None,
    keep: tuple[Iterable[int], Iterable[int]] | Iterable[int] | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Convert open-loop subspaces into the form the SBD solver requires.

    SBD needs the Hartree-Fock string of each spin first: ``(1 << num_elec_a) - 1``
    in the alpha list and ``(1 << num_elec_b) - 1`` in the beta list.

    - A closed-shell batch (``num_elec_a == num_elec_b`` and the same alpha and beta
      strings) becomes one list for both spins: Hartree-Fock, then at most
      ``int(sqrt(subspace_dim)) - 1`` other strings of the batch.
    - Any other batch is open shell: each spin gets its own list with its own
      Hartree-Fock string first. With Hartree-Fock added, the subsampled sizes
      ``n_a`` and ``n_b`` are kept when ``n_a * n_b <= subspace_dim``. Otherwise both
      are scaled by ``r = sqrt(subspace_dim / (n_a * n_b))`` to
      ``max(1, int(n * r))``. When one list is held at 1, the other is cut to at
      most ``subspace_dim`` strings. So ``len(alpha) * len(beta) <= subspace_dim``
      always holds.

    When a list must be cut, its strings in ``keep`` (for example the carryover strings
    of the previous iteration) are kept first, in the order of the batch, and the rest
    are chosen without replacement, weighted by ``probabilities`` (uniform if omitted),
    with ``rng``. ``keep`` is either one set of strings for both spins or a pair
    ``(alpha, beta)``; a closed-shell batch keeps the strings of both.

    This always includes Hartree-Fock, so the subspace differs from the one DICE gets.

    Raises:
        ValueError: If ``subspace_dim`` is below 1, or a list must be cut without ``rng``.
    """
    if subspace_dim < 1:
        raise ValueError("subspace_dim must be at least 1")
    hf_a, hf_b = (1 << num_elec_a) - 1, (1 << num_elec_b) - 1
    keep_a, keep_b = _keep_per_spin(keep)
    closed_limit = int(np.sqrt(subspace_dim)) - 1
    for alpha, beta in batches:
        alpha = np.asarray(alpha, dtype=np.longlong).reshape(-1)
        beta = np.asarray(beta, dtype=np.longlong).reshape(-1)
        if num_elec_a == num_elec_b and np.array_equal(alpha, beta):
            strings = _hf_first(alpha, hf_a, closed_limit + 1, keep_a | keep_b, probabilities, rng)
            yield strings, strings
            continue
        others_a = np.unique(alpha[alpha != hf_a])
        others_b = np.unique(beta[beta != hf_b])
        cap_a, cap_b = _open_shell_caps(len(others_a) + 1, len(others_b) + 1, subspace_dim)
        yield (
            _hf_first(alpha, hf_a, cap_a, keep_a, probabilities, rng),
            _hf_first(beta, hf_b, cap_b, keep_b, probabilities, rng),
        )


def to_sbd_closed_shell(
    batches: Iterable[tuple[np.ndarray, np.ndarray]],
    *,
    subspace_dim: int,
    num_elec_a: int,
    probabilities: Mapping[int, float] | None = None,
    rng: np.random.Generator | None = None,
    keep: Iterable[int] | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """The closed-shell conversion of :func:`to_sbd_subspace`.

    Each batch ``(s, s)`` becomes ``(t, t)`` where ``t`` starts with the Hartree-Fock
    determinant ``(1 << num_elec_a) - 1``, followed by at most
    ``int(sqrt(subspace_dim)) - 1`` other determinants of ``s``, chosen as in
    :func:`to_sbd_subspace`. Batches with different alpha and beta sets are passed
    through unchanged.
    """
    if subspace_dim < 1:
        raise ValueError("subspace_dim must be at least 1")
    for alpha, beta in batches:
        alpha = np.asarray(alpha, dtype=np.longlong).reshape(-1)
        beta = np.asarray(beta, dtype=np.longlong).reshape(-1)
        if not np.array_equal(alpha, beta):
            yield alpha, beta
            continue
        yield from to_sbd_subspace(
            [(alpha, beta)],
            subspace_dim=subspace_dim,
            num_elec_a=num_elec_a,
            num_elec_b=num_elec_a,
            probabilities=probabilities,
            rng=rng,
            keep=keep,
        )


def _keep_per_spin(
    keep: tuple[Iterable[int], Iterable[int]] | Iterable[int] | None,
) -> tuple[set[int], set[int]]:
    """The kept strings of each spin; one flat set applies to both."""
    if keep is None:
        return set(), set()
    if isinstance(keep, tuple) and len(keep) == 2 and all(_is_strings(k) for k in keep):
        alpha, beta = keep
        return _int_set(alpha), _int_set(beta)
    strings = _int_set(keep)
    return strings, set(strings)


def _is_strings(value) -> bool:
    """Whether ``value`` is a set of strings rather than one string."""
    return value is None or (isinstance(value, Iterable) and np.ndim(value) > 0)


def _int_set(strings: Iterable[int] | None) -> set[int]:
    if strings is None:
        return set()
    return {int(ci) for ci in np.asarray(list(strings), dtype=object).reshape(-1)}


def _open_shell_caps(n_a: int, n_b: int, subspace_dim: int) -> tuple[int, int]:
    """List sizes, Hartree-Fock included, whose product fits in ``subspace_dim``."""
    if n_a * n_b <= subspace_dim:
        return n_a, n_b
    rate = np.sqrt(subspace_dim / (n_a * n_b))
    cap_a, cap_b = max(1, int(n_a * rate)), max(1, int(n_b * rate))
    # A list held at 1 can leave the other too long; cut that one to fit.
    if cap_a * cap_b > subspace_dim:
        if cap_a >= cap_b:
            cap_a = max(1, subspace_dim // cap_b)
        else:
            cap_b = max(1, subspace_dim // cap_a)
    return cap_a, cap_b


def _hf_first(
    strings: np.ndarray,
    hartree_fock: int,
    size: int,
    kept_strings: set[int],
    probabilities: Mapping[int, float] | None,
    rng: np.random.Generator | None,
) -> np.ndarray:
    """Hartree-Fock, then at most ``size - 1`` other strings of ``strings``."""
    limit = size - 1
    others = np.unique(strings[strings != hartree_fock])
    if len(others) > limit:
        if rng is None:
            raise ValueError("rng is required to subsample the SBD subspace")
        is_kept = np.array([int(ci) in kept_strings for ci in others], dtype=bool)
        forced = strings[np.isin(strings, others[is_kept])]
        forced = forced[np.sort(np.unique(forced, return_index=True)[1])][:limit]
        candidates = others[~np.isin(others, forced)]
        weights = None
        if probabilities is not None:
            weights = np.array([probabilities.get(int(ci), 0.0) for ci in candidates])
            weights = weights / weights.sum() if weights.sum() > 0 else None
        drawn = rng.choice(candidates, size=limit - len(forced), replace=False, p=weights)
        others = np.concatenate((forced, drawn))
    return np.concatenate(([hartree_fock], others)).astype(np.longlong)
