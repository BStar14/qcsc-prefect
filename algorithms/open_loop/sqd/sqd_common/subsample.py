"""Post-selection and subsampling with precise control of the subspace dimension.

Both SQD workflows share this module, so the DICE and fulqrum solvers see the
same subspaces for the same RNG state.

- CI strings are ``int64`` for ``norb < 64`` and Python ints (``object`` dtype)
  otherwise, which the fulqrum solver accepts.
- Open-shell batches keep at least one string per spin, so a very small
  ``subspace_dim`` cannot produce an empty subspace.
- Carryover CI strings, the heaviest strings of the previous iteration's ground
  state, come first in every batch and count toward its size. The rest of a
  batch is drawn from the sampled strings without them. Without carryover the
  batches and the RNG draws are those of plain subsampling.
"""

from typing import Iterator

import numpy as np
from qiskit_addon_sqd.subsampling import postselect_by_hamming_right_and_left


def postselect(
    bitstring_matrix: np.ndarray,
    probabilities: np.ndarray,
    *,
    hamming_right: int,
    hamming_left: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Post-select only bitstrings with correct hamming weight.

    Args:
        bitstring_matrix: A 2D array of ``bool`` representations of bit
            values such that each row represents a single bitstring.
        probabilities: A 1D array specifying a probability distribution over the bitstrings.
        hamming_right: The target hamming weight for the right half of sampled bitstrings.
        hamming_left: The target hamming weight for the left half of sampled bitstrings.
    """
    return postselect_by_hamming_right_and_left(
        bitstring_matrix,
        probabilities,
        hamming_right=hamming_right,
        hamming_left=hamming_left,
    )


def bitstrings_to_ci_strs(bitstring_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert each row into (alpha, beta) CI strings without deduplication.

    The right half of a row is alpha (spin-up), the left half is beta, and the
    rightmost bit of each half is orbital 0.

    Returns:
        ``(ci_strs_a, ci_strs_b)`` with one entry per row: ``int64`` for
        ``norb < 64``, Python ints (``object`` dtype) otherwise.
    """
    norb = bitstring_matrix.shape[1] // 2
    num_configs = bitstring_matrix.shape[0]
    dtype = np.int64 if norb < 64 else object
    ci_strs_a = np.zeros(num_configs, dtype=dtype)
    ci_strs_b = np.zeros(num_configs, dtype=dtype)

    # For performance, we accumulate the left and right CI strings together, column-wise,
    # across the two halves of the input bitstring matrix.
    for i in range(norb):
        weight = 1 << (norb - 1 - i)
        ci_strs_b += bitstring_matrix[:, i].astype(dtype) * weight
        ci_strs_a += bitstring_matrix[:, norb + i].astype(dtype) * weight
    return ci_strs_a, ci_strs_b


def subsample(
    bitstring_matrix: np.ndarray,
    probabilities: np.ndarray,
    subspace_dim: int,
    num_batches: int,
    rng: np.random.Generator,
    open_shell: bool = False,
    carryover: tuple[np.ndarray, np.ndarray] | None = None,
    nelec: tuple[int, int] | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Subsample batches of integer representation of determinants from unique set.

    Args:
        bitstring_matrix: A 2D array of ``bool`` representations of bit
            values such that each row represents a single bitstring.
        probabilities: A 1D array specifying a probability distribution over the bitstrings.
        subspace_dim: A target dimension of subspace for diagonalization.
        num_batches: Number of subspaces to draw when the full set exceeds ``subspace_dim``.
        rng: A random number generator to control random behavior.
        open_shell: A flag specifying whether unique configurations from the left and right
            halves of the bitstrings should be kept separate. If ``False``, configurations
            from the left and right halves of the bitstrings are combined into a single
            set of unique configurations. That combined set will be returned for both the left
            and right bitstrings.
        carryover: Alpha and beta CI strings kept in every batch, heaviest first. They
            are kept even when the samples lack them. Strings with a wrong electron
            number are dropped. A closed-shell run uses the alpha strings followed by
            the beta strings, without repeats, for both spins.
        nelec: Number of alpha and beta electrons, to check the carryover strings.
            Defaults to the Hamming weights of the first row of ``bitstring_matrix``.

    Batches have at most ``max(1, int(sqrt(subspace_dim)))`` strings (closed shell)
    or the per-spin sizes of plain subsampling (open shell). The carryover strings
    fill a batch first, truncated to that size, and the rest is drawn without
    replacement from the other sampled strings, with renormalized probabilities.
    When the sampled and carryover strings together fit in ``subspace_dim``, the one
    batch is their union.

    Yields:
        A length-2 tuple of determinant lists representing the right (spin-up)
        and left (spin-down) halves of the bitstrings, respectively.
    """
    ci_strs_a, ci_strs_b = bitstrings_to_ci_strs(bitstring_matrix)
    norb = bitstring_matrix.shape[1] // 2
    if nelec is None and len(bitstring_matrix):
        row = np.asarray(bitstring_matrix[0], dtype=bool)
        nelec = (int(row[norb:].sum()), int(row[:norb].sum()))
    carry_a, carry_b = _carryover_ci_strs(carryover, norb, nelec, open_shell)

    # Reduce duplicated elements from CI strings and accumulate probabilities.
    if not open_shell:
        ci_strs_unique, ci_probs_unique = _unique_and_accumulate_probs(
            ci_strings=np.concatenate((ci_strs_a, ci_strs_b)),
            probabilities=np.tile(probabilities, 2) / 2.0,
        )
        forced = _unique_in_order(np.concatenate((carry_a, carry_b)))
        ci_strs_all = _union(ci_strs_unique, forced)
        net_dim = len(ci_strs_all) ** 2
        if net_dim > subspace_dim:
            for _ in range(num_batches):
                ci_strs_subsampled = _draw_with_forced(
                    ci_strs_unique,
                    ci_probs_unique,
                    forced,
                    size=max(1, int(np.sqrt(subspace_dim))),
                    rng=rng,
                )
                yield (ci_strs_subsampled, ci_strs_subsampled)
        else:
            # Only returns a single set because running diagonalization
            # on the same subspace doesn't improve results.
            yield (ci_strs_all, ci_strs_all)
    else:
        ci_strs_a_unique, ci_probs_a_unique = _unique_and_accumulate_probs(
            ci_strings=ci_strs_a,
            probabilities=probabilities,
        )
        ci_strs_b_unique, ci_probs_b_unique = _unique_and_accumulate_probs(
            ci_strings=ci_strs_b,
            probabilities=probabilities,
        )
        ci_strs_a_all = _union(ci_strs_a_unique, carry_a)
        ci_strs_b_all = _union(ci_strs_b_unique, carry_b)
        net_dim = len(ci_strs_a_all) * len(ci_strs_b_all)
        truncate_rate = np.sqrt(subspace_dim / net_dim)
        if net_dim > subspace_dim:
            for _ in range(num_batches):
                ci_strs_a_subsampled = _draw_with_forced(
                    ci_strs_a_unique,
                    ci_probs_a_unique,
                    carry_a,
                    size=max(1, int(len(ci_strs_a_all) * truncate_rate)),
                    rng=rng,
                )
                ci_strs_b_subsampled = _draw_with_forced(
                    ci_strs_b_unique,
                    ci_probs_b_unique,
                    carry_b,
                    size=max(1, int(len(ci_strs_b_all) * truncate_rate)),
                    rng=rng,
                )
                yield (ci_strs_a_subsampled, ci_strs_b_subsampled)
        else:
            # Only returns a single set because running diagonalization
            # on the same subspace doesn't improve results.
            yield (ci_strs_a_all, ci_strs_b_all)


def _carryover_ci_strs(
    carryover: tuple[np.ndarray, np.ndarray] | None,
    norb: int,
    nelec: tuple[int, int] | None,
    open_shell: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Carryover strings of each spin with a valid electron number, without repeats.

    The strings keep their order and get the dtype of :func:`bitstrings_to_ci_strs`.
    A closed-shell run shares one set between the spins, so it accepts either
    electron number.
    """
    dtype = np.int64 if norb < 64 else object
    if carryover is None:
        return np.zeros(0, dtype=dtype), np.zeros(0, dtype=dtype)
    out = []
    for spin, strings in enumerate(carryover):
        if nelec is None:
            allowed = None
        else:
            allowed = {int(nelec[spin])} if open_shell else {int(n) for n in nelec}
        values = [] if strings is None else [int(x) for x in np.asarray(strings).reshape(-1)]
        kept = [
            v
            for v in values
            if 0 <= v < (1 << norb) and (allowed is None or v.bit_count() in allowed)
        ]
        out.append(_unique_in_order(np.array(kept, dtype=dtype)))
    return out[0], out[1]


def _unique_in_order(ci_strings: np.ndarray) -> np.ndarray:
    """``ci_strings`` without repeats, in the order of their first occurrence."""
    if len(ci_strings) == 0:
        return ci_strings
    _, first = np.unique(ci_strings, return_index=True)
    return ci_strings[np.sort(first)]


def _union(ci_strs_unique: np.ndarray, forced: np.ndarray) -> np.ndarray:
    """Sorted union of the unique sampled strings and the carryover strings."""
    if len(forced) == 0:
        return ci_strs_unique
    return np.unique(np.concatenate((ci_strs_unique, forced)))


def _draw_with_forced(
    ci_strs_unique: np.ndarray,
    ci_probs_unique: np.ndarray,
    forced: np.ndarray,
    *,
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """``size`` strings: the forced ones first, then ones drawn from the rest.

    Without forced strings, this is the plain draw of ``size`` strings, so the RNG
    stream is the same as without carryover.
    """
    if len(forced) == 0:
        return rng.choice(ci_strs_unique, size=size, replace=False, p=ci_probs_unique)
    forced = forced[:size]
    remaining = size - len(forced)
    if remaining <= 0:
        return forced
    mask = ~np.isin(ci_strs_unique, forced)
    candidates, weights = ci_strs_unique[mask], ci_probs_unique[mask]
    if len(candidates) <= remaining:
        drawn = candidates
    else:
        total = weights.sum()
        drawn = rng.choice(
            candidates,
            size=remaining,
            replace=False,
            p=weights / total if total > 0 else None,
        )
    return np.concatenate((forced, drawn))


def _unique_and_accumulate_probs(
    ci_strings: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce duplicated elements from CI string and accumulate probability.

    Args:
        ci_strings: CI strings to reduce overlapped element.
        probabilities: Probabilities corresponding to the CI strings.

    Returns:
        Unique CI strings and associated probabilities.
    """
    ci_strs_unique, ci_strs_inv = np.unique(
        ci_strings,
        return_inverse=True,
    )
    ci_probs_unique = np.bincount(
        ci_strs_inv.reshape(-1),
        weights=probabilities,
        minlength=len(ci_strs_unique),
    )
    return ci_strs_unique, ci_probs_unique
