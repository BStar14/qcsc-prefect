"""Modification of subsample code for presice control of subspace dimension."""

from typing import Iterator

import numpy as np
from qiskit_addon_sqd.configuration_recovery import post_select_by_hamming_weight


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
    mask_postsel = post_select_by_hamming_weight(
        bitstring_matrix,
        hamming_right=hamming_right,
        hamming_left=hamming_left,
    )
    bs_mat_postsel = bitstring_matrix[mask_postsel]
    probs_postsel = probabilities[mask_postsel]
    probs_postsel = np.abs(probs_postsel) / np.sum(np.abs(probs_postsel))

    return bs_mat_postsel, probs_postsel


def subsample(
    bitstring_matrix: np.ndarray,
    probabilities: np.ndarray,
    subspace_dim: int,
    num_batches: int,
    rng: np.random.Generator,
    open_shell: bool = False,
) -> Iterator[np.ndarray]:
    """Subsample batches of integer representation of determinants from unique set.

    Args:
        bitstring_matrix: A 2D array of ``bool`` representations of bit
            values such that each row represents a single bitstring.
        probabilities: A 1D array specifying a probability distribution over the bitstrings.
        subspace_dim: A target dimension of subspace for diagonalization.
        rng: A random number generator to control random behavior.
        open_shell: A flag specifying whether unique configurations from the left and right
            halves of the bitstrings should be kept separate. If ``False``, configurations
            from the left and right halves of the bitstrings are combined into a single
            set of unique configurations. That combined set will be returned for both the left
            and right bitstrings.

    Yields:
        A length-2 tuple of determinant lists representing the right (spin-up)
        and left (spin-down) halves of the bitstrings, respectively.
    """
    norb = bitstring_matrix.shape[1] // 2
    num_configs = bitstring_matrix.shape[0]
    ci_strs_a = np.zeros(num_configs, dtype=np.longlong)
    ci_strs_b = np.zeros(num_configs, dtype=np.longlong)

    # For performance, we accumulate the left and right CI strings together, column-wise,
    # across the two halves of the input bitstring matrix.
    for i in range(norb):
        ci_strs_b[:] += bitstring_matrix[:, i] * 2 ** (norb - 1 - i)
        ci_strs_a[:] += bitstring_matrix[:, norb + i] * 2 ** (norb - 1 - i)

    # Reduce duplicated elements from CI strings and accumurate probabilities.
    if not open_shell:
        ci_strs_unique, ci_probs_unique = _unique_and_accumurate_probs(
            ci_strings=np.concatenate((ci_strs_a, ci_strs_b)),
            probabilities=np.tile(probabilities, 2) / 2.0,
        )
        net_dim = len(ci_strs_unique) ** 2
        if net_dim > subspace_dim:
            for _ in range(num_batches):
                ci_strs_subsampled = rng.choice(
                    ci_strs_unique,
                    size=int(np.sqrt(subspace_dim)),
                    replace=False,
                    p=ci_probs_unique,
                )
                yield (ci_strs_subsampled, ci_strs_subsampled)
        else:
            # Only returns a single set because running diagonalization
            # on the same subspace doesn't improve results.
            yield (ci_strs_unique, ci_strs_unique)
    else:
        ci_strs_a_unique, ci_probs_a_unique = _unique_and_accumurate_probs(
            ci_strings=ci_strs_a,
            probabilities=probabilities,
        )
        ci_strs_b_unique, ci_probs_b_unique = _unique_and_accumurate_probs(
            ci_strings=ci_strs_b,
            probabilities=probabilities,
        )
        net_dim = len(ci_strs_a_unique) * len(ci_strs_b_unique)
        truncate_rate = np.sqrt(subspace_dim / net_dim)
        if net_dim > subspace_dim:
            for _ in range(num_batches):
                ci_strs_a_subsampled = rng.choice(
                    ci_strs_a_unique,
                    size=int(len(ci_strs_a_unique) * truncate_rate),
                    replace=False,
                    p=ci_probs_a_unique,
                )
                ci_strs_b_subsampled = rng.choice(
                    ci_strs_b_unique,
                    size=int(len(ci_strs_b_unique) * truncate_rate),
                    replace=False,
                    p=ci_probs_b_unique,
                )
                yield (ci_strs_a_subsampled, ci_strs_b_subsampled)
        else:
            # Only returns a single set because running diagonalization
            # on the same subspace doesn't improve results.
            yield (ci_strs_a_unique, ci_strs_b_unique)


def _unique_and_accumurate_probs(
    ci_strings: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce duplicated elements from CI string and accumurate probability.

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
        ci_strs_inv,
        weights=probabilities,
        minlength=len(ci_strs_unique),
    )
    return ci_strs_unique, ci_probs_unique
