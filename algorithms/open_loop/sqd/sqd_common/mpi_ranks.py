"""Check the MPI rank count of a solver job against ``num_nodes x mpiprocs``.

``sqd-create-blocks`` checks the SBD grid and the fulqrum ``process_grid``
against ``num_nodes x mpiprocs``. The launcher may take the count from elsewhere:
``mpiexec -n`` or ``mpirun -np`` in ``mpi_options`` (Miyabi, Fugaku), and on
Fugaku ``max-proc-per-node`` in ``fugaku_mpi_options_for_pjm`` places the ranks on
the nodes. These counts must agree with the blocks, or the grid the solver gets
is not the one that was checked.
"""

from __future__ import annotations

from typing import Any

_COUNT_FLAGS = ("-n", "-np", "--np", "--ntasks")
_PER_NODE_OPTION = "max-proc-per-node"


def _tokens(options: list[str] | None) -> list[str]:
    """The options split on whitespace, so ``["-np 4"]`` reads like ``["-np", "4"]``."""
    return [token for option in options or [] for token in str(option).split()]


def mpi_rank_count(mpi_options: list[str] | None) -> int | None:
    """The rank count that ``-n``, ``-np``, ``--np`` or ``--ntasks`` sets, or None."""
    tokens = _tokens(mpi_options)
    for index, token in enumerate(tokens):
        flag, has_value, value = token.partition("=")
        if flag not in _COUNT_FLAGS:
            continue
        if not has_value:
            if index + 1 == len(tokens):
                raise RuntimeError(f"mpi_options ends with {flag} and no rank count.")
            value = tokens[index + 1]
        try:
            return int(value)
        except ValueError:
            raise RuntimeError(f"mpi_options sets {flag} {value}, which is not a number.") from None
    return None


def max_proc_per_node(pjm_options: list[str] | None) -> int | None:
    """The ``max-proc-per-node`` value of ``fugaku_mpi_options_for_pjm``, or None."""
    for token in _tokens(pjm_options):
        for part in token.split(","):
            key, _, value = part.partition("=")
            if key.strip() == _PER_NODE_OPTION:
                try:
                    return int(value)
                except ValueError:
                    raise RuntimeError(
                        f"fugaku_mpi_options_for_pjm sets {_PER_NODE_OPTION}={value}, which is "
                        "not a number."
                    ) from None
    return None


def check_rank_count(block_kwargs: dict[str, Any]) -> None:
    """Stop when ``mpi_options`` or ``max-proc-per-node`` disagrees with the blocks."""
    if block_kwargs.get("launcher") == "single":
        return
    num_nodes, mpiprocs = int(block_kwargs["num_nodes"]), int(block_kwargs["mpiprocs"])
    ranks = num_nodes * mpiprocs
    count = mpi_rank_count(block_kwargs.get("mpi_options"))
    if count is not None and count != ranks:
        raise RuntimeError(
            f"mpi_options starts {count} MPI ranks, but num_nodes x mpiprocs = "
            f"{num_nodes} x {mpiprocs} = {ranks}. Set the count in mpi_options to {ranks}, "
            "or change num_nodes or mpiprocs."
        )
    if block_kwargs.get("hpc_target") == "fugaku":
        per_node = max_proc_per_node(block_kwargs.get("mpi_options_for_pjm"))
        if per_node is not None and per_node != mpiprocs:
            raise RuntimeError(
                f"fugaku_mpi_options_for_pjm places {_PER_NODE_OPTION}={per_node} ranks on "
                f"each node, but mpiprocs = {mpiprocs}. Set {_PER_NODE_OPTION}={mpiprocs}, "
                "or change mpiprocs."
            )
