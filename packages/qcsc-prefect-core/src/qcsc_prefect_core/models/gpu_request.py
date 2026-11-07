"""Checks of a GPU request against the HPC target and the resource class."""

from __future__ import annotations

# The most GPUs a node of each target has; None means no limit is known.
MAX_GPUS_PER_NODE: dict[str, int | None] = {
    "local": 0,
    "fugaku": 0,
    "miyabi": 1,
    "slurm": None,
}


def validate_gpu_request(
    *,
    hpc_target: str,
    resource_class: str,
    gpus_per_node: int | None,
) -> None:
    """Reject a ``gpus_per_node`` that the target cannot schedule.

    Block creation and the job submission both call this, so a bad request fails
    before any job runs, not when the first job script is built.

    - ``local`` and ``fugaku``: no GPU request (local jobs use the GPUs visible to
      the worker; Fugaku nodes have none).
    - ``miyabi``: at most 1, the one GPU of a Miyabi-G node.
    - ``slurm``: any positive count.
    - A GPU request needs ``resource_class="gpu"``, which selects the GPU queue.

    Raises:
        ValueError: If the request does not fit the target.
    """
    if gpus_per_node is None:
        return
    if gpus_per_node < 1:
        raise ValueError(f"gpus_per_node must be at least 1 or unset, got {gpus_per_node}.")
    limit = MAX_GPUS_PER_NODE.get(hpc_target)
    if limit == 0:
        raise ValueError(f"hpc_target {hpc_target!r} takes no GPU request; unset gpus_per_node.")
    if limit is not None and gpus_per_node > limit:
        raise ValueError(
            f"hpc_target {hpc_target!r} nodes have {limit} GPU(s); gpus_per_node is "
            f"{gpus_per_node}. Request more nodes for more GPUs."
        )
    if resource_class != "gpu":
        raise ValueError(
            f"gpus_per_node={gpus_per_node} needs resource_class='gpu', which selects the "
            f"GPU queue; got resource_class={resource_class!r}."
        )
