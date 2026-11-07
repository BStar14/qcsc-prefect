"""The GPU request checks shared by block creation and job submission."""

from __future__ import annotations

import pytest
from qcsc_prefect_core.models.gpu_request import validate_gpu_request


@pytest.mark.parametrize(
    ("hpc_target", "resource_class", "gpus"),
    [
        ("slurm", "gpu", 8),
        ("miyabi", "gpu", 1),
        ("local", "cpu", None),
        ("fugaku", "cpu", None),
        ("slurm", "cpu", None),
    ],
)
def test_valid_requests(hpc_target, resource_class, gpus):
    validate_gpu_request(hpc_target=hpc_target, resource_class=resource_class, gpus_per_node=gpus)


@pytest.mark.parametrize(
    ("hpc_target", "resource_class", "gpus", "match"),
    [
        ("local", "gpu", 1, "takes no GPU request"),
        ("fugaku", "gpu", 2, "takes no GPU request"),
        ("miyabi", "gpu", 4, "have 1 GPU"),
        ("slurm", "cpu", 4, "needs resource_class='gpu'"),
        ("miyabi", "cpu", 1, "needs resource_class='gpu'"),
        ("slurm", "gpu", 0, "at least 1"),
    ],
)
def test_invalid_requests(hpc_target, resource_class, gpus, match):
    with pytest.raises(ValueError, match=match):
        validate_gpu_request(
            hpc_target=hpc_target, resource_class=resource_class, gpus_per_node=gpus
        )
