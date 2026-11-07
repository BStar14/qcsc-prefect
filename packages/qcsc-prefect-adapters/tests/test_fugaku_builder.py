from __future__ import annotations

from pathlib import Path

import pytest
from qcsc_prefect_adapters.fugaku.builder import (
    FugakuJobRequest,
    render_manifest_bulk_script,
    render_script,
)
from qcsc_prefect_core.models.execution_profile import ExecutionProfile


def test_render_fugaku_script_with_modules_env_and_extra_resources(tmp_path: Path):
    profile = ExecutionProfile(
        command_key="hello",
        num_nodes=4,
        mpiprocs=1,
        ompthreads=48,
        walltime="00:10:00",
        launcher="mpirun",
        mpi_options=["-n", "4"],
        modules=["LLVM/llvmorg-21.1.0"],
        environments={
            "OMP_NUM_THREADS": "48",
            "UTOFU_SWAP_PROTECT": "1",
            "LD_LIBRARY_PATH": "/lib64:$LD_LIBRARY_PATH",
        },
        arguments=["--foo", "bar"],
    )
    req = FugakuJobRequest(
        queue_name="small",
        project="ra010014",
        executable="/path/to/gb-demo",
        job_name="gbsqd-test",
        gfscache="/vol0004:/vol0002",
        mpi_options_for_pjm=["max-proc-per-node=1"],
        pjm_resources=["freq=2000,eco_state=2"],
    )

    text = render_script(work_dir=tmp_path, exec_profile=profile, req=req)

    assert '#PJM -L "rscgrp=small"' in text
    assert '#PJM -L "node=4"' in text
    assert '#PJM -L "elapse=00:10:00"' in text
    assert '#PJM -L "freq=2000,eco_state=2"' in text
    assert "#PJM -g ra010014" in text
    assert '#PJM --mpi "max-proc-per-node=1"' in text
    assert "#PJM -x PJM_LLIO_GFSCACHE=/vol0004:/vol0002" in text
    assert "module load LLVM/llvmorg-21.1.0" in text
    assert 'export OMP_NUM_THREADS="48"' in text
    assert 'export UTOFU_SWAP_PROTECT="1"' in text
    assert 'export LD_LIBRARY_PATH="/lib64:$LD_LIBRARY_PATH"' in text
    assert 'QCSC_PREFECT_EXECUTABLE="/path/to/gb-demo"' in text
    assert "QCSC Prefect preflight failed: executable '${QCSC_PREFECT_EXECUTABLE}'" in text
    assert 'mpirun -n 4 "${QCSC_PREFECT_EXECUTABLE}" --foo bar' in text


def test_render_manifest_bulk_script_selects_manifest_with_pjm_bulknum(
    tmp_path: Path,
):
    profile = ExecutionProfile(
        command_key="manifest-command",
        num_nodes=1,
        mpiprocs=1,
        walltime="00:05:00",
        launcher="single",
        arguments=[
            "-m",
            "example_app.qpy_batch",
            "--manifest",
            '"$QCSC_BULK_MANIFEST"',
        ],
    )
    req = FugakuJobRequest(
        queue_name="small",
        project="ra010014",
        executable="python",
        job_name="manifest-bulk",
    )

    text = render_manifest_bulk_script(
        work_dir=tmp_path / "bulk-group-0001",
        bulk_manifest_dir=tmp_path / "bulk-group-0001" / "manifests",
        exec_profile=profile,
        req=req,
    )

    assert 'export QCSC_BULK_MANIFEST_DIR="' in text
    assert 'MANIFEST="${QCSC_BULK_MANIFEST_DIR}/${PJM_BULKNUM}.json"' in text
    assert 'export QCSC_BULK_MANIFEST="${MANIFEST}"' in text
    assert 'export QCSC_BULK_NUM="${PJM_BULKNUM}"' in text
    assert '[ ! -r "${QCSC_BULK_MANIFEST}" ]' in text
    assert (
        '"${QCSC_PREFECT_EXECUTABLE}" -m example_app.qpy_batch --manifest "$QCSC_BULK_MANIFEST"'
    ) in text
    assert "gb_demo" not in text
    assert "TrimSQD" not in text
    assert "/${PJM_BULKNUM}/" not in text
    assert "input_dir" not in text


def test_fugaku_rejects_gpus_per_node(tmp_path: Path):
    profile = ExecutionProfile(command_key="hello", num_nodes=1, gpus_per_node=1)
    req = FugakuJobRequest(queue_name="small", project="ra000000", executable="/path/to/hello")

    with pytest.raises(ValueError, match="Fugaku nodes have no GPUs"):
        render_script(work_dir=tmp_path, exec_profile=profile, req=req)
