"""Automation script for DICE solver MPI performance tuning."""

import inspect
import json
import os
import subprocess
import uuid

from prefect import flow, get_client, get_run_logger
from prefect.artifacts import ArtifactFilter, create_table_artifact
from pydantic import BaseModel, Field
from qcsc_prefect_blocks.common.blocks import ExecutionProfileBlock
from qcsc_prefect_dice import DiceSHCISolverJob
from sqd_dice.main import DEFAULT_RUNTIME_BLOCK_NAME, Parameters, sqd_2405_05068


async def _maybe_await(value):
    if inspect.isawaitable(value):
        return await value
    return value


class DICESetup(BaseModel):
    """Control parameters in a single experiment attempt."""

    mpiprocs: int = Field(
        description="Number of MPI process per node to request.",
        gt=0,
        title="Num MPI Processes",
    )

    num_nodes: int = Field(
        description=(
            "Number of compute node to request. "
            "Maximum number of available node per job depends on the queue name."
        ),
        gt=0,
        title="Num Nodes",
    )

    subspace_dim: int = Field(
        description="SQD: Dimension d of subsampled bitstrings for diagonalization.",
        title="Subspace Dimension",
        ge=1,
    )


@flow(
    name="dice_perf_tuning",
)
async def main(
    base_parameters: Parameters,
    experiments: list[DICESetup],
    runner_name: str = DEFAULT_RUNTIME_BLOCK_NAME,
    solver_name: str = "sqd-solver",
    option_name: str = "sampler_options",
) -> int:
    """MPI parameter tuning of DICE solver.

    .. note::
        SQD setup max_iterations and num_batches are fixed to 1.
        Only one HPC job is scheduled per sub-experiment.

    Args:
        base_parameters: Base parameter of the SQD experiment.
        experiments: Parameters to overwrite.
        runner_name: Name of QuantumRuntime block to load.
        solver_name: Name of DiceSHCISolverJob block to load.
        option_name: Name of Variable storing sampler primitive options to load.
    """
    logger = get_run_logger()

    records = []
    for experiment in experiments:
        logger.info(f"Running a sub experiment with {experiment}")

        tmp_name = f"tmp-solver-{uuid.uuid4().hex}"
        tmp_exec_name = f"tmp-exec-{uuid.uuid4().hex}"

        # Create temporary solver configuration
        current_solver = await _maybe_await(DiceSHCISolverJob.load(solver_name))
        current_exec = await _maybe_await(
            ExecutionProfileBlock.load(current_solver.execution_profile_block_name)
        )
        current_exec.num_nodes = experiment.num_nodes
        current_exec.mpiprocs = experiment.mpiprocs
        await _maybe_await(current_exec.save(tmp_exec_name, overwrite=True))
        current_solver.execution_profile_block_name = tmp_exec_name
        await _maybe_await(current_solver.save(tmp_name, overwrite=True))
        logger.debug(
            f"Created temporary solver/execution-profile blocks {tmp_name}, {tmp_exec_name}"
        )

        # Update parameters
        current_params = base_parameters.model_copy()
        current_params.sqd.subspace_dim = experiment.subspace_dim
        current_params.sqd.max_iterations = 1
        current_params.sqd.num_batches = 1

        try:
            state = await sqd_2405_05068(
                parameters=current_params,
                runner_name=runner_name,
                solver_name=tmp_name,
                option_name=option_name,
                cache_compute_integrals=True,
                cache_sampling=True,
                return_state=True,
            )
            # Check result
            record = {
                "d": experiment.subspace_dim,
                "mpiprocs": experiment.mpiprocs,
                "nodes": experiment.num_nodes,
                "energy": None,
                "token": None,
                "cpupercent": None,
                "cput": None,
                "walltime": None,
                "mem": None,
                "avg_node_mem": None,
            }
            if state.is_completed():
                logger.info(
                    f"Sub experiment {str(state.state_details.flow_run_id)} completed successfully."
                )
                record["energy"] = await state.result()
            # Read job report artifact
            async with get_client() as client:
                read_job_reports = ArtifactFilter(
                    flow_run_id={"any_": [state.state_details.flow_run_id]},
                    key={"any_": [current_solver.metrics_artifact_key]},
                )
                reports = await client.read_artifacts(
                    artifact_filter=read_job_reports,
                )
            if len(reports) != 0:
                logger.info(
                    f"Found {len(reports)} '{current_solver.metrics_artifact_key}' artifacts."
                )
                report_data = json.loads(reports[0].data)
                report_data = dict(zip(*report_data))
                record["token"] = report_data["token"]
                record["cpupercent"] = report_data.get("cpupercent", None)
                record["cput"] = report_data.get("cput", None)
                record["walltime"] = report_data.get("walltime", None)
                record["mem"] = report_data.get("mem", None)
                node_mems = []
                for key, val in report_data.items():
                    if key.startswith("mem_per_nodes"):
                        val_num = float(val[:-2])
                        node_mems.append(val_num)
                if len(node_mems) > 0:
                    record["avg_node_mem"] = f"{sum(node_mems) / len(node_mems):.5f}gb"
            records.append(record)
        finally:
            await _maybe_await(DiceSHCISolverJob.delete(tmp_name))
            await _maybe_await(ExecutionProfileBlock.delete(tmp_exec_name))

    await create_table_artifact(
        table=records,
        key="dice-performance",
    )

    return 0


@main.on_completion
def terminate_if_job(flow, flow_run, state):
    if job_id := os.getenv("PBS_JOBID"):
        subprocess.run(["qdel", job_id])


if __name__ == "__main__":
    main.serve(
        name="dice_perf_tuning",
        description="MPI parameter tuning of DICE solver for SQD application.",
    )
