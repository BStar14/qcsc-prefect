"""Definition of the SQD workflow with the fulqrum eigensolver.

The stages (integrals, LUCJ parameters, quantum sampling, configuration
recovery) come from ``sqd_common``. Each subspace of the recovery loop is
diagonalized by a ``FulqrumSolverJob`` block, one scheduler job per batch.
"""

import os
import pathlib

from prefect import flow, get_run_logger
from pydantic import Field
from qcsc_prefect_fulqrum import FulqrumSolverJob
from sqd_common import flow_params
from sqd_common.flow import run_sqd_pattern

DEFAULT_SOLVER_BLOCK_NAME = "sqd-fulqrum-solver"
FLOW_NAME = "sqd_2405_05068_fulqrum"
ARTIFACT_KEY = "sqd-fulqrum-configuration-recovery"


class FlowParameters(flow_params.FlowParameters):
    """Parameters of an SQD run diagonalized with fulqrum."""

    solver_block_name: str = Field(
        default=DEFAULT_SOLVER_BLOCK_NAME,
        description="Name of the FulqrumSolverJob block that diagonalizes each subspace.",
        title="Solver Block",
    )


@flow(name=FLOW_NAME)
async def sqd_2405_05068_fulqrum(parameters: FlowParameters) -> float:
    """SQD Experiment from arXiv2405.05068, diagonalized with fulqrum.

    Args:
        parameters: Workflow parameters.

    Returns:
        Minimum ground state energy solved by SQD.
    """
    if parameters.molecule.spin_sq:
        get_run_logger().warning(
            f"spin_sq={parameters.molecule.spin_sq} is ignored: "
            "the fulqrum solver applies no spin penalty."
        )
    return await run_sqd_pattern(parameters, FulqrumSolverJob, ARTIFACT_KEY)


def deploy():
    """Deploy workflow with a local worker."""
    # Prefect deploys with relative path.
    # Workflow is now installed in site-packages.
    os.chdir(pathlib.Path(__file__).parent)

    sqd_2405_05068_fulqrum.with_options(version=os.getenv("WF_VERSION", "unknown")).serve(
        name=FLOW_NAME,
        description="SQD experiment from arXiv2405.05068 with the fulqrum eigensolver.",
    )
