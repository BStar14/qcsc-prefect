"""Definition of the SQD workflow with the DICE SHCI eigensolver.

The stages (integrals, LUCJ parameters, quantum sampling, configuration
recovery) come from ``sqd_common``. Each subspace of the recovery loop is
diagonalized by a ``DiceSHCISolverJob`` block, one MPI job per batch.
"""

import os
import pathlib

from prefect import flow
from pydantic import Field
from qcsc_prefect_dice import DiceSHCISolverJob
from sqd_common import flow_params
from sqd_common.flow import run_sqd_pattern

DEFAULT_SOLVER_BLOCK_NAME = "sqd-solver"
DEPLOYMENT_NAME = "sqd_2405_05068"
ARTIFACT_KEY = "sqd-configuration-recovery"


class FlowParameters(flow_params.FlowParameters):
    """Parameters of an SQD run diagonalized with DICE."""

    solver_block_name: str = Field(
        default=DEFAULT_SOLVER_BLOCK_NAME,
        description="Name of the DiceSHCISolverJob block that diagonalizes each subspace.",
        title="Solver Block",
    )


@flow
async def sqd_2405_05068(parameters: FlowParameters) -> float:
    """SQD Experiment from arXiv2405.05068.

    Args:
        parameters: Workflow parameters.

    Returns:
        Minimum ground state energy solved by SQD.
    """
    return await run_sqd_pattern(parameters, DiceSHCISolverJob, ARTIFACT_KEY)


def deploy():
    """Deploy workflow with a local worker."""
    # Prefect deploys with relative path.
    # Workflow is now installed in site-packages.
    os.chdir(pathlib.Path(__file__).parent)

    sqd_2405_05068.with_options(version=os.getenv("WF_VERSION", "unknown")).serve(
        name=DEPLOYMENT_NAME,
        description="SQD experiment from arXiv2405.05068.",
    )
