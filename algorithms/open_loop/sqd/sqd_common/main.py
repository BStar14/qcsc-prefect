"""The SQD workflow (arXiv:2405.05068) as one Prefect flow, ``sqd``.

The stages (integrals, LUCJ parameters, quantum sampling, configuration
recovery) live in ``sqd_common``. ``parameters.solver_block_ref`` selects the
solver block that diagonalizes each subspace: fulqrum (default), DICE or SBD.
"""

import importlib.metadata
import os
import pathlib

from prefect import flow

from sqd_common.flow import run_sqd_pattern
from sqd_common.flow_params import FlowParameters

FLOW_NAME = "sqd"
DEPLOYMENT_NAME = "sqd"
ARTIFACT_KEY = "sqd-configuration-recovery"


@flow(name=FLOW_NAME)
async def sqd(parameters: FlowParameters) -> float:
    """SQD experiment from arXiv:2405.05068.

    Args:
        parameters: Workflow parameters.

    Returns:
        Minimum ground state energy solved by SQD.
    """
    return await run_sqd_pattern(parameters, ARTIFACT_KEY)


def flow_version() -> str:
    """``WF_VERSION``, else the installed ``sqd`` version."""
    try:
        default = importlib.metadata.version("sqd")
    except importlib.metadata.PackageNotFoundError:
        default = "unknown"
    return os.getenv("WF_VERSION", default)


def deploy():
    """Deploy workflow with a local worker."""
    # Prefect deploys with relative path.
    # Workflow is now installed in site-packages.
    os.chdir(pathlib.Path(__file__).parent)

    sqd.with_options(version=flow_version()).serve(
        name=DEPLOYMENT_NAME,
        description="SQD experiment from arXiv:2405.05068 with the fulqrum, DICE or SBD solver.",
    )
