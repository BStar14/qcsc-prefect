"""Default names of the SQD workflow's Prefect blocks, for each solver and system.

``sqd-create-blocks`` saves the blocks of the solver it is run for under these
names, and records ``solver_ref(...)`` as the Solver Block of the Advanced SQD
Settings block. ``[advanced]`` overrides any of them.

The QRMI sampler blocks and the runtime and advanced settings blocks do not depend
on the solver, so they have one name each.
"""

from __future__ import annotations

from sqd_common.advanced import DEFAULT_ADVANCED_BLOCK_NAME
from sqd_common.flow_params import DEFAULT_QRMI_BLOCK_NAME

# Solver in sqd_blocks.toml -> its solver block type slug.
SOLVER_SLUGS = {
    "fulqrum": "fulqrum_solver_job",
    "dice": "dice_shci_solver_job",
    "sbd": "sbd_solver_job",
}
SCRIPT_SUFFIXES = {"local": "", "slurm": ".slurm", "miyabi": ".pbs", "fugaku": ".pjm"}

QRMI_BLOCK_NAMES = {
    "command_block_name": "cmd-sqd-qrmi",
    "execution_profile_block_name": "exec-sqd-qrmi",
    "hpc_profile_block_name": "hpc-slurm-sqd-qrmi",
}
RUNTIME_BLOCK_NAME = DEFAULT_QRMI_BLOCK_NAME
ADVANCED_BLOCK_NAME = DEFAULT_ADVANCED_BLOCK_NAME


def solver_block_names(solver: str, system: str) -> dict[str, str]:
    """The default names of ``solver``'s blocks on ``system``, plus its script and metrics key.

    Args:
        solver: ``fulqrum``, ``dice`` or ``sbd``.
        system: ``slurm``, ``miyabi``, ``fugaku`` or ``local``.

    Raises:
        ValueError: For an unknown solver or system.
    """
    if solver not in SOLVER_SLUGS:
        raise ValueError(f"Unknown solver {solver!r}; use one of {', '.join(SOLVER_SLUGS)}.")
    if system not in SCRIPT_SUFFIXES:
        raise ValueError(f"Unknown system {system!r}; use one of {', '.join(SCRIPT_SUFFIXES)}.")
    return {
        "solver_block_name": f"sqd-{solver}-solver",
        "command_block_name": f"cmd-sqd-{solver}-solver",
        "execution_profile_block_name": f"exec-sqd-{solver}-{system}",
        "hpc_profile_block_name": f"hpc-{system}-sqd-{solver}",
        "script_filename": f"{solver}_solver{SCRIPT_SUFFIXES[system]}",
        "metrics_artifact_key": f"{system}-sqd-{solver}-metrics",
    }


def solver_ref(solver: str, solver_block_name: str) -> str:
    """The ``solver_block_ref`` of ``solver``'s block ``solver_block_name``."""
    return f"{SOLVER_SLUGS[solver]}/{solver_block_name}"
