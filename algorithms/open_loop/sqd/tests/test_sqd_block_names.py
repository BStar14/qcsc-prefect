"""Default block names of the SQD workflow (``sqd_common.block_names``)."""

from __future__ import annotations

import pytest

pytest.importorskip("prefect")

from sqd_common import block_names  # noqa: E402


@pytest.mark.parametrize(
    "solver, system, expected",
    [
        (
            "fulqrum",
            "slurm",
            {
                "solver_block_name": "sqd-fulqrum-solver",
                "command_block_name": "cmd-sqd-fulqrum-solver",
                "execution_profile_block_name": "exec-sqd-fulqrum-slurm",
                "hpc_profile_block_name": "hpc-slurm-sqd-fulqrum",
                "script_filename": "fulqrum_solver.slurm",
                "metrics_artifact_key": "slurm-sqd-fulqrum-metrics",
            },
        ),
        (
            "dice",
            "fugaku",
            {
                "solver_block_name": "sqd-dice-solver",
                "command_block_name": "cmd-sqd-dice-solver",
                "execution_profile_block_name": "exec-sqd-dice-fugaku",
                "hpc_profile_block_name": "hpc-fugaku-sqd-dice",
                "script_filename": "dice_solver.pjm",
                "metrics_artifact_key": "fugaku-sqd-dice-metrics",
            },
        ),
    ],
)
def test_names_follow_the_solver_and_system(solver, system, expected):
    assert block_names.solver_block_names(solver, system) == expected


def test_shared_blocks_name_no_solver():
    names = [*block_names.QRMI_BLOCK_NAMES.values(), block_names.RUNTIME_BLOCK_NAME]
    names.append(block_names.ADVANCED_BLOCK_NAME)
    assert names == [
        "cmd-sqd-qrmi",
        "exec-sqd-qrmi",
        "hpc-slurm-sqd-qrmi",
        "qrmi-runtime",
        "sqd-advanced-settings",
    ]
    assert not any(solver in name for name in names for solver in block_names.SOLVER_SLUGS)


def test_solver_ref_uses_the_block_type_slug():
    assert block_names.solver_ref("dice", "sqd-dice-solver") == (
        "dice_shci_solver_job/sqd-dice-solver"
    )
    assert block_names.solver_ref("sbd", "x") == "sbd_solver_job/x"


@pytest.mark.parametrize("solver, system", [("cuda", "slurm"), ("fulqrum", "summit")])
def test_unknown_solver_or_system_is_rejected(solver, system):
    with pytest.raises(ValueError, match="Unknown"):
        block_names.solver_block_names(solver, system)
