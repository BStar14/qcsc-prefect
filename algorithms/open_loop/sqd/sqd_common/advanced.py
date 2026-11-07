"""Advanced SQD settings: kept in a Prefect block instead of the run form.

They change rarely, so the run form leaves them out. Each run loads the block
named by ``FlowParameters.advanced_block_name`` (``sqd-advanced-settings``), and
without that block it uses the defaults below. ``sqd-create-blocks`` saves the
block and records the solver block it created there, so changing the solver does
not change the run form.

The stages take :class:`AdvancedSettings`, a plain model with the block's values,
so their persisted results are keyed on the values, not on a block document.
"""

from __future__ import annotations

from typing import Any

from prefect.blocks.core import Block
from pydantic import BaseModel, Field

from sqd_common.flow_params import DEFAULT_ADVANCED_BLOCK_NAME as DEFAULT_ADVANCED_BLOCK_NAME
from sqd_common.flow_params import DEFAULT_SOLVER_BLOCK_REF

# The transpiler settings, in the order of build_pass_manager's keyword arguments.
TRANSPILER_KEYS = (
    "optimization_level",
    "sabre_layout_trials",
    "sabre_swap_trials",
    "sabre_max_iterations",
)


def _at(position: int) -> dict[str, int]:
    """Field order in the Prefect block form, which otherwise sorts fields alphabetically."""
    return {"position": position}


class AdvancedSettings(BaseModel):
    """CCSD restarts, reset mitigation, the transpiler, and the solver block of the runs."""

    solver_block_ref: str = Field(
        default=DEFAULT_SOLVER_BLOCK_REF,
        title="Solver Block",
        description=(
            "Solver block that diagonalizes each subspace, as '<block type slug>/<block "
            "name>': fulqrum_solver_job/<name>, dice_shci_solver_job/<name> (needs "
            "sqd[dice]) or sbd_solver_job/<name> (needs sqd[sbd]). sqd-create-blocks sets "
            "it to the solver block it created."
        ),
        json_schema_extra=_at(0),
    )
    ccsd_maxiter: int = Field(
        default=10,
        ge=1,
        title="CCSD Restarts",
        description=(
            "Number of CCSD kernel runs. Each run after the first restarts from the "
            "previous amplitudes, which removes the memory effect of the DIIS history."
        ),
        json_schema_extra=_at(1),
    )
    use_reset_mitigation: bool = Field(
        default=False,
        title="Reset Mitigation",
        description=(
            "Measure every qubit before the circuit and keep only the shots that started in "
            "the ground state. This lowers the shot count; how much depends on the quality "
            "of the hardware reset. A reload of earlier samples uses this setting too."
        ),
        json_schema_extra=_at(2),
    )
    optimization_level: int = Field(
        default=3,
        ge=0,
        le=3,
        title="Optimization Level",
        description="Optimization level of the preset pass manager.",
        json_schema_extra=_at(3),
    )
    sabre_layout_trials: int = Field(
        default=10000,
        ge=1,
        title="SABRE Layout Trials",
        description=(
            "The number of random initial layouts tested, selecting the one that minimizes "
            "SWAP gates."
        ),
        json_schema_extra=_at(4),
    )
    sabre_swap_trials: int = Field(
        default=10,
        ge=1,
        title="SABRE SWAP Trials",
        description=(
            "The number of routing trials for each layout, refining gate placement for "
            "better routing."
        ),
        json_schema_extra=_at(5),
    )
    sabre_max_iterations: int = Field(
        default=8,
        ge=1,
        title="SABRE Max Iteration",
        description=(
            "The number of forward-backward routing iterations to refine the layout and "
            "reduce routing costs."
        ),
        json_schema_extra=_at(6),
    )


class SQDAdvancedSettingsBlock(Block, AdvancedSettings):
    """Advanced SQD settings that every run of the ``sqd`` flow reads."""

    _block_type_name = "Advanced SQD Settings"
    _block_type_slug = "sqd_advanced_settings"

    def settings(self) -> AdvancedSettings:
        """The block's values as the plain model the stages take."""
        return AdvancedSettings.model_validate(
            {name: getattr(self, name) for name in AdvancedSettings.model_fields}
        )


async def load_advanced_settings(name: str, logger: Any) -> AdvancedSettings:
    """The settings of block ``name``, or the defaults if there is no such block."""
    try:
        block = await SQDAdvancedSettingsBlock.aload(name)
    except ValueError:
        logger.info(
            f"No Advanced SQD Settings block '{name}'; using the defaults. sqd-create-blocks "
            "saves one."
        )
        return AdvancedSettings()
    return block.settings()
