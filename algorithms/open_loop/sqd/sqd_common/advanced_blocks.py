"""Save the Advanced SQD Settings block from the ``[sqd_advanced]`` config table.

Used by the ``create_blocks`` script of every solver. The block records the solver
block the script created, so a run uses it without a solver in the run form.
"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from sqd_common.advanced import DEFAULT_ADVANCED_BLOCK_NAME, AdvancedSettings

# Keys of [sqd_advanced]; sqd-create-blocks sets solver_block_ref itself.
ADVANCED_KEYS = tuple(name for name in AdvancedSettings.model_fields if name != "solver_block_ref")


def advanced_settings(config: dict[str, Any], solver_block_ref: str) -> AdvancedSettings:
    """The Advanced SQD Settings of a config, for the solver block ``solver_block_ref``.

    Raises:
        RuntimeError: If ``[sqd_advanced]`` is not a table, has an unknown key or an
            invalid value.
    """
    table = config.get("sqd_advanced", {})
    if not isinstance(table, dict):
        raise RuntimeError("'sqd_advanced' must be a table.")
    unknown = sorted(set(table) - set(ADVANCED_KEYS))
    if unknown:
        raise RuntimeError(
            f"Unknown [sqd_advanced] settings: {', '.join(unknown)}; "
            f"use {', '.join(ADVANCED_KEYS)}."
        )
    try:
        return AdvancedSettings(**table, solver_block_ref=solver_block_ref)
    except ValidationError as exc:
        raise RuntimeError(f"Invalid [sqd_advanced] settings: {exc}") from exc


def advanced_block_name(config: dict[str, Any]) -> str:
    """The block name: ``advanced_block_name`` of the config, else the default."""
    name = str(config.get("advanced_block_name") or "").strip()
    return name or DEFAULT_ADVANCED_BLOCK_NAME


def save_advanced_block(settings: AdvancedSettings, name: str) -> None:
    """Save ``settings`` as the Advanced SQD Settings block ``name``."""
    from sqd_common.advanced import SQDAdvancedSettingsBlock

    SQDAdvancedSettingsBlock(**settings.model_dump()).save(name, overwrite=True)
