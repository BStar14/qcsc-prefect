"""Build the ``QrmiRuntimeBlock`` of a create-blocks ``[runtime]`` config table.

Used by the ``create_blocks`` scripts of every solver. The SQD flow sets the
transpiler hook and its configuration on each run, so the table must not.
"""

from __future__ import annotations

import os
from typing import Any

from sqd_common.flow_params import DEFAULT_QRMI_BLOCK_NAME

# QRMI settings that are secrets: read from the environment, never from the config file.
SECRET_RESOURCE_KEYS = ("QRMI_IBM_QCS_IAM_APIKEY",)
FLOW_OWNED_KEYS = frozenset({"transpiler", "transpile_config", "resource_config", "api_key"})
# Plain QrmiRuntimeBlock fields a config may set, by their QRMI setting in [runtime.qrmi].
PLAIN_QRMI_FIELDS = {
    "QRMI_IBM_QCS_SERVICE_CRN": "service_crn",
    "QRMI_IBM_QCS_ENDPOINT": "endpoint",
    "QRMI_IBM_QCS_IAM_ENDPOINT": "iam_endpoint",
}


def runtime_block_name(cli_value: str | None, config: dict[str, Any]) -> str:
    """The block name: CLI, then ``runtime_block_name`` of the config, then the default."""
    for value in (cli_value, config.get("runtime_block_name")):
        if value is not None and str(value).strip():
            return str(value).strip()
    return DEFAULT_QRMI_BLOCK_NAME


def runtime_block_settings(runtime: dict[str, Any]) -> dict[str, Any]:
    """QrmiRuntimeBlock fields of a ``[runtime]`` table (scheduler defaults to ``iqp``).

    ``service_crn``, ``endpoint`` and ``iam_endpoint`` are plain block fields, set
    in the table or, as their QRMI names, in ``[runtime.qrmi]``. For IQP the API key
    comes from the ``QRMI_IBM_QCS_IAM_APIKEY`` environment variable into the block's
    secret ``api_key``; it is never read from the file. Other ``[runtime.qrmi]`` keys
    stay in ``resource_config`` (IQP only).

    Raises:
        RuntimeError: If the table sets a field the flow owns, writes a secret, or
            misses a required setting.
    """
    if not isinstance(runtime, dict):
        raise RuntimeError("'runtime' must be a table.")
    settings = {"scheduler": "iqp", **runtime}
    qrmi = {str(key): str(value) for key, value in settings.pop("qrmi", {}).items()}
    unsupported = sorted(settings.keys() & FLOW_OWNED_KEYS)
    if unsupported:
        raise RuntimeError(
            f"[runtime] must not set {', '.join(unsupported)}; the SQD flow sets the "
            "transpiler, and resource_config is built from [runtime.qrmi] and the environment"
        )
    if not str(settings.get("resource") or "").strip():
        raise RuntimeError("Missing required setting: runtime.resource")
    secrets = sorted(qrmi.keys() & set(SECRET_RESOURCE_KEYS))
    if secrets:
        raise RuntimeError(
            f"Do not write {', '.join(secrets)} in the config file; set it in the environment"
        )
    for key, name in PLAIN_QRMI_FIELDS.items():
        if key in qrmi:
            settings.setdefault(name, qrmi.pop(key))
    if settings["scheduler"] == "iqp":
        value = os.environ.get(SECRET_RESOURCE_KEYS[0])
        if not value:
            raise RuntimeError(
                f"Missing required setting: environment variable {SECRET_RESOURCE_KEYS[0]}"
            )
        settings["api_key"] = value
        if qrmi:
            settings["resource_config"] = qrmi
    elif qrmi:
        raise RuntimeError(
            f"[runtime.qrmi] keys {', '.join(sorted(qrmi))} are only used by scheduler='iqp'; "
            "Slurm reads the others from qrmi_config.json on the compute nodes"
        )
    return settings


def build_runtime_block(settings: dict[str, Any]):
    """A validated ``QrmiRuntimeBlock`` (not saved)."""
    from qcsc_prefect.integrations.qiskit.qrmi import QrmiRuntimeBlock

    block = QrmiRuntimeBlock(**settings)
    block.get_runtime()  # Validate the settings before saving.
    return block


def describe(settings: dict[str, Any]) -> dict[str, Any]:
    """``settings`` with the API key masked, for printing."""
    shown = dict(settings)
    if "api_key" in shown:
        shown["api_key"] = "set" if shown["api_key"] else "unset"
    if shown.get("resource_config"):
        shown["resource_config"] = {
            key: "***" if key in SECRET_RESOURCE_KEYS else value
            for key, value in shown["resource_config"].items()
        }
    return shown
