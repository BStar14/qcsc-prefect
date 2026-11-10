"""Build and save the ``QrmiRuntimeBlock`` of a create-blocks ``[qrmi]`` or ``[runtime]`` table.

Used by the ``create_blocks`` scripts of every solver. The SQD flow sets the
transpiler hook and its configuration on each run, so the table must not.

``qcsc-qrmi login`` saves the IBM Quantum instances and their backends in the block,
and the API key in ``qrmi_config.json``. ``sqd-create-blocks`` keeps them, and
changes the instance and the backend only when the table sets ``instance`` or
``resource``.
"""

from __future__ import annotations

from typing import Any

from sqd_common.flow_params import DEFAULT_QRMI_BLOCK_NAME

# QRMI settings that are secrets: never read from the config file.
SECRET_RESOURCE_KEYS = ("QRMI_IBM_QCS_IAM_APIKEY",)
FLOW_OWNED_KEYS = frozenset(
    {"transpiler", "transpile_config", "resource_config", "instances", "instance_backends"}
)
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

    ``instance``, ``resource``, ``service_crn``, ``endpoint`` and ``iam_endpoint``
    are plain block fields, set in the table or, as their QRMI names, in
    ``[runtime.qrmi]``. The API key is never read from the file: ``qcsc-qrmi login``
    stores it in ``qrmi_config.json``. Other ``[runtime.qrmi]`` keys stay in
    ``resource_config`` (IQP only).

    Raises:
        RuntimeError: If the table sets a field the flow owns or writes a secret.
    """
    if not isinstance(runtime, dict):
        raise RuntimeError("'runtime' must be a table.")
    settings = {"scheduler": "iqp", **runtime}
    qrmi = {str(key): str(value) for key, value in settings.pop("qrmi", {}).items()}
    secrets = sorted(
        key
        for key in {*qrmi, *settings}
        if key == "api_key" or any(s in key for s in SECRET_RESOURCE_KEYS)
    )
    if secrets:
        raise RuntimeError(
            f"Do not write {', '.join(secrets)} in the config file; qcsc-qrmi login stores "
            "the API key in qrmi_config.json"
        )
    unsupported = sorted(settings.keys() & FLOW_OWNED_KEYS)
    if unsupported:
        raise RuntimeError(
            f"[runtime] must not set {', '.join(unsupported)}; the SQD flow sets the "
            "transpiler, and qcsc-qrmi login sets the instances"
        )
    for key, name in PLAIN_QRMI_FIELDS.items():
        if key not in qrmi:
            continue
        value = qrmi.pop(key)
        if settings.get(name) not in (None, value):
            raise RuntimeError(f"[runtime] {name} and [runtime.qrmi] {key} differ; set only {name}")
        settings[name] = value
    if qrmi:
        if settings["scheduler"] != "iqp":
            raise RuntimeError(
                f"[runtime.qrmi] keys {', '.join(sorted(qrmi))} are only used by "
                "scheduler='iqp'; set the CRN and endpoints as [runtime] keys"
            )
        settings["resource_config"] = qrmi
    return settings


def _saved_block(name: str):
    from qcsc_prefect.integrations.qiskit.qrmi import QrmiRuntimeBlock

    try:
        return QrmiRuntimeBlock.load(name, _sync=True)
    except ValueError:
        return None


def merged_runtime_settings(
    settings: dict[str, Any], saved: Any | None
) -> tuple[dict[str, Any], list[str]]:
    """``settings`` over the saved block's, keeping its login choices; and the changes.

    The instances, their backends and the CRN override come from the saved block.
    The instance and the backend come from ``settings`` only when it sets them.
    """
    if saved is None:
        return dict(settings), []
    merged = saved.settings()
    changes = []
    for key, value in settings.items():
        if value is None:
            continue
        if key in ("instance", "resource") and merged.get(key) and merged[key] != value:
            title = "QPU Backend" if key == "resource" else "IBM Quantum Instance"
            changes.append(f"{title}: {merged[key]} -> {value}")
        merged[key] = value
    if "resource_config" not in settings and saved.resource_config is not None:
        if merged.get("scheduler") == "iqp":
            merged["resource_config"] = saved.resource_config.get_secret_value()
    return merged, changes


def build_runtime_block(settings: dict[str, Any]):
    """A validated ``QrmiRuntimeBlock`` (not saved), with dropdowns of its saved lists.

    Raises:
        RuntimeError: For an unknown key, a block without a QPU Backend, or an
            instance that is not in the saved Instances.
    """
    from qcsc_prefect.integrations.qiskit.qrmi import QrmiRuntimeBlock
    from qcsc_prefect.integrations.qiskit.qrmi.block import qrmi_runtime_block_class

    # The block keeps unknown keys without using them, so a misspelled field would be lost.
    unknown = sorted(set(settings) - set(QrmiRuntimeBlock.model_fields))
    if unknown:
        raise RuntimeError(f"Unknown [runtime] settings: {', '.join(unknown)}")
    if not str(settings.get("resource") or "").strip():
        raise RuntimeError(
            "Missing setting: the QPU backend. Run qcsc-qrmi login, or set resource in "
            "[qrmi] or [runtime]."
        )
    try:
        block = QrmiRuntimeBlock(**settings)
    except ValueError as exc:
        raise RuntimeError(f"Invalid QRMI Runtime block settings: {exc}") from exc
    labels = list(block.instances)
    if block.instance and labels and block.instance_label() not in labels:
        raise RuntimeError(
            f"Instance {block.instance!r} is not one of {', '.join(labels)}; run "
            "qcsc-qrmi refresh if it is new."
        )
    backends = sorted({name for names in block.instance_backends.values() for name in names})
    # A value outside the saved lists keeps a free-text field; the backend check reports it.
    block_class = qrmi_runtime_block_class(
        labels if block.instance_label() in labels else [],
        backends if block.resource in backends else [],
    )
    data = block.settings()
    data["instance"] = block.instance_label()
    if block.resource_config is not None:
        data["resource_config"] = block.resource_config.get_secret_value()
    block = block_class(**data)
    if block.scheduler == "slurm":
        block.get_runtime()  # Validate the Slurm settings before saving.
    return block


def save_runtime_block(settings: dict[str, Any], name: str) -> list[str]:
    """Save ``settings`` over block ``name``, keeping its login choices; return the changes."""
    merged, changes = merged_runtime_settings(settings, _saved_block(name))
    build_runtime_block(merged).save(name, overwrite=True)
    return changes


def describe(settings: dict[str, Any]) -> dict[str, Any]:
    """``settings`` for printing, with secrets masked."""
    shown = dict(settings)
    if shown.get("resource_config"):
        shown["resource_config"] = {
            key: "***" if key in SECRET_RESOURCE_KEYS else value
            for key, value in shown["resource_config"].items()
        }
    return shown
