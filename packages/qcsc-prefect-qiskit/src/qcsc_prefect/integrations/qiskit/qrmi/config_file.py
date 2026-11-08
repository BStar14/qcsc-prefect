"""Read and write ``qrmi_config.json``: the API key once, the instances, the backends.

``qcsc-qrmi login`` writes the file. Its layout::

    {
      "environment": {"QRMI_IBM_QCS_IAM_APIKEY": "..."},
      "default_instance": "main-instance",
      "instances": [
        {"name": "main-instance", "plan": "premium", "crn": "crn:v1:...",
         "backends": ["ibm_fez", "ibm_kingston"]}
      ],
      "resources": [
        {"name": "ibm_fez", "type": "ibm-quantum-compute-service",
         "environment": {"QRMI_IBM_QCS_IAM_APIKEY": "..."}}
      ]
    }

- ``environment`` holds the API key, once. The workflow reads it for IQP and for the
  backend check.
- ``instances`` lists each IBM Quantum instance with its CRN and backends. Only this
  package reads it; QRMI and the Slurm SPANK plugin ignore keys other than
  ``resources``.
- ``resources`` is what QRMI and SPANK read: one entry per backend, whichever
  instances can use it. SPANK reads only an entry's own ``environment``, so each IBM
  entry carries a copy of the key, which ``write_config`` keeps equal to the one above.
  The CRN, the endpoints and the session mode come from the ``qrmi-runtime`` block
  for every job, so the entries hold nothing else. Entries of other resource types
  are kept as they are.
"""

from __future__ import annotations

import json
import os
import stat
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

API_KEY_SETTING = "QRMI_IBM_QCS_IAM_APIKEY"
IBM_RESOURCE_TYPE = "ibm-quantum-compute-service"
CONFIG_ENV = "QRMI_CONFIG"
DEFAULT_CONFIG_PATH = Path("~/.qiskit/qrmi_config.json")


def config_path(path: str | os.PathLike | None = None) -> Path:
    """``path``, else ``$QRMI_CONFIG``, else ``~/.qiskit/qrmi_config.json``."""
    chosen = path or os.environ.get(CONFIG_ENV) or DEFAULT_CONFIG_PATH
    return Path(chosen).expanduser()


@dataclass
class Instance:
    """One IBM Quantum instance: its name, plan, CRN and the backends it can use."""

    name: str
    crn: str
    plan: str = ""
    backends: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        """The name shown in the block form, ``name (plan)``."""
        return f"{self.name} ({self.plan})" if self.plan else self.name

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "plan": self.plan,
            "crn": self.crn,
            "backends": sorted(set(self.backends)),
        }


@dataclass
class QrmiConfig:
    """The parts of ``qrmi_config.json`` this package manages, plus the rest as read."""

    api_key: str | None = field(default=None, repr=False)
    instances: list[Instance] = field(default_factory=list)
    default_instance: str | None = None
    # Entries of other resource types, and other top-level keys, kept unchanged.
    other_resources: list[dict[str, Any]] = field(default_factory=list)
    other_keys: dict[str, Any] = field(default_factory=dict)

    @property
    def backends(self) -> list[str]:
        """Every backend of every instance, once each, sorted."""
        return sorted({name for instance in self.instances for name in instance.backends})

    def instance(self, name_or_label: str) -> Instance | None:
        """The instance named ``name_or_label`` (its name or its ``name (plan)`` label)."""
        for instance in self.instances:
            if name_or_label in (instance.name, instance.label):
                return instance
        return None


def read_config(path: str | os.PathLike | None = None) -> QrmiConfig:
    """Read ``qrmi_config.json``; a missing file gives an empty config.

    A file written by hand, without ``instances``, still gives its IBM entries'
    API key, so the workflow can use it.
    """
    path = config_path(path)
    if not path.exists():
        return QrmiConfig()
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object")
    environment = data.get("environment") or {}
    resources = data.get("resources") or []
    api_key = environment.get(API_KEY_SETTING)
    other_resources = []
    for entry in resources:
        if entry.get("type") == IBM_RESOURCE_TYPE:
            api_key = api_key or (entry.get("environment") or {}).get(API_KEY_SETTING)
        else:
            other_resources.append(entry)
    instances = [
        Instance(
            name=str(item["name"]),
            crn=str(item["crn"]),
            plan=str(item.get("plan") or ""),
            backends=[str(name) for name in item.get("backends") or []],
        )
        for item in data.get("instances") or []
    ]
    managed = {"environment", "instances", "default_instance", "resources"}
    return QrmiConfig(
        api_key=api_key or None,
        instances=instances,
        default_instance=data.get("default_instance"),
        other_resources=other_resources,
        other_keys={key: value for key, value in data.items() if key not in managed},
    )


def config_document(config: QrmiConfig) -> dict[str, Any]:
    """The JSON document of ``config``, with the key copied into every IBM entry."""
    if not config.api_key:
        raise ValueError("The QRMI config needs an API key")
    entry_environment = {API_KEY_SETTING: config.api_key}
    resources = [
        {"name": name, "type": IBM_RESOURCE_TYPE, "environment": dict(entry_environment)}
        for name in config.backends
    ]
    other_names = {entry.get("name") for entry in config.other_resources}
    clashes = sorted(other_names & set(config.backends))
    if clashes:
        raise ValueError(
            f"Backends {', '.join(clashes)} are also resources of another type in the file"
        )
    return {
        **config.other_keys,
        "environment": dict(entry_environment),
        "default_instance": config.default_instance,
        "instances": [instance.to_json() for instance in config.instances],
        "resources": [*resources, *config.other_resources],
    }


def write_config(config: QrmiConfig, path: str | os.PathLike | None = None) -> Path:
    """Write ``config`` to ``path`` with mode 600, in place.

    The file is rewritten in place rather than replaced, so a container that mounts
    this single file sees the new content.
    """
    path = config_path(path)
    text = json.dumps(config_document(config), indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT, 0o600)
    try:
        os.fchmod(fd, stat.S_IRUSR | stat.S_IWUSR)
        os.ftruncate(fd, 0)
        os.write(fd, text.encode("utf-8"))
    finally:
        os.close(fd)
    return path


def merge_instances(
    config: QrmiConfig, instances: Iterable[Instance], *, replace: bool = True
) -> QrmiConfig:
    """``config`` with ``instances`` added, or replacing the instances of the same name."""
    found = {instance.name: instance for instance in instances}
    kept = [] if replace else [i for i in config.instances if i.name not in found]
    return QrmiConfig(
        api_key=config.api_key,
        instances=[*kept, *found.values()],
        default_instance=config.default_instance,
        other_resources=list(config.other_resources),
        other_keys=dict(config.other_keys),
    )


def instance_backends(instances: Iterable[Instance]) -> dict[str, list[str]]:
    """``label -> backends``, as the ``qrmi-runtime`` block stores them."""
    return {instance.label: sorted(set(instance.backends)) for instance in instances}


def instance_crns(instances: Iterable[Instance]) -> dict[str, str]:
    """``label -> CRN``, as the ``qrmi-runtime`` block stores them."""
    return {instance.label: instance.crn for instance in instances}


def describe(config: QrmiConfig) -> Mapping[str, Any]:
    """A summary without the key: instances with backend counts, and the backends."""
    return {
        "api_key": "set" if config.api_key else "unset",
        "default_instance": config.default_instance,
        "instances": [
            {"label": instance.label, "backends": len(instance.backends)}
            for instance in config.instances
        ],
        "backends": config.backends,
    }
