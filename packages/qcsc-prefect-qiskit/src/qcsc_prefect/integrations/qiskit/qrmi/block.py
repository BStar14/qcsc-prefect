"""Prefect Block that stores QrmiRuntime settings and builds the runtime."""

from __future__ import annotations

import logging
import os
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal

from prefect.blocks.core import Block
from prefect.types import SecretDict
from pydantic import ConfigDict, Field, ValidationInfo, create_model, field_validator

from qiskit import QuantumCircuit

from .config_file import API_KEY_SETTING, config_path, crn_region, read_config
from .runtime import DEFAULT_TRANSPILER, PreparedCircuit, QrmiRuntime

DEFAULT_ENDPOINT = "https://quantum.cloud.ibm.com/api/v1"
DEFAULT_IAM_ENDPOINT = "https://iam.cloud.ibm.com"


def regional_endpoint(crn: str) -> str:
    """The IBM Quantum API endpoint of the CRN's region, as qiskit-ibm-runtime builds it.

    ``us-east`` uses ``quantum.cloud.ibm.com``; another region, for example ``eu-de``,
    uses ``eu-de.quantum.cloud.ibm.com``.
    """
    region = crn_region(crn)
    if not region or region == "us-east":
        return DEFAULT_ENDPOINT
    return f"https://{region}.quantum.cloud.ibm.com/api/v1"


SESSION_MODES = ("batch", "dedicated")
_DEFAULTS = {
    "instance": "",
    "service_crn": "",
    "endpoint": DEFAULT_ENDPOINT,
    "iam_endpoint": DEFAULT_IAM_ENDPOINT,
    "session_mode": "batch",
}

LOGGER = logging.getLogger(__name__)


class BackendAccessError(ValueError):
    """The selected IBM Quantum instance cannot use the selected backend."""


class QrmiRuntimeBlock(Block):
    """Saved ``QrmiRuntime`` settings: the instance and backend, and where to sample.

    The instance, the backend (``resource``), the CRN and the endpoints are plain
    fields of the block form. The API key is not in the block: it stays in
    ``qrmi_config.json``, written by ``qcsc-qrmi login``.

    - ``scheduler="slurm"`` passes the CRN, the endpoints and the session mode to the
      job's environment, where the SPANK plugin reads them. SPANK reads the API key
      from the cluster's ``qrmi_config.json``.
    - ``scheduler="iqp"`` passes them, with the API key of the workflow host's
      ``qrmi_config.json`` (``$QRMI_CONFIG``, default ``~/.qiskit/qrmi_config.json``)
      or of ``QRMI_IBM_QCS_IAM_APIKEY``, to QRMI in the workflow process.

    ``qcsc-qrmi login`` fills ``instances`` and ``instance_backends`` and saves the
    block with dropdowns for IBM Quantum Instance and QPU Backend (see
    ``qrmi_runtime_block_class``). ``check_backend_access`` checks that the instance
    can use the backend.

    ``prepare`` is a Python callable and cannot be stored, so pass it to
    ``get_runtime``. ``load`` keeps Prefect's behavior and returns the block;
    ``await QrmiRuntimeBlock.aload_runtime(name)`` (or ``load_runtime`` from
    synchronous code) returns the ``QrmiRuntime``.
    """

    # Validation errors would otherwise echo the input, the resource_config secret included.
    model_config = ConfigDict(hide_input_in_errors=True)

    _block_type_name = "QRMI Runtime"
    _block_type_slug = "qrmi_runtime"

    scheduler: Literal["slurm", "iqp"] = Field(
        default="slurm",
        title="Scheduler",
        description="slurm: run in a Slurm job; iqp: run in the workflow process.",
    )
    instance: str = Field(
        default="",
        title="IBM Quantum Instance",
        description=(
            "Instance to run on, as 'name (plan, region)' from Instances. qcsc-qrmi login lists "
            "them; its CRN is used unless Service CRN is set."
        ),
    )
    resource: str = Field(
        title="QPU Backend",
        description=(
            "Backend to sample on, for example ibm_kingston. The instance must be able to "
            "use it. For Slurm it is also the job's --qpu and must be a resource of the "
            "cluster's qrmi_config.json (qcsc-qrmi login writes them)."
        ),
    )
    service_crn: str = Field(
        default="",
        title="Service CRN",
        description="Empty: the CRN of IBM Quantum Instance. Set it to use another CRN.",
    )
    instances: dict[str, str] = Field(
        default_factory=dict,
        title="Instances",
        description=(
            "IBM Quantum instances, 'name (plan, region)' -> CRN. qcsc-qrmi login fills it."
        ),
    )
    endpoint: str = Field(
        default=DEFAULT_ENDPOINT,
        title="IBM Quantum Endpoint",
        description=(
            "IBM Quantum API endpoint (QRMI_IBM_QCS_ENDPOINT). The default follows the "
            "instance's region: eu-de instances use https://eu-de.quantum.cloud.ibm.com/api/v1."
        ),
    )
    iam_endpoint: str = Field(
        default=DEFAULT_IAM_ENDPOINT,
        title="IAM Endpoint",
        description="IBM Cloud IAM endpoint (QRMI_IBM_QCS_IAM_ENDPOINT).",
    )
    session_mode: Literal["batch", "dedicated"] = Field(
        default="batch",
        title="Session Mode",
        description=(
            "Slurm only: QRMI session of the job. batch works with every plan, including "
            "Open; dedicated needs a plan with sessions."
        ),
    )
    instance_backends: dict[str, list[str]] = Field(
        default_factory=dict,
        title="Backend Access",
        description=(
            "Backends each instance can use, 'name (plan, region)' -> backends; checked before a "
            "run. qcsc-qrmi login and refresh fill it."
        ),
    )
    root_dir: str | None = Field(
        default=None,
        title="Root Directory",
        description="Slurm only: shared directory for per-job work directories.",
    )
    command_block_name: str | None = Field(default=None, title="Command Block Name")
    execution_profile_block_name: str | None = Field(
        default=None, title="Execution Profile Block Name"
    )
    hpc_profile_block_name: str | None = Field(default=None, title="HPC Profile Block Name")
    transpiler: str = Field(
        default=DEFAULT_TRANSPILER,
        title="Transpiler",
        description="Importable module:function(circuit, target, config).",
    )
    transpile_config: dict = Field(default_factory=dict, title="Transpile Config")
    watch_poll_interval: float = Field(default=2.0, gt=0, title="Poll Interval Seconds")
    timeout_seconds: float | None = Field(
        default=None, title="Timeout Seconds", description="Slurm only: monitoring timeout."
    )
    resource_config: SecretDict | None = Field(
        default=None,
        title="QRMI Resource Config",
        description=(
            "Legacy, IQP only: other QRMI settings without the '<resource>_' prefix. The "
            "fields above take precedence."
        ),
    )

    @field_validator(
        "instance", "resource", "service_crn", "endpoint", "iam_endpoint", mode="before"
    )
    @classmethod
    def _strip(cls, value: Any, info: ValidationInfo) -> Any:
        """Strip pasted whitespace; a block saved before these fields keeps the defaults."""
        if value is None:
            return _DEFAULTS.get(info.field_name, value)
        if isinstance(value, str):
            value = value.strip()
            # An instance label is 'name (plan, region)'; the other fields are single values.
            if info.field_name != "instance" and any(char.isspace() for char in value):
                raise ValueError("must be one value without spaces")
            if not value and info.field_name == "resource":
                raise ValueError("QPU Backend is required")
            return value or _DEFAULTS.get(info.field_name, value)
        return value

    @field_validator("session_mode", mode="before")
    @classmethod
    def _default_session_mode(cls, value: Any) -> Any:
        return value or "batch"

    @field_validator("resource_config")
    @classmethod
    def _no_resource_config_for_slurm(cls, value: Any, info: ValidationInfo) -> Any:
        if value and info.data.get("scheduler") == "slurm":
            raise ValueError(
                "QRMI Resource Config is only used by scheduler 'iqp'; for Slurm set the "
                "instance in the block and the API key with qcsc-qrmi login."
            )
        return value

    def settings(self) -> dict[str, Any]:
        """The block's field values, without ``resource_config`` or Prefect's own keys."""
        return {
            name: getattr(self, name)
            for name in type(self).model_fields
            if name != "resource_config"
        }

    def instance_label(self) -> str:
        """The label in Instances of IBM Quantum Instance, which may be the bare name."""
        if not self.instance or self.instance in self.instances:
            return self.instance
        matches = [label for label in self.instances if label.split(" (")[0] == self.instance]
        return matches[0] if len(matches) == 1 else self.instance

    def crn(self) -> str:
        """Service CRN, else the CRN of IBM Quantum Instance, else ``""``."""
        return self.service_crn or self.instances.get(self.instance_label(), "")

    def api_endpoint(self) -> str:
        """IBM Quantum Endpoint, or the regional one of the CRN while it is the default."""
        if self.endpoint != DEFAULT_ENDPOINT:
            return self.endpoint
        return regional_endpoint(self.crn())

    def qrmi_settings(self) -> dict[str, str]:
        """The QRMI settings of the fields, without the API key."""
        settings = {
            "QRMI_IBM_QCS_ENDPOINT": self.api_endpoint(),
            "QRMI_IBM_QCS_IAM_ENDPOINT": self.iam_endpoint,
            "QRMI_IBM_QCS_SESSION_MODE": self.session_mode,
        }
        if self.crn():
            settings["QRMI_IBM_QCS_SERVICE_CRN"] = self.crn()
        return settings

    def api_key(self) -> str | None:
        """The API key of this host: ``qrmi_config.json``, the environment, or old blocks.

        Read only for IQP and the live backend check; never stored in the block.
        """
        try:
            key = read_config().api_key
        except OSError as exc:
            LOGGER.warning("Cannot read %s: %s", config_path(), exc)
            key = None
        if not key:
            key = os.environ.get(API_KEY_SETTING)
        if not key and self.resource_config:
            key = self.resource_config.get_secret_value().get(API_KEY_SETTING)
        return key or None

    def check_backend_access(self, *, live: bool = True, logger: Any = LOGGER) -> None:
        """Check that IBM Quantum Instance can use QPU Backend, before any job starts.

        Offline, the backend must be in the instance's Backend Access list. With
        ``live`` and an API key on this host, IBM Quantum is asked too, through
        qiskit-ibm-runtime, and its answer wins over a stale list. A failed live
        request (for example no network) is logged, and the offline answer stands.

        Raises:
            BackendAccessError: Without an instance or CRN, or if the instance cannot
                use the backend.
        """
        label = self.instance_label()
        crn = self.crn()
        if not crn:
            raise BackendAccessError(
                "The QRMI Runtime block names no IBM Quantum Instance or Service CRN. Run "
                "qcsc-qrmi login, or choose an IBM Quantum Instance in the block."
            )
        listed = self.instance_backends.get(label)
        # qiskit-ibm-runtime reaches the instance's regional endpoint from its CRN.
        if live and self.api_endpoint() == regional_endpoint(crn):
            key = self.api_key()
            if key:
                try:
                    available = _live_backend_names(key, crn, self.resource)
                except Exception as exc:  # noqa: BLE001 - any failure falls back to offline
                    logger.warning(
                        "Could not ask IBM Quantum whether %s can use %s (%s); using the "
                        "Backend Access list.",
                        label or "the Service CRN",
                        self.resource,
                        type(exc).__name__,
                    )
                else:
                    if self.resource not in available:
                        raise BackendAccessError(self._access_message(label))
                    return
        if listed is not None and self.resource not in listed:
            raise BackendAccessError(self._access_message(label))

    def _access_message(self, label: str) -> str:
        others = sorted(
            name for name, names in self.instance_backends.items() if self.resource in names
        )
        where = f"; it is available to: {', '.join(others)}" if others else ""
        return (
            f"{self.resource} is not available to {label or 'the Service CRN'}{where}. Choose "
            "another IBM Quantum Instance or QPU Backend in the QRMI Runtime block, or run "
            "qcsc-qrmi refresh."
        )

    def get_runtime(
        self,
        prepare: Callable[[QuantumCircuit], PreparedCircuit] | None = None,
    ) -> QrmiRuntime:
        """Build a validated ``QrmiRuntime`` from the saved settings."""
        config = None
        submit_settings = None
        if self.scheduler == "iqp":
            extra = self.resource_config.get_secret_value() if self.resource_config else {}
            # A legacy block's resource_config keeps its endpoints and key; a changed
            # field still wins.
            config = {**self.qrmi_settings(), **extra}
            if self.endpoint != DEFAULT_ENDPOINT or self.instance or self.service_crn:
                config["QRMI_IBM_QCS_ENDPOINT"] = self.api_endpoint()
            if self.iam_endpoint != DEFAULT_IAM_ENDPOINT:
                config["QRMI_IBM_QCS_IAM_ENDPOINT"] = self.iam_endpoint
            if self.crn():
                config["QRMI_IBM_QCS_SERVICE_CRN"] = self.crn()
            # qrmi_config.json, then the environment, then a legacy resource_config.
            key = self.api_key()
            if key:
                config[API_KEY_SETTING] = key
            # QRMI's IQP path submits jobs without a session.
            config.pop("QRMI_IBM_QCS_SESSION_MODE", None)
            if not config.get(API_KEY_SETTING):
                raise ValueError(
                    f"scheduler 'iqp' needs an API key: run qcsc-qrmi login (it writes "
                    f"{config_path()}), or set {API_KEY_SETTING}."
                )
        else:
            submit_settings = self.qrmi_settings()
        return QrmiRuntime(
            root_dir=Path(self.root_dir) if self.root_dir else None,
            command_block_name=self.command_block_name,
            execution_profile_block_name=self.execution_profile_block_name,
            hpc_profile_block_name=self.hpc_profile_block_name,
            resource=self.resource,
            prepare=prepare,
            transpiler=self.transpiler,
            transpile_config=dict(self.transpile_config),
            watch_poll_interval=self.watch_poll_interval,
            timeout_seconds=self.timeout_seconds,
            scheduler=self.scheduler,
            resource_config=config,
            job_settings=submit_settings,
        )

    @classmethod
    async def aload_runtime(
        cls, name: str, prepare: Callable[[QuantumCircuit], PreparedCircuit] | None = None
    ) -> QrmiRuntime:
        """Load the saved block ``name`` and build its ``QrmiRuntime``."""
        block = await cls.aload(name)
        return block.get_runtime(prepare)

    @classmethod
    def load_runtime(
        cls, name: str, prepare: Callable[[QuantumCircuit], PreparedCircuit] | None = None
    ) -> QrmiRuntime:
        """Synchronous ``aload_runtime`` for tasks that are not coroutines."""
        block = cls.load(name, _sync=True)
        return block.get_runtime(prepare)


def _live_backend_names(api_key: str, crn: str, backend: str) -> set[str]:
    """``{backend}`` if instance ``crn`` can use it, else an empty set (IBM Quantum)."""
    from qiskit_ibm_runtime import QiskitRuntimeService
    from qiskit_ibm_runtime.exceptions import QiskitBackendNotFoundError

    from .cli import quiet_ibm_runtime

    with quiet_ibm_runtime():
        service = QiskitRuntimeService(channel="ibm_quantum_platform", token=api_key, instance=crn)
        try:
            return {item.name for item in service.backends(name=backend, instance=crn)}
        except QiskitBackendNotFoundError:
            return set()


def qrmi_runtime_block_class(
    instances: Sequence[str], backends: Sequence[str]
) -> type[QrmiRuntimeBlock]:
    """``QrmiRuntimeBlock`` whose IBM Quantum Instance and QPU Backend are dropdowns.

    Save a block with this class so that its form offers ``instances`` (labels) and
    ``backends``. The class has the same block type and fields; only the schema of
    the two fields lists the choices. Load blocks with ``QrmiRuntimeBlock``, which
    accepts any value, so a block saved with other choices always loads.
    """
    instances = list(dict.fromkeys(instances))
    backends = sorted(set(backends))
    fields: dict[str, Any] = {}
    base_fields = QrmiRuntimeBlock.model_fields
    if instances:
        info = base_fields["instance"]
        fields["instance"] = (
            Literal[tuple(instances)],
            Field(default=instances[0], title=info.title, description=info.description),
        )
    if backends:
        info = base_fields["resource"]
        fields["resource"] = (
            Literal[tuple(backends)],
            Field(default=backends[0], title=info.title, description=info.description),
        )
    if not fields:
        return QrmiRuntimeBlock
    registry = _block_registry()
    registered = registry.get(QrmiRuntimeBlock.get_block_type_slug())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        subclass = create_model("QrmiRuntimeBlock", __base__=QrmiRuntimeBlock, **fields)
    # Loading by block type must keep returning the base class, which accepts any value.
    if registered is not None:
        registry[QrmiRuntimeBlock.get_block_type_slug()] = registered
    return subclass


def _block_registry() -> dict:
    from prefect.utilities.dispatch import get_registry_for_type

    return get_registry_for_type(Block) or {}
