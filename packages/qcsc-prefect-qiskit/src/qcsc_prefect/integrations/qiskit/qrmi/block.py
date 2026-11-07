"""Prefect Block that stores QrmiRuntime settings and builds the runtime."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Literal

from prefect.blocks.core import Block
from prefect.types import SecretDict
from pydantic import Field, SecretStr, model_validator

from qiskit import QuantumCircuit

from .runtime import DEFAULT_TRANSPILER, PreparedCircuit, QrmiRuntime

# QRMI settings that the block's plain fields set, by field name.
QRMI_SETTING_FIELDS = {
    "service_crn": "QRMI_IBM_QCS_SERVICE_CRN",
    "endpoint": "QRMI_IBM_QCS_ENDPOINT",
    "iam_endpoint": "QRMI_IBM_QCS_IAM_ENDPOINT",
    "session_mode": "QRMI_IBM_QCS_SESSION_MODE",
}
API_KEY_SETTING = "QRMI_IBM_QCS_IAM_APIKEY"
DEFAULT_ENDPOINT = "https://quantum.cloud.ibm.com/api/v1"
DEFAULT_IAM_ENDPOINT = "https://iam.cloud.ibm.com"


class QrmiRuntimeBlock(Block):
    """Saved ``QrmiRuntime`` settings.

    The backend (``resource``), the service CRN and the endpoints are plain fields,
    visible in the block form. The IBM Cloud API key is the only secret.

    - ``scheduler="iqp"`` passes all of them to QRMI in the workflow process.
    - ``scheduler="slurm"`` passes the plain ones to the job's environment, where the
      SPANK plugin reads them over the cluster's ``qrmi_config.json``. The API key
      stays in that file on the compute nodes: a per-job key could only reach SPANK
      through the job environment, which Slurm stores with the job, so a Slurm
      block rejects ``api_key``.

    Stores the JSON-serializable settings; ``resource_config`` is a secret.
    ``prepare`` is a Python callable and cannot be stored, so pass it to
    ``get_runtime``. ``load`` keeps Prefect's behavior and returns the block;
    ``await QrmiRuntimeBlock.aload_runtime(name)`` (or ``load_runtime`` from
    synchronous code) returns the ``QrmiRuntime``.
    """

    _block_type_name = "QRMI Runtime"
    _block_type_slug = "qrmi_runtime"

    scheduler: Literal["slurm", "iqp"] = Field(
        default="slurm",
        title="Scheduler",
        description="slurm: run in a Slurm job; iqp: run in the workflow process.",
    )
    resource: str = Field(
        title="QRMI Resource",
        description=(
            "QRMI resource ID: the backend name, for example ibm_kingston. For Slurm it is "
            "also the --qpu resource and must be listed in the cluster's qrmi_config.json."
        ),
    )
    service_crn: str | None = Field(
        default=None,
        title="Service CRN",
        description=(
            "IBM Quantum instance CRN. Empty: for Slurm, the CRN of the cluster's "
            "qrmi_config.json; required for IQP unless QRMI Resource Config has it."
        ),
    )
    endpoint: str | None = Field(
        default=None,
        title="IBM Quantum Endpoint",
        description=(
            "IBM Quantum API endpoint (QRMI_IBM_QCS_ENDPOINT). Empty: "
            f"{DEFAULT_ENDPOINT} for IQP, the cluster's qrmi_config.json for Slurm."
        ),
    )
    iam_endpoint: str | None = Field(
        default=None,
        title="IAM Endpoint",
        description=(
            "IBM Cloud IAM endpoint (QRMI_IBM_QCS_IAM_ENDPOINT). Empty: "
            f"{DEFAULT_IAM_ENDPOINT} for IQP, the cluster's qrmi_config.json for Slurm."
        ),
    )
    session_mode: Literal["batch"] | None = Field(
        default=None,
        title="Session Mode",
        description=(
            "Slurm only: QRMI session mode for the job. batch works with every plan, "
            "including Open. Empty: the cluster's qrmi_config.json."
        ),
    )
    api_key: SecretStr | None = Field(
        default=None,
        title="IBM Cloud API Key",
        description=(
            "IQP only; kept secret. For Slurm leave it empty: the compute nodes read the "
            "API key from the cluster's qrmi_config.json."
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
            "Advanced, IQP only: other QRMI settings without the '<resource>_' prefix. "
            "The fields above take precedence; set the CRN, endpoints and API key there."
        ),
    )

    @model_validator(mode="after")
    def _no_api_key_for_slurm(self) -> "QrmiRuntimeBlock":
        if self.scheduler == "slurm" and self.api_key is not None:
            raise ValueError(
                "Slurm reads the API key from qrmi_config.json on the compute nodes; leave "
                "IBM Cloud API Key empty for scheduler 'slurm'."
            )
        return self

    def qrmi_settings(self) -> dict[str, str]:
        """The plain QRMI settings of the fields, without the API key."""
        settings = {}
        for name, key in QRMI_SETTING_FIELDS.items():
            value = getattr(self, name)
            if value:
                settings[key] = str(value)
        return settings

    def get_runtime(
        self, prepare: Callable[[QuantumCircuit], PreparedCircuit] | None = None
    ) -> QrmiRuntime:
        """Build a validated ``QrmiRuntime`` from the saved settings."""
        config = None
        submit_settings = None
        if self.scheduler == "iqp":
            extra = self.resource_config.get_secret_value() if self.resource_config else {}
            config = {**extra, **self.qrmi_settings()}
            if self.api_key is not None:
                config[API_KEY_SETTING] = self.api_key.get_secret_value()
            # QRMI's IQP path submits jobs without a session.
            config.pop("QRMI_IBM_QCS_SESSION_MODE", None)
            if config:
                # A block from before these fields keeps its resource_config endpoints.
                config.setdefault("QRMI_IBM_QCS_ENDPOINT", DEFAULT_ENDPOINT)
                config.setdefault("QRMI_IBM_QCS_IAM_ENDPOINT", DEFAULT_IAM_ENDPOINT)
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
