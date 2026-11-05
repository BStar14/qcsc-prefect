"""Prefect Block that stores QrmiRuntime settings and builds the runtime."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Literal

from prefect.blocks.core import Block
from prefect.types import SecretDict
from pydantic import Field

from qiskit import QuantumCircuit

from .runtime import DEFAULT_TRANSPILER, PreparedCircuit, QrmiRuntime


class QrmiRuntimeBlock(Block):
    """Saved ``QrmiRuntime`` settings.

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
    resource: str = Field(title="QRMI Resource", description="QRMI resource ID (backend name).")
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
            "IQP only: QRMI settings without the '<resource>_' prefix, for example "
            "QRMI_IBM_QCS_ENDPOINT, QRMI_IBM_QCS_IAM_ENDPOINT, QRMI_IBM_QCS_IAM_APIKEY "
            "and QRMI_IBM_QCS_SERVICE_CRN."
        ),
    )

    def get_runtime(
        self, prepare: Callable[[QuantumCircuit], PreparedCircuit] | None = None
    ) -> QrmiRuntime:
        """Build a validated ``QrmiRuntime`` from the saved settings."""
        config = self.resource_config.get_secret_value() if self.resource_config else None
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
