"""A reusable Prefect Block selecting a quantum source for any application."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from prefect.blocks.core import Block
from pydantic import Field, model_validator

from qiskit import QuantumCircuit

from ..qrmi import PreparedCircuit
from .base import QuantumSource
from .factory import (
    QuantumSourceConfig,
    SourceLoadError,
    load_quantum_source,
    validate_qfw_transpile_config,
)


class QuantumSourceBlock(Block):
    """Saved source selection and platform settings, shared by all algorithms.

    References existing runtime/resource Blocks rather than copying credentials.
    Loading this Block does not load a platform Block or import a platform SDK.
    Call ``get_source`` in the sampling worker to construct its own source. Dynamic
    algorithm policy can be supplied there without being stored in this Block.
    """

    _block_type_name = "Quantum Source"
    _block_type_slug = "quantum-source"

    config: QuantumSourceConfig = Field(
        title="Source Configuration",
        description="Source kind and its runtime/resource Block, tags and node script settings.",
    )
    num_bits: int | None = Field(
        default=None,
        gt=0,
        strict=True,
        description="Default random width; the application may supply it.",
    )
    seed: int | None = Field(default=None, description="Default random seed.")
    transpiler: str | None = Field(
        default=None, description="Optional QRMI compiler hook, as module:function."
    )
    transpile_config: dict | None = Field(
        default=None,
        description="Optional compilation settings; omitted values keep the runtime's settings.",
    )

    @model_validator(mode="before")
    @classmethod
    def reject_unknown_settings(cls, values):
        # Block's serializer includes this metadata in round trips. Preserve its
        # standard handling while rejecting misspelled application settings.
        metadata = {
            "block_type_slug",
            "_block_document_id",
            "_block_document_name",
            "_is_anonymous",
        }
        if isinstance(values, Mapping):
            unknown = set(values) - set(cls.model_fields) - metadata
            if unknown:
                raise ValueError(
                    "Unknown QuantumSourceBlock settings: " + ", ".join(sorted(unknown))
                )
        return values

    @model_validator(mode="after")
    def validate_saved_compilation(self):
        if self.config.kind == "qfw":
            if self.transpiler is not None:
                raise ValueError(
                    "QFw uses config.sampler for compilation; do not save a QRMI transpiler hook"
                )
            validate_qfw_transpile_config(self.config, self.transpile_config)
        return self

    def get_source(
        self,
        *,
        num_bits: int | None = None,
        seed: int | None = None,
        prepare: Callable[[QuantumCircuit], PreparedCircuit] | None = None,
        transpiler: str | None = None,
        transpile_config: Mapping | None = None,
    ) -> QuantumSource:
        """Build one source with saved defaults and explicit application overrides.

        The application owns its circuit preparation, random width/seed calculation
        and compilation policy. Supplied values override this Block's defaults;
        an explicit compilation configuration replaces rather than merges them.
        No backend submission happens until ``source.sample()`` is called.
        """
        return load_quantum_source(
            self.config,
            num_bits=self.num_bits if num_bits is None else num_bits,
            seed=self.seed if seed is None else seed,
            prepare=prepare,
            transpiler=self.transpiler if transpiler is None else transpiler,
            transpile_config=(
                self.transpile_config if transpile_config is None else transpile_config
            ),
        )

    @classmethod
    def load_source(
        cls,
        name: str,
        *,
        num_bits: int | None = None,
        seed: int | None = None,
        prepare: Callable[[QuantumCircuit], PreparedCircuit] | None = None,
        transpiler: str | None = None,
        transpile_config: Mapping | None = None,
    ) -> QuantumSource:
        """Load a named source Block and construct a source in synchronous code.

        ``load`` retains the standard Prefect behavior and returns the saved Block.
        Pass only the Block name between workers, then call this method there.
        """
        try:
            block = cls.load(name, _sync=True)
        except ValueError as exc:
            raise SourceLoadError(
                f"QuantumSourceBlock '{name}' could not be loaded. Create it with "
                f"QuantumSourceBlock(config=...).save({name!r}) in the same Prefect workspace, "
                "or select the name of an existing source Block. This API expects a saved "
                "Block name, not a built-in platform selector."
            ) from exc
        return block.get_source(
            num_bits=num_bits,
            seed=seed,
            prepare=prepare,
            transpiler=transpiler,
            transpile_config=transpile_config,
        )
