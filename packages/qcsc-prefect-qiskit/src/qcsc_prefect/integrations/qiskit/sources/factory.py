"""Build a quantum source from deployment settings and application sampling policy."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Literal

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from qiskit import QuantumCircuit

from ..qrmi import PreparedCircuit, QrmiRuntimeBlock
from .base import QuantumSource
from .qfw import DEFAULT_WORK_DIR, QFwSource
from .qfw_resource import QFwResource
from .qrmi import QrmiSource
from .random import RandomSource


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class QuantumSourceConfig:
    """Serializable connection settings, separate from algorithm sampling policy.

    ``block_name`` selects a saved runtime (``qrmi``) or resource (``qfw``).
    For QFw, ``sampler_arg_names`` maps compilation setting names to CLI flags,
    in argument order; values are taken from the application's ``transpile_config``.
    No algorithm hook, block name or CLI flag is supplied by the library.
    """

    kind: Literal["random", "qrmi", "qfw"]
    block_name: str | None = None
    tags: tuple[str, ...] = ()
    sampler: str | None = None
    sampler_args: tuple = ()
    sampler_arg_names: tuple[tuple[str, str], ...] = ()
    work_dir: str = DEFAULT_WORK_DIR

    def __post_init__(self) -> None:
        if self.kind in ("qrmi", "qfw") and (
            self.block_name is None or not self.block_name.strip()
        ):
            raise ValueError(f"{self.kind} requires a block_name")
        if self.kind == "qfw":
            if self.sampler is None or not self.sampler.strip():
                raise ValueError("qfw requires a sampler script path")
            names = [name for name, _ in self.sampler_arg_names]
            flags = [flag for _, flag in self.sampler_arg_names]
            if any(not name.strip() for name in names):
                raise ValueError("sampler_arg_names must have nonempty setting names")
            if any(not flag.startswith("-") or "=" in flag for flag in flags):
                raise ValueError("sampler_arg_names must use CLI flags without values")
            if len(set(names)) != len(names) or len(set(flags)) != len(flags):
                raise ValueError("sampler_arg_names must not repeat a setting or CLI flag")
            fixed_flags = {
                arg.split("=", 1)[0] for arg in self.sampler_args if isinstance(arg, str)
            }
            duplicates = fixed_flags.intersection(flags)
            if duplicates:
                raise ValueError(
                    "sampler_args duplicates mapped CLI flags: " + ", ".join(sorted(duplicates))
                )


def validate_qfw_transpile_config(config: QuantumSourceConfig, settings: Mapping | None) -> None:
    """Reject supplied compilation values that the configured script cannot receive.

    Missing values may still be provided by an algorithm at source construction.
    This validation performs no platform imports or Block reads.
    """
    unmapped = set(settings or {}) - {name for name, _ in config.sampler_arg_names}
    if unmapped:
        raise ValueError(
            "QFw compilation settings have no sampler_arg_names mapping: "
            + ", ".join(sorted(str(name) for name in unmapped))
        )


class SourceLoadError(RuntimeError):
    """A saved source block could not be loaded or its runtime is invalid."""


def load_quantum_source(
    config: QuantumSourceConfig,
    *,
    num_bits: int | None = None,
    seed: int | None = None,
    prepare: Callable[[QuantumCircuit], PreparedCircuit] | None = None,
    transpiler: str | None = None,
    transpile_config: Mapping | None = None,
) -> QuantumSource:
    """Load only the selected source's block, in the calling worker.

    ``num_bits`` and ``seed`` configure random sampling. Device sources use the
    optional application preparation and compilation policy. ``transpiler`` is a
    QRMI hook; QFw compilation uses the configured node script instead. Omitted QRMI policy
    values preserve the saved runtime's settings; a supplied configuration replaces
    its configuration. QFw uses the prepared circuit and maps the configuration to
    the configured node script's arguments. Source construction never submits work.
    """
    if config.kind == "random":
        if num_bits is None:
            raise ValueError("random requires num_bits")
        return RandomSource(num_bits=num_bits, seed=seed)

    if config.kind == "qrmi":
        try:
            runtime = QrmiRuntimeBlock.load_runtime(config.block_name)
        except ValueError as exc:
            raise SourceLoadError(
                f"QrmiRuntimeBlock '{config.block_name}' could not be loaded."
            ) from exc
        overrides = {}
        if prepare is not None:
            overrides["prepare"] = prepare
        if transpiler is not None:
            overrides["transpiler"] = transpiler
        if transpile_config is not None:
            overrides["transpile_config"] = dict(transpile_config)
        return QrmiSource(replace(runtime, **overrides), tags=config.tags)

    validate_qfw_transpile_config(config, transpile_config)
    args = list(config.sampler_args)
    for name, flag in config.sampler_arg_names:
        if transpile_config is None or name not in transpile_config:
            raise ValueError(f"Missing compilation setting for sampler argument: {name}")
        args.extend((flag, transpile_config[name]))
    try:
        resource = QFwResource.load(config.block_name)
    except ValueError as exc:
        raise SourceLoadError(f"QFwResource block '{config.block_name}' is not defined.") from exc

    def prepare_circuit(circuit):
        prepared = prepare(circuit)
        if prepared.config:
            raise ValueError("QFw does not support per-circuit compilation settings")
        return prepared.circuit

    return QFwSource(
        resource.device_id,
        sampler=config.sampler,
        sampler_args=args,
        prepare=prepare_circuit if prepare is not None else None,
        work_dir=config.work_dir,
    )
