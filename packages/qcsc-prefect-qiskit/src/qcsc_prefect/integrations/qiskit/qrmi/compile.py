"""Target-aware compilation shared by the Slurm worker and direct IQP execution."""

from __future__ import annotations

import importlib

from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from .contract import circuit_registers
from .transpilation import preset_pass_manager_options


def generic_transpile(circuit, target, config):
    """Configure a preset pass manager using JSON options and the actual Target."""
    options = preset_pass_manager_options(config, num_qubits=circuit.num_qubits)
    return generate_preset_pass_manager(target=target, **options).run(circuit)


def validate_isa(circuit, target) -> None:
    for instruction in circuit.data:
        if instruction.operation.name == "barrier":
            continue
        qargs = tuple(circuit.find_bit(bit).index for bit in instruction.qubits)
        if not target.instruction_supported(
            operation_name=instruction.operation.name,
            qargs=qargs,
            parameters=instruction.operation.params,
        ):
            raise RuntimeError(f"Unsupported ISA instruction: {instruction.operation.name} {qargs}")


def load_transpiler(path: str):
    module_name, function_name = path.split(":")
    return getattr(importlib.import_module(module_name), function_name)


def compile_pubs(circuits, configs, registers, target, transpiler: str):
    """Compile circuit ``i`` with ``configs[i]``; return ISA circuits and per-PUB reports."""
    transpile = load_transpiler(transpiler)
    isa_circuits = []
    pub_reports = []
    for circuit, config, pub_registers in zip(circuits, configs, registers):
        isa = transpile(circuit, target, config)
        if circuit_registers(isa) != pub_registers:
            raise ValueError("transpiler must preserve classical register names and widths")
        validate_isa(isa, target)
        isa_circuits.append(isa)
        pub_reports.append(
            {
                **config,
                "input_ops": dict(circuit.count_ops()),
                "isa_ops": dict(isa.count_ops()),
                "isa_num_qubits": isa.num_qubits,
                "registers": pub_registers,
            }
        )
    return isa_circuits, pub_reports
