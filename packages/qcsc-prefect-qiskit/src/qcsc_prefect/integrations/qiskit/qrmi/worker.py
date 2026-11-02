"""Compute-node entrypoint: QPY -> target-aware hook -> QRMI task_runner."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

from qiskit import qasm3, qpy
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from .contract import (
    SCHEMA_VERSION,
    circuit_registers,
    decode_result,
    positive_shots,
)
from .transpilation import preset_pass_manager_options


def require_batch_environment(resource: str) -> None:
    # SPANK acquires the Batch before Python starts; never change its mode here.
    mode = os.environ.get(f"{resource}_QRMI_IBM_QCS_SESSION_MODE", "")
    if mode.lower() != "batch":
        raise RuntimeError(
            f'Configure {resource} QRMI_IBM_QCS_SESSION_MODE="batch" on every compute node'
        )
    if not os.environ.get(f"{resource}_QRMI_JOB_ACQUISITION_TOKEN"):
        raise RuntimeError(f"Missing SPANK acquisition token; submit with --qpu={resource}")


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


def make_sampler_payload(isa_circuits, shots: int) -> dict:
    return {
        "program_id": "sampler",
        "parameters": {
            "pubs": [
                [qasm3.dumps(isa, disable_constants=True, allow_aliasing=True), None, shots]
                for isa in isa_circuits
            ],
            "version": 2,
            "support_qiskit": True,
            "shots": shots,
            "options": {},
        },
    }


def execute_task_runner(isa_circuits, shots: int, executable: Path, resource: str) -> None:
    Path("sampler_input.json").write_text(
        json.dumps(make_sampler_payload(isa_circuits, shots), indent=2), encoding="utf-8"
    )
    # No retries: a zero CLI exit code alone does not prove quantum success.
    subprocess.run([str(executable), resource, "sampler_input.json", "result.json"], check=True)
    decode_result(
        Path("result.json").resolve(), shots, [circuit_registers(isa) for isa in isa_circuits]
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", default="sampler_request.json")
    parser.add_argument(
        "--task-runner", type=Path, default=Path(sys.executable).with_name("task_runner")
    )
    args = parser.parse_args(argv)
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    if request["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported sampler request schema_version")
    shots = positive_shots(request["shots"])
    resource_id = request["resource"]
    require_batch_environment(resource_id)

    # QRMI reads SPANK's environment; import only inside the allocation.
    from qrmi import QRMIService
    from qrmi.primitives.ibm import get_target

    resource = next((r for r in QRMIService().resources() if r.resource_id() == resource_id), None)
    if resource is None:
        raise RuntimeError(f"{resource_id} is not present in this Slurm allocation")
    with Path("circuit.qpy").open("rb") as stream:
        circuits = qpy.load(stream)
    registers = request["registers"]
    configs = json.loads(Path("transpile_config.json").read_text(encoding="utf-8"))
    if not isinstance(configs, list) or not isinstance(registers, list):
        raise ValueError("registers and transpile_config must have one entry per PUB")
    if not circuits or not len(circuits) == len(registers) == len(configs):
        raise ValueError(
            f"QPY has {len(circuits)} circuits, request has {len(registers)} registers "
            f"and {len(configs)} transpile configs; expected one of each per PUB"
        )
    if [circuit_registers(circuit) for circuit in circuits] != registers:
        raise ValueError("QPY registers do not match sampler request")
    module_name, function_name = request["transpiler"].split(":")
    transpile = getattr(importlib.import_module(module_name), function_name)
    target = get_target(resource)
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
    report = {
        "transpiler": request["transpiler"],
        "target_num_qubits": target.num_qubits,
        "host": socket.gethostname(),
        "resource": resource_id,
        "mode": "batch",
        "pubs": pub_reports,
    }
    Path("transpile_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with Path("isa.qpy").open("wb") as stream:
        qpy.dump(isa_circuits, stream)
    print(f">>> Host: {socket.gethostname()}, resource: {resource_id}, mode: batch", flush=True)
    execute_task_runner(isa_circuits, shots, args.task_runner, resource_id)


if __name__ == "__main__":
    main()
