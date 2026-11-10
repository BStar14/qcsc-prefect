"""Compute-node entrypoint: QPY -> target-aware hook -> QRMI task_runner."""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
from pathlib import Path

from qiskit import qpy

# Re-exported: the default transpiler path and existing hooks import these names here.
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager  # noqa: F401

from .compile import (  # noqa: F401
    compile_pubs,
    generic_transpile,
    make_sampler_parameters,
    validate_isa,
)
from .contract import (
    SCHEMA_VERSION,
    TASK_ENDED_FILE,
    circuit_registers,
    decode_result,
    positive_shots,
)


def require_batch_environment(resource: str) -> None:
    # SPANK acquires the Batch before Python starts; never change its mode here.
    mode = os.environ.get(f"{resource}_QRMI_IBM_QCS_SESSION_MODE", "")
    if mode.lower() != "batch":
        raise RuntimeError(
            f'Configure {resource} QRMI_IBM_QCS_SESSION_MODE="batch" on every compute node'
        )
    if not os.environ.get(f"{resource}_QRMI_JOB_ACQUISITION_TOKEN"):
        raise RuntimeError(f"Missing SPANK acquisition token; submit with --qpu={resource}")


def make_sampler_payload(isa_circuits, shots: int) -> dict:
    return {"program_id": "sampler", "parameters": make_sampler_parameters(isa_circuits, shots)}


class QuantumTaskEnded(RuntimeError):
    """The quantum task ended without a result: cancelled, failed or stopped."""


# task_runner logs the final status of a task that ended without a result, for example
# "ERROR TaskStatus.Cancelled", and exits 0 without writing the result file.
_TASK_STATUS = re.compile(r"TaskStatus\.(\w+)")
_TASK_ID = re.compile(r"Task ID: (\S+)")


def _task_ended_message(output: str, resource: str) -> str:
    statuses = _TASK_STATUS.findall(output)
    task_ids = _TASK_ID.findall(output)
    task = f"Quantum task {task_ids[-1]} on {resource}" if task_ids else f"The task on {resource}"
    status = statuses[-1] if statuses else None
    if status == "Cancelled":
        return f"{task} was cancelled on IBM Quantum; no samples were returned."
    if status == "Failed":
        return f"{task} failed on IBM Quantum; see its details on the IBM Quantum dashboard."
    return (
        f"{task} ended without a result (task_runner wrote no result.json); the job was "
        "probably stopped, for example cancelled from Prefect or by Slurm."
    )


def run_task_runner(command: list[str]) -> str:
    """Run task_runner, copying its log to this job's output as it comes; return the log.

    Raises:
        subprocess.CalledProcessError: If task_runner exits with an error.
    """
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace"
    )
    lines = []
    try:
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            lines.append(line)
        returncode = process.wait()
    except BaseException:
        # Stopping the worker stops task_runner, which then stops its quantum task.
        process.terminate()
        process.wait()
        raise
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, command)
    return "".join(lines)


def execute_task_runner(isa_circuits, shots: int, executable: Path, resource: str) -> None:
    Path("sampler_input.json").write_text(
        json.dumps(make_sampler_payload(isa_circuits, shots), indent=2), encoding="utf-8"
    )
    # No retries: a zero CLI exit code alone does not prove quantum success.
    output = run_task_runner([str(executable), resource, "sampler_input.json", "result.json"])
    result_path = Path("result.json").resolve()
    if not result_path.is_file():
        raise QuantumTaskEnded(_task_ended_message(output, resource))
    decode_result(result_path, shots, [circuit_registers(isa) for isa in isa_circuits])


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
    from qrmi.primitives.ibm import get_backend

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
    target = get_backend(resource).target
    isa_circuits, pub_reports = compile_pubs(
        circuits, configs, registers, target, request["transpiler"]
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


def run() -> None:
    """Console entry point: a quantum task that ended without a result exits with one line."""
    try:
        main()
    except QuantumTaskEnded as exc:
        # The ended task file tells the workflow host why, without a traceback to read.
        Path(TASK_ENDED_FILE).write_text(str(exc) + "\n", encoding="utf-8")
        print(f"qcsc-qrmi-sampler: {exc}", file=sys.stderr)
        raise SystemExit(3) from None


if __name__ == "__main__":
    run()
