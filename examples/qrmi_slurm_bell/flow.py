"""Sample a measured two-qubit Bell circuit through the QRMI/Slurm library."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from prefect import flow
from qcsc_prefect.integrations.qiskit.qrmi import QrmiSlurmRuntime
from qiskit import QuantumCircuit


@flow(name="qrmi-slurm-bell", retries=0, persist_result=False)
async def bell_flow(
    *,
    shots: int = 100,
    work_dir: str = "/shared/qrmi-bell",
    resource: str = "test_heron",
    command_block_name: str = "qrmi-bell-command",
    execution_profile_block_name: str = "qrmi-bell-execution",
    hpc_profile_block_name: str = "qrmi-bell-hpc",
) -> dict:
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure_all()

    runtime = QrmiSlurmRuntime(
        root_dir=Path(work_dir),
        command_block_name=command_block_name,
        execution_profile_block_name=execution_profile_block_name,
        hpc_profile_block_name=hpc_profile_block_name,
        resource=resource,
    )
    result = await runtime.sampler([(circuit,)], shots=shots)
    samples = result[0].data.meas
    return {
        "num_bits": samples.num_bits,
        "num_shots": samples.num_shots,
        "counts": samples.get_counts(),
        "slurm": result.metadata["qcsc_slurm"],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots", type=int, default=100)
    parser.add_argument("--work-dir", default="/shared/qrmi-bell")
    parser.add_argument("--resource", default="test_heron")
    parser.add_argument("--command-block-name", default="qrmi-bell-command")
    parser.add_argument("--execution-profile-block-name", default="qrmi-bell-execution")
    parser.add_argument("--hpc-profile-block-name", default="qrmi-bell-hpc")
    args = parser.parse_args(argv)
    print(json.dumps(asyncio.run(bell_flow(**vars(args))), indent=2))


if __name__ == "__main__":
    main()
