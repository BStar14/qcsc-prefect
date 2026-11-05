"""Sample a measured two-qubit Bell circuit directly on IQP through QRMI."""

from __future__ import annotations

import argparse
import asyncio
import json

from prefect import flow
from qcsc_prefect.integrations.qiskit.qrmi import QrmiRuntimeBlock
from qiskit import QuantumCircuit


@flow(name="qrmi-iqp-bell", retries=0, persist_result=False)
async def bell_flow(*, shots: int = 100, runtime_block_name: str = "qrmi-iqp-bell") -> dict:
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure_all()

    runtime = await QrmiRuntimeBlock.aload_runtime(runtime_block_name)
    result = await runtime.sampler([(circuit,)], shots=shots)
    samples = result[0].data.meas
    record = result.metadata["qcsc_qrmi"]
    return {
        "num_bits": samples.num_bits,
        "num_shots": samples.num_shots,
        "counts": samples.get_counts(),
        "job_id": record["job_id"],
        "resource": record["resource"],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots", type=int, default=100)
    parser.add_argument("--runtime-block-name", default="qrmi-iqp-bell")
    args = parser.parse_args(argv)
    print(json.dumps(asyncio.run(bell_flow(**vars(args))), indent=2))


if __name__ == "__main__":
    main()
