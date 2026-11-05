#!/usr/bin/env python3

import argparse
import gzip
import json
import logging
import os
import socket
import time

from qfw_qiskit import QFwBackend
from qfw_qiskit.reservation_set import (
    parse_qfw_reservations,
    select_qpm_reservation,
)
from qiskit import qpy
from transpile_core import (
    _find_optimal_layout,
    _transpile_circuit,
)

GZIP_MAGIC = b"\x1f\x8b"


def load_circuit(path):
    with open(path, "rb") as handle:
        magic = handle.read(2)

    opener = gzip.open if magic == GZIP_MAGIC else open

    with opener(path, "rb") as handle:
        circuits = qpy.load(handle)

    if len(circuits) != 1:
        raise SystemExit(f"{path} holds {len(circuits)} circuits, expected one")

    return circuits[0]


def reservation_options(service_id=None):
    reservations = parse_qfw_reservations(required=False)

    if not reservations:
        return {}, None

    selected = select_qpm_reservation(
        reservations,
        service_id=service_id,
    )

    options = {
        "reservation_id": selected.reservation_id,
    }

    token = os.environ.get("QFW_RESERVATION_TOKEN")
    if token:
        options["token"] = token

    return options, selected


def main():
    parser = argparse.ArgumentParser(description="Run the SQD quantum stage through QFw-SLURM.")

    parser.add_argument("--circuit", required=True)
    parser.add_argument("--shots", type=int, required=True)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--service-id", default=None)
    parser.add_argument("--out", required=True)

    parser.add_argument("--optimization-level", type=int, default=3)
    parser.add_argument("--sabre-max-iterations", type=int, default=8)
    parser.add_argument("--sabre-swap-trials", type=int, default=10)
    parser.add_argument("--sabre-layout-trials", type=int, default=1024)

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logger = logging.getLogger("qfw-sqd-sampler")

    circuit = load_circuit(args.circuit)

    options, reservation = reservation_options(args.service_id)

    service_id = args.service_id
    if service_id is None and reservation is not None:
        service_id = reservation.service_id

    logger.info(
        "Using QPM service %s for %s-qubit circuit.",
        service_id,
        circuit.num_qubits,
    )

    backend = QFwBackend(
        provider=None,
        service_id=service_id,
    )

    logger.info("QFw backend created: %s", backend.name)

    target_start = time.time()
    target = backend.target

    logger.info(
        "Loaded backend target in %.2fs.",
        time.time() - target_start,
    )

    layout, _ = _find_optimal_layout(
        test_circuit=circuit,
        target=target,
        optimization_level=args.optimization_level,
        max_iterations=args.sabre_max_iterations,
        swap_trials=args.sabre_swap_trials,
        layout_trials=args.sabre_layout_trials,
        logger=logger,
        use_pre_init=False,
    )

    logger.info("Completed SQD layout search.")

    transpile_start = time.time()

    isa_circuit = _transpile_circuit(
        circuit=circuit,
        target=target,
        layout=layout,
        optimization_level=args.optimization_level,
        use_pre_init=False,
    )

    logger.info(
        "Completed SQD transpilation in %.2fs. ISA circuit depth=%s, qubits=%s.",
        time.time() - transpile_start,
        isa_circuit.depth(),
        isa_circuit.num_qubits,
    )

    run_start = time.time()

    job = backend.run(
        isa_circuit,
        shots=args.shots,
        **options,
    )

    result = job.result()
    counts = result.get_counts(isa_circuit)

    logger.info(
        "Completed IBM execution in %.2fs with %s distinct outcomes.",
        time.time() - run_start,
        len(counts),
    )

    record = {
        "counts": {str(key): int(value) for key, value in counts.items()},
        "shots": args.shots,
        "provider": args.provider,
        "backend": backend.name,
        "service_id": reservation.service_id if reservation else None,
        "reservation_id": reservation.reservation_id if reservation else None,
        "circuit": {
            "name": isa_circuit.name,
            "num_qubits": isa_circuit.num_qubits,
            "num_clbits": isa_circuit.num_clbits,
            "depth": isa_circuit.depth(),
        },
        "job_id": getattr(job, "job_id", lambda: None)(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "host": socket.gethostname(),
        "time_taken_sec": getattr(result, "time_taken", None),
    }

    tmp = f"{args.out}.tmp"

    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")

    os.replace(tmp, args.out)

    logger.info(
        "Wrote %s shots to %s.",
        sum(counts.values()),
        args.out,
    )


if __name__ == "__main__":
    main()
