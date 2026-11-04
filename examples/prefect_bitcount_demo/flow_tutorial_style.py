from __future__ import annotations

import argparse
import asyncio

from prefect import flow
from prefect.artifacts import create_table_artifact
from prefect.variables import Variable

try:
    from get_counts_integration import BITLEN, BitCounter
    from options_resolver import resolve_sampler_options_and_work_dir
    from quantum_sampling import QuantumSource, sample_bitstrings
except ModuleNotFoundError:
    from examples.prefect_bitcount_demo.get_counts_integration import BITLEN, BitCounter
    from examples.prefect_bitcount_demo.options_resolver import (
        resolve_sampler_options_and_work_dir,
    )
    from examples.prefect_bitcount_demo.quantum_sampling import (
        QuantumSource,
        sample_bitstrings,
    )


@flow(name="miyabi_tutorial")
async def main(
    quantum_source: QuantumSource = "real-device",
    runtime_block_name: str = "qrmi-runtime",
    bitcounter_block_name: str = "miyabi-tutorial",
    options_variable_name: str = "miyabi-tutorial",
    hpc_profile_block_override: str | None = None,
    random_seed: int = 24,
):
    counter = await BitCounter.load(bitcounter_block_name)
    if hpc_profile_block_override:
        counter = counter.model_copy(update={"hpc_profile_block_name": hpc_profile_block_override})
    options_raw = await Variable.get(options_variable_name)
    sampler_options, _ = resolve_sampler_options_and_work_dir(
        options_raw,
        default_shots=100000,
    )
    bitstrings = await sample_bitstrings(
        quantum_source=quantum_source,
        runtime_block_name=runtime_block_name,
        sampler_options=sampler_options,
        bitlen=BITLEN,
        default_shots=100000,
        random_seed=random_seed,
    )
    counts = await counter.get(bitstrings)

    await create_table_artifact(
        table=[list(counts.keys()), list(counts.values())],
        key="sampler-count-dict",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run tutorial-style BitCount flow with an optional HPC profile override."
    )
    parser.add_argument(
        "--quantum-source",
        choices=("real-device", "random"),
        default="real-device",
        help="Choose QrmiRuntime sampling or deterministic random bitstrings.",
    )
    parser.add_argument("--runtime-block", default="qrmi-runtime")
    parser.add_argument("--bitcounter-block", default="miyabi-tutorial")
    parser.add_argument("--options-variable", default="miyabi-tutorial")
    parser.add_argument("--hpc-profile-block-override", default=None)
    parser.add_argument(
        "--random-seed",
        type=int,
        default=24,
        help="Base seed used when --quantum-source random is selected.",
    )
    args = parser.parse_args()

    asyncio.run(
        main(
            quantum_source=args.quantum_source,
            runtime_block_name=args.runtime_block,
            bitcounter_block_name=args.bitcounter_block,
            options_variable_name=args.options_variable,
            hpc_profile_block_override=args.hpc_profile_block_override,
            random_seed=args.random_seed,
        )
    )
