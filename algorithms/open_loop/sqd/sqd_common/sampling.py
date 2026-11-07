"""Execute stage: sample bitstrings on a QPU, or load them from an earlier job.

Every path returns a 2D bool array, one row per shot, in the ``BitArray``
big-endian order (the right half of a row is alpha, orbital 0 rightmost).
The flow hands this array unchanged to ``sqd_loop``, which is what makes
Prefect draw the Execute → Post-process edge.
"""

import json
from pathlib import Path

import numpy as np
from prefect import get_run_logger, task
from prefect.artifacts import create_table_artifact
from prefect_qiskit.runtime import QuantumRuntime
from qiskit.primitives import PrimitiveResult
from qiskit.primitives.containers import BitArray
from qiskit_addon_sqd.counts import generate_bit_array_uniform

from sqd_common.advanced import AdvancedSettings
from sqd_common.flow_params import SamplingParameters
from sqd_common.lucj import LUCJParameters, build_lucj_circuit, circuit_transpile_options

SAMPLING_ARTIFACT_KEY = "sqd-sampling-source"
# Seed of the transpiler RNG; the compilation hook gets the integer it draws.
TRANSPILER_SEED = 24


def extract_bool_array(
    pub_result: PrimitiveResult, use_reset_mitigation: bool
) -> tuple[np.ndarray, int]:
    """Measured bitstrings of a sampler result, after optional reset postselection.

    With reset mitigation, keep only the shots whose ``test`` register reads all
    zeros.

    Returns:
        The bool array and the number of shots before postselection.
    """
    data = pub_result[0].data
    meas_bits = data.meas
    if not use_reset_mitigation:
        return meas_bits.to_bool_array(), meas_bits.num_shots
    if "test" not in data:
        raise ValueError(
            "Reset mitigation is on, but the result has no 'test' register. "
            "Sample with use_reset_mitigation=True or turn it off."
        )
    bitstrings = meas_bits.get_bitstrings(data.test.bitcount() == 0)
    bit_array = BitArray.from_samples(bitstrings, num_bits=meas_bits.num_bits)
    get_run_logger().info(
        "Reset mitigation result:\n"
        f"  Before: {meas_bits.num_shots} bitstrings\n"
        f"  After: {bit_array.num_shots} bitstrings\n"
        f"  Retention rate: {bit_array.num_shots / meas_bits.num_shots}\n"
    )
    return bit_array.to_bool_array(), meas_bits.num_shots


def random_bool_array(sampling_params: SamplingParameters, norb: int) -> np.ndarray:
    """Deterministic uniform random bitstrings for a run without a QPU."""
    get_run_logger().info(
        "Quantum source is random. Sampling %s deterministic pseudo-random bitstrings.",
        sampling_params.shots,
    )
    return generate_bit_array_uniform(
        sampling_params.shots,
        2 * norb,
        rand_seed=sampling_params.random_seed,
    ).to_bool_array()


async def record_sampling_source(*, bool_array: np.ndarray, shots: int, **source) -> None:
    """Write the ``sqd-sampling-source`` artifact, so a later run can reload the samples.

    ``qpu_topology`` is the topology the LUCJ interaction pairs were chosen for. A
    reload does not build the circuit and leaves it empty.
    """
    row = {
        "source": source.pop("source"),
        "resource": source.pop("resource", None),
        "qiskit_runtime_job_id": source.pop("qiskit_runtime_job_id", None),
        "qrmi_result_dir": source.pop("qrmi_result_dir", None),
        "qpu_topology": source.pop("qpu_topology", None),
        **source,
        "shots": int(shots),
        "shots_kept": int(len(bool_array)),
    }
    await create_table_artifact(
        table=[row],
        key=SAMPLING_ARTIFACT_KEY,
        description=(
            "Where the samples of this run came from. Paste qiskit_runtime_job_id (IQP) or "
            "qrmi_result_dir (Slurm) into the Sampling parameters of a new run to reuse them."
        ),
    )


async def _load_runtime(runner_name: str) -> QuantumRuntime:
    try:
        return await QuantumRuntime.load(runner_name)
    except ValueError as exc:
        raise RuntimeError(
            f"QuantumRuntime block '{runner_name}' is not defined. It is needed only to "
            "reload an IBM Quantum job by qiskit_runtime_job_id."
        ) from exc


async def _load_qrmi_runtime(block_name: str):
    from qcsc_prefect.integrations.qiskit.qrmi import QrmiRuntimeBlock

    try:
        return await QrmiRuntimeBlock.aload_runtime(block_name)
    except ValueError as exc:
        raise RuntimeError(
            f"Quantum source 'real-device' requested but QrmiRuntimeBlock '{block_name}' "
            "could not be loaded. Set quantum_source='random' to use deterministic random "
            "sampling instead."
        ) from exc


@task(
    name="quantum_sampling",
    tags=["execute"],
    persist_result=True,
    result_serializer="compressed/pickle",
)
async def quantum_sampling(
    lucj_params: LUCJParameters,
    advanced: AdvancedSettings,
    sampling_params: SamplingParameters,
) -> np.ndarray:
    """Build the LUCJ circuit and sample it through the QRMI runtime block.

    The host runs the ffsim pre-init passes and expands the circuit for QPY. The
    block's scheduler decides where the SQD compilation hook runs: in this
    process for ``iqp``, on the compute node for ``slurm``. Both compile with the
    same pass manager and seed.

    Args:
        lucj_params: LUCJ operator parameters, including the system size.
        advanced: Advanced SQD Settings: reset mitigation and the transpiler.
        sampling_params: Shots and the QRMI runtime block.

    Returns:
        Sampled bitstrings as a 2D bool array.
    """
    from sqd_common.qrmi import compiled_circuit_summary, make_qrmi_runtime

    logger = get_run_logger()
    rng = np.random.default_rng(TRANSPILER_SEED)
    shots = sampling_params.shots
    runtime = make_qrmi_runtime(
        await _load_qrmi_runtime(sampling_params.runtime_block_name),
        transpile_config=circuit_transpile_options(advanced, rng),
        resource=sampling_params.qpu_resource,
    )
    lucj_circ = build_lucj_circuit(
        lucj_params=lucj_params, use_reset_mitigation=advanced.use_reset_mitigation
    )
    logger.info(
        "Submitting sampler workload to QRMI resource %s (scheduler=%s, shots=%s).",
        runtime.resource,
        runtime.scheduler,
        shots,
    )
    pub_result = await runtime.sampler(sampler_pubs=[(lucj_circ,)], shots=shots)
    bool_array, num_shots = extract_bool_array(pub_result, advanced.use_reset_mitigation)

    slurm = pub_result.metadata.get("qcsc_slurm")
    if slurm is not None:
        summary = compiled_circuit_summary(pub_result)
        if summary is None:
            logger.warning("The compiled circuit (isa.qpy) is not readable from this host.")
        else:
            logger.info(summary)
        source = dict(qrmi_result_dir=slurm.get("work_dir"), slurm_job_id=slurm.get("job_id"))
    else:
        iqp = pub_result.metadata.get("qcsc_qrmi") or {}
        (report,) = iqp.get("pubs") or [{}]
        logger.info(f"Instruction counts = {report.get('isa_ops')}")
        source = dict(qiskit_runtime_job_id=iqp.get("job_id"))
    await record_sampling_source(
        bool_array=bool_array,
        shots=num_shots,
        source=f"qrmi-{runtime.scheduler}",
        resource=runtime.resource,
        qpu_topology=lucj_params.qpu_topology,
        **source,
    )
    return bool_array


@task(
    name="load_runtime_job",
    tags=["execute"],
    persist_result=True,
    result_serializer="compressed/pickle",
)
async def load_runtime_job(
    job_id: str,
    runner_name: str,
    use_reset_mitigation: bool,
) -> np.ndarray:
    """Load the samples of a finished IBM Quantum Runtime job.

    Args:
        job_id: Qiskit Runtime job ID.
        runner_name: QuantumRuntime block whose credentials can read the job.
        use_reset_mitigation: Whether the job was sampled with reset mitigation.

    Returns:
        Sampled bitstrings as a 2D bool array.
    """
    logger = get_run_logger()
    runtime = await _load_runtime(runner_name)
    logger.info("Loading samples from Qiskit Runtime job %s.", job_id)
    client = runtime.credentials.get_client()
    pub_result = await client.get_primitive_result(job_id=job_id)
    bool_array, num_shots = extract_bool_array(pub_result, use_reset_mitigation)
    await record_sampling_source(
        bool_array=bool_array,
        shots=num_shots,
        source="ibm-runtime (reloaded)",
        resource=runtime.resource_name,
        qiskit_runtime_job_id=job_id,
    )
    return bool_array


@task(
    name="load_qrmi_result",
    tags=["execute"],
    persist_result=True,
    result_serializer="compressed/pickle",
)
async def load_qrmi_result(
    work_dir: str,
    use_reset_mitigation: bool,
) -> np.ndarray:
    """Load the samples of a finished QRMI Slurm job from its job directory.

    Args:
        work_dir: QRMI job directory holding ``sampler_request.json`` and ``result.json``.
        use_reset_mitigation: Whether the job was sampled with reset mitigation.

    Returns:
        Sampled bitstrings as a 2D bool array.
    """
    from qcsc_prefect.integrations.qiskit.qrmi.contract import decode_result

    logger = get_run_logger()
    job_dir = Path(work_dir).expanduser()
    request_path = job_dir / "sampler_request.json"
    result_path = job_dir / "result.json"
    for path in (request_path, result_path):
        if not path.is_file():
            raise FileNotFoundError(f"QRMI job directory has no {path.name}: {job_dir}")
    request = json.loads(request_path.read_text(encoding="utf-8"))
    logger.info("Loading samples from QRMI job directory %s.", job_dir)
    registers = request["registers"]
    # Schema 1 job directories hold one PUB's registers; schema 2 holds a list.
    if isinstance(registers, dict):
        registers = [registers]
    pub_result = decode_result(result_path, request["shots"], registers)
    bool_array, num_shots = extract_bool_array(pub_result, use_reset_mitigation)
    await record_sampling_source(
        bool_array=bool_array,
        shots=num_shots,
        source="qrmi (reloaded)",
        resource=request.get("resource"),
        qrmi_result_dir=str(job_dir),
    )
    return bool_array
