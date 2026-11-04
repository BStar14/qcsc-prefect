"""Flow body shared by the SQD workflows, laid out as the Qiskit Patterns stages.

::

    compute_molecular_integrals [map] ──────────────┐
       ▼ elec_props                                 │
    get_lucj_params [optimize]                      │
       ▼ lucj_params                                │
    quantum_sampling | load_qrmi_result |           │
    load_runtime_job [execute]                      │
       ▼ bool_array                                 │
    sqd_loop [post-process] ◄───────────────────────┘

Prefect draws an edge between two tasks when the result of one is passed
unchanged into the other, so each stage hands its return value straight on.
The Map, Optimize and Execute results are persisted. Retrying a failed flow
run therefore reuses them and repeats only ``sqd_loop``.
"""

from prefect import get_run_logger
from qcsc_workflow_utility import (
    ElectronicProperties,
    compute_molecular_integrals_from_fcidump,
    compute_molecular_integrals_from_geometry,
)

from sqd_common.flow_params import FCIDumpFile, FlowParameters, MoleculeGeometry
from sqd_common.lucj import get_lucj_params
from sqd_common.postprocess import SolverBlock, sqd_loop
from sqd_common.sampling import (
    load_qrmi_result,
    load_runtime_job,
    quantum_sampling,
    random_bool_array,
)


async def load_solver_block(solver_cls: type, name: str) -> SolverBlock:
    """Load the solver block before any expensive stage, so a bad name fails fast."""
    logger = get_run_logger()
    try:
        solver = await solver_cls.load(name)
    except Exception:
        logger.exception("Failed to load %s block '%s'.", solver_cls.__name__, name)
        raise
    logger.info("Solver OK: %s '%s'.", solver_cls.__name__, name)
    return solver


def compute_molecular_integrals(parameters: FlowParameters) -> ElectronicProperties:
    """Map stage: molecular integrals, CCSD amplitudes and initial occupancies."""
    molecule = parameters.molecule
    if isinstance(molecule, MoleculeGeometry):
        return compute_molecular_integrals_from_geometry.with_options(tags=["map"])(
            atom=molecule.atom,
            basis=molecule.basis,
            symmetry=molecule.symmetry,
            spin_sq=molecule.spin_sq,
            ccsd_maxiter=molecule.ccsd_maxiter,
        )
    if isinstance(molecule, FCIDumpFile):
        return compute_molecular_integrals_from_fcidump.with_options(tags=["map"])(
            fcidump_file=molecule.fcidump_file,
            spin_sq=molecule.spin_sq,
            ccsd_maxiter=molecule.ccsd_maxiter,
        )
    raise TypeError("Unsupported molecule type")


async def run_sqd_pattern(
    parameters: FlowParameters,
    solver_cls: type,
    artifact_key: str,
) -> float:
    """Run Map → Optimize → Execute → Post-process and return the lowest energy.

    Args:
        parameters: Flow parameters.
        solver_cls: Solver block class to load ``parameters.solver_block_name`` with.
        artifact_key: Key of the configuration recovery table artifact.

    Returns:
        Minimum ground state energy solved by SQD.
    """
    logger = get_run_logger()
    solver = await load_solver_block(solver_cls, parameters.solver_block_name)
    sampling = parameters.sampling

    # Map
    elec_props = compute_molecular_integrals(parameters)

    if sampling.quantum_source == "random":
        # No circuit runs, so the flow skips Optimize and Execute.
        bool_array = random_bool_array(sampling, elec_props.num_orbitals)
    elif sampling.qrmi_result_dir is not None:
        bool_array = await load_qrmi_result(
            work_dir=sampling.qrmi_result_dir,
            circuit_params=parameters.circuit,
        )
    elif sampling.qiskit_runtime_job_id is not None:
        bool_array = await load_runtime_job(
            job_id=sampling.qiskit_runtime_job_id,
            runner_name=sampling.runner_name,
            circuit_params=parameters.circuit,
        )
    else:
        # Optimize
        lucj_params = get_lucj_params(
            circuit_params=parameters.circuit,
            elec_props=elec_props,
        )
        # Execute
        bool_array = await quantum_sampling(
            lucj_params=lucj_params,
            circuit_params=parameters.circuit,
            sampling_params=sampling,
        )

    # Post-process
    logger.debug("Starting configuration recovery and diagonalization.")
    return await sqd_loop(
        bool_array=bool_array,
        elec_props=elec_props,
        sqd_params=parameters.sqd,
        solver=solver,
        artifact_key=artifact_key,
    )
