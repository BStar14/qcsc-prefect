"""Flow body shared by the SQD workflows, laid out as the Qiskit Patterns stages.

::

    compute_molecular_integrals [map] ──────────────┐
       ▼ elec_props                                 │
    get_lucj_params [optimize] ◄ resolve_topology   │
       ▼ lucj_params                                │
    quantum_sampling | load_qrmi_result |           │
    load_runtime_job [execute]                      │
       ▼ bool_array                                 │
    sqd_loop [post-process] ◄───────────────────────┘

Prefect draws an edge between two tasks when the result of one is passed
unchanged into the other, so each stage hands its return value straight on.
``resolve_topology`` is a plain function: it picks the LUCJ interaction
pattern for the QPU topology before Optimize.
The Map, Optimize and Execute results are persisted. Retrying a failed flow
run therefore reuses them and repeats only ``sqd_loop``, which continues from
its checkpoint after the last completed recovery round.
"""

import numpy as np
from prefect import get_run_logger
from qcsc_workflow_utility import (
    ElectronicProperties,
    compute_molecular_integrals_from_fcidump,
    compute_molecular_integrals_from_geometry,
    to_sbd_closed_shell,
)

from sqd_common.flow_params import FCIDumpFile, FlowParameters, MoleculeGeometry
from sqd_common.lucj import get_lucj_params
from sqd_common.postprocess import SolverBlock, sqd_loop
from sqd_common.recovery import BatchTransformFn
from sqd_common.sampling import (
    load_qrmi_result,
    load_runtime_job,
    quantum_sampling,
    random_bool_array,
)
from sqd_common.solvers import FULQRUM_SLUG, SBD_SLUG, solver_class, solver_slug
from sqd_common.topology import resolve_topology


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


def check_solver_input(slug: str, elec_props: ElectronicProperties) -> None:
    """Reject a molecule the selected solver cannot solve, before any quantum job.

    SBD takes closed-shell systems with ``spin_sq`` None only. fulqrum has no spin
    penalty, so it ignores ``spin_sq``.
    """
    if slug == SBD_SLUG:
        from qcsc_prefect_sbd import check_sbd_input

        hartree_fock = np.array([(1 << elec_props.num_electrons[0]) - 1], dtype=np.int64)
        check_sbd_input(
            ci_strings=(hartree_fock, hartree_fock),
            nelec=elec_props.num_electrons,
            spin_sq=elec_props.spin_sq,
        )
    elif slug == FULQRUM_SLUG and elec_props.spin_sq:
        get_run_logger().warning(
            f"spin_sq={elec_props.spin_sq} is ignored: the fulqrum solver applies no spin penalty."
        )


def batch_transform(
    slug: str, subspace_dim: int, elec_props: ElectronicProperties
) -> BatchTransformFn | None:
    """SBD needs the Hartree-Fock determinant first and one set for alpha and beta."""
    if slug != SBD_SLUG:
        return None

    def to_sbd(batches, rng):
        return to_sbd_closed_shell(
            batches,
            subspace_dim=subspace_dim,
            num_elec_a=elec_props.num_electrons[0],
            rng=rng,
        )

    return to_sbd


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


async def run_sqd_pattern(parameters: FlowParameters, artifact_key: str) -> float:
    """Run Map → Optimize → Execute → Post-process and return the lowest energy.

    The solver block of ``parameters.solver_block_ref`` is loaded first and the
    molecule is checked against it right after Map, so a bad block name or input
    the solver cannot solve fails before any quantum job.

    Args:
        parameters: Flow parameters.
        artifact_key: Key of the configuration recovery table artifact.

    Returns:
        Minimum ground state energy solved by SQD.
    """
    logger = get_run_logger()
    slug, block_name = solver_slug(parameters.solver_block_ref)
    solver = await load_solver_block(solver_class(slug), block_name)
    sampling = parameters.sampling

    # Map
    elec_props = compute_molecular_integrals(parameters)
    check_solver_input(slug, elec_props)

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
            qpu_topology=resolve_topology(parameters.circuit, sampling),
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
        batch_transform=batch_transform(slug, parameters.sqd.subspace_dim, elec_props),
    )
