# Workflow for observability demo on Miyabi
#
# Author: Naoki Kanazawa (knzwnao@jp.ibm.com)

from time import perf_counter

from qiskit import QuantumCircuit
from qiskit.passmanager import ConditionalController
from qiskit.transpiler import Layout, Target, generate_preset_pass_manager
from qiskit.transpiler.passes import (
    ApplyLayout,
    BarrierBeforeFinalMeasurements,
    EnlargeWithAncilla,
    FullAncillaAllocation,
    Optimize1qGatesDecomposition,
    RemoveIdentityEquivalent,
    SabreLayout,
    SetLayout,
)
from qiskit.transpiler.passmanager import PassManager
from qiskit_ibm_runtime.transpiler.passes import FoldRzzAngle

TRANSPILE_SEED = 6538
SLOW_LAYOUT_TRIAL_THRESHOLD = 4_096


def _transpile_circuit(
    circuit: QuantumCircuit,
    target: Target,
    layout: Layout,
    optimization_level: int,
    use_pre_init: bool = True,
) -> QuantumCircuit:
    custom_pm = generate_preset_pass_manager(
        optimization_level=optimization_level,
        seed_transpiler=TRANSPILE_SEED,
        target=target,
    )
    if use_pre_init:
        from ffsim.qiskit import PRE_INIT

        custom_pm.pre_init = PRE_INIT
    custom_pm.layout = PassManager(
        [
            BarrierBeforeFinalMeasurements(
                label="qiskit.transpiler.internal.routing.protection.barrier",
            ),
            SetLayout(
                layout=layout,
            ),
            FullAncillaAllocation(coupling_map=target.build_coupling_map()),
            EnlargeWithAncilla(),
            ApplyLayout(),
        ]
    )
    if "rzz" in target.operation_names:
        custom_pm.post_optimization = PassManager(
            [
                FoldRzzAngle(),
                Optimize1qGatesDecomposition(target=target),
                RemoveIdentityEquivalent(target=target),
            ]
        )
    return custom_pm.run(circuit)


def _find_optimal_layout(
    test_circuit: QuantumCircuit,
    target: Target,
    optimization_level: int,
    max_iterations: int,
    swap_trials: int,
    layout_trials: int,
    logger=None,
    use_pre_init: bool = True,
) -> tuple[Layout, list[dict]]:
    coupling_map = target.build_coupling_map()
    num_qubits = len(test_circuit.qubits)

    if logger is not None:
        logger.info(
            "Starting SABRE layout search for %s qubits "
            "(optimization_level=%s, max_iterations=%s, swap_trials=%s, layout_trials=%s).",
            num_qubits,
            optimization_level,
            max_iterations,
            swap_trials,
            layout_trials,
        )

    if layout_trials > SLOW_LAYOUT_TRIAL_THRESHOLD and logger is not None:
        logger.warning(
            "SABRE layout search is configured with %s layout_trials. "
            "IBM Quantum submission starts only after this search finishes, so large values "
            "can make the closed loop look stuck. For troubleshooting, start around "
            "256-1024 layout trials.",
            layout_trials,
        )

    test_pm = generate_preset_pass_manager(
        optimization_level=optimization_level,
        seed_transpiler=TRANSPILE_SEED,
        target=target,
    )
    if use_pre_init:
        from ffsim.qiskit import PRE_INIT

        test_pm.pre_init = PRE_INIT

    if "rzz" in target.operation_names:
        test_pm.post_optimization = PassManager(
            [
                FoldRzzAngle(),
                Optimize1qGatesDecomposition(target=target),
                RemoveIdentityEquivalent(target=target),
            ]
        )

    test_pm.layout = PassManager(
        [
            BarrierBeforeFinalMeasurements(
                label="qiskit.transpiler.internal.routing.protection.barrier",
            ),
            SabreLayout(
                coupling_map=coupling_map,
                seed=TRANSPILE_SEED,
                max_iterations=max_iterations,
                layout_trials=layout_trials,
                swap_trials=swap_trials,
            ),
            ConditionalController(
                tasks=[
                    FullAncillaAllocation(coupling_map=coupling_map),
                    EnlargeWithAncilla(),
                    ApplyLayout(),
                ],
                condition=lambda propset: propset["final_layout"] is None,
            ),
        ]
    )

    start = perf_counter()
    isa_trial = test_pm.run(test_circuit)
    elapsed = perf_counter() - start

    depth = isa_trial.depth(lambda inst: inst.operation.name not in ("rz", "barrier", "measure"))

    if logger is not None:
        logger.info(
            "Completed SABRE layout search in %.2fs. Circuit depth = %s. Instruction counts = %s",
            elapsed,
            depth,
            dict(isa_trial.count_ops()),
        )

    final_sabre_layout = isa_trial.layout.initial_virtual_layout(filter_ancillas=True)

    layout_info = []
    for vi, pi in final_sabre_layout.get_virtual_bits().items():
        qubit = {
            "v_index": test_circuit.qubits.index(vi),
            "p_index": pi,
            "t1": None,
            "t2": None,
        }
        try:
            qubit_prop = target.qubit_properties[pi]
            qubit["t1"] = qubit_prop.t1 * 1e6
            qubit["t2"] = qubit_prop.t2 * 1e6
        except (IndexError, TypeError):
            pass

        layout_info.append(qubit)

    return final_sabre_layout, layout_info
