"""SQD transpiler hooks for QrmiRuntime: preset pass manager with a massive SABRE layout search.

``QrmiRuntime`` runs ``prepare`` on the workflow host and calls
``transpile(circuit, target, config)`` once per circuit with the actual Target: in the
workflow process for ``scheduler="iqp"``, and on the compute node for
``scheduler="slurm"``, where this package must be installed.

The ffsim ``PRE_INIT`` stage runs in ``prepare``. Its ``MergeOrbitalRotations`` pass needs
the ffsim gate objects of the LUCJ circuit, which QPY transport to a compute node turns
into plain gates. ``transpile`` runs the rest of the former ``transpile_circuit`` pass
manager.
"""

from __future__ import annotations

from ffsim.qiskit import PRE_INIT
from qcsc_prefect.integrations.qiskit.qrmi import PreparedCircuit
from qiskit.passmanager import ConditionalController
from qiskit.transpiler import generate_preset_pass_manager
from qiskit.transpiler.passes import (
    ApplyLayout,
    BarrierBeforeFinalMeasurements,
    EnlargeWithAncilla,
    FullAncillaAllocation,
    Optimize1qGatesDecomposition,
    RemoveIdentityEquivalent,
    SabreLayout,
)
from qiskit.transpiler.passmanager import PassManager
from qiskit_ibm_runtime.transpiler.passes import FoldRzzAngle

TRANSPILER = "sqd_dice.transpilation:transpile"
DEFAULTS = {
    "optimization_level": 3,
    "seed_transpiler": None,
    "sabre_max_iterations": 8,
    "sabre_swap_trials": 10,
    "sabre_layout_trials": 10000,
}


def transpile_config(
    *,
    optimization_level: int,
    seed_transpiler: int | None,
    sabre_max_iterations: int,
    sabre_swap_trials: int,
    sabre_layout_trials: int,
) -> dict:
    """JSON-serializable configuration for ``transpile``."""
    config = {
        "optimization_level": optimization_level,
        "seed_transpiler": seed_transpiler,
        "sabre_max_iterations": sabre_max_iterations,
        "sabre_swap_trials": sabre_swap_trials,
        "sabre_layout_trials": sabre_layout_trials,
    }
    _validate(config)
    return config


def _validate(config: dict) -> None:
    unknown = config.keys() - DEFAULTS.keys()
    if unknown:
        raise ValueError(f"Unsupported SQD transpile options: {sorted(map(str, unknown))}")
    level = config.get("optimization_level", DEFAULTS["optimization_level"])
    if type(level) is not int or not 0 <= level <= 3:
        raise ValueError("optimization_level must be an integer from 0 to 3")
    seed = config.get("seed_transpiler")
    if seed is not None and (type(seed) is not int or seed < 0):
        raise ValueError("seed_transpiler must be a nonnegative integer or None")
    for name in ("sabre_max_iterations", "sabre_swap_trials", "sabre_layout_trials"):
        value = config.get(name, DEFAULTS[name])
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")


def prepare(circuit) -> PreparedCircuit:
    """QrmiRuntime ``prepare`` hook: run the ffsim pre-init stage on the workflow host."""
    return PreparedCircuit(PRE_INIT.run(circuit))


def build_pass_manager(target, config: dict):
    """Preset pass manager for ``target`` with the SQD layout stage, without pre-init."""
    _validate(config)
    options = {**DEFAULTS, **config}
    seed = options["seed_transpiler"]
    coupling_map = target.build_coupling_map()
    passmanager = generate_preset_pass_manager(
        optimization_level=options["optimization_level"],
        seed_transpiler=seed,
        target=target,
    )
    passmanager.pre_init = None  # Run by prepare on the workflow host.
    passmanager.layout = PassManager(
        [
            BarrierBeforeFinalMeasurements(
                label="qiskit.transpiler.internal.routing.protection.barrier",
            ),
            SabreLayout(
                coupling_map=coupling_map,
                seed=seed,
                max_iterations=options["sabre_max_iterations"],
                layout_trials=options["sabre_layout_trials"],
                swap_trials=options["sabre_swap_trials"],
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
    if "rzz" in target.operation_names:
        passmanager.post_optimization = PassManager(
            [
                FoldRzzAngle(),
                Optimize1qGatesDecomposition(target=target),  # Cancel added local gates
                RemoveIdentityEquivalent(target=target),  # Remove GlobalPhaseGate
            ]
        )
    return passmanager


def transpile(circuit, target, config):
    """QrmiRuntime transpiler hook."""
    return build_pass_manager(target, config).run(circuit)
