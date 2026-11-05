"""SBD transpiler hooks for QrmiRuntime on the real-device path.

``QrmiRuntime`` runs ``prepare`` on the workflow host and calls
``transpile(circuit, target, config)`` with the actual Target: in the workflow process
for ``scheduler="iqp"``, and on the compute node for ``scheduler="slurm"``, where this
package must be installed.

The ffsim ``PRE_INIT`` stage runs in ``prepare``. It needs the ffsim gate objects, which
QPY transport to a compute node turns into plain gates. The QFw-SLURM path does the same
before sending its circuit. ``transpile`` then searches the SABRE layout and transpiles
with that layout, as ``find_optimal_layout`` and ``transpile_circuit`` did.
"""

from __future__ import annotations

from ffsim.qiskit import PRE_INIT
from qcsc_prefect.integrations.qiskit.qrmi import PreparedCircuit

from .transpile_core import _find_optimal_layout, _transpile_circuit

TRANSPILER = "sbd.transpilation:transpile"
OPTIONS = ("optimization_level", "sabre_max_iterations", "sabre_swap_trials")
OPTIONS += ("sabre_layout_trials",)


def transpile_config(circuit_params) -> dict:
    """JSON-serializable ``transpile`` configuration from ``CircuitParameters``."""
    config = {name: getattr(circuit_params, name) for name in OPTIONS}
    _validate(config)
    return config


def _validate(config: dict) -> None:
    if set(config) != set(OPTIONS):
        raise ValueError(f"SBD transpile config must have exactly the keys {list(OPTIONS)}")
    level = config["optimization_level"]
    if type(level) is not int or not 0 <= level <= 3:
        raise ValueError("optimization_level must be an integer from 0 to 3")
    for name in OPTIONS[1:]:
        if type(config[name]) is not int or config[name] < 1:
            raise ValueError(f"{name} must be a positive integer")


def prepare(circuit) -> PreparedCircuit:
    """QrmiRuntime ``prepare`` hook: run the ffsim pre-init stage on the workflow host."""
    return PreparedCircuit(PRE_INIT.run(circuit))


def transpile(circuit, target, config):
    """QrmiRuntime transpiler hook: SABRE layout search, then transpile with that layout."""
    _validate(config)
    level = config["optimization_level"]
    layout, _ = _find_optimal_layout(
        test_circuit=circuit,
        target=target,
        optimization_level=level,
        max_iterations=config["sabre_max_iterations"],
        swap_trials=config["sabre_swap_trials"],
        layout_trials=config["sabre_layout_trials"],
        use_pre_init=False,
    )
    return _transpile_circuit(
        circuit=circuit,
        target=target,
        layout=layout,
        optimization_level=level,
        use_pre_init=False,
    )
