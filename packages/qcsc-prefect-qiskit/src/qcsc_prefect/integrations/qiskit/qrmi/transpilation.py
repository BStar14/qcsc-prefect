"""JSON-safe options for compilation against the compute-node Target."""

from __future__ import annotations

PRESET_OPTIONS = frozenset(
    {
        "optimization_level",
        "initial_layout",
        "seed_transpiler",
        "layout_method",
        "routing_method",
        "translation_method",
        "scheduling_method",
        "approximation_degree",
        "unitary_synthesis_method",
        "unitary_synthesis_plugin_config",
        "init_method",
        "optimization_method",
        "qubits_initially_zero",
    }
)


def preset_pass_manager_options(config: dict, *, num_qubits: int) -> dict:
    """Validate portable preset options without acquiring a Target on the host.

    Qiskit validates target-dependent constraints and plugin settings on the node.
    Target/backend overrides and Python objects such as Layout are not portable.
    """
    unknown = config.keys() - PRESET_OPTIONS
    if unknown:
        raise ValueError(
            f"Unsupported preset transpile_config options: {sorted(map(str, unknown))}"
        )
    options = {"optimization_level": 1, **config}
    level = options["optimization_level"]
    if type(level) is not int or not 0 <= level <= 3:
        raise ValueError("optimization_level must be an integer from 0 to 3")
    seed = options.get("seed_transpiler")
    if seed is not None and (type(seed) is not int or seed < 0):
        raise ValueError("seed_transpiler must be a nonnegative integer or None")
    layout = options.get("initial_layout")
    if layout is not None:
        if not isinstance(layout, list) or any(type(q) is not int or q < 0 for q in layout):
            raise ValueError("initial_layout must be a list of nonnegative physical qubit indices")
        if len(set(layout)) != len(layout):
            raise ValueError("initial_layout must not contain duplicate physical qubit indices")
        if len(layout) != num_qubits:
            raise ValueError(
                "initial_layout must have exactly one entry per circuit qubit "
                f"(expected {num_qubits}, got {len(layout)})"
            )
    for name in (
        "layout_method",
        "routing_method",
        "translation_method",
        "scheduling_method",
        "init_method",
        "optimization_method",
        "unitary_synthesis_method",
    ):
        method = options.get(name)
        if method is not None and not isinstance(method, str):
            raise ValueError(f"{name} must be a string or None")
    degree = options.get("approximation_degree")
    if degree is not None and (type(degree) not in (int, float) or not 0 <= degree <= 1):
        raise ValueError("approximation_degree must be a number from 0 to 1 or None")
    if "qubits_initially_zero" in options and type(options["qubits_initially_zero"]) is not bool:
        raise ValueError("qubits_initially_zero must be a bool")
    plugin_config = options.get("unitary_synthesis_plugin_config")
    if plugin_config is not None and not isinstance(plugin_config, dict):
        raise ValueError("unitary_synthesis_plugin_config must be a dict or None")
    return options
