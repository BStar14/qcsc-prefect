"""QPU topology of the LUCJ ansatz: its interaction pairs and the topology of a backend.

This module imports neither ffsim nor Qiskit. It reads a Qiskit ``Target`` or
``CouplingMap`` through their methods only.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any, Literal, get_args

from prefect import get_run_logger

from sqd_common.flow_params import CircuitParameters, SamplingParameters

Topology = Literal["heavy-hex", "square"]
TOPOLOGIES: tuple[str, ...] = get_args(Topology)
# Alpha-beta pairs kept on a heavy-hex (Heron) lattice.
MAX_AB_PAIRS = 5


def interaction_pairs(
    norb: int, topology: Topology = "heavy-hex"
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """LUCJ interactions that ``topology`` implements without SWAP gates.

    Alpha-alpha (and beta-beta) pairs couple neighbouring orbitals. Alpha-beta
    pairs ``(p, p)`` depend on how the two spin chains meet:

    - ``heavy-hex`` (IBM Heron): through an ancilla qubit, every 4th orbital. At
      most ``MAX_AB_PAIRS`` pairs are kept, starting from orbital 0.
    - ``square`` (IBM Nighthawk): directly, on every orbital.

    Raises:
        ValueError: For any other topology.
    """
    aa_indices = [(p, p + 1) for p in range(norb - 1)]
    if topology == "heavy-hex":
        ab_indices = [(p, p) for p in range(0, norb, 4)][:MAX_AB_PAIRS]
    elif topology == "square":
        ab_indices = [(p, p) for p in range(norb)]
    else:
        raise ValueError(f"Unknown QPU topology {topology!r}; expected one of {TOPOLOGIES}.")
    return aa_indices, ab_indices


def _coupling_edges(coupling: Any) -> Iterable[Iterable[int]]:
    build_coupling_map = getattr(coupling, "build_coupling_map", None)
    if build_coupling_map is not None:  # A Target.
        coupling = build_coupling_map()
        if coupling is None:
            raise ValueError(
                "The backend target has no coupling map (all-to-all connectivity). "
                "Set Circuit > QPU Topology to heavy-hex or square."
            )
    get_edges = getattr(coupling, "get_edges", None)  # A CouplingMap.
    return get_edges() if get_edges is not None else coupling


def classify_coupling_map(coupling: Any) -> Topology:
    """Tell heavy-hex from square by the highest qubit degree of the coupling graph.

    Args:
        coupling: A ``Target``, a ``CouplingMap`` or an iterable of qubit pairs.
            Edge direction and duplicate edges do not matter.

    Returns:
        ``heavy-hex`` for a maximum degree of 2 or 3, ``square`` for 4.

    Raises:
        ValueError: For a graph without couplings or with a qubit of degree 5 or
            more, which is neither pattern.
    """
    edges = {frozenset(edge) for edge in _coupling_edges(coupling)}
    degree = Counter(qubit for edge in edges if len(edge) == 2 for qubit in edge)
    max_degree = max(degree.values(), default=0)
    if 0 < max_degree <= 3:
        return "heavy-hex"
    if max_degree == 4:
        return "square"
    raise ValueError(
        f"Cannot tell the QPU topology from a coupling map whose maximum qubit degree is "
        f"{max_degree}: heavy-hex has at most 3 and square 4. Set Circuit > QPU Topology "
        "to heavy-hex or square."
    )


def resolve_topology(
    circuit_params: CircuitParameters,
    sampling_params: SamplingParameters,
    coupling: Any = None,
) -> Topology:
    """The QPU topology the LUCJ interaction pairs of a run are chosen for.

    This is the QPU Topology of the run. With ``coupling``, the backend ``Target``
    or ``CouplingMap``, a run whose QPU Topology does not match the backend fails
    before Optimize. The workflow host holds no backend before Optimize today, as
    QRMI reads the target only when it compiles, so the flow passes none.

    Raises:
        ValueError: If ``coupling`` is not of the topology set, or neither
            heavy-hex nor square.
    """
    topology = circuit_params.qpu_topology
    if coupling is not None and sampling_params.quantum_source != "random":
        backend_topology = classify_coupling_map(coupling)
        if backend_topology != topology:
            raise ValueError(
                f"QPU Topology is {topology}, but the backend coupling map is {backend_topology}. "
                f"Set Circuit > QPU Topology to {backend_topology}."
            )
    get_run_logger().info("QPU topology: %s.", topology)
    return topology
