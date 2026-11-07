"""QPU topology of the LUCJ ansatz: interaction pairs and how ``auto`` is resolved.

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
        ValueError: For any other topology. Resolve ``auto`` with
            ``resolve_topology`` first.
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
    """The concrete QPU topology of a run, needed before Optimize.

    A topology set in the circuit parameters is used as is. ``auto`` classifies
    ``coupling``, the backend ``Target`` or ``CouplingMap`` if the workflow host
    holds one. Otherwise it falls back to heavy-hex with a warning, as when QRMI
    reads the target only when it compiles, after Optimize. A random quantum
    source runs no circuit and gets heavy-hex without a warning.

    Raises:
        ValueError: If ``coupling`` is neither heavy-hex nor square.
    """
    logger = get_run_logger()
    topology = circuit_params.qpu_topology
    if topology != "auto":
        logger.info("QPU topology: %s, as set in QPU Topology.", topology)
        return topology
    if sampling_params.quantum_source == "random":
        return "heavy-hex"
    if coupling is not None:
        topology = classify_coupling_map(coupling)
        logger.info("QPU topology: %s, read from the backend coupling map.", topology)
        return topology
    logger.warning(
        "QPU Topology is auto, but the workflow host cannot read the backend coupling map "
        "before Optimize. Using heavy-hex (IBM Heron); set Circuit > QPU Topology to square "
        "for a square-lattice QPU (IBM Nighthawk)."
    )
    return "heavy-hex"
