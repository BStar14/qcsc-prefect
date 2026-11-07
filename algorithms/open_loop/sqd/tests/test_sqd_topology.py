"""QPU topology of the LUCJ ansatz (``sqd_common.topology``), without ffsim."""

from __future__ import annotations

import logging

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("prefect")

from qiskit.providers.fake_provider import GenericBackendV2  # noqa: E402
from qiskit.transpiler import CouplingMap, Target  # noqa: E402
from sqd_common import topology  # noqa: E402
from sqd_common.flow_params import CircuitParameters, SamplingParameters  # noqa: E402

LOGGER = logging.getLogger("sqd_common.topology.test")


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(topology, "get_run_logger", lambda: LOGGER)


# ---------------------------------------------------------------- interaction pairs


def test_heavy_hex_pairs_are_the_default():
    aa, ab = topology.interaction_pairs(10)
    assert aa == [(p, p + 1) for p in range(9)]
    assert ab == [(0, 0), (4, 4), (8, 8)]
    assert topology.interaction_pairs(10, "heavy-hex") == (aa, ab)
    # At most MAX_AB_PAIRS alpha-beta pairs, from orbital 0.
    assert topology.MAX_AB_PAIRS == 5
    assert topology.interaction_pairs(40)[1] == [(p, p) for p in range(0, 20, 4)]


def test_square_pairs_couple_every_orbital_without_a_cap():
    aa, ab = topology.interaction_pairs(10, "square")
    assert aa == [(p, p + 1) for p in range(9)]
    assert ab == [(p, p) for p in range(10)]
    assert len(topology.interaction_pairs(40, "square")[1]) == 40


@pytest.mark.parametrize("name", ["auto", "grid", "Heavy-Hex"])
def test_unknown_topology_is_rejected(name):
    with pytest.raises(ValueError, match="Unknown QPU topology"):
        topology.interaction_pairs(4, name)


# ---------------------------------------------------------------- classification


@pytest.mark.parametrize(
    "coupling",
    [
        CouplingMap.from_heavy_hex(3),
        CouplingMap.from_heavy_hex(5, bidirectional=False),
        CouplingMap.from_hexagonal_lattice(2, 3),
        CouplingMap.from_line(4),
    ],
    ids=["heavy-hex-3", "heavy-hex-5-directed", "hexagonal", "line"],
)
def test_degree_at_most_three_is_heavy_hex(coupling):
    assert topology.classify_coupling_map(coupling) == "heavy-hex"


@pytest.mark.parametrize(
    "coupling",
    [CouplingMap.from_grid(4, 5), CouplingMap.from_heavy_square(3)],
    ids=["grid", "heavy-square"],
)
def test_degree_four_is_square(coupling):
    assert topology.classify_coupling_map(coupling) == "square"


def test_a_target_and_an_edge_list_are_classified():
    heavy_hex = CouplingMap.from_heavy_hex(3)
    target = GenericBackendV2(
        num_qubits=heavy_hex.size(), coupling_map=heavy_hex.get_edges(), seed=1
    ).target
    assert topology.classify_coupling_map(target) == "heavy-hex"
    # Direction and duplicates do not count: qubit 0 has neighbours 1 to 4.
    edges = [(0, 1), (1, 0), (0, 2), (2, 0), (3, 0), (0, 4), (0, 4)]
    assert topology.classify_coupling_map(edges) == "square"


@pytest.mark.parametrize(
    "coupling, degree",
    [
        (CouplingMap.from_full(6), 5),
        ([(0, q) for q in range(1, 7)], 6),
        (CouplingMap(), 0),
        (Target(num_qubits=3), 0),
    ],
    ids=["all-to-all", "star", "empty", "target-without-couplings"],
)
def test_irregular_coupling_map_asks_for_the_topology(coupling, degree):
    with pytest.raises(ValueError, match="QPU Topology") as error:
        topology.classify_coupling_map(coupling)
    if degree:
        assert f"maximum qubit degree is {degree}" in str(error.value)


def test_target_with_all_to_all_connectivity_asks_for_the_topology():
    class AllToAll:
        def build_coupling_map(self):
            return None

    with pytest.raises(ValueError, match="all-to-all"):
        topology.classify_coupling_map(AllToAll())


# ---------------------------------------------------------------- resolution


def _resolve(caplog, qpu_topology="heavy-hex", coupling=None, **sampling):
    with caplog.at_level(logging.INFO, logger=LOGGER.name):
        return topology.resolve_topology(
            CircuitParameters(qpu_topology=qpu_topology),
            SamplingParameters(**sampling),
            coupling,
        )


def _warnings(caplog):
    return [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]


def test_the_default_topology_is_heavy_hex(caplog):
    assert CircuitParameters().qpu_topology == "heavy-hex"
    assert _resolve(caplog) == "heavy-hex"
    assert "QPU topology: heavy-hex." in caplog.text
    assert _warnings(caplog) == []


@pytest.mark.parametrize("name", ["heavy-hex", "square"])
def test_a_set_topology_is_used_as_is(caplog, name):
    assert _resolve(caplog, name) == name
    assert _warnings(caplog) == []


@pytest.mark.parametrize(
    "coupling, name",
    [(CouplingMap.from_heavy_hex(3), "heavy-hex"), (CouplingMap.from_grid(3, 3), "square")],
)
def test_a_matching_backend_is_accepted(caplog, coupling, name):
    assert _resolve(caplog, name, coupling=coupling) == name


def test_a_backend_of_the_other_topology_fails_before_optimize(caplog):
    with pytest.raises(ValueError, match="backend coupling map is square.*QPU Topology to square"):
        _resolve(caplog, "heavy-hex", coupling=CouplingMap.from_grid(3, 3))


def test_a_random_source_ignores_the_backend(caplog):
    coupling = CouplingMap.from_grid(3, 3)
    assert _resolve(caplog, "heavy-hex", coupling=coupling, quantum_source="random") == "heavy-hex"


def test_an_irregular_backend_raises(caplog):
    with pytest.raises(ValueError, match="QPU Topology"):
        _resolve(caplog, coupling=CouplingMap.from_full(6))


# ---------------------------------------------------------------- run form


def test_qpu_topology_keeps_the_circuit_form_positions_contiguous():
    properties = CircuitParameters.model_json_schema()["properties"]
    positions = sorted(field["position"] for field in properties.values())
    assert positions == list(range(len(properties)))
    assert properties["qpu_topology"]["position"] == 4
    assert properties["qpu_topology"]["enum"] == list(topology.TOPOLOGIES)
    assert properties["qpu_topology"]["default"] == "heavy-hex"


def test_removed_max_ab_pairs_points_to_qpu_topology():
    with pytest.raises(ValueError, match=r"max_ab_pairs was removed.*QPU Topology"):
        CircuitParameters(max_ab_pairs=2)
