"""Fixtures for the SBD tests; helpers are in sbd_helpers."""

import pytest
from sbd_helpers import lucj_circuit, ucj_parameters


@pytest.fixture
def circuits():
    return [lucj_circuit(parameter) for parameter in ucj_parameters(3)]
