"""The example FlowParameters files under algorithms/open_loop/sqd/default_params."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("pydantic")

from sqd_common.flow_params import FlowParameters  # noqa: E402

SQD = Path(__file__).resolve().parents[1]
PARAMS = sorted((SQD / "default_params").glob("*.json"))
N2_FCIDUMP = SQD / "data" / "n2_631g_fc.fcidump"


def test_every_example_is_present():
    assert {p.name for p in PARAMS} == {
        "n2.json",
        "n2_dice.json",
        "n2_random.json",
        "n2_reload.json",
        "n2_sbd.json",
    }


@pytest.mark.parametrize("path", PARAMS, ids=lambda p: p.name)
def test_example_validates_and_finds_the_fcidump_from_any_directory(path, monkeypatch, tmp_path):
    # A deployment serves from the package directory, not the repository root.
    monkeypatch.chdir(tmp_path)
    params = FlowParameters.model_validate(json.loads(path.read_text()))
    assert Path(params.molecule.fcidump_file) == N2_FCIDUMP
    # The solver comes from the Advanced SQD Settings block that sqd-create-blocks saves;
    # the SBD example names its block as a Solver Block Override.
    expected_solver = {"n2_sbd.json": "sbd_solver_job/sqd-sbd-solver"}.get(path.name)
    assert params.solver_block_ref == expected_solver
    # Only the DICE example sets a spin penalty; the others keep the default, null.
    assert params.molecule.spin_sq == (0.0 if path.name == "n2_dice.json" else None)
    assert params.circuit.compress_maxiter == 50


def test_reuse_examples_select_one_source():
    by_name = {p.name: FlowParameters.model_validate_json(p.read_text()) for p in PARAMS}
    assert by_name["n2_random.json"].sampling.quantum_source == "random"
    assert by_name["n2_sbd.json"].sampling.quantum_source == "random"
    assert by_name["n2.json"].sampling.runtime_block_name == "qrmi-runtime"
    assert by_name["n2_reload.json"].sampling.qiskit_runtime_job_id


def test_n2_fcidump_is_the_frozen_core_active_space():
    header = N2_FCIDUMP.read_text().splitlines()[0]
    assert "NORB=  16" in header and "NELEC=10" in header and "MS2=0" in header


def test_absolute_and_missing_paths_are_kept(tmp_path):
    absolute = tmp_path / "x.fcidump"
    for value in (str(absolute), "does/not/exist.fcidump"):
        params = FlowParameters.model_validate(
            {
                "molecule": {"fcidump_file": value},
                "sqd": {"subspace_dim": 1, "num_batches": 1, "max_iterations": 1},
            }
        )
        assert params.molecule.fcidump_file == value
