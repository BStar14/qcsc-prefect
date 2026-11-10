"""The closed-loop flow runs to the end in random mode with SBD or DICE replaced by fakes.

The flow body runs without a Prefect server: its tasks are called directly, walkers run one
after another, and artifacts are recorded instead of uploaded.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from qcsc_prefect_dice import DiceSHCISolverJob
from qcsc_prefect_sbd import SBDResult, SBDSolverJob, check_sbd_input
from qcsc_workflow_utility import chem
from qiskit_addon_sqd.fermion import SCIResult
from sbd import main, sqd
from sbd.flow_params import FlowParameters

# N2 (18 orbitals, 7 + 7 electrons): large enough that carryover and new determinants
# both fit in the subspace of every generation.
N2 = str(Path(__file__).resolve().parents[1] / "data/fcidump_N2_MO.txt")
HARTREE_FOCK = (1 << 7) - 1


class Done:
    """A finished PrefectFuture-like value."""

    def __init__(self, value):
        self.value = value

    def result(self):
        return self.value


class DoneList(list):
    """A PrefectFutureList-like list of finished futures."""

    def result(self):
        return [future.result() for future in self]


@pytest.fixture
def flow(monkeypatch, tmp_path):
    logger = SimpleNamespace(
        info=lambda *a, **k: None,
        debug=lambda *a, **k: None,
        exception=lambda *a, **k: None,
        warning=lambda *a, **k: None,
    )
    telemetry = []
    for module in (main, sqd, chem):
        monkeypatch.setattr(module, "get_run_logger", lambda: logger)
    monkeypatch.setattr(main, "create_table_artifact", lambda **kwargs: None)
    monkeypatch.setattr(
        main, "extend_table_artifact", lambda artifact_key, new_table: telemetry.extend(new_table)
    )
    monkeypatch.setattr(
        main,
        "compute_molecular_integrals_from_fcidump",
        chem.compute_molecular_integrals_from_fcidump.fn,
    )
    for name in ("differential_evolution_trial", "mutation_and_crossover", "selection"):
        monkeypatch.setattr(main, name, getattr(main, name).fn)
    monkeypatch.setattr(main, "initialize_ucj_parameters", main.initialize_ucj_parameters.fn)
    monkeypatch.setattr(main, "PrefectFutureList", DoneList)
    monkeypatch.setattr(
        main, "walker_sqd", SimpleNamespace(submit=lambda **kw: Done(sqd.walker_sqd.fn(**kw)))
    )
    # Few shots keep configuration recovery fast; the default is 100_000 per walker.
    monkeypatch.setattr(
        sqd.Variable, "get", staticmethod(lambda name, default=None: {"params": {"shots": 2000}})
    )
    monkeypatch.setattr(sqd, "save_ndarray", lambda **arrays: f"file://{tmp_path}/data.npz")
    for name in ("recover_configurations", "postselect_bitstrings", "subsample_close_shell"):
        monkeypatch.setattr(sqd, name, getattr(sqd, name).fn)

    def run(solver_block_ref, iterations=2):
        parameters = FlowParameters(
            fcidump=N2,
            sqd_dim=400,
            quantum_source="random",
            solver_block_ref=solver_block_ref,
            # randomization_factor=0: with ffsim 0.0.85, the default random perturbation of
            # t2 breaks its pair symmetry and the LUCJ parameter conversion rejects it, for
            # every solver (also before solver selection was added).
            de_params={"num_walkers": 4, "iterations": iterations, "randomization_factor": 0.0},
        )
        return main.riken_sqd_de.fn(parameters=parameters)

    run.telemetry = telemetry
    return run


def fake_solver(monkeypatch, cls, result):
    inputs = []

    class Fake:
        solver_mode = "fake"

        async def run(self, **kwargs):
            inputs.append(kwargs)
            return result(kwargs)

    monkeypatch.setattr(cls, "load", classmethod(lambda c, name: Fake()))
    return inputs


def test_closed_loop_with_sbd_passes_carryover_to_the_next_generation(flow, monkeypatch):
    def result(kwargs):
        check_sbd_input(ci_strings=kwargs["ci_strings"], nelec=kwargs["nelec"], spin_sq=None)
        occupancy = np.full(kwargs["norb"], 0.5)
        return SBDResult(
            energy=-1.0 - 0.1 * len(kwargs["ci_strings"][0]),
            orbital_occupancies=(occupancy, occupancy),
            # A non-Hartree-Fock determinant, so that it reaches the next generation.
            carryover_bitstrings=np.array([[False] * 10 + [True] * 7 + [False]]),
        )

    inputs = fake_solver(monkeypatch, SBDSolverJob, result)
    energy = flow("sbd_solver_job/davidson-solver")

    assert np.isfinite(energy)
    assert len(inputs) == 8  # four walkers in each of two generations
    carried = int("0" * 10 + "1" * 7 + "0", 2)
    first, second = inputs[:4], inputs[4:]
    assert all(carried not in call["ci_strings"][0] for call in first)
    assert all(carried in call["ci_strings"][0] for call in second)
    assert all(call["ci_strings"][0][0] == HARTREE_FOCK for call in inputs)
    assert {(row["solver"], row["carryover"]) for row in flow.telemetry} == {("SBD", "sbd")}


def test_closed_loop_with_dice_runs_without_carryover(flow, monkeypatch):
    def result(kwargs):
        occupancy = np.full(kwargs["norb"], 0.5)
        return SCIResult(
            energy=-1.0 - 0.1 * len(kwargs["ci_strings"][0]),
            sci_state=None,
            orbital_occupancies=(occupancy, occupancy),
        )

    inputs = fake_solver(monkeypatch, DiceSHCISolverJob, result)
    energy = flow("dice_shci_solver_job/dice-solver")

    assert np.isfinite(energy)
    assert len(inputs) == 8
    # Every walker starts with the Hartree-Fock determinant and no carryover.
    assert all(call["ci_strings"][0][0] == HARTREE_FOCK for call in inputs)
    assert len(flow.telemetry) == 8
    assert {(row["solver"], row["carryover"]) for row in flow.telemetry} == {("DICE", "none")}


def test_closed_loop_rejects_an_unknown_solver_before_any_walker(flow, monkeypatch):
    with pytest.raises(ValueError, match="solver_block_ref"):
        flow("other_solver/x")
    assert flow.telemetry == []
