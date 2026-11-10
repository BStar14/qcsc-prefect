from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import prefect.blocks.core
import pytest
from prefect.blocks.core import Block
from prefect.client.schemas.objects import BlockDocument, BlockType
from qcsc_prefect_sbd import SBDResult, SBDSolverJob
from qcsc_prefect_sbd import solver_job as mod

# Saved document data of a block created before the move, when SBDSolverJob was defined in
# algorithms/sbd/sbd/solver_job.py. The block type name, slug and fields are unchanged.
SAVED_DATA = {
    "root_dir": "/work/sbd",
    "command_block_name": "cmd-sbd-diag",
    "execution_profile_block_name": "exec-sbd-mpi",
    "hpc_profile_block_name": "hpc-miyabi-sbd",
    "script_filename": "sbd_solver.pbs",
    "metrics_artifact_key": "miyabi-sbd-metrics",
    "timeout_seconds": 7200.0,
    "user_args": ["--x", "1"],
    "task_comm_size": 1,
    "adet_comm_size": 1,
    "bdet_comm_size": 1,
    "block": 4,
    "iteration": 1,
    "tolerance": 0.01,
    "carryover_ratio": 0.1,
    "solver_mode": "gpu",
}
# Schema checksum of the current class, with the carryover_type field. A change here means
# saving the block registers a new schema version.
SCHEMA_CHECKSUM = "sha256:f5f57489e6d89ed3dbcf3bae556df81d79a523c1"


@pytest.fixture(autouse=True)
def quiet_logger(monkeypatch):
    logger = SimpleNamespace(debug=lambda *a, **k: None, info=lambda *a, **k: None)
    monkeypatch.setattr(mod, "get_run_logger", lambda: logger)


def test_block_type_is_unchanged_by_the_move():
    assert SBDSolverJob._block_type_slug == "sbd_solver_job"
    assert SBDSolverJob.get_block_type_slug() == "sbd-solver-job"
    assert SBDSolverJob.get_block_type_name() == "SBD Solver Job"
    schema = SBDSolverJob._to_block_schema(block_type_id=uuid.uuid4())
    assert schema.checksum.startswith(SCHEMA_CHECKSUM)


@pytest.mark.parametrize("loader", [SBDSolverJob, Block])
def test_block_saved_before_the_move_loads(monkeypatch, loader):
    # SBDSolverJob.load(name) restores with the class; Block.load("sbd-solver-job/name")
    # looks the class up from the saved schema's block type slug.
    # Restoring a block emits a Prefect "loaded" event, which would start a temporary
    # server. Record it instead.
    events = []
    monkeypatch.setattr(prefect.blocks.core, "emit_event", lambda **event: events.append(event))
    block_type = BlockType(
        id=uuid.uuid4(), name="SBD Solver Job", slug="sbd-solver-job", created=None
    )
    schema = SBDSolverJob._to_block_schema(block_type_id=block_type.id)
    schema = schema.model_copy(update={"id": uuid.uuid4(), "block_type": block_type})
    document = BlockDocument(
        id=uuid.uuid4(),
        name="davidson-solver",
        data=dict(SAVED_DATA),
        block_schema_id=schema.id,
        block_schema=schema,
        block_type_id=block_type.id,
        block_type_name=block_type.name,
        block_type=block_type,
    )

    solver = loader._from_block_document(document)

    assert type(solver) is SBDSolverJob
    assert {name: getattr(solver, name) for name in SAVED_DATA} == SAVED_DATA
    # Saved before carryover_type existed: the default keeps the carryover the flow uses.
    assert solver.carryover_type == 1
    assert [event["event"] for event in events] == ["prefect.block.sbd-solver-job.loaded"]


def test_prep_files_writes_fcidump_and_big_endian_alpha_dets(tmp_path: Path):
    norb = 10  # Two bytes per determinant.
    mod._prep_files(
        work_dir=tmp_path,
        ci_strings=(np.array([0b11, 0b1000000001]), np.array([0b11])),
        one_body_tensor=np.eye(norb),
        two_body_tensor=np.zeros((norb,) * 4),
        norb=norb,
        nelec=(2, 2),
    )

    assert (tmp_path / "fcidump.txt").read_text().startswith(" &FCI NORB=")
    assert (tmp_path / "AlphaDets.bin").read_bytes() == bytes([0, 3, 2, 1])


@pytest.mark.parametrize("alpha", [-1, 1 << 4])
def test_prep_files_rejects_out_of_range_determinants(tmp_path: Path, alpha):
    with pytest.raises(ValueError, match="Alpha determinant"):
        mod._prep_files(
            work_dir=tmp_path,
            ci_strings=(np.array([alpha]), np.array([alpha])),
            one_body_tensor=np.eye(4),
            two_body_tensor=np.zeros((4,) * 4),
            norb=4,
            nelec=(1, 1),
        )


def test_prep_files_writes_beta_dets_only_when_the_sets_differ(tmp_path: Path):
    norb = 4
    common = dict(one_body_tensor=np.eye(norb), two_body_tensor=np.zeros((norb,) * 4), norb=norb)
    same = tmp_path / "same"
    same.mkdir()
    mod._prep_files(
        work_dir=same, ci_strings=(np.array([3, 5]), np.array([3, 5])), nelec=(2, 2), **common
    )
    assert not (same / "BetaDets.bin").exists()

    rohf = tmp_path / "rohf"
    rohf.mkdir()
    mod._prep_files(
        work_dir=rohf, ci_strings=(np.array([3, 5]), np.array([1, 2, 8])), nelec=(2, 1), **common
    )
    assert (rohf / "AlphaDets.bin").read_bytes() == bytes([3, 5])
    assert (rohf / "BetaDets.bin").read_bytes() == bytes([1, 2, 8])
    header = (rohf / "fcidump.txt").read_text().splitlines()[0]
    assert "NELEC= 3" in header and "MS2=1" in header


@pytest.mark.parametrize(
    "ci_strings,nelec,message",
    [
        ((np.array([3, 1]), np.array([3, 1])), (2, 2), "1 of 2 Alpha determinants do not have 2"),
        ((np.array([3, 5]), np.array([1, 3])), (2, 1), "1 of 2 Beta determinants do not have 1"),
        ((np.array([3, 5, 3]), np.array([3, 5, 3])), (2, 2), "Alpha .* 1 duplicate"),
        ((np.array([3, 5]), np.array([1, 2, 2])), (2, 1), "Beta .* 1 duplicate"),
    ],
)
def test_prep_files_rejects_wrong_sector_and_duplicate_determinants(
    tmp_path: Path, ci_strings, nelec, message
):
    with pytest.raises(ValueError, match=message):
        mod._prep_files(
            work_dir=tmp_path,
            ci_strings=ci_strings,
            one_body_tensor=np.eye(4),
            two_body_tensor=np.zeros((4,) * 4),
            norb=4,
            nelec=nelec,
        )


def write_outputs(work_dir: Path, carryover: bytes) -> None:
    (work_dir / "occ_a.txt").write_text("0.9\n0.1\n0.0\n")
    (work_dir / "occ_b.txt").write_text("0.8\n0.2\n0.0\n")
    (work_dir / "davidson_energy.txt").write_text("-1.25\n")
    (work_dir / "carryover.bin").write_bytes(carryover)


def test_read_files_parses_energy_occupancies_and_carryover(tmp_path: Path):
    write_outputs(tmp_path, bytes([0b10100000, 0b01100000]))

    result = mod._read_files(work_dir=tmp_path, norb=3)

    assert isinstance(result, SBDResult)
    assert result.energy == -1.25
    np.testing.assert_allclose(result.orbital_occupancies[0], [0.9, 0.1, 0.0])
    np.testing.assert_allclose(result.orbital_occupancies[1], [0.8, 0.2, 0.0])
    assert result.carryover_bitstrings.tolist() == [[True, False, True], [False, True, True]]
    assert result.rdm1 is None and result.rdm2 is None


def test_read_files_accepts_empty_carryover_and_rejects_misaligned(tmp_path: Path):
    write_outputs(tmp_path, b"")
    assert mod._read_files(work_dir=tmp_path, norb=10).carryover_bitstrings.shape == (0, 10)

    write_outputs(tmp_path, bytes([1, 2, 3]))
    with pytest.raises(ValueError, match="not aligned"):
        mod._read_files(work_dir=tmp_path, norb=10)


def test_read_files_reads_the_beta_carryover_for_separate_beta(tmp_path: Path):
    write_outputs(tmp_path, bytes([0b11000000]))
    (tmp_path / "carryover_b.bin").write_bytes(bytes([0b10000000, 0b00100000]))

    shared = mod._read_files(work_dir=tmp_path, norb=3)
    assert shared.carryover_bitstrings_b is None

    result = mod._read_files(work_dir=tmp_path, norb=3, separate_beta=True)
    assert result.carryover_bitstrings.tolist() == [[True, True, False]]
    assert result.carryover_bitstrings_b.tolist() == [[True, False, False], [False, False, True]]
    assert result.carryover_bitstrings_b.dtype == bool


def test_read_files_requires_the_beta_carryover_for_separate_beta(tmp_path: Path):
    write_outputs(tmp_path, b"")
    with pytest.raises(FileNotFoundError, match="--bdetfile"):
        mod._read_files(work_dir=tmp_path, norb=3, separate_beta=True)


@pytest.mark.parametrize("mode,extra", [("cpu", []), ("gpu", ["--adetfile", "AlphaDets.bin"])])
def test_solver_args(mode, extra):
    solver = SBDSolverJob(root_dir="/work", solver_mode=mode, block=4, user_args=["--y"])
    args = mod._build_solver_args(solver)
    assert args[:2] == ["--task_comm_size", "1"]
    assert args[args.index("--block") + 1] == "4"
    assert args[args.index("--rdm") + 1] == "0"
    assert args[args.index("--carryover_type") + 1] == "1"
    assert "--dump_matrix_form_wf" not in args
    assert "--bdetfile" not in args
    if extra:
        assert args[args.index("--adetfile") : args.index("--adetfile") + 2] == extra
        assert "--carryoverfile" in args
    else:
        assert "--adetfile" not in args
    assert args[-1] == "--y"


@pytest.mark.parametrize("carryover_type", [0, 2, 3])
def test_solver_args_pass_the_carryover_type(carryover_type):
    solver = SBDSolverJob(root_dir="/work", carryover_type=carryover_type)
    args = mod._build_solver_args(solver)
    assert args[args.index("--carryover_type") + 1] == str(carryover_type)


@pytest.mark.parametrize("carryover_type", [-1, 4])
def test_carryover_type_is_range_checked(carryover_type):
    with pytest.raises(ValueError, match="carryover_type"):
        SBDSolverJob(root_dir="/work", carryover_type=carryover_type)


@pytest.mark.parametrize("mode", ["cpu", "gpu"])
def test_separate_beta_args_name_each_determinant_file_once(mode):
    solver = SBDSolverJob(root_dir="/work", solver_mode=mode, user_args=["--y"])
    args = mod._build_solver_args(solver, separate_beta=True)
    assert args.count("--adetfile") == 1
    assert args.count("--bdetfile") == 1
    assert args[args.index("--adetfile") + 1] == "AlphaDets.bin"
    assert args[args.index("--bdetfile") + 1] == "BetaDets.bin"
    assert ("--carryoverfile" in args) == (mode == "gpu")
    assert args[-1] == "--y"


def test_run_sbd_inner_runs_through_block_executor(monkeypatch, tmp_path: Path):
    executed = {}

    async def fake_run_job_from_blocks(**kwargs):
        executed.update(kwargs)
        write_outputs(kwargs["work_dir"], bytes([0b10100000]))
        return SimpleNamespace(exit_status=0)

    monkeypatch.setattr(mod, "run_job_from_blocks", fake_run_job_from_blocks)
    solver = SBDSolverJob(root_dir=str(tmp_path / "sbd_jobs"))

    result = asyncio.run(
        mod._run_sbd_inner.fn(
            solver=solver,
            ci_strings=(np.array([0b011]), np.array([0b011])),
            one_body_tensor=np.eye(3),
            two_body_tensor=np.zeros((3,) * 4),
            norb=3,
            nelec=(2, 2),
        )
    )

    assert result.energy == -1.25
    assert executed["command_block_name"] == "cmd-sbd-diag"
    assert executed["execution_profile_block_name"] == "exec-sbd-mpi"
    assert executed["hpc_profile_block_name"] == "hpc-miyabi-sbd"
    assert executed["script_filename"] == "sbd_solver.pbs"
    assert Path(executed["work_dir"]).parent == (tmp_path / "sbd_jobs").resolve()
    assert (Path(executed["work_dir"]) / "AlphaDets.bin").is_file()
    assert not (Path(executed["work_dir"]) / "BetaDets.bin").exists()
    assert "--bdetfile" not in executed["user_args"]
    assert result.carryover_bitstrings_b is None


def test_run_sbd_inner_passes_separate_beta_for_an_open_shell_subspace(monkeypatch, tmp_path: Path):
    executed = {}

    async def fake_run_job_from_blocks(**kwargs):
        executed.update(kwargs)
        write_outputs(kwargs["work_dir"], bytes([0b11000000]))
        (kwargs["work_dir"] / "carryover_b.bin").write_bytes(bytes([0b10000000]))
        return SimpleNamespace(exit_status=0)

    monkeypatch.setattr(mod, "run_job_from_blocks", fake_run_job_from_blocks)
    result = asyncio.run(
        mod._run_sbd_inner.fn(
            solver=SBDSolverJob(root_dir=str(tmp_path)),
            ci_strings=(np.array([0b011, 0b101]), np.array([0b001, 0b010])),
            one_body_tensor=np.eye(3),
            two_body_tensor=np.zeros((3,) * 4),
            norb=3,
            nelec=(2, 1),
        )
    )

    args = executed["user_args"]
    assert args[args.index("--bdetfile") + 1] == "BetaDets.bin"
    assert (Path(executed["work_dir"]) / "BetaDets.bin").is_file()
    assert result.carryover_bitstrings_b.tolist() == [[True, False, False]]


def test_run_sbd_inner_fails_on_nonzero_exit(monkeypatch, tmp_path: Path):
    async def fake_run_job_from_blocks(**kwargs):
        return SimpleNamespace(exit_status=1)

    monkeypatch.setattr(mod, "run_job_from_blocks", fake_run_job_from_blocks)
    with pytest.raises(RuntimeError, match="exit_status=1"):
        asyncio.run(
            mod._run_sbd_inner.fn(
                solver=SBDSolverJob(root_dir=str(tmp_path)),
                ci_strings=(np.array([1]), np.array([1])),
                one_body_tensor=np.eye(1),
                two_body_tensor=np.zeros((1, 1, 1, 1)),
                norb=1,
                nelec=(1, 1),
            )
        )


HF_2 = 0b11  # Hartree-Fock determinant for two electrons per spin.
HF_1 = 0b1  # Hartree-Fock determinant for one electron.


@pytest.mark.parametrize(
    "ci_strings,nelec,spin_sq,message",
    [
        ((np.array([5, HF_2]), np.array([5, HF_2])), (2, 2), None, "alpha Hartree-Fock"),
        ((np.array([HF_2, 5]), np.array([5, HF_2])), (2, 2), None, "beta Hartree-Fock"),
        ((np.array([HF_2, 5]), np.array([HF_2, 5])), (2, 1), None, "beta Hartree-Fock"),
        ((np.array([HF_1, 2]), np.array([HF_2, 5])), (2, 2), None, "alpha Hartree-Fock"),
        ((np.array([], dtype=int), np.array([], dtype=int)), (2, 2), None, "Hartree-Fock"),
        ((np.array([HF_2, 5]), np.array([], dtype=int)), (2, 2), None, "beta Hartree-Fock"),
        ((np.array([HF_2, 5]), np.array([HF_2, 5])), (2, 2), 0.0, "spin_sq"),
        ((np.array([HF_2, 5]), np.array([HF_1, 2])), (2, 1), 0.75, "spin_sq"),
    ],
)
def test_run_rejects_input_sbd_cannot_solve(monkeypatch, ci_strings, nelec, spin_sq, message):
    calls = []

    async def fake_run_job_from_blocks(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(exit_status=0)

    monkeypatch.setattr(mod, "run_job_from_blocks", fake_run_job_from_blocks)
    solver = SBDSolverJob(root_dir="/unused")
    with pytest.raises(ValueError, match=message):
        asyncio.run(
            solver.run(
                ci_strings=ci_strings,
                one_body_tensor=np.eye(4),
                two_body_tensor=np.zeros((4,) * 4),
                norb=4,
                nelec=nelec,
                spin_sq=spin_sq,
            )
        )
    assert calls == []


def test_the_hartree_fock_error_names_the_expected_and_found_determinant():
    with pytest.raises(ValueError, match="beta Hartree-Fock determinant 1 first.*got 2"):
        mod.check_sbd_input(
            ci_strings=(np.array([HF_2, 5]), np.array([2, HF_1])), nelec=(2, 1), spin_sq=None
        )


def test_valid_closed_shell_input_passes():
    mod.check_sbd_input(
        ci_strings=(np.array([HF_2, 5, 6]), [HF_2, 5, 6]), nelec=(2, 2), spin_sq=None
    )


def test_closed_shell_input_with_different_alpha_and_beta_sets_passes():
    mod.check_sbd_input(
        ci_strings=(np.array([HF_2, 5, 6]), np.array([HF_2, 9])), nelec=(2, 2), spin_sq=None
    )


def test_restricted_open_shell_input_passes():
    # Doublet with two alpha and one beta electron: the sets differ, each starts with HF.
    mod.check_sbd_input(
        ci_strings=(np.array([HF_2, 5, 6]), np.array([HF_1, 2, 4])), nelec=(2, 1), spin_sq=None
    )
