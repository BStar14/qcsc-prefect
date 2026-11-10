"""``sqd_sbd.create_blocks``: the SBD blocks from the profiles and the ``[sbd]`` table."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest
from sqd_common.block_config import load_block_config
from sqd_sbd import create_blocks

EXAMPLE = Path(__file__).resolve().parents[1] / "sqd_blocks.example.toml"
SBD_EXECUTABLE = "/path/to/qcsc-prefect/packages/qcsc-prefect-sbd/native/diag"


def _resolve(argv=(), config=None):
    args = create_blocks._parse_args(list(argv))
    return create_blocks.resolve_settings(args, config if config is not None else {})


def _sbd_config(tmp_path, profile="slurm", *, qrmi=False, extra=""):
    """The example switched to SBD on ``profile``; [qrmi] is removed unless ``qrmi``."""
    text = EXAMPLE.read_text()
    text = text.replace('solver = "fulqrum"', 'solver = "sbd"', 1)
    text = text.replace('profile = "slurm" ', f'profile = "{profile}" ', 1)
    text = text.replace('# project = "YOUR_ACCOUNT"', 'project = "gz00"', 1)
    if not qrmi:
        start, end = text.index("\n[qrmi]\n"), text.index("# IQP from the workflow host")
        text = text[:start] + "\n" + text[end:]
    if extra:
        text = text.replace("[sbd]\n", f"[sbd]\n{extra}\n", 1)
    path = tmp_path / "sqd_blocks.toml"
    path.write_text(text)
    return path


def _minimal(**overrides):
    config = {
        "hpc_target": "slurm",
        "queue": "normal",
        "work_dir": "/data/x",
        "sbd_executable": "/opt/sbd/diag",
    }
    return {**config, **overrides}


@pytest.mark.parametrize(
    ("profile", "expected"),
    [
        (
            "slurm",
            {
                "hpc_target": "slurm",
                "launcher": "srun",
                "mpi_options": ["--mpi=pmix"],
                "mpiprocs": 4,
                "ompthreads": 1,
                "resource_class": "cpu",
                "solver_mode": "cpu",
                "gpus_per_node": None,
                "environments": None,
                "script_filename": "sbd_solver.slurm",
            },
        ),
        (
            "slurm-gpu",
            {
                "hpc_target": "slurm",
                "launcher": "srun",
                "mpi_options": ["--mpi=pmix"],
                "mpiprocs": 4,
                "resource_class": "gpu",
                "solver_mode": "gpu",
                "gpus_per_node": 4,
                "environments": None,
                "script_filename": "sbd_solver.slurm",
            },
        ),
        (
            "miyabi",
            {
                "hpc_target": "miyabi",
                "launcher": "mpiexec.hydra",
                "mpi_options": ["-np", "4"],
                "mpiprocs": 4,
                "ompthreads": 1,
                "modules": ["intel/2023.2.0", "impi/2021.10.0"],
                "resource_class": "cpu",
                "solver_mode": "cpu",
                "script_filename": "sbd_solver.pbs",
            },
        ),
        (
            "miyabi-gpu",
            {
                "hpc_target": "miyabi",
                "launcher": "mpirun",
                "mpi_options": ["-n", "1"],
                "mpiprocs": 1,
                "modules": [],
                "pre_commands": ["unset OMPI_MCA_mca_base_env_list"],
                "environments": {"MIYABI": "G"},
                "resource_class": "gpu",
                "solver_mode": "gpu",
                "gpus_per_node": 1,
                "script_filename": "sbd_solver.pbs",
            },
        ),
        (
            "fugaku",
            {
                "hpc_target": "fugaku",
                "launcher": "mpiexec",
                "mpi_options": ["-n", "2"],
                "mpi_options_for_pjm": ["max-proc-per-node=2"],
                "mpiprocs": 2,
                "resource_class": "cpu",
                "gpus_per_node": None,
                "script_filename": "sbd_solver.pjm",
            },
        ),
    ],
)
def test_each_profile_resolves_its_sbd_table(tmp_path, profile, expected):
    kwargs = _resolve(config=load_block_config(_sbd_config(tmp_path, profile)))
    assert {key: kwargs[key] for key in expected} == expected
    target = expected["hpc_target"]
    assert kwargs["solver_block_name"] == "sqd-sbd-solver"
    assert kwargs["command_block_name"] == "cmd-sqd-sbd-solver"
    assert kwargs["execution_profile_block_name"] == f"exec-sqd-sbd-{target}"
    assert kwargs["hpc_profile_block_name"] == f"hpc-{target}-sqd-sbd"
    assert kwargs["metrics_artifact_key"] == f"{target}-sqd-sbd-metrics"
    assert (kwargs["command_name"], kwargs["executable_key"], kwargs["profile_name"]) == (
        "sqd-sbd",
        "sqd_sbd_solver",
        "sqd-sbd-mpi",
    )
    assert kwargs["sbd_executable"] == SBD_EXECUTABLE
    assert kwargs["root_dir"] == "/shared/work/sqd"


def test_the_fulqrum_keys_of_a_profile_do_not_reach_the_sbd_blocks(tmp_path):
    kwargs = _resolve(config=load_block_config(_sbd_config(tmp_path, "slurm-gpu")))
    assert not {"backend", "eigensolver", "fulqrum_executable", "process_grid"} & set(kwargs)
    # [sbd.environments] of the profile replaces the fulqrum FQ_DIST_BACKEND.
    assert kwargs["environments"] is None


def test_the_example_defaults_reach_the_solver_block():
    kwargs = _resolve(config=_minimal())
    assert {
        key: kwargs[key]
        for key in (
            "task_comm_size",
            "adet_comm_size",
            "bdet_comm_size",
            "block",
            "iteration",
            "tolerance",
            "carryover_type",
            "solver_mode",
            "timeout_seconds",
            "walltime",
        )
    } == {
        "task_comm_size": 1,
        "adet_comm_size": 1,
        "bdet_comm_size": 1,
        "block": 10,
        "iteration": 2,
        "tolerance": 1e-4,
        "carryover_type": 1,
        "solver_mode": "cpu",
        "timeout_seconds": 7200.0,
        # The profile's [sbd] table keeps one hour under the 12 hours of fulqrum.
        "walltime": "01:00:00",
    }
    # The block keeps its carryover_ratio default; each run sets it.
    assert "carryover_ratio" not in kwargs
    assert (kwargs["launcher"], kwargs["mpiprocs"]) == ("srun", 4)


def test_tuning_in_the_sbd_table_and_flags(tmp_path):
    path = _sbd_config(tmp_path, extra="adet_comm_size = 2\nbdet_comm_size = 2\nblock = 6")
    kwargs = _resolve(config=load_block_config(path))
    assert (kwargs["adet_comm_size"], kwargs["bdet_comm_size"], kwargs["block"]) == (2, 2, 6)
    argv = ["--iteration", "5", "--tolerance", "1e-6", "--carryover-type", "0"]
    kwargs = _resolve(argv, load_block_config(path))
    assert (kwargs["iteration"], kwargs["tolerance"], kwargs["carryover_type"]) == (5, 1e-6, 0)


@pytest.mark.parametrize(
    ("overrides", "argv"),
    [
        ({"adet_comm_size": 3}, ()),
        ({"task_comm_size": 2, "bdet_comm_size": 4}, ()),
        ({}, ("--adet-comm-size", "8")),
    ],
)
def test_comm_sizes_must_divide_the_ranks(overrides, argv):
    with pytest.raises(RuntimeError, match="must divide the 4 MPI ranks"):
        _resolve(argv, _minimal(**overrides))


def test_comm_sizes_that_divide_the_ranks_pass():
    kwargs = _resolve(config=_minimal(num_nodes=2, adet_comm_size=2, bdet_comm_size=4))
    assert kwargs["num_nodes"] * kwargs["mpiprocs"] == 8


def test_the_mpi_options_count_must_match_the_ranks(tmp_path):
    # The Miyabi profile starts -np 4; four nodes need -np 16 too.
    config = load_block_config(_sbd_config(tmp_path, "miyabi", extra="num_nodes = 4"))
    with pytest.raises(RuntimeError, match=r"starts 4 MPI ranks.*4 x 4 = 16"):
        _resolve(config=config)
    config["mpi_options"] = ["-np", "16"]
    assert _resolve(config=config)["num_nodes"] == 4


def test_the_fugaku_ranks_per_node_must_match_mpiprocs(tmp_path):
    config = load_block_config(_sbd_config(tmp_path, "fugaku"))
    assert _resolve(config=config)["mpi_options_for_pjm"] == ["max-proc-per-node=2"]
    config["mpiprocs"], config["mpi_options"] = 4, ["-n", "4"]
    with pytest.raises(RuntimeError, match=r"max-proc-per-node=2 ranks.*mpiprocs = 4"):
        _resolve(config=config)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"solver_mode": "gpu", "resource_class": "cpu"}, "solver_mode = 'gpu'.*resource_class"),
        ({"solver_mode": "tpu"}, "'solver_mode' must be one of"),
        ({"gpus_per_node": 2}, "needs resource_class='gpu'"),
        (
            {"hpc_target": "fugaku", "project": "g", "resource_class": "gpu", "gpus_per_node": 1},
            "takes no GPU request",
        ),
        (
            {"hpc_target": "miyabi", "project": "g", "resource_class": "gpu", "gpus_per_node": 2},
            "have 1 GPU",
        ),
    ],
)
def test_a_bad_gpu_setting_fails_in_dry_run(overrides, match):
    with pytest.raises(RuntimeError, match=match):
        _resolve(config=_minimal(**overrides))


def test_gpu_mode_runs_one_rank_per_gpu():
    config = _minimal(solver_mode="gpu", gpus_per_node=2, mpiprocs=4)
    kwargs = _resolve(config=config)
    assert (kwargs["resource_class"], kwargs["mpiprocs"]) == ("gpu", 2)
    with pytest.raises(RuntimeError, match="one MPI rank per GPU.*--mpiprocs 3"):
        _resolve(["--mpiprocs", "3"], config)


@pytest.mark.parametrize(
    ("key", "parameter"),
    [
        ("carryover_ratio", "sqd.carryover_fraction"),
        ("carryover_fraction", "sqd.carryover_fraction"),
        ("shots", "sampling.shots"),
    ],
)
def test_run_settings_are_rejected_as_block_settings(tmp_path, key, parameter):
    path = _sbd_config(tmp_path, extra=f"{key} = 0.5")
    with pytest.raises(RuntimeError, match=f"'{key}' is not a block setting.*{parameter}"):
        _resolve(config=load_block_config(path))


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"hpc_target": "pbs"}, "hpc_target"),
        (_minimal(queue=None), "queue"),
        (_minimal(work_dir=None), "work_dir"),
        (_minimal(sbd_executable=None), "sbd_executable"),
        (_minimal(hpc_target="miyabi"), "project/group"),
        (_minimal(mpiprocs=0), "'mpiprocs' must be at least 1"),
        (_minimal(tolerance=0), "'tolerance' must be above 0"),
    ],
)
def test_missing_or_invalid_settings(config, message):
    with pytest.raises(RuntimeError, match=message):
        _resolve(config={k: v for k, v in config.items() if v is not None})


def test_local_target_runs_one_process_without_a_queue():
    kwargs = _resolve(config={"hpc_target": "local", "work_dir": "/x", "sbd_executable": "diag"})
    assert (kwargs["launcher"], kwargs["mpiprocs"], kwargs["queue"]) == ("single", 1, "")
    assert kwargs["script_filename"] == "sbd_solver"
    # A bare command name stays a bare name, looked up on the compute node's PATH.
    assert kwargs["sbd_executable"] == "diag"


def test_dry_run_prints_settings_without_prefect(tmp_path, capsys):
    create_blocks.main(["--config", str(_sbd_config(tmp_path, "miyabi")), "--dry-run"])
    printed = json.loads(capsys.readouterr().out)
    assert set(printed) == {"blocks", "advanced_block"}
    assert printed["blocks"]["project"] == "gz00"
    assert printed["advanced_block"]["solver_block_ref"] == "sbd_solver_job/sqd-sbd-solver"


def test_dry_run_with_the_example_qrmi_table(tmp_path, capsys):
    """Needs the top-level python_executable that sqd_common.qrmi_blocks reads."""
    create_blocks.main(["--config", str(_sbd_config(tmp_path, qrmi=True)), "--dry-run"])
    printed = json.loads(capsys.readouterr().out)
    assert set(printed) == {"blocks", "qrmi_blocks", "runtime_block", "advanced_block"}
    assert printed["qrmi_blocks"]["python_executable"].endswith("/venv/qcsc-node/bin/python")
    assert printed["runtime_block"]["scheduler"] == "slurm"


@pytest.fixture
def fake_sbd(monkeypatch):
    """A ``qcsc_prefect_sbd`` module whose ``create_sbd_blocks`` records its call."""
    calls = []

    def create_sbd_blocks(**kwargs):
        calls.append(kwargs)
        return {
            name: kwargs[name]
            for name in (
                "command_block_name",
                "execution_profile_block_name",
                "hpc_profile_block_name",
                "solver_block_name",
            )
        }

    module = types.ModuleType("qcsc_prefect_sbd")
    module.create_sbd_blocks = create_sbd_blocks
    monkeypatch.setitem(sys.modules, "qcsc_prefect_sbd", module)
    return calls


def test_main_creates_the_blocks_and_the_advanced_block(tmp_path, monkeypatch, capsys, fake_sbd):
    saved = []
    monkeypatch.setattr(
        create_blocks, "save_advanced_block", lambda settings, name: saved.append((name, settings))
    )
    monkeypatch.setattr(
        create_blocks, "create_qrmi_blocks", lambda qrmi: pytest.fail("no [qrmi] table")
    )
    path = _sbd_config(tmp_path, "slurm")
    create_blocks.main(["--config", str(path)])

    (kwargs,) = fake_sbd
    assert kwargs == _resolve(config=load_block_config(path))
    ((name, settings),) = saved
    assert name == "sqd-advanced-settings"
    assert settings.solver_block_ref == "sbd_solver_job/sqd-sbd-solver"
    out = capsys.readouterr().out
    assert "Saved blocks for SQD SBD workflow (target=slurm)" in out
    assert f"SBD executable: {SBD_EXECUTABLE}" in out
    assert "Solver Block sbd_solver_job/sqd-sbd-solver" in out


def test_a_value_error_of_create_sbd_blocks_is_a_runtime_error(tmp_path, monkeypatch, fake_sbd):
    import qcsc_prefect_sbd

    def reject(**kwargs):
        raise ValueError("bad SBD setting")

    monkeypatch.setattr(qcsc_prefect_sbd, "create_sbd_blocks", reject)
    monkeypatch.setattr(
        create_blocks, "save_advanced_block", lambda *a: pytest.fail("no block may be saved")
    )
    with pytest.raises(RuntimeError, match="bad SBD setting"):
        create_blocks.main(["--config", str(_sbd_config(tmp_path))])


def test_a_missing_sbd_package_names_the_extra(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "qcsc_prefect_sbd", None)
    with pytest.raises(SystemExit, match=r'pip install "sqd\[sbd\]"'):
        create_blocks.main(["--config", str(_sbd_config(tmp_path))])
