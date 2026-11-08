"""``sqd-create-blocks``: the solver dispatch and the DICE ``[runtime]`` table."""

from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path

import pytest
from sqd_common import create_blocks

SQD = Path(__file__).resolve().parents[1]


def test_fulqrum_is_the_default_solver(monkeypatch):
    seen = []
    monkeypatch.setattr(
        "sqd_fulqrum.create_blocks.main", lambda argv: seen.append(("fulqrum", argv))
    )
    create_blocks.main(["--config", "x.toml", "--dry-run"])
    assert seen == [("fulqrum", ["--config", "x.toml", "--dry-run"])]


def test_solver_flag_selects_dice(monkeypatch):
    pytest.importorskip("qcsc_prefect_dice")
    seen = []
    monkeypatch.setattr("sqd_dice.create_blocks.main", lambda: seen.append(sys.argv[1:]))
    create_blocks.main(["--config", "x.toml", "--solver", "dice", "--hpc-target", "miyabi"])
    assert seen == [["--config", "x.toml", "--hpc-target", "miyabi"]]


@pytest.mark.parametrize("argv, message", [(["--solver"], "needs one"), (["--solver", "x"], "x")])
def test_bad_solver_flags_exit_with_a_hint(argv, message):
    with pytest.raises(SystemExit, match=message):
        create_blocks.main(argv)


def test_solver_flag_selects_sbd(monkeypatch):
    seen = []
    monkeypatch.setattr("sqd_sbd.create_blocks.main", lambda argv: seen.append(("sbd", argv)))
    create_blocks.main(["--config", "x.toml", "--solver", "sbd", "--dry-run"])
    assert seen == [("sbd", ["--config", "x.toml", "--dry-run"])]


def test_sbd_dry_run_resolves_its_own_blocks(tmp_path, capsys):
    path = tmp_path / "sqd_blocks.toml"
    path.write_text(
        'solver = "sbd"\nprofile = "slurm"\nqueue = "normal"\nwork_dir = "/data/sqd"\n'
        '[sbd]\nsbd_executable = "/opt/sbd/diag"\n[sqd_advanced]\nccsd_maxiter = 6\n'
    )
    create_blocks.main(["--config", str(path), "--dry-run"])
    printed = json.loads(capsys.readouterr().out)
    assert printed["blocks"]["solver_block_name"] == "sqd-sbd-solver"
    assert printed["blocks"]["sbd_executable"] == "/opt/sbd/diag"
    assert printed["advanced_block"]["solver_block_ref"] == "sbd_solver_job/sqd-sbd-solver"
    assert printed["advanced_block"]["ccsd_maxiter"] == 6
    assert "qrmi_blocks" not in printed


EXAMPLE = SQD / "sqd_blocks.example.toml"


def _dice_config(tmp_path, profile="miyabi", runtime=True):
    """The example switched to DICE on ``profile``, with an IQP [runtime] table."""
    text = EXAMPLE.read_text()
    text = text.replace('solver = "fulqrum"', 'solver = "dice"')
    text = text.replace('profile = "slurm"', f'profile = "{profile}"')
    text = text.replace('# project = "YOUR_ACCOUNT"', 'project = "your-group"')
    if runtime:
        # Replace [qrmi] (Slurm only) with the commented IQP [runtime] tables.
        start, end = text.index("\n[qrmi]\n"), text.index("# To sample from the workflow")
        text = text[:start] + "\n" + text[end:]
        text = text.replace("# [runtime]\n", "[runtime]\n", 1)
        text = text.replace('# scheduler = "iqp"\n', 'scheduler = "iqp"\n', 1)
        text = text.replace(
            '# resource = "ibm_kingston"       # Optional', 'resource = "ibm_kingston"  #', 1
        )
    path = tmp_path / "sqd_blocks.toml"
    path.write_text(text)
    return path


def test_example_keeps_run_settings_out_of_the_blocks():
    with EXAMPLE.open("rb") as stream:
        config = tomllib.load(stream)
    assert config["solver"] == "fulqrum" and config["profile"] == "slurm"
    assert not {"shots", "sampler_options_json", "options_variable_name"} & set(config)


def test_solver_key_of_the_config_selects_dice(monkeypatch, tmp_path):
    pytest.importorskip("qcsc_prefect_dice")
    seen = []
    monkeypatch.setattr("sqd_dice.create_blocks.main", lambda: seen.append(sys.argv[1:]))
    path = _dice_config(tmp_path)
    create_blocks.main(["--config", str(path)])
    assert seen == [["--config", str(path)]]


@pytest.mark.parametrize(
    ("profile", "expected"),
    [
        ("miyabi", {"hpc_target": "miyabi", "launcher": "mpiexec.hydra", "mpiprocs": 4}),
        ("fugaku", {"hpc_target": "fugaku", "launcher": "mpiexec", "mpiprocs": 2}),
    ],
)
def test_dice_profiles_merge_their_mpi_settings(tmp_path, profile, expected):
    from sqd_common.block_config import load_block_config

    config = load_block_config(_dice_config(tmp_path, profile))
    assert {key: config[key] for key in expected} == expected
    assert config["dice_executable"].endswith("/Dice")
    # Only the selected solver's table is read.
    assert "fulqrum_executable" not in config


def test_dice_runtime_table_saves_the_block_after_the_solver(monkeypatch, tmp_path, capsys):
    pytest.importorskip("qcsc_prefect_dice")
    from sqd_dice import create_blocks as dice

    monkeypatch.setenv("QRMI_IBM_QCS_IAM_APIKEY", "secret-key")
    saved = []
    monkeypatch.setattr(
        dice,
        "create_dice_blocks",
        lambda **kwargs: {
            "solver_block_name": kwargs["solver_block_name"],
            "command_block_name": "c",
            "execution_profile_block_name": "e",
            "hpc_profile_block_name": "h",
        },
    )
    monkeypatch.setattr(
        dice,
        "save_runtime_block",
        lambda settings, name: saved.append((name, settings)) or [],
    )
    advanced = []
    monkeypatch.setattr(
        dice, "save_advanced_block", lambda settings, name: advanced.append((name, settings))
    )
    monkeypatch.setattr(sys, "argv", ["sqd-create-blocks", "--config", str(_dice_config(tmp_path))])
    dice.main()
    ((name, settings),) = saved
    assert name == "qrmi-runtime" and settings["scheduler"] == "iqp"
    # The API key stays in qrmi_config.json or the environment, never in the block.
    assert "api_key" not in settings and "secret-key" not in str(settings)
    # The runs use the DICE block through the Advanced SQD Settings block.
    ((name, settings),) = advanced
    assert name == "sqd-advanced-settings"
    assert settings.solver_block_ref == "dice_shci_solver_job/sqd-dice-solver"
    out = capsys.readouterr().out
    assert "Solver Block dice_shci_solver_job/sqd-dice-solver" in out
    assert "secret-key" not in out


def test_dice_runtime_table_rejects_a_key_before_any_block(monkeypatch, tmp_path):
    pytest.importorskip("qcsc_prefect_dice")
    from sqd_dice import create_blocks as dice

    monkeypatch.setattr(
        dice, "create_dice_blocks", lambda **kwargs: pytest.fail("no block may be saved")
    )
    path = _dice_config(tmp_path)
    path.write_text(path.read_text().replace("[runtime]\n", '[runtime]\napi_key = "x"\n', 1))
    monkeypatch.setattr(sys, "argv", ["sqd-create-blocks", "--config", str(path)])
    with pytest.raises(RuntimeError, match="qcsc-qrmi login stores the API key"):
        dice.main()


def test_the_solver_option_selects_its_table(tmp_path, capsys):
    path = tmp_path / "sqd_blocks.toml"
    path.write_text(
        'solver = "fulqrum"\nprofile = "slurm"\nqueue = "normal"\nwork_dir = "/data/sqd"\n'
        '[fulqrum]\nfulqrum_executable = "f"\n'
        '[sbd]\nsbd_executable = "/opt/sbd/diag"\nsolver_block_name = "my-sbd"\n'
    )
    create_blocks.main(["--config", str(path), "--solver", "sbd", "--dry-run"])
    printed = json.loads(capsys.readouterr().out)
    assert printed["advanced_block"]["solver_block_ref"] == "sbd_solver_job/my-sbd"
    assert "fulqrum_executable" not in printed["blocks"]


def test_sbd_saves_the_shared_qrmi_blocks(tmp_path, capsys):
    config = tmp_path / "sqd_blocks.toml"
    config.write_text(EXAMPLE.read_text().replace('solver = "fulqrum"', 'solver = "sbd"'))
    create_blocks.main(["--config", str(config), "--dry-run"])
    printed = json.loads(capsys.readouterr().out)
    assert printed["advanced_block"]["solver_block_ref"] == "sbd_solver_job/sqd-sbd-solver"
    assert printed["qrmi_blocks"]["command_block_name"] == "cmd-sqd-qrmi"
    assert printed["qrmi_blocks"]["queue"] == "normal"
    # The QRMI job runs one process; the SBD ranks and threads stay with the solver job.
    assert printed["blocks"]["mpiprocs"] == 4 and printed["blocks"]["launcher"] == "srun"
    assert printed["runtime_block"]["scheduler"] == "slurm"
