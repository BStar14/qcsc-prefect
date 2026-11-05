"""``sqd-create-blocks``: the solver dispatch and the DICE ``[runtime]`` table."""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

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


@pytest.mark.parametrize(
    "argv, message", [(["--solver", "sbd"], "closed_loop/sqd_de"), (["--solver"], "needs one")]
)
def test_sbd_and_bad_solver_flags_exit_with_a_hint(argv, message):
    with pytest.raises(SystemExit, match=message):
        create_blocks.main(argv)


@pytest.mark.parametrize("name", ["sqd_blocks.example.toml", "sqd_blocks.fugaku.example.toml"])
def test_dice_examples_keep_run_settings_out_of_the_blocks(name):
    with (SQD / name).open("rb") as stream:
        config = tomllib.load(stream)
    assert not {"shots", "sampler_options_json", "options_variable_name"} & set(config)


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
        "build_runtime_block",
        lambda settings: SimpleNamespace(
            save=lambda name, overwrite: saved.append((name, settings))
        ),
    )
    monkeypatch.setattr(
        sys, "argv", ["sqd-create-blocks", "--config", str(SQD / "sqd_blocks.example.toml")]
    )
    dice.main()
    ((name, settings),) = saved
    assert name == "qrmi-runtime" and settings["scheduler"] == "iqp"
    assert settings["resource_config"]["QRMI_IBM_QCS_IAM_APIKEY"] == "secret-key"
    out = capsys.readouterr().out
    assert "solver_block_ref = 'dice_shci_solver_job/sqd-solver'" in out
    assert "secret-key" not in out


def test_dice_runtime_table_needs_the_api_key_before_any_block(monkeypatch):
    pytest.importorskip("qcsc_prefect_dice")
    from sqd_dice import create_blocks as dice

    monkeypatch.delenv("QRMI_IBM_QCS_IAM_APIKEY", raising=False)
    monkeypatch.setattr(
        dice, "create_dice_blocks", lambda **kwargs: pytest.fail("no block may be saved")
    )
    monkeypatch.setattr(
        sys, "argv", ["sqd-create-blocks", "--config", str(SQD / "sqd_blocks.example.toml")]
    )
    with pytest.raises(RuntimeError, match="QRMI_IBM_QCS_IAM_APIKEY"):
        dice.main()
