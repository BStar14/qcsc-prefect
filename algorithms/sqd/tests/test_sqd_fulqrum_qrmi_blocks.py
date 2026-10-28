"""The optional [qrmi] table of sqd-fulqrum-create-blocks."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest
from sqd_fulqrum import create_blocks

EXAMPLE = Path(__file__).resolve().parents[1] / "sqd_fulqrum_blocks.slurm.example.toml"
QRMI = {"resource": "test_heron", "python_executable": "/opt/venvs/qrmi/bin/python"}


def _config(**qrmi):
    with EXAMPLE.open("rb") as f:
        config = tomllib.load(f)
    if qrmi:
        config["qrmi"] = qrmi
    return config


def _resolve(argv=(), config=None):
    args = create_blocks._parse_args(list(argv))
    config = _config() if config is None else config
    block_kwargs, _, _ = create_blocks.resolve_settings(args, config)
    return create_blocks.resolve_qrmi_settings(args, config, block_kwargs)


def test_example_without_qrmi_table_creates_no_qrmi_blocks():
    assert _resolve() is None


def test_qrmi_table_inherits_solver_settings():
    qrmi = _resolve(config=_config(**QRMI))
    assert qrmi["resource"] == "test_heron"
    assert qrmi["python_executable"] == "/opt/venvs/qrmi/bin/python"
    assert qrmi["root_dir"] == "/data/sqd-fulqrum/qrmi"
    assert (qrmi["queue"], qrmi["project"]) == ("normal", "")
    assert qrmi["ompthreads"] == 4 and qrmi["walltime"] == "00:30:00"
    assert qrmi["command_block_name"] == "cmd-sqd-fulqrum-qrmi"
    assert qrmi["execution_profile_block_name"] == "exec-sqd-fulqrum-qrmi"
    assert qrmi["hpc_profile_block_name"] == "hpc-slurm-sqd-fulqrum-qrmi"
    assert create_blocks.qrmi_runtime_config(qrmi) == {
        "root_dir": "/data/sqd-fulqrum/qrmi",
        "command_block_name": "cmd-sqd-fulqrum-qrmi",
        "execution_profile_block_name": "exec-sqd-fulqrum-qrmi",
        "hpc_profile_block_name": "hpc-slurm-sqd-fulqrum-qrmi",
        "resource": "test_heron",
    }


def test_qrmi_table_overrides_and_cli():
    qrmi = _resolve(
        ["--qrmi-resource", "ibm_kingston"],
        _config(
            **QRMI,
            work_dir="/shared/qrmi",
            queue="qpu",
            ompthreads=8,
            extra_directives="--exclusive",
            environments={"RAYON_NUM_THREADS": 8},
            timeout_seconds=600,
        ),
    )
    assert qrmi["resource"] == "ibm_kingston"
    assert (qrmi["root_dir"], qrmi["queue"], qrmi["ompthreads"]) == ("/shared/qrmi", "qpu", 8)
    assert qrmi["extra_directives"] == ["--exclusive"]
    assert qrmi["environments"] == {"RAYON_NUM_THREADS": "8"}
    runtime = create_blocks.qrmi_runtime_config(qrmi)
    assert runtime["resource"] == "ibm_kingston" and runtime["timeout_seconds"] == 600.0
    assert "watch_poll_interval" not in runtime


@pytest.mark.parametrize(
    ("argv", "config", "message"),
    [
        ((), _config(python_executable="/p"), "qrmi.resource"),
        (("--qrmi-python-executable", "/p"), _config(), "qrmi.resource"),
        ((), _config(resource="test_heron"), "qrmi.python_executable"),
        ((), _config(resource="bad id", python_executable="/p"), "whitespace"),
        ((), _config(**QRMI, shots=10), "Unknown \\[qrmi\\] settings: shots"),
        (
            (),
            {
                "hpc_target": "local",
                "work_dir": "/x",
                "fulqrum_executable": "f",
                "qrmi": QRMI,
            },
            "hpc_target = 'slurm'",
        ),
    ],
)
def test_invalid_qrmi_settings(argv, config, message):
    with pytest.raises(RuntimeError, match=message):
        _resolve(argv, config)


def test_dry_run_prints_qrmi_runtime_config(tmp_path, capsys):
    path = tmp_path / "blocks.json"
    path.write_text(json.dumps(_config(**QRMI)))
    create_blocks.main(["--config", str(path), "--dry-run"])
    printed = json.loads(capsys.readouterr().out)
    assert printed["qrmi_blocks"]["resource"] == "test_heron"
    assert printed["qrmi_runtime_config"]["hpc_profile_block_name"] == "hpc-slurm-sqd-fulqrum-qrmi"


def test_main_saves_separate_qrmi_blocks(tmp_path, monkeypatch, capsys):
    pytest.importorskip("prefect")
    pytest.importorskip("qiskit_addon_sqd")
    from qcsc_prefect_fulqrum import block_utils

    saved = {}

    def fake_save(self, name, overwrite=False):
        saved[name] = self
        return self

    for cls in (
        block_utils.CommandBlock,
        block_utils.ExecutionProfileBlock,
        block_utils.HPCProfileBlock,
        block_utils.FulqrumSolverJob,
    ):
        monkeypatch.setattr(cls, "save", fake_save, raising=False)
    monkeypatch.setattr(block_utils, "register_fulqrum_block_types", lambda: None)
    monkeypatch.setattr(create_blocks, "_set_variable", lambda *args: None)
    path = tmp_path / "blocks.json"
    path.write_text(json.dumps(_config(**QRMI)))

    create_blocks.main(["--config", str(path)])

    cmd = saved["cmd-sqd-fulqrum-qrmi"]
    exec_profile = saved["exec-sqd-fulqrum-qrmi"]
    hpc = saved["hpc-slurm-sqd-fulqrum-qrmi"]
    assert cmd.command_name == exec_profile.command_name == "sqd-fulqrum-qrmi"
    assert cmd.default_args == ["-m", "qcsc_prefect.integrations.qiskit.qrmi.worker"]
    assert exec_profile.launcher == "single" and exec_profile.ompthreads == 4
    assert exec_profile.environments == {"OMP_NUM_THREADS": "4"}
    assert hpc.hpc_target == "slurm" and hpc.slurm_qpu == "test_heron"
    assert hpc.slurm_ntasks == 1
    assert hpc.executable_map == {cmd.executable_key: "/opt/venvs/qrmi/bin/python"}
    # The solver's HPC profile never requests the QPU.
    assert saved["hpc-slurm-sqd-fulqrum"].slurm_qpu is None
    out = capsys.readouterr().out
    printed = json.loads(out[out.index("{") :])
    assert printed["resource"] == "test_heron"
    assert printed["root_dir"] == "/data/sqd-fulqrum/qrmi"
