"""The optional [qrmi] and [runtime] tables of the fulqrum create-blocks script."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqd_common import qrmi_blocks  # noqa: E402
from sqd_common.block_config import load_block_config
from sqd_fulqrum import create_blocks

EXAMPLE = Path(__file__).resolve().parents[1] / "sqd_blocks.example.toml"
QRMI = {"resource": "test_heron", "python_executable": "/opt/venvs/qrmi/bin/python"}


def _config(**qrmi):
    """The example config with its default [qrmi] table replaced by ``qrmi``, or removed."""
    config = load_block_config(EXAMPLE)
    config.pop("qrmi")
    if qrmi:
        config["qrmi"] = qrmi
    return config


def test_example_keeps_the_qrmi_table_active():
    with EXAMPLE.open("rb") as f:
        config = tomllib.load(f)
    assert config["qrmi"] == {
        "resource": "YOUR_QRMI_RESOURCE",
        "python_executable": "/opt/venvs/qrmi/bin/python",
    }
    assert "runtime" not in config
    # The profile adds the QRMI job's walltime, threads and timeout.
    merged = load_block_config(EXAMPLE)["qrmi"]
    assert (merged["walltime"], merged["ompthreads"], merged["timeout_seconds"]) == (
        "00:30:00",
        4,
        3600,
    )


def _resolve(argv=(), config=None):
    args = create_blocks._parse_args(list(argv))
    config = _config() if config is None else config
    block_kwargs = create_blocks.resolve_settings(args, config)
    return qrmi_blocks.resolve_qrmi_settings(args, config, block_kwargs)


def test_config_without_qrmi_table_creates_no_qrmi_blocks():
    assert _resolve() is None


def test_qrmi_table_inherits_solver_settings():
    qrmi = _resolve(config=_config(**QRMI))
    assert qrmi["resource"] == "test_heron"
    assert qrmi["python_executable"] == "/opt/venvs/qrmi/bin/python"
    assert qrmi["root_dir"] == "/data/sqd/qrmi"
    assert (qrmi["queue"], qrmi["project"]) == ("normal", "")
    assert qrmi["ompthreads"] == 4 and qrmi["walltime"] == "00:30:00"
    assert qrmi["command_block_name"] == "cmd-sqd-qrmi"
    assert qrmi["execution_profile_block_name"] == "exec-sqd-qrmi"
    assert qrmi["hpc_profile_block_name"] == "hpc-slurm-sqd-qrmi"
    assert qrmi_blocks.qrmi_runtime_settings(qrmi) == {
        "scheduler": "slurm",
        "root_dir": "/data/sqd/qrmi",
        "command_block_name": "cmd-sqd-qrmi",
        "execution_profile_block_name": "exec-sqd-qrmi",
        "hpc_profile_block_name": "hpc-slurm-sqd-qrmi",
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
    runtime = qrmi_blocks.qrmi_runtime_settings(qrmi)
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


def test_dry_run_prints_the_runtime_block(tmp_path, capsys):
    path = tmp_path / "blocks.json"
    path.write_text(json.dumps(_config(**QRMI)))
    create_blocks.main(["--config", str(path), "--dry-run"])
    printed = json.loads(capsys.readouterr().out)
    assert printed["qrmi_blocks"]["resource"] == "test_heron"
    runtime = printed["runtime_block"]
    assert runtime["name"] == "qrmi-runtime" and runtime["scheduler"] == "slurm"
    assert runtime["hpc_profile_block_name"] == "hpc-slurm-sqd-qrmi"


IQP_RUNTIME = {
    "scheduler": "iqp",
    "resource": "ibm_kingston",
    "qrmi": {"QRMI_IBM_QCS_ENDPOINT": "https://quantum.cloud.ibm.com/api/v1"},
}


def test_runtime_table_reads_the_api_key_from_the_environment(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("QRMI_IBM_QCS_IAM_APIKEY", "secret-key")
    path = tmp_path / "blocks.json"
    path.write_text(json.dumps({**_config(), "runtime": IQP_RUNTIME, "runtime_block_name": "rt"}))
    create_blocks.main(["--config", str(path), "--dry-run"])
    out = capsys.readouterr().out
    assert "secret-key" not in out
    runtime = json.loads(out)["runtime_block"]
    assert runtime["name"] == "rt" and runtime["scheduler"] == "iqp"
    # The key goes to the block's secret api_key field; the endpoint is a plain field.
    assert runtime["api_key"] == "set"
    assert runtime["endpoint"] == "https://quantum.cloud.ibm.com/api/v1"
    assert "resource_config" not in runtime


def test_qrmi_table_takes_the_crn_and_rejects_an_api_key():
    qrmi = _resolve(config=_config(**QRMI, service_crn="crn:v1:x"))
    runtime = qrmi_blocks.qrmi_runtime_settings(qrmi)
    assert runtime["service_crn"] == "crn:v1:x" and "api_key" not in runtime
    with pytest.raises(RuntimeError, match="Do not write api_key in \\[qrmi\\]"):
        _resolve(config=_config(**QRMI, api_key="secret"))


@pytest.mark.parametrize(
    ("runtime", "env", "message"),
    [
        (IQP_RUNTIME, {}, "QRMI_IBM_QCS_IAM_APIKEY"),
        (
            {**IQP_RUNTIME, "qrmi": {"QRMI_IBM_QCS_IAM_APIKEY": "x"}},
            {"QRMI_IBM_QCS_IAM_APIKEY": "k"},
            "set it in the environment",
        ),
        ({**IQP_RUNTIME, "transpiler": "m:f"}, {"QRMI_IBM_QCS_IAM_APIKEY": "k"}, "must not set"),
        ({"scheduler": "slurm", "resource": "r", "qrmi": {"A": "b"}}, {}, "only used by"),
        ({"scheduler": "iqp"}, {}, "runtime.resource"),
    ],
)
def test_invalid_runtime_table(monkeypatch, runtime, env, message):
    monkeypatch.delenv("QRMI_IBM_QCS_IAM_APIKEY", raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    with pytest.raises(RuntimeError, match=message):
        create_blocks.resolve_runtime_settings({"runtime": runtime}, None)


def test_qrmi_and_runtime_tables_are_exclusive():
    with pytest.raises(RuntimeError, match="not both"):
        create_blocks.resolve_runtime_settings({"runtime": IQP_RUNTIME}, {"resource": "r"})


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
    runtime_blocks = []
    monkeypatch.setattr(
        create_blocks,
        "build_runtime_block",
        lambda settings: SimpleNamespace(
            save=lambda name, overwrite: runtime_blocks.append((name, settings))
        ),
    )
    monkeypatch.setattr(create_blocks, "save_advanced_block", lambda settings, name: None)
    path = tmp_path / "blocks.json"
    path.write_text(json.dumps(_config(**QRMI)))

    create_blocks.main(["--config", str(path)])

    cmd = saved["cmd-sqd-qrmi"]
    exec_profile = saved["exec-sqd-qrmi"]
    hpc = saved["hpc-slurm-sqd-qrmi"]
    assert cmd.command_name == exec_profile.command_name == "sqd-qrmi"
    assert cmd.default_args == ["-m", "qcsc_prefect.integrations.qiskit.qrmi.worker"]
    assert exec_profile.launcher == "single" and exec_profile.ompthreads == 4
    assert exec_profile.environments == {"OMP_NUM_THREADS": "4"}
    assert hpc.hpc_target == "slurm" and hpc.slurm_qpu == "test_heron"
    assert hpc.slurm_ntasks == 1
    assert hpc.executable_map == {cmd.executable_key: "/opt/venvs/qrmi/bin/python"}
    # The solver's HPC profile never requests the QPU.
    assert saved["hpc-slurm-sqd-fulqrum"].slurm_qpu is None
    ((name, settings),) = runtime_blocks
    assert name == "qrmi-runtime" and settings["scheduler"] == "slurm"
    assert settings["resource"] == "test_heron"
    assert settings["root_dir"] == "/data/sqd/qrmi"
    assert settings["hpc_profile_block_name"] == "hpc-slurm-sqd-qrmi"
    assert "QRMI runtime block: qrmi-runtime (scheduler=slurm)" in capsys.readouterr().out


def test_qrmi_keys_without_their_header_get_a_hint(tmp_path):
    # In the example, a commented-out [qrmi] header leaves its keys in [dice].
    text = EXAMPLE.read_text().replace("\n[qrmi]\n", "\n# [qrmi]\n")
    path = tmp_path / "blocks.toml"
    path.write_text(text)
    with pytest.raises(RuntimeError, match=r"python_executable, resource in \[dice\].*header"):
        create_blocks.main(["--config", str(path), "--dry-run"])


def test_duplicate_keys_get_a_hint(tmp_path):
    path = tmp_path / "blocks.toml"
    path.write_text('walltime = "01:00:00"\n# [qrmi]\nwalltime = "00:30:00"\n')
    with pytest.raises(RuntimeError, match=r"Invalid TOML in .*blocks\.toml.*header line"):
        create_blocks.main(["--config", str(path), "--dry-run"])
