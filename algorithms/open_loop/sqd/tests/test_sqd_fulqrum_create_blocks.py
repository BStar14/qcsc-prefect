from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqd_common.block_config import load_block_config
from sqd_fulqrum import create_blocks

EXAMPLE = Path(__file__).resolve().parents[1] / "sqd_blocks.example.toml"


def _resolve(argv=(), config=None):
    args = create_blocks._parse_args(list(argv))
    return create_blocks.resolve_settings(args, config if config is not None else {})


def _example_config():
    """The example with its profile merged in, as create_blocks reads it."""
    return load_block_config(EXAMPLE)


def test_example_toml_resolves_to_expected_settings():
    kwargs = _resolve(config=_example_config())
    assert kwargs["hpc_target"] == "slurm"
    assert kwargs["queue"] == "normal"
    assert kwargs["project"] == ""
    assert kwargs["root_dir"] == "/data/sqd-fulqrum"
    assert kwargs["fulqrum_executable"] == "/opt/venvs/fulqrum/bin/qcsc-fulqrum-solve"
    assert kwargs["launcher"] == "single"
    assert (kwargs["num_nodes"], kwargs["mpiprocs"], kwargs["ompthreads"]) == (1, 1, 4)
    assert kwargs["solver_block_name"] == "sqd-fulqrum-solver"
    assert kwargs["execution_profile_block_name"] == "exec-sqd-fulqrum-slurm"
    assert kwargs["hpc_profile_block_name"] == "hpc-slurm-sqd-fulqrum"
    assert kwargs["script_filename"] == "fulqrum_solver.slurm"
    assert kwargs["metrics_artifact_key"] == "slurm-sqd-fulqrum-metrics"
    assert kwargs["eigensolver"] == "scipy" and kwargs["tol"] == 1e-8
    assert kwargs["max_iter"] is None and kwargs["return_sci_state"] is False
    assert kwargs["modules"] == [] and kwargs["environments"] is None


def test_defaults_match_explicit_example_names():
    """The example spells out the defaults; dropping them must not change anything."""

    config = _example_config()
    minimal = {k: config[k] for k in ("hpc_target", "queue", "work_dir", "fulqrum_executable")}
    full = _resolve(config=config)
    reduced = _resolve(config=minimal)
    for key in (
        "command_block_name",
        "execution_profile_block_name",
        "hpc_profile_block_name",
        "solver_block_name",
        "script_filename",
        "metrics_artifact_key",
        "launcher",
        "walltime",
        "eigensolver",
        "hamiltonian_format",
        "dense_max_dim",
        "return_sci_state",
    ):
        assert reduced[key] == full[key], key


def test_cli_overrides_config():
    argv = [
        "--queue",
        "debug",
        "--ompthreads",
        "8",
        "--eigensolver",
        "primme",
        "--max-iter",
        "500",
        "--return-sci-state",
        "--modules",
        "gcc",
        "openblas",
        "--slurm-memory",
        "16G",
    ]
    kwargs = _resolve(argv, _example_config())
    assert kwargs["queue"] == "debug"
    assert kwargs["ompthreads"] == 8
    assert kwargs["eigensolver"] == "primme"
    assert kwargs["max_iter"] == 500
    assert kwargs["return_sci_state"] is True
    assert kwargs["modules"] == ["gcc", "openblas"]
    assert kwargs["slurm_memory"] == "16G"


def test_environments_and_csv_lists():
    config = {
        "hpc_target": "slurm",
        "queue": "normal",
        "work_dir": "/data/x",
        "fulqrum_executable": "qcsc-fulqrum-solve",
        "modules": "gcc, openblas",
        "environments": {"OMP_NUM_THREADS": 2, "OPENBLAS_NUM_THREADS": "1"},
    }
    kwargs = _resolve(config=config)
    assert kwargs["modules"] == ["gcc", "openblas"]
    assert kwargs["environments"] == {"OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1"}
    # A bare command name stays a bare name, looked up on the compute node's PATH.
    assert kwargs["fulqrum_executable"] == "qcsc-fulqrum-solve"


@pytest.mark.parametrize("key", ["shots", "sampler_options_json", "options_variable_name"])
def test_run_settings_are_rejected_as_block_settings(key):
    config = {**_example_config(), key: 100}
    with pytest.raises(RuntimeError, match=f"'{key}' is not a block setting.*flow parameters"):
        _resolve(config=config)


@pytest.mark.parametrize("flag", ["--shots", "--sampler-options-json", "--options-variable-name"])
def test_run_settings_have_no_cli_flags(flag):
    with pytest.raises(SystemExit):
        create_blocks._parse_args([flag, "1"])


@pytest.mark.parametrize(
    ("target", "suffix"),
    [("local", ""), ("miyabi", ".pbs"), ("fugaku", ".pjm")],
)
def test_other_targets(target, suffix):
    config = {
        "hpc_target": target,
        "work_dir": "/data/x",
        "fulqrum_executable": "/opt/fq/bin/qcsc-fulqrum-solve",
    }
    if target != "local":
        config.update(group="grp", queue="q")
    kwargs = _resolve(config=config)
    assert kwargs["hpc_target"] == target
    assert kwargs["script_filename"] == f"fulqrum_solver{suffix}"
    assert kwargs["hpc_profile_block_name"] == f"hpc-{target}-sqd-fulqrum"
    if target != "local":
        assert kwargs["project"] == "grp"


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"hpc_target": "pbs"}, "hpc_target"),
        ({"hpc_target": "slurm", "work_dir": "/x", "fulqrum_executable": "f"}, "queue"),
        ({"hpc_target": "slurm", "queue": "q", "fulqrum_executable": "f"}, "work_dir"),
        ({"hpc_target": "slurm", "queue": "q", "work_dir": "/x"}, "fulqrum_executable"),
        (
            {"hpc_target": "miyabi", "queue": "q", "work_dir": "/x", "fulqrum_executable": "f"},
            "project/group",
        ),
    ],
)
def test_missing_or_invalid_settings(config, message):
    with pytest.raises(RuntimeError, match=message):
        _resolve(config=config)


def test_dry_run_prints_settings_without_prefect(capsys):
    create_blocks.main(["--config", str(EXAMPLE), "--dry-run"])
    printed = json.loads(capsys.readouterr().out)
    # The example keeps [qrmi] active, so the QRMI job and its runtime block resolve too.
    assert set(printed) == {"blocks", "qrmi_blocks", "runtime_block"}
    assert printed["blocks"]["solver_block_name"] == "sqd-fulqrum-solver"
    assert printed["qrmi_blocks"]["resource"] == "YOUR_QRMI_RESOURCE"
    assert printed["runtime_block"]["name"] == "qrmi-runtime"
    assert printed["runtime_block"]["scheduler"] == "slurm"


def test_resolved_kwargs_build_the_block_set(monkeypatch, capsys):
    """main() feeds create_fulqrum_blocks; the block saves are mocked."""

    pytest.importorskip("prefect")
    pytest.importorskip("qiskit_addon_sqd")
    from qcsc_prefect_fulqrum import block_utils

    kwargs = _resolve(config=_example_config())
    inspect.signature(block_utils.create_fulqrum_blocks).bind(**kwargs)

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
    # The example's [qrmi] table also saves the QRMI job and runtime blocks; stub them.
    qrmi_saved = []
    monkeypatch.setattr(create_blocks, "create_qrmi_blocks", qrmi_saved.append)
    monkeypatch.setattr(
        create_blocks,
        "build_runtime_block",
        lambda settings: SimpleNamespace(save=lambda name, overwrite: qrmi_saved.append(name)),
    )

    create_blocks.main(["--config", str(EXAMPLE)])

    assert set(saved) == {
        "cmd-sqd-fulqrum-solver",
        "exec-sqd-fulqrum-slurm",
        "hpc-slurm-sqd-fulqrum",
        "sqd-fulqrum-solver",
    }
    cmd = saved["cmd-sqd-fulqrum-solver"]
    exec_profile = saved["exec-sqd-fulqrum-slurm"]
    hpc = saved["hpc-slurm-sqd-fulqrum"]
    solver = saved["sqd-fulqrum-solver"]
    assert cmd.command_name == exec_profile.command_name == "sqd-fulqrum"
    assert exec_profile.launcher == "single" and exec_profile.ompthreads == 4
    assert exec_profile.environments == {"OMP_NUM_THREADS": "4"}
    assert hpc.hpc_target == "slurm" and hpc.queue_cpu == "normal"
    assert hpc.executable_map == {"sqd_fulqrum_solver": "/opt/venvs/fulqrum/bin/qcsc-fulqrum-solve"}
    assert solver.root_dir == str(Path("/data/sqd-fulqrum").resolve())
    assert solver.execution_profile_block_name == "exec-sqd-fulqrum-slurm"
    assert solver.hpc_profile_block_name == "hpc-slurm-sqd-fulqrum"
    assert solver.return_sci_state is False
    assert qrmi_saved[0]["resource"] == "YOUR_QRMI_RESOURCE" and qrmi_saved[1] == "qrmi-runtime"
    assert "Saved blocks for SQD fulqrum workflow" in capsys.readouterr().out
