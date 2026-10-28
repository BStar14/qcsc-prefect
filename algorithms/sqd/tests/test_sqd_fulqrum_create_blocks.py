from __future__ import annotations

import inspect
import json
import tomllib
from pathlib import Path

import pytest
from sqd_fulqrum import create_blocks

EXAMPLE = Path(__file__).resolve().parents[1] / "sqd_fulqrum_blocks.slurm.example.toml"


def _resolve(argv=(), config=None):
    args = create_blocks._parse_args(list(argv))
    return create_blocks.resolve_settings(args, config if config is not None else {})


def _example_config():
    with EXAMPLE.open("rb") as f:
        return tomllib.load(f)


def test_example_toml_resolves_to_expected_settings():
    kwargs, var_name, sampler = _resolve(config=_example_config())
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
    assert var_name == "sampler_options"
    assert sampler == {"params": {"shots": 100000}}


def test_defaults_match_explicit_example_names():
    """The example spells out the defaults; dropping them must not change anything."""

    config = _example_config()
    minimal = {k: config[k] for k in ("hpc_target", "queue", "work_dir", "fulqrum_executable")}
    full, _, _ = _resolve(config=config)
    reduced, _, _ = _resolve(config=minimal)
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
        "--sampler-options-json",
        '{"params": {"shots": 2000}}',
        "--modules",
        "gcc",
        "openblas",
        "--slurm-memory",
        "16G",
    ]
    kwargs, _, sampler = _resolve(argv, _example_config())
    assert kwargs["queue"] == "debug"
    assert kwargs["ompthreads"] == 8
    assert kwargs["eigensolver"] == "primme"
    assert kwargs["max_iter"] == 500
    assert kwargs["return_sci_state"] is True
    assert kwargs["modules"] == ["gcc", "openblas"]
    assert kwargs["slurm_memory"] == "16G"
    assert sampler == {"params": {"shots": 2000}}


def test_environments_and_csv_lists():
    config = {
        "hpc_target": "slurm",
        "queue": "normal",
        "work_dir": "/data/x",
        "fulqrum_executable": "qcsc-fulqrum-solve",
        "modules": "gcc, openblas",
        "environments": {"OMP_NUM_THREADS": 2, "OPENBLAS_NUM_THREADS": "1"},
        "shots": 500,
    }
    kwargs, _, sampler = _resolve(config=config)
    assert kwargs["modules"] == ["gcc", "openblas"]
    assert kwargs["environments"] == {"OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1"}
    # A bare command name stays a bare name, looked up on the compute node's PATH.
    assert kwargs["fulqrum_executable"] == "qcsc-fulqrum-solve"
    assert sampler == {"params": {"shots": 500}}


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
    kwargs, _, _ = _resolve(config=config)
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


def test_dry_run_prints_settings_without_prefect(capsys, monkeypatch):
    def fail(*_args, **_kwargs):
        raise AssertionError("dry run must not save anything")

    monkeypatch.setattr(create_blocks, "_set_variable", fail)
    create_blocks.main(["--config", str(EXAMPLE), "--dry-run"])
    printed = json.loads(capsys.readouterr().out)
    assert printed["blocks"]["solver_block_name"] == "sqd-fulqrum-solver"
    assert printed["sampler_options"] == {"params": {"shots": 100000}}


def test_resolved_kwargs_build_the_block_set(monkeypatch, capsys):
    """main() feeds create_fulqrum_blocks; saves and the variable are mocked."""

    pytest.importorskip("prefect")
    pytest.importorskip("qiskit_addon_sqd")
    from qcsc_prefect_fulqrum import block_utils

    kwargs, _, _ = _resolve(config=_example_config())
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
    variables = {}
    monkeypatch.setattr(create_blocks, "_set_variable", variables.__setitem__)

    create_blocks.main(["--config", str(EXAMPLE)])

    assert variables == {"sampler_options": {"params": {"shots": 100000}}}
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
    assert "Saved blocks and variable" in capsys.readouterr().out
