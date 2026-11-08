from __future__ import annotations

import inspect
import json
import tomllib
from pathlib import Path

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
    assert kwargs["root_dir"] == "/data/sqd"
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


@pytest.mark.parametrize(
    "key", ["shots", "sampler_options_json", "options_variable_name", "carryover_fraction"]
)
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
    assert set(printed) == {"blocks", "qrmi_blocks", "runtime_block", "advanced_block"}
    assert printed["advanced_block"]["name"] == "sqd-advanced-settings"
    assert printed["advanced_block"]["solver_block_ref"] == "fulqrum_solver_job/sqd-fulqrum-solver"
    assert printed["advanced_block"]["ccsd_maxiter"] == 10
    assert printed["blocks"]["solver_block_name"] == "sqd-fulqrum-solver"
    assert printed["qrmi_blocks"]["resource"] is None  # kept from qcsc-qrmi login
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
        "save_runtime_block",
        lambda settings, name: qrmi_saved.append(name) or [],
    )

    advanced_saved = []
    monkeypatch.setattr(
        create_blocks, "save_advanced_block", lambda settings, name: advanced_saved.append(name)
    )
    create_blocks.main(["--config", str(EXAMPLE)])

    assert advanced_saved == ["sqd-advanced-settings"]
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
    assert solver.root_dir == str(Path("/data/sqd").resolve())
    assert solver.execution_profile_block_name == "exec-sqd-fulqrum-slurm"
    assert solver.hpc_profile_block_name == "hpc-slurm-sqd-fulqrum"
    assert solver.return_sci_state is False
    assert qrmi_saved[0]["resource"] is None and qrmi_saved[1] == "qrmi-runtime"
    assert "Saved blocks for SQD fulqrum workflow" in capsys.readouterr().out


def _gpu_config(tmp_path, profile, *, qrmi=True, advanced=""):
    """The example switched to a GPU profile, optionally without [qrmi] (Slurm only)."""
    text = EXAMPLE.read_text()
    text = text.replace('profile = "slurm" ', f'profile = "{profile}" ', 1)
    text = text.replace('queue = "normal"', 'queue = "regular-g"', 1)
    text = text.replace('# project = "YOUR_ACCOUNT"', 'project = "gz00"', 1)
    if not qrmi:
        start, end = text.index("\n[qrmi]\n"), text.index("# To sample from the workflow")
        text = text[:start] + "\n" + text[end:]
    if advanced:
        text += f"\n[advanced]\n{advanced}\n"
    path = tmp_path / "sqd_blocks.toml"
    path.write_text(text)
    return path


def test_gpu_profiles_are_bundled():
    from sqd_common.block_config import PROFILES_DIR, profile_names

    assert {"slurm-gpu", "miyabi-gpu"} <= set(profile_names())
    pyproject = tomllib.loads((EXAMPLE.parent / "pyproject.toml").read_text())
    globs = pyproject["tool"]["setuptools"]["package-data"]["sqd_common"]
    packaged = {path.stem for glob in globs for path in (PROFILES_DIR.parent).glob(glob)}
    assert {"slurm-gpu", "miyabi-gpu"} <= packaged


@pytest.mark.parametrize(
    ("profile", "qrmi", "expected"),
    [
        (
            "slurm-gpu",
            True,
            {
                "hpc_target": "slurm",
                "launcher": "srun",
                "mpi_options": ["--mpi=pmix"],
                "num_nodes": 1,
                "gpus_per_node": 4,
                "mpiprocs": 4,
                "ompthreads": 8,
            },
        ),
        (
            "miyabi-gpu",
            False,
            {
                "hpc_target": "miyabi",
                "launcher": "mpirun",
                "mpi_options": None,
                "num_nodes": 1,
                "gpus_per_node": 1,
                "mpiprocs": 1,
                "ompthreads": 72,
            },
        ),
    ],
)
def test_gpu_profiles_resolve_to_gpu_mpi(tmp_path, profile, qrmi, expected):
    kwargs = _resolve(config=load_block_config(_gpu_config(tmp_path, profile, qrmi=qrmi)))
    assert {key: kwargs[key] for key in expected} == expected
    assert kwargs["resource_class"] == "gpu"
    assert kwargs["backend"] == "gpu-mpi"
    assert "carryover_fraction" not in kwargs
    assert kwargs["environments"] == {"FQ_DIST_BACKEND": "auto"}
    inspect.signature(_create_fulqrum_blocks_signature()).bind(**kwargs)


def _create_fulqrum_blocks_signature():
    """``create_fulqrum_blocks``; skips when its Prefect dependencies are missing."""
    try:
        from qcsc_prefect_fulqrum.block_utils import create_fulqrum_blocks
    except ModuleNotFoundError:
        pytest.skip("qcsc_prefect_fulqrum.block_utils needs Prefect")
    return create_fulqrum_blocks


def test_user_file_and_advanced_override_the_gpu_profile(tmp_path):
    path = _gpu_config(
        tmp_path,
        "slurm-gpu",
        advanced='backend = "gpu"\ngpus_per_node = 2',
    )
    config = load_block_config(path)
    assert (config["backend"], config["gpus_per_node"]) == ("gpu", 2)
    kwargs = _resolve(config=config)
    # [advanced] overrides the profile; the profile's mpiprocs is kept.
    assert (kwargs["backend"], kwargs["gpus_per_node"], kwargs["mpiprocs"]) == ("gpu", 2, 4)

    # A top-level key of the user file overrides [advanced].
    text = path.read_text().replace(
        "# Keys after a [table] header", "gpus_per_node = 3\n# Keys after a [table] header", 1
    )
    path.write_text(text)
    assert _resolve(config=load_block_config(path))["gpus_per_node"] == 3


def test_gpu_mpi_runs_one_rank_per_gpu():
    config = {
        "hpc_target": "slurm",
        "queue": "gpu",
        "work_dir": "/data/x",
        "fulqrum_executable": "qcsc-fulqrum-solve",
        "launcher": "srun",
        "resource_class": "gpu",
        "gpus_per_node": 8,
        "backend": "gpu-mpi",
    }
    assert _resolve(config=config)["mpiprocs"] == 8
    # A config's mpiprocs, for example a CPU profile's 1, gives way to the GPU count.
    assert _resolve(config={**config, "mpiprocs": 1})["mpiprocs"] == 8
    with pytest.raises(RuntimeError, match="one MPI rank per GPU.*--mpiprocs 2"):
        _resolve(["--mpiprocs", "2"], config)
    cpu = _resolve(config={**config, "backend": "cpu"})
    assert (cpu["mpiprocs"], cpu["gpus_per_node"]) == (1, 8)


def test_cpu_profile_with_gpu_mpi_gets_one_rank_per_gpu(tmp_path):
    """The slurm profile sets mpiprocs = 1; gpu-mpi with 4 GPUs must still run 4 ranks."""
    path = _gpu_config(
        tmp_path,
        "slurm",
        advanced='backend = "gpu-mpi"\nlauncher = "srun"\nresource_class = "gpu"\n'
        "gpus_per_node = 4",
    )
    kwargs = _resolve(config=load_block_config(path))
    assert (kwargs["mpiprocs"], kwargs["gpus_per_node"]) == (4, 4)


def test_slurm_gpu_profile_follows_a_lower_gpu_count(tmp_path):
    path = _gpu_config(tmp_path, "slurm-gpu", advanced="gpus_per_node = 2")
    kwargs = _resolve(config=load_block_config(path))
    assert (kwargs["mpiprocs"], kwargs["gpus_per_node"]) == (2, 2)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"hpc_target": "fugaku", "project": "g", "gpus_per_node": 1}, "takes no GPU request"),
        ({"hpc_target": "miyabi", "project": "g", "gpus_per_node": 4}, "have 1 GPU"),
        ({"gpus_per_node": 4}, "needs resource_class='gpu'"),
    ],
)
def test_a_bad_gpu_request_fails_at_block_creation(overrides, match):
    with pytest.raises(RuntimeError, match=match):
        _resolve(config={**_example_config(), **overrides})


def test_gpu_cli_flags_override_config():
    argv = ["--backend", "gpu", "--gpus-per-node", "2", "--resource-class", "gpu"]
    kwargs = _resolve(argv, _example_config())
    assert (kwargs["backend"], kwargs["gpus_per_node"]) == ("gpu", 2)
    assert _resolve(config=_example_config())["gpus_per_node"] is None


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"backend": "gpu-mpi"}, "MPI launcher"),
        ({"backend": "tpu"}, "'backend' must be one of"),
    ],
)
def test_invalid_gpu_settings(overrides, message):
    with pytest.raises(RuntimeError, match=message):
        _resolve(config={**_example_config(), **overrides})


@pytest.mark.parametrize(("profile", "qrmi"), [("slurm-gpu", True), ("miyabi-gpu", False)])
def test_dry_run_of_the_gpu_profiles(tmp_path, capsys, profile, qrmi):
    path = _gpu_config(tmp_path, profile, qrmi=qrmi)
    create_blocks.main(["--config", str(path), "--dry-run"])
    printed = json.loads(capsys.readouterr().out)
    blocks = printed["blocks"]
    assert blocks["resource_class"] == "gpu" and blocks["backend"] == "gpu-mpi"
    assert blocks["queue"] == "regular-g"
    if profile == "slurm-gpu":
        assert (blocks["gpus_per_node"], blocks["mpiprocs"], blocks["launcher"]) == (4, 4, "srun")
        assert blocks["script_filename"] == "fulqrum_solver.slurm"
        assert printed["qrmi_blocks"]["walltime"] == "00:30:00"
    else:
        assert (blocks["gpus_per_node"], blocks["mpiprocs"], blocks["launcher"]) == (1, 1, "mpirun")
        assert blocks["script_filename"] == "fulqrum_solver.pbs"
        assert blocks["project"] == "gz00"
        assert "qrmi_blocks" not in printed


def test_tuning_keys_and_flags_reach_the_solver_block():
    config = {**_example_config(), "eigensolver": "primme", "max_basis_size": 32, "block_size": 2}
    kwargs = _resolve(config=config)
    assert (kwargs["max_basis_size"], kwargs["block_size"]) == (32, 2)
    assert kwargs["process_grid"] is None and kwargs["comm_backend"] == "auto"
    argv = ["--min-restart-size", "6", "--primme-method", "PRIMME_DYNAMIC"]
    kwargs = _resolve(argv, config)
    assert (kwargs["min_restart_size"], kwargs["primme_method"]) == (6, "PRIMME_DYNAMIC")


def test_gpu_mpi_tuning_resolves_from_the_gpu_profile(tmp_path):
    path = _gpu_config(
        tmp_path,
        "slurm-gpu",
        advanced='process_grid = [2, 2]\nconvergence = "energy"\ncomm_backend = "nccl"',
    )
    kwargs = _resolve(config=load_block_config(path))
    assert (kwargs["process_grid"], kwargs["convergence"], kwargs["comm_backend"]) == (
        (2, 2),
        "energy",
        "nccl",
    )


def test_tuning_the_eigensolver_ignores_fails_in_dry_run():
    with pytest.raises(RuntimeError, match="apply to PRIMME only"):
        _resolve(config={**_example_config(), "block_size": 2})
    with pytest.raises(RuntimeError, match="apply to backend='gpu-mpi' only"):
        _resolve(config={**_example_config(), "process_grid": [1, 2]})


def test_a_dice_energy_tol_in_advanced_is_ignored_on_cpu(tmp_path):
    """Configs from before [dice] keep DICE's energy_tol in [advanced]."""
    path = _gpu_config(tmp_path, "slurm", advanced="energy_tol = 1e-10")
    assert _resolve(config=load_block_config(path))["energy_tol"] == 1e-6
    (tmp_path / "gpu").mkdir()
    gpu = _gpu_config(tmp_path / "gpu", "slurm-gpu", advanced="energy_tol = 1e-9")
    assert _resolve(config=load_block_config(gpu))["energy_tol"] == 1e-9
    with pytest.raises(RuntimeError, match="energy_tol apply to backend='gpu-mpi' only"):
        _resolve(["--energy-tol", "1e-9"], config=load_block_config(path))
