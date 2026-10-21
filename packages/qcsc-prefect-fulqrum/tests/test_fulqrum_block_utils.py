from __future__ import annotations

import pytest

# Prefect-side modules; the compute-node venv has no Prefect.
pytest.importorskip("prefect")
pytest.importorskip("qiskit_addon_sqd")

from qcsc_prefect_fulqrum import block_utils as mod  # noqa: E402

EXE = "/opt/venvs/fulqrum/bin/qcsc-fulqrum-solve"


@pytest.fixture
def saved(monkeypatch):
    records: list[tuple[str, str, bool, object]] = []

    def fake_save(kind):
        def _save(self, name, overwrite=False):
            records.append((kind, name, overwrite, self))
            return self

        return _save

    monkeypatch.setattr(mod, "register_fulqrum_block_types", lambda: None)
    monkeypatch.setattr(mod.CommandBlock, "save", fake_save("command"), raising=False)
    monkeypatch.setattr(mod.ExecutionProfileBlock, "save", fake_save("execution"), raising=False)
    monkeypatch.setattr(mod.HPCProfileBlock, "save", fake_save("hpc"), raising=False)
    monkeypatch.setattr(mod.FulqrumSolverJob, "save", fake_save("solver"), raising=False)
    return records


def _block(records, kind):
    return next(model for k, _, _, model in records if k == kind)


def test_create_fulqrum_blocks_slurm(saved, tmp_path):
    names = mod.create_fulqrum_blocks(
        hpc_target="slurm",
        queue="normal",
        root_dir=str(tmp_path / "jobs"),
        fulqrum_executable=EXE,
        ompthreads=8,
        slurm_memory="16G",
        eigensolver="primme",
        hamiltonian_format="matrix_free",
        tol=1e-10,
    )

    assert names == {
        "command_block_name": "cmd-fulqrum-solver",
        "execution_profile_block_name": "exec-fulqrum-slurm",
        "hpc_profile_block_name": "hpc-slurm-fulqrum",
        "solver_block_name": "fulqrum-solver",
    }
    assert [kind for kind, *_ in saved] == ["command", "execution", "hpc", "solver"]
    assert all(overwrite for _, _, overwrite, _ in saved)

    command = _block(saved, "command")
    execution = _block(saved, "execution")
    hpc = _block(saved, "hpc")
    solver = _block(saved, "solver")

    assert command.command_name == execution.command_name == "fulqrum"
    assert command.default_args == []
    assert execution.launcher == "single"
    assert (execution.num_nodes, execution.mpiprocs, execution.ompthreads) == (1, 1, 8)
    assert execution.environments == {"OMP_NUM_THREADS": "8"}
    assert hpc.hpc_target == "slurm"
    assert hpc.queue_cpu == "normal"
    assert hpc.slurm_memory == "16G"
    assert hpc.executable_map == {"fulqrum_solver": EXE}
    assert solver.root_dir == str((tmp_path / "jobs").resolve())
    assert solver.script_filename == "fulqrum_solver.slurm"
    assert solver.metrics_artifact_key == "slurm-fulqrum-metrics"
    assert solver.hpc_profile_block_name == "hpc-slurm-fulqrum"
    assert (solver.eigensolver, solver.hamiltonian_format, solver.tol) == (
        "primme",
        "matrix_free",
        1e-10,
    )


def test_explicit_omp_num_threads_is_kept(saved, tmp_path):
    mod.create_fulqrum_blocks(
        hpc_target="slurm",
        root_dir=str(tmp_path),
        fulqrum_executable=EXE,
        ompthreads=8,
        environments={"OMP_NUM_THREADS": "4", "OMP_PROC_BIND": "close"},
    )
    assert _block(saved, "execution").environments == {
        "OMP_NUM_THREADS": "4",
        "OMP_PROC_BIND": "close",
    }


def test_no_ompthreads_leaves_environment_alone(saved, tmp_path):
    mod.create_fulqrum_blocks(hpc_target="slurm", root_dir=str(tmp_path), fulqrum_executable=EXE)
    execution = _block(saved, "execution")
    assert execution.ompthreads is None
    assert execution.environments == {}


@pytest.mark.parametrize(
    ("target", "script", "kwargs"),
    [
        ("local", "fulqrum_solver", {}),
        ("miyabi", "fulqrum_solver.pbs", {"project": "gz00", "queue": "regular-c"}),
        (
            "fugaku",
            "fulqrum_solver.pjm",
            {"project": "ra000000", "queue": "small", "pjm_resources": ["freq=2200"]},
        ),
    ],
)
def test_create_fulqrum_blocks_other_targets(saved, tmp_path, target, script, kwargs):
    names = mod.create_fulqrum_blocks(
        hpc_target=target, root_dir=str(tmp_path), fulqrum_executable=EXE, **kwargs
    )
    assert names["execution_profile_block_name"] == f"exec-fulqrum-{target}"
    assert names["hpc_profile_block_name"] == f"hpc-{target}-fulqrum"
    hpc = _block(saved, "hpc")
    solver = _block(saved, "solver")
    assert hpc.hpc_target == target
    assert _block(saved, "execution").launcher == "single"
    assert solver.script_filename == script
    assert solver.metrics_artifact_key == f"{target}-fulqrum-metrics"
    if target != "local":
        assert hpc.project_cpu == kwargs["project"]
        assert hpc.queue_cpu == kwargs["queue"]
    if target == "fugaku":
        assert hpc.pjm_resources == ["freq=2200"]


def test_custom_block_names(saved, tmp_path):
    names = mod.create_fulqrum_blocks(
        hpc_target="slurm",
        root_dir=str(tmp_path),
        fulqrum_executable=EXE,
        command_block_name="cmd-x",
        execution_profile_block_name="exec-x",
        hpc_profile_block_name="hpc-x",
        solver_block_name="solver-x",
        script_filename="x.slurm",
        metrics_artifact_key="x-metrics",
    )
    assert names == {
        "command_block_name": "cmd-x",
        "execution_profile_block_name": "exec-x",
        "hpc_profile_block_name": "hpc-x",
        "solver_block_name": "solver-x",
    }
    solver = _block(saved, "solver")
    assert (solver.command_block_name, solver.execution_profile_block_name) == ("cmd-x", "exec-x")
    assert solver.script_filename == "x.slurm"
    assert solver.metrics_artifact_key == "x-metrics"


def test_local_rejects_shell_setup(tmp_path):
    with pytest.raises(ValueError, match="modules or pre_commands"):
        mod.create_fulqrum_blocks(
            hpc_target="local",
            root_dir=str(tmp_path),
            fulqrum_executable=EXE,
            modules=["gcc"],
            pre_commands=["source env.sh"],
        )
