from __future__ import annotations

import pytest
from qcsc_prefect_sbd import block_utils as mod

EXE = "/opt/sbd/bin/diag"


@pytest.fixture
def saved(monkeypatch):
    records: list[tuple[str, str, bool, object]] = []

    def fake_save(kind):
        def _save(self, name, overwrite=False):
            records.append((kind, name, overwrite, self))
            return self

        return _save

    monkeypatch.setattr(mod, "register_sbd_block_types", lambda: None)
    monkeypatch.setattr(mod.CommandBlock, "save", fake_save("command"), raising=False)
    monkeypatch.setattr(mod.ExecutionProfileBlock, "save", fake_save("execution"), raising=False)
    monkeypatch.setattr(mod.HPCProfileBlock, "save", fake_save("hpc"), raising=False)
    monkeypatch.setattr(mod.SBDSolverJob, "save", fake_save("solver"), raising=False)
    return records


def _block(records, kind):
    return next(model for k, _, _, model in records if k == kind)


def test_create_sbd_blocks_slurm(saved, tmp_path):
    names = mod.create_sbd_blocks(
        hpc_target="slurm",
        queue="normal",
        root_dir=str(tmp_path / "jobs"),
        sbd_executable=EXE,
        num_nodes=2,
        mpiprocs=2,
        ompthreads=4,
        mpi_options=["--mpi=pmix"],
        slurm_memory="8G",
        timeout_seconds=600.0,
        adet_comm_size=2,
        bdet_comm_size=2,
        block=6,
        iteration=3,
        tolerance=1e-8,
        carryover_ratio=0.25,
        carryover_type=2,
    )

    assert names == {
        "command_block_name": "cmd-sbd-solver",
        "execution_profile_block_name": "exec-sbd-slurm",
        "hpc_profile_block_name": "hpc-slurm-sbd",
        "solver_block_name": "sbd-solver",
    }
    assert [kind for kind, *_ in saved] == ["command", "execution", "hpc", "solver"]
    assert all(overwrite for _, _, overwrite, _ in saved)

    command = _block(saved, "command")
    execution = _block(saved, "execution")
    hpc = _block(saved, "hpc")
    solver = _block(saved, "solver")

    assert command.command_name == execution.command_name == "sbd"
    assert command.executable_key == "sbd_solver"
    assert command.default_args == []
    assert execution.profile_name == "sbd-mpi"
    assert execution.launcher == "srun"
    assert execution.mpi_options == ["--mpi=pmix"]
    assert (execution.num_nodes, execution.mpiprocs, execution.ompthreads) == (2, 2, 4)
    assert execution.environments == {"OMP_NUM_THREADS": "4"}
    assert hpc.hpc_target == "slurm"
    assert hpc.queue_cpu == hpc.queue_gpu == "normal"
    assert hpc.slurm_memory == "8G"
    assert hpc.executable_map == {"sbd_solver": EXE}
    assert solver.root_dir == str((tmp_path / "jobs").resolve())
    assert solver.script_filename == "sbd_solver.slurm"
    assert solver.metrics_artifact_key == "slurm-sbd-metrics"
    assert solver.timeout_seconds == 600.0
    assert (solver.command_block_name, solver.execution_profile_block_name) == (
        "cmd-sbd-solver",
        "exec-sbd-slurm",
    )
    assert solver.hpc_profile_block_name == "hpc-slurm-sbd"
    assert (solver.task_comm_size, solver.adet_comm_size, solver.bdet_comm_size) == (1, 2, 2)
    assert (solver.block, solver.iteration, solver.tolerance) == (6, 3, 1e-8)
    assert (solver.carryover_ratio, solver.carryover_type, solver.solver_mode) == (
        0.25,
        2,
        "cpu",
    )


@pytest.mark.parametrize(
    ("target", "launcher", "script", "kwargs"),
    [
        ("local", "single", "sbd_solver", {}),
        ("miyabi", "mpiexec.hydra", "sbd_solver.pbs", {"project": "gz00", "queue": "regular-c"}),
        (
            "fugaku",
            "mpiexec",
            "sbd_solver.pjm",
            {"project": "ra000000", "queue": "small", "pjm_resources": ["freq=2200"]},
        ),
    ],
)
def test_create_sbd_blocks_other_targets(saved, tmp_path, target, launcher, script, kwargs):
    names = mod.create_sbd_blocks(
        hpc_target=target, root_dir=str(tmp_path), sbd_executable=EXE, **kwargs
    )
    assert names["execution_profile_block_name"] == f"exec-sbd-{target}"
    assert names["hpc_profile_block_name"] == f"hpc-{target}-sbd"
    hpc = _block(saved, "hpc")
    solver = _block(saved, "solver")
    assert hpc.hpc_target == target
    assert _block(saved, "execution").launcher == launcher
    assert solver.script_filename == script
    assert solver.metrics_artifact_key == f"{target}-sbd-metrics"
    if target != "local":
        assert hpc.project_cpu == kwargs["project"]
        assert hpc.queue_cpu == kwargs["queue"]
    if target == "fugaku":
        assert hpc.pjm_resources == ["freq=2200"]


def test_fugaku_puts_the_binary_dir_on_ld_library_path(saved, tmp_path):
    mod.create_sbd_blocks(
        hpc_target="fugaku",
        root_dir=str(tmp_path),
        sbd_executable="/vol0001/sbd/bin/diag",
        gfscache="/vol0004",
        spack_modules=["openblas"],
        mpi_options_for_pjm=["-std-proc out"],
    )
    hpc = _block(saved, "hpc")
    assert _block(saved, "execution").environments == {
        "LD_LIBRARY_PATH": "/vol0001/sbd/bin:$LD_LIBRARY_PATH"
    }
    assert hpc.gfscache == "/vol0004"
    assert hpc.spack_modules == ["openblas"]
    assert hpc.mpi_options_for_pjm == ["-std-proc out"]


def test_explicit_environment_is_kept(saved, tmp_path):
    mod.create_sbd_blocks(
        hpc_target="fugaku",
        root_dir=str(tmp_path),
        sbd_executable="/vol0001/sbd/bin/diag",
        ompthreads=8,
        environments={"OMP_NUM_THREADS": "4", "LD_LIBRARY_PATH": "/lib"},
    )
    assert _block(saved, "execution").environments == {
        "OMP_NUM_THREADS": "4",
        "LD_LIBRARY_PATH": "/lib",
    }


def test_explicit_launcher_wins(saved, tmp_path):
    mod.create_sbd_blocks(
        hpc_target="slurm", root_dir=str(tmp_path), sbd_executable=EXE, launcher="mpirun"
    )
    assert _block(saved, "execution").launcher == "mpirun"


def test_custom_block_names(saved, tmp_path):
    names = mod.create_sbd_blocks(
        hpc_target="slurm",
        root_dir=str(tmp_path),
        sbd_executable=EXE,
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
    assert [name for _, name, _, _ in saved] == ["cmd-x", "exec-x", "hpc-x", "solver-x"]
    solver = _block(saved, "solver")
    assert (solver.command_block_name, solver.execution_profile_block_name) == ("cmd-x", "exec-x")
    assert solver.hpc_profile_block_name == "hpc-x"
    assert solver.script_filename == "x.slurm"
    assert solver.metrics_artifact_key == "x-metrics"


def test_local_rejects_shell_setup(saved, tmp_path):
    with pytest.raises(ValueError, match="modules or pre_commands"):
        mod.create_sbd_blocks(
            hpc_target="local",
            root_dir=str(tmp_path),
            sbd_executable=EXE,
            modules=["gcc"],
            pre_commands=["source env.sh"],
        )
    assert saved == []


def test_gpu_solver_mode_on_the_gpu_queue(saved, tmp_path):
    mod.create_sbd_blocks(
        hpc_target="slurm",
        queue="gpu",
        root_dir=str(tmp_path),
        sbd_executable="/opt/sbd/bin/diag-gpu",
        resource_class="gpu",
        gpus_per_node=4,
        solver_mode="gpu",
    )
    execution = _block(saved, "execution")
    assert (execution.resource_class, execution.gpus_per_node) == ("gpu", 4)
    assert _block(saved, "solver").solver_mode == "gpu"


def test_gpu_solver_mode_needs_the_gpu_resource_class(saved, tmp_path):
    with pytest.raises(ValueError, match="solver_mode='gpu'.*resource_class='gpu'"):
        mod.create_sbd_blocks(
            hpc_target="slurm", root_dir=str(tmp_path), sbd_executable=EXE, solver_mode="gpu"
        )
    assert saved == []


@pytest.mark.parametrize(
    ("hpc_target", "gpus", "resource_class", "match"),
    [
        ("fugaku", 1, "gpu", "takes no GPU request"),
        ("local", 1, "gpu", "takes no GPU request"),
        ("miyabi", 4, "gpu", "have 1 GPU"),
        ("slurm", 4, "cpu", "needs resource_class='gpu'"),
    ],
)
def test_a_bad_gpu_request_fails_before_any_block_is_saved(
    saved, tmp_path, hpc_target, gpus, resource_class, match
):
    with pytest.raises(ValueError, match=match):
        mod.create_sbd_blocks(
            hpc_target=hpc_target,
            queue="q",
            project="p",
            root_dir=str(tmp_path),
            sbd_executable=EXE,
            resource_class=resource_class,
            gpus_per_node=gpus,
        )
    assert saved == []


@pytest.mark.parametrize(
    ("num_nodes", "mpiprocs", "comm"),
    [
        (2, 4, (1, 2, 4)),  # Product equals the rank count.
        (2, 4, (1, 2, 2)),  # Product divides it; the rest is the Hamiltonian communicator.
        (1, 6, (3, 1, 2)),
    ],
)
def test_comm_sizes_that_divide_the_rank_count(saved, tmp_path, num_nodes, mpiprocs, comm):
    task, adet, bdet = comm
    mod.create_sbd_blocks(
        hpc_target="slurm",
        root_dir=str(tmp_path),
        sbd_executable=EXE,
        num_nodes=num_nodes,
        mpiprocs=mpiprocs,
        task_comm_size=task,
        adet_comm_size=adet,
        bdet_comm_size=bdet,
    )
    solver = _block(saved, "solver")
    assert (solver.task_comm_size, solver.adet_comm_size, solver.bdet_comm_size) == comm


@pytest.mark.parametrize(
    ("target", "num_nodes", "mpiprocs", "comm", "ranks"),
    [
        ("slurm", 2, 4, (1, 3, 1), 8),
        ("slurm", 1, 4, (1, 2, 4), 4),
        ("miyabi", 1, 4, (3, 1, 1), 4),
        ("local", 1, 4, (1, 2, 1), 1),  # The single launcher runs one rank.
    ],
)
def test_comm_sizes_that_do_not_divide_the_rank_count_fail_before_saving(
    saved, tmp_path, target, num_nodes, mpiprocs, comm, ranks
):
    task, adet, bdet = comm
    with pytest.raises(ValueError, match=rf"= {task * adet * bdet} must divide the {ranks} MPI"):
        mod.create_sbd_blocks(
            hpc_target=target,
            root_dir=str(tmp_path),
            sbd_executable=EXE,
            num_nodes=num_nodes,
            mpiprocs=mpiprocs,
            task_comm_size=task,
            adet_comm_size=adet,
            bdet_comm_size=bdet,
        )
    assert saved == []


def test_package_exports_the_helpers():
    import qcsc_prefect_sbd

    assert qcsc_prefect_sbd.create_sbd_blocks is mod.create_sbd_blocks
    assert qcsc_prefect_sbd.register_sbd_block_types is mod.register_sbd_block_types
