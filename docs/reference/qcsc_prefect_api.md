# QCSC Prefect API Reference

This page is generated from Python docstrings in the core `qcsc-prefect`
packages. Native Qiskit primitives are documented separately in
[Native Qiskit API](native_qiskit_api.md).

## Core Models

::: qcsc_prefect_core.models.execution_profile.ExecutionProfile
    options:
      show_source: false

## Prefect Blocks

::: qcsc_prefect_blocks.common.blocks.CommandBlock
    options:
      show_source: false

::: qcsc_prefect_blocks.common.blocks.ExecutionProfileBlock
    options:
      show_source: false

::: qcsc_prefect_blocks.common.blocks.HPCProfileBlock
    options:
      show_source: false

## Block-Based Execution

::: qcsc_prefect_executor.from_blocks.SubmissionTarget
    options:
      show_source: false

::: qcsc_prefect_executor.from_blocks.resolve_hpc_target
    options:
      show_source: false

::: qcsc_prefect_executor.from_blocks.resolve_submission_target
    options:
      show_source: false

::: qcsc_prefect_executor.from_blocks.build_scheduler_script_filename
    options:
      show_source: false

::: qcsc_prefect_executor.from_blocks.resolve_scheduler_script_filename
    options:
      show_source: false

::: qcsc_prefect_executor.from_blocks.run_job_from_blocks
    options:
      show_source: false

## Bulk HPC Execution

::: qcsc_prefect_executor.bulk.BulkJobStatus
    options:
      show_source: false

::: qcsc_prefect_executor.bulk.BulkJobSpec
    options:
      show_source: false

::: qcsc_prefect_executor.bulk.BulkJobRecord
    options:
      show_source: false

::: qcsc_prefect_executor.bulk.SubmittedJob
    options:
      show_source: false

::: qcsc_prefect_executor.bulk.BulkRunResult
    options:
      show_source: false

::: qcsc_prefect_executor.bulk.BulkTickResult
    options:
      show_source: false

::: qcsc_prefect_executor.bulk.BulkJobRegistry
    options:
      show_source: false

::: qcsc_prefect_executor.bulk.GlobalFugakuBulkRunner
    options:
      show_source: false

::: qcsc_prefect_core.queue.QueueCapacity
    options:
      show_source: false

::: qcsc_prefect_core.queue.QueueAwareSubmitGate
    options:
      show_source: false

::: qcsc_prefect_executor.bulk.QueueFullError
    options:
      show_source: false

::: qcsc_prefect_executor.bulk.TemporarySubmitError
    options:
      show_source: false

::: qcsc_prefect_executor.bulk.DuplicateJobKeyError
    options:
      show_source: false

::: qcsc_prefect_executor.from_blocks.submit_job_from_blocks
    options:
      show_source: false

::: qcsc_prefect_executor.from_blocks.monitor_jobs_many
    options:
      show_source: false

::: qcsc_prefect_executor.from_blocks.run_jobs_from_blocks_bulk
    options:
      show_source: false

## Miyabi API

::: qcsc_prefect_adapters.miyabi.builder.MiyabiJobRequest
    options:
      show_source: false

::: qcsc_prefect_adapters.miyabi.builder.to_miyabi_template_kwargs
    options:
      show_source: false

::: qcsc_prefect_adapters.miyabi.builder.render_script
    options:
      show_source: false

::: qcsc_prefect_adapters.miyabi.builder.write_script_file
    options:
      show_source: false

::: qcsc_prefect_adapters.miyabi.runtime.SubmitResult
    options:
      show_source: false

::: qcsc_prefect_adapters.miyabi.runtime.MiyabiPBSRuntime
    options:
      show_source: false

::: qcsc_prefect_executor.miyabi.run.MiyabiRunResult
    options:
      show_source: false

::: qcsc_prefect_executor.miyabi.run.run_miyabi_job
    options:
      show_source: false

::: qcsc_prefect_executor.miyabi.from_blocks.run_miyabi_job_from_blocks
    options:
      show_source: false

## Fugaku API

::: qcsc_prefect_adapters.fugaku.builder.FugakuJobRequest
    options:
      show_source: false

::: qcsc_prefect_adapters.fugaku.builder.to_fugaku_template_kwargs
    options:
      show_source: false

::: qcsc_prefect_adapters.fugaku.builder.render_script
    options:
      show_source: false

::: qcsc_prefect_adapters.fugaku.builder.write_script_file
    options:
      show_source: false

::: qcsc_prefect_adapters.fugaku.runtime.SubmitResult
    options:
      show_source: false

::: qcsc_prefect_adapters.fugaku.runtime.FugakuPJMRuntime
    options:
      show_source: false

::: qcsc_prefect_executor.fugaku.run.FugakuRunResult
    options:
      show_source: false

::: qcsc_prefect_executor.fugaku.run.run_fugaku_job
    options:
      show_source: false

## Slurm API

::: qcsc_prefect_adapters.slurm.builder.SlurmJobRequest
    options:
      show_source: false

::: qcsc_prefect_adapters.slurm.builder.to_slurm_template_kwargs
    options:
      show_source: false

::: qcsc_prefect_adapters.slurm.builder.render_script
    options:
      show_source: false

::: qcsc_prefect_adapters.slurm.builder.write_script_file
    options:
      show_source: false

::: qcsc_prefect_adapters.slurm.runtime.SubmitResult
    options:
      show_source: false

::: qcsc_prefect_adapters.slurm.runtime.SlurmRuntime
    options:
      show_source: false

::: qcsc_prefect_executor.slurm.run.SlurmRunResult
    options:
      show_source: false

::: qcsc_prefect_executor.slurm.run.run_slurm_job
    options:
      show_source: false

::: qcsc_prefect_executor.slurm.from_blocks.run_slurm_job_from_blocks
    options:
      show_source: false

## DICE API

::: qcsc_prefect_dice.solver_job.DiceSHCISolverJob
    options:
      show_source: false

::: qcsc_prefect_dice.block_utils.register_dice_block_types
    options:
      show_source: false

::: qcsc_prefect_dice.block_utils.create_dice_blocks
    options:
      show_source: false

::: qcsc_prefect_dice.io_utils.make_job_work_dir
    options:
      show_source: false

::: qcsc_prefect_dice.io_utils.prep_dice_input_files
    options:
      show_source: false

::: qcsc_prefect_dice.io_utils.read_dice_output_files
    options:
      show_source: false

## Fulqrum API

::: qcsc_prefect_fulqrum.solver_job.FulqrumSolverJob
    options:
      show_source: false

::: qcsc_prefect_fulqrum.block_utils.register_fulqrum_block_types
    options:
      show_source: false

::: qcsc_prefect_fulqrum.block_utils.create_fulqrum_blocks
    options:
      show_source: false

::: qcsc_prefect_fulqrum.solver.solve_fulqrum
    options:
      show_source: false

::: qcsc_prefect_fulqrum.solver.FulqrumSolverOptions
    options:
      show_source: false

::: qcsc_prefect_fulqrum.solver.FulqrumSolution
    options:
      show_source: false

::: qcsc_prefect_fulqrum.io_utils.make_job_work_dir
    options:
      show_source: false
