# QCSC Prefect Adapters

Local process and scheduler adapters for QCSC Prefect workflows. Local commands
are executed without a shell or generated job script; scheduler targets use
their Jinja2 script templates.

Slurm single-job execution uses a matching `scontrol` record while the controller
retains the job. Pending and running jobs do not require Slurm accounting.
`sacct` remains the fallback when the record is absent or `scontrol` fails;
transient scheduler failures are retried until the configured timeout.
This does not add historical recovery after a job is purged. Bulk monitoring
continues to use `sacct`.
