"""Test stand-ins for a loaded QrmiRuntimeBlock."""


class QrmiBlockStub:
    """A loaded QrmiRuntimeBlock stand-in that builds ``runtime`` and passes the check."""

    def __init__(self, runtime, resource=None):
        self.runtime = runtime
        self.resource = resource or getattr(runtime, "resource", "test_heron")
        self.scheduler = getattr(runtime, "scheduler", "slurm")
        self.checked = 0

    def get_runtime(self, prepare=None):
        return self.runtime

    def check_backend_access(self, **kwargs):
        self.checked += 1

    def instance_label(self):
        return "main (premium)"
