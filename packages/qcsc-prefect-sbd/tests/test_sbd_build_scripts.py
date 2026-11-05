"""Build presets must select the requested API revision in reused SBD checkouts."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

NATIVE = Path(__file__).resolve().parents[1] / "native"
SCRIPTS = ["build_sbd.sh", "build_sbd_fugaku.sh", "build_sbd_gpu.sh", "build_sbd_slurm.sh"]


def git(directory, *arguments):
    return subprocess.check_output(
        [
            "git",
            "-c",
            "user.name=SBD test",
            "-c",
            "user.email=sbd-test@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "core.hooksPath=/dev/null",
            "-C",
            str(directory),
            *arguments,
        ],
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()


@pytest.mark.parametrize("script", SCRIPTS)
@pytest.mark.parametrize("missing_revision", [False, True])
def test_build_reused_checkout_at_requested_revision(tmp_path, script, missing_revision):
    remote = tmp_path / "remote"
    remote.mkdir()
    git(remote, "init", "-q")
    marker = remote / "api.txt"
    marker.write_text("old API\n")
    git(remote, "add", "api.txt")
    git(remote, "commit", "-qm", "old API")
    old = git(remote, "rev-parse", "HEAD")
    marker.write_text("current API\n")
    git(remote, "commit", "-qam", "current API")
    requested = git(remote, "rev-parse", "HEAD")

    dependency = tmp_path / "dependency"
    dependency.mkdir()
    git(dependency, "init", "-q")
    git(dependency, "remote", "add", "origin", str(remote))
    if missing_revision:
        git(dependency, "fetch", "--depth=1", "origin", old)
        git(dependency, "checkout", "--detach", "FETCH_HEAD")
    else:
        git(dependency, "fetch", "origin")
        git(dependency, "checkout", "--detach", old)

    native = tmp_path / "native"
    native.mkdir()
    shutil.copy(NATIVE / script, native / script)
    shutil.copy(NATIVE / "main.cc", native / "main.cc")
    # Set the default revision file, rather than overriding SBD_COMMIT, so the test
    # checks that every preset uses the common default as well as checking out it.
    (native / "SBD_REVISION").write_text(requested + "\n")
    tools = tmp_path / "bin"
    tools.mkdir()
    (tools / "module").write_text("#!/bin/sh\nexit 0\n")
    compiler = tools / "compiler"
    compiler.write_text(
        "#!/bin/sh\n"
        'while [ "$#" -gt 0 ]; do\n'
        '  if [ "$1" = "-o" ]; then shift; touch "$1"; fi\n'
        "  shift\n"
        "done\n"
    )
    for executable in (tools / "module", compiler):
        executable.chmod(0o755)
    for name in ("mpiicpc", "mpiFCCpx", "mpic++", "mpicxx"):
        (tools / name).symlink_to(compiler)
    environment = dict(os.environ, PATH=f"{tools}:{os.environ['PATH']}", SBD_DIR=str(dependency))
    environment.pop("SBD_COMMIT", None)
    environment.pop("CCCOM", None)
    run = subprocess.run(
        ["bash", str(native / script)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    assert git(dependency, "rev-parse", "HEAD") == requested
    assert (dependency / "api.txt").read_text() == "current API\n"
    assert f"Using SBD commit: {requested}" in run.stdout
    expected_binary = "diag-gpu" if script == "build_sbd_gpu.sh" else "diag"
    assert (native / expected_binary).exists()
