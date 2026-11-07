# PyPI Release Guide

This page describes the production PyPI release steps for maintainers.

Use the same version for all `qcsc-prefect` packages in one release.

When a release adds a new package under `packages/`, register a pending Trusted Publisher
for that PyPI project (workflow `pypi-publish.yml`, environment `pypi`) before pushing the tag.
`qcsc-prefect-sbd` and `qcsc-prefect-fulqrum` are new in `0.3.1`.

## Step 1: Choose the Release Version

Choose the next release version, for example:

```text
0.1.1
```

The release tag will use the same version with a leading `v`:

```text
v0.1.1
```

## Step 2: Prepare the Release Branch

Open GitHub and run the release preparation workflow from GitHub Actions.

1. Open the `qiskit-community/qcsc-prefect` repository on GitHub.
2. Select the `Actions` tab.
3. Select the `Prepare release` workflow.
4. Click `Run workflow`.
5. Select the `main` branch.
6. Enter `old_version`.
7. Enter `new_version`.
8. Click `Run workflow`.

Example input:

```text
old_version: 0.1.0
new_version: 0.1.1
```

When the workflow succeeds, it creates this branch:

```text
release/v<new_version>
```

The workflow does not create a pull request automatically.

## Step 3: Open the Release Pull Request

Open a pull request manually.

Use this source branch:

```text
release/v<new_version>
```

Use this target branch:

```text
main
```

Use this PR title:

```text
Release v<new_version>
```

Use a short PR body like this:

```markdown
Release preparation for `v<new_version>`.

- Updated package versions.
- Updated internal dependency pins.

After this PR is merged, publish by pushing tag `v<new_version>`.
```

## Step 4: Merge the Release Pull Request

Review the release pull request and merge it into `main`.

After merging, update your local `main` branch:

```bash
git switch main
git pull --ff-only origin main
```

## Step 5: Create and Push the Release Tag

Create a tag that matches the release version exactly.

Example:

```bash
git tag v0.1.1
git push origin v0.1.1
```

Pushing the tag starts the `Publish packages to PyPI` workflow.

## Step 6: Approve the PyPI Publish Job

Open GitHub Actions and select the `Publish packages to PyPI` workflow run for the tag.

If GitHub asks for `pypi` environment approval, approve the deployment.

Wait until the workflow finishes successfully.

Do not add PyPI API tokens or password secrets for this release process.

## Step 7: Verify the Published Packages

After the publish workflow succeeds, verify installation from PyPI in a clean environment.

Core install:

```bash
python -m venv /tmp/qcsc-prefect-release-core
/tmp/qcsc-prefect-release-core/bin/python -m pip install --upgrade pip
/tmp/qcsc-prefect-release-core/bin/python -m pip install "qcsc-prefect==<version>"
/tmp/qcsc-prefect-release-core/bin/python -c "import qcsc_prefect_core, qcsc_prefect_blocks, qcsc_prefect_adapters, qcsc_prefect_executor"
```

All extras install:

```bash
python -m venv /tmp/qcsc-prefect-release-all
/tmp/qcsc-prefect-release-all/bin/python -m pip install --upgrade pip
/tmp/qcsc-prefect-release-all/bin/python -m pip install "qcsc-prefect[all]==<version>"
/tmp/qcsc-prefect-release-all/bin/python -c "import qcsc_prefect.integrations.qiskit, qcsc_prefect_dice, qcsc_prefect_sbd, qcsc_prefect_fulqrum"
```

The `fulqrum` extra installs only the Python-side integration. The fulqrum solver itself is
built per site with `packages/qcsc-prefect-fulqrum/native/build_fulqrum.sh`, because the
fulqrum version it needs is not on PyPI.

Replace `<version>` with the release version without the leading `v`, for example `0.1.1`.
