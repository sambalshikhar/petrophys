# Release Process

This repository uses annotated Git tags and GitHub Releases for package
releases. The release branch is `main`.

## Branch Roles

- `main`: public Python package release branch. Keep this focused on the core
  petrophysics library, workflow tracing, context JSON export, docs, tests, and
  examples that belong in the package.
- `dev`: integration branch for package changes before they are merged to
  `main`.
- `agent_dev`: experimental agent, MCP, CAMEL, multi-agent, benchmark, and
  evaluation work. Do not merge agent-only packages or infrastructure into a
  normal package release from `main`.

## Versioning

Use semantic versioning:

- `MAJOR`: incompatible API or workflow behavior changes.
- `MINOR`: backward-compatible modules, workflows, scripts, or docs.
- `PATCH`: bug fixes and documentation corrections.

Release tags use the format `vMAJOR.MINOR.PATCH`, for example `v2.0.0`.

## Standard Release Checklist

Start from a clean `main` branch:

```bash
git fetch origin --prune --tags
git switch main
git pull --ff-only origin main
git status --short
```

Confirm the version is updated in:

- `pyproject.toml`
- `petrophys/version.py`
- `docs/conf.py`
- `docs/release_notes.rst`

Run the release checks:

```bash
python -m pip install --upgrade pip build twine
python -m pip install -e ".[dev,docs]"
pytest
python -m build
twine check dist/*
python -m sphinx -b html docs docs/_build/html
```

Commit the release prep:

```bash
git add pyproject.toml petrophys/version.py docs/conf.py docs/release_notes.rst README.md
git commit -m "Release v2.0.0"
git push origin main
```

Create and push an annotated tag from `main`:

```bash
git tag -a v2.0.0 -m "Release v2.0.0"
git push origin v2.0.0
```

Create a GitHub Release from that tag:

1. Go to GitHub Releases.
2. Draft a new release using tag `v2.0.0`.
3. Use the matching `docs/release_notes.rst` section as the release body.
4. Publish the release.

Publishing the GitHub Release triggers `.github/workflows/publish.yml`, which
builds the tagged source distribution and wheel, checks the artifacts, uploads
the distribution as a workflow artifact, publishes to TestPyPI, then publishes
to PyPI after the configured `pypi` environment approval.

## Hotfix Releases

For a package hotfix:

```bash
git switch main
git pull --ff-only origin main
git switch -c hotfix/v2.0.1
```

Apply the fix, update the patch version and release notes, run the release
checks, merge the hotfix back to `main`, and tag `v2.0.1` from `main`.

## Post-Release

After the release is published:

- Verify the GitHub Release points at the intended tag.
- Verify the PyPI package page shows the new version.
- Install the package in a fresh environment and import it:

```bash
python -m venv /tmp/petrophys-release-check
source /tmp/petrophys-release-check/bin/activate
python -m pip install --upgrade pip
python -m pip install petrophys==2.0.0
python -c "import petrophys; print(petrophys.__version__)"
```

