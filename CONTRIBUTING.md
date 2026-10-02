# Contributing to Petrophys

Thank you for helping improve Petrophys. Contributions should preserve both
scientific traceability and predictable behavior on incomplete well-log data.

## Development setup

```bash
git clone https://github.com/sambalshikhar/petrophys.git
cd petrophys
python -m pip install -e ".[all,dev,docs,tutorials]"
```

Execute the affected tutorials in fresh kernels before opening a pull request:

```bash
python scripts/execute_tutorials.py --output-dir tutorial-results
python scripts/check_notebook_outputs.py
```

Build the documentation when changing public APIs, examples, or docstrings:

```bash
sphinx-build -W -b html docs docs/_build/html
```

## Where code goes

- A measurement-to-property calculation goes in its scientific domain under
  `petrophys.interpretation`.
- A cutoff, classification, aggregation or reservoir decision goes under
  `petrophys.evaluation`.
- File parsing and basic serialization go in `petrophys.io`; complete delivery
  products go in `petrophys.reporting`.
- Cleaning, standardization and quality operations go in
  `petrophys.preprocessing`.
- Sequencing and automatic method selection go in `petrophys.workflow`.
- Plotting stays out of scientific calculation modules.
- Shared data structures stay dependency-light and live in `petrophys.model`.
- Supported user APIs are exported from the relevant package `__init__.py`.

See the [architecture page](https://petrophys.netlify.app/architecture.html)
for the package map and dependency direction.

## Scientific changes

- State the equation, units, assumptions, and literature or vendor reference.
- Verify scalar and array inputs, missing values, invalid physical values,
  and ascending or descending depth where applicable.
- Preserve imported interpreted curves unless replacement is explicitly
  requested and recorded in the workflow trace.
- Record routing decisions, defaults, warnings, and skipped prerequisites.
- Avoid silently inventing unavailable measurements or metadata.

## Data and examples

- Do not commit proprietary data or machine-specific paths.
- Document dataset provenance, redistribution permission, and transformations.
- Keep generated plots and reports out of commits unless they are intentional
  documentation outputs.
- Use Git LFS for approved native-format fixtures that cannot be kept small.

## Pull requests

Keep changes focused, describe the scientific and software behavior affected,
and include the commands used for verification. New public functions should
have docstrings, exports, and a reproducible example.

## Tutorials

Library plots use the shared style in `petrophys.visualization.style`. Use
`use_plot_style()` in notebooks that also draw directly with Matplotlib, and
`save_figure(fig, path)` for consistent exports. Preserve the conventional
colours of curves, minerals and image logs. Publisher figures and source-data
images are evidence, not generated plots, and must not be restyled.

The notebooks in `examples/` are rendered in the documentation with their
saved outputs, so re-run any notebook you change in a fresh kernel before
committing it:

```bash
python -m pip install -e ".[tutorials,mcp]"
python scripts/execute_tutorials.py examples/part_a_foundations/03_clay_volume.ipynb --inplace
python scripts/check_notebook_outputs.py
```

The second script fails on notebooks with unexecuted cells or with outputs
that contain local paths.

## Releasing (maintainers)

Repository pushes, documentation deployments and PyPI releases are separate
steps.

**Documentation.** Build with warnings as errors, then deploy the built
`docs/_build/html` folder with the landing page copied over it (the steps are
in `netlify.toml`):

```bash
python -m pip install -e ".[docs,mcp]"
python -m sphinx -W --keep-going -b html docs docs/_build/html
cp -r docs/landing/. docs/_build/html/
```

After deploying, check that the site serves the intended commit and open the
quickstart, context, input-template and tutorial pages.

**PyPI.** `.github/workflows/publish.yml` builds and checks the
distributions, tests the wheel and publishes with PyPI trusted publishing.
Before triggering it, check the version, the data licences in
`THIRD_PARTY_DATA.md` and the wheel contents, and configure the PyPI/TestPyPI
trusted publisher and GitHub environments. Never put upload tokens in source
files or notebooks.
