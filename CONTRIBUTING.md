# Contributing to Petrophys

Thank you for helping improve Petrophys. Contributions should preserve both
scientific traceability and predictable behavior on incomplete well-log data.

## Development setup

```bash
git clone https://github.com/sambalshikhar/petrophys.git
cd petrophys
python -m pip install -e ".[all,dev,docs]"
```

Run the test suite before opening a pull request:

```bash
python -m pytest
```

Build the documentation when changing public APIs, examples, or docstrings:

```bash
sphinx-build -W -b html docs docs/_build/html
```

## Scientific changes

- State the equation, units, assumptions, and literature or vendor reference.
- Add tests for scalar and array inputs, missing values, invalid physical
  values, and ascending or descending depth where applicable.
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
have docstrings, exports, and focused regression tests.
