# Tests

Run everything from the repository root:

```bash
python -m unittest discover -s tests -v
```

or one file, from inside `tests/`: `python -m unittest test_reference_values -v`.
The suite tests the checkout it sits in (it puts `src/` first on the import
path) and uses only the bundled sample data. Tests that need an optional
package skip with the install command: `pp.impute` needs scikit-learn
(`petrophys[ml]`), the MCP server Python 3.10+ and `petrophys[mcp]`, the
pressure DLIS samples dlisio (`petrophys[dlis]`).

| File | What it holds fixed |
|---|---|
| `test_public_api.py` | Every public entry point (`sample`, `read`, `read_formations`, `Well`, `interpret`, results, `plot`, `impute`, `export_context`) with each documented input form, and the error each bad input raises; every bundled well interpreted, plotted and exported. The cases are in `api_cases.py`. |
| `test_reference_values.py` | The interpretation of nine bundled wells, including the GeoLink and COSTA benchmark profiles: steps and methods, net thickness, statistics of every curve, and the exported context of the benchmark profiles, to a relative 1e-6. Scenarios are in `reference_cases.py`, stored numbers in `reference/`. |
| `test_science.py` | The textbook equations (Archie and the shaly-sand models in clean rock, density and sonic porosity, Larionov, Timur, Coates-Denoo, RQI/FZI, Arps, brine resistivity from salinity) against values worked by hand; the order of the Rw sources; the depth-shift tools; the plot object. Runs in seconds. |
| `test_nnls.py` | The mineral solvers' non-negative least squares reaches the optimum, whatever the scipy version. |
| `test_mcp_server.py` | An MCP client session: tool list, sample well, interpretation, step trace, zone summaries, plot, export, refused requests. |
| `test_check_scripts.py` | `scripts/check_input_validation.py`, `check_diagnostic_plots.py`, `check_function_schema.py` and `check_notebook_outputs.py` pass. |

## When a reference test fails

The failure lists every quantity that moved, for example
`/curves/PHIE/mean: 0.2555 -> 0.2561`. If the change was not meant to move
it, find the cause. If it was, rewrite that scenario's reference and commit
the JSON with the change, so the diff shows reviewers what moved:

```bash
python tests/make_reference.py 30_6-5
```

## Adding tests

- A new public function or setting: add its accepted inputs and its refused
  inputs to `api_cases.py`, with the error each refusal raises.
- A new calculation that changes outputs of the bundled wells: rewrite the
  references as above; a new kind of well or profile: add a scenario to
  `reference_cases.py` and run `make_reference.py` for it.
- Anything else: a `test_*.py` file using `unittest`, importing `_support`
  first.
