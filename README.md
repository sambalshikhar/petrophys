# Petrophys

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Documentation](https://img.shields.io/badge/docs-petrophys.netlify.app-087f73.svg)](https://petrophys.netlify.app)
[![CI](https://github.com/sambalshikhar/petrophys/actions/workflows/ci.yml/badge.svg)](https://github.com/sambalshikhar/petrophys/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Petrophys turns well logs, core, and specialist measurements into traceable
petrophysical interpretations: clay volume, porosity, water saturation,
permeability, rock quality, and net pay.

- **Input-aware.** The workflow checks which curves and parameters a well
  actually has and runs every step it can. Incomplete log suites give a
  reduced interpretation, not a crash.
- **Traceable.** Every step records its method, inputs, outputs, and
  parameters; every skipped step records why. Results export as JSON for
  review or for LLM agents.
- **Honest about data.** Implausible or mislabelled curves, geology that misses
  the logged interval, and invalid settings are reported, never silently
  "fixed".

## Installation

```bash
pip install petrophys
```

Until the first PyPI release, install from GitHub:

```bash
pip install "petrophys @ git+https://github.com/sambalshikhar/petrophys.git"
```

Optional extras:

| Extra | Adds | Install |
|-------|------|---------|
| `dlis` | DLIS reading (logs and borehole images) | `pip install "petrophys[dlis]"` |
| `ml` | Pretrained curve imputation (Python 3.11+) | `pip install "petrophys[ml]"` |
| `mcp` | MCP server for LLM clients (Python 3.10+) | `pip install "petrophys[mcp]"` |
| `all` | Everything above | `pip install "petrophys[all]"` |

Development checkout:

```bash
git clone https://github.com/sambalshikhar/petrophys.git
cd petrophys
pip install -e ".[all,dev,docs]"
```

## Quick start

```python
import petrophys as pp

well = pp.read(pp.sample("30_6-5"))        # or pp.read("my_well.las")
result = pp.interpret(well, rw=0.07)       # Rw in ohm.m; use your own value
result.raise_for_errors()

result.print_trace()                       # what ran, what was skipped, and why
pp.plot(result, depth_range=(2800, 3000)).save("interpretation.png")

result.to_las("interpreted.las")           # curves plus the settings used
result.to_json("trace.json")               # full machine-readable trace
```

`pp.read`, `pp.interpret`, and `pp.plot` are the front door. `interpret` works
on a copy of the well; `result.well` holds the interpreted curves (`VCL`,
`PHIE`, `SW`, `PERM`, `NET_PAY`, ...) and `result.config` the settings that
were actually used, including automatically resolved matrix and lithology.

### Your own data

```python
well = pp.read(
    "my_well.las",
    formations="formations.csv",   # formation_name, top_depth[, bottom_depth]
    lithology="lithology.csv",     # top, bottom, lithology
)
result = pp.interpret(well, rw=0.05, bht_f=210)
```

CSV and TSV logs need their depth unit: `pp.read("logs.csv", depth_unit="m")`.
Vendor mnemonics (`GR_EDTC`, `RHOZ`, `TNPH`, ...) and units (kg/m3, percent
porosity) are standardized automatically, in plots and in interpretation.
Templates for every input ship with the package, e.g.
`pp.sample("formation_tops_template")`; see the
[input templates guide](https://petrophys.netlify.app/input_templates) and the
[own-data guide](https://petrophys.netlify.app/own_data).

### Configuring the interpretation

Every setting of `pp.interpret` is a field of `DynamicWorkflowConfig`, with
units in the name (`bht_f` in degF, `td_ft` in feet) and validation on
creation:

```python
config = pp.DynamicWorkflowConfig(
    rw=0.05, rw_meas_temp_f=75,      # formation water
    bht_f=220, td_ft=8500,           # temperature profile
    sw_model="indonesia",            # or "auto" to route per sample
    a=1.0, m=2.0, n=2.0,             # Archie parameters, e.g. from SCAL
    skip_steps=["permeability"],
)
result = pp.interpret(well, config)
```

A misspelled or invalid setting fails immediately with the accepted values,
e.g. `hc_type='gass' is not valid; use one of: oil, gas (did you mean 'gas'?)`.

### Checking the result

```python
result.status                  # "interpreted", "no_interpretation" or "failed"
result.summary(top=2841, bottom=2958)   # compact JSON-ready overview of a depth window
result.warnings                # data problems found on the way
result.specific_steps_skipped  # e.g. {"core_correction": "Missing core porosity (PHIC/CPOR), ..."}
result.to_trace_dict()         # every step's method, inputs, outputs, parameters
```

Data problems also raise `petrophys.DataQualityWarning`, for example when a
density curve labelled g/cc is really in kg/m3. Turn them into errors with
`warnings.simplefilter("error", pp.DataQualityWarning)`.

## Use with LLM agents (MCP)

Petrophys includes a [Model Context Protocol](https://modelcontextprotocol.io)
server, so Claude Desktop, Claude Code, Cursor, VS Code and other MCP clients
can load wells, interpret them, read the trace, summarize zones, plot, and
export results on your machine:

```bash
pip install "petrophys[mcp]"                       # Python 3.10+
claude mcp add petrophys -- petrophys-mcp --root /path/to/wells
```

The model calls the same validated workflow as `pp.interpret`; invalid settings
come back with the accepted values, and file access is limited to the `--root`
folders. See the [LLM agents guide](https://petrophys.netlify.app/mcp). For
code-generating assistants, the docs also serve
[llms.txt](https://petrophys.netlify.app/llms.txt), and
`petrophys.workflow.config_schema()` describes every setting as data.

## Workflow steps

`pp.interpret` runs these steps in order. Each runs only when its inputs are
available; `result.specific_steps_skipped` explains every skip.

| # | Step | Computes |
|---|------|----------|
| 1 | `normalization` | Canonical curve aliases and units, implausible-value screening |
| 2 | `temperature` | Formation temperature profile from BHT or gradient |
| 3 | `bad_hole` | Washout and density-correction quality flags |
| 4 | `spectral_gamma` | Clay indicators from potassium, thorium, uranium |
| 5 | `elemental_spectroscopy` | Mineral indicators from elemental yields |
| 6 | `vclay` | Clay volume from GR, SP, and neutron-density |
| 7 | `rcl` | Clay resistivity from shale intervals |
| 8 | `rmf` | Mud-filtrate resistivity at formation temperature |
| 9 | `hc_correction` | Neutron-density hydrocarbon correction (opt-in) |
| 10 | `porosity` | Density, neutron, sonic, total, effective porosity |
| 11 | `lithology` | Mineral volumes from an adaptive multi-mineral solve |
| 12 | `porosity_refinement` | Porosity with per-depth lithology endpoints |
| 13 | `core_correction` | Porosity and permeability calibrated to core |
| 14 | `rw` | Formation water resistivity at formation temperature |
| 15 | `invasion` | Invasion profile from shallow and deep resistivity |
| 16 | `saturation` | Water and flushed-zone saturation |
| 17 | `permeability` | Core-calibrated or empirical permeability |
| 18 | `rock_quality` | RQI, FZI, normalized porosity |
| 19 | `bvw` | Bulk volume water |
| 20 | `mhi` | Moveable hydrocarbon index |
| 21 | `net_pay` | Net rock, net reservoir, and net pay flags |

With `sw_model="auto"`, water saturation is routed per sample: Archie below the
clay-volume cutoff (0.3); in shaly rock, Dual Water for carbonates,
Waxman-Smits when CEC and B are given, Simandoux for salinity of at least
20,000 ppm, and Indonesia otherwise. North Sea and Nigeria-Schlumberger are
available by name.

## Individual calculations

Every model is also a plain function on NumPy arrays or floats:

```python
import numpy as np
from petrophys.interpretation.saturation import archie, dual_water
from petrophys.interpretation.clay_volume import vcl_from_gr
from petrophys.evaluation import calculate_rock_quality

vcl = vcl_from_gr(np.array([35.0, 80.0, 120.0]), gr_min=30.0, gr_max=130.0)
sw = archie(rt=20.0, rw=0.05, phi=0.22)
sw_shaly = dual_water(phi_t=0.2, rt=10.0, rw=0.05, rwb=0.02, vcl=0.3, phi_cl=0.15)
quality = calculate_rock_quality(perm_k=100.0, phi_e=0.2)   # {"RQI", "phi_z", "FZI"}
```

## Package layout

| Package | Contents |
|---------|----------|
| `petrophys.model` | Wells, curves, core data, intervals, units, constants |
| `petrophys.io` | LAS, DLIS, CSV/ASCII, and geology readers; LAS/CSV/Excel writers |
| `petrophys.preprocessing` | Standardization, quality control, curve imputation |
| `petrophys.interpretation` | Scientific calculations by property and specialist domain |
| `petrophys.evaluation` | Net pay, rock quality, diagnostics, formation summaries |
| `petrophys.workflow` | The dynamic workflow, zoned and batch processing |
| `petrophys.reporting` | Delivery packages and LLM context export |
| `petrophys.visualization` | Log plots, crossplots, specialist displays |
| `petrophys.datasets` | Packaged sample wells and input templates |

See the [architecture guide](https://petrophys.netlify.app/architecture) for
dependency rules.

## Documentation

- [Documentation home](https://petrophys.netlify.app)
- [Installation](https://petrophys.netlify.app/installation)
- [Quickstart](https://petrophys.netlify.app/quickstart)
- [Tutorials](https://petrophys.netlify.app/tutorials/)
- [API reference](https://petrophys.netlify.app/api/)
- [Changelog](CHANGELOG.md)

## Contributing

Contributions are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md) for the
development setup, tests, and pull-request checklist.

## License

The Petrophys source code is available under the MIT License; see
[LICENSE](LICENSE). Bundled third-party data and trained artifacts retain their
source terms and attribution requirements; see
[THIRD_PARTY_DATA.md](THIRD_PARTY_DATA.md) and
[`petrophys.datasets`](src/petrophys/datasets/README.md).

## Citation

If you use Petrophys in your research, please cite:

```bibtex
@software{petrophys,
  title = {Petrophys: A Python Library for Petrophysical Analysis},
  author = {Petrophys Contributors},
  year = {2026},
  url = {https://github.com/sambalshikhar/petrophys}
}
```
