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
  "fixed". Values the workflow has to assume, such as the temperature of a
  given Rw or a temperature gradient, are reported as warnings.

## Installation

```bash
python -m pip install petrophys
```

Until the first release is on PyPI, install from source:

```bash
git clone https://github.com/sambalshikhar/petrophys.git
cd petrophys
python -m pip install -e ".[all]"
```

Optional extras (add in square brackets, e.g. `pip install "petrophys[dlis,mcp]"`):

| Extra | Adds |
|-------|------|
| `dlis` | DLIS reading (logs and borehole images) |
| `ml` | Pretrained curve imputation (Python 3.11+) |
| `mcp` | MCP server for LLM clients (Python 3.10+) |
| `all` | Everything above |

Contributors: `python -m pip install -e ".[all,dev,docs]"`.

## Quick start

```python
import petrophys as pp

well = pp.read(pp.sample("30_6-5"))        # or pp.read("my_well.las")
result = pp.interpret(well, rw=0.07, rw_meas_temp_f=230)   # Rw (ohm.m) at 230 degF; use your own
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
result = pp.interpret(well, rw=0.05, rw_meas_temp_f=75, bht_f=210)
```

Give Rw (ohm.m) with the temperature it was measured at, `rw_meas_temp_f`
(degF); the workflow corrects it along the well's temperature profile. Without
that temperature, a given Rw, or a header RW without RWT, is taken at the
well's median formation temperature, with a warning.

CSV and TSV logs need their depth unit: `pp.read("logs.csv", depth_unit="m")`.
Vendor mnemonics (`GR_EDTC`, `RHOZ`, `TNPH`, ...) and units (kg/m3, percent
porosity) are standardized automatically, in plots and in interpretation.
Physically implausible values are removed on the way, and a curve that is
mostly implausible, which usually means a wrong unit label, raises a warning.
A per-sample lithology code log is read from the curve named by the
`lithology_curve` setting, else from `LITHOLOGY_GEOLINK`, else from a curve
under a lithology-code name, such as `LITHOLOGY` or `FACIES`, that holds
lithology codes. Its codes are decoded with the GeoLink/Kingdom code table
whatever the curve is called, and a warning names any curve other than
`LITHOLOGY_GEOLINK`.
Templates for every input ship with the package, e.g.
`pp.sample("formation_tops_template")`; see the
[input templates guide](https://petrophys.netlify.app/input_templates) and the
[own-data guide](https://petrophys.netlify.app/own_data).

### Configuring the interpretation

Every setting of `pp.interpret` is a field of `DynamicWorkflowConfig`, with
units in the name (`bht_f` in degF, `td_ft` in feet) and validation on
creation. The values here are placeholders for your well's:

```python
config = pp.DynamicWorkflowConfig(
    rw=0.05, rw_meas_temp_f=75,      # formation water
    bht_f=220, td_ft=8500,           # temperature profile
    sw_model="indonesia",            # or "auto" (the default) to route per sample
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
result.warnings                # data problems and assumed values
result.specific_steps_skipped  # e.g. {"core_correction": "Missing core porosity (PHIC/CPOR), ..."}
result.to_trace_dict()         # every step's method, inputs, outputs, parameters
```

Data problems also raise `petrophys.DataQualityWarning`, for example when a
density curve labelled g/cc is really in kg/m3. Turn them into errors with
`warnings.simplefilter("error", pp.DataQualityWarning)`. Values the workflow
had to assume because the data lacked them, such as the temperature of a given
Rw, a temperature gradient or Swirr, are listed in `result.warnings` and in
the `~Other` section of `result.to_las()`.

**Read whole-well totals per formation.** A default run uses one
formation-water salinity for the whole well, with Rw corrected only for
temperature, while shallow formation waters are usually fresher; the clay
indicators also disagree in undercompacted mudstones. Shallow Tertiary
sections that no lithology log covers can therefore show marginal pay, with
SW about 0.5. In the quick-start run on 30_6-5, 21.3 m of the 49.7 m of net
pay lies in the overburden above 1500 m and 25.5 m in the oil column at
2841-2907 m; a default run on 25_5-1 puts 259 m of its 310.4 m of net pay
above its oil zones, which start at 2984 m. Attach formation tops and read
`result.formation_summary()` (or pass `top` and `bottom` to `result.summary`),
or give such a section its own Rw in a zoned run (header zones such as
`TOP_Z1`, `BOT_Z1`, `RW_Z1`; see the
[workflows guide](https://petrophys.netlify.app/workflows.html)). A
formation-water salinity that varies with depth is not available yet.

## Custom workflows

Build your own reusable recipe from built-in steps and your own functions,
check it against a well, run it on one well or a whole field, and share it as
JSON:

```python
from petrophys.workflow import steps

wf = pp.Workflow("Brent quick look", [
    steps.clay_volume(method="gr", gr_clean=20, gr_shale=110),
    steps.porosity(method="density", matrix_density=2.65),
    steps.formation_water(rw=0.07, rw_temp_f=230),
    steps.saturation(model="archie"),
    steps.net_pay(vcl_cutoff=0.35, phie_cutoff=0.10, sw_cutoff=0.60),
])
run = wf.run(well)                    # checks inputs first; full trace
run.summary(top=2841, bottom=2958)
wf.save("brent_quick_look.json")      # pp.Workflow.load(...) later
```

`steps.interpret(...)` runs the whole prebuilt workflow as one step, so you
can add your own steps around it. See the
[custom workflows guide](https://petrophys.netlify.app/custom_workflows.html).

## Context JSON

```python
context = pp.build_context(result)
pp.export_context(result, "well_context.json")
```

The versioned API enforces byte budgets and preserves existing files by default.
It supports formation/lithology packages and the `petrophys-context` command.
Packages use your interpretation settings. [Calculation profiles](#calculation-profiles)
explains how to select dataset configurations for benchmark context.
By default, the context classifies each property with its table in
`petrophys.model.CLASS_TABLES`, so a property gets the same class in every
category; reservoir quality needs PHIE, PERM and VCL (`NO_QUALITY_DATA`
otherwise) and uses the limits of each sample's lithology family, clastic or
carbonate.
See the [context guide](https://petrophys.netlify.app/context.html) and [interpretation review guide](https://petrophys.netlify.app/data_quality.html).
For GeoLink, COSTA and Danish benchmark reproduction, see the
[benchmark context guide](docs/benchmark_context.md), with commands, required
inputs, output layouts and known reproduction limits.
Net-pay output is cutoff screening, not confirmed pay; review warnings and calibration.

## Use with LLM agents (MCP)

Petrophys includes a [Model Context Protocol](https://modelcontextprotocol.io)
server, so Claude, ChatGPT/OpenAI, Cursor, VS Code, Codex and other MCP clients
can load wells, interpret them, read the trace, summarize zones, plot, and
export results:

```bash
python -m pip install "petrophys[mcp]"                         # Python 3.10+
petrophys-mcp config claude-code --root /path/to/wells         # prints the exact client setup
petrophys-mcp serve --http --root /path/to/wells --token auto  # or serve remote clients over HTTP
petrophys-mcp check --root /path/to/wells                      # verify folders and a test session
```

`docker compose up` runs the HTTP server in a container, and
`examples/mcp_openai_agent.py` connects an OpenAI model to a local server.

The model calls the same validated workflow as `pp.interpret`; invalid settings
come back with the accepted values, and file access is limited to the `--root`
folders. To interpret with the calculations of the GeoLink eval context
benchmark, start the server with `--calculation-profile geolink` (see
[Calculation profiles](#calculation-profiles)). See the
[LLM agents guide](https://petrophys.netlify.app/mcp). For
code-generating assistants, the docs also serve
[llms.txt](https://petrophys.netlify.app/llms.txt), and
`petrophys.workflow.config_schema()` describes every setting as data.

## Workflow steps

`pp.interpret` runs these steps in order. Each runs only when its inputs are
available; `result.specific_steps_skipped` explains every skip.

| # | Step | Computes |
|---|------|----------|
| 1 | `normalization` | Canonical curve aliases and units, implausible-value screening |
| 2 | `input_context` | Records which formation and lithology sources were used |
| 3 | `temperature` | Formation temperature profile from BHT or gradient |
| 4 | `bad_hole` | Washout and density-correction quality flags |
| 5 | `vclay` | Clay volume: the lower of GR and neutron-density; SP reported |
| 6 | `rcl` | Clay resistivity from shale intervals |
| 7 | `rmf` | Mud-filtrate resistivity at formation temperature |
| 8 | `hc_correction` | Neutron-density hydrocarbon correction (opt-in) |
| 9 | `porosity` | Density, neutron, sonic, total, effective porosity |
| 10 | `lithology` | Mineral volumes: clastic multi-mineral or carbonate solve |
| 11 | `core_correction` | Porosity and permeability calibrated to core (porosity fitted to PHIT, used only with >= 8 pairs, slope > 0, r^2 >= 0.5) |
| 12 | `rw` | Formation water resistivity at formation temperature |
| 13 | `invasion` | Rxo/Rt from the flushed-zone RXO, then shallow RSHA/RESS, with RMED/RESM as a proxy |
| 14 | `saturation` | Water and flushed-zone saturation |
| 15 | `permeability` | Core-calibrated or empirical permeability |
| 16 | `rock_quality` | RQI, FZI, normalized porosity |
| 17 | `bvw` | Bulk volume water |
| 18 | `mhi` | Moveable hydrocarbon index |
| 19 | `net_pay` | Net rock, net reservoir, and net pay flags |

Spectral-gamma and elemental-spectroscopy interpretation are separate
functions in `petrophys.interpretation.geochemistry`; `pp.interpret` lists
them, and `porosity_refinement`, as skipped. The
[glossary](https://petrophys.netlify.app/glossary.html) lists every curve the
workflow adds.

With `sw_model="auto"` (the default), water saturation is routed per sample.
Samples with VCL below `sw_routing_vcl_cutoff` (by default `vcl_cutoff`, or 0.3
when that is unset) get Archie. In shaly rock, carbonate samples get Dual Water
(Clavier, Coates and Dumanoir, 1984) when the bound-water resistivity `rwb` is
given; Rwb is never estimated. All other shaly samples get Waxman-Smits
(Waxman and Smits, 1968) when `cec` (meq/100 g) and `b_coeff`
((S/m)/(meq/cm3)) are given, Simandoux when the formation-water salinity
`salinity_ppm` is at least 20,000 ppm, and otherwise Indonesia, the published
equation of Poupon and Leveaux (1971). The Waxman-Smits saturation, solved on
total porosity, is reported on the effective-porosity basis of the other
models; `pp.interpret` and `steps.saturation` convert `cec` to the meq/g that
the `waxman_smits()` function takes. North Sea and Nigeria-Schlumberger are
available by name. `sw_model="dual_water"` without `rwb`, or `"waxman_smits"`
without `cec` and `b_coeff`, fails the saturation step with a message naming
the missing input. With oil-based mud (`mud_type="OBM"`) no SXO or MHI is
computed, and Rw is not estimated from the SP or Rxo/Rt.

Net pay comes from cutoffs on VCL, PHIE and SW. Cutoffs left unset take the
standard of each sample's lithology family
(`petrophys.evaluation.STANDARD_CUTOFFS`): clastic VCL < 0.40, PHIE > 0.08,
SW < 0.60; carbonate (limestone, chalk, dolomite, marl) VCL < 0.30,
PHIE > 0.045, SW < 0.70. Samples that no lithology log or interval table
classifies take the clastic standards, or the carbonate ones with
`lithology_domain="carbonate"`. Evaporites (from the lithology log or
intervals, or halite recognized from its log response), coal and bad hole are
left out of net reservoir and net pay. The caliper washout and large-hole
tests for `BAD_HOLE` run only where a bit size is known (the BS curve, else a
plausible 3-36 in header value); elsewhere `BAD_HOLE` comes from |DRHO| >
0.15 g/cm3 alone, and a warning names the depths where a caliper reading has
no bit size. The trace lists the cutoffs used per family and the thickness
each exclusion removed.
`steps.net_pay` applies the same rules (`exclude_non_reservoir=False` for the
cutoffs alone).

## Calculation profiles

`calculation_profile` selects an interpretation configuration: `"current"` is
the default, and `"geolink"` provides the GeoLink evaluation configuration.
Use a dataset `context_profile` to generate benchmark context: it applies
the collection's preparation, interpretation, calibration and export rules.
See the [benchmark context guide](docs/benchmark_context.md) for commands and
required inputs.

For an MCP evaluation harness, `petrophys-mcp serve --calculation-profile geolink`
or `PETROPHYS_CALCULATION_PROFILE=geolink` selects the GeoLink configuration.
Settings supplied in a tool call take precedence. The environment variable
is read by the MCP server only.

Dataset profiles are available for GeoLink, COSTA and Danish:

```python
pp.export_context(
    "PATH/eval", "outputs/geolink_context",
    layout="category_formation_lithology", context_profile="geolink",
    tops_file="PATH/NPD stratigraphic picks north sea.xlsx",
    lithology_mapping_file="PATH/Litho_Nomenclature_Kingdom.xlsx",
)
pp.export_context(
    None, "outputs/costa_context",  # None: the COSTA LAS bundled with petrophys
    layout="category_formation_lithology", context_profile="costa",
)
```

The Danish profile takes its raw LAS, its tops workbook and the same codebook.
Use `pp.list_context_profiles()` and `pp.describe_context_profile("danish")` to
inspect them. One script reproduces any of the three eval collections from a
LAS file or folder, keeping the interpreted LAS as well:

```bash
python scripts/generate_benchmark_context.py --input-dir PATH/HW-31.las --output-dir outputs/HW-31
```

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
| `petrophys.model` | Wells, curves, core data, intervals, units, constants, class tables |
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
development setup, verification commands, and pull-request checklist.

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
