# Petrophys

[![PyPI version](https://badge.fury.io/py/petrophys.svg)](https://badge.fury.io/py/petrophys)
[![Python Version](https://img.shields.io/pypi/pyversions/petrophys.svg)](https://pypi.org/project/petrophys/)
[![Documentation Status](https://readthedocs.org/projects/petrophys/badge/?version=latest)](https://petrophys.readthedocs.io/en/latest/?badge=latest)
[![CI](https://github.com/sambalshikhar/petrophys/actions/workflows/ci.yml/badge.svg)](https://github.com/sambalshikhar/petrophys/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Petrophys is a Python library for deterministic petrophysical well log analysis,
formation evaluation, workflow tracing, and agent-ready context generation.

Version 2 turns the package into a reproducible interpretation environment:
it can read LAS files, standardize curves, extract LAS `~Params`, route the
dynamic workflow, compute petrophysical curves, export audit JSON, and build
context packages that an LLM agent can inspect or reproduce.

## Version 2 Highlights

- Dynamic workflow with full step traces, decisions, parameters, curve inputs,
  curve outputs, warnings, and final curve summaries.
- Zone-aware LAS processing that maps `~Params` such as `RW`, `RWT`, `RMF`,
  `BHT`, `CLFL`, `WSAL`, `RBFL`, `MUD`, `A`, `M`, `N`, `RTSH`, and `CEC`
  into `DynamicWorkflowConfig`.
- Context JSON exporters for whole-well, formation-level, formation/lithology,
  and category-first work packages.
- New `petrophys.analysis` package for category-oriented porosity, saturation,
  permeability, lithology, hydrocarbon, statistical, and wellbore QC summaries.
- Expanded petrophysics calculations for RW/RMF routing, saturation routing,
  apparent Swirr, core correction, elastic properties, lithology-aware
  permeability, and log-derived QC.
- Reproducible CLI scripts for workflow trace generation and context package
  generation.
- Template-driven log viewer and batch visualization helpers.
- Broader tests for workflow routing, trace export, standardization, value
  range QC, saturation, permeability, porosity, log viewing, and analysis
  modules.

## Installation

From PyPI:

```bash
pip install petrophys
```

For optional features:

```bash
pip install petrophys[dlis]   # DLIS support
pip install petrophys[dev]    # tests and development tools
pip install petrophys[docs]   # documentation build tools
pip install petrophys[all]    # all optional dependencies
```

From source:

```bash
git clone https://github.com/sambalshikhar/petrophys.git
cd petrophys
pip install -e ".[dev]"
```

## Quick Start

### Generate workflow JSON traces for LAS files

This is the recommended entry point for reproducible, agent-facing runs.

```python
from pathlib import Path
from petrophys.workflow import write_workflow_json_traces_directory

results = write_workflow_json_traces_directory(
    input_dir=Path("geolink_las_with_params"),
    output_dir=Path("workflow_trace_output"),
    pattern="*.las",
)

for item in results:
    print(item.source_file, item.success, item.json_log_file)
```

The generated `workflow_json` files include:

- source LAS file and well name
- input, output, and computed curve names
- global LAS parameter overrides
- zone parameters, when present
- executed workflow steps
- step method names, parameters, decisions, warnings, and errors
- input and output curve summaries per step
- final curve summaries for reproduced statistics
- lithology group summaries and derived-log notes

### Run the dynamic workflow programmatically

```python
from petrophys.io import read_las
from petrophys.workflow import DynamicWorkflowConfig, run_dynamic_workflow

well = read_las("well_data.las")

config = DynamicWorkflowConfig(
    bht_f=220.0,
    td_ft=8500.0,
    rw=0.05,
    salinity_ppm=25000.0,
    lithology="clastic",
    matrix_type="sandstone",
)

result = run_dynamic_workflow(well, config)

print([step.step for step in result.steps_executed])
print(result.steps_skipped)
print(result.warnings)

phie = result.well.get("PHIE")
sw = result.well.get("SW")
perm = result.well.get("PERM")
```

### Export context JSON packages

```python
from pathlib import Path
from petrophys.io import (
    export_reference_json_work_package_by_category_formation_lithology,
    write_reference_curve_name_mapping,
)

results = export_reference_json_work_package_by_category_formation_lithology(
    input_dir=Path("las_with_petrophys_params"),
    output_dir=Path("context_json_by_category"),
    tops_file=Path("formation_tops.xlsx"),
    lithology_mapping_file=Path("lithology_mapping.xlsx"),
    pattern="*.las",
)

mapping_path = write_reference_curve_name_mapping("context_json_by_category")
```

### Use individual calculation modules

```python
from petrophys.clay_volume import calculate_rcl, vcl_from_gr
from petrophys.petrophysics.saturation import archie, dual_water, waxman_smits
from petrophys.petrophysics.permeability import coates_denoo, flag_impermeable
from petrophys.reservoir import calculate_rock_quality

vcl_gr = vcl_from_gr(gr, gr_min=20.0, gr_max=120.0, method="linear")
rcl = calculate_rcl(rdep, vcl_gr, vcl_threshold=0.9)

sw_archie = archie(rt, rw=0.05, phi=0.18, a=1.0, m=2.0, n=2.0)
sw_dual = dual_water(phi_t=0.22, phi_e=0.18, rt=8.0, rw=0.05, rwb=0.02, vcl=0.25, phi_cl=0.15)
sw_ws = waxman_smits(phi_t=0.22, rt=8.0, rw=0.05, cec=10.0, rho_ma=2.65, b_coeff=0.5)

perm = coates_denoo(phie, swirr=0.22)
imperm = flag_impermeable(phi_s, phi_d, phi_n)
rock_quality = calculate_rock_quality(perm_k=100.0, phi_e=0.2)
```

## Dynamic Workflow

The dynamic workflow inspects the available curves and configuration, then
runs only the applicable modules. The current Version 2 order is:

| Step | Workflow step | Main outputs |
|------|---------------|--------------|
| 1 | Temperature | `TEMP` |
| 2 | Bad-hole QC | `BAD_HOLE` |
| 3 | Clay volume | `VCL`, method curves during processing |
| 4 | Clay resistivity | `rcl_computed` in trace |
| 5 | Mud filtrate resistivity | `RMF` |
| 6 | Hydrocarbon correction | `NPHI_HC`, `RHOB_HC` when enabled |
| 7 | Porosity | `PHIN`, `PHID`, `PHIS`, `PHIT`, `PHIE`, `PHI_SEC` |
| 8 | Core correction | `PHIE_CC`, `PERM_CC` when core data is available |
| 9 | Lithology | `LITHO`, mineral fractions, or authoritative input lithology |
| 10 | Formation water resistivity | `RW` |
| 11 | Invasion | `INVASION_RATIO` |
| 12 | Saturation | `SW`, `SXO` |
| 13 | Permeability | `PERM`, `SWIRR_APP`, optional flags |
| 14 | Rock quality | `RQI`, `FZI`, `PHI_Z` |
| 15 | Bulk volume water | `BVW` |
| 16 | Moveable hydrocarbon index | `MHI` |
| 17 | Net pay | `NET_ROCK`, `NET_RES`, `NET_PAY`, `RESFLAG`, `SEALFLAG` |

Steps can be disabled with `DynamicWorkflowConfig(skip_steps=[...])`.

## Saturation Routing

Water saturation can be forced with `sw_model`, or auto-routed. Auto-routing
uses VCL, lithology, salinity, CEC, Rcl, and available curves.

| Formation context | Typical routed model |
|-------------------|----------------------|
| Clean rock below VCL cutoff | Archie |
| Shaly clastic with high salinity | Simandoux |
| Shaly clastic with lower salinity | Indonesia |
| Shaly carbonate | Dual Water |
| Shaly clastic with CEC support | Waxman-Smits |
| North Sea style VSH exponent input | North Sea shaly sand |

The workflow JSON records the selected model, per-model sample counts, Archie
coefficients, RW/RMF ranges, SXO method, and the lithology source.

## LAS Parameter Mapping

When using the zone-aware batch path, recognized LAS `~Params` are mapped into
workflow configuration fields.

| LAS parameter | Config field |
|---------------|--------------|
| `RW` | `rw` |
| `RWT` | `rw_meas_temp_f` |
| `RMF` | `rmf_at_meas_temp` |
| `RMFT` | `rmf_meas_temp_f` |
| `RM`, `RMT` | `rm`, `rm_meas_temp_f` |
| `CLFL` | `rmf_salinity_ppm` |
| `WSAL`, `SAL`, `WATSAL` | `salinity_ppm` |
| `BHT` | `bht_f` |
| `TD` | `td_ft` |
| `MUD`, `MUDTYPE` | `mud_type` |
| `MUDWATER` | `mud_water` |
| `LITH`, `LITHOLOGY` | `lithology` |
| `MATRIX`, `MATRIX_TYPE` | `matrix_type` |
| `A`, `M`, `N` | Archie coefficients |
| `RTSH`, `CEC`, `B_COEFF`, `RWB`, `PNSH` | shaly-sand and dual-water parameters |
| `RBFL`, `RHO_FL`, `RHO_MA`, `RHMA`, `PNMA` | porosity matrix/fluid endpoints |

Unmapped parameters are retained in the workflow JSON as audit metadata.

## Module Map

| Package | Purpose |
|---------|---------|
| `petrophys.base` | Core data structures: `Well`, `Curve`, data types, units, classifications |
| `petrophys.core` | Core plug data, core-to-log calibration, poro-perm transforms, calibration plots |
| `petrophys.formation` | Formation tops, zones, manual intervals, and interval-level summaries |
| `petrophys.io` | LAS, DLIS, CSV/ASCII readers; LAS/CSV/Excel writers; context JSON exporters |
| `petrophys.qaqc` | Curve standardization, unit conversion, value range QC, bad-hole flags, normalization |
| `petrophys.environmental` | Temperature profiles, gradients, and environmental corrections |
| `petrophys.clay_volume` | GR, SP, neutron-density VCL, clay resistivity, VCL classification |
| `petrophys.resistivity` | RMF, RW, salinity conversion, SP/Rxo/Rt heuristics, invasion analysis |
| `petrophys.lithology` | Lithology logs, matrix properties, crossplots, multi-mineral solver |
| `petrophys.petrophysics` | Porosity, saturation, permeability, HC correction, cementation, elastic properties |
| `petrophys.reservoir` | BVW, MHI, net pay, rock quality, reservoir flags |
| `petrophys.analysis` | Category-level analysis summaries for LLM/context workflows |
| `petrophys.workflow` | Standard workflow, dynamic workflow, zone-aware batch workflow, trace export |
| `petrophys.visualization` | Log plots, triple combos, interpretation plots, log viewer, Pickett plots |
| `petrophys.summation` | Formation and zone summaries, averages, NTG, HCPV |
| `petrophys.utils` | Statistics, validation, interpolation, analysis helpers |

## Public Workflow APIs

```python
from petrophys.workflow import (
    DynamicWorkflow,
    DynamicWorkflowConfig,
    run_dynamic_workflow,
    run_dynamic_workflow_with_zones,
    write_workflow_json_traces_directory,
    extract_zoned_parameters,
    process_las_directory_with_workflow,
    write_batch_report,
    write_curve_audit_report,
)
```

## Public Analysis APIs

```python
from petrophys.analysis import (
    PorosityAnalysis,
    SaturationAnalysis,
    PermeabilityAnalysis,
    LithologyAnalysis,
    HydrocarbonAnalysis,
    NumericalStatisticalAnalysis,
    WellboreQC,
)
```

## CLI Scripts

```bash
# Generate per-well workflow_json traces
python scripts/generate_workflow_json_traces.py \
  --input-dir /path/to/las_with_params \
  --output-dir /path/to/output

# Generate formation/lithology context packages
python scripts/generate_context_json_by_formation_lithology.py \
  --input-dir /path/to/las \
  --tops-file /path/to/tops.xlsx \
  --output-dir /path/to/context_json

# Generate category-first formation/lithology context packages
python scripts/generate_category_context_json_by_formation_lithology.py \
  --input-dir /path/to/las \
  --tops-file /path/to/tops.xlsx \
  --output-dir /path/to/category_context_json
```

## Reproducibility for LLM Agents

For an LLM agent, the safest way to reproduce calculations is:

1. Read the LAS through the public workflow trace API.
2. Let the library standardize curves and map LAS `~Params`.
3. Use `write_workflow_json_traces_directory` or `run_dynamic_workflow_with_zones`.
4. Compare `global_parameter_overrides`, `step_results`, and `final_curve_summaries`.
5. Pin the package version or git commit when comparing historical traces.

Exact reproduction of older workflow JSON files can depend on the RW/RMF
temperature reference behavior used at generation time. Current Version 2 traces
record the effective RW/RMF parameters and temperature correction path.

## Tests

```bash
pytest

# Focused checks for Version 2 workflow behavior
pytest tests/test_workflow_trace_export.py
pytest tests/test_dynamic_workflow_routing.py
pytest tests/test_analysis_modules.py
pytest tests/test_saturation_routing_logic.py
pytest tests/test_value_range_qc.py
```

## Documentation

Full documentation is available at
[https://petrophys.readthedocs.io](https://petrophys.readthedocs.io).

- [Installation Guide](https://petrophys.readthedocs.io/en/latest/installation.html)
- [Quick Start Tutorial](https://petrophys.readthedocs.io/en/latest/quickstart.html)
- [API Reference](https://petrophys.readthedocs.io/en/latest/api/modules.html)
- [Tutorials](https://petrophys.readthedocs.io/en/latest/tutorials/index.html)
- [Release Notes](https://petrophys.readthedocs.io/en/latest/release_notes.html)

## Release Notes

See [docs/release_notes.rst](docs/release_notes.rst) for Version 2.0.0 changes,
migration notes, new APIs, behavior changes, and verification guidance.

## License

MIT License. See [LICENSE](LICENSE) for details.

## Citation

```bibtex
@software{petrophys,
  title = {Petrophys: A Python Library for Petrophysical Analysis},
  author = {Petrophys Development Team},
  year = {2026},
  url = {https://github.com/sambalshikhar/petrophys}
}
```
