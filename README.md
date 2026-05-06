# Petrophys

[![PyPI version](https://badge.fury.io/py/petrophys.svg)](https://badge.fury.io/py/petrophys)
[![Python Version](https://img.shields.io/pypi/pyversions/petrophys.svg)](https://pypi.org/project/petrophys/)
[![Documentation Status](https://readthedocs.org/projects/petrophys/badge/?version=latest)](https://petrophys.readthedocs.io/en/latest/?badge=latest)
[![CI](https://github.com/sambalshikhar/petrophys/actions/workflows/ci.yml/badge.svg)](https://github.com/sambalshikhar/petrophys/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Petrophys is a Python library for deterministic petrophysical well-log
interpretation. It reads LAS/DLIS/CSV data, standardizes curves, runs
petrophysical calculations, and records enough workflow detail to audit or
reproduce the result.

The main path is the dynamic workflow:

1. Load a well.
2. Standardize available curves.
3. Apply explicit parameters from code or LAS `~Params`.
4. Run only the calculations supported by the available inputs.
5. Add interpreted curves back to the well.
6. Export workflow JSON traces when reproducibility matters.

## Install

```bash
pip install petrophys
```

Optional extras:

```bash
pip install petrophys[dlis]   # DLIS support
pip install petrophys[dev]    # tests and development tools
pip install petrophys[docs]   # documentation build tools
pip install petrophys[all]    # all optional dependencies
```

Install from source:

```bash
git clone https://github.com/sambalshikhar/petrophys.git
cd petrophys
pip install -e ".[dev]"
```

## First Workflow

### 1. Read a LAS file

```python
from petrophys.io import read_las

well = read_las("well_data.las")
print(well.name)
print(well.curves)
```

### 2. Define interpretation parameters

Only provide what you know. The workflow will use defaults or skip unsupported
steps when inputs are missing.

```python
from petrophys.workflow import DynamicWorkflowConfig

config = DynamicWorkflowConfig(
    bht_f=220.0,
    td_ft=8500.0,
    rw=0.05,
    salinity_ppm=25000.0,
    lithology="clastic",
    matrix_type="sandstone",
)
```

Common parameters include:

- `rw`, `rw_meas_temp_f`: formation water resistivity and measurement
  temperature.
- `rmf_at_meas_temp`, `rmf_meas_temp_f`: mud filtrate resistivity and
  measurement temperature.
- `bht_f`, `td_ft`: bottom-hole temperature and total depth.
- `lithology`, `matrix_type`: routing controls for porosity, saturation, and
  net-pay cutoffs.
- `a`, `m`, `n`: Archie coefficients.
- `cec`, `rwb`, `b_coeff`: shaly-sand and dual-water model inputs.

### 3. Run the dynamic workflow

```python
from petrophys.workflow import run_dynamic_workflow

result = run_dynamic_workflow(well, config)

print("Executed:", [step.step for step in result.steps_executed])
print("Skipped:", result.steps_skipped)
print("Warnings:", result.warnings)
```

### 4. Use the interpreted curves

Computed curves are added to `result.well`.

```python
interpreted = result.well

phie = interpreted.get("PHIE")      # effective porosity
sw = interpreted.get("SW")          # water saturation
perm = interpreted.get("PERM")      # permeability
net_pay = interpreted.get("NET_PAY")
```

Typical workflow outputs include:

- `VCL`: clay volume
- `PHIT`, `PHIE`: total and effective porosity
- `RW`, `RMF`: resistivity inputs after routing/correction
- `SW`, `SXO`: water saturation and flushed-zone saturation
- `PERM`, `SWIRR_APP`: permeability and apparent irreducible water saturation
- `RQI`, `FZI`, `BVW`, `MHI`: reservoir quality indicators
- `NET_ROCK`, `NET_RES`, `NET_PAY`: reservoir and pay flags

## Reproducible JSON Traces

For audit, benchmarking, or LLM-assisted review, generate workflow JSON traces
directly from a directory of LAS files.

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

Each trace records:

- source file and well name
- original, standardized, and computed curve names
- LAS parameter overrides and unmapped metadata
- zone parameters, when present
- executed and skipped workflow steps
- methods, parameters, decisions, warnings, and errors per step
- final curve summaries for comparison across runs

## LAS Parameters

The zone-aware workflow can read useful LAS `~Params` and map them into
`DynamicWorkflowConfig`. Common aliases include `RW`, `RWT`, `RMF`, `RMFT`,
`BHT`, `TD`, `WSAL`, `SAL`, `CLFL`, `LITH`, `MATRIX`, `A`, `M`, `N`,
`RTSH`, `CEC`, `B_COEFF`, and `RWB`.

```python
from petrophys.io import read_las
from petrophys.workflow import run_dynamic_workflow_with_zones

well = read_las("well_with_params.las")
result = run_dynamic_workflow_with_zones(well)
```

## Calculation Modules

You can use the workflow as the high-level entry point, or call modules
directly for focused calculations.

```python
from petrophys.clay_volume import vcl_from_gr
from petrophys.petrophysics.saturation import archie
from petrophys.petrophysics.permeability import coates_denoo
from petrophys.reservoir import calculate_rock_quality

vcl = vcl_from_gr(gr, gr_min=20.0, gr_max=120.0)
sw = archie(rt, rw=0.05, phi=0.18, a=1.0, m=2.0, n=2.0)
perm = coates_denoo(phie, swi=0.22)
quality = calculate_rock_quality(perm_k=100.0, phi_e=0.2)
```

Main packages:

- `petrophys.io`: readers, writers, and context JSON exporters.
- `petrophys.qaqc`: curve standardization, units, bad-hole, and range QC.
- `petrophys.clay_volume`, `petrophys.resistivity`, `petrophys.petrophysics`:
  core interpretation calculations.
- `petrophys.reservoir`: BVW, MHI, net pay, and rock quality.
- `petrophys.workflow`: standard, dynamic, zone-aware, and batch workflows.
- `petrophys.analysis` and `petrophys.visualization`: review summaries and
  plots.

## Command-Line Scripts

```bash
# Generate per-well workflow JSON traces
python scripts/generate_workflow_json_traces.py \
  --input-dir /path/to/las_with_params \
  --output-dir /path/to/output
```

The `scripts/` folder also includes context-package generators for
formation/lithology and category-first review packages.

## Tests

```bash
pytest
```

## Documentation And Releases

- Documentation: [https://petrophys.readthedocs.io](https://petrophys.readthedocs.io)
- Release notes: [docs/release_notes.rst](docs/release_notes.rst)
- Release process: [RELEASE.md](RELEASE.md)
- License: [LICENSE](LICENSE)

## Citation

```bibtex
@software{petrophys,
  title = {Petrophys: A Python Library for Petrophysical Analysis},
  author = {Petrophys Development Team},
  year = {2026},
  url = {https://github.com/sambalshikhar/petrophys}
}
```
