# Petrophys

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Documentation Status](https://readthedocs.org/projects/petrophys/badge/?version=latest)](https://petrophys.readthedocs.io/en/latest/?badge=latest)
[![CI](https://github.com/sambalshikhar/petrophys/actions/workflows/ci.yml/badge.svg)](https://github.com/sambalshikhar/petrophys/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive petrophysics library for well log analysis and formation evaluation.

## Features

- **Curve Standardization**: Unified mnemonic names and unit handling
- **Quality Control**: Bad hole detection, log normalization, washout flagging
- **Formation Evaluation**: Clay volume (GR, SP, N-D methods), clay resistivity (Rcl), lithology determination (multi-mineral solver)
- **Porosity Calculation**: Density, neutron, sonic, total (weighted N-D), effective (clay-corrected), and secondary porosity
- **Hydrocarbon Correction**: Preliminary porosity-based NPHI and RHOB corrections for oil/gas zones
- **Water Saturation**: 5 models with configurable, input-aware heuristic routing — Archie, Simandoux, Indonesia, Waxman-Smits, Dual Water
- **Permeability Estimation**: Coates-Denoo, Timur, Wyllie-Rose equations with impermeability detection
- **Rock Quality**: Reservoir Quality Index (RQI), Flow Zone Indicator (FZI), normalized porosity (phi_z)
- **Reservoir Characterization**: Net pay (lithology-aware cutoffs), BVW, MHI calculations
- **Dynamic Workflow**: Input-aware pipeline that auto-detects available curves and runs applicable modules
- **Formation Analysis**: Per-formation summaries from common wide or long CSV top tables with report export
- **Per-Interval Lithology**: CSV-driven per-depth lithology for Sw model routing and net pay cutoffs
- **Visualization**: Log plots, crossplots, interpretation displays

## Installation

### From GitHub

```bash
pip install "petrophys @ git+https://github.com/sambalshikhar/petrophys.git"
```

### With optional dependencies

The pretrained ML extra requires Python 3.11 or newer and installs the model's
supported scikit-learn, XGBoost, and joblib version series.

```bash
# For DLIS file support
pip install "petrophys[dlis] @ git+https://github.com/sambalshikhar/petrophys.git"

# For pretrained ML and DLIS runtime support
pip install "petrophys[all] @ git+https://github.com/sambalshikhar/petrophys.git"
```

### Development checkout

```bash
git clone https://github.com/sambalshikhar/petrophys.git
cd petrophys
pip install -e ".[all,dev,docs]"
```

## Quick Start

### First Plot

```python
import petrophys as pp
well = pp.read(pp.sample("30_6-5"))  # or your own LAS path
pp.plot(well, depth_range=(2800, 3000)).show()
```

Formations and lithology are optional. Plotting displays existing measurements;
it does not invent missing saturation or require interpretation parameters.

To interpret, use `result = pp.interpret(well, rw=0.07)`, then
`result.raise_for_errors().plot().show()`. The Rw here is an illustrative
assumption, not a value to reuse blindly. This API works on a copy; the
lower-level `run_dynamic_workflow` below modifies its input.
Inspect `result.warnings`, `result.specific_steps_skipped`, and
`result.to_trace_dict()` before using calculated results.

### Detailed Interpretation Configuration

```python
from petrophys.io import read_las
from petrophys.workflow import run_dynamic_workflow, DynamicWorkflowConfig

well = read_las("well_data.las")

# The dynamic workflow inspects the well's curves and runs only
# applicable modules in the correct pipeline order
config = DynamicWorkflowConfig(
    # Temperature (optional — skipped if BHT not provided)
    bht_f=220,
    td_ft=8500,

    # Formation water
    rw=0.05,
    salinity_ppm=25000,

    # Lithology controls Sw model routing and net pay cutoffs
    lithology="clastic",              # or "carbonate"
    # lithology_csv="lithology.csv",  # optional; enable only with a real file
    matrix_type="sandstone",

    # HC correction
    hc_type="oil",           # or "gas"

    # Archie parameters (from SCAL if available)
    a=1.0, m=2.0, n=2.0,
)

result = run_dynamic_workflow(well, config)

# See which steps ran vs were skipped
print(f"Executed: {[s.step for s in result.steps_executed]}")
print(f"Skipped:  {result.steps_skipped}")
print(f"Warnings: {result.warnings}")

# All computed curves are added to the well automatically:
# VCL, PHIE, PHIT, PHIS, PHI_SEC, SW, SXO, PERM, BVW, MHI,
# RQI, FZI, PHI_Z, NET_ROCK, NET_RES, NET_PAY, etc.
# (MHI needs invaded-zone resistivity, RXO/RSHA; it is not written
# when Sxo rests on the quick-look fallback)
```

`StandardWorkflow` remains available for compatibility with fixed legacy
configurations, but it is deprecated. New applications and batch processing
use `DynamicWorkflow`.

### Plot LAS With Formations and Lithology

```python
from petrophys.io import read_las
from petrophys.visualization import create_log_viewer

well = read_las("my_well.las")
viewer = create_log_viewer(well, formation_tops="formations.csv",
                          lithology_intervals="lithology.csv")
viewer.show()
```

When supplying optional geology, use the [input templates and format guide](docs/input_templates.rst).
Templates ship with the package, for example `petrophys.sample("lithology_intervals_template")`.
Depth units and datum must match the LAS. External lithology overrides embedded
lithology for display only. Core saturation percentages are normalized to
fractions for plotting; inferred conversions are reported in `viewer.warnings`.
The viewer displays existing measurements, not a newly calculated `SW` curve.

To use external geology for both plotting and interpretation, attach it with
`pp.read("my_well.las", formations="formations.csv", lithology="lithology.csv")`.
Then use `pp.plot(well)` or `pp.interpret(well)`. See the
[own-data guide](docs/own_data.rst) for mappings, units, source selection, and diagnostics.

### Granular Module Usage (individual calculations)

```python
# Each module can be used independently
from petrophys.interpretation.saturation import dual_water, waxman_smits, archie
from petrophys.evaluation import calculate_rock_quality, calculate_rqi, calculate_fzi
from petrophys.interpretation.clay_volume import calculate_rcl, vcl_from_gr
from petrophys.interpretation.permeability import flag_impermeable

# Direct function calls for Sw
sw = archie(rt=10.0, rw=0.05, phi=0.15)
sw = dual_water(phi_t=0.2, rt=10.0, rw=0.05, rwb=0.02, vcl=0.3, phi_cl=0.15)
sw = waxman_smits(phi_t=0.2, rt=10.0, rw=0.05, cec=0.1, rho_ma=2.65, b_coeff=0.5)  # cec in meq/g

# Rock quality
rq = calculate_rock_quality(perm_k=100, phi_e=0.2)
# Returns: {'RQI': ..., 'phi_z': ..., 'FZI': ...}

# Clay resistivity from pure-shale zones
rcl = calculate_rcl(rdep_array, vcl_array, vcl_threshold=0.9)

# Impermeability detection
imperm_flags = flag_impermeable(phi_s, phi_d, phi_n)
```

## Water Saturation Model Selection

The library can route Sw calculations using this configurable quick-look heuristic:

```
VCL < cutoff (0.3)  →  CLEAN  →  Archie

VCL >= cutoff        →  SHALY:
    Carbonate                    →  Dual Water
    Clastic + CEC + b_coeff      →  Waxman-Smits
    Clastic + salinity >= 20kppm →  Simandoux
    Clastic + salinity < 20kppm  →  Indonesia
```

Override with `sw_model="archie"` (or any model name) to force a specific model.

## Pipeline Execution Order

The dynamic workflow follows this sequence:

| Step | Module | Description |
|------|--------|-------------|
| 1 | Temperature | Linear gradient from BHT |
| 2 | Bad Hole | Washout and data quality flags |
| 3 | VCLAY | GR + SP + N-D methods, minimum selection |
| 4 | Rcl | Clay resistivity from pure-shale zones |
| 5 | Rmf | Mud filtrate resistivity (temp-corrected) |
| 6 | HC Correction | NPHI and RHOB hydrocarbon corrections |
| 7 | Porosity | Total, effective, density, sonic, secondary |
| 8 | Lithology | Adaptive multi-mineral solver using RHOB and available NPHI, PEF, or DTC responses |
| 9 | Porosity Refinement | Re-run porosity with computed-lithology matrices |
| 10 | Core Correction | Calibrate porosity/perm to core (if core curves present) |
| 11 | Rw | Formation water resistivity (given / salinity / SP) |
| 12 | Invasion | Rxo/Rt classification |
| 13 | Saturation | Sw via auto-routed model + Sxo |
| 14 | Permeability | Coates/Timur + impermeability flags |
| 15 | Rock Quality | RQI, FZI, phi_z |
| 16 | BVW | Bulk Volume Water |
| 17 | MHI | Moveable Hydrocarbon Index |
| 18 | Net Pay | 3-stage cutoff filtering |

> Note: Lithology runs **after** the first porosity pass; the *Porosity Refinement*
> step then recomputes porosity using the computed-lithology matrix endpoints.

Each step runs only if the required input curves are available. Steps can be explicitly skipped via `config.skip_steps`.

## Module Organization

| Module | Description |
|--------|-------------|
| `petrophys.model` | Wells, curves, core data, intervals, units, and metadata |
| `petrophys.io` | LAS, DLIS, ASCII, CSV, and basic file writers |
| `petrophys.preprocessing` | Standardization, normalization, quality control, and imputation |
| `petrophys.interpretation` | Scientific calculations organized by property and specialist domain |
| `petrophys.evaluation` | Net pay, reservoir quality, flow units, volumetrics, and summaries |
| `petrophys.workflow` | Standard, dynamic, zoned, and batch orchestration |
| `petrophys.reporting` | Delivery LAS/JSON packages and audit reports |
| `petrophys.visualization` | Log panels, crossplots, and specialist displays |
| `petrophys.datasets` | Small packaged sample datasets |

See the [architecture guide](docs/architecture.rst) for package ownership and
dependency rules.

## Documentation

Full documentation is available at [https://petrophys.readthedocs.io](https://petrophys.readthedocs.io)

- [Installation Guide](https://petrophys.readthedocs.io/en/latest/installation.html)
- [Quick Start Tutorial](https://petrophys.readthedocs.io/en/latest/quickstart.html)
- [API Reference](https://petrophys.readthedocs.io/en/latest/api/index.html)
- [Tutorials](https://petrophys.readthedocs.io/en/latest/tutorials/index.html)

## Contributing

Contributions are welcome! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

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
