# Petrophys

[![PyPI version](https://badge.fury.io/py/petrophys.svg)](https://badge.fury.io/py/petrophys)
[![Python Version](https://img.shields.io/pypi/pyversions/petrophys.svg)](https://pypi.org/project/petrophys/)
[![Documentation Status](https://readthedocs.org/projects/petrophys/badge/?version=latest)](https://petrophys.readthedocs.io/en/latest/?badge=latest)
[![CI](https://github.com/petrophys/petrophys/actions/workflows/ci.yml/badge.svg)](https://github.com/petrophys/petrophys/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive petrophysics library for well log analysis and formation evaluation.

## Features

- **Curve Standardization**: Unified mnemonic names and unit handling
- **Quality Control**: Bad hole detection, log normalization, washout flagging
- **Formation Evaluation**: Clay volume (GR, SP, N-D methods), clay resistivity (Rcl), lithology determination (multi-mineral solver)
- **Porosity Calculation**: Density, neutron, sonic, total (weighted N-D), effective (clay-corrected), and secondary porosity
- **Hydrocarbon Correction**: Preliminary porosity-based NPHI and RHOB corrections for oil/gas zones
- **Water Saturation**: 5 models with automatic routing — Archie, Simandoux, Indonesia, Waxman-Smits, Dual Water
- **Permeability Estimation**: Coates-Denoo, Timur, Wyllie-Rose equations with impermeability detection
- **Rock Quality**: Reservoir Quality Index (RQI), Flow Zone Indicator (FZI), normalized porosity (phi_z)
- **Reservoir Characterization**: Net pay (lithology-aware cutoffs), BVW, MHI calculations
- **Dynamic Workflow**: Input-aware pipeline that auto-detects available curves and runs applicable modules
- **Formation Analysis**: Per-formation summaries from CSV tops (simple, NPD, Danish formats) with report export
- **Per-Interval Lithology**: CSV-driven per-depth lithology for Sw model routing and net pay cutoffs
- **Visualization**: Log plots, crossplots, interpretation displays

## Installation

### From PyPI (recommended)

```bash
pip install petrophys
```

### With optional dependencies

```bash
# For DLIS file support
pip install petrophys[dlis]

# For development
pip install petrophys[dev]

# For documentation building
pip install petrophys[docs]

# All optional dependencies
pip install petrophys[all]
```

### From Source

```bash
git clone https://github.com/petrophys/petrophys.git
cd petrophys
pip install -e ".[dev]"
```

## Quick Start

### Standard Workflow (fixed configuration)

```python
from petrophys.io import read_las
from petrophys.workflow import run_standard_workflow, WorkflowConfig

# Load well data
well = read_las("well_data.las")

# Configure parameters
config = WorkflowConfig(
    rw=0.05,              # Formation water resistivity
    matrix_density=2.65,  # Sandstone matrix
    a=1.0,                # Archie tortuosity
    m=2.0,                # Cementation exponent
    n=2.0,                # Saturation exponent
    vcl_cutoff=0.3,       # VCL cutoff
    phie_cutoff=0.08,     # Porosity cutoff
    sw_cutoff=0.7,        # Water saturation cutoff
)

# Run interpretation
result = run_standard_workflow(well, config)

# Access computed curves
vcl = result.well.get("VCL")       # Clay volume
phie = result.well.get("PHIE")     # Effective porosity
sw = result.well.get("SW")         # Water saturation
net_pay = result.well.get("NET_PAY")  # Net pay flags
```

### Dynamic Workflow (auto-detects available inputs)

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
    lithology_csv="lithology.csv",    # per-interval lithology (optional)
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
```

### Granular Module Usage (individual calculations)

```python
# Each module can be used independently
from petrophys.petrophysics.saturation import dual_water, waxman_smits, archie
from petrophys.reservoir import calculate_rock_quality, calculate_rqi, calculate_fzi
from petrophys.clay_volume import calculate_rcl, vcl_from_gr
from petrophys.petrophysics.permeability import flag_impermeable

# Direct function calls for Sw
sw = archie(rt, rw=0.05, phi=0.15)
sw = dual_water(phi_t=0.2, phi_e=0.15, rt=10.0, rw=0.05, rwb=0.02, vcl=0.3, phi_cl=0.15)
sw = waxman_smits(phi_t=0.2, rt=10.0, rw=0.05, cec=10.0, rho_ma=2.65, b_coeff=0.5)

# Rock quality
rq = calculate_rock_quality(perm_k=100, phi_e=0.2)
# Returns: {'RQI': ..., 'phi_z': ..., 'FZI': ...}

# Clay resistivity from pure-shale zones
rcl = calculate_rcl(rdep_array, vcl_array, vcl_threshold=0.9)

# Impermeability detection
imperm_flags = flag_impermeable(phi_s, phi_d, phi_n)
```

## Water Saturation Model Selection

The library automatically routes Sw calculations based on formation properties:

```
VCL < cutoff (0.3)  →  CLEAN  →  Archie

VCL >= cutoff        →  SHALY:
    Carbonate                    →  Dual Water
    Clastic + CEC available      →  Waxman-Smits
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
| 5 | Lithology | Multi-mineral solver (RHOB, NPHI, PEF) |
| 6 | Rmf | Mud filtrate resistivity (temp-corrected) |
| 7 | Rw | Formation water resistivity (given / salinity / SP) |
| 8 | Invasion | Rxo/Rt classification |
| 9 | HC Correction | NPHI and RHOB hydrocarbon corrections |
| 10 | Porosity | Total, effective, density, sonic, secondary |
| 11 | Saturation | Sw via auto-routed model + Sxo |
| 12 | Permeability | Coates/Timur + impermeability flags |
| 13 | Rock Quality | RQI, FZI, phi_z |
| 14 | BVW | Bulk Volume Water |
| 15 | MHI | Moveable Hydrocarbon Index |
| 16 | Net Pay | 3-stage cutoff filtering |

Each step runs only if the required input curves are available. Steps can be explicitly skipped via `config.skip_steps`.

## Module Organization

| Module | Description |
|--------|-------------|
| `petrophys.core` | Core data structures (Well, Curve, Units, Classifications) |
| `petrophys.io` | File I/O (LAS, DLIS, ASCII) |
| `petrophys.qaqc` | Quality control (bad hole, normalization, standardization) |
| `petrophys.environmental` | Environmental parameters (temperature profiles) |
| `petrophys.resistivity` | Resistivity calculations (Rmf, Rw, invasion) |
| `petrophys.clay_volume` | Clay volume (GR, SP, N-D methods) and clay resistivity (Rcl) |
| `petrophys.lithology` | Lithology determination (multi-mineral solver, crossplots, CSV intervals) |
| `petrophys.formation` | Interval-based analysis (backward-compatible re-exports) |
| `petrophys.petrophysics` | Core calculations (porosity, saturation, HC correction, permeability) |
| `petrophys.reservoir` | Reservoir evaluation (BVW, MHI, net pay, rock quality) |
| `petrophys.visualization` | Plotting and display |
| `petrophys.summation` | Summary, reporting, and formation-level analysis |
| `petrophys.workflow` | Workflow orchestration (standard, dynamic, batch) |

## Documentation

Full documentation is available at [https://petrophys.readthedocs.io](https://petrophys.readthedocs.io)

- [Installation Guide](https://petrophys.readthedocs.io/en/latest/installation.html)
- [Quick Start Tutorial](https://petrophys.readthedocs.io/en/latest/quickstart.html)
- [API Reference](https://petrophys.readthedocs.io/en/latest/api/modules.html)
- [Tutorials](https://petrophys.readthedocs.io/en/latest/tutorials/index.html)
- [Release Notes](https://petrophys.readthedocs.io/en/latest/release_notes.html)

## Contributing

Contributions are welcome! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

If you use Petrophys in your research, please cite:

```bibtex
@software{petrophys,
  title = {Petrophys: A Python Library for Petrophysical Analysis},
  author = {Petrophys Development Team},
  year = {2024},
  url = {https://github.com/petrophys/petrophys}
}
```
