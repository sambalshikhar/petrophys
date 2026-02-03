# Petrophys

[![PyPI version](https://badge.fury.io/py/petrophys.svg)](https://badge.fury.io/py/petrophys)
[![Python Version](https://img.shields.io/pypi/pyversions/petrophys.svg)](https://pypi.org/project/petrophys/)
[![Documentation Status](https://readthedocs.org/projects/petrophys/badge/?version=latest)](https://petrophys.readthedocs.io/en/latest/?badge=latest)
[![CI](https://github.com/petrophys/petrophys/actions/workflows/ci.yml/badge.svg)](https://github.com/petrophys/petrophys/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive petrophysics library for well log analysis and formation evaluation.

## Features

- **Curve Standardization**: Unified mnemonic names and unit handling
- **Quality Control**: Bad hole detection, log normalization
- **Formation Evaluation**: Clay volume, lithology determination
- **Porosity Calculation**: Density, neutron, sonic, and combined methods
- **Water Saturation**: Archie, Simandoux, Indonesia, Waxman-Smits, Dual Water models
- **Permeability Estimation**: Timur, Wyllie-Rose, Coates-Denoo equations
- **Reservoir Characterization**: Net pay, BVW, MHI calculations
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

# Create visualization
from petrophys.visualization import create_interpretation_plot
plot = create_interpretation_plot(result.well, depth_range=(8000, 8500))
plot.save("interpretation.png")
```

## Module Organization

The library is organized into umbrella modules:

| Module | Description |
|--------|-------------|
| `petrophys.core` | Core data structures (Well, Curve, Units) |
| `petrophys.io` | File I/O (LAS, DLIS, ASCII) |
| `petrophys.qaqc` | Quality control (bad hole, normalization, standardization) |
| `petrophys.environmental` | Environmental parameters (temperature) |
| `petrophys.resistivity` | Resistivity calculations (Rmf, Rw, invasion) |
| `petrophys.formation` | Formation characterization (clay volume, lithology) |
| `petrophys.petrophysics` | Core calculations (porosity, saturation, permeability) |
| `petrophys.reservoir` | Reservoir evaluation (BVW, MHI, net pay) |
| `petrophys.visualization` | Plotting and display |
| `petrophys.summation` | Summary and reporting |
| `petrophys.workflow` | Workflow orchestration |

## Documentation

Full documentation is available at [https://petrophys.readthedocs.io](https://petrophys.readthedocs.io)

- [Installation Guide](https://petrophys.readthedocs.io/en/latest/installation.html)
- [Quick Start Tutorial](https://petrophys.readthedocs.io/en/latest/quickstart.html)
- [API Reference](https://petrophys.readthedocs.io/en/latest/api/modules.html)
- [Tutorials](https://petrophys.readthedocs.io/en/latest/tutorials/index.html)

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
