"""
Petrophys - A comprehensive petrophysics library for well log analysis.

This library provides tools for:
- Curve standardization and quality control
- Formation evaluation (clay volume, lithology)
- Porosity calculation (density, neutron, sonic)
- Water saturation modeling (Archie, Simandoux, Indonesia, etc.)
- Permeability estimation
- Net pay determination
- Visualization and reporting
"""

from petrophys.version import __version__
from petrophys.core.well import Well
from petrophys.core.curve import Curve
from petrophys.core.data_types import DataType
from petrophys.core.classifications import (
    VclClassification,
    InvasionClassification,
    PorosityClassification,
    HcSaturationClassification,
    PermeabilityClassification,
    MhiClassification,
)

__all__ = [
    "__version__",
    "Well",
    "Curve",
    "DataType",
    "VclClassification",
    "InvasionClassification",
    "PorosityClassification",
    "HcSaturationClassification",
    "PermeabilityClassification",
    "MhiClassification",
]
