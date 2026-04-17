"""
Petrophys - A comprehensive petrophysics library for well log analysis.

This library provides tools for:
- Curve standardization and quality control
- Clay volume (GR, SP, N-D methods) and clay resistivity
- Lithology determination (multi-mineral solver, crossplots, CSV intervals)
- Porosity calculation (density, neutron, sonic)
- Water saturation modeling (Archie, Simandoux, Indonesia, etc.)
- Permeability estimation
- Net pay determination
- Visualization and reporting
"""

from petrophys.version import __version__
from petrophys.base.well import Well
from petrophys.base.curve import Curve
from petrophys.base.data_types import DataType
from petrophys.base.classifications import (
    VclClassification,
    InvasionClassification,
    PorosityClassification,
    HcSaturationClassification,
    PermeabilityClassification,
    MhiClassification,
)
from petrophys.lithology import LithologyAnalysis
from petrophys.petrophysics.permeability import PermeabilityAnalysis
from petrophys.petrophysics.porosity import PorosityAnalysis
from petrophys.petrophysics.saturation import SaturationAnalysis
from petrophys.qaqc import WellboreQC
from petrophys.reservoir import HydrocarbonAnalysis
from petrophys.utils import NumericalStatisticalAnalysis
from petrophys.workflow import run_comprehensive_analysis

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
    "HydrocarbonAnalysis",
    "LithologyAnalysis",
    "PorosityAnalysis",
    "PermeabilityAnalysis",
    "SaturationAnalysis",
    "NumericalStatisticalAnalysis",
    "WellboreQC",
    "run_comprehensive_analysis",
]
