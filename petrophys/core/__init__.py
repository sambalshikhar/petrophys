"""
Core module - Fundamental data structures for petrophysics.

This module provides:
- Well: Container for well data and metadata
- Curve: Data structure for log curves with units and metadata
- DataType: Enumeration for data type classification
- Units: Unit conversion utilities
- Constants: Matrix and fluid properties tables
"""

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
from petrophys.core.units import UnitConverter, UnitRegistry
from petrophys.core.constants import MatrixProperties, FluidProperties

__all__ = [
    "Well",
    "Curve",
    "DataType",
    "VclClassification",
    "InvasionClassification",
    "PorosityClassification",
    "HcSaturationClassification",
    "PermeabilityClassification",
    "MhiClassification",
    "UnitConverter",
    "UnitRegistry",
    "MatrixProperties",
    "FluidProperties",
]
