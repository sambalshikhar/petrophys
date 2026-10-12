"""
Wells, curves, intervals, core data and the reference properties shared by the library.

This module provides:
- Well: Container for well data and metadata
- Curve: Data structure for log curves with units and metadata
- DataType: Enumeration for data type classification
- Units: Unit conversion utilities
- Constants: Matrix and fluid properties tables
- Class tables: the canonical classes of each petrophysical property
"""

from petrophys.model.core import CoreData
from petrophys.model.formation_tops import FormationTops, label_depths
from petrophys.model.intervals import (
    Formation,
    Interval,
    IntervalKind,
    ManualInterval,
    SubZone,
    Zone,
)

from petrophys.model.well import Well
from petrophys.model.curve import Curve
from petrophys.model.data_types import DataType
from petrophys.model.class_tables import (
    CLASS_TABLES,
    Bands,
    ClassTable,
    ReservoirGrade,
    ReservoirQualityTable,
    TwoCurveBands,
    TwoCurveClassTable,
)
from petrophys.model.classifications import (
    VclClassification,
    InvasionClassification,
    PorosityClassification,
    HcSaturationClassification,
    PermeabilityClassification,
    RQIClassification,
    MhiClassification,
)
from petrophys.model.units import UnitConverter, UnitRegistry
from petrophys.model.constants import (
    MatrixProperties,
    FluidProperties,
    ArchieCoefficients,
    ArchieParameterDefaults,
)

__all__ = [
    "Well",
    "Curve",
    "DataType",
    "VclClassification",
    "InvasionClassification",
    "PorosityClassification",
    "HcSaturationClassification",
    "PermeabilityClassification",
    "RQIClassification",
    "MhiClassification",
    "CLASS_TABLES",
    "ClassTable",
    "TwoCurveClassTable",
    "Bands",
    "TwoCurveBands",
    "ReservoirQualityTable",
    "ReservoirGrade",
    "UnitConverter",
    "UnitRegistry",
    "MatrixProperties",
    "FluidProperties",
    "ArchieCoefficients",
    "ArchieParameterDefaults",
    "CoreData",
    "Formation",
    "FormationTops",
    "label_depths",
    "Interval",
    "IntervalKind",
    "ManualInterval",
    "SubZone",
    "Zone",
]
