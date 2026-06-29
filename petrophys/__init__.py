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
from petrophys.analysis import (
    HydrocarbonAnalysis,
    LithologyAnalysis,
    NumericalStatisticalAnalysis,
    PermeabilityAnalysis,
    PorosityAnalysis,
    SaturationAnalysis,
    WellboreQC,
)
from petrophys.data import sample, list_samples
from petrophys.workflow import run_comprehensive_analysis
from petrophys.imputation import (
    CurveImputationResult,
    PretrainedCurveImputer,
    impute_las_file,
    impute_missing_curve,
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
    "HydrocarbonAnalysis",
    "LithologyAnalysis",
    "PorosityAnalysis",
    "PermeabilityAnalysis",
    "SaturationAnalysis",
    "NumericalStatisticalAnalysis",
    "WellboreQC",
    "run_comprehensive_analysis",
    "sample",
    "list_samples",
    "CurveImputationResult",
    "PretrainedCurveImputer",
    "impute_las_file",
    "impute_missing_curve",
]
