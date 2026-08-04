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
from petrophys.model.well import Well
from petrophys.model.curve import Curve
from petrophys.model.data_types import DataType
from petrophys.model.classifications import (
    VclClassification,
    InvasionClassification,
    PorosityClassification,
    HcSaturationClassification,
    PermeabilityClassification,
    RQIClassification,
    MhiClassification,
)
from petrophys.evaluation.diagnostics import (
    HydrocarbonAnalysis,
    LithologyAnalysis,
    NumericalStatisticalAnalysis,
    PermeabilityAnalysis,
    PorosityAnalysis,
    SaturationAnalysis,
)
from petrophys.preprocessing.quality.wellbore import WellboreQC
from petrophys.datasets import sample, list_samples
from petrophys.workflow import run_comprehensive_analysis
from petrophys.preprocessing.imputation import (
    CurveImputationResult,
    ModelCompatibilityError,
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
    "RQIClassification",
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
    "ModelCompatibilityError",
    "PretrainedCurveImputer",
    "impute_las_file",
    "impute_missing_curve",
]
