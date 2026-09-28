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
from petrophys.api import read, plot, interpret
from petrophys.exceptions import DataQualityWarning
from petrophys.io import read_formations, read_lithology_intervals
from petrophys.io.lithology_reader import read_lithology_codes
from petrophys.reporting.context import ContextSizeError, build_context, export_context
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
from petrophys.workflow import (
    DynamicWorkflowConfig,
    DynamicWorkflowResult,
    run_comprehensive_analysis,
    run_dynamic_workflow,
)
from petrophys.preprocessing.imputation import (
    CurveImputationResult,
    ModelCompatibilityError,
    PretrainedCurveImputer,
    impute_las_file,
    impute_missing_curve,
)

__all__ = [
    "read",
    "plot",
    "interpret",
    "build_context",
    "export_context",
    "ContextSizeError",
    "read_lithology_codes",
    "DynamicWorkflowConfig",
    "DynamicWorkflowResult",
    "run_dynamic_workflow",
    "DataQualityWarning",
    "read_formations",
    "read_lithology_intervals",
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
