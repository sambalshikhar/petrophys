"""
Analysis module — category-level classification analyses.

Each class takes a Well, produces a CategoryAnalysisResult with
sections, derived curves, and metadata aligned with the reference
petrophysical QC / interpretation workflow.
"""

from petrophys.analysis.hydrocarbon_analysis import HydrocarbonAnalysis
from petrophys.analysis.lithology_analysis import LithologyAnalysis
from petrophys.analysis.porosity_analysis import PorosityAnalysis
from petrophys.analysis.permeability_analysis import PermeabilityAnalysis
from petrophys.analysis.saturation_analysis import SaturationAnalysis
from petrophys.analysis.numerical_statistical_analysis import NumericalStatisticalAnalysis
from petrophys.analysis.quality_check import WellboreQC

__all__ = [
    "HydrocarbonAnalysis",
    "LithologyAnalysis",
    "PorosityAnalysis",
    "PermeabilityAnalysis",
    "SaturationAnalysis",
    "NumericalStatisticalAnalysis",
    "WellboreQC",
]
