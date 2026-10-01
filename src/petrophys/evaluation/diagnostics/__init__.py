"""
Diagnostic analyses of an interpreted well: lithology, porosity, saturation, permeability and hydrocarbons.

Each class takes a Well, produces a CategoryAnalysisResult with
sections, derived curves, and metadata aligned with the reference
petrophysical QC / interpretation workflow.
"""

from petrophys.evaluation.diagnostics.hydrocarbon_analysis import HydrocarbonAnalysis
from petrophys.evaluation.diagnostics.lithology_analysis import LithologyAnalysis
from petrophys.evaluation.diagnostics.porosity_analysis import PorosityAnalysis
from petrophys.evaluation.diagnostics.permeability_analysis import PermeabilityAnalysis
from petrophys.evaluation.diagnostics.saturation_analysis import SaturationAnalysis
from petrophys.evaluation.diagnostics.numerical_statistical_analysis import NumericalStatisticalAnalysis

__all__ = [
    "HydrocarbonAnalysis",
    "LithologyAnalysis",
    "PorosityAnalysis",
    "PermeabilityAnalysis",
    "SaturationAnalysis",
    "NumericalStatisticalAnalysis",
]
