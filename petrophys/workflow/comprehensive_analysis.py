"""
Convenience runner for the full document-aligned analysis suite.
"""

from __future__ import annotations

from petrophys.analysis import (
    HydrocarbonAnalysis,
    LithologyAnalysis,
    NumericalStatisticalAnalysis,
    PermeabilityAnalysis,
    PorosityAnalysis,
    SaturationAnalysis,
    WellboreQC,
)
from petrophys.utils.analysis_helpers import ComprehensiveAnalysisResult


def run_comprehensive_analysis(
    well: "Well",
    add_curves: bool = True,
    store_results: bool = True,
) -> ComprehensiveAnalysisResult:
    """Run all document-aligned analysis categories."""
    qc = WellboreQC().analyze(well, add_curves=add_curves)
    numerical = NumericalStatisticalAnalysis().analyze(well, add_curves=add_curves)
    lithology = LithologyAnalysis().analyze(well, add_curves=add_curves)
    porosity = PorosityAnalysis().analyze(well, add_curves=add_curves)
    permeability = PermeabilityAnalysis().analyze(well, add_curves=add_curves)
    saturation = SaturationAnalysis().analyze(well, add_curves=add_curves)
    hydrocarbon = HydrocarbonAnalysis().analyze(well, add_curves=add_curves)

    result = ComprehensiveAnalysisResult(
        categories={
            qc.category_key: qc,
            lithology.category_key: lithology,
            porosity.category_key: porosity,
            permeability.category_key: permeability,
            saturation.category_key: saturation,
            hydrocarbon.category_key: hydrocarbon,
            numerical.category_key: numerical,
        }
    )

    if store_results:
        for key, category_result in result.categories.items():
            well.set_result(key, category_result.to_dict()[key])
        well.set_result("COMPREHENSIVE_ANALYSIS", result.to_dict())

    return result
