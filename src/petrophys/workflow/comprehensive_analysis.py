"""
Convenience runner for the full document-aligned analysis suite.
"""

from __future__ import annotations

from petrophys.evaluation.diagnostics import (
    HydrocarbonAnalysis,
    LithologyAnalysis,
    NumericalStatisticalAnalysis,
    PermeabilityAnalysis,
    PorosityAnalysis,
    SaturationAnalysis,
)
from petrophys.preprocessing.quality.wellbore import WellboreQC
from petrophys._internal.analysis_helpers import ComprehensiveAnalysisResult
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from petrophys.model.well import Well


def run_comprehensive_analysis(
    well: "Well",
    add_curves: bool = True,
    store_results: bool = True,
    class_tables: str = "current",
    lithology_family=None,
) -> ComprehensiveAnalysisResult:
    """Run all document-aligned analysis categories.

    ``class_tables`` is passed to the diagnostics analyzers: ``"current"``
    (default) classifies with the canonical tables of
    :data:`~petrophys.model.class_tables.CLASS_TABLES`, so a property gets the
    same class in every category; ``"benchmark"`` keeps the GeoLink eval
    package's tables. ``lithology_family`` (``"clastic"``, ``"carbonate"`` or
    a lithology, for every sample or one per sample; default clastic) selects
    the limits of the reservoir-quality table under ``"current"``.
    """
    graded = {"class_tables": class_tables, "lithology_family": lithology_family}
    qc = WellboreQC().analyze(well, add_curves=add_curves)
    numerical = NumericalStatisticalAnalysis().analyze(well, add_curves=add_curves)
    lithology = LithologyAnalysis(**graded).analyze(well, add_curves=add_curves)
    porosity = PorosityAnalysis(**graded).analyze(well, add_curves=add_curves)
    permeability = PermeabilityAnalysis(**graded).analyze(well, add_curves=add_curves)
    saturation = SaturationAnalysis(class_tables=class_tables).analyze(well, add_curves=add_curves)
    hydrocarbon = HydrocarbonAnalysis(**graded).analyze(well, add_curves=add_curves)

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
