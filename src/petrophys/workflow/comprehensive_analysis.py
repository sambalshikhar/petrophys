"""
Convenience runner for the full document-aligned analysis suite.
"""

from __future__ import annotations

import warnings

import numpy as np

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
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Tuple

if TYPE_CHECKING:
    from petrophys.model.well import Well


def _keeping_existing_curves(well: "Well", kept: List[str]) -> Callable[[Callable[[], Any]], Any]:
    """Run an analyzer and restore every curve the well had before it ran.

    The analyzers write their derived curves (BAD_HOLE, CROSSOVER_FLAG, ...)
    with their own thresholds; a curve already on the well, such as the
    interpretation's BAD_HOLE that its NET_PAY excludes, is put back as it
    was, and its name is added to ``kept``.
    """
    originals: Dict[str, Tuple[Any, np.ndarray]] = {
        name: (well.get(name), np.array(well.get(name).data, copy=True)) for name in well.curves
    }

    def run(analyze: Callable[[], Any]) -> Any:
        try:
            return analyze()
        finally:
            for name, (curve, data) in originals.items():
                current = well.get(name)
                same = current is curve and np.array_equal(
                    np.asarray(current.data), data, equal_nan=data.dtype.kind == "f"
                )
                if not same:
                    curve.data = data
                    well._curves[name.upper()] = curve
                    if name not in kept:
                        kept.append(name)

    return run


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
    same class in every category; ``"geolink"`` keeps the GeoLink eval
    package's tables. ``lithology_family`` (``"clastic"``, ``"carbonate"`` or
    a lithology, for every sample or one per sample; default clastic) selects
    the limits of the reservoir-quality table under ``"current"``.

    With ``add_curves`` only curves the well does not have are written: a
    curve already on it (for example the BAD_HOLE and CROSSOVER_FLAG of an
    interpretation, which its NET_PAY is consistent with) is kept as it is,
    and the analyzers that run after it read it. The names kept are given
    in a warning and, with ``store_results``, under ``"curves_kept"`` of the
    ``COMPREHENSIVE_ANALYSIS`` result; the category reports themselves are
    computed with each analyzer's own thresholds.
    """
    graded = {"class_tables": class_tables, "lithology_family": lithology_family}
    kept: List[str] = []
    guard = _keeping_existing_curves(well, kept) if add_curves else (lambda analyze: analyze())
    qc = guard(lambda: WellboreQC().analyze(well, add_curves=add_curves))
    numerical = guard(lambda: NumericalStatisticalAnalysis().analyze(well, add_curves=add_curves))
    lithology = guard(lambda: LithologyAnalysis(**graded).analyze(well, add_curves=add_curves))
    porosity = guard(lambda: PorosityAnalysis(**graded).analyze(well, add_curves=add_curves))
    permeability = guard(lambda: PermeabilityAnalysis(**graded).analyze(well, add_curves=add_curves))
    saturation = guard(
        lambda: SaturationAnalysis(class_tables=class_tables).analyze(well, add_curves=add_curves)
    )
    hydrocarbon = guard(lambda: HydrocarbonAnalysis(**graded).analyze(well, add_curves=add_curves))
    if kept:
        warnings.warn(
            f"Curves already on the well were kept, not overwritten: {', '.join(kept)}",
            UserWarning, stacklevel=2,
        )

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
        well.set_result("COMPREHENSIVE_ANALYSIS", {**result.to_dict(), "curves_kept": list(kept)})

    return result
