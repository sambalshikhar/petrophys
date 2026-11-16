"""
Porosity analysis classifications.
"""

from __future__ import annotations

import numpy as np

from petrophys._internal.benchmark import reproducing_benchmark
from petrophys._internal.analysis_helpers import (
    CategoryAnalysisResult,
    DerivedCurve,
    build_section,
    format_curve_source,
    get_curve_data,
)
from petrophys.evaluation.diagnostics._tables import (
    class_table_set,
    reservoir_quality,
    reservoir_quality_text,
    sample_families,
)
from petrophys.evaluation.diagnostics.bands import Bands
from petrophys.evaluation.rock_quality.rock_quality_module import RockQualityModule
from petrophys.model.class_tables import CLASS_TABLES
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from petrophys.model.well import Well


class PorosityAnalysis:
    """Porosity and pore-system classification analysis.

    Parameters
    ----------
    class_tables : {"current", "geolink"}, default "current"
        ``"current"`` grades porosity, clay volume, RQI and FZI with their
        canonical tables (:data:`~petrophys.model.class_tables.CLASS_TABLES`)
        and flow-unit quality with the reservoir-quality table on effective
        porosity (PHIE_CC, PHIE, PHIT), permeability and clay volume (VCL,
        VCLAY, VSH). ``"geolink"`` keeps the GeoLink eval package's classes
        and texts: the class attributes below and the inline flow-unit grades.
    lithology_family : str or sequence of str, optional
        Lithology family (``"clastic"``, ``"carbonate"``) or lithology of every
        sample, or one per sample, whose limits the reservoir-quality table
        applies under ``"current"``. None (default): clastic.
    """

    CATEGORY_KEY = "POROSITY_AND_PORE_SYSTEMS"
    # The "geolink" profile's one-curve classes of the sections below (Bands: base label,
    # then each step in order). An analyzer with the "current" class tables holds
    # the canonical tables under the same names.
    TOTAL_POROSITY_BANDS = Bands("POOR_POROSITY", ((">", 0.06, "FAIR_POROSITY"), (">", 0.12, "GOOD_POROSITY"),
                                                   (">", 0.20, "EXCELLENT_POROSITY")))
    CLAY_IMPACT_BANDS = Bands("SHALY_FORMATION", (("<=", 0.50, "MODERATE_CLAY_IMPACT"), ("<", 0.25, "CLEAN_FORMATION")))
    RQI_BANDS = Bands("LOW_RQI", ((">=", 0.5, "MODERATE_RQI"), (">", 1.0, "HIGH_RQI")))
    FZI_BANDS = Bands("FLOW_UNIT_4", ((">", 0.5, "FLOW_UNIT_3"), (">", 1.0, "FLOW_UNIT_2"), (">", 2.0, "FLOW_UNIT_1")))

    def __init__(self, class_tables: str = "current", lithology_family=None):
        self.class_tables = class_table_set(class_tables)
        self.lithology_family = lithology_family
        if self.class_tables == "current":
            self.TOTAL_POROSITY_BANDS = CLASS_TABLES["PHI"].bands
            self.CLAY_IMPACT_BANDS = CLASS_TABLES["VCL"].bands
            self.RQI_BANDS = CLASS_TABLES["RQI"].bands
            self.FZI_BANDS = CLASS_TABLES["FZI"].bands

    def analyze(
        self,
        well: "Well",
        add_curves: bool = True,
    ) -> CategoryAnalysisResult:
        """Classify each sample's porosity and pore-system indicators and summarize them by interval.

        Parameters
        ----------
        well : Well
            Interpreted well. Total and effective porosity, the density,
            sonic and neutron porosities (PHID, PHIS, PHIN), permeability,
            clay volume, RQI and FZI are read where present; a class that
            needs an absent curve is undetermined.
        add_curves : bool, default True
            Add the derived curves that hold values to ``well``, replacing
            curves of the same name: ``PORO_QUALITY_INDEX``,
            ``DS_SEPARATION`` (density minus sonic porosity, v/v) where both
            exist, and ``RQI`` and ``FZI`` where the well has none.

        Returns
        -------
        CategoryAnalysisResult
            Sections ``total_porosity_quality_distribution``,
            ``pore_system_architecture_classification``,
            ``clay_content_impact_on_porosity_variation``,
            ``porosity_quality_assessment_characteristics`` and
            ``integrated_porosity_reservoir_assessment``, each with its
            description, method, thresholds and the depth intervals of every
            class, and the derived curves in ``derived_curves``.
        """
        benchmark = self.class_tables == "geolink"
        n_samples = well.n_samples
        phit, phit_source = get_curve_data(well, ["PHIE_CC", "PHIT"])
        phid, phid_source = get_curve_data(well, ["PHID"])
        phis, phis_source = get_curve_data(well, ["PHIS"])
        phin, phin_source = get_curve_data(well, ["PHIN", "NPHI"])
        perm, perm_source = get_curve_data(well, ["PERM_CC", "PERM", "PERM_CORE", "PERM_TIMUR"])
        vsh, vsh_source = get_curve_data(well, ["VSH", "VCL", "VCLAY"])
        rqi, rqi_source = get_curve_data(well, ["RQI"])
        fzi, fzi_source = get_curve_data(well, ["FZI"])
        phie, phie_source = get_curve_data(well, ["PHIE_CC", "PHIE", "PHIT"])
        if benchmark or reproducing_benchmark():
            porosity_basis_text = (
                f"{format_curve_source(phit_source)}; fallback priority PHID={format_curve_source(phid_source)}, "
                f"PHIS={format_curve_source(phis_source)}, PHIN={format_curve_source(phin_source)}"
            )
            porosity_methodology = (
                f"Uses {porosity_basis_text}. Priority is PHIE_CC, PHIT, PHID, PHIS, then PHIN/NPHI."
            )
        else:
            used = [format_curve_source(source)
                    for source in (phit_source, phid_source, phis_source, phin_source) if source]
            porosity_methodology = (
                "Each sample takes the first finite porosity of PHIE_CC, PHIT, PHID, PHIS, then "
                "PHIN/NPHI (PHIE is not in this order); curves with data here: "
                f"{', '.join(used) if used else 'none'}."
            )
        phie_source_text = format_curve_source(phie_source)
        perm_source_text = format_curve_source(perm_source)
        vsh_source_text = format_curve_source(vsh_source)

        derived_curves: dict[str, DerivedCurve] = {}

        if (rqi is None or fzi is None) and perm is not None and phie is not None:
            try:
                rq_result = RockQualityModule().calculate(well, perm=perm, phi_e=phie)
                if rqi is None:
                    rqi = rq_result.rqi
                    derived_curves["RQI"] = DerivedCurve(
                        rqi,
                        "microns",
                        f"Reservoir Quality Index from {perm_source_text} and {phie_source_text}",
                    )
                if fzi is None:
                    fzi = rq_result.fzi
                    derived_curves["FZI"] = DerivedCurve(
                        fzi,
                        "microns",
                        f"Flow Zone Indicator from {perm_source_text} and {phie_source_text}",
                    )
            except Exception:
                pass

        porosity_missing = "NO_POROSITY_DATA" if benchmark else CLASS_TABLES["PHI"].missing
        porosity_source = self._first_valid_porosity(phit, phid, phis, phin)
        valid_phi = np.isfinite(porosity_source)
        total_labels = self.TOTAL_POROSITY_BANDS.classify(porosity_source, valid_phi, missing=porosity_missing)
        if total_labels.shape != (n_samples,):
            # No porosity curve at all: _first_valid_porosity returns an empty array.
            total_labels = np.full(n_samples, porosity_missing, dtype=object)

        ds_separation = np.full(n_samples, np.nan, dtype=float)
        pore_labels = np.full(n_samples, "UNDETERMINED_PORE_TYPE", dtype=object)
        if phid is not None and phis is not None:
            ds_separation = phid - phis
            valid = np.isfinite(ds_separation)
            pore_labels[valid] = "PRIMARY_POROSITY"
            pore_labels[valid & (ds_separation > 0.05)] = "SECONDARY_POROSITY"
            pore_labels[valid & (ds_separation < -0.05)] = "MATRIX_EFFECT"
            derived_curves["DS_SEPARATION"] = DerivedCurve(
                ds_separation,
                "v/v",
                "Density-sonic porosity separation",
            )

        clay_missing = "NO_VARIATION_DATA" if benchmark else CLASS_TABLES["VCL"].missing
        clay_labels = np.full(n_samples, clay_missing, dtype=object)
        if vsh is not None:
            clay_labels = self.CLAY_IMPACT_BANDS.classify(vsh, missing=clay_missing)

        if benchmark:
            quality_labels = np.full(n_samples, "NO_QUALITY_DATA", dtype=object)
            if phit is not None and perm is not None:
                valid = np.isfinite(phit) & np.isfinite(perm)
                quality_labels[valid] = "POOR_FLOW_UNIT"
                quality_labels[valid & (phit > 0.06) & (perm > 0.1)] = "FAIR_FLOW_UNIT"
                quality_labels[valid & (phit > 0.10) & (perm > 1.0)] = "GOOD_FLOW_UNIT"
                quality_labels[valid & (phit > 0.15) & (perm > 10.0)] = "EXCELLENT_FLOW_UNIT"
            elif rqi is not None:
                quality_labels = self.RQI_BANDS.classify(rqi, missing="NO_QUALITY_DATA")
            elif fzi is not None:
                quality_labels = self.FZI_BANDS.classify(fzi, missing="NO_QUALITY_DATA")

            integrated_labels = np.full(n_samples, "UNDETERMINED", dtype=object)
            integrated_labels[total_labels == "POOR_POROSITY"] = "TIGHT_FORMATION"
            integrated_labels[total_labels == "FAIR_POROSITY"] = "MARGINAL_RESERVOIR"
            integrated_labels[
                np.isin(total_labels, ["EXCELLENT_POROSITY", "GOOD_POROSITY"])
                & np.isin(quality_labels, ["EXCELLENT_FLOW_UNIT", "GOOD_FLOW_UNIT"])
            ] = "GOOD_RESERVOIR"
            integrated_labels[
                (total_labels == "EXCELLENT_POROSITY")
                & (quality_labels == "EXCELLENT_FLOW_UNIT")
            ] = "EXCELLENT_RESERVOIR"
        else:
            quality_table = CLASS_TABLES["RESERVOIR_QUALITY"]
            families = sample_families(self.lithology_family, n_samples)
            two_curve = phie is not None and perm is not None
            quality_labels, quality_clay_source = reservoir_quality(well, phie, perm, families)
            if not two_curve and rqi is not None:
                quality_labels = self.RQI_BANDS.classify(rqi, missing=CLASS_TABLES["RQI"].missing)
            elif not two_curve and fzi is not None:
                quality_labels = self.FZI_BANDS.classify(fzi, missing=CLASS_TABLES["FZI"].missing)

            # The reservoir-quality class where it exists, so the label means
            # the same as in the other categories; otherwise porosity alone.
            integrated_labels = np.full(n_samples, "UNDETERMINED", dtype=object)
            integrated_labels[total_labels == "POOR_POROSITY"] = "TIGHT_FORMATION"
            integrated_labels[total_labels == "FAIR_POROSITY"] = "MARGINAL_RESERVOIR"
            if two_curve:
                graded = np.isin(quality_labels, quality_table.labels)
                integrated_labels[graded] = quality_labels[graded]

        poro_quality_index = np.full(n_samples, np.nan, dtype=float)
        if phit is not None and perm is not None:
            valid = np.isfinite(phit) & np.isfinite(perm) & (phit > 0.001) & (perm > 0.001)
            raw = np.maximum(phit[valid] * np.log10(np.maximum(perm[valid], 0.001)), 1e-6)
            poro_quality_index[valid] = np.clip(np.sqrt(raw), 0.001, 5.0)
            derived_curves["PORO_QUALITY_INDEX"] = DerivedCurve(
                poro_quality_index,
                "",
                "Composite porosity quality index",
            )

        if benchmark:
            texts = {
                "total": "EXCELLENT > 0.20; GOOD 0.12-0.20; FAIR 0.06-0.12; POOR <= 0.06.",
                "clay": "CLEAN < 0.25; MODERATE 0.25-0.50; SHALY > 0.50.",
                "quality_description": "Flow-unit quality using selected porosity/permeability, then RQI, then FZI.",
                "quality_methodology": (
                    f"Direct route uses porosity={format_curve_source(phit_source)} and permeability={perm_source_text}; "
                    f"rock-quality fallback uses {format_curve_source(rqi_source, 'RQI derived when possible')} "
                    f"then {format_curve_source(fzi_source, 'FZI derived when possible')}."
                ),
                "quality": (
                    "EXCELLENT_FLOW_UNIT: selected porosity > 0.15 and selected permeability > 10; "
                    "GOOD_FLOW_UNIT: selected porosity > 0.10 and selected permeability > 1; "
                    "FAIR_FLOW_UNIT: selected porosity > 0.06 and selected permeability > 0.1. "
                    "RQI fallback: HIGH_RQI > 1.0, MODERATE_RQI >= 0.5, LOW_RQI < 0.5. "
                    "FZI fallback: FLOW_UNIT_1 > 2.0, FLOW_UNIT_2 > 1.0, FLOW_UNIT_3 > 0.5, FLOW_UNIT_4 <= 0.5."
                ),
                "integrated_description": "Integrated reservoir quality from porosity and flow-unit classes.",
                "integrated_methodology": "Combines total-porosity quality with the flow-unit section.",
                "integrated": (
                    "EXCELLENT_RESERVOIR, GOOD_RESERVOIR, MARGINAL_RESERVOIR, "
                    "TIGHT_FORMATION, otherwise UNDETERMINED."
                ),
            }
        else:
            texts = {
                "total": CLASS_TABLES["PHI"].describe(),
                "clay": CLASS_TABLES["VCL"].describe(),
                "quality_description": (
                    "Reservoir quality from effective porosity, permeability and clay volume, then RQI, then FZI "
                    "classes."
                ),
                "quality_methodology": (
                    f"Direct route uses effective porosity={phie_source_text}, permeability={perm_source_text} "
                    f"and clay volume={format_curve_source(quality_clay_source)}, with the limits of each "
                    "sample's lithology family; "
                    f"rock-quality fallback uses {format_curve_source(rqi_source, 'RQI derived when possible')} "
                    f"then {format_curve_source(fzi_source, 'FZI derived when possible')}."
                ),
                "quality": (
                    f"{reservoir_quality_text(families)} "
                    f"RQI fallback: {CLASS_TABLES['RQI'].describe()} "
                    f"FZI fallback (global hydraulic elements): {CLASS_TABLES['FZI'].describe()}"
                ),
                "integrated_description": "Integrated reservoir quality from the reservoir-quality and porosity classes.",
                "integrated_methodology": (
                    "The reservoir-quality class where effective porosity, permeability and clay volume grade "
                    "the sample; otherwise the total-porosity class."
                ),
                "integrated": (
                    "NON_RESERVOIR, FAIR_RESERVOIR, GOOD_RESERVOIR or EXCELLENT_RESERVOIR as in the quality section; "
                    "without them TIGHT_FORMATION for POOR_POROSITY, MARGINAL_RESERVOIR for FAIR_POROSITY, "
                    "otherwise UNDETERMINED."
                ),
            }

        result = CategoryAnalysisResult(
            category_key=self.CATEGORY_KEY,
            sections={
                "total_porosity_quality_distribution": build_section(
                    well,
                    total_labels,
                    description="Porosity quality from the best available porosity measurement.",
                    methodology=porosity_methodology,
                    calculation_thresholds=texts["total"],
                ),
                "pore_system_architecture_classification": build_section(
                    well,
                    pore_labels,
                    description="Pore-system architecture from density-sonic separation.",
                    methodology="DS_SEPARATION = PHID - PHIS.",
                    calculation_thresholds=(
                        "SECONDARY > 0.05; PRIMARY between -0.05 and 0.05; "
                        "MATRIX_EFFECT < -0.05."
                    ),
                ),
                "clay_content_impact_on_porosity_variation": build_section(
                    well,
                    clay_labels,
                    description="Clay impact on porosity variation from shale volume.",
                    methodology=f"Uses {vsh_source_text}; priority is VSH, then VCL, VCLAY.",
                    calculation_thresholds=texts["clay"],
                ),
                "porosity_quality_assessment_characteristics": build_section(
                    well,
                    quality_labels,
                    description=texts["quality_description"],
                    methodology=texts["quality_methodology"],
                    calculation_thresholds=texts["quality"],
                ),
                "integrated_porosity_reservoir_assessment": build_section(
                    well,
                    integrated_labels,
                    description=texts["integrated_description"],
                    methodology=texts["integrated_methodology"],
                    calculation_thresholds=texts["integrated"],
                ),
            },
            derived_curves=derived_curves,
        )

        if add_curves:
            result.add_curves_to_well(well)

        return result

    @staticmethod
    def _first_valid_porosity(
        phit: np.ndarray | None,
        phid: np.ndarray | None,
        phis: np.ndarray | None,
        phin: np.ndarray | None,
    ) -> np.ndarray:
        arrays = [arr for arr in [phit, phid, phis, phin] if arr is not None]
        if not arrays:
            return np.array([], dtype=float)
        result = np.full(len(arrays[0]), np.nan, dtype=float)
        for arr in arrays:
            mask = ~np.isfinite(result) & np.isfinite(arr)
            result[mask] = arr[mask]
        return result
