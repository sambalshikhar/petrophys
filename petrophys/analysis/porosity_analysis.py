"""
Porosity analysis classifications.
"""

from __future__ import annotations

import numpy as np

from petrophys.utils.analysis_helpers import (
    CategoryAnalysisResult,
    DerivedCurve,
    build_section,
    format_curve_source,
    get_curve_data,
)
from petrophys.reservoir.rock_quality.rock_quality_module import RockQualityModule


class PorosityAnalysis:
    """Porosity and pore-system classification analysis."""

    CATEGORY_KEY = "POROSITY_AND_PORE_SYSTEMS"

    def analyze(
        self,
        well: "Well",
        add_curves: bool = True,
    ) -> CategoryAnalysisResult:
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
        porosity_basis_text = (
            f"{format_curve_source(phit_source)}; fallback priority PHID={format_curve_source(phid_source)}, "
            f"PHIS={format_curve_source(phis_source)}, PHIN={format_curve_source(phin_source)}"
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

        porosity_source = self._first_valid_porosity(phit, phid, phis, phin)
        total_labels = np.full(n_samples, "NO_POROSITY_DATA", dtype=object)
        valid_phi = np.isfinite(porosity_source)
        total_labels[valid_phi] = "POOR_POROSITY"
        total_labels[valid_phi & (porosity_source > 0.06)] = "FAIR_POROSITY"
        total_labels[valid_phi & (porosity_source > 0.12)] = "GOOD_POROSITY"
        total_labels[valid_phi & (porosity_source > 0.20)] = "EXCELLENT_POROSITY"

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

        clay_labels = np.full(n_samples, "NO_VARIATION_DATA", dtype=object)
        if vsh is not None:
            valid = np.isfinite(vsh)
            clay_labels[valid] = "SHALY_FORMATION"
            clay_labels[valid & (vsh <= 0.50)] = "MODERATE_CLAY_IMPACT"
            clay_labels[valid & (vsh < 0.25)] = "CLEAN_FORMATION"

        quality_labels = np.full(n_samples, "NO_QUALITY_DATA", dtype=object)
        if phit is not None and perm is not None:
            valid = np.isfinite(phit) & np.isfinite(perm)
            quality_labels[valid] = "POOR_FLOW_UNIT"
            quality_labels[valid & (phit > 0.06) & (perm > 0.1)] = "FAIR_FLOW_UNIT"
            quality_labels[valid & (phit > 0.10) & (perm > 1.0)] = "GOOD_FLOW_UNIT"
            quality_labels[valid & (phit > 0.15) & (perm > 10.0)] = "EXCELLENT_FLOW_UNIT"
        elif rqi is not None:
            valid = np.isfinite(rqi)
            quality_labels[valid] = "LOW_RQI"
            quality_labels[valid & (rqi >= 0.5)] = "MODERATE_RQI"
            quality_labels[valid & (rqi > 1.0)] = "HIGH_RQI"
        elif fzi is not None:
            valid = np.isfinite(fzi)
            quality_labels[valid] = "FLOW_UNIT_4"
            quality_labels[valid & (fzi > 0.5)] = "FLOW_UNIT_3"
            quality_labels[valid & (fzi > 1.0)] = "FLOW_UNIT_2"
            quality_labels[valid & (fzi > 2.0)] = "FLOW_UNIT_1"

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

        result = CategoryAnalysisResult(
            category_key=self.CATEGORY_KEY,
            sections={
                "total_porosity_quality_distribution": build_section(
                    well,
                    total_labels,
                    description="Porosity quality from the best available porosity measurement.",
                    methodology=(
                        f"Uses {porosity_basis_text}. Priority is PHIE_CC, PHIT, PHID, PHIS, then PHIN/NPHI."
                    ),
                    calculation_thresholds=(
                        "EXCELLENT > 0.20; GOOD 0.12-0.20; FAIR 0.06-0.12; "
                        "POOR <= 0.06."
                    ),
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
                    calculation_thresholds=(
                        "CLEAN < 0.25; MODERATE 0.25-0.50; SHALY > 0.50."
                    ),
                ),
                "porosity_quality_assessment_characteristics": build_section(
                    well,
                    quality_labels,
                    description="Flow-unit quality using selected porosity/permeability, then RQI, then FZI.",
                    methodology=(
                        f"Direct route uses porosity={format_curve_source(phit_source)} and permeability={perm_source_text}; "
                        f"rock-quality fallback uses {format_curve_source(rqi_source, 'RQI derived when possible')} "
                        f"then {format_curve_source(fzi_source, 'FZI derived when possible')}."
                    ),
                    calculation_thresholds=(
                        "EXCELLENT_FLOW_UNIT: selected porosity > 0.15 and selected permeability > 10; "
                        "GOOD_FLOW_UNIT: selected porosity > 0.10 and selected permeability > 1; "
                        "FAIR_FLOW_UNIT: selected porosity > 0.06 and selected permeability > 0.1. "
                        "RQI fallback: HIGH_RQI > 1.0, MODERATE_RQI >= 0.5, LOW_RQI < 0.5. "
                        "FZI fallback: FLOW_UNIT_1 > 2.0, FLOW_UNIT_2 > 1.0, FLOW_UNIT_3 > 0.5, FLOW_UNIT_4 <= 0.5."
                    ),
                ),
                "integrated_porosity_reservoir_assessment": build_section(
                    well,
                    integrated_labels,
                    description="Integrated reservoir quality from porosity and flow-unit classes.",
                    methodology="Combines total-porosity quality with the flow-unit section.",
                    calculation_thresholds=(
                        "EXCELLENT_RESERVOIR, GOOD_RESERVOIR, MARGINAL_RESERVOIR, "
                        "TIGHT_FORMATION, otherwise UNDETERMINED."
                    ),
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
