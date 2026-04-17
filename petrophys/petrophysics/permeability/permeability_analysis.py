"""
Permeability and flow-unit analysis.
"""

from __future__ import annotations

import numpy as np

from petrophys.utils.analysis_helpers import (
    CategoryAnalysisResult,
    DerivedCurve,
    build_section,
    depth_step_array,
    get_curve_data,
)
from petrophys.reservoir.rock_quality.rock_quality_module import RockQualityModule


class PermeabilityAnalysis:
    """Permeability distribution and flow-unit analysis."""

    CATEGORY_KEY = "PERMEABILITY_AND_FLOW_UNITS"

    def analyze(
        self,
        well: "Well",
        add_curves: bool = True,
    ) -> CategoryAnalysisResult:
        n_samples = well.n_samples
        perm, _ = get_curve_data(well, ["PERM", "PERM_CC", "PERM_CORE", "PERM_TIMUR"])
        fzi, _ = get_curve_data(well, ["FZI"])
        rqi, _ = get_curve_data(well, ["RQI"])
        phie, _ = get_curve_data(well, ["PHIE", "PHIT"])
        vsh, _ = get_curve_data(well, ["VSH", "VCL", "VCLAY"])
        rxo, _ = get_curve_data(well, ["RXO", "RSHA", "RESS"])
        rdep, _ = get_curve_data(well, ["RDEP", "RT"])
        mhi, _ = get_curve_data(well, ["MHI"])

        derived_curves: dict[str, DerivedCurve] = {}
        if (fzi is None or rqi is None) and perm is not None and phie is not None:
            try:
                rq_result = RockQualityModule().calculate(well, perm=perm, phi_e=phie)
                if rqi is None:
                    rqi = rq_result.rqi
                    derived_curves["RQI"] = DerivedCurve(rqi, "microns", "Reservoir Quality Index")
                if fzi is None:
                    fzi = rq_result.fzi
                    derived_curves["FZI"] = DerivedCurve(fzi, "microns", "Flow Zone Indicator")
            except Exception:
                pass

        distribution_labels = np.full(n_samples, "NO_DATA", dtype=object)
        if perm is not None:
            valid = np.isfinite(perm)
            distribution_labels[valid] = "POOR_PERMEABILITY"
            distribution_labels[valid & (perm > 1.0)] = "FAIR_PERMEABILITY"
            distribution_labels[valid & (perm > 10.0)] = "GOOD_PERMEABILITY"
            distribution_labels[valid & (perm > 100.0)] = "VERY_GOOD_PERMEABILITY"
            distribution_labels[valid & (perm > 1000.0)] = "EXCELLENT_PERMEABILITY"

        flow_unit_labels = np.full(n_samples, "NO_DATA", dtype=object)
        if fzi is not None:
            valid = np.isfinite(fzi)
            flow_unit_labels[valid] = "MICRO_PORES"
            flow_unit_labels[valid & (fzi >= 0.5)] = "MESO_PORES"
            flow_unit_labels[valid & (fzi >= 2.0)] = "MACRO_PORES"
            flow_unit_labels[valid & (fzi >= 5.0)] = "MEGA_PORES"

        variation_labels = np.full(n_samples, "NO_DATA", dtype=object)
        if vsh is not None:
            valid = np.isfinite(vsh)
            variation_labels[valid] = "SHALE_DOMINATED"
            variation_labels[valid & (vsh <= 0.50)] = "MIXED_LITHOLOGY"
            variation_labels[valid & (vsh < 0.20)] = "CLEAN_RESERVOIR"

        invasion_labels = np.full(n_samples, "NO_DATA", dtype=object)
        ratio = np.full(n_samples, np.nan, dtype=float)
        if rxo is not None and rdep is not None:
            valid = np.isfinite(rxo) & np.isfinite(rdep) & (rdep > 0.0)
            ratio[valid] = rxo[valid] / rdep[valid]
            invasion_labels[valid] = "MINIMAL_INVASION"
            invasion_labels[valid & (ratio < 0.8)] = "CONDUCTIVE_INVASION"
            invasion_labels[valid & (ratio > 1.2)] = "RESISTIVE_INVASION"
            derived_curves["RXO_RDEP_RATIO"] = DerivedCurve(
                ratio,
                "",
                "Flushed-to-deep resistivity ratio",
            )

        overall_labels = np.full(n_samples, "TIGHT_FORMATION", dtype=object)
        overall_labels[distribution_labels == "FAIR_PERMEABILITY"] = "MARGINAL_RESERVOIR"
        overall_labels[distribution_labels == "GOOD_PERMEABILITY"] = "GOOD_RESERVOIR"
        overall_labels[distribution_labels == "VERY_GOOD_PERMEABILITY"] = "VERY_GOOD_RESERVOIR"
        overall_labels[distribution_labels == "EXCELLENT_PERMEABILITY"] = "EXCELLENT_RESERVOIR"

        if perm is not None:
            kh = np.clip(perm * depth_step_array(well.depth), 0.001, None)
            derived_curves["KH"] = DerivedCurve(kh, "mD*depth", "Flow capacity (PERM * dz)")

        if perm is not None or rqi is not None or fzi is not None:
            perm_qi = np.full(n_samples, np.nan, dtype=float)
            components: list[np.ndarray] = []
            weights: list[float] = []
            if perm is not None:
                components.append(np.asarray(perm, dtype=float))
                weights.append(0.5)
            if rqi is not None:
                components.append(np.asarray(rqi, dtype=float) * 10.0)
                weights.append(0.3)
            if fzi is not None:
                components.append(np.asarray(fzi, dtype=float) * 5.0)
                weights.append(0.2)
            stack = np.vstack(components)
            weight_arr = np.asarray(weights, dtype=float)
            for idx in range(n_samples):
                values = stack[:, idx]
                valid = np.isfinite(values)
                if np.any(valid):
                    perm_qi[idx] = float(
                        np.sum(values[valid] * weight_arr[valid]) / np.sum(weight_arr[valid])
                    )
            derived_curves["PERM_QUALITY_INDEX"] = DerivedCurve(
                perm_qi,
                "",
                "Composite permeability quality index",
            )

        metadata = {}
        if mhi is not None:
            metadata["mhi_confirmation"] = {
                "high_mhi_samples": int(np.count_nonzero(np.isfinite(mhi) & (mhi > 0.3)))
            }

        result = CategoryAnalysisResult(
            category_key=self.CATEGORY_KEY,
            sections={
                "permeability_distribution_analysis": build_section(
                    well,
                    distribution_labels,
                    description="Permeability quality from absolute permeability.",
                    methodology="Uses the first available permeability curve.",
                    calculation_thresholds=(
                        "EXCELLENT > 1000; VERY_GOOD 100-1000; GOOD 10-100; "
                        "FAIR 1-10; POOR <= 1."
                    ),
                ),
                "flow_unit_classification_system": build_section(
                    well,
                    flow_unit_labels,
                    description="Flow-unit characterization from FZI.",
                    methodology="Uses FZI classes with RQI as supplementary context.",
                    calculation_thresholds=(
                        "MEGA_PORES >= 5; MACRO_PORES 2-5; MESO_PORES 0.5-2; MICRO_PORES < 0.5."
                    ),
                ),
                "permeability_variation_controls": build_section(
                    well,
                    variation_labels,
                    description="Geological permeability controls from shale content.",
                    methodology="Uses the first available VSH/VCL curve.",
                    calculation_thresholds=(
                        "CLEAN_RESERVOIR < 0.20; MIXED_LITHOLOGY 0.20-0.50; SHALE_DOMINATED > 0.50."
                    ),
                ),
                "invasion_profile_permeability_indicators": build_section(
                    well,
                    invasion_labels,
                    description="Permeability clues from invasion behavior.",
                    methodology="Uses RXO / RDEP when resistivity data are available.",
                    calculation_thresholds=(
                        "CONDUCTIVE_INVASION < 0.8; MINIMAL_INVASION 0.8-1.2; "
                        "RESISTIVE_INVASION > 1.2."
                    ),
                ),
                "overall_permeability_interpretation": build_section(
                    well,
                    overall_labels,
                    description="Reservoir interpretation mapped from permeability class.",
                    methodology="Direct mapping from permeability distribution classes.",
                    calculation_thresholds=(
                        "EXCELLENT_RESERVOIR, VERY_GOOD_RESERVOIR, GOOD_RESERVOIR, "
                        "MARGINAL_RESERVOIR, otherwise TIGHT_FORMATION."
                    ),
                ),
            },
            derived_curves=derived_curves,
            metadata=metadata,
        )

        if add_curves:
            result.add_curves_to_well(well)

        return result
