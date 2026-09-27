"""
Comprehensive lithology analysis classifications.
"""

from __future__ import annotations

import numpy as np

from petrophys._internal.analysis_helpers import (
    CategoryAnalysisResult,
    DerivedCurve,
    build_section,
    classification_code,
    format_curve_source,
    get_curve_data,
)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from petrophys.model.well import Well


GROUND_TRUTH_NAMES = {
    1: "Sand",
    2: "Sand",
    3: "CrossBedded Sand",
    4: "Shaly Sandstone",
    5: "Silt",
    6: "Silt",
    7: "Shale",
    8: "Shale",
    9: "Limestone",
    10: "Calcareous Sandstone",
    11: "Limestone",
    12: "Limestone",
    13: "Limestone",
    14: "Calcareous Shale",
    15: "Dolomite",
    16: "Sandy Limestone",
    17: "Conglomerate",
    18: "Coal",
    19: "Extrusive",
    20: "Extrusive",
    21: "Halite",
    22: "Anhydrite",
    23: "Intrusive",
    24: "Intrusive",
    25: "Sandy Shale",
    26: "Gypsum",
    27: "Halite",
    28: "Extrusive",
    29: "Chert",
    30: "Shaly Sand",
    31: "Breccia",
    32: "Halite",
    33: "Halite",
    34: "Halite",
    35: "CrossBedded Sand",
    36: "Gravel",
}

GROUND_TRUTH_GROUPS = {
    "LIMESTONE": {9, 11, 12, 13, 16},
    "DOLOMITE": {15},
    "ANHYDRITE": {22},
    "COAL": {18},
    "CALCAREOUS_SHALE": {14},
    "HALITE": {21, 27, 32, 33, 34},
    "SANDSTONE": {1, 2, 3, 4, 10, 30, 35},
    "SHALE": {7, 8, 25},
    "SILT": {5, 6},
    "CLASTIC_COARSE": {17, 31, 36},
    "EVAPORITE": {26},
    "IGNEOUS": {19, 20, 23, 24, 28},
    "CHERT": {29},
}

GROUP_CODES = {
    "NO_LITHOLOGY_DATA": 0,
    "LIMESTONE": 1,
    "DOLOMITE": 2,
    "ANHYDRITE": 3,
    "COAL": 4,
    "CALCAREOUS_SHALE": 5,
    "HALITE": 6,
    "SANDSTONE": 7,
    "SHALE": 8,
    "SILT": 9,
    "CLASTIC_COARSE": 10,
    "EVAPORITE": 11,
    "IGNEOUS": 12,
    "CHERT": 13,
    "SPECIAL_LITHOLOGY": 14,
}

COMPUTED_LITHO_GROUPS = {
    1: "SANDSTONE",
    2: "LIMESTONE",
    3: "DOLOMITE",
    4: "SHALE",
    5: "ANHYDRITE",
    6: "HALITE",
    7: "COAL",
}


class LithologyAnalysis:
    """Ground-truth and log-based lithology analysis."""

    CATEGORY_KEY = "COMPREHENSIVE_LITHOLOGY_ANALYSIS"

    def analyze(
        self,
        well: "Well",
        add_curves: bool = True,
    ) -> CategoryAnalysisResult:
        n_samples = well.n_samples
        gt_codes, _ = get_curve_data(well, ["LITHOLOGY_GEOLINK"])
        computed_litho, _ = get_curve_data(well, ["LITHO"])
        pef, _ = get_curve_data(well, ["PEF"])
        rhob, _ = get_curve_data(well, ["RHOB"])
        nphi, _ = get_curve_data(well, ["NPHI", "PHIN"])
        gr, _ = get_curve_data(well, ["GR"])
        phit, phit_source = get_curve_data(well, ["PHIE_CC", "PHIE", "PHIT"])
        vsh, vsh_source = get_curve_data(well, ["VSH", "VCL", "VCLAY"])
        perm, perm_source = get_curve_data(well, ["PERM_CC", "PERM", "PERM_CORE", "PERM_TIMUR"])
        porosity_source_text = format_curve_source(phit_source)
        perm_source_text = format_curve_source(perm_source)
        vsh_source_text = format_curve_source(vsh_source)

        gt_groups = np.full(n_samples, "NO_LITHOLOGY_DATA", dtype=object)
        if gt_codes is not None:
            rounded = np.rint(gt_codes).astype(int)
            for group_name, codes in GROUND_TRUTH_GROUPS.items():
                gt_groups[np.isin(rounded, list(codes))] = group_name

        working_groups = gt_groups.copy()
        if computed_litho is not None:
            rounded = np.rint(computed_litho).astype(int)
            missing = gt_groups == "NO_LITHOLOGY_DATA"
            for code, group_name in COMPUTED_LITHO_GROUPS.items():
                working_groups[missing & (rounded == code)] = group_name
            working_groups[missing & (rounded >= 8)] = "SPECIAL_LITHOLOGY"

        limestone_indicator = np.full(n_samples, np.nan, dtype=float)
        dolomite_indicator = np.full(n_samples, np.nan, dtype=float)
        carbonate_indicator = np.full(n_samples, np.nan, dtype=float)
        if pef is not None and rhob is not None:
            valid = np.isfinite(pef) & np.isfinite(rhob)
            limestone_indicator[valid] = ((pef[valid] > 4.5) & (rhob[valid] > 2.7)).astype(float)
            dolomite_indicator[valid] = (
                (pef[valid] >= 2.8)
                & (pef[valid] <= 3.2)
                & (rhob[valid] >= 2.8)
                & (rhob[valid] <= 2.9)
            ).astype(float)
            carbonate_indicator[valid] = ((pef[valid] > 2.8) & (rhob[valid] > 2.6)).astype(float)

        gt_distribution = np.full(n_samples, "NO_LITHOLOGY_DATA", dtype=object)
        valid_gt = gt_groups != "NO_LITHOLOGY_DATA"
        gt_distribution[valid_gt] = np.char.add("GT_", gt_groups[valid_gt].astype(str))

        carbonate_labels = np.full(n_samples, "NON_CARBONATE", dtype=object)
        gt_carbonate = np.isin(working_groups, ["LIMESTONE", "DOLOMITE"])
        valid_carbonate = np.isfinite(carbonate_indicator)
        carbonate_labels[gt_carbonate & valid_carbonate & (carbonate_indicator > 0.5)] = "CARBONATE_CONFIRMED"
        carbonate_labels[gt_carbonate & valid_carbonate & (limestone_indicator > 0.5)] = "LIMESTONE_CONFIRMED"
        carbonate_labels[gt_carbonate & valid_carbonate & (dolomite_indicator > 0.5)] = "DOLOMITE_CONFIRMED"
        carbonate_labels[gt_carbonate & valid_carbonate & (carbonate_indicator <= 0.5)] = "CARBONATE_LOG_MISMATCH"
        carbonate_labels[(~gt_carbonate) & valid_carbonate & (carbonate_indicator > 0.5)] = "NON_CARBONATE_CARBONATE_SIGNATURE"

        gr_clay = np.full(n_samples, np.nan, dtype=float)
        nphi_clay = np.full(n_samples, np.nan, dtype=float)
        pef_clay = np.full(n_samples, np.nan, dtype=float)
        if gr is not None:
            valid = np.isfinite(gr)
            gr_clay[valid] = (gr[valid] > 100.0).astype(float)
        if nphi is not None:
            valid = np.isfinite(nphi)
            nphi_clay[valid] = (nphi[valid] > 0.30).astype(float)
        if pef is not None:
            valid = np.isfinite(pef)
            pef_clay[valid] = (pef[valid] < 2.5).astype(float)
        composite_clay = np.full(n_samples, np.nan, dtype=float)
        clay_stack = np.vstack([gr_clay, nphi_clay, pef_clay])
        for idx in range(n_samples):
            values = clay_stack[:, idx]
            valid = np.isfinite(values)
            if np.any(valid):
                composite_clay[idx] = float(np.nanmean(values[valid]))

        clay_labels = np.full(n_samples, "NO_CLAY_DATA", dtype=object)
        valid_clay = np.isfinite(composite_clay)
        clay_labels[valid_clay] = "LOW_CLAY_INDICATION"
        clay_labels[valid_clay & (composite_clay >= 0.33)] = "MODERATE_CLAY_INDICATION"
        clay_labels[valid_clay & (composite_clay >= 0.67)] = "HIGH_CLAY_INDICATION"

        quality = np.full(n_samples, "POOR", dtype=object)
        if phit is not None:
            valid = np.isfinite(phit)
            quality[valid] = "POOR"
            excellent = valid & (phit > 0.15)
            if perm is not None:
                excellent &= np.isfinite(perm) & (perm > 10.0)
            if vsh is not None:
                excellent &= np.isfinite(vsh) & (vsh < 0.25)
            good = valid & (phit > 0.10)
            if perm is not None:
                good &= np.isfinite(perm) & (perm > 1.0)
            if vsh is not None:
                good &= np.isfinite(vsh) & (vsh < 0.40)
            fair = valid & (phit > 0.05)
            quality[fair] = "FAIR"
            quality[good] = "GOOD"
            quality[excellent] = "EXCELLENT"

        petro_labels = np.full(n_samples, "SPECIAL_LITHOLOGY", dtype=object)
        petro_labels[np.isin(working_groups, ["EVAPORITE", "HALITE", "ANHYDRITE"])] = "EVAPORITE_SEAL"
        if vsh is not None:
            petro_labels[(working_groups == "SHALE") & np.isfinite(vsh)] = "TYPICAL_SHALE"
            petro_labels[(working_groups == "SHALE") & np.isfinite(vsh) & (vsh <= 0.7)] = "SANDY_SHALE"
        petro_labels[np.isin(working_groups, ["SANDSTONE"]) & np.isin(quality, ["FAIR", "POOR"])] = "TIGHT_SANDSTONE"
        petro_labels[np.isin(working_groups, ["SANDSTONE"]) & (quality == "GOOD")] = "GOOD_SANDSTONE_RESERVOIR"
        petro_labels[np.isin(working_groups, ["SANDSTONE"]) & (quality == "EXCELLENT")] = "EXCELLENT_SANDSTONE_RESERVOIR"
        petro_labels[np.isin(working_groups, ["LIMESTONE", "DOLOMITE"]) & np.isin(quality, ["FAIR", "POOR"])] = "TIGHT_CARBONATE"
        petro_labels[np.isin(working_groups, ["LIMESTONE", "DOLOMITE"]) & (quality == "GOOD")] = "GOOD_CARBONATE_RESERVOIR"
        petro_labels[np.isin(working_groups, ["LIMESTONE", "DOLOMITE"]) & (quality == "EXCELLENT")] = "EXCELLENT_CARBONATE_RESERVOIR"

        gr_consistency = np.full(n_samples, np.nan, dtype=float)
        if gr is not None:
            valid = np.isfinite(gr)
            gr_consistency[valid] = 0.0
            gr_consistency[valid & (working_groups == "SHALE") & (gr > 80.0)] = 1.0
            gr_consistency[valid & np.isin(working_groups, ["SANDSTONE", "LIMESTONE", "DOLOMITE"]) & (gr < 60.0)] = 1.0

        pef_consistency = np.full(n_samples, np.nan, dtype=float)
        if pef is not None:
            valid = np.isfinite(pef)
            pef_consistency[valid] = 0.0
            pef_consistency[valid & np.isin(working_groups, ["LIMESTONE", "DOLOMITE"]) & (pef > 3.5)] = 1.0
            pef_consistency[valid & (working_groups == "SANDSTONE") & (pef >= 1.8) & (pef <= 3.0)] = 1.0
            pef_consistency[valid & (working_groups == "SHALE") & (pef < 2.5)] = 1.0

        consistency_score = np.full(n_samples, np.nan, dtype=float)
        consistency_stack = np.vstack([gr_consistency, pef_consistency])
        for idx in range(n_samples):
            values = consistency_stack[:, idx]
            valid = np.isfinite(values)
            if np.any(valid):
                consistency_score[idx] = float(np.nanmean(values[valid]))

        consistency_labels = np.full(n_samples, "INSUFFICIENT_DATA_FOR_CONSISTENCY", dtype=object)
        valid_consistency = np.isfinite(consistency_score)
        consistency_labels[valid_consistency] = "NO_LOG_LITHOLOGY_MATCH"
        consistency_labels[valid_consistency & (consistency_score >= 0.25)] = "POOR_LOG_LITHOLOGY_MATCH"
        consistency_labels[valid_consistency & (consistency_score >= 0.50)] = "GOOD_LOG_LITHOLOGY_MATCH"
        consistency_labels[valid_consistency & (consistency_score >= 0.75)] = "EXCELLENT_LOG_LITHOLOGY_MATCH"

        overall_labels = np.full(n_samples, "UNDETERMINED", dtype=object)
        overall_labels[np.char.find(petro_labels.astype(str), "SEAL") >= 0] = "SEAL_ROCK"
        overall_labels[np.char.find(petro_labels.astype(str), "TIGHT") >= 0] = "TIGHT_FORMATION"
        overall_labels[np.char.find(petro_labels.astype(str), "GOOD") >= 0] = "GOOD_RESERVOIR"
        overall_labels[np.char.find(petro_labels.astype(str), "EXCELLENT") >= 0] = "EXCELLENT_RESERVOIR"
        overall_labels[(overall_labels == "UNDETERMINED") & valid_gt] = gt_distribution[(overall_labels == "UNDETERMINED") & valid_gt]

        strong_res_fraction = float(
            np.count_nonzero(np.isin(petro_labels, [
                "EXCELLENT_SANDSTONE_RESERVOIR",
                "GOOD_SANDSTONE_RESERVOIR",
                "EXCELLENT_CARBONATE_RESERVOIR",
                "GOOD_CARBONATE_RESERVOIR",
            ]))
            / max(n_samples, 1)
        )
        seal_fraction = float(
            np.count_nonzero(petro_labels == "EVAPORITE_SEAL") / max(n_samples, 1)
        )

        formation_assessment = "LIMITED formation"
        if strong_res_fraction > 0.50:
            formation_assessment = "EXCELLENT formation"
        elif strong_res_fraction > 0.30:
            formation_assessment = "GOOD formation"
        elif strong_res_fraction > 0.15:
            formation_assessment = "FAIR formation"

        seal_assessment = "NO seal potential"
        if seal_fraction > 0.30:
            seal_assessment = "SIGNIFICANT seal potential"
        elif seal_fraction > 0.0:
            seal_assessment = "LIMITED seal potential"

        result = CategoryAnalysisResult(
            category_key=self.CATEGORY_KEY,
            sections={
                "lithology_distribution_classification": build_section(
                    well,
                    gt_distribution,
                    description="Ground-truth lithology distribution from the lithology curve.",
                    methodology="Maps raw lithology codes to grouped lithology classes.",
                    calculation_thresholds=(
                        "GT_{GROUP} for mapped codes: limestone 9/11/12/13/16, dolomite 15, "
                        "anhydrite 22, coal 18, calcareous shale 14, halite 21/27/32/33/34, "
                        "sandstone 1/2/3/4/10/30/35, shale 7/8/25, silt 5/6, coarse clastic 17/31/36, "
                        "evaporite 26, igneous 19/20/23/24/28, chert 29; otherwise NO_LITHOLOGY_DATA."
                    ),
                ),
                "carbonate_analysis": build_section(
                    well,
                    carbonate_labels,
                    description="Carbonate confirmation and mismatch analysis.",
                    methodology="Cross-checks PEF and RHOB carbonate indicators against lithology groups.",
                    calculation_thresholds=(
                        "LIMESTONE_INDICATOR: PEF > 4.5 and RHOB > 2.7; "
                        "DOLOMITE_INDICATOR: PEF 2.8-3.2 and RHOB 2.8-2.9; "
                        "CARBONATE_INDICATOR: PEF > 2.8 and RHOB > 2.6."
                    ),
                ),
                "clay_content_indicators": build_section(
                    well,
                    clay_labels,
                    description="Composite clay indication from GR, NPHI, and PEF.",
                    methodology="COMPOSITE_CLAY = mean(available indicators: GR>100 API, NPHI>0.30 v/v, PEF<2.5).",
                    calculation_thresholds=(
                        "HIGH_CLAY_INDICATION >= 0.67; MODERATE 0.33-0.67; LOW < 0.33."
                    ),
                ),
                "lithology_petrophysics_relationships": build_section(
                    well,
                    petro_labels,
                    description="Reservoir quality interpreted within lithology context.",
                    methodology=(
                        f"Combines grouped lithology with selected porosity={porosity_source_text}, "
                        f"permeability={perm_source_text}, and shale volume={vsh_source_text}."
                    ),
                    calculation_thresholds=(
                        "Quality base: FAIR when selected porosity >0.05, GOOD when >0.10, EXCELLENT when >0.15. "
                        "When selected permeability is available, GOOD also requires >1 mD and EXCELLENT requires >10 mD. "
                        "When shale volume is available, GOOD also requires <0.40 and EXCELLENT requires <0.25."
                    ),
                ),
                "lithology_log_consistency": build_section(
                    well,
                    consistency_labels,
                    description="Consistency between lithology labels and GR/PEF responses.",
                    methodology=(
                        "CONSISTENCY_SCORE is the mean of available GR and PEF checks: "
                        "GR consistent when shale has GR>80 API or sandstone/limestone/dolomite has GR<60 API; "
                        "PEF consistent when carbonate has PEF>3.5, sandstone has PEF 1.8-3.0, or shale has PEF<2.5."
                    ),
                    calculation_thresholds=(
                        "EXCELLENT >= 0.75; GOOD 0.50-0.75; POOR 0.25-0.50; "
                        "NO_MATCH < 0.25."
                    ),
                ),
                "overall_interpretation": build_section(
                    well,
                    overall_labels,
                    description="Primary lithology interpretation hierarchy.",
                    methodology="Uses petrophysical lithology labels first, then falls back to mapped lithology.",
                    calculation_thresholds=(
                        "SEAL_ROCK from EVAPORITE_SEAL; TIGHT_FORMATION from tight sandstone/carbonate; "
                        "GOOD_RESERVOIR and EXCELLENT_RESERVOIR from good/excellent sandstone or carbonate; "
                        "otherwise GT_{GROUP}."
                    ),
                ),
            },
            derived_curves={
                "GT_LITHOLOGY_GROUP_CODE": DerivedCurve(
                    classification_code(gt_groups, GROUP_CODES),
                    "",
                    "Ground-truth lithology group code",
                ),
                "LIMESTONE_INDICATOR": DerivedCurve(
                    limestone_indicator,
                    "",
                    "Limestone log indicator",
                ),
                "DOLOMITE_INDICATOR": DerivedCurve(
                    dolomite_indicator,
                    "",
                    "Dolomite log indicator",
                ),
                "CARBONATE_INDICATOR": DerivedCurve(
                    carbonate_indicator,
                    "",
                    "General carbonate log indicator",
                ),
                "GR_CLAY_INDICATOR": DerivedCurve(gr_clay, "", "GR clay indicator"),
                "NPHI_CLAY_INDICATOR": DerivedCurve(nphi_clay, "", "NPHI clay indicator"),
                "PEF_CLAY_INDICATOR": DerivedCurve(pef_clay, "", "PEF clay indicator"),
                "COMPOSITE_CLAY": DerivedCurve(composite_clay, "", "Composite clay indicator"),
                "CONSISTENCY_SCORE": DerivedCurve(
                    consistency_score,
                    "",
                    "Lithology-log consistency score",
                ),
            },
            metadata={
                "formation_level_summary": {
                    "reservoir_fraction": strong_res_fraction,
                    "seal_fraction": seal_fraction,
                    "formation_assessment": formation_assessment,
                    "seal_assessment": seal_assessment,
                }
            },
        )

        if add_curves:
            result.add_curves_to_well(well)

        return result
