"""
Comprehensive lithology analysis classifications.
"""

from __future__ import annotations

import numpy as np

from petrophys._internal.benchmark import reproducing_benchmark
from petrophys._internal.analysis_helpers import (
    CategoryAnalysisResult,
    DerivedCurve,
    build_section,
    classification_code,
    format_curve_source,
    get_curve_data,
)
from petrophys.interpretation.lithology.matrix_inference import interpreted_lithology_code_curve
from petrophys.evaluation.diagnostics._tables import (
    class_table_set,
    reservoir_quality,
    reservoir_quality_text,
    sample_families,
)
from petrophys.model.class_tables import CLASS_TABLES
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

#: Lithology codes the "current" calculation profile names differently from the "geolink" profile (see
#: ``_CURRENT_GROUND_TRUTH_CODE_TO_GROUP`` in
#: :mod:`petrophys.workflow.dynamic_workflow`): code 10 is "Porous Limestone"
#: in the Kingdom lithology codebook ("Calcareous Sandstone" is only its display
#: pattern), and code 26, "Gap", records no lithology (None).
CURRENT_PROFILE_CODE_NAMES = {10: "Porous Limestone", 26: None}

#: GROUND_TRUTH_GROUPS as the "current" calculation profile groups the codes:
#: code 10 with limestone, and code 26 in no group.
CURRENT_GROUND_TRUTH_GROUPS = {
    group: (codes | {10}) if group == "LIMESTONE" else codes - {10, 26}
    for group, codes in GROUND_TRUTH_GROUPS.items()
}

# Group names that the threshold text spells differently.
_GROUP_TEXT_NAMES = {"CLASTIC_COARSE": "coarse clastic"}


def _ground_truth_groups_text(groups) -> str:
    """The code list of each lithology group, for the threshold text."""
    parts = [
        f"{_GROUP_TEXT_NAMES.get(group, group.lower().replace('_', ' '))} "
        + "/".join(str(code) for code in sorted(codes))
        for group, codes in groups.items()
        if codes
    ]
    return "GT_{GROUP} for mapped codes: " + ", ".join(parts) + "; otherwise NO_LITHOLOGY_DATA."


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

# Rock families whose samples the lithology-petrophysics section grades with
# the reservoir-quality table, and their lithology groups ("current" class tables).
_RESERVOIR_ROCKS = {"SANDSTONE": ("SANDSTONE",), "CARBONATE": ("LIMESTONE", "DOLOMITE")}

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
    """Ground-truth and log-based lithology analysis.

    The ground-truth codes are read from the lithology code curve the
    interpretation of the well read: ``LITHOLOGY_GEOLINK``, or the curve the
    dynamic workflow recorded on the well (another lithology-code curve that
    a ``"current"`` run read, or the one ``lithology_curve`` named).

    Parameters
    ----------
    ground_truth_groups : dict, optional
        Lithology group -> set of ``LITHOLOGY_GEOLINK`` codes. Default
        ``GROUND_TRUTH_GROUPS``; ``CURRENT_GROUND_TRUTH_GROUPS`` groups the
        codes as the "current" calculation profile does.
    class_tables : {"current", "geolink"}, default "current"
        ``"current"`` grades sandstone and carbonate samples with the
        reservoir-quality table of
        :data:`~petrophys.model.class_tables.CLASS_TABLES` (effective porosity,
        permeability and clay volume, the classes the porosity, permeability
        and hydrocarbon analyses give), and the overall interpretation takes
        that class. ``"geolink"`` keeps the GeoLink eval package's grades
        (FAIR from porosity alone; GOOD and EXCELLENT also below a shale
        volume) and texts.
    lithology_family : str or sequence of str, optional
        Lithology family (``"clastic"``, ``"carbonate"``) or lithology of every
        sample, or one per sample, whose limits the reservoir-quality table
        applies under ``"current"``. None (default): clastic.
    """

    CATEGORY_KEY = "COMPREHENSIVE_LITHOLOGY_ANALYSIS"

    def __init__(self, ground_truth_groups=None, class_tables: str = "current", lithology_family=None):
        self.ground_truth_groups = (
            GROUND_TRUTH_GROUPS if ground_truth_groups is None else ground_truth_groups
        )
        self.class_tables = class_table_set(class_tables)
        self.lithology_family = lithology_family

    def analyze(
        self,
        well: "Well",
        add_curves: bool = True,
    ) -> CategoryAnalysisResult:
        benchmark = self.class_tables == "geolink"
        n_samples = well.n_samples
        gt_codes, _ = get_curve_data(well, [interpreted_lithology_code_curve(well)])
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
            for group_name, codes in self.ground_truth_groups.items():
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
        mismatch = gt_carbonate & valid_carbonate & (carbonate_indicator <= 0.5)
        if not (benchmark or reproducing_benchmark()):
            # PEF = 2.8 meets the dolomite indicator (>= 2.8) but not the carbonate one (> 2.8).
            mismatch &= ~(dolomite_indicator > 0.5)
        carbonate_labels[mismatch] = "CARBONATE_LOG_MISMATCH"
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

        quality_table = CLASS_TABLES["RESERVOIR_QUALITY"]
        if benchmark:
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
            petro_methodology = (
                f"Combines grouped lithology with selected porosity={porosity_source_text}, "
                f"permeability={perm_source_text}, and shale volume={vsh_source_text}."
            )
        else:
            families = sample_families(self.lithology_family, n_samples)
            quality, quality_clay_source = reservoir_quality(well, phit, perm, families)
            petro_methodology = (
                f"Combines grouped lithology with the reservoir quality of selected porosity={porosity_source_text}, "
                f"permeability={perm_source_text} and clay volume={format_curve_source(quality_clay_source)}, "
                "with the limits of each sample's lithology family, and with shale "
                f"volume={vsh_source_text} for shale samples."
            )

        petro_labels = np.full(n_samples, "SPECIAL_LITHOLOGY", dtype=object)
        petro_labels[np.isin(working_groups, ["EVAPORITE", "HALITE", "ANHYDRITE"])] = "EVAPORITE_SEAL"
        if vsh is not None:
            petro_labels[(working_groups == "SHALE") & np.isfinite(vsh)] = "TYPICAL_SHALE"
            petro_labels[(working_groups == "SHALE") & np.isfinite(vsh) & (vsh <= 0.7)] = "SANDY_SHALE"
        if benchmark:
            petro_labels[np.isin(working_groups, ["SANDSTONE"]) & np.isin(quality, ["FAIR", "POOR"])] = "TIGHT_SANDSTONE"
            petro_labels[np.isin(working_groups, ["SANDSTONE"]) & (quality == "GOOD")] = "GOOD_SANDSTONE_RESERVOIR"
            petro_labels[np.isin(working_groups, ["SANDSTONE"]) & (quality == "EXCELLENT")] = "EXCELLENT_SANDSTONE_RESERVOIR"
            petro_labels[np.isin(working_groups, ["LIMESTONE", "DOLOMITE"]) & np.isin(quality, ["FAIR", "POOR"])] = "TIGHT_CARBONATE"
            petro_labels[np.isin(working_groups, ["LIMESTONE", "DOLOMITE"]) & (quality == "GOOD")] = "GOOD_CARBONATE_RESERVOIR"
            petro_labels[np.isin(working_groups, ["LIMESTONE", "DOLOMITE"]) & (quality == "EXCELLENT")] = "EXCELLENT_CARBONATE_RESERVOIR"
        else:
            for rock, groups in _RESERVOIR_ROCKS.items():
                in_rock = np.isin(working_groups, groups)
                petro_labels[in_rock] = f"{rock}_NO_QUALITY_DATA"
                petro_labels[in_rock & (quality == "NON_RESERVOIR")] = f"TIGHT_{rock}"
                for grade in ("FAIR", "GOOD", "EXCELLENT"):
                    petro_labels[in_rock & (quality == f"{grade}_RESERVOIR")] = f"{grade}_{rock}_RESERVOIR"

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
        if benchmark:
            overall_labels[np.char.find(petro_labels.astype(str), "SEAL") >= 0] = "SEAL_ROCK"
            overall_labels[np.char.find(petro_labels.astype(str), "TIGHT") >= 0] = "TIGHT_FORMATION"
            overall_labels[np.char.find(petro_labels.astype(str), "GOOD") >= 0] = "GOOD_RESERVOIR"
            overall_labels[np.char.find(petro_labels.astype(str), "EXCELLENT") >= 0] = "EXCELLENT_RESERVOIR"
        else:
            overall_labels[petro_labels == "EVAPORITE_SEAL"] = "SEAL_ROCK"
            reservoir_rock = (
                np.isin(working_groups, [group for groups in _RESERVOIR_ROCKS.values() for group in groups])
                & np.isin(quality, quality_table.labels)
            )
            overall_labels[reservoir_rock] = quality[reservoir_rock]
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
                    calculation_thresholds=_ground_truth_groups_text(self.ground_truth_groups),
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
                    methodology=petro_methodology,
                    calculation_thresholds=(
                        (
                            "Quality base: FAIR when selected porosity >0.05, GOOD when >0.10, EXCELLENT when >0.15. "
                            "When selected permeability is available, GOOD also requires >1 mD and EXCELLENT requires >10 mD. "
                            "When shale volume is available, GOOD also requires <0.40 and EXCELLENT requires <0.25."
                        )
                        if benchmark
                        else (
                            f"Sandstone and carbonate quality: {reservoir_quality_text(families)} "
                            "TIGHT_{ROCK} for NON_RESERVOIR, {GRADE}_{ROCK}_RESERVOIR for FAIR, GOOD and "
                            "EXCELLENT, {ROCK}_NO_QUALITY_DATA where effective porosity, permeability or clay "
                            "volume is missing."
                        )
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
                        (
                            "SEAL_ROCK from EVAPORITE_SEAL; TIGHT_FORMATION from tight sandstone/carbonate; "
                            "GOOD_RESERVOIR and EXCELLENT_RESERVOIR from good/excellent sandstone or carbonate; "
                            "otherwise GT_{GROUP}."
                        )
                        if benchmark
                        else (
                            "SEAL_ROCK from EVAPORITE_SEAL; sandstone and carbonate samples take their "
                            f"reservoir-quality class ({', '.join(quality_table.labels)}); otherwise GT_{{GROUP}}."
                        )
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
