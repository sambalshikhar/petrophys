"""
Numerical and statistical analysis classifications.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.cluster.vq import kmeans2

from petrophys.petrophysics.elastic import calculate_elastic_properties
from petrophys.utils.analysis_helpers import (
    CategoryAnalysisResult,
    DerivedCurve,
    build_section,
    depth_step_array,
    depth_to_meters,
    format_curve_source,
    get_curve_data,
)
from petrophys.reservoir.net_pay.net_pay_module import NetPayModule


class NumericalStatisticalAnalysis:
    """Numerical and statistical data-quality analysis."""

    CATEGORY_KEY = "NUMERICAL_AND_STATISTICAL_ANALYSIS"

    def analyze(
        self,
        well: "Well",
        add_curves: bool = True,
    ) -> CategoryAnalysisResult:
        n_samples = well.n_samples
        numeric_curve_names = self._numeric_curve_names(well)
        curve_arrays = {
            name: np.asarray(well.get(name).data, dtype=float)
            for name in numeric_curve_names
            if well.get(name) is not None
        }

        derived_curves: dict[str, DerivedCurve] = {}
        total_logs = max(len(curve_arrays), 1)
        completeness = np.full(n_samples, np.nan, dtype=float)
        for idx in range(n_samples):
            available = sum(int(np.isfinite(values[idx])) for values in curve_arrays.values())
            completeness[idx] = available / total_logs
        derived_curves["DATA_COMPLETENESS"] = DerivedCurve(
            completeness,
            "",
            "Fraction of available numerical logs per sample",
        )

        coverage_labels = np.full(n_samples, "POOR_DATA_COVERAGE", dtype=object)
        valid = np.isfinite(completeness)
        coverage_labels[valid & (completeness >= 0.40)] = "MODERATE_DATA_COVERAGE"
        coverage_labels[valid & (completeness >= 0.60)] = "GOOD_DATA_COVERAGE"
        coverage_labels[valid & (completeness >= 0.80)] = "EXCELLENT_DATA_COVERAGE"

        gradient_curves = {}
        for name in ["GR", "RHOB", "RDEP", "NPHI", "DTC"]:
            data, _ = get_curve_data(well, [name, "RT" if name == "RDEP" else name, "DT" if name == "DTC" else name])
            if data is None:
                continue
            grad = self._gradient(data, well.depth)
            gradient_curves[f"{name}_GRAD"] = grad
            derived_curves[f"{name}_GRAD"] = DerivedCurve(grad, "", f"{name} gradient")

        trend_labels = np.full(n_samples, "STABLE_ZONE", dtype=object)
        if gradient_curves:
            stack = np.vstack(list(gradient_curves.values()))
            significant = np.sum(np.isfinite(stack) & (np.abs(stack) > 0.1), axis=0)
            trend_labels[significant >= 1] = "MODERATE_TREND_ZONE"
            trend_labels[significant >= 3] = "STRONG_TREND_ZONE"

        correlation_ratio = np.full(n_samples, np.nan, dtype=float)
        correlation_labels = np.full(n_samples, "UNKNOWN_CONSISTENCY", dtype=object)
        relation_specs = [
            (["GR"], ["VSH", "VCL", "VCLAY"], "positive"),
            (["RHOB"], ["PHIE_CC", "PHIT", "PHIE"], "negative"),
            (["NPHI", "PHIN"], ["PHIE_CC", "PHIT", "PHIE"], "positive"),
            (["RDEP", "RT"], ["SWA", "SW", "SWT-LOG"], "negative"),
            (["PHIE_CC", "PHIT", "PHIE"], ["PERM_CC", "PERM", "PERM_CORE"], "positive"),
        ]
        relation_cache: list[tuple[np.ndarray, np.ndarray, str]] = []
        relation_sources: list[str] = []
        for left_names, right_names, relation in relation_specs:
            left, left_source = get_curve_data(well, left_names)
            right, right_source = get_curve_data(well, right_names)
            if left is None or right is None:
                continue
            relation_cache.append((left, right, relation))
            relation_sources.append(
                f"{format_curve_source(left_source)} vs {format_curve_source(right_source)} ({relation})"
            )
        relation_source_text = "; ".join(relation_sources) if relation_sources else "no supported pairs"

        for idx in range(n_samples):
            valid_pairs = 0
            consistent_pairs = 0
            for left, right, relation in relation_cache:
                lv = left[idx]
                rv = right[idx]
                if not (np.isfinite(lv) and np.isfinite(rv)):
                    continue
                valid_pairs += 1
                lmean = float(np.nanmean(left))
                rmean = float(np.nanmean(right))
                if relation == "positive":
                    consistent = (lv >= lmean and rv >= rmean) or (lv < lmean and rv < rmean)
                else:
                    consistent = (lv >= lmean and rv < rmean) or (lv < lmean and rv >= rmean)
                consistent_pairs += int(consistent)
            if valid_pairs > 0:
                correlation_ratio[idx] = consistent_pairs / valid_pairs

        valid_ratio = np.isfinite(correlation_ratio)
        correlation_labels[valid_ratio] = "POOR_LOG_CONSISTENCY"
        correlation_labels[valid_ratio & (correlation_ratio >= 0.60)] = "GOOD_LOG_CONSISTENCY"
        correlation_labels[valid_ratio & (correlation_ratio >= 0.80)] = "EXCELLENT_LOG_CONSISTENCY"
        derived_curves["CONSISTENCY_RATIO"] = DerivedCurve(
            correlation_ratio,
            "",
            "Expected log-relation consistency ratio",
        )

        deviation_count = np.zeros(n_samples, dtype=int)
        extreme_count = np.zeros(n_samples, dtype=int)
        for name, values in curve_arrays.items():
            if name.endswith("FLAG") or name.endswith("_CODE"):
                continue
            mean = float(np.nanmean(values))
            std = float(np.nanstd(values))
            if not np.isfinite(std) or std <= 0.0:
                continue
            z = np.abs((values - mean) / std)
            deviation_count += np.where(np.isfinite(z) & (z > 2.0), 1, 0)
            extreme_count += np.where(np.isfinite(z) & (z > 3.0), 1, 0)

        deviation_labels = np.full(n_samples, "NORMAL_ZONE", dtype=object)
        deviation_labels[(deviation_count >= 3) | (extreme_count >= 1)] = "SIGNIFICANT_DEVIATION_ZONE"
        deviation_labels[extreme_count >= 2] = "EXTREME_DEVIATION_ZONE"

        badhole, _ = get_curve_data(well, ["BAD_HOLE", "BAD_HOLE_FLAG"])
        drho, _ = get_curve_data(well, ["DRHO"])
        quality_labels = np.full(n_samples, "UNKNOWN_LOG_QUALITY", dtype=object)
        if badhole is not None:
            valid = np.isfinite(badhole)
            quality_labels[valid] = "GOOD_LOG_QUALITY"
            quality_labels[valid & (badhole > 0.5)] = "POOR_LOG_QUALITY"
        if drho is not None:
            poor = np.isfinite(drho) & (np.abs(drho) > 0.1)
            quality_labels[poor] = "POOR_LOG_QUALITY"
            if badhole is None:
                quality_labels[np.isfinite(drho) & ~poor] = "GOOD_LOG_QUALITY"

        rhob, _ = get_curve_data(well, ["RHOB"])
        dtc, _ = get_curve_data(well, ["DTC", "DT"])
        dts, _ = get_curve_data(well, ["DTS"])
        acoustic_impedance = np.full(n_samples, np.nan, dtype=float)
        poisson_ratio = np.full(n_samples, np.nan, dtype=float)
        young_mod = np.full(n_samples, np.nan, dtype=float)
        brittleness = np.full(n_samples, np.nan, dtype=float)
        electrofacies = np.full(n_samples, np.nan, dtype=float)

        if rhob is not None and dtc is not None:
            elastic = calculate_elastic_properties(rhob, dtc, dts)
            acoustic_impedance = np.clip(elastic.acoustic_impedance, 1e6, 30e6)
            poisson_ratio = elastic.poisson_ratio
            young_mod = elastic.young_mod_gpa

        if np.any(np.isfinite(acoustic_impedance)):
            derived_curves["ACOUSTIC_IMPEDANCE"] = DerivedCurve(
                acoustic_impedance,
                "kg/m2/s",
                "Acoustic impedance",
            )

        elastic_valid = np.isfinite(poisson_ratio) & np.isfinite(young_mod)
        if np.any(elastic_valid):
            e_norm = np.clip((young_mod[elastic_valid] - 5.0) / 80.0, 0.0, 1.0)
            v_norm = np.clip((0.45 - poisson_ratio[elastic_valid]) / 0.35, 0.0, 1.0)
            brittleness[elastic_valid] = np.clip((e_norm + v_norm) / 2.0, 0.0, 1.0)
            derived_curves["POISSON_RATIO"] = DerivedCurve(poisson_ratio, "", "Poisson ratio")
            derived_curves["YOUNG_MOD"] = DerivedCurve(young_mod, "GPa", "Young's modulus")
            derived_curves["BRITTLENESS_INDEX"] = DerivedCurve(
                brittleness,
                "",
                "Brittleness index",
            )

        electrofacies = self._electrofacies(well, n_samples)
        if np.any(np.isfinite(electrofacies)):
            derived_curves["ELECTROFACIES"] = DerivedCurve(
                electrofacies,
                "",
                "Electrofacies cluster label",
            )

        advanced_labels = np.full(n_samples, "NO_ADVANCED_DATA", dtype=object)
        advanced_available = (
            (np.isfinite(brittleness) & (brittleness > 0.6))
            | (np.isfinite(acoustic_impedance) & (acoustic_impedance > 15e6))
            | (np.isfinite(young_mod) & (young_mod > 30.0))
            | np.isfinite(electrofacies)
        )
        advanced_labels[advanced_available] = "ADVANCED_DATA_AVAILABLE"

        depth_m = depth_to_meters(well.depth, well.depth_unit)
        dz_m = depth_step_array(depth_m)
        sv = np.full(n_samples, np.nan, dtype=float)
        pp = np.full(n_samples, np.nan, dtype=float)
        sh_min = np.full(n_samples, np.nan, dtype=float)
        if rhob is not None:
            valid = np.isfinite(rhob)
            sv[valid] = np.cumsum((rhob[valid] * 1000.0) * 9.81 * dz_m[valid]) / 1e6
        if depth_m.size:
            seawater = (1025.0 * 9.81 * 100.0 / 1e6) if np.nanmax(np.abs(depth_m)) > 200.0 else 0.0
            pp = (1000.0 * 9.81 * np.abs(depth_m) / 1e6) + seawater
        valid_sh = np.isfinite(poisson_ratio) & np.isfinite(sv) & np.isfinite(pp) & (poisson_ratio < 0.999)
        sh_min[valid_sh] = ((poisson_ratio[valid_sh] / (1.0 - poisson_ratio[valid_sh])) * (sv[valid_sh] - pp[valid_sh])) + pp[valid_sh]
        derived_curves["S_V"] = DerivedCurve(sv, "MPa", "Overburden stress")
        derived_curves["PP_HYDROSTATIC"] = DerivedCurve(pp, "MPa", "Hydrostatic pore pressure")
        derived_curves["SH_MIN"] = DerivedCurve(sh_min, "MPa", "Minimum horizontal stress")

        geomech_labels = np.full(n_samples, "NO_GEOMECH_DATA", dtype=object)
        geomech_count = np.sum(
            np.vstack([np.isfinite(sv), np.isfinite(pp), np.isfinite(sh_min)]),
            axis=0,
        )
        geomech_labels[geomech_count >= 2] = "GEOMECH_DATA_AVAILABLE"

        net_pay_curve, _ = get_curve_data(well, ["NET_PAY"])
        if net_pay_curve is None:
            try:
                net_pay_curve = NetPayModule().calculate(well).net_pay.astype(float)
                derived_curves["NET_PAY"] = DerivedCurve(net_pay_curve, "", "Net pay flag")
            except Exception:
                net_pay_curve = None
        net_pay_labels = np.full(n_samples, "NO_PAY_DATA", dtype=object)
        if net_pay_curve is not None:
            valid = np.isfinite(net_pay_curve)
            net_pay_labels[valid] = "NON_PAY"
            net_pay_labels[valid & (net_pay_curve > 0.5)] = "NET_PAY"

        integrated_labels = np.full(n_samples, "NORMAL_ZONE", dtype=object)
        integrated_labels[quality_labels == "GOOD_LOG_QUALITY"] = "GOOD_QUALITY_ZONE"
        integrated_labels[correlation_labels == "EXCELLENT_LOG_CONSISTENCY"] = "HIGH_CONFIDENCE_ZONE"
        integrated_labels[quality_labels == "POOR_LOG_QUALITY"] = "POOR_LOG_QUALITY_ZONE"
        integrated_labels[deviation_labels == "EXTREME_DEVIATION_ZONE"] = "EXTREME_DEVIATION_ZONE"
        integrated_labels[net_pay_labels == "NET_PAY"] = "NET_PAY_ZONE"

        result = CategoryAnalysisResult(
            category_key=self.CATEGORY_KEY,
            sections={
                "statistical_data_coverage_analysis": build_section(
                    well,
                    coverage_labels,
                    description="Per-sample numerical log completeness.",
                    methodology="Available numerical logs divided by total numerical logs.",
                    calculation_thresholds=(
                        "EXCELLENT >= 0.80; GOOD 0.60-0.79; MODERATE 0.40-0.59; POOR < 0.40."
                    ),
                ),
                "log_gradient_trend_analysis": build_section(
                    well,
                    trend_labels,
                    description="Trend intensity from key-log gradients.",
                    methodology="Counts significant gradients across GR, RHOB, RDEP, NPHI, and DTC.",
                    calculation_thresholds=(
                        "STRONG_TREND_ZONE when >= 3 gradients exceed |0.1|; "
                        "MODERATE when 1-2; STABLE when none."
                    ),
                ),
                "log_correlation_consistency_evaluation": build_section(
                    well,
                    correlation_labels,
                    description="Expected petrophysical relationship consistency.",
                    methodology=(
                        "Checks each supported log pair against its expected sign relative to the mean. "
                        f"Pairs used: {relation_source_text}."
                    ),
                    calculation_thresholds=(
                        "EXCELLENT >= 0.80; GOOD 0.60-0.80; POOR < 0.60; UNKNOWN when no pairs are available."
                    ),
                ),
                "statistical_deviation_detection": build_section(
                    well,
                    deviation_labels,
                    description="Per-sample outlier counts across numerical logs.",
                    methodology="Uses z-scores relative to whole-well statistics.",
                    calculation_thresholds=(
                        "EXTREME when >= 2 z-scores exceed 3.0; SIGNIFICANT when >= 1 exceeds 3.0 "
                        "or >= 3 exceed 2.0."
                    ),
                ),
                "log_quality_control_assessment": build_section(
                    well,
                    quality_labels,
                    description="Log quality from BAD_HOLE and DRHO.",
                    methodology="Poor quality is triggered by bad-hole flags or density-correction magnitude.",
                    calculation_thresholds="POOR_LOG_QUALITY if BAD_HOLE > 0.5 or |DRHO| > 0.1.",
                ),
                "advanced_parameter_characterization": build_section(
                    well,
                    advanced_labels,
                    description="Acoustic, elastic, and electrofacies characterization.",
                    methodology=(
                        "Computes Vp=304800/DTC from DTC 20-300 us/ft and Vs=304800/DTS from DTS "
                        "20-600 us/ft; then computes AI, Young's modulus, brittleness, and electrofacies."
                    ),
                    calculation_thresholds=(
                        "ADVANCED_DATA_AVAILABLE when brittleness > 0.6, AI > 15e6, "
                        "Young's modulus > 30 GPa, or electrofacies are available."
                    ),
                ),
                "geomechanical_stress_analysis": build_section(
                    well,
                    geomech_labels,
                    description="Geomechanical stress availability screening.",
                    methodology=(
                        "Computes S_V from cumulative RHOB*9.81*dz, PP_HYDROSTATIC from water gradient, "
                        "and SH_MIN from Poisson ratio with S_V and PP."
                    ),
                    calculation_thresholds="GEOMECH_DATA_AVAILABLE when at least two geomechanical parameters are finite.",
                ),
                "net_pay_flag_analysis": build_section(
                    well,
                    net_pay_labels,
                    description="Net pay interpretation using the library's existing net-pay behavior.",
                    methodology="Uses existing NET_PAY when present or computes it with the current NetPayModule defaults.",
                    calculation_thresholds="NET_PAY when NET_PAY = 1, otherwise NON_PAY.",
                ),
                "integrated_numerical_assessment": build_section(
                    well,
                    integrated_labels,
                    description="Integrated data-confidence classification.",
                    methodology="Prioritizes net pay, deviation severity, log quality, and consistency.",
                    calculation_thresholds=(
                        "NET_PAY_ZONE, EXTREME_DEVIATION_ZONE, POOR_LOG_QUALITY_ZONE, "
                        "HIGH_CONFIDENCE_ZONE, GOOD_QUALITY_ZONE, otherwise NORMAL_ZONE."
                    ),
                ),
            },
            derived_curves=derived_curves,
        )

        if add_curves:
            result.add_curves_to_well(well)

        return result

    @staticmethod
    def _numeric_curve_names(well: "Well") -> Sequence[str]:
        excluded = {
            "DEPT",
            "DEPTH",
            "LITHOLOGY_GEOLINK",
            "FORMATION",
            "GDEN",
            "PHIC",
            "KH_CORE",
            "KV_CORE",
            "SWC",
            "SOC",
        }
        return [name for name in well.curves if name.upper() not in excluded]

    @staticmethod
    def _gradient(values: np.ndarray, depth: np.ndarray) -> np.ndarray:
        result = np.full_like(values, np.nan, dtype=float)
        valid = np.isfinite(values) & np.isfinite(depth)
        if np.count_nonzero(valid) < 2:
            return result
        result[valid] = np.gradient(values[valid], depth[valid])
        return result

    @staticmethod
    def _electrofacies(well: "Well", n_samples: int) -> np.ndarray:
        features = []
        for names in (["RHOB"], ["NPHI", "PHIN"], ["GR"], ["RDEP", "RT"]):
            values, _ = get_curve_data(well, names)
            if values is not None:
                features.append(values)
        if len(features) < 2:
            return np.full(n_samples, np.nan, dtype=float)

        matrix = np.vstack(features).T
        valid = np.all(np.isfinite(matrix), axis=1)
        if np.count_nonzero(valid) < 2:
            return np.full(n_samples, np.nan, dtype=float)

        x = matrix[valid]
        mean = np.nanmean(x, axis=0)
        std = np.nanstd(x, axis=0)
        std[std <= 0.0] = 1.0
        x_scaled = (x - mean) / std
        n_clusters = min(5, x_scaled.shape[0])
        if n_clusters < 2:
            return np.full(n_samples, np.nan, dtype=float)

        _, labels = kmeans2(x_scaled, n_clusters, minit="points")
        electrofacies = np.full(n_samples, np.nan, dtype=float)
        electrofacies[valid] = labels.astype(float)
        return electrofacies
