"""
Numerical and statistical analysis classifications.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np
from scipy.cluster.vq import ClusterError, kmeans2

from petrophys.interpretation.rock_physics.elastic import elastic_properties
from petrophys._internal.benchmark import reproducing_benchmark
from petrophys._internal.analysis_helpers import (
    CategoryAnalysisResult,
    DerivedCurve,
    build_section,
    depth_step_array,
    depth_to_meters,
    format_curve_source,
    get_curve_data,
)
from petrophys.evaluation.net_pay.net_pay_module import NetPayModule
from petrophys.interpretation.lithology.matrix_inference import interpreted_lithology_code_curve

if TYPE_CHECKING:
    from petrophys.model.well import Well


class NumericalStatisticalAnalysis:
    """Numerical and statistical data-quality analysis."""

    CATEGORY_KEY = "NUMERICAL_AND_STATISTICAL_ANALYSIS"

    #: Bulk density (g/cc) assumed between the surface and the top of the density log.
    RHO_OVERBURDEN = 2.3
    #: Seawater density (kg/m3) of the water column above the sea floor.
    RHO_SEAWATER = 1025.0

    def analyze(
        self,
        well: "Well",
        add_curves: bool = True,
        rho_overburden: float = RHO_OVERBURDEN,
        water_depth_m: float = 0.0,
    ) -> CategoryAnalysisResult:
        """Classify the well's numerical data quality and derive screening curves.

        ``rho_overburden`` (g/cc) is the bulk density assumed from the surface
        (or the sea floor) down to the first density sample, and
        ``water_depth_m`` the water column above the sea floor, both measured
        from the depth datum; S_V and PP_HYDROSTATIC are absolute stresses
        from that datum. Inside the benchmark reproduction S_V is the
        increment from the top of the density log and PP_HYDROSTATIC adds a
        fixed 100 m of seawater below 200 m, as the packages were built.
        """
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
        if all(values.shape == (n_samples,) for values in curve_arrays.values()):
            available = np.zeros(n_samples, dtype=np.int64)
            for values in curve_arrays.values():
                available += np.isfinite(values)
            completeness[:] = available / total_logs
        else:
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
            if reproducing_benchmark():
                stack = np.vstack(list(gradient_curves.values()))
            else:
                # Each gradient in units of its curve's own P10-P90 spread per metre
                # (resistivity on log10), so GR in API and RHOB in g/cc are judged
                # alike; a feet well's gradients are per foot.
                stack = np.vstack([
                    self._relative_gradient(well, name[: -len("_GRAD")])
                    for name in gradient_curves
                ]) / self._metres_per_depth_unit(well)
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
            finite_values = values[np.isfinite(values)]
            if finite_values.size == 0:
                continue
            mean = float(np.nanmean(values))
            std = float(np.nanstd(values))
            if not np.isfinite(std) or std <= 0.0:
                continue
            z = np.abs((values - mean) / std)
            deviation_count += np.where(
                np.isfinite(z) & (z > 2.0),
                1,
                0,
            )
            extreme_count += np.where(
                np.isfinite(z) & (z > 3.0),
                1,
                0,
            )

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
            elastic = elastic_properties(rhob, dtc, dts)
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
        benchmark = reproducing_benchmark()
        sv = np.full(n_samples, np.nan, dtype=float)
        pp = np.full(n_samples, np.nan, dtype=float)
        sh_min = np.full(n_samples, np.nan, dtype=float)
        if rhob is not None and benchmark:
            dz_m = depth_step_array(depth_m)
            valid = np.isfinite(rhob)
            if np.any(valid):
                # Context stress is an increment across the analyzed interval.
                sv[valid] = np.cumsum(
                    (rhob[valid] * 1000.0) * 9.81 * dz_m[valid]
                ) / 1e6
        elif rhob is not None:
            sv = self._overburden_stress(rhob, depth_m, rho_overburden, water_depth_m)
        finite_depth_m = depth_m[np.isfinite(depth_m)]
        if finite_depth_m.size and benchmark:
            seawater = (1025.0 * 9.81 * 100.0 / 1e6) if np.nanmax(np.abs(finite_depth_m)) > 200.0 else 0.0
            pp = (1000.0 * 9.81 * np.abs(depth_m) / 1e6) + seawater
        elif finite_depth_m.size:
            # Water gradient from the datum; the seawater column (if any) is the
            # denser part of it, so it only adds the density excess over fresh water.
            seawater = (self.RHO_SEAWATER - 1000.0) * 9.81 * float(water_depth_m) / 1e6
            pp = (1000.0 * 9.81 * np.abs(depth_m) / 1e6) + seawater
        valid_sh = np.isfinite(poisson_ratio) & np.isfinite(sv) & np.isfinite(pp) & (poisson_ratio < 0.999)
        sh_min[valid_sh] = ((poisson_ratio[valid_sh] / (1.0 - poisson_ratio[valid_sh])) * (sv[valid_sh] - pp[valid_sh])) + pp[valid_sh]
        if benchmark:
            derived_curves["S_V"] = DerivedCurve(
                sv, "MPa", "Overburden stress increment from the top of the analysed interval",
                basis="an increment from the top of the analysed interval, not absolute",
            )
            derived_curves["PP_HYDROSTATIC"] = DerivedCurve(pp, "MPa", "Hydrostatic pore pressure")
            derived_curves["SH_MIN"] = DerivedCurve(
                sh_min, "MPa", "Minimum horizontal stress from the S_V increment, not absolute",
                basis="from the S_V increment, not absolute",
            )
        else:
            datum = (f"from the datum ({rho_overburden:g} g/cc above the density log, "
                     f"{float(water_depth_m):g} m of water)")
            derived_curves["S_V"] = DerivedCurve(sv, "MPa", f"Overburden stress {datum}", basis=datum)
            water = f"from the datum ({float(water_depth_m):g} m of water)"
            derived_curves["PP_HYDROSTATIC"] = DerivedCurve(pp, "MPa", f"Hydrostatic pore pressure {water}",
                                                            basis=water)
            derived_curves["SH_MIN"] = DerivedCurve(
                sh_min, "MPa", "Minimum horizontal stress from Poisson ratio with absolute S_V and PP",
                basis="from Poisson ratio with absolute S_V and PP",
            )

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
                        if benchmark else
                        "A gradient is significant above |0.1| of its curve's P10-P90 range per metre "
                        "(RDEP on log10). STRONG_TREND_ZONE when >= 3 gradients are significant; "
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
                        if benchmark else
                        f"Computes S_V from the datum ({rho_overburden:g} g/cc above the density log, "
                        f"{float(water_depth_m):g} m of water, then the integrated RHOB with internal "
                        "gaps interpolated), PP_HYDROSTATIC from the water gradient over the same "
                        "column, and SH_MIN from Poisson ratio with S_V and PP."
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
            # The lithology code curve holds categories, not measurements.
            interpreted_lithology_code_curve(well),
        }
        return [name for name in well.curves if name.upper() not in excluded]

    @staticmethod
    def _metres_per_depth_unit(well: "Well") -> float:
        return float(depth_to_meters(np.ones(1), well.depth_unit)[0])

    @classmethod
    def _overburden_stress(
        cls,
        rhob: np.ndarray,
        depth_m: np.ndarray,
        rho_overburden: float,
        water_depth_m: float,
    ) -> np.ndarray:
        """Absolute overburden stress (MPa) from the datum.

        The water column weighs ``RHO_SEAWATER``, the sediment above the first
        density sample ``rho_overburden`` (g/cc), and below it the density log
        is integrated with the trapezoid rule over the actual depths; samples
        inside an internal RHOB gap take the linearly interpolated density, so
        the gap still adds its weight. Samples above the first or below the
        last density sample, or without a finite depth, are NaN.
        """
        sv = np.full(rhob.shape, np.nan, dtype=float)
        valid = np.isfinite(rhob) & np.isfinite(depth_m)
        if not np.any(valid):
            return sv
        order = np.argsort(depth_m, kind="stable")
        depth_sorted = depth_m[order]
        rhob_sorted = rhob[order]
        finite = np.isfinite(rhob_sorted) & np.isfinite(depth_sorted)
        first, last = np.flatnonzero(finite)[[0, -1]]
        covered = np.zeros(rhob.shape, dtype=bool)
        covered[first:last + 1] = np.isfinite(depth_sorted[first:last + 1])
        z = depth_sorted[covered]
        rho = np.interp(z, depth_sorted[finite], rhob_sorted[finite]) * 1000.0
        water = min(max(float(water_depth_m), 0.0), max(float(z[0]), 0.0))
        above = (cls.RHO_SEAWATER * water + rho_overburden * 1000.0 * (z[0] - water)) * 9.81
        increments = 0.5 * (rho[1:] + rho[:-1]) * np.diff(z) * 9.81
        sv[order[covered]] = (above + np.concatenate([[0.0], np.cumsum(increments)])) / 1e6
        return sv

    @classmethod
    def _relative_gradient(cls, well: "Well", name: str) -> np.ndarray:
        """Gradient of a curve divided by its P10-P90 range (RDEP on log10), per depth unit."""
        aliases = {"RDEP": ["RDEP", "RT"], "DTC": ["DTC", "DT"]}.get(name, [name])
        data, _ = get_curve_data(well, aliases)
        values = np.asarray(data, dtype=float)
        if name == "RDEP":
            with np.errstate(divide="ignore", invalid="ignore"):
                values = np.where(values > 0, np.log10(values), np.nan)
        finite = values[np.isfinite(values)]
        if finite.size < 2:
            return np.full(well.n_samples, np.nan)
        spread = float(np.percentile(finite, 90) - np.percentile(finite, 10))
        if not spread > 0:
            return np.full(well.n_samples, np.nan)
        return cls._gradient(values, well.depth) / spread

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
        unique_count = np.unique(x_scaled, axis=0).shape[0]
        n_clusters = min(5, x_scaled.shape[0], unique_count)

        if n_clusters == 1:
            labels = np.zeros(x_scaled.shape[0], dtype=int)
        else:
            # Sort samples before seeded initialization so the result does not
            # depend on LAS row order. Canonical centroid ordering also keeps
            # cluster IDs stable if the solver returns permuted labels.
            sort_keys = tuple(
                x_scaled[:, column]
                for column in range(x_scaled.shape[1] - 1, -1, -1)
            )
            sample_order = np.lexsort(sort_keys)
            sorted_samples = x_scaled[sample_order]
            centroids = None
            sorted_labels = None
            for candidate_count in range(n_clusters, 1, -1):
                try:
                    centroids, sorted_labels = kmeans2(
                        sorted_samples,
                        candidate_count,
                        minit="points",
                        missing="raise",
                        seed=0,
                    )
                    n_clusters = candidate_count
                    break
                except ClusterError:
                    continue

            if centroids is None or sorted_labels is None:
                labels = np.zeros(x_scaled.shape[0], dtype=int)
                electrofacies = np.full(n_samples, np.nan, dtype=float)
                electrofacies[valid] = labels.astype(float)
                return electrofacies

            labels = np.empty_like(sorted_labels)
            labels[sample_order] = sorted_labels

            centroid_keys = tuple(
                centroids[:, column]
                for column in range(centroids.shape[1] - 1, -1, -1)
            )
            centroid_order = np.lexsort(centroid_keys)
            canonical_labels = np.empty(n_clusters, dtype=int)
            canonical_labels[centroid_order] = np.arange(n_clusters)
            labels = canonical_labels[labels]

        electrofacies = np.full(n_samples, np.nan, dtype=float)
        electrofacies[valid] = labels.astype(float)
        return electrofacies
