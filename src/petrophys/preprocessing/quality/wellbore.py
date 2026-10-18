"""
Wellbore quality-control analysis aligned with the reference QC docs.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

from petrophys._internal.benchmark import reproducing_benchmark
from petrophys._internal.analysis_helpers import (
    CategoryAnalysisResult,
    DerivedCurve,
    build_section,
    get_curve_data_depthwise,
    rolling_std,
)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from petrophys.model.well import Well


class WellboreQC:
    """Bad-hole detection using caliper roughness, washout and DRHO fallback.

    ``cali_variation_threshold`` is the roughness limit in inches per metre
    (the rolling standard deviation of dCALI/dDepth, with the depth converted
    to metres whatever the well's depth unit); ``washout_threshold`` is the
    overgauge limit in inches, flagging CALI - BS above it; ``drho_threshold``
    is the |DRHO| limit in g/cc used where the caliper cannot be evaluated.
    """

    CATEGORY_KEY = "WELLBORE_QUALITY_CONTROL"

    def __init__(
        self,
        drho_threshold: float = 0.15,
        cali_variation_threshold: float = 0.5,
        washout_threshold: float = 1.5,
        bitsize_exclusion_distance: float = 50.0,
    ):
        self.drho_threshold = drho_threshold
        self.cali_variation_threshold = cali_variation_threshold
        self.washout_threshold = washout_threshold
        self.bitsize_exclusion_distance = bitsize_exclusion_distance

    def analyze(
        self,
        well: "Well",
        add_curves: bool = True,
    ) -> CategoryAnalysisResult:
        n_samples = well.n_samples
        depth = np.asarray(well.depth, dtype=float)
        # The benchmark packages took the roughness per depth unit as logged
        # and never applied the washout threshold.
        benchmark = reproducing_benchmark()
        depth_in_ft = str(getattr(well, "depth_unit", "m") or "m").lower().startswith("f")
        cali, _, _ = get_curve_data_depthwise(well, ["CALI", "CAL"])
        bs, _, _ = get_curve_data_depthwise(well, ["BS"])
        drho, _, _ = get_curve_data_depthwise(well, ["DRHO"])

        cali_smooth = np.full(n_samples, np.nan, dtype=float)
        cali_roughness = np.full(n_samples, np.nan, dtype=float)
        badholeflag = np.full(n_samples, np.nan, dtype=float)
        detection_method = np.full(n_samples, "", dtype=object)

        valid_cali = (
            cali is not None
            and bs is not None
            and np.count_nonzero(np.isfinite(cali) & np.isfinite(bs) & (cali > 0.0) & (bs > 0.0)) >= 5
        )

        if valid_cali:
            cali_valid = np.isfinite(cali) & np.isfinite(bs) & (cali > 0.0) & (bs > 0.0)
            indices = np.arange(n_samples)
            cali_interp = np.interp(indices, indices[cali_valid], cali[cali_valid])
            cali_smooth = savgol_filter(cali_interp, window_length=5, polyorder=2, mode="interp")
            gradient = np.gradient(cali_smooth, depth)
            if depth_in_ft and not benchmark:
                gradient = gradient / 0.3048  # in/ft -> in/m
            cali_roughness = rolling_std(gradient, window=5)
            roughness_valid = np.isfinite(cali_roughness) & cali_valid
            badholeflag[roughness_valid] = (
                cali_roughness[roughness_valid] > self.cali_variation_threshold
            ).astype(float)
            detection_method[roughness_valid] = "caliper_roughness"
            if not benchmark:
                washout = cali_valid & ((cali - bs) > self.washout_threshold)
                detection_method[washout & ~(badholeflag > 0.5)] = "caliper_washout"
                badholeflag[washout] = 1.0

        if drho is not None:
            drho_valid = np.isfinite(drho)
            fallback = (~np.isfinite(badholeflag)) & drho_valid
            badholeflag[fallback] = (np.abs(drho[fallback]) > self.drho_threshold).astype(float)
            detection_method[fallback] = "drho_fallback"

        labels = np.full(n_samples, "NO_QC_DATA", dtype=object)
        good = np.isfinite(badholeflag) & (badholeflag < 0.5)
        bad = np.isfinite(badholeflag) & (badholeflag > 0.5)
        labels[good] = "GOOD_HOLE"
        labels[bad] = "BAD_HOLE"

        exclusion_zones = self._bit_size_exclusion_zones(depth, bs)
        bad_intervals = self._intervals_from_flag(depth, bad)

        if benchmark:
            description = "Bad-hole screening from caliper roughness with DRHO fallback."
            methodology = (
                "Primary method smooths CALI with Savitzky-Golay window=5/polyorder=2, "
                "computes rolling standard deviation of dCALI/dDepth over window=5, "
                "and uses DRHO only where caliper roughness cannot be evaluated."
            )
            thresholds = (
                f"CALI roughness > {self.cali_variation_threshold}; "
                f"fallback |DRHO| > {self.drho_threshold}; needs at least 5 finite CALI/BS samples."
            )
            roughness_unit = ""
        else:
            description = "Bad-hole screening from caliper roughness and washout with DRHO fallback."
            methodology = (
                "Primary method smooths CALI with Savitzky-Golay window=5/polyorder=2, "
                "computes rolling standard deviation of dCALI/dDepth (depth in metres) over window=5, "
                "flags washout where CALI - BS exceeds the washout threshold, "
                "and uses DRHO only where the caliper cannot be evaluated."
            )
            thresholds = (
                f"CALI roughness > {self.cali_variation_threshold} in/m; "
                f"washout CALI - BS > {self.washout_threshold} in; "
                f"fallback |DRHO| > {self.drho_threshold} g/cc; needs at least 5 finite CALI/BS samples."
            )
            roughness_unit = "in/m"
        method_counts = {
            "caliper_roughness": int(np.count_nonzero(detection_method == "caliper_roughness")),
            "drho_fallback": int(np.count_nonzero(detection_method == "drho_fallback")),
        }
        if not benchmark:
            method_counts["caliper_washout"] = int(np.count_nonzero(detection_method == "caliper_washout"))

        result = CategoryAnalysisResult(
            category_key=self.CATEGORY_KEY,
            sections={
                "bad_hole_detection": build_section(
                    well,
                    labels,
                    description=description,
                    methodology=methodology,
                    calculation_thresholds=thresholds,
                )
            },
            derived_curves={
                "CALI_SMOOTH": DerivedCurve(cali_smooth, "in", "Smoothed caliper"),
                "CALI_ROUGHNESS": DerivedCurve(cali_roughness, roughness_unit, "Caliper roughness"),
                "BAD_HOLE": DerivedCurve(badholeflag, "", "Bad-hole flag"),
            },
            metadata={
                "qc_summary": {
                    # None when no sample could be assessed (no CALI/BS or DRHO).
                    "bad_hole_percentage": float(np.nanmean(np.where(np.isfinite(badholeflag), badholeflag, np.nan)) * 100.0)
                    if np.any(np.isfinite(badholeflag))
                    else None,
                    "bad_hole_intervals": bad_intervals,
                    "detection_method_counts": method_counts,
                    "bit_size_exclusion_zones": exclusion_zones,
                }
            },
        )

        if add_curves:
            result.add_curves_to_well(well)

        return result

    def _bit_size_exclusion_zones(
        self,
        depth: np.ndarray,
        bit_size: np.ndarray | None,
    ) -> list[dict[str, float]]:
        if bit_size is None or depth.size == 0:
            return []
        valid = np.isfinite(bit_size)
        if np.count_nonzero(valid) < 2:
            return []

        zones: list[dict[str, float]] = []
        changes = np.where(np.abs(np.diff(bit_size)) > 0.0)[0]
        for idx in changes:
            change_depth = float(depth[min(idx + 1, len(depth) - 1)])
            zones.append(
                {
                    "top": change_depth - self.bitsize_exclusion_distance,
                    "bottom": change_depth + self.bitsize_exclusion_distance,
                }
            )
        return zones

    @staticmethod
    def _intervals_from_flag(depth: np.ndarray, flag: np.ndarray) -> list[str]:
        intervals: list[str] = []
        if depth.size == 0 or not np.any(flag):
            return intervals

        padded = np.concatenate([[False], flag.astype(bool), [False]])
        changes = np.diff(padded.astype(int))
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]
        for start, end in zip(starts, ends):
            intervals.append(f"{depth[start]:.1f}-{depth[end - 1]:.1f}")
        return intervals
