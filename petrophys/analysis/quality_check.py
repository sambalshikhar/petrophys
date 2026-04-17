"""
Wellbore quality-control analysis aligned with the reference QC docs.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

from petrophys.utils.analysis_helpers import (
    CategoryAnalysisResult,
    DerivedCurve,
    build_section,
    rolling_std,
)


class WellboreQC:
    """Bad-hole detection using caliper roughness and DRHO fallback."""

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
        cali_curve = well.get("CALI") or well.get("CAL")
        bs_curve = well.get("BS")
        drho_curve = well.get("DRHO")

        cali = np.asarray(cali_curve.data, dtype=float) if cali_curve is not None else None
        bs = np.asarray(bs_curve.data, dtype=float) if bs_curve is not None else None
        drho = np.asarray(drho_curve.data, dtype=float) if drho_curve is not None else None

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
            cali_roughness = rolling_std(gradient, window=5)
            roughness_valid = np.isfinite(cali_roughness) & cali_valid
            badholeflag[roughness_valid] = (
                cali_roughness[roughness_valid] > self.cali_variation_threshold
            ).astype(float)
            detection_method[roughness_valid] = "caliper_roughness"

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

        result = CategoryAnalysisResult(
            category_key=self.CATEGORY_KEY,
            sections={
                "bad_hole_detection": build_section(
                    well,
                    labels,
                    description="Bad-hole screening from caliper roughness with DRHO fallback.",
                    methodology=(
                        "Primary method smooths caliper with a Savitzky-Golay filter, "
                        "computes gradient roughness, and flags rough intervals."
                    ),
                    calculation_thresholds=(
                        f"CALI roughness > {self.cali_variation_threshold}; "
                        f"fallback |DRHO| > {self.drho_threshold}."
                    ),
                )
            },
            derived_curves={
                "CALI_SMOOTH": DerivedCurve(cali_smooth, "in", "Smoothed caliper"),
                "CALI_ROUGHNESS": DerivedCurve(cali_roughness, "", "Caliper roughness"),
                "BADHOLEFLAG": DerivedCurve(badholeflag, "", "Bad-hole flag"),
            },
            metadata={
                "qc_summary": {
                    "bad_hole_percentage": float(np.nanmean(np.where(np.isfinite(badholeflag), badholeflag, np.nan)) * 100.0)
                    if np.any(np.isfinite(badholeflag))
                    else 0.0,
                    "bad_hole_intervals": bad_intervals,
                    "detection_method_counts": {
                        "caliper_roughness": int(np.count_nonzero(detection_method == "caliper_roughness")),
                        "drho_fallback": int(np.count_nonzero(detection_method == "drho_fallback")),
                    },
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
