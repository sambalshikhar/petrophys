"""
Main invasion profile module.

This module provides the InvasionModule class for characterizing
invasion profiles from resistivity logs.
"""

from typing import Optional, Union
from dataclasses import dataclass
import numpy as np

from petrophys.interpretation.electrical.invasion.invasion_classification import (
    classify_invasion,
    InvasionType,
)
from petrophys._internal.analysis_helpers import get_curve, get_curve_data_depthwise


@dataclass
class InvasionResult:
    """
    Result container for invasion analysis.

    Attributes
    ----------
    ratio : np.ndarray
        Rxo/Rt ratio
    classification : np.ndarray
        Invasion classification at each depth
    statistics : dict
        Summary statistics
    """

    ratio: np.ndarray
    classification: np.ndarray
    statistics: dict


class InvasionModule:
    """
    Invasion profile characterization.

    Analyzes the relationship between shallow (Rxo) and deep (Rt)
    resistivity to characterize invasion profile.

    Parameters
    ----------
    rxo_curve : str, optional
        Name of shallow resistivity curve
    rt_curve : str, optional
        Name of deep resistivity curve

    Examples
    --------
    >>> invasion = InvasionModule()
    >>> result = invasion.analyze(well)
    >>> print(f"Mean ratio: {result.statistics['mean_ratio']:.2f}")
    """

    def __init__(
        self,
        rxo_curve: str = "RSHA",
        rt_curve: str = "RDEP"
    ):
        self.rxo_curve = rxo_curve
        self.rt_curve = rt_curve

    def analyze(self, well: "Well") -> InvasionResult:
        """
        Analyze invasion profile.

        Parameters
        ----------
        well : Well
            Well with resistivity curves

        Returns
        -------
        InvasionResult
            Invasion analysis results
        """
        # Get resistivity curves. Prefer true flushed/shallow curves, then use
        # medium/deep separation as a fallback proxy indicator where
        # flushed/shallow curves are absent or non-finite at a depth sample.
        shallow_candidates = list(dict.fromkeys([self.rxo_curve, "RXO", "RESS", "RSHA"]))
        medium_candidates = ["RMED", "RESM"]
        ratio_candidates = list(dict.fromkeys([*shallow_candidates, *medium_candidates]))
        rxo_data, rxo_name, rxo_meta = get_curve_data_depthwise(well, ratio_candidates)
        shallow_data, _, shallow_meta = get_curve_data_depthwise(well, shallow_candidates)
        medium_data, _, medium_meta = get_curve_data_depthwise(well, medium_candidates)
        rt_data, rt_name, rt_meta = get_curve_data_depthwise(well, [self.rt_curve, "RT", "RDEP"])
        rxo_existing = get_curve(well, ratio_candidates, require_finite=False)
        rt_existing = get_curve(well, [self.rt_curve, "RT", "RDEP"], require_finite=False)

        if rxo_data is None:
            if rxo_existing is not None:
                raise ValueError("Flushed/shallow resistivity or medium fallback curve present but has no finite samples")
            raise ValueError("Flushed/shallow resistivity or medium fallback curve not found")
        if rt_data is None:
            if rt_existing is not None:
                raise ValueError("Deep resistivity curve present but has no finite samples")
            raise ValueError("Deep resistivity curve not found")

        # Calculate ratio
        ratio = calculate_invasion_ratio(rxo_data, rt_data)
        rxo_data, rt_data = np.broadcast_arrays(
            np.asarray(rxo_data, dtype=float),
            np.asarray(rt_data, dtype=float),
        )
        rt_min = 0.0
        max_ratio = 100.0
        core_valid = (
            np.isfinite(rxo_data)
            & np.isfinite(rt_data)
            & (rxo_data > 0.0)
            & (rt_data > rt_min)
        )
        raw_ratio = np.full(rxo_data.shape, np.nan, dtype=float)
        raw_ratio[core_valid] = rxo_data[core_valid] / rt_data[core_valid]

        # Classify
        classification = classify_invasion(ratio)

        # Calculate statistics
        finite_ratio = np.isfinite(ratio)
        valid_ratio = ratio[finite_ratio]
        valid_count = int(np.count_nonzero(finite_ratio))
        sample_count = max(len(ratio), 1)
        percent_count = max(valid_count, 1)
        stats = {
            "mean_ratio": float(np.mean(valid_ratio)) if len(valid_ratio) > 0 else np.nan,
            "median_ratio": float(np.median(valid_ratio)) if len(valid_ratio) > 0 else np.nan,
            "min_ratio": float(np.min(valid_ratio)) if len(valid_ratio) > 0 else np.nan,
            "max_ratio": float(np.max(valid_ratio)) if len(valid_ratio) > 0 else np.nan,
            "valid_percent": float(valid_count / sample_count * 100),
            "conductive_percent": float(np.sum(finite_ratio & (ratio < 0.8)) / percent_count * 100),
            "minimal_percent": float(np.sum(finite_ratio & (ratio >= 0.8) & (ratio <= 1.2)) / percent_count * 100),
            "resistive_percent": float(np.sum(finite_ratio & (ratio > 1.2)) / percent_count * 100),
            "total_samples": int(ratio.size),
            "valid_samples": valid_count,
            "nan_samples": int(ratio.size - valid_count),
            "rt_min_used": rt_min,
            "max_ratio_used": max_ratio,
            "nonfinite_rxo_samples": int(np.count_nonzero(~np.isfinite(rxo_data))),
            "nonfinite_rt_samples": int(np.count_nonzero(~np.isfinite(rt_data))),
            "nonpositive_rxo_samples": int(np.count_nonzero(np.isfinite(rxo_data) & (rxo_data <= 0.0))),
            "rt_below_min_samples": int(np.count_nonzero(np.isfinite(rt_data) & (rt_data <= rt_min))),
            "ratio_above_max_samples": int(np.count_nonzero(core_valid & (raw_ratio > max_ratio))),
            "rxo_curve": rxo_name,
            "rt_curve": rt_name,
            "rxo_source_counts": rxo_meta.get("source_counts", {}),
            "rt_source_counts": rt_meta.get("source_counts", {}),
            "shallow_source_counts": shallow_meta.get("source_counts", {}) if shallow_meta else {},
            "medium_proxy_source_counts": medium_meta.get("source_counts", {}) if medium_meta else {},
            "medium_deep_proxy_samples": int(
                sum((rxo_meta.get("source_counts", {}) or {}).get(name, 0) for name in medium_candidates)
            ),
            "true_shallow_samples": int(
                sum((rxo_meta.get("source_counts", {}) or {}).get(name, 0) for name in shallow_candidates)
            ),
            "fallback_proxy_used": any(
                ((rxo_meta.get("source_counts", {}) or {}).get(name, 0) > 0)
                for name in medium_candidates
            ),
        }

        return InvasionResult(
            ratio=ratio,
            classification=classification,
            statistics=stats
        )

    def add_to_well(
        self,
        well: "Well",
        result: Optional[InvasionResult] = None
    ) -> "Well":
        """
        Add invasion ratio curve to well.

        Parameters
        ----------
        well : Well
            Well object
        result : InvasionResult, optional
            Pre-computed result

        Returns
        -------
        Well
            Well with invasion ratio curve
        """
        if result is None:
            result = self.analyze(well)

        well.add_curve(
            "INV_RATIO",
            result.ratio,
            unit="",
            description="Invasion Ratio (Rxo/Rt)"
        )

        well.set_result("invasion_analysis", result.statistics)

        return well


def calculate_invasion_ratio(
    rxo: Union[float, np.ndarray],
    rt: Union[float, np.ndarray],
    rt_min: float = 0.0,
    max_ratio: float = 100.0,
) -> Union[float, np.ndarray]:
    """
    Calculate invasion ratio (Rxo/Rt).

    Parameters
    ----------
    rxo : float or np.ndarray
        Flushed zone resistivity
    rt : float or np.ndarray
        True formation resistivity
    rt_min : float, default 0.0
        Minimum usable true resistivity. Non-positive values are treated as
        invalid; low positive brine-formation resistivity is retained.
    max_ratio : float, default 100.0
        Maximum plausible ratio retained in the output.

    Returns
    -------
    float or np.ndarray
        Invasion ratio

    Examples
    --------
    >>> ratio = calculate_invasion_ratio(rxo_data, rt_data)
    """
    rxo_arr, rt_arr = np.broadcast_arrays(
        np.asarray(rxo, dtype=float),
        np.asarray(rt, dtype=float),
    )
    ratio = np.full(rxo_arr.shape, np.nan, dtype=float)
    valid = (
        np.isfinite(rxo_arr)
        & np.isfinite(rt_arr)
        & (rxo_arr > 0)
        & (rt_arr > rt_min)
    )
    ratio[valid] = rxo_arr[valid] / rt_arr[valid]
    ratio[ratio > max_ratio] = np.nan

    if np.isscalar(rxo) and np.isscalar(rt):
        return float(ratio.reshape(-1)[0])
    return ratio
