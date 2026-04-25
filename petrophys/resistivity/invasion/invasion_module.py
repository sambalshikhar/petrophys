"""
Main invasion profile module.

This module provides the InvasionModule class for characterizing
invasion profiles from resistivity logs.
"""

from typing import Optional, Union
from dataclasses import dataclass
import numpy as np

from petrophys.resistivity.invasion.invasion_classification import (
    classify_invasion,
    InvasionType,
)


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
        # Get resistivity curves
        rxo = well.get(self.rxo_curve) or well.get("RXO") or well.get("RESS")
        rt = well.get(self.rt_curve) or well.get("RT")

        if rxo is None:
            raise ValueError(f"Shallow resistivity curve not found")
        if rt is None:
            raise ValueError(f"Deep resistivity curve not found")

        # Calculate ratio
        ratio = calculate_invasion_ratio(rxo.data, rt.data)
        rxo_data, rt_data = np.broadcast_arrays(
            np.asarray(rxo.data, dtype=float),
            np.asarray(rt.data, dtype=float),
        )
        rt_min = 0.2
        max_ratio = 100.0
        core_valid = (
            np.isfinite(rxo_data)
            & np.isfinite(rt_data)
            & (rxo_data > 0.0)
            & (rt_data >= rt_min)
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
            "rt_below_min_samples": int(np.count_nonzero(np.isfinite(rt_data) & (rt_data < rt_min))),
            "ratio_above_max_samples": int(np.count_nonzero(core_valid & (raw_ratio > max_ratio))),
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
    rt_min: float = 0.2,
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
    rt_min : float, default 0.2
        Minimum usable true resistivity. Lower or non-positive values are
        treated as invalid instead of creating artificial ratio spikes.
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
        & (rt_arr >= rt_min)
    )
    ratio[valid] = rxo_arr[valid] / rt_arr[valid]
    ratio[ratio > max_ratio] = np.nan

    if np.isscalar(rxo) and np.isscalar(rt):
        return float(ratio.reshape(-1)[0])
    return ratio
