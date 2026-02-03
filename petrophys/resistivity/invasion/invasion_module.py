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
        rxo = well.get(self.rxo_curve) or well.get("RXO")
        rt = well.get(self.rt_curve) or well.get("RT")

        if rxo is None:
            raise ValueError(f"Shallow resistivity curve not found")
        if rt is None:
            raise ValueError(f"Deep resistivity curve not found")

        # Calculate ratio
        ratio = calculate_invasion_ratio(rxo.data, rt.data)

        # Classify
        classification = classify_invasion(ratio)

        # Calculate statistics
        valid_ratio = ratio[~np.isnan(ratio)]
        stats = {
            "mean_ratio": float(np.mean(valid_ratio)) if len(valid_ratio) > 0 else np.nan,
            "median_ratio": float(np.median(valid_ratio)) if len(valid_ratio) > 0 else np.nan,
            "min_ratio": float(np.min(valid_ratio)) if len(valid_ratio) > 0 else np.nan,
            "max_ratio": float(np.max(valid_ratio)) if len(valid_ratio) > 0 else np.nan,
            "conductive_percent": float(np.sum(ratio < 0.8) / len(ratio) * 100),
            "minimal_percent": float(np.sum((ratio >= 0.8) & (ratio <= 1.2)) / len(ratio) * 100),
            "resistive_percent": float(np.sum(ratio > 1.2) / len(ratio) * 100),
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
    rt: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Calculate invasion ratio (Rxo/Rt).

    Parameters
    ----------
    rxo : float or np.ndarray
        Flushed zone resistivity
    rt : float or np.ndarray
        True formation resistivity

    Returns
    -------
    float or np.ndarray
        Invasion ratio

    Examples
    --------
    >>> ratio = calculate_invasion_ratio(rxo_data, rt_data)
    """
    # Avoid division by zero
    if isinstance(rt, np.ndarray):
        rt = np.where(rt == 0, 1e-10, rt)
    else:
        rt = max(rt, 1e-10)

    return rxo / rt
