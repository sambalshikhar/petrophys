"""
Main log normalization module.

This module provides the NormalizationModule class for normalizing
well logs to standard ranges or reference wells.
"""

from typing import Optional, Union, Tuple, List
from dataclasses import dataclass
import numpy as np


@dataclass
class NormalizationParams:
    """
    Parameters for log normalization.

    Attributes
    ----------
    log_min : float
        Target minimum value
    log_max : float
        Target maximum value
    log_low : float
        Current/source minimum value
    log_high : float
        Current/source maximum value
    """

    log_min: float
    log_max: float
    log_low: float
    log_high: float


class NormalizationModule:
    """
    Log normalization for multi-well consistency.

    This class provides methods for normalizing well logs to ensure
    consistency when comparing or combining data from multiple wells.

    The linear normalization equation is:
    Norm = logmin + (logmax - logmin) * (Log - loglow) / (loghigh - loglow)

    Parameters
    ----------
    method : str, optional
        Normalization method ("linear", "percentile", "z-score")
        Default: "linear"

    Examples
    --------
    >>> normalizer = NormalizationModule()
    >>> normalized_gr = normalizer.normalize(
    ...     gr_curve,
    ...     log_min=0, log_max=150,
    ...     log_low=20, log_high=120
    ... )
    """

    def __init__(self, method: str = "linear"):
        self.method = method
        self._reference_stats: dict = {}

    def normalize(
        self,
        data: Union[np.ndarray, "Curve"],
        log_min: float,
        log_max: float,
        log_low: Optional[float] = None,
        log_high: Optional[float] = None,
        percentile_low: float = 5,
        percentile_high: float = 95
    ) -> np.ndarray:
        """
        Normalize log data to target range.

        Parameters
        ----------
        data : np.ndarray or Curve
            Log data to normalize
        log_min : float
            Target minimum value
        log_max : float
            Target maximum value
        log_low : float, optional
            Source minimum (auto-detect if not provided)
        log_high : float, optional
            Source maximum (auto-detect if not provided)
        percentile_low : float, optional
            Percentile for auto-detecting low (default: 5)
        percentile_high : float, optional
            Percentile for auto-detecting high (default: 95)

        Returns
        -------
        np.ndarray
            Normalized log data
        """
        # Extract array from Curve if needed
        if hasattr(data, 'data'):
            values = data.data.copy()
        else:
            values = np.asarray(data, dtype=np.float64)

        # Auto-detect source range if not provided
        valid_data = values[~np.isnan(values)]
        if len(valid_data) == 0:
            return values

        if log_low is None:
            log_low = np.percentile(valid_data, percentile_low)
        if log_high is None:
            log_high = np.percentile(valid_data, percentile_high)

        # Apply normalization
        return linear_normalization(values, log_min, log_max, log_low, log_high)

    def normalize_curve(
        self,
        curve: "Curve",
        log_min: float,
        log_max: float,
        log_low: Optional[float] = None,
        log_high: Optional[float] = None
    ) -> "Curve":
        """
        Normalize a Curve object.

        Parameters
        ----------
        curve : Curve
            Curve to normalize
        log_min : float
            Target minimum
        log_max : float
            Target maximum
        log_low : float, optional
            Source minimum
        log_high : float, optional
            Source maximum

        Returns
        -------
        Curve
            Normalized curve
        """
        normalized_data = self.normalize(
            curve.data, log_min, log_max, log_low, log_high
        )

        new_curve = curve.copy()
        new_curve._data = normalized_data
        new_curve.mnemonic = f"{curve.mnemonic}_NORM"

        return new_curve

    def set_reference(
        self,
        mnemonic: str,
        log_min: float,
        log_max: float,
        log_low: float,
        log_high: float
    ):
        """
        Set reference parameters for a curve type.

        Parameters
        ----------
        mnemonic : str
            Curve mnemonic
        log_min, log_max : float
            Target range
        log_low, log_high : float
            Reference source range
        """
        self._reference_stats[mnemonic.upper()] = NormalizationParams(
            log_min=log_min,
            log_max=log_max,
            log_low=log_low,
            log_high=log_high
        )

    def normalize_to_reference(
        self,
        curve: "Curve"
    ) -> "Curve":
        """
        Normalize curve using stored reference parameters.

        Parameters
        ----------
        curve : Curve
            Curve to normalize

        Returns
        -------
        Curve
            Normalized curve

        Raises
        ------
        KeyError
            If no reference is set for this curve type
        """
        mnemonic = curve.mnemonic.upper()
        if mnemonic not in self._reference_stats:
            raise KeyError(f"No reference set for {mnemonic}")

        params = self._reference_stats[mnemonic]
        return self.normalize_curve(
            curve,
            params.log_min,
            params.log_max,
            params.log_low,
            params.log_high
        )

    def calculate_shift(
        self,
        data: np.ndarray,
        reference_data: np.ndarray
    ) -> Tuple[float, float]:
        """
        Calculate shift and scale to match reference.

        Parameters
        ----------
        data : np.ndarray
            Data to shift
        reference_data : np.ndarray
            Reference data to match

        Returns
        -------
        tuple
            (shift, scale) values
        """
        # Use percentiles for robustness
        data_valid = data[~np.isnan(data)]
        ref_valid = reference_data[~np.isnan(reference_data)]

        if len(data_valid) == 0 or len(ref_valid) == 0:
            return 0.0, 1.0

        data_p10 = np.percentile(data_valid, 10)
        data_p90 = np.percentile(data_valid, 90)
        ref_p10 = np.percentile(ref_valid, 10)
        ref_p90 = np.percentile(ref_valid, 90)

        # Calculate scale
        data_range = data_p90 - data_p10
        ref_range = ref_p90 - ref_p10

        if data_range == 0:
            scale = 1.0
        else:
            scale = ref_range / data_range

        # Calculate shift (after scaling)
        shift = ref_p10 - (data_p10 * scale)

        return shift, scale


def linear_normalization(
    data: np.ndarray,
    log_min: float,
    log_max: float,
    log_low: float,
    log_high: float
) -> np.ndarray:
    """
    Apply linear normalization to data.

    Norm = logmin + (logmax - logmin) * (Log - loglow) / (loghigh - loglow)

    Parameters
    ----------
    data : np.ndarray
        Input data
    log_min : float
        Target minimum
    log_max : float
        Target maximum
    log_low : float
        Source minimum
    log_high : float
        Source maximum

    Returns
    -------
    np.ndarray
        Normalized data
    """
    if log_high == log_low:
        return np.full_like(data, (log_min + log_max) / 2)

    normalized = log_min + (log_max - log_min) * (data - log_low) / (log_high - log_low)
    return normalized


def normalize_log(
    data: Union[np.ndarray, "Curve"],
    log_min: float,
    log_max: float,
    log_low: Optional[float] = None,
    log_high: Optional[float] = None
) -> np.ndarray:
    """
    Convenience function for log normalization.

    Parameters
    ----------
    data : np.ndarray or Curve
        Data to normalize
    log_min : float
        Target minimum
    log_max : float
        Target maximum
    log_low : float, optional
        Source minimum (auto-detect if None)
    log_high : float, optional
        Source maximum (auto-detect if None)

    Returns
    -------
    np.ndarray
        Normalized data

    Examples
    --------
    >>> normalized = normalize_log(gr_data, log_min=0, log_max=150)
    """
    normalizer = NormalizationModule()
    return normalizer.normalize(data, log_min, log_max, log_low, log_high)
