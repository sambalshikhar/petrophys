"""
Statistical utilities for petrophysics calculations.
"""

from typing import Optional, Dict, Union, Tuple
import numpy as np


def finite_min_max(values: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    """Return finite min/max values, or ``(None, None)`` when none exist."""
    arr = np.asarray(values, dtype=float).reshape(-1)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None, None
    return float(np.nanmin(finite)), float(np.nanmax(finite))


def weighted_average(
    values: np.ndarray,
    weights: Optional[np.ndarray] = None,
    ignore_nan: bool = True
) -> float:
    """
    Calculate weighted average.

    Parameters
    ----------
    values : np.ndarray
        Data values
    weights : np.ndarray, optional
        Weights for each value (default: equal weights)
    ignore_nan : bool
        Whether to ignore NaN values

    Returns
    -------
    float
        Weighted average
    """
    values = np.asarray(values, dtype=float)

    if weights is None:
        weights = np.ones_like(values)
    else:
        weights = np.asarray(weights, dtype=float)

    if ignore_nan:
        mask = ~np.isnan(values) & ~np.isnan(weights)
        values = values[mask]
        weights = weights[mask]

    if len(values) == 0 or np.sum(weights) == 0:
        return np.nan

    return float(np.sum(values * weights) / np.sum(weights))


def harmonic_mean(
    values: np.ndarray,
    weights: Optional[np.ndarray] = None,
    ignore_nan: bool = True
) -> float:
    """
    Calculate harmonic mean (useful for permeability averaging).

    Parameters
    ----------
    values : np.ndarray
        Data values (must be positive)
    weights : np.ndarray, optional
        Weights for each value
    ignore_nan : bool
        Whether to ignore NaN values

    Returns
    -------
    float
        Harmonic mean
    """
    values = np.asarray(values, dtype=float)

    if weights is None:
        weights = np.ones_like(values)
    else:
        weights = np.asarray(weights, dtype=float)

    if ignore_nan:
        mask = ~np.isnan(values) & ~np.isnan(weights) & (values > 0)
        values = values[mask]
        weights = weights[mask]
    else:
        mask = values > 0
        values = values[mask]
        weights = weights[mask]

    if len(values) == 0 or np.sum(weights) == 0:
        return np.nan

    return float(np.sum(weights) / np.sum(weights / values))


def geometric_mean(
    values: np.ndarray,
    weights: Optional[np.ndarray] = None,
    ignore_nan: bool = True
) -> float:
    """
    Calculate geometric mean.

    Parameters
    ----------
    values : np.ndarray
        Data values (must be positive)
    weights : np.ndarray, optional
        Weights for each value
    ignore_nan : bool
        Whether to ignore NaN values

    Returns
    -------
    float
        Geometric mean
    """
    values = np.asarray(values, dtype=float)

    if weights is None:
        weights = np.ones_like(values)
    else:
        weights = np.asarray(weights, dtype=float)

    if ignore_nan:
        mask = ~np.isnan(values) & ~np.isnan(weights) & (values > 0)
        values = values[mask]
        weights = weights[mask]
    else:
        mask = values > 0
        values = values[mask]
        weights = weights[mask]

    if len(values) == 0 or np.sum(weights) == 0:
        return np.nan

    log_values = np.log(values)
    weighted_log_mean = np.sum(weights * log_values) / np.sum(weights)
    return float(np.exp(weighted_log_mean))


def trim_mean(
    values: np.ndarray,
    trim_percent: float = 0.1,
    ignore_nan: bool = True
) -> float:
    """
    Calculate trimmed mean (removes outliers).

    Parameters
    ----------
    values : np.ndarray
        Data values
    trim_percent : float
        Fraction to trim from each end (0-0.5)
    ignore_nan : bool
        Whether to ignore NaN values

    Returns
    -------
    float
        Trimmed mean
    """
    values = np.asarray(values, dtype=float)

    if ignore_nan:
        values = values[~np.isnan(values)]

    if len(values) == 0:
        return np.nan

    n = len(values)
    trim_count = int(n * trim_percent)

    if trim_count > 0:
        sorted_values = np.sort(values)
        trimmed = sorted_values[trim_count:-trim_count]
    else:
        trimmed = values

    if len(trimmed) == 0:
        return np.nan

    return float(np.mean(trimmed))


def percentile_stats(
    values: np.ndarray,
    percentiles: Optional[list] = None,
    ignore_nan: bool = True
) -> Dict[str, float]:
    """
    Calculate percentile statistics.

    Parameters
    ----------
    values : np.ndarray
        Data values
    percentiles : list, optional
        Percentiles to calculate (default: [10, 25, 50, 75, 90])
    ignore_nan : bool
        Whether to ignore NaN values

    Returns
    -------
    dict
        Dictionary with statistics
    """
    values = np.asarray(values, dtype=float)

    if ignore_nan:
        values = values[~np.isnan(values)]

    if len(values) == 0:
        return {
            "count": 0,
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
        }

    if percentiles is None:
        percentiles = [10, 25, 50, 75, 90]

    stats = {
        "count": len(values),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }

    for p in percentiles:
        stats[f"p{p}"] = float(np.percentile(values, p))

    return stats


def calculate_rmse(
    observed: np.ndarray,
    predicted: np.ndarray,
    ignore_nan: bool = True
) -> float:
    """
    Calculate Root Mean Square Error.

    Parameters
    ----------
    observed : np.ndarray
        Observed values
    predicted : np.ndarray
        Predicted values
    ignore_nan : bool
        Whether to ignore NaN values

    Returns
    -------
    float
        RMSE value
    """
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    if ignore_nan:
        mask = ~np.isnan(observed) & ~np.isnan(predicted)
        observed = observed[mask]
        predicted = predicted[mask]

    if len(observed) == 0:
        return np.nan

    return float(np.sqrt(np.mean((observed - predicted) ** 2)))


def calculate_r_squared(
    observed: np.ndarray,
    predicted: np.ndarray,
    ignore_nan: bool = True
) -> float:
    """
    Calculate coefficient of determination (R²).

    Parameters
    ----------
    observed : np.ndarray
        Observed values
    predicted : np.ndarray
        Predicted values
    ignore_nan : bool
        Whether to ignore NaN values

    Returns
    -------
    float
        R² value
    """
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    if ignore_nan:
        mask = ~np.isnan(observed) & ~np.isnan(predicted)
        observed = observed[mask]
        predicted = predicted[mask]

    if len(observed) == 0:
        return np.nan

    ss_res = np.sum((observed - predicted) ** 2)
    ss_tot = np.sum((observed - np.mean(observed)) ** 2)

    if ss_tot == 0:
        return np.nan

    return float(1 - ss_res / ss_tot)
