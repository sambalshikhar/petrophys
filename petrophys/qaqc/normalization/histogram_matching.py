"""
Histogram matching and equalization for log normalization.

This module provides histogram-based normalization methods
for matching log distributions between wells.
"""

from typing import Optional, Tuple
import numpy as np


def histogram_matching(
    source: np.ndarray,
    reference: np.ndarray,
    n_bins: int = 256
) -> np.ndarray:
    """
    Match histogram of source to reference distribution.

    This method transforms the source data so that its histogram
    matches that of the reference data.

    Parameters
    ----------
    source : np.ndarray
        Source data to transform
    reference : np.ndarray
        Reference data to match
    n_bins : int, optional
        Number of bins for histogram (default: 256)

    Returns
    -------
    np.ndarray
        Transformed source data

    Examples
    --------
    >>> # Normalize well B to match well A
    >>> gr_b_normalized = histogram_matching(gr_well_b, gr_well_a)
    """
    # Handle NaN values
    source_valid_mask = ~np.isnan(source)
    ref_valid_mask = ~np.isnan(reference)

    source_valid = source[source_valid_mask]
    ref_valid = reference[ref_valid_mask]

    if len(source_valid) == 0 or len(ref_valid) == 0:
        return source.copy()

    # Get value range
    all_values = np.concatenate([source_valid, ref_valid])
    value_min = np.min(all_values)
    value_max = np.max(all_values)
    if value_min == value_max:
        return source.copy()

    # Compute histograms
    source_hist, bins = np.histogram(
        source_valid, bins=n_bins, range=(value_min, value_max)
    )
    ref_hist, _ = np.histogram(
        ref_valid, bins=n_bins, range=(value_min, value_max)
    )

    # Compute CDFs
    source_cdf = np.cumsum(source_hist).astype(float)
    source_cdf /= source_cdf[-1]

    ref_cdf = np.cumsum(ref_hist).astype(float)
    ref_cdf /= ref_cdf[-1]

    # Build lookup table
    lookup = np.interp(source_cdf, ref_cdf, bins[:-1])

    # Apply transformation
    result = source.copy()
    bin_indices = np.digitize(source_valid, bins[:-1]) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    result[source_valid_mask] = lookup[bin_indices]

    return result


def histogram_equalization(
    data: np.ndarray,
    n_bins: int = 256,
    clip_limit: Optional[float] = None
) -> np.ndarray:
    """
    Apply histogram equalization to enhance contrast.

    Parameters
    ----------
    data : np.ndarray
        Input data
    n_bins : int, optional
        Number of bins (default: 256)
    clip_limit : float, optional
        Clip limit for contrast limiting (CLAHE-style)

    Returns
    -------
    np.ndarray
        Equalized data
    """
    # Handle NaN values
    valid_mask = ~np.isnan(data)
    valid_data = data[valid_mask]

    if len(valid_data) == 0:
        return data.copy()

    # Get range
    data_min = np.min(valid_data)
    data_max = np.max(valid_data)

    if data_min == data_max:
        return data.copy()

    # Compute histogram
    hist, bins = np.histogram(valid_data, bins=n_bins, range=(data_min, data_max))

    # Apply clip limit if specified (CLAHE)
    if clip_limit is not None:
        clip_value = clip_limit * len(valid_data) / n_bins
        excess = np.sum(np.maximum(hist - clip_value, 0))
        hist = np.minimum(hist, clip_value)
        hist += int(excess / n_bins)

    # Compute CDF
    cdf = np.cumsum(hist).astype(float)
    cdf_normalized = (cdf - cdf.min()) / (cdf.max() - cdf.min())

    # Map values
    result = data.copy()
    bin_indices = np.digitize(valid_data, bins[:-1]) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    # Scale back to original range
    equalized = data_min + cdf_normalized[bin_indices] * (data_max - data_min)
    result[valid_mask] = equalized

    return result


def compute_histogram_statistics(
    data: np.ndarray,
    n_bins: int = 100
) -> dict:
    """
    Compute histogram statistics for a log curve.

    Parameters
    ----------
    data : np.ndarray
        Log data
    n_bins : int, optional
        Number of bins

    Returns
    -------
    dict
        Statistics including histogram, percentiles, mode
    """
    valid_data = data[~np.isnan(data)]

    if len(valid_data) == 0:
        return {
            "histogram": np.array([]),
            "bins": np.array([]),
            "mode": np.nan,
            "percentiles": {},
        }

    hist, bins = np.histogram(valid_data, bins=n_bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Find mode (most frequent value)
    mode_idx = np.argmax(hist)
    mode = bin_centers[mode_idx]

    # Compute percentiles
    percentiles = {
        "p5": np.percentile(valid_data, 5),
        "p10": np.percentile(valid_data, 10),
        "p25": np.percentile(valid_data, 25),
        "p50": np.percentile(valid_data, 50),
        "p75": np.percentile(valid_data, 75),
        "p90": np.percentile(valid_data, 90),
        "p95": np.percentile(valid_data, 95),
    }

    return {
        "histogram": hist,
        "bins": bins,
        "bin_centers": bin_centers,
        "mode": mode,
        "percentiles": percentiles,
        "mean": np.mean(valid_data),
        "std": np.std(valid_data),
    }


def match_percentiles(
    source: np.ndarray,
    reference: np.ndarray,
    percentiles: Tuple[float, float] = (10, 90)
) -> np.ndarray:
    """
    Simple normalization by matching percentiles.

    Parameters
    ----------
    source : np.ndarray
        Source data
    reference : np.ndarray
        Reference data
    percentiles : tuple, optional
        Low and high percentiles to match (default: 10, 90)

    Returns
    -------
    np.ndarray
        Normalized source data
    """
    source_valid = source[~np.isnan(source)]
    ref_valid = reference[~np.isnan(reference)]

    if len(source_valid) == 0 or len(ref_valid) == 0:
        return source.copy()

    p_low, p_high = percentiles

    # Get percentile values
    src_low = np.percentile(source_valid, p_low)
    src_high = np.percentile(source_valid, p_high)
    ref_low = np.percentile(ref_valid, p_low)
    ref_high = np.percentile(ref_valid, p_high)

    # Avoid division by zero
    if src_high == src_low:
        return source.copy()

    # Apply linear transformation
    scale = (ref_high - ref_low) / (src_high - src_low)
    offset = ref_low - src_low * scale

    return source * scale + offset
