"""
Interpolation utilities for log data.
"""

from typing import Optional, Union
import numpy as np
from scipy import interpolate as scipy_interp


def linear_interpolate(
    x: np.ndarray,
    y: np.ndarray,
    x_new: np.ndarray,
    fill_value: Union[float, str] = np.nan
) -> np.ndarray:
    """
    Linear interpolation of data.

    Parameters
    ----------
    x : np.ndarray
        Original x values (e.g., depth)
    y : np.ndarray
        Original y values (e.g., curve data)
    x_new : np.ndarray
        New x values to interpolate to
    fill_value : float or str
        Value for extrapolation ('extrapolate' or a float)

    Returns
    -------
    np.ndarray
        Interpolated values
    """
    # Remove NaN values
    mask = ~np.isnan(y)
    if np.sum(mask) < 2:
        return np.full_like(x_new, np.nan, dtype=float)

    f = scipy_interp.interp1d(
        x[mask], y[mask],
        kind='linear',
        bounds_error=False,
        fill_value=fill_value
    )
    return f(x_new)


def log_interpolate(
    x: np.ndarray,
    y: np.ndarray,
    x_new: np.ndarray
) -> np.ndarray:
    """
    Logarithmic interpolation (useful for resistivity).

    Parameters
    ----------
    x : np.ndarray
        Original x values
    y : np.ndarray
        Original y values (must be positive)
    x_new : np.ndarray
        New x values

    Returns
    -------
    np.ndarray
        Interpolated values
    """
    # Work in log space
    mask = ~np.isnan(y) & (y > 0)
    if np.sum(mask) < 2:
        return np.full_like(x_new, np.nan, dtype=float)

    log_y = np.log10(y[mask])
    f = scipy_interp.interp1d(
        x[mask], log_y,
        kind='linear',
        bounds_error=False,
        fill_value=np.nan
    )
    return np.power(10, f(x_new))


def resample_curve(
    depth: np.ndarray,
    data: np.ndarray,
    new_depth: np.ndarray,
    method: str = "linear"
) -> np.ndarray:
    """
    Resample curve data to new depth sampling.

    Parameters
    ----------
    depth : np.ndarray
        Original depth values
    data : np.ndarray
        Original curve data
    new_depth : np.ndarray
        Target depth values
    method : str
        Interpolation method ('linear', 'cubic', 'nearest')

    Returns
    -------
    np.ndarray
        Resampled data
    """
    mask = ~np.isnan(data)
    if np.sum(mask) < 2:
        return np.full_like(new_depth, np.nan, dtype=float)

    if method == "nearest":
        f = scipy_interp.interp1d(
            depth[mask], data[mask],
            kind='nearest',
            bounds_error=False,
            fill_value=np.nan
        )
    elif method == "cubic":
        if np.sum(mask) < 4:
            method = "linear"
            f = scipy_interp.interp1d(
                depth[mask], data[mask],
                kind='linear',
                bounds_error=False,
                fill_value=np.nan
            )
        else:
            f = scipy_interp.interp1d(
                depth[mask], data[mask],
                kind='cubic',
                bounds_error=False,
                fill_value=np.nan
            )
    else:  # linear
        f = scipy_interp.interp1d(
            depth[mask], data[mask],
            kind='linear',
            bounds_error=False,
            fill_value=np.nan
        )

    return f(new_depth)


def fill_gaps(
    data: np.ndarray,
    max_gap: Optional[int] = None,
    method: str = "linear"
) -> np.ndarray:
    """
    Fill NaN gaps in data using interpolation.

    Parameters
    ----------
    data : np.ndarray
        Data with NaN gaps
    max_gap : int, optional
        Maximum gap size to fill (in samples)
    method : str
        Fill method ('linear', 'nearest', 'forward', 'backward')

    Returns
    -------
    np.ndarray
        Data with gaps filled
    """
    result = data.copy()
    nans = np.isnan(result)

    if not np.any(nans):
        return result

    if np.all(nans):
        return result

    # Get indices
    indices = np.arange(len(result))
    valid_mask = ~nans

    if method == "forward":
        # Forward fill
        for i in range(len(result)):
            if nans[i]:
                prev_valid = np.where(valid_mask[:i])[0]
                if len(prev_valid) > 0:
                    result[i] = result[prev_valid[-1]]
    elif method == "backward":
        # Backward fill
        for i in range(len(result) - 1, -1, -1):
            if nans[i]:
                next_valid = np.where(valid_mask[i + 1:])[0]
                if len(next_valid) > 0:
                    result[i] = result[i + 1 + next_valid[0]]
    elif method == "nearest":
        f = scipy_interp.interp1d(
            indices[valid_mask], result[valid_mask],
            kind='nearest',
            bounds_error=False,
            fill_value=np.nan
        )
        result[nans] = f(indices[nans])
    else:  # linear
        f = scipy_interp.interp1d(
            indices[valid_mask], result[valid_mask],
            kind='linear',
            bounds_error=False,
            fill_value=np.nan
        )
        result[nans] = f(indices[nans])

    # Apply max_gap constraint if specified
    if max_gap is not None:
        # Find continuous NaN regions
        nan_regions = []
        in_region = False
        start = 0

        for i in range(len(nans)):
            if nans[i] and not in_region:
                start = i
                in_region = True
            elif not nans[i] and in_region:
                nan_regions.append((start, i))
                in_region = False

        if in_region:
            nan_regions.append((start, len(nans)))

        # Reset gaps larger than max_gap
        for start, end in nan_regions:
            if (end - start) > max_gap:
                result[start:end] = np.nan

    return result
