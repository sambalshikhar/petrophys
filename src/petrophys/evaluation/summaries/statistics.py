"""
Curve statistics: the summary statistics petrophys reports for the samples of a
curve over an interval or a lithology group.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import numpy as np

# The statistics the reference context reports per curve, in its order.
STATISTICS = (
    "minimum",
    "maximum",
    "arithmetic_mean",
    "geometric_mean",
    "harmonic_mean",
    "third_power_average",
    "median",
    "mode",
    "standard_deviation",
)


def _as_float_array(values: Any) -> np.ndarray:
    arr = np.asarray(getattr(values, "data", values))
    if np.issubdtype(arr.dtype, np.number):
        return arr.astype(float).reshape(-1)
    output = np.full(arr.size, np.nan, dtype=float)
    for idx, value in enumerate(arr.reshape(-1)):
        try:
            output[idx] = float(value)
        except (TypeError, ValueError):
            output[idx] = np.nan
    return output


def percentile_key(p: float) -> str:
    """The key of a percentile in :func:`curve_statistics`: ``10`` -> ``p10``, ``2.5`` -> ``p2.5``."""
    return f"p{float(p):g}"


def curve_statistics(
    values: Any,
    percentiles: Iterable[float] = (),
) -> Optional[Dict[str, float]]:
    """Summary statistics of the finite samples of a curve.

    The statistics the reference context reports for a curve over a lithology
    group, unrounded: minimum, maximum, arithmetic mean, geometric and harmonic
    means over the positive samples (0 when there are none), the third-power
    average (cube root of the mean cube), median, mode of the values rounded to
    4 decimals and the population standard deviation; with the variance, the
    sample count and any requested percentiles.

    Parameters
    ----------
    values : array-like or Curve
        Curve samples. Non-numeric and non-finite values are ignored.
    percentiles : iterable of float, optional
        Percentiles (0-100) to add, each under :func:`percentile_key`
        (``p10``, ``p2.5``).

    Returns
    -------
    dict or None
        ``count``, the :data:`STATISTICS`, ``variance`` and the requested
        percentiles; None when there is no finite sample.

    Examples
    --------
    >>> stats = curve_statistics([0.1, 0.2, float("nan"), 0.3], percentiles=[50])
    >>> stats["count"], round(stats["arithmetic_mean"], 6), round(stats["p50"], 6)
    (3, 0.2, 0.2)
    """
    arr = _as_float_array(values)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None

    positive = finite[finite > 0]
    rounded = np.round(finite, 4)
    unique, counts = np.unique(rounded, return_counts=True)
    mode_value = float(unique[int(np.argmax(counts))]) if unique.size else float(finite[0])

    if positive.size:
        geometric_mean = float(np.exp(np.nanmean(np.log(positive))))
        harmonic_mean = float(positive.size / np.sum(1.0 / positive))
    else:
        geometric_mean = 0.0
        harmonic_mean = 0.0

    stats: Dict[str, float] = {
        "count": int(finite.size),
        "minimum": float(np.nanmin(finite)),
        "maximum": float(np.nanmax(finite)),
        "arithmetic_mean": float(np.nanmean(finite)),
        "geometric_mean": geometric_mean,
        "harmonic_mean": harmonic_mean,
        "third_power_average": float(np.cbrt(np.nanmean(np.power(finite, 3)))),
        "median": float(np.nanmedian(finite)),
        "mode": mode_value,
        "standard_deviation": float(np.nanstd(finite)),
        "variance": float(np.nanvar(finite)),
    }
    for p in percentiles:
        if not 0.0 <= float(p) <= 100.0:
            raise ValueError(f"percentile {p} is outside 0-100")
        stats[percentile_key(p)] = float(np.percentile(finite, float(p)))
    return stats


__all__ = ["STATISTICS", "curve_statistics", "percentile_key"]
