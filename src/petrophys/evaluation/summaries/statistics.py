"""
Curve statistics: the summary statistics petrophys reports for the samples of a
curve over an interval or a lithology group.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import numpy as np

from petrophys._internal.benchmark import reproducing_benchmark

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
    means, the third-power average (cube root of the mean cube), median, mode
    of the values rounded to 4 decimals and the population standard deviation;
    with the variance, the sample count and any requested percentiles.

    The geometric and harmonic means are over all finite samples, as defined
    for non-negative values: a zero sample makes both 0 (its logarithm and
    reciprocal are unbounded), and a negative sample leaves them undefined
    (NaN). So geometric mean <= arithmetic mean always holds. The benchmark
    reproduction takes both over the positive samples only (0 without any).

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

    if reproducing_benchmark():
        if positive.size:
            geometric_mean = float(np.exp(np.nanmean(np.log(positive))))
            harmonic_mean = float(positive.size / np.sum(1.0 / positive))
        else:
            geometric_mean = 0.0
            harmonic_mean = 0.0
    elif np.any(finite < 0):
        geometric_mean = harmonic_mean = float("nan")
    elif positive.size < finite.size:
        geometric_mean = harmonic_mean = 0.0
    else:
        geometric_mean = float(np.exp(np.mean(np.log(finite))))
        harmonic_mean = float(finite.size / np.sum(1.0 / finite))

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


def zone_statistics(
    well: Any,
    zones: Any,
    curves: Optional[Iterable[str]] = None,
    *,
    percentiles: Iterable[float] = (10, 50, 90),
    rank: Optional[str] = None,
    **zone_options: Any,
):
    """Statistics of each curve in each zone, as a long table.

    Parameters
    ----------
    well : Well
        Well with the curves.
    zones : DataFrame, path, list or mapping
        Zones as :func:`petrophys.read_formations` reads them: a CSV/Excel
        file or DataFrame of formation tops, a list of ``{"name", "top",
        "bottom"}`` dicts, or ``{name: top}``. Give bottoms for every zone
        or for none; without bottoms a zone ends at the next zone's top (the
        last at the well's bottom).
    curves : iterable of str, optional
        Curves to summarise; by default every numeric curve with data.
    percentiles : iterable of float
        Percentiles added to each row (``p10``, ``p50``, ``p90``).
    rank : str, optional
        Keep one level of a hierarchical tops table (``"formation"``).
    **zone_options
        Passed to :func:`petrophys.read_formations`, e.g. ``sheet_name``,
        ``columns={"formation_name": "Surface", "top_depth": "MD"}`` or
        ``well_name`` for a multi-well table.

    Returns
    -------
    pandas.DataFrame
        One row per zone and curve: ``zone``, ``top``, ``bottom``,
        ``thickness``, ``curve``, ``unit``, ``samples`` (in the zone),
        ``coverage`` (percent with a value) and the :func:`curve_statistics`
        of the finite samples. A sample at a boundary belongs to the zone
        below it.
    """
    import pandas as pd

    from petrophys.io.geology_reader import read_formations

    table = read_formations(zones, well=well, rank=rank, **zone_options)
    depth = np.asarray(well.depth, dtype=float)
    well_bottom = float(np.nanmax(depth))
    tops = table["top_depth"].astype(float).to_numpy()
    bottoms = table["bottom_depth"].astype(float).to_numpy() if "bottom_depth" in table else np.full(tops.size, np.nan)
    order = np.argsort(tops, kind="stable")
    names = list(curves) if curves is not None else [
        name for name in well.curves
        if np.issubdtype(np.asarray(well.get(name).data).dtype, np.number)
        and np.isfinite(np.asarray(well.get(name).data, dtype=float)).any()
    ]
    missing = [name for name in names if well.get(name) is None]
    if missing:
        raise ValueError(f"No curve {missing} on the well")
    rows = []
    for position, index in enumerate(order):
        top = tops[index]
        bottom = bottoms[index]
        if not np.isfinite(bottom):
            later = tops[order[position + 1:]]
            later = later[later > top]
            bottom = float(later.min()) if later.size else well_bottom
        last = bottom >= well_bottom
        in_zone = (depth >= top) & ((depth <= bottom) if last else (depth < bottom))
        count = int(in_zone.sum())
        for name in names:
            curve = well.get(name)
            values = np.asarray(curve.data, dtype=float)[in_zone]
            stats = curve_statistics(values, percentiles) if count else None
            row = {
                "zone": str(table["formation_name"].iloc[index]), "top": float(top), "bottom": float(bottom),
                "thickness": float(bottom - top), "curve": name, "unit": str(getattr(curve, "unit", "") or ""),
                "samples": count,
                "coverage": round(100.0 * (stats["count"] if stats else 0) / count, 2) if count else 0.0,
            }
            if stats:
                row.update(stats)
            rows.append(row)
    return pd.DataFrame(rows)


__all__ = ["STATISTICS", "curve_statistics", "percentile_key", "zone_statistics"]
