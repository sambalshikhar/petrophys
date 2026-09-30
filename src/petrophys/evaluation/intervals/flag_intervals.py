"""Turn a per-sample flag into a table of depth intervals."""

from __future__ import annotations

from typing import Optional

import numpy as np

from petrophys._internal.analysis_helpers import depth_step_array


def flag_intervals(
    flag,
    depth,
    *,
    max_gap: Optional[float] = None,
    min_thickness: float = 0.0,
):
    """List the depth intervals where a flag is set.

    Consecutive flagged samples form one interval. Each interval spans the
    full cells of its samples: from half a sample step above the first
    flagged sample to half a step below the last, so a single flagged sample
    has the thickness of one sample step and the thicknesses of contiguous
    intervals add up to the flagged thickness.

    Parameters
    ----------
    flag : array-like or Curve
        Per-sample flag: booleans, or numbers where values above 0.5 count
        as set (the convention of ``NET_PAY``-style flag curves). NaN is
        treated as not set.
    depth : array-like or Well
        Sample depths (or a well, whose depth is used), same length as
        ``flag``, in increasing or decreasing order.
    max_gap : float, optional
        Largest distance, in depth units, between two flagged sample depths
        that still joins them into one interval; unflagged samples inside a
        bridged gap are included in the interval. Defaults to 1.5 median
        sample steps, which joins only adjacent samples.
    min_thickness : float, default 0.0
        Drop intervals thinner than this (depth units).

    Returns
    -------
    pandas.DataFrame
        One row per interval, shallowest first, with ``top`` and ``bottom``
        (cell boundaries), ``thickness`` and ``samples`` (flagged samples in
        the interval).

    Raises
    ------
    ValueError
        If the lengths differ or ``max_gap`` / ``min_thickness`` are invalid.

    Examples
    --------
    >>> import numpy as np
    >>> depth = np.arange(2000.0, 2005.0, 0.5)
    >>> pay = np.array([0, 1, 1, 1, 0, 0, 1, 0, 1, 1])
    >>> table = flag_intervals(pay, depth)
    >>> table["top"].tolist(), table["bottom"].tolist()
    ([2000.25, 2002.75, 2003.75], [2001.75, 2003.25, 2004.75])
    >>> table["thickness"].tolist(), table["samples"].tolist()
    ([1.5, 0.5, 1.0], [3, 1, 2])
    >>> flag_intervals(pay, depth, max_gap=1.0)["samples"].tolist()
    [3, 3]
    """
    import pandas as pd

    if hasattr(depth, "depth") and not isinstance(depth, np.ndarray):
        depth = depth.depth
    depth = np.asarray(getattr(depth, "data", depth), dtype=float).reshape(-1)
    values = np.asarray(getattr(flag, "data", flag))
    if values.dtype == bool:
        flagged = values.reshape(-1).copy()
    else:
        numeric = np.asarray(values, dtype=float).reshape(-1)
        with np.errstate(invalid="ignore"):
            flagged = np.isfinite(numeric) & (numeric > 0.5)
    if flagged.size != depth.size:
        raise ValueError(f"flag has {flagged.size} samples but depth has {depth.size}")
    min_thickness = float(min_thickness)
    if not np.isfinite(min_thickness) or min_thickness < 0:
        raise ValueError("min_thickness must be a non-negative number")

    columns = ["top", "bottom", "thickness", "samples"]
    finite = np.isfinite(depth)
    flagged &= finite
    if not np.any(flagged):
        return pd.DataFrame(columns=columns).astype({"samples": int})

    order = np.argsort(depth[finite], kind="stable")
    sorted_depth = depth[finite][order]
    sorted_flag = flagged[finite][order]
    dz = depth_step_array(sorted_depth)
    if max_gap is None:
        positive = np.diff(sorted_depth)
        positive = positive[positive > 0]
        max_gap = 1.5 * float(np.median(positive)) if positive.size else 0.0
    max_gap = float(max_gap)
    if not np.isfinite(max_gap) or max_gap < 0:
        raise ValueError("max_gap must be a non-negative number")

    index = np.flatnonzero(sorted_flag)
    breaks = np.flatnonzero(np.diff(sorted_depth[index]) > max_gap)
    starts = np.r_[0, breaks + 1]
    stops = np.r_[breaks, index.size - 1]
    rows = []
    for start, stop in zip(starts, stops):
        first, last = index[start], index[stop]
        top = sorted_depth[first] - 0.5 * dz[first]
        bottom = sorted_depth[last] + 0.5 * dz[last]
        if bottom - top < min_thickness:
            continue
        rows.append({
            "top": float(top),
            "bottom": float(bottom),
            "thickness": float(bottom - top),
            "samples": int(stop - start + 1),
        })
    return pd.DataFrame(rows, columns=columns)


__all__ = ["flag_intervals"]
