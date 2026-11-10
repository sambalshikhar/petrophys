"""Change-point zonation: split a log into intervals of statistically different level.

:func:`change_point_zonation` finds the boundaries where the mean of one or
more curves shifts, with the PELT algorithm (Killick et al., 2012): the
segmentation minimising the within-zone sum of squares plus a penalty per
boundary, solved exactly. Several curves are standardised by their
variability below the zone scale (the robust spread about a running median
over ``min_thickness``) and segmented together. ``min_thickness`` keeps
zones above the vertical resolution of the tools; the penalty sets how big
a shift must be to start a zone (by default a BIC-type penalty scaled by
the curve noise, so a larger ``penalty_factor`` gives fewer zones).

The zones are statistical: compare them with formation tops and core
before naming them.

References
----------
Killick, R., Fearnhead, P. and Eckley, I.A. (2012). Optimal detection of
changepoints with a linear computational cost. *Journal of the American
Statistical Association* 107(500), 1590-1598.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence
import warnings

import numpy as np


@dataclass
class Zonation:
    """Zones found by :func:`change_point_zonation`."""

    #: One dict per zone: ``top``, ``bottom``, ``thickness``, ``samples``
    #: and the mean of each curve.
    zones: List[Dict[str, float]]
    #: Depths of the boundaries between zones.
    boundaries: np.ndarray
    curves: List[str] = field(default_factory=list)
    penalty: float = float("nan")

    def to_dataframe(self):
        import pandas as pd

        return pd.DataFrame(self.zones)

    def labels(self, depth: Any) -> np.ndarray:
        """Zone number (1, 2, ...) of each depth; NaN outside the zoned interval."""
        d = np.asarray(depth, dtype=float)
        out = np.full(d.shape, np.nan)
        for number, zone in enumerate(self.zones, start=1):
            out[(d >= zone["top"]) & (d <= zone["bottom"])] = number
        return out


def _noise_scale(x: np.ndarray, window: int) -> float:
    """Variability of a series below the zone scale: robust spread of its
    residual from a running median over ``window`` samples.

    Logs vary bed by bed, not as white noise; measuring the spread inside
    the thinnest allowed zone puts that bed-scale variation in the noise, so
    only shifts larger than it start a zone.
    """
    from scipy.ndimage import median_filter

    size = max(3, int(window) | 1)
    residual = x - median_filter(x, size=size, mode="nearest")
    mad = 1.4826 * np.median(np.abs(residual - np.median(residual)))
    if mad > 0:
        return float(mad)
    spread = float(np.std(residual))
    return spread if spread > 0 else 1.0


def pelt(values: np.ndarray, penalty: float, min_size: int) -> List[int]:
    """Exact change points of a (samples x curves) array under a mean-shift model.

    Returns segment end indices (exclusive), the last being ``len(values)``.
    """
    x = np.asarray(values, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    n = x.shape[0]
    min_size = max(1, int(min_size))
    if n < 2 * min_size:
        return [n]
    s1 = np.vstack([np.zeros(x.shape[1]), np.cumsum(x, axis=0)])
    s2 = np.vstack([np.zeros(x.shape[1]), np.cumsum(x**2, axis=0)])

    def cost(starts: np.ndarray, end: int) -> np.ndarray:
        length = (end - starts)[:, None]
        sums = s1[end] - s1[starts]
        return np.sum(s2[end] - s2[starts] - sums**2 / length, axis=1)

    best = np.full(n + 1, np.inf)
    best[0] = -penalty
    last = np.zeros(n + 1, dtype=int)
    candidates = np.array([0])
    # Sample at which each candidate start stops being needed (pruning).
    expiry = np.array([n + 1])
    for end in range(min_size, n + 1):
        keep = expiry > end
        candidates, expiry = candidates[keep], expiry[keep]
        eligible = end - candidates >= min_size
        if not eligible.any():
            continue
        starts = candidates[eligible]
        fit = best[starts] + cost(starts, end)
        pick = int(np.argmin(fit))
        best[end] = fit[pick] + penalty
        last[end] = starts[pick]
        # Pruning: a start that cannot beat the best at ``end`` cannot beat a
        # split at ``end`` later, but that split needs a full zone after it,
        # so the start is retired only ``min_size`` samples on.
        dominated = np.zeros(candidates.size, dtype=bool)
        dominated[eligible] = fit > best[end]
        expiry = np.where(dominated & (expiry > end + min_size), end + min_size, expiry)
        # ``end`` can start the next zone once it is a valid zone end itself.
        candidates = np.append(candidates, end)
        expiry = np.append(expiry, n + 1)
    ends = []
    position = n
    while position > 0:
        ends.append(position)
        position = last[position]
    return sorted(ends)


def change_point_zonation(
    depth: Any,
    curves: Dict[str, Any],
    *,
    min_thickness: float,
    penalty: Optional[float] = None,
    penalty_factor: float = 1.0,
    log_curves: Sequence[str] = (),
    max_zones: Optional[int] = None,
) -> Zonation:
    """Zone a depth interval by shifts in the mean of one or more curves.

    Parameters
    ----------
    depth : array-like
        Sample depths, increasing.
    curves : dict
        Curve name to samples (same length as ``depth``), e.g.
        ``{"GR": gr, "RHOB": rhob}``.
    min_thickness : float
        Thinnest zone, in depth units (e.g. 2 m, above the tools' resolution).
    penalty : float, optional
        Cost of one boundary in standardised units; overrides the default
        ``penalty_factor * 2 * p * ln(n)`` (p curves, n samples).
    penalty_factor : float
        Scales the default penalty: above 1 fewer zones, below 1 more.
    log_curves : sequence of str
        Curves segmented as log10 (resistivity, permeability).
    max_zones : int, optional
        Raise the penalty (bisection) until there are at most this many
        zones; the strongest boundaries are kept.

    Returns
    -------
    Zonation
        Zones with top, bottom, thickness, sample count and curve means.
        Samples where any curve is missing are skipped (zones bridge them).
    """
    d = np.asarray(depth, dtype=float)
    names = list(curves)
    if not names:
        raise ValueError("Give at least one curve")
    columns = []
    for name in names:
        values = np.asarray(curves[name], dtype=float).reshape(-1)
        if values.shape != d.shape:
            raise ValueError(f"{name}: {values.size} samples, depth has {d.size}")
        if name in log_curves:
            with np.errstate(divide="ignore", invalid="ignore"):
                values = np.where(values > 0, np.log10(values), np.nan)
        columns.append(values)
    raw = np.column_stack(columns)
    use = np.isfinite(d) & np.all(np.isfinite(raw), axis=1)
    if use.sum() < 2:
        raise ValueError("No depth with all curves present")
    dd = d[use]
    if np.any(np.diff(dd) <= 0):
        raise ValueError("depth must increase")
    step = float(np.median(np.diff(dd))) if dd.size > 1 else 1.0
    min_size = max(2, int(round(min_thickness / step)))
    if int(round(min_thickness / step)) < 2:
        warnings.warn(
            f"min_thickness={min_thickness:g} is under two samples ({2 * step:g} at this "
            f"{step:g} step); zones are at least two samples thick.",
            UserWarning, stacklevel=2,
        )
    standardised = raw[use] / np.array([_noise_scale(raw[use][:, i], min_size) for i in range(raw.shape[1])])
    if penalty is None:
        penalty = penalty_factor * 2.0 * standardised.shape[1] * np.log(standardised.shape[0])
    ends = pelt(standardised, float(penalty), min_size)
    if max_zones is not None and len(ends) > max_zones:
        if max_zones < 1:
            raise ValueError("max_zones must be at least 1")
        low, high = float(penalty), float(penalty)
        while len(pelt(standardised, high, min_size)) > max_zones:
            low, high = high, high * 4.0
        for _ in range(30):
            middle = 0.5 * (low + high)
            if len(pelt(standardised, middle, min_size)) > max_zones:
                low = middle
            else:
                high = middle
            if high - low < 1e-3 * high:
                break
        penalty = high
        ends = pelt(standardised, penalty, min_size)
    zones = []
    start = 0
    for end in ends:
        segment = slice(start, end)
        top, bottom = float(dd[start]), float(dd[end - 1])
        zone = {"zone": len(zones) + 1, "top": top, "bottom": bottom,
                "thickness": bottom - top + step, "samples": end - start}
        for i, name in enumerate(names):
            values = raw[use][segment, i]
            zone[f"{name}_mean"] = float(10 ** np.mean(values)) if name in log_curves else float(np.mean(values))
        zones.append(zone)
        start = end
    boundaries = np.array([0.5 * (dd[end - 1] + dd[end]) for end in ends[:-1]])
    return Zonation(zones=zones, boundaries=boundaries, curves=names, penalty=float(penalty))


def zone_well(
    well: Any,
    curves: Sequence[str] = ("GR",),
    *,
    min_thickness: float,
    top: Optional[float] = None,
    bottom: Optional[float] = None,
    **options: Any,
) -> Zonation:
    """:func:`change_point_zonation` on curves of a well, optionally in a depth window."""
    depth = np.asarray(well.depth, dtype=float)
    window = np.ones(depth.shape, dtype=bool)
    if top is not None:
        window &= depth >= top
    if bottom is not None:
        window &= depth <= bottom
    data = {}
    for name in curves:
        curve = well.get(name)
        if curve is None:
            raise ValueError(f"No curve {name!r} on the well")
        data[name] = np.asarray(curve.data, dtype=float)[window]
    return change_point_zonation(depth[window], data, min_thickness=min_thickness, **options)


__all__ = ["Zonation", "change_point_zonation", "pelt", "zone_well"]
