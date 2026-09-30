"""Detection of held (stuck) tool readings.

A real measurement never repeats exactly, to every stored decimal, for many
samples in a row. Long runs of identical values appear where a file fills
depths a sensor never reached: the last metres of hole below a sensor that
sits high on the tool string, the top of a run before the tool was logging,
or a tool that stuck. The fill is usually the last good reading held
constant, and sometimes zeros.

:func:`held_reading_mask` flags such samples in one array, and
:func:`find_held_readings` checks several curves of a well and reports where
the runs are. The flags are QC evidence to inspect, not an automatic
deletion: quantized curves (caliper in gauge hole, flags, codes) and logs
resampled from a coarser sampling legitimately repeat values, which is why
the default run length is long.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional

import numpy as np


def _as_values(values) -> np.ndarray:
    data = getattr(values, "data", values)
    return np.asarray(data, dtype=float).reshape(-1)


def _runs(values: np.ndarray, min_run: int, tolerance: float):
    """Yield (start, stop) index pairs of runs of at least ``min_run`` equal samples."""
    finite = np.isfinite(values)
    same = np.zeros(values.size, dtype=bool)
    if values.size > 1:
        with np.errstate(invalid="ignore"):
            same[1:] = (
                finite[1:] & finite[:-1] & (np.abs(values[1:] - values[:-1]) <= tolerance)
            )
    # A run is one sample that starts it plus consecutive samples equal to
    # the one above.
    index = 0
    size = values.size
    while index < size:
        if index + 1 < size and same[index + 1] and finite[index]:
            stop = index + 1
            while stop < size and same[stop]:
                stop += 1
            if stop - index >= min_run:
                yield index, stop
            index = stop
        else:
            index += 1


def _validate(min_run, tolerance):
    if int(min_run) != min_run or int(min_run) < 2:
        raise ValueError("min_run must be an integer of at least 2")
    tolerance = float(tolerance)
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be a non-negative number")
    return int(min_run), tolerance


def held_reading_mask(values, *, min_run: int = 10, tolerance: float = 0.0) -> np.ndarray:
    """Flag samples that repeat the reading above them in a long run.

    A run is a sequence of consecutive finite samples whose values agree
    within ``tolerance``. In every run of at least ``min_run`` samples, all
    samples after the first are flagged. The first sample is the reading
    being repeated and is kept (at the bottom of a logging run it is the
    deepest real measurement). Missing values break a run.

    Parameters
    ----------
    values : array-like or Curve
        One log curve, in any unit.
    min_run : int, default 10
        Minimum run length, in samples, counting the first sample. Short
        repeats are common in quantized or resampled curves; lower this only
        after looking at the curve.
    tolerance : float, default 0.0
        Largest absolute difference, in the curve's unit, still treated as
        a repeat. The default requires an exact repeat.

    Returns
    -------
    numpy.ndarray of bool
        True where a sample repeats the one above inside a long run.

    Raises
    ------
    ValueError
        If ``min_run`` is not an integer of at least 2 or ``tolerance`` is
        negative.

    Examples
    --------
    >>> import numpy as np
    >>> gr = np.array([61.2, 63.8, 80.19, 80.19, 80.19, 80.19, 0.0, 0.0])
    >>> held_reading_mask(gr, min_run=3).astype(int).tolist()
    [0, 0, 0, 1, 1, 1, 0, 0]
    """
    min_run, tolerance = _validate(min_run, tolerance)
    data = _as_values(values)
    mask = np.zeros(data.size, dtype=bool)
    for start, stop in _runs(data, min_run, tolerance):
        mask[start + 1:stop] = True
    return mask


@dataclass(repr=False)
class HeldReadings:
    """Held-reading flags for several curves of one well.

    Attributes
    ----------
    depth : numpy.ndarray
        Depth of every sample checked (the whole well, or the requested
        window).
    masks : dict of str to numpy.ndarray
        Per-curve boolean flags from :func:`held_reading_mask`.
    runs : pandas.DataFrame
        One row per flagged run: ``curve``, ``top`` and ``bottom`` (depths of
        the first and last flagged sample), ``samples`` (flagged samples) and
        ``value`` (the held reading).
    min_run : int
        Run length used.
    tolerance : float
        Repeat tolerance used.
    depth_unit : str
        Depth unit of the well.
    """

    depth: np.ndarray
    masks: Dict[str, np.ndarray]
    runs: "object"
    min_run: int
    tolerance: float = 0.0
    depth_unit: str = ""
    curves: list = field(default_factory=list)

    def any(self) -> np.ndarray:
        """Return True where at least one checked curve is flagged."""
        combined = np.zeros(self.depth.size, dtype=bool)
        for mask in self.masks.values():
            combined |= mask
        return combined

    @property
    def shallowest_depth(self) -> Optional[float]:
        """Shallowest flagged depth over all checked curves, or None."""
        flagged = self.any()
        if not np.any(flagged):
            return None
        return float(np.nanmin(self.depth[flagged]))

    def summary(self):
        """Return one row per checked curve.

        Columns: ``held_samples``, ``runs``, ``top`` and ``bottom`` (shallowest
        and deepest flagged depth, NaN when none) and ``longest_run`` (flagged
        samples in the longest run).
        """
        import pandas as pd

        rows = []
        for name in self.curves:
            mask = self.masks[name]
            curve_runs = self.runs[self.runs["curve"] == name]
            flagged_depth = self.depth[mask]
            rows.append({
                "curve": name,
                "held_samples": int(np.count_nonzero(mask)),
                "runs": int(len(curve_runs)),
                "top": float(np.nanmin(flagged_depth)) if flagged_depth.size else np.nan,
                "bottom": float(np.nanmax(flagged_depth)) if flagged_depth.size else np.nan,
                "longest_run": int(curve_runs["samples"].max()) if len(curve_runs) else 0,
            })
        return pd.DataFrame(rows).set_index("curve")

    def __repr__(self) -> str:
        unit = f" {self.depth_unit}" if self.depth_unit else ""
        parts = [f"{name}: {int(np.count_nonzero(self.masks[name]))}" for name in self.curves]
        where = (
            f"; shallowest at {self.shallowest_depth:.2f}{unit}"
            if self.shallowest_depth is not None else ""
        )
        return (
            f"HeldReadings(min_run={self.min_run}, held samples "
            f"{{{', '.join(parts)}}}{where})"
        )


def find_held_readings(
    well,
    curves: Optional[Iterable[str]] = None,
    *,
    min_run: int = 10,
    tolerance: float = 0.0,
    top: Optional[float] = None,
    bottom: Optional[float] = None,
) -> HeldReadings:
    """Find held (stuck or padded) readings in the curves of a well.

    Parameters
    ----------
    well : Well
        Well to check.
    curves : iterable of str, optional
        Curves to check. Defaults to every curve on the well.
    min_run : int, default 10
        Minimum run length in samples (see :func:`held_reading_mask`).
    tolerance : float, default 0.0
        Repeat tolerance in each curve's unit.
    top, bottom : float, optional
        Depth window to check, in well depth units (inclusive). Defaults to
        the whole well.

    Returns
    -------
    HeldReadings
        Per-curve flags, the list of runs and a per-curve summary. Use
        ``result.shallowest_depth`` to end an analysis window above the
        first held sample.

    Raises
    ------
    ValueError
        If a requested curve is missing or the settings are invalid.

    Examples
    --------
    >>> import petrophys as pp
    >>> well = pp.read(pp.sample("25_5-1"))
    >>> held = find_held_readings(well, ["GR", "THOR", "URAN"])
    >>> held.summary()["held_samples"].to_dict()
    {'GR': 57, 'THOR': 45, 'URAN': 45}
    >>> round(held.shallowest_depth, 1)
    3423.4
    """
    import pandas as pd

    min_run, tolerance = _validate(min_run, tolerance)
    if well.depth is None:
        raise ValueError("Well has no depth data")
    depth = np.asarray(well.depth, dtype=float).reshape(-1)
    names = list(well.curves) if curves is None else [str(name) for name in curves]
    missing = [name for name in names if well.get(name) is None]
    if missing:
        raise ValueError(f"Curve(s) not found on well: {', '.join(missing)}")

    window = np.ones(depth.size, dtype=bool)
    if top is not None:
        window &= depth >= float(top)
    if bottom is not None:
        window &= depth <= float(bottom)
    depth = depth[window]

    masks: Dict[str, np.ndarray] = {}
    rows = []
    for name in names:
        values = _as_values(well.get(name))
        if values.size != window.size:
            raise ValueError(
                f"Curve '{name}' length ({values.size}) does not match well depth "
                f"length ({window.size})"
            )
        values = values[window]
        mask = np.zeros(values.size, dtype=bool)
        for start, stop in _runs(values, min_run, tolerance):
            mask[start + 1:stop] = True
            held_depth = depth[start + 1:stop]
            rows.append({
                "curve": name,
                "top": float(np.nanmin(held_depth)),
                "bottom": float(np.nanmax(held_depth)),
                "samples": int(stop - start - 1),
                "value": float(values[start]),
            })
        masks[name] = mask

    runs = pd.DataFrame(rows, columns=["curve", "top", "bottom", "samples", "value"])
    return HeldReadings(
        depth=depth,
        masks=masks,
        runs=runs,
        min_run=min_run,
        tolerance=tolerance,
        depth_unit=str(getattr(well, "depth_unit", "") or ""),
        curves=names,
    )


__all__ = ["HeldReadings", "find_held_readings", "held_reading_mask"]
