"""Core-to-log depth matching.

Core plug depths are usually driller's depths, measured on the drill pipe
while the core was cut. Logs are measured on the wireline cable. The two
often differ by a metre or more, and each core run can carry its own offset.
An offset silently weakens every core-to-log comparison, because the log is
read at the wrong depth for every plug.

:func:`find_core_depth_shift` scans candidate shifts of the plug depths and
scores each one by the Pearson correlation between a core property and a log
curve read at the shifted depths. :func:`shift_core_depths` applies the chosen
shift and records it in the core metadata.

A shift picked this way is fitted to the same data it is judged on, so the
correlation at the best shift is optimistic. Two checks guard against a chance
peak: the scan should show a clear maximum away from the ends of the search
window, and separate core runs (``run_gap``) should prefer shifts of similar
sign and size. Where core gamma ray is available, matching it to the log GR is
the stronger test.
"""

from __future__ import annotations

import dataclasses
import warnings
from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np

from petrophys.interpretation.core_analysis.calibration import _log_at_depth
from petrophys.model.core import CoreData

_CORE_PROPERTIES = ("porosity", "permeability", "grain_density", "sw", "so")
_POSITIVE_PROPERTIES = ("porosity", "permeability")


@dataclass(repr=False)
class CoreDepthShift:
    """Result of a core-to-log depth-shift scan.

    Attributes
    ----------
    shift : float
        Best shift, in well depth units. It is **added** to the core depths:
        a positive shift moves the plugs deeper (log depth = core depth +
        shift).
    correlation : float
        Pearson correlation between the core property and the log at the best
        shift.
    correlation_unshifted : float
        The same correlation with the plug depths as stored (shift 0).
    n_points : int
        Number of plug/log pairs behind ``correlation``.
    shifts, correlations : numpy.ndarray
        Every scanned shift and its correlation (NaN where fewer than
        ``min_points`` pairs were valid).
    log_curve : str
        Log curve that was matched.
    core_property : str
        Core property that was matched (a :class:`~petrophys.model.CoreData`
        attribute name).
    log_scale : bool
        Whether both series were compared as log10 values.
    depth_unit : str
        Depth unit of the well (``"m"`` or ``"ft"``).
    runs : pandas.DataFrame or None
        Per-core-run scan when ``run_gap`` was given: one row per run with
        ``top``, ``bottom``, ``plugs``, ``shift`` and ``correlation``.
    """

    shift: float
    correlation: float
    correlation_unshifted: float
    n_points: int
    shifts: np.ndarray
    correlations: np.ndarray
    log_curve: str
    core_property: str
    log_scale: bool = False
    depth_unit: str = ""
    runs: Optional["object"] = None

    @property
    def r_squared(self) -> float:
        """Coefficient of determination (squared correlation) at the best shift."""
        return float(self.correlation) ** 2

    @property
    def at_search_edge(self) -> bool:
        """True when the best shift is the first or last scanned shift.

        The correlation might keep rising outside the scanned window; widen
        ``max_shift`` before trusting the result.
        """
        finite = np.flatnonzero(np.isfinite(self.correlations))
        if finite.size == 0:
            return False
        edges = (self.shifts[finite[0]], self.shifts[finite[-1]])
        return bool(np.isclose(self.shift, edges[0]) or np.isclose(self.shift, edges[1]))

    def to_dataframe(self):
        """Return the scan as a DataFrame with ``shift``, ``correlation`` and ``r_squared``."""
        import pandas as pd

        return pd.DataFrame({
            "shift": self.shifts,
            "correlation": self.correlations,
            "r_squared": self.correlations ** 2,
        })

    def __repr__(self) -> str:
        unit = f" {self.depth_unit}" if self.depth_unit else ""
        return (
            f"CoreDepthShift({self.core_property} vs {self.log_curve}: "
            f"shift {self.shift:+.2f}{unit}, r = {self.correlation:.3f} "
            f"(unshifted {self.correlation_unshifted:.3f}), n = {self.n_points}, "
            f"searched {self.shifts.min():+.2f} to {self.shifts.max():+.2f}{unit})"
        )


def _core_values(core: CoreData, core_property: str, log_scale: bool) -> np.ndarray:
    if core_property not in _CORE_PROPERTIES:
        raise ValueError(
            f"core_property must be one of {', '.join(_CORE_PROPERTIES)}; got {core_property!r}"
        )
    values = getattr(core, core_property)
    if values is None:
        raise ValueError(f"CoreData has no {core_property} measurements")
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size != np.asarray(core.depth).size:
        raise ValueError(
            f"CoreData.{core_property} has {values.size} values for "
            f"{np.asarray(core.depth).size} plug depths"
        )
    values = values.copy()
    if core_property in _POSITIVE_PROPERTIES or log_scale:
        # Zero porosity or permeability marks an unmeasured plug, and log10
        # needs positive values.
        values[~(values > 0)] = np.nan
    if log_scale:
        values = np.log10(values)
    return values


def _correlation(core_values: np.ndarray, log_values: np.ndarray, min_points: int):
    valid = np.isfinite(core_values) & np.isfinite(log_values)
    count = int(np.count_nonzero(valid))
    if count < min_points:
        return np.nan, count
    x = core_values[valid]
    y = log_values[valid]
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan, count
    return float(np.corrcoef(x, y)[0, 1]), count


def _scan(log_depth, log_values, plug_depth, core_values, shifts, min_points, log_scale):
    correlations = np.full(shifts.shape, np.nan)
    counts = np.zeros(shifts.shape, dtype=int)
    for index, shift in enumerate(shifts):
        at_plugs = _log_at_depth(log_depth, log_values, plug_depth + shift)
        if log_scale:
            at_plugs = np.where(at_plugs > 0, at_plugs, np.nan)
            with np.errstate(divide="ignore", invalid="ignore"):
                at_plugs = np.log10(at_plugs)
        correlations[index], counts[index] = _correlation(core_values, at_plugs, min_points)
    return correlations, counts


def find_core_depth_shift(
    well,
    core: CoreData,
    log_curve: str = "PHIE",
    *,
    core_property: str = "porosity",
    shifts: Optional[Sequence[float]] = None,
    max_shift: Optional[float] = None,
    step: Optional[float] = None,
    log_scale: bool = False,
    run_gap: Optional[float] = None,
    min_run_plugs: int = 30,
    min_points: int = 10,
) -> CoreDepthShift:
    """Find the depth shift that best ties core plugs to a log curve.

    Each candidate shift is added to the plug depths, the log is interpolated
    at the shifted depths, and the Pearson correlation between the core
    property and the log is computed. The shift with the highest (signed)
    correlation wins; a negative correlation is never treated as a good tie.

    Parameters
    ----------
    well : Well
        Well carrying ``log_curve``. The log must be computed independently
        of the core (comparing core porosity with a core-corrected curve
        gives a meaningless tie).
    core : CoreData
        Core plugs, for example from
        :func:`~petrophys.interpretation.core_analysis.core_data_from_well`.
    log_curve : str, default "PHIE"
        Log curve to match.
    core_property : str, default "porosity"
        ``CoreData`` attribute to match: ``"porosity"``, ``"permeability"``,
        ``"grain_density"``, ``"sw"`` or ``"so"``. Zero or negative porosity
        and permeability are treated as unmeasured.
    shifts : sequence of float, optional
        Explicit shifts to scan, in well depth units. Overrides
        ``max_shift`` and ``step``.
    max_shift : float, optional
        Largest shift to scan in either direction, in well depth units.
        Defaults to 30 log samples (about 4.6 m, or 15 ft, at the standard
        0.1524 m / 0.5 ft sampling).
    step : float, optional
        Shift increment in well depth units. Defaults to the median log
        sample step.
    log_scale : bool, default False
        Compare log10 values; use it for permeability.
    run_gap : float, optional
        When given, also scan each core run on its own. Runs are separated
        where consecutive plugs are more than ``run_gap`` apart (depth units).
    min_run_plugs : int, default 30
        Runs with fewer valid plugs are not scanned on their own.
    min_points : int, default 10
        Minimum valid plug/log pairs for a correlation to be computed.

    Returns
    -------
    CoreDepthShift
        Best shift, the correlation before and after, the full scan and,
        with ``run_gap``, the per-run shifts.

    Raises
    ------
    ValueError
        If the curve or core property is missing, the scan settings are
        invalid, or no shift has ``min_points`` valid pairs.

    Warns
    -----
    UserWarning
        If the best shift lies at the edge of the scanned window, or its
        correlation is below 0.5.

    See Also
    --------
    shift_core_depths : Apply the shift to a ``CoreData``.
    petrophys.interpretation.core_analysis.calibrate_porosity : Calibration
        statistics at the stored (or shifted) plug depths.

    Examples
    --------
    >>> import numpy as np
    >>> from petrophys import Well
    >>> from petrophys.interpretation.core_analysis import CoreData
    >>> well = Well("DEMO")
    >>> well.depth = np.arange(1000.0, 1100.0, 0.5)
    >>> _ = well.add_curve("PHID", 0.2 + 0.05 * np.sin(well.depth / 3.0), "v/v")
    >>> plugs = np.arange(1020.0, 1080.0, 1.0)
    >>> core = CoreData(depth=plugs - 2.0,                  # stored 2 m too shallow
    ...                 porosity=0.2 + 0.05 * np.sin(plugs / 3.0))
    >>> scan = find_core_depth_shift(well, core, "PHID", max_shift=5.0)
    >>> float(scan.shift)
    2.0
    >>> tied = shift_core_depths(core, scan)
    """
    curve = well.get(log_curve)
    if curve is None:
        raise ValueError(f"Curve '{log_curve}' not found on well")
    if well.depth is None:
        raise ValueError("Well has no depth data")
    log_depth = np.asarray(well.depth, dtype=float).reshape(-1)
    log_values = np.asarray(curve.data, dtype=float).reshape(-1)
    if log_depth.size != log_values.size:
        raise ValueError(
            f"Curve '{log_curve}' length ({log_values.size}) does not match "
            f"well depth length ({log_depth.size})"
        )
    if int(min_points) < 3:
        raise ValueError("min_points must be at least 3")

    core_values = _core_values(core, core_property, log_scale)
    plug_depth = np.asarray(core.depth, dtype=float).reshape(-1)

    if shifts is not None:
        scan_shifts = np.asarray(shifts, dtype=float).reshape(-1)
        if scan_shifts.size == 0 or not np.all(np.isfinite(scan_shifts)):
            raise ValueError("shifts must be a non-empty sequence of finite numbers")
        scan_shifts = np.unique(scan_shifts)
    else:
        if step is None:
            increments = np.abs(np.diff(log_depth[np.isfinite(log_depth)]))
            increments = increments[increments > 0]
            if increments.size == 0:
                raise ValueError("Cannot infer the log sample step; pass step=")
            step = float(np.median(increments))
        step = float(step)
        if not np.isfinite(step) or step <= 0:
            raise ValueError("step must be a positive number")
        if max_shift is None:
            max_shift = 30.0 * step
        max_shift = float(max_shift)
        if not np.isfinite(max_shift) or max_shift < 0:
            raise ValueError("max_shift must be a non-negative number")
        count = int(np.floor(max_shift / step + 1e-9))
        scan_shifts = np.arange(-count, count + 1) * step

    correlations, counts = _scan(
        log_depth, log_values, plug_depth, core_values, scan_shifts, int(min_points), log_scale
    )
    if not np.any(np.isfinite(correlations)):
        raise ValueError(
            f"No scanned shift had {int(min_points)} valid {core_property}/{log_curve} pairs; "
            "check that the log covers the cored interval"
        )
    best = int(np.nanargmax(correlations))
    unshifted, _ = _scan(
        log_depth, log_values, plug_depth, core_values, np.zeros(1), int(min_points), log_scale
    )

    runs = None
    if run_gap is not None:
        runs = _scan_runs(
            log_depth, log_values, plug_depth, core_values, scan_shifts,
            float(run_gap), int(min_run_plugs), int(min_points), log_scale,
        )

    result = CoreDepthShift(
        shift=float(scan_shifts[best]),
        correlation=float(correlations[best]),
        correlation_unshifted=float(unshifted[0]),
        n_points=int(counts[best]),
        shifts=scan_shifts,
        correlations=correlations,
        log_curve=log_curve,
        core_property=core_property,
        log_scale=bool(log_scale),
        depth_unit=str(getattr(well, "depth_unit", "") or ""),
        runs=runs,
    )
    if result.at_search_edge and scan_shifts.size > 1:
        warnings.warn(
            f"The best core depth shift ({result.shift:+.2f}) lies at the edge of the "
            "scanned window; widen max_shift before trusting it.",
            UserWarning,
            stacklevel=2,
        )
    elif scan_shifts.size > 1 and result.correlation < 0.5:
        warnings.warn(
            f"The best core depth shift ({result.shift:+.2f}) correlates weakly with "
            f"{log_curve} (r = {result.correlation:.2f}); it may be a local peak. Widen "
            "max_shift, or check the core depth reference, before applying it.",
            UserWarning,
            stacklevel=2,
        )
    return result


def _scan_runs(log_depth, log_values, plug_depth, core_values, shifts,
               run_gap, min_run_plugs, min_points, log_scale):
    import pandas as pd

    if not np.isfinite(run_gap) or run_gap <= 0:
        raise ValueError("run_gap must be a positive number")
    has_value = np.isfinite(core_values) & np.isfinite(plug_depth)
    order = np.argsort(plug_depth[has_value])
    depths = plug_depth[has_value][order]
    values = core_values[has_value][order]
    rows = []
    if depths.size:
        run_id = np.r_[0, np.cumsum(np.diff(depths) > run_gap)]
        for rid in np.unique(run_id):
            in_run = run_id == rid
            if np.count_nonzero(in_run) < min_run_plugs:
                continue
            correlations, _ = _scan(
                log_depth, log_values, depths[in_run], values[in_run], shifts,
                min(min_points, int(np.count_nonzero(in_run))), log_scale,
            )
            if not np.any(np.isfinite(correlations)):
                continue
            best = int(np.nanargmax(correlations))
            rows.append({
                "top": float(depths[in_run].min()),
                "bottom": float(depths[in_run].max()),
                "plugs": int(np.count_nonzero(in_run)),
                "shift": float(shifts[best]),
                "correlation": float(correlations[best]),
            })
    return pd.DataFrame(rows, columns=["top", "bottom", "plugs", "shift", "correlation"])


def shift_core_depths(
    core: CoreData,
    shift: Union[float, Sequence[float], np.ndarray, CoreDepthShift],
) -> CoreData:
    """Return a copy of ``core`` with every plug depth moved by ``shift``.

    Parameters
    ----------
    core : CoreData
        Core plugs to shift. The input is not modified.
    shift : float, array-like or CoreDepthShift
        Shift added to the plug depths, in well depth units (positive moves
        the plugs deeper). An array gives one shift per plug, for example a
        different shift for each core run. A :class:`CoreDepthShift` applies
        its best shift.

    Returns
    -------
    CoreData
        Shifted copy. ``metadata["depth_shift"]`` holds the total shift
        applied so far, so repeated shifts stay traceable.

    Raises
    ------
    ValueError
        If the shift is not finite or does not match the number of plugs.

    Examples
    --------
    >>> import numpy as np
    >>> from petrophys.interpretation.core_analysis import CoreData
    >>> core = CoreData(depth=np.array([2990.3, 2990.6]), porosity=np.array([0.21, 0.19]))
    >>> moved = shift_core_depths(core, 2.9)
    >>> moved.depth.round(1).tolist(), moved.metadata["depth_shift"]
    ([2993.2, 2993.5], 2.9)
    """
    if isinstance(shift, CoreDepthShift):
        shift = shift.shift
    depth = np.asarray(core.depth, dtype=float)
    delta = np.asarray(shift, dtype=float)
    if delta.ndim == 0:
        delta = float(delta)
        if not np.isfinite(delta):
            raise ValueError("shift must be finite")
    else:
        delta = delta.reshape(-1)
        if delta.size != depth.size:
            raise ValueError(
                f"shift has {delta.size} values for {depth.size} plugs; pass one "
                "value, or one per plug"
            )
        if not np.all(np.isfinite(delta)):
            raise ValueError("shift must be finite")
    metadata = dict(core.metadata)
    previous = metadata.get("depth_shift", 0.0)
    metadata["depth_shift"] = previous + delta
    return dataclasses.replace(core, depth=depth + delta, metadata=metadata)


__all__ = ["CoreDepthShift", "find_core_depth_shift", "shift_core_depths"]
