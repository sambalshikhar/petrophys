"""Depth matching of one log curve to another in the same well.

Tools on one string are recorded against one depth, but a delivered file can
still carry a curve that is off depth: a run spliced at the wrong depth, a
merge that shifted one tool, a pad tool corrected and its neighbours not.
Two curves that respond to the same beds then line up best at a non-zero
lag. :func:`find_log_depth_shift` scans the lags and returns the shift that
lines a curve up with a reference curve; :func:`shift_curves` applies it.

A density log and a neutron log respond to the same porosity in opposite
directions, so they correlate negatively when on depth; the scan maximises
the size of the correlation by default and reports its sign.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class LogDepthShift:
    """Result of :func:`find_log_depth_shift`.

    Attributes
    ----------
    shift : float
        Depth correction for ``curve``, in well depth units: the curve's
        reading at depth ``z`` belongs at ``z + shift``. Pass it to
        :func:`shift_curves`. Negative: the curve reads too deep.
    correlation : float
        Correlation of the two curves at the best shift.
    correlation_unshifted : float
        Correlation as recorded (shift 0).
    n_points : int
        Number of sample pairs behind ``correlation``.
    shifts, correlations : numpy.ndarray
        Every scanned shift and its correlation.
    curve, reference : str
        The curve that was moved and the curve it was matched to.
    depth_unit : str
        Depth unit of the well.
    """

    shift: float
    correlation: float
    correlation_unshifted: float
    n_points: int
    shifts: np.ndarray
    correlations: np.ndarray
    curve: str
    reference: str
    depth_unit: str = ""

    @property
    def at_search_edge(self) -> bool:
        """True when the best shift is the first or last one scanned: widen ``max_shift``."""
        finite = np.flatnonzero(np.isfinite(self.correlations))
        if finite.size == 0:
            return False
        return bool(np.isclose(self.shift, self.shifts[finite[0]]) or np.isclose(self.shift, self.shifts[finite[-1]]))

    @property
    def off_depth(self) -> bool:
        """True when the curves clearly line up better shifted than as recorded.

        The best correlation is strong (size at least 0.6), at least 0.3
        better than as recorded, the shift is more than two samples and not at
        the edge of the scan. On 208 wells with density and neutron logs
        (the bundled samples, the GeoLink and COSTA wells) only HW-3 passes:
        its RHOB lines up with NPHI 55 ft (16.8 m) higher, r -0.79 against
        +0.11 as recorded.
        """
        step = float(np.nanmedian(np.abs(np.diff(self.shifts)))) if self.shifts.size > 1 else 0.0
        return bool(
            np.isfinite(self.correlation)
            and abs(self.correlation) >= 0.6
            and abs(self.correlation) - abs(np.nan_to_num(self.correlation_unshifted)) >= 0.3
            and abs(self.shift) > 2 * step
            and not self.at_search_edge
        )

    def to_dataframe(self):
        """Return the scan as a DataFrame with ``shift`` and ``correlation``."""
        import pandas as pd

        return pd.DataFrame({"shift": self.shifts, "correlation": self.correlations})

    def __repr__(self) -> str:
        unit = f" {self.depth_unit}" if self.depth_unit else ""
        return (
            f"LogDepthShift({self.curve} to {self.reference}: shift {self.shift:+.2f}{unit}, "
            f"r = {self.correlation:.3f} (as recorded {self.correlation_unshifted:.3f}), "
            f"n = {self.n_points}, searched {self.shifts.min():+.2f} to {self.shifts.max():+.2f}{unit})"
        )


def _find(well, name: str):
    """The well's curve by its own name or a standard name (DTC finds DT)."""
    from petrophys.preprocessing.standardization.alias_map import get_aliases

    for candidate in [name, *get_aliases(str(name))]:
        curve = well.get(candidate)
        if curve is not None:
            return curve
    raise ValueError(f"No curve {name!r} on the well, by its own name or a standard name; "
                     f"it has {list(well.curves)}")


def _curve(well, name: str) -> np.ndarray:
    return np.asarray(_find(well, name).data, dtype=float)


def find_log_depth_shift(
    well,
    curve: str,
    reference: str,
    *,
    max_shift: Optional[float] = None,
    top: Optional[float] = None,
    bottom: Optional[float] = None,
    sign: Optional[str] = None,
    min_points: int = 200,
) -> LogDepthShift:
    """Find the depth shift that lines ``curve`` up with ``reference``.

    Every whole-sample lag up to ``max_shift`` is tried; at each, the Pearson
    correlation of the two curves over their common samples is computed. The
    lag with the strongest correlation wins.

    Parameters
    ----------
    well : Well
        Well carrying both curves, on one depth index.
    curve : str
        Curve to move, e.g. ``"RHOB"``.
    reference : str
        Curve taken as on depth, e.g. ``"NPHI"`` or ``"GR"``.
    max_shift : float, optional
        Largest shift to try either way, in well depth units. Default 20 m
        (65.6 ft).
    top, bottom : float, optional
        Depth window of the comparison, in well depth units. Default: the
        whole log.
    sign : {"positive", "negative"}, optional
        Expected sign of the correlation when on depth: ``"negative"`` for
        density against neutron, ``"positive"`` for neutron against gamma
        ray. Default: the strongest correlation of either sign.
    min_points : int, default 200
        Fewest sample pairs a lag needs to be scored.

    Returns
    -------
    LogDepthShift
        ``.shift`` to pass to :func:`shift_curves`, the correlation as
        recorded and shifted, and ``.off_depth`` when the shift is clear.

    Examples
    --------
    >>> import petrophys as pp
    >>> from petrophys.preprocessing import find_log_depth_shift
    >>> hw3 = pp.read(pp.sample("HW-3"))
    >>> found = find_log_depth_shift(hw3, "RHOB", "NPHI")
    >>> round(found.shift), round(found.correlation, 2), found.off_depth
    (-55, -0.79, True)
    """
    if sign not in (None, "positive", "negative"):
        raise ValueError(f"sign must be 'positive', 'negative' or None; got {sign!r}")
    if well.depth is None:
        raise ValueError("Well has no depth data")
    depth = np.asarray(well.depth, dtype=float)
    moved, fixed = _curve(well, curve), _curve(well, reference)
    window = np.isfinite(depth)
    if top is not None:
        window &= depth >= float(top)
    if bottom is not None:
        window &= depth <= float(bottom)
    step = float(np.nanmedian(np.diff(depth))) if depth.size > 1 else float("nan")
    if not np.isfinite(step) or step <= 0:
        raise ValueError("The depth index must increase with a regular step to scan lags")
    unit = str(getattr(well, "depth_unit", "") or "")
    if max_shift is None:
        max_shift = 20.0 / 0.3048 if unit.lower().startswith("f") else 20.0
    n_lags = int(round(float(max_shift) / step))

    lags = np.arange(-n_lags, n_lags + 1)
    correlations = np.full(lags.size, np.nan)
    counts = np.zeros(lags.size, dtype=int)
    index = np.flatnonzero(window)
    for k, lag in enumerate(lags):
        # The curve's sample at i + lag is compared with the reference at i.
        source = index + lag
        inside = (source >= 0) & (source < moved.size)
        a, b = moved[source[inside]], fixed[index[inside]]
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < min_points or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
            continue
        correlations[k] = np.corrcoef(a[ok], b[ok])[0, 1]
        counts[k] = int(ok.sum())
    if not np.any(np.isfinite(correlations)):
        raise ValueError(f"{curve} and {reference} share fewer than {min_points} samples at every lag")

    score = {None: np.abs(correlations), "positive": correlations, "negative": -correlations}[sign]
    best = int(np.nanargmax(score))
    zero = int(np.flatnonzero(lags == 0)[0])
    # A reading at i + lag belongs at i: the correction is minus the lag.
    shifts = -lags * step
    return LogDepthShift(
        shift=float(shifts[best]) + 0.0,
        correlation=float(correlations[best]),
        correlation_unshifted=float(correlations[zero]),
        n_points=int(counts[best]),
        shifts=shifts[::-1].copy(),
        correlations=correlations[::-1].copy(),
        curve=str(curve),
        reference=str(reference),
        depth_unit=unit,
    )


def shift_curves(well, curves, shift):
    """Return a copy of ``well`` with ``curves`` moved by ``shift``.

    ``shift`` is one correction in well depth units, a correction per sample,
    or a :class:`~petrophys.preprocessing.DepthMatch`: the moved curve at depth
    ``z`` is the recorded curve at ``z - shift``, so a reading recorded at
    ``z`` moves to ``z + shift`` and a negative shift moves a curve up. Values
    are interpolated onto the well's depth index; depths the moved curve does
    not reach become NaN, and gaps stay gaps. Each moved curve's description
    records the shift.

    Examples
    --------
    >>> import petrophys as pp
    >>> from petrophys.preprocessing import find_log_depth_shift, shift_curves
    >>> hw3 = pp.read(pp.sample("HW-3"))
    >>> aligned = shift_curves(hw3, ["RHOB", "DTC"], find_log_depth_shift(hw3, "RHOB", "NPHI").shift)
    """
    from copy import deepcopy

    from petrophys.preprocessing.quality.depth_match import _moved

    names = [str(name) for name in ([curves] if isinstance(curves, str) else curves)]
    out = deepcopy(well)
    depth = np.asarray(out.depth, dtype=float)
    values_of_shift = np.asarray(getattr(shift, "shift", shift), dtype=float)
    if values_of_shift.ndim == 0:
        values_of_shift = np.full(depth.shape, float(values_of_shift))
    if values_of_shift.shape != depth.shape:
        raise ValueError(f"shift has {values_of_shift.size} values; the well has {depth.size} samples")
    if not np.all(np.isfinite(values_of_shift)):
        raise ValueError("shift must be finite")
    unit = str(getattr(well, "depth_unit", "") or "")
    low, high = float(values_of_shift.min()), float(values_of_shift.max())
    note = (f"shifted {low:+g} {unit}" if low == high else f"shifted {low:+g} to {high:+g} {unit}").strip()
    for name in names:
        curve = _find(out, name)
        curve.data = _moved(depth, np.asarray(curve.data, dtype=float), values_of_shift)
        curve.metadata.description = f"{curve.description}; {note}" if curve.description else note
    return out


__all__ = ["LogDepthShift", "find_log_depth_shift", "shift_curves"]
