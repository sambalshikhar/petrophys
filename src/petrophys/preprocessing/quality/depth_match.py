"""Depth matching of one log to another in the same well, the shift varying with depth.

The method is Kerzner's (1984): at every depth a short window of the
reference log is correlated with the log to match at every shift up to a
maximum; the local maxima of each correlogram are the candidate shifts there;
dynamic programming then picks one shift per depth so that the shifts are
consistent, as an analyst would: the matched log must not cross itself, its
total stretching (the change of shift from depth to depth) is kept small, and
every strong correlation left out costs its correlation. A log may follow the
reference (``sign="positive"``), mirror it (``"negative"``, density against
neutron) or do either from bed to bed (``None``, the absolute correlation).

The match is judged by correlation, which Le et al. (2019) found the best of
thirteen measures on manually matched gamma-ray pairs. The defaults were set
on the bundled wells: a known splice and stretch put into a real gamma ray is
recovered to within one sample at 90 % of depths, and curves already on depth
are left where they are.

References: Kerzner, M. G., 1984, A solution to the problem of automatic depth
matching, SPWLA 25th Annual Logging Symposium, paper VV. Le, T., Liang, L.,
Zimmermann, T., Zeroug, S. and Heliot, D., 2019, A machine-learning framework
for automating well-log depth matching, Petrophysics 60(5), 585-595.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from petrophys.preprocessing.quality.depth_alignment import _find


@dataclass
class DepthMatch:
    """Result of :func:`depth_match`.

    Attributes
    ----------
    depth : numpy.ndarray
        The well's depth index.
    shift : numpy.ndarray
        Correction of ``curve`` at each depth, in well depth units: the matched
        curve at depth ``z`` is the recorded curve at ``z - shift``. Negative:
        the curve reads too deep there. Pass the result to
        :func:`~petrophys.preprocessing.shift_curves`.
    centers, center_shift : numpy.ndarray
        The depths where shifts were chosen (window centres) and the shift at each.
    matched : numpy.ndarray of bool
        Per centre: the chosen shift is a correlogram maximum of at least
        ``min_correlation`` (else it is carried from the neighbours).
    correlation_before, correlation_after : float
        Median of the windowed correlations of the two curves, as recorded and
        after matching (signed as ``sign`` asks: higher is better).
    curve, reference : str
    depth_unit : str
    """

    depth: np.ndarray
    shift: np.ndarray
    centers: np.ndarray
    center_shift: np.ndarray
    matched: np.ndarray
    correlation_before: float
    correlation_after: float
    curve: str
    reference: str
    depth_unit: str = ""

    @property
    def max_shift(self) -> float:
        """Largest correction applied, in well depth units."""
        finite = self.shift[np.isfinite(self.shift)]
        return float(np.max(np.abs(finite))) if finite.size else 0.0

    @property
    def off_depth(self) -> bool:
        """True when matching clearly improves the tie: the median windowed
        correlation up by 0.2 or more to at least 0.35, with a correction of more
        than two samples somewhere."""
        step = float(np.nanmedian(np.diff(self.depth))) if self.depth.size > 1 else 0.0
        return bool(self.correlation_after >= 0.35
                    and self.correlation_after - self.correlation_before >= 0.2
                    and self.max_shift > 2 * step)

    def to_dataframe(self):
        """The shift at each window centre, with whether a correlation fixed it."""
        import pandas as pd

        return pd.DataFrame({"depth": self.centers, "shift": self.center_shift, "matched": self.matched})

    def __repr__(self) -> str:
        unit = f" {self.depth_unit}" if self.depth_unit else ""
        finite = self.shift[np.isfinite(self.shift)]
        low, high = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 0.0)
        return (f"DepthMatch({self.curve} to {self.reference}: shift {low:+.2f} to {high:+.2f}{unit}, "
                f"r {self.correlation_before:.2f} -> {self.correlation_after:.2f}, "
                f"{int(self.matched.sum())} of {self.matched.size} windows matched)")


def _rolling_sum(values: np.ndarray, centers: np.ndarray, half: int) -> np.ndarray:
    total = np.concatenate([[0.0], np.cumsum(values)])
    lo = np.clip(centers - half, 0, values.size)
    hi = np.clip(centers + half + 1, 0, values.size)
    return np.asarray(total[hi] - total[lo], dtype=float)


def _correlograms(x: np.ndarray, y: np.ndarray, centers: np.ndarray, half: int, lags: np.ndarray,
                  min_fraction: float) -> np.ndarray:
    """Pearson correlation of x around each centre with y shifted by each lag."""
    n = x.size
    out = np.full((centers.size, lags.size), np.nan)
    x0 = np.where(np.isfinite(x), x, 0.0)
    need = min_fraction * (2 * half + 1)
    for j, lag in enumerate(lags):
        shifted = np.full(n, np.nan)
        if lag >= 0:
            shifted[: n - lag] = y[lag:]
        else:
            shifted[-lag:] = y[: n + lag]
        ok = np.isfinite(x) & np.isfinite(shifted)
        okf = ok.astype(float)
        xs = np.where(ok, x0, 0.0)
        ys = np.where(ok, shifted, 0.0)
        count = _rolling_sum(okf, centers, half)
        sx, sy = _rolling_sum(xs, centers, half), _rolling_sum(ys, centers, half)
        sxx, syy = _rolling_sum(xs * xs, centers, half), _rolling_sum(ys * ys, centers, half)
        sxy = _rolling_sum(xs * ys, centers, half)
        with np.errstate(invalid="ignore", divide="ignore"):
            cov = sxy - sx * sy / count
            vx = sxx - sx * sx / count
            vy = syy - sy * sy / count
            r = cov / np.sqrt(vx * vy)
        r[(count < need) | ~(vx > 1e-12 * np.maximum(count, 1)) | ~(vy > 1e-12 * np.maximum(count, 1))] = np.nan
        out[:, j] = r
    return out


def depth_match(
    well,
    curve: str,
    reference: str = "GR",
    *,
    sign: Optional[str] = "positive",
    max_shift: Optional[float] = None,
    local_shift: Optional[float] = None,
    window: Optional[float] = None,
    stiffness: float = 0.5,
    min_correlation: float = 0.35,
    min_gain: float = 0.2,
    local_gain: float = 0.1,
    top: Optional[float] = None,
    bottom: Optional[float] = None,
) -> DepthMatch:
    """Match ``curve`` to ``reference`` in depth, the shift free to vary with depth.

    In two passes, as Kerzner (1984) advises: the major shift first, then the
    detail. The bulk shift is the one whose windowed correlations have the
    highest median, so that many windows must agree, kept when that median is
    ``min_correlation`` or more and ``min_gain`` above the median as recorded. Around it, within ``local_shift``, correlograms in
    short windows give candidate shifts at every depth, a window rewarded by
    how much better it correlates than at the bulk shift; dynamic programming
    keeps the consistent set. The result is a shift at every sample, for
    :func:`~petrophys.preprocessing.shift_curves`.

    Parameters
    ----------
    well : Well
        Well carrying both curves (own or standard names: ``"DTC"`` finds DT).
    curve : str
        Curve to move.
    reference : str, default "GR"
        Curve taken as on depth. Gamma ray is recorded on every logging pass so
        that passes can be tied to it.
    sign : {"positive", "negative", None}, default "positive"
        How ``curve`` follows the reference: in step (``"positive"``), mirrored
        (``"negative"``: density against neutron, resistivity against gamma ray
        in clastics) or either, bed by bed (``None``).
    max_shift : float, optional
        Largest bulk correction, in well depth units. Default 20 m (65.6 ft).
    local_shift : float, optional
        How far the correction may vary around the bulk shift. Default 2 m
        (6.6 ft).
    window : float, optional
        Length of the correlation window of the detail pass. Default 10 m
        (32.8 ft).
    stiffness : float, default 0.5
        Cost of changing the shift by one sample from one window to the next,
        in units of correlation. Higher keeps the shift steadier (a bulk
        shift); lower lets it follow every wiggle.
    min_correlation : float, default 0.35
        A correlogram maximum below this is not a candidate, and a bulk shift
        whose median windowed correlation is below it is not kept.
    min_gain : float, default 0.2
        The bulk shift is kept when it raises the median windowed correlation
        by this much; otherwise the curves are taken as tied (bulk 0). On 621
        density, neutron and sonic pairs of 221 wells the best shift raised it
        by less than 0.11 in 95 % of them.
    local_gain : float, default 0.1
        A window counts towards a local shift by its correlation gain over the
        bulk shift beyond this margin: in a window of tens of samples a
        correlation varies by about this much from noise alone.
    top, bottom : float, optional
        Depth window to match; the shift is 0 outside it.

    Returns
    -------
    DepthMatch

    Examples
    --------
    >>> import petrophys as pp
    >>> from petrophys.preprocessing import depth_match, shift_curves
    >>> hw3 = pp.read(pp.sample("HW-3"))
    >>> match = depth_match(hw3, "RHOB", "NPHI", sign="negative")
    >>> aligned = shift_curves(hw3, ["RHOB", "DTC"], match)
    """
    if sign not in (None, "positive", "negative"):
        raise ValueError(f"sign must be 'positive', 'negative' or None; got {sign!r}")
    if not 0.0 <= float(stiffness):
        raise ValueError("stiffness must be 0 or more")
    depth = np.asarray(well.depth, dtype=float)
    if depth.size < 3:
        raise ValueError("The well has too few samples to match")
    step = float(np.nanmedian(np.diff(depth)))
    if not np.isfinite(step) or step <= 0:
        raise ValueError("The depth index must increase with a regular step")
    unit = str(getattr(well, "depth_unit", "") or "")
    feet = unit.lower().startswith("f")
    max_shift = (20.0 / 0.3048 if feet else 20.0) if max_shift is None else float(max_shift)
    local_shift = (2.0 / 0.3048 if feet else 2.0) if local_shift is None else float(local_shift)
    window = (10.0 / 0.3048 if feet else 10.0) if window is None else float(window)
    x = np.asarray(_find(well, reference).data, dtype=float)
    y = np.asarray(_find(well, curve).data, dtype=float)
    inside = np.isfinite(depth)
    if top is not None:
        inside &= depth >= float(top)
    if bottom is not None:
        inside &= depth <= float(bottom)
    x = np.where(inside, x, np.nan)

    half = max(int(round(window / step / 2)), 3)
    grid = max(half // 2, 1)  # centres a quarter window apart
    rows = np.flatnonzero(inside & np.isfinite(x))
    if rows.size < 2 * half + 1:
        raise ValueError(f"{reference} has too few samples in the window to match")

    centers = np.arange(rows[0], rows[-1] + 1, grid)

    def scored(lags: np.ndarray) -> np.ndarray:
        r = _correlograms(x, y, centers, half, lags, min_fraction=0.6)
        return np.asarray({"positive": r, "negative": -r, None: np.abs(r)}[sign], dtype=float)

    # Pass 1: the major shift. Each shift is judged by the median of its
    # windowed correlations, so many windows must agree; one interval of bad
    # readings cannot carry a whole-log correlation.
    n_lag = max(int(round(max_shift / step)), 1)
    all_lags = np.arange(-n_lag, n_lag + 1)
    windows = scored(all_lags)
    enough = np.isfinite(windows).sum(axis=0) >= max(3, 0.5 * np.isfinite(windows[:, n_lag]).sum())
    with np.errstate(all="ignore"):
        typical = np.where(enough, np.nanmedian(np.where(np.isfinite(windows), windows, np.nan), axis=0), np.nan)
    bulk_lag = 0
    if np.isfinite(typical).any():
        best_lag = int(np.nanargmax(typical))
        at_edge = best_lag in (0, all_lags.size - 1)
        if (not at_edge and typical[best_lag] >= min_correlation
                and typical[best_lag] - np.nan_to_num(typical[n_lag], nan=0.0) >= min_gain):
            bulk_lag = int(all_lags[best_lag])

    # Pass 2: the detail, within local_shift of the major shift.
    n_local = max(int(round(local_shift / step)), 1)
    lags = np.arange(bulk_lag - n_local, bulk_lag + n_local + 1)
    inside_lags = (lags >= -n_lag) & (lags <= n_lag)
    score = np.full((centers.size, lags.size), np.nan)
    score[:, inside_lags] = windows[:, lags[inside_lags] + n_lag]

    # Candidates: correlogram maxima along the shift axis, strong enough, and the
    # bulk shift wherever it correlates; each scores its correlation, and a shift
    # other than the bulk one pays the noise margin, so it must correlate better
    # by that much to be chosen.
    padded = np.pad(np.nan_to_num(score, nan=-np.inf), ((0, 0), (1, 1)), constant_values=-np.inf)
    peak = (padded[:, 1:-1] >= padded[:, :-2]) & (padded[:, 1:-1] >= padded[:, 2:])
    peak[:, n_local] = True
    strong = peak & (np.nan_to_num(score, nan=-1.0) >= min_correlation)
    margin = np.where(lags == bulk_lag, 0.0, local_gain)[None, :]
    reward = np.where(strong, np.maximum(score - margin, 0.0), 0.0)
    # Where nothing correlates, a slight pull towards the bulk shift.
    reward = reward - 1e-3 * np.abs(lags - bulk_lag)[None, :]

    # Dynamic programming: a shift per centre, the most correlation kept for
    # the least stretching. Between centres `grid` samples apart the shift may
    # change by less than `grid` samples, so the matched log never folds.
    reach = grid - 1
    best = reward[0].copy()
    back = np.zeros((centers.size, lags.size), dtype=int)
    for k in range(1, centers.size):
        candidate = np.full((2 * reach + 1, lags.size), -np.inf)
        for i, d in enumerate(range(-reach, reach + 1)):
            moved = np.full(lags.size, -np.inf)
            if d >= 0:
                moved[d:] = best[: lags.size - d]
            else:
                moved[:d] = best[-d:]
            candidate[i] = moved - stiffness * abs(d)
        pick = np.argmax(candidate, axis=0)
        back[k] = np.arange(lags.size) - (pick - reach)
        best = candidate[pick, np.arange(lags.size)] + reward[k]
    path = np.empty(centers.size, dtype=int)
    path[-1] = int(np.argmax(best))
    for k in range(centers.size - 1, 0, -1):
        path[k - 1] = back[k, path[k]]
    chosen_lag = lags[path]
    matched = reward[np.arange(centers.size), path] > 0

    # A reading at sample i + lag belongs at i: the correction is minus the lag.
    center_shift = -chosen_lag * step
    shift = np.zeros(depth.size)
    span = np.zeros(depth.size, dtype=bool)
    span[centers[0]: centers[-1] + 1] = True
    shift[span] = np.interp(np.flatnonzero(span), centers, center_shift)
    shift[:centers[0]] = center_shift[0]
    shift[centers[-1] + 1:] = center_shift[-1]
    shift[~inside] = 0.0

    # The tie as recorded and after matching: the median of the windowed correlations.
    before = windows[:, n_lag]
    after = windows[np.arange(centers.size), np.clip(chosen_lag + n_lag, 0, all_lags.size - 1)]
    with np.errstate(all="ignore"):
        tie_before = float(np.nanmedian(before)) if np.isfinite(before).any() else float("nan")
        tie_after = float(np.nanmedian(after)) if np.isfinite(after).any() else float("nan")
    return DepthMatch(depth=depth, shift=shift, centers=depth[centers], center_shift=center_shift,
                      matched=matched, correlation_before=tie_before, correlation_after=tie_after,
                      curve=str(curve), reference=str(reference), depth_unit=unit)


def _moved(depth: np.ndarray, values: np.ndarray, shift: np.ndarray) -> np.ndarray:
    """``values`` read at ``depth - shift``: the curve after its correction; gaps stay gaps."""
    finite = np.isfinite(values)
    if finite.sum() < 2:
        return np.full(depth.shape, np.nan)
    source = depth - shift
    out = np.interp(source, depth[finite], values[finite], left=np.nan, right=np.nan)
    nearest = np.clip(np.searchsorted(depth, source), 0, depth.size - 1)
    out[~finite[nearest]] = np.nan
    return np.asarray(out, dtype=float)


__all__ = ["DepthMatch", "depth_match"]
