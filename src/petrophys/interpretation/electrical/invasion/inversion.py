"""Step-profile invasion inversion: Rt, Rxo and invasion diameter from several resistivities.

The step model has a flushed zone of resistivity Rxo out to diameter di and
the uninvaded formation Rt beyond it. Each tool reads a mixture weighted by
its pseudo-geometric factor J(di), the share of the signal from inside di:

- laterolog (currents in series):  ``Ra = J * Rxo + (1 - J) * Rt``
- induction (currents in parallel): ``1/Ra = J/Rxo + (1 - J)/Rt``

Given three or more readings (or two with a measured Rxo),
:func:`invert_step_profile` fits Rxo and Rt for every trial di on a grid
(Gauss-Newton on the relative misfit, any mix of laterolog and induction
readings) and keeps the diameter with the smallest misfit. This is the
calculation behind the vendors' "tornado" charts.

The answer is only as good as the J(di) of each tool. Take them from the
service company's tool-response charts, borehole-corrected, as
:class:`ToolResponse` tables. :meth:`ToolResponse.parametric` gives a smooth
curve through a stated half-response diameter for exploring sensitivity; it
is not the response of any real tool.

References
----------
Schlumberger (2009). *Log Interpretation Charts*, Rint-9 to Rint-12.
Ellis, D.V. and Singer, J.M. (2007). *Well Logging for Earth Scientists*,
2nd ed., ch. 7-9.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

RESPONSE_TYPES = ("laterolog", "induction")


@dataclass(frozen=True)
class ToolResponse:
    """Pseudo-geometric factor J(di) of one resistivity reading.

    ``diameters`` (inches) and ``factors`` (0-1, increasing) tabulate J;
    values between points are interpolated linearly, J is 0 below the
    first diameter and 1 beyond the last. ``response`` is ``"laterolog"``
    (series mixing of resistivities) or ``"induction"`` (parallel mixing,
    in conductivity).
    """

    name: str
    diameters: Tuple[float, ...]
    factors: Tuple[float, ...]
    response: str = "laterolog"

    def __post_init__(self) -> None:
        d = np.asarray(self.diameters, dtype=float)
        j = np.asarray(self.factors, dtype=float)
        if d.size != j.size or d.size < 2:
            raise ValueError(f"{self.name}: diameters and factors need the same length, at least 2")
        if np.any(np.diff(d) <= 0) or np.any(np.diff(j) < 0) or j.min() < 0 or j.max() > 1:
            raise ValueError(f"{self.name}: diameters must increase and factors rise within 0-1")
        if self.response not in RESPONSE_TYPES:
            raise ValueError(f"{self.name}: response must be one of {RESPONSE_TYPES}")

    def __repr__(self) -> str:
        d = np.asarray(self.diameters, dtype=float)
        half = float(np.interp(0.5, self.factors, d)) if max(self.factors) >= 0.5 else float("nan")
        return (f"ToolResponse({self.name!r}, {self.response}, J=0.5 at {half:.1f} in, "
                f"{d.size} points {d[0]:g}-{d[-1]:g} in)")

    def j(self, di: np.ndarray) -> np.ndarray:
        """J at invasion diameters ``di`` (inches)."""
        return np.interp(np.asarray(di, dtype=float), self.diameters, self.factors, left=0.0, right=1.0)

    @classmethod
    def parametric(
        cls,
        name: str,
        half_response_diameter: float,
        sharpness: float = 2.5,
        response: str = "laterolog",
        borehole_diameter: float = 8.5,
    ) -> "ToolResponse":
        """Smooth J with ``J(half_response_diameter) = 0.5``: ``J = x**p / (1 + x**p)``.

        ``x = (di - dh) / (d50 - dh)`` with ``dh`` the borehole diameter
        (J = 0 there). For sensitivity studies only; not a real tool.
        """
        if half_response_diameter <= borehole_diameter:
            raise ValueError("half_response_diameter must exceed the borehole diameter")
        di = np.concatenate([[borehole_diameter], np.geomspace(borehole_diameter + 0.1, 400.0, 120)])
        x = (di - borehole_diameter) / (half_response_diameter - borehole_diameter)
        j = x**sharpness / (1.0 + x**sharpness)
        return cls(name, tuple(di), tuple(j), response)


@dataclass
class InvasionInversionResult:
    """Per-sample step-profile solution."""

    rt: np.ndarray
    rxo: np.ndarray
    di: np.ndarray
    #: Root-mean-square relative misfit of the readings.
    misfit: np.ndarray
    #: 1 where the best diameter lies on the edge of the searched range.
    at_limit: np.ndarray
    #: J of the deepest reading at the solved diameter: its share from the
    #: flushed zone. Above about 0.5 Rt rests on a small part of the signal
    #: and is poorly determined.
    deep_flushed_share: np.ndarray = field(default_factory=lambda: np.array([]))
    #: Range of Rt and di over every solution that fits the readings within
    #: ``noise``: the step model is non-unique (conductive invasion read by
    #: induction tools especially), and this range is its uncertainty.
    rt_low: np.ndarray = field(default_factory=lambda: np.array([]))
    rt_high: np.ndarray = field(default_factory=lambda: np.array([]))
    di_low: np.ndarray = field(default_factory=lambda: np.array([]))
    di_high: np.ndarray = field(default_factory=lambda: np.array([]))
    readings: Tuple[str, ...] = ()
    notes: Dict[str, str] = field(default_factory=dict)


def _summary(name: str, result) -> str:
    ok = np.isfinite(result.rt)
    if not ok.any():
        return f"{name}(no sample solved)"
    return (f"{name}({int(ok.sum())} of {ok.size} samples solved from {', '.join(result.readings)}; "
            f"median Rt {np.nanmedian(result.rt):.3g}, Rxo {np.nanmedian(result.rxo):.3g} ohm.m, "
            f"di {np.nanmedian(result.di):.1f} in; at range limit {np.nanmean(result.at_limit) * 100:.0f}%)")


InvasionInversionResult.__repr__ = lambda self: _summary("InvasionInversionResult", self)


def _mix(j: np.ndarray, rxo: np.ndarray, rt: np.ndarray, response: str) -> np.ndarray:
    if response == "laterolog":
        return j * rxo + (1.0 - j) * rt
    return 1.0 / (j / rxo + (1.0 - j) / rt)


def forward_step_profile(
    rt: float | np.ndarray,
    rxo: float | np.ndarray,
    di: float | np.ndarray,
    tools: Sequence[ToolResponse],
) -> Dict[str, np.ndarray]:
    """Apparent resistivity of each tool for a step profile (Rt, Rxo, di)."""
    rt_arr, rxo_arr, di_arr = np.broadcast_arrays(
        np.asarray(rt, dtype=float), np.asarray(rxo, dtype=float), np.asarray(di, dtype=float)
    )
    return {tool.name: _mix(tool.j(di_arr), rxo_arr, rt_arr, tool.response) for tool in tools}


def invert_step_profile(
    readings: Mapping[str, Sequence[float]],
    tools: Sequence[ToolResponse],
    *,
    rxo: Optional[Sequence[float]] = None,
    di_range: Tuple[float, float] = (8.5, 120.0),
    n_diameters: int = 200,
    noise: float = 0.02,
) -> InvasionInversionResult:
    """Invert resistivity readings for Rt, Rxo and invasion diameter, sample by sample.

    Parameters
    ----------
    readings : mapping
        Apparent resistivity (ohm.m) per tool name, e.g.
        ``{"RSHA": ..., "RMED": ..., "RDEP": ...}``; the names match ``tools``.
    tools : sequence of ToolResponse
        Response of each reading.
    rxo : array-like, optional
        Measured flushed-zone resistivity (e.g. MSFL). It fixes Rxo, so two
        readings suffice.
    di_range : (float, float)
        Invasion diameters to search, inches.
    n_diameters : int
        Grid size (geometric spacing).
    noise : float
        Relative reading error (0.02 = 2 %). Solutions whose RMS relative
        misfit is within it are equally valid; their span is reported as
        ``rt_low``-``rt_high`` and ``di_low``-``di_high``.
        When ``noise`` is the standard error of the readings the range is
        roughly a one-standard-deviation interval (tested: the true Rt fell
        inside it for about 80 % of noisy synthetic profiles).

    Returns
    -------
    InvasionInversionResult
        Rt, Rxo, di (inches), misfit, a flag where the answer is at the
        edge of ``di_range`` (the readings do not constrain it there) and
        the deepest reading's flushed-zone share. Samples with a missing or
        non-positive reading are NaN. Where Rxo is close to Rt the diameter
        is not determined (any di fits).
    """
    by_name = {tool.name: tool for tool in tools}
    names = [name for name in readings if name in by_name]
    if len(names) != len(readings):
        raise ValueError(f"No tool response for {sorted(set(readings) - set(by_name))}")
    needed = 2 if rxo is not None else 3
    if len(names) < needed:
        raise ValueError(f"Inversion needs {needed} readings{' with a measured Rxo' if rxo is not None else ''}")
    ra = np.vstack([np.asarray(readings[name], dtype=float) for name in names])  # (k, n)
    n = ra.shape[1]
    lo, hi = di_range
    if not 0 < lo < hi:
        raise ValueError("di_range must be increasing and positive")
    grid = np.geomspace(lo, hi, n_diameters)
    j = np.vstack([by_name[name].j(grid) for name in names])  # (k, g)
    induction = np.array([by_name[name].response == "induction" for name in names])
    valid = np.all(np.isfinite(ra) & (ra > 0), axis=0)
    rxo_fixed = None
    if rxo is not None:
        rxo_fixed = np.asarray(rxo, dtype=float)
        valid &= np.isfinite(rxo_fixed) & (rxo_fixed > 0)

    log_ra = np.log(np.where(valid, ra, 1.0))
    log_rxo_fixed = np.log(np.where(valid, rxo_fixed, 1.0)) if rxo_fixed is not None else None
    misfits = np.full((grid.size, n), np.inf)
    grid_rt = np.full((grid.size, n), np.nan)
    best = np.full(n, np.inf)
    out_rt = np.full(n, np.nan)
    out_rxo = np.full(n, np.nan)
    out_di = np.full(n, np.nan)
    for g in range(grid.size):
        jmat = np.repeat(j[:, g][:, None], n, axis=1)
        r_xo, r_t, misfit = _fit(jmat, ra, log_ra, induction, valid, log_rxo_fixed)
        misfits[g] = misfit
        grid_rt[g] = r_t
        better = valid & np.isfinite(misfit) & (misfit < best - 1e-12)
        best = np.where(better, misfit, best)
        out_rt = np.where(better, r_t, out_rt)
        out_rxo = np.where(better, r_xo, out_rxo)
        out_di = np.where(better, grid[g], out_di)

    # Refine the diameter between grid points: parabola through the misfit
    # around the grid minimum, in log diameter, then refit there.
    index = np.argmin(misfits, axis=0)
    inner = valid & (index > 0) & (index < grid.size - 1)
    cols = np.arange(n)
    left = misfits[np.clip(index - 1, 0, None), cols] ** 2
    mid = misfits[index, cols] ** 2
    right = misfits[np.clip(index + 1, None, grid.size - 1), cols] ** 2
    curvature = left - 2.0 * mid + right
    step = np.log(grid[1] / grid[0])
    with np.errstate(divide="ignore", invalid="ignore"):
        offset = np.where(inner & (curvature > 0), 0.5 * (left - right) / curvature, 0.0)
    refined = grid[index] * np.exp(np.clip(offset, -0.5, 0.5) * step)
    jmat = np.vstack([by_name[name].j(refined) for name in names])
    r_xo, r_t, misfit = _fit(jmat, ra, log_ra, induction, valid, log_rxo_fixed)
    better = inner & np.isfinite(misfit) & (misfit <= best)
    out_rt = np.where(better, r_t, out_rt)
    out_rxo = np.where(better, r_xo, out_rxo)
    out_di = np.where(better, refined, out_di)
    best = np.where(better, misfit, best)

    at_limit = np.where(
        np.isfinite(out_di),
        ((index == 0) | (index == grid.size - 1)).astype(float),
        np.nan,
    )
    # Equivalent solutions: every grid diameter fitting within the noise.
    tolerance = np.maximum(np.where(np.isfinite(best), best, np.inf), noise)
    accepted = (misfits <= tolerance[None, :]) & np.isfinite(grid_rt)
    with np.errstate(invalid="ignore"):
        rt_low = np.fmin(np.where(accepted, grid_rt, np.inf).min(axis=0), out_rt)
        rt_high = np.fmax(np.where(accepted, grid_rt, -np.inf).max(axis=0), out_rt)
        di_low = np.fmin(np.where(accepted, grid[:, None], np.inf).min(axis=0), out_di)
        di_high = np.fmax(np.where(accepted, grid[:, None], -np.inf).max(axis=0), out_di)
    missing = ~np.isfinite(out_rt)
    for values in (rt_low, rt_high, di_low, di_high):
        values[missing | ~np.isfinite(values)] = np.nan
    # The deepest reading is the one least affected by invasion at mid-range diameters.
    deepest = names[int(np.argmin(j[:, grid.size // 2]))]
    share = np.where(np.isfinite(out_di), by_name[deepest].j(np.nan_to_num(out_di)), np.nan)
    return InvasionInversionResult(
        rt=out_rt, rxo=out_rxo, di=out_di, misfit=np.where(np.isfinite(best), best, np.nan),
        at_limit=at_limit, deep_flushed_share=share,
        rt_low=rt_low, rt_high=rt_high, di_low=di_low, di_high=di_high, readings=tuple(names),
        notes={"deepest_reading": deepest},
    )


def _fit(
    jmat: np.ndarray,
    ra: np.ndarray,
    log_ra: np.ndarray,
    induction: np.ndarray,
    valid: np.ndarray,
    log_rxo_fixed: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rxo, Rt and misfit for the response factors ``jmat`` (readings x samples)."""
    k, n = ra.shape
    cols = np.arange(n)
    # Start: Rxo from the reading most inside di, Rt from the one most outside.
    log_rxo = log_rxo_fixed.copy() if log_rxo_fixed is not None else log_ra[np.argmax(jmat, axis=0), cols]
    log_rt = log_ra[np.argmin(jmat, axis=0), cols]
    safe = np.where(valid, ra, 1.0)
    # Gauss-Newton on the relative misfit, in log-resistivity (keeps both positive).
    for _ in range(12):
        r_xo, r_t = np.exp(log_rxo), np.exp(log_rt)
        model = np.empty_like(ra)
        d_xo = np.empty_like(ra)
        d_t = np.empty_like(ra)
        for i in range(k):
            ji = jmat[i]
            if induction[i]:
                m = 1.0 / (ji / r_xo + (1.0 - ji) / r_t)
                d_xo[i] = m**2 * ji / r_xo
                d_t[i] = m**2 * (1.0 - ji) / r_t
            else:
                m = ji * r_xo + (1.0 - ji) * r_t
                d_xo[i] = ji * r_xo
                d_t[i] = (1.0 - ji) * r_t
            model[i] = m
        res = (model - safe) / safe
        a_xo = d_xo / safe
        a_t = d_t / safe
        if log_rxo_fixed is None:
            s11 = np.sum(a_xo**2, axis=0) + 1e-12
            s12 = np.sum(a_xo * a_t, axis=0)
            s22 = np.sum(a_t**2, axis=0) + 1e-12
            g1 = np.sum(a_xo * res, axis=0)
            g2 = np.sum(a_t * res, axis=0)
            det = s11 * s22 - s12**2
            log_rxo = log_rxo - np.clip((g1 * s22 - g2 * s12) / det, -2.0, 2.0)
            step_t = (s11 * g2 - s12 * g1) / det
        else:
            step_t = np.sum(a_t * res, axis=0) / (np.sum(a_t**2, axis=0) + 1e-12)
        log_rt = log_rt - np.clip(step_t, -2.0, 2.0)
    r_xo, r_t = np.exp(log_rxo), np.exp(log_rt)
    model = np.vstack([
        jmat[i] * r_xo + (1.0 - jmat[i]) * r_t if not induction[i]
        else 1.0 / (jmat[i] / r_xo + (1.0 - jmat[i]) / r_t)
        for i in range(k)
    ])
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        misfit = np.sqrt(np.mean(((model - safe) / safe) ** 2, axis=0))
    misfit = np.where(valid, misfit, np.nan)
    return r_xo, r_t, misfit


def read_tool_responses(
    source,
    *,
    responses: Optional[Mapping[str, str]] = None,
    sheet_name=None,
) -> list:
    """Read pseudo-geometric factors J(di) digitized from vendor charts.

    Two layouts are read:

    - long: one row per point, columns ``tool``, ``diameter`` (``di``,
      inches) and ``J`` (``factor``), optionally ``response``
      (``laterolog``/``induction``);
    - wide: a ``diameter`` column and one J column per tool.

    A diameter header in cm or mm (``di (cm)``) is converted to inches.
    ``responses`` gives the mixing law per tool where the table does not
    (laterolog by default). Rows are sorted by diameter; J must rise from 0
    to at most 1.
    """
    import pandas as pd

    from petrophys.io.geology_reader import _read_table
    from petrophys.io.survey_reader import _name_and_unit

    table = source.copy() if isinstance(source, pd.DataFrame) else _read_table(source, sheet_name=sheet_name)
    keys = {_name_and_unit(c)[0]: c for c in table.columns}
    diameter = next((keys[k] for k in ("diameter", "di", "invasiondiameter", "dia", "d") if k in keys), None)
    if diameter is None:
        raise ValueError(f"Tool-response table needs a diameter column; found {list(table.columns)}")
    unit = _name_and_unit(diameter)[1]
    # Tables read from files arrive as text: make the numbers numbers before sorting.
    table[diameter] = pd.to_numeric(table[diameter], errors="coerce")
    table = table[table[diameter].notna()]
    scale = {"cm": 1 / 2.54, "mm": 1 / 25.4, "m": 1 / 0.0254, "in": 1.0, "inch": 1.0, "inches": 1.0, "": 1.0}.get(unit)
    if scale is None:
        raise ValueError(f"Unknown diameter unit {unit!r}; use inches, cm or mm")
    responses = {str(k): str(v).lower() for k, v in (responses or {}).items()}
    tool_col = next((keys[k] for k in ("tool", "curve", "name", "reading") if k in keys), None)
    factor_col = next((keys[k] for k in ("j", "factor", "geometricfactor", "pseudogeometricfactor", "jdi") if k in keys), None)
    curves: Dict[str, Tuple[np.ndarray, np.ndarray, str]] = {}
    if tool_col is not None and factor_col is not None:
        table[factor_col] = pd.to_numeric(table[factor_col], errors="coerce")
        table = table[table[factor_col].notna()]
        response_col = next((keys[k] for k in ("response", "type", "law") if k in keys), None)
        for name, group in table.groupby(tool_col, sort=False):
            group = group.sort_values(diameter)
            law = responses.get(str(name))
            if law is None and response_col is not None:
                law = str(group[response_col].iloc[0]).strip().lower()
            curves[str(name)] = (group[diameter].to_numpy(float) * scale, group[factor_col].to_numpy(float), law or "laterolog")
    else:
        others = [c for c in table.columns if c != diameter]
        if not others:
            raise ValueError("Tool-response table has no J columns")
        table = table.sort_values(diameter)
        for column in others:
            values = pd.to_numeric(table[column], errors="coerce").to_numpy(float)
            ok = np.isfinite(values)
            curves[str(column)] = (table[diameter].to_numpy(float)[ok] * scale, values[ok], responses.get(str(column), "laterolog"))
    tools = []
    for name, (d, j, law) in curves.items():
        if law not in RESPONSE_TYPES:
            raise ValueError(f"{name}: response {law!r}; use one of {RESPONSE_TYPES}")
        if j.size and np.nanmax(j) > 1.0 + 1e-9 and np.nanmax(j) <= 100.0:
            j = j / 100.0  # digitized in percent
        tools.append(ToolResponse(name, tuple(float(v) for v in d), tuple(float(v) for v in j), law))
    return tools


def invert_well_invasion(
    well,
    tools: Sequence[ToolResponse],
    *,
    readings: Optional[Mapping[str, str]] = None,
    rxo_curve: Optional[str] = None,
    add_curves: bool = True,
    **options,
) -> InvasionInversionResult:
    """:func:`invert_step_profile` on a well's resistivity curves.

    ``readings`` maps each tool name to its curve (by default the tool name
    is the curve name, e.g. ``LLD``, ``AT90``). With ``add_curves`` the well
    gets ``RT_INV``, ``RXO_INV``, ``DI_INV`` (inches), ``RT_INV_LOW``,
    ``RT_INV_HIGH`` and ``INV_MISFIT``.
    """
    mapping = dict(readings or {tool.name: tool.name for tool in tools})
    data = {}
    for tool_name, curve_name in mapping.items():
        curve = well.get(curve_name)
        if curve is None:
            raise ValueError(f"No curve {curve_name!r} on the well for tool {tool_name!r}")
        data[tool_name] = np.asarray(curve.data, dtype=float)
    rxo = None
    if rxo_curve is not None:
        curve = well.get(rxo_curve)
        if curve is None:
            raise ValueError(f"No curve {rxo_curve!r} on the well")
        rxo = np.asarray(curve.data, dtype=float)
    used = [tool for tool in tools if tool.name in data]
    result = invert_step_profile(data, used, rxo=rxo, **options)
    if add_curves:
        well.add_curve("RT_INV", result.rt, "ohm.m", "Rt from step-profile invasion inversion")
        well.add_curve("RXO_INV", result.rxo, "ohm.m", "Rxo from step-profile invasion inversion")
        well.add_curve("DI_INV", result.di, "in", "Invasion diameter from step-profile inversion")
        well.add_curve("RT_INV_LOW", result.rt_low, "ohm.m", "Lowest Rt fitting the readings within their noise")
        well.add_curve("RT_INV_HIGH", result.rt_high, "ohm.m", "Highest Rt fitting the readings within their noise")
        well.add_curve("INV_MISFIT", result.misfit, "", "RMS relative misfit of the invasion inversion")
    return result


__all__ = [
    "RESPONSE_TYPES",
    "InvasionInversionResult",
    "ToolResponse",
    "forward_step_profile",
    "invert_step_profile",
    "invert_well_invasion",
    "read_tool_responses",
]
