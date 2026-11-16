"""Deviation surveys, true vertical depth and the depth datum of a well.

Saturation-height functions, free-water levels and pore pressure all work in
true vertical depth (TVD), usually below sea level (TVDSS), while logs are
recorded in measured depth (MD) from a rig reference (kelly bushing, drill
floor or ground level). This module closes that gap:

- :func:`read_survey` reads a deviation survey (MD, inclination, azimuth)
  from CSV, Excel, LAS or a DataFrame, finding the columns by name;
- :func:`minimum_curvature` computes TVD and the horizontal offsets;
- :func:`depth_reference_elevation` reads the elevation of the log's depth
  reference from the LAS header (EKB/EDF/EGL by LMF, or APD + EPD);
- :func:`attach_survey` adds ``TVD`` (below the depth reference) and
  ``TVDSS`` (below sea level, positive down) curves to a well.

``pp.read(path, survey=...)`` runs :func:`attach_survey` on reading.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

_FT_PER_M = 1.0 / 0.3048

_MD = ("md", "measureddepth", "depth", "dept", "mdrkb", "mdkb", "mddf", "mdbrt")
_INC = ("inc", "incl", "inclination", "dev", "devi", "deviation", "angle")
_AZI = ("azi", "azim", "azimuth", "hazi", "azimuthtrue", "azimuthgrid", "dir", "direction")
_TVD = ("tvd", "tvdrkb", "tvdkb", "truevertical", "truevenicaldepth", "truverticaldepth")


@dataclass
class Survey:
    """A deviation survey with its minimum-curvature trajectory."""

    md: np.ndarray
    inclination: np.ndarray
    azimuth: np.ndarray
    tvd: np.ndarray
    north: np.ndarray
    east: np.ndarray
    #: Dogleg severity: degrees per 30 m, or per 100 ft for a survey in feet
    #: (:func:`attach_survey` rescales it when it converts the unit).
    dogleg: np.ndarray
    depth_unit: str = ""
    report: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (f"Survey({self.md.size} stations, MD {self.md[0]:g}-{self.md[-1]:g} {self.depth_unit}, "
                f"TVD at end {self.tvd[-1]:.1f}, max inclination {np.max(self.inclination):.1f} deg)")

    def to_dataframe(self) -> pd.DataFrame:
        """The survey as a table, one row per station.

        Returns
        -------
        DataFrame
            ``md``, ``tvd``, ``north`` and ``east`` in ``depth_unit``,
            ``inclination`` and ``azimuth`` in degrees, and ``dogleg``
            (degrees per 30 m, or per 100 ft for a survey in feet).
        """
        return pd.DataFrame({
            "md": self.md, "inclination": self.inclination, "azimuth": self.azimuth,
            "tvd": self.tvd, "north": self.north, "east": self.east, "dogleg": self.dogleg,
        })

    def tvd_at(self, md: Any) -> np.ndarray:
        """TVD at measured depths ``md`` (between stations, along the minimum-curvature arc)."""
        return _tvd_between_stations(self, np.asarray(md, dtype=float))


def _name_and_unit(column: Any) -> Tuple[str, str]:
    text = str(column)
    match = re.match(r"^\s*(.*?)\s*[\(\[]\s*([^\)\]]*)\s*[\)\]]\s*$", text)
    if match:
        return re.sub(r"[^a-z0-9]", "", match.group(1).lower()), match.group(2).strip().lower()
    return re.sub(r"[^a-z0-9]", "", text.lower()), ""


def _depth_unit_from(text: str) -> Optional[str]:
    text = str(text or "").strip().lower().rstrip(".")
    if text in {"m", "meter", "meters", "metre", "metres"}:
        return "m"
    if text in {"ft", "f", "feet", "foot", "usft", "ft(us)"}:
        return "ft"
    return None


def _course_length(unit: Any) -> float:
    """The length dogleg severity is quoted over: 100 ft for feet, else 30 m."""
    return 100.0 if str(unit or "").strip().lower() in {"ft", "feet", "f"} else 30.0


def minimum_curvature(md: Any, inclination: Any, azimuth: Any,
                      course_length: float = 30.0) -> Dict[str, np.ndarray]:
    """True vertical depth and horizontal offsets by the minimum-curvature method.

    Angles in degrees; TVD, north and east in the unit of ``md``. When the
    first station is below MD 0, the hole above it is taken as straight at
    its inclination; north and east start at zero at the first station.
    Dogleg severity is in degrees per ``course_length`` units of ``md``
    (30 for metres; :func:`read_survey` uses 100 for a survey in feet).

    Examples
    --------
    >>> traj = minimum_curvature([0, 1000], [0, 0], [0, 0])
    >>> float(traj["tvd"][-1])
    1000.0
    """
    md = np.asarray(md, dtype=float)
    inc = np.radians(np.asarray(inclination, dtype=float))
    azi = np.radians(np.asarray(azimuth, dtype=float))
    n = md.size
    tvd = np.zeros(n)
    north = np.zeros(n)
    east = np.zeros(n)
    dogleg = np.zeros(n)
    if n == 0:
        return {"tvd": tvd, "north": north, "east": east, "dogleg": dogleg}
    tvd[0] = md[0] * np.cos(inc[0]) if md[0] > 0 else 0.0
    for i in range(1, n):
        dmd = md[i] - md[i - 1]
        cos_dl = np.cos(inc[i] - inc[i - 1]) - np.sin(inc[i - 1]) * np.sin(inc[i]) * (1 - np.cos(azi[i] - azi[i - 1]))
        beta = np.arccos(np.clip(cos_dl, -1.0, 1.0))
        rf = 1.0 if beta < 1e-9 else 2.0 / beta * np.tan(beta / 2.0)
        half = 0.5 * dmd * rf
        tvd[i] = tvd[i - 1] + half * (np.cos(inc[i - 1]) + np.cos(inc[i]))
        north[i] = north[i - 1] + half * (np.sin(inc[i - 1]) * np.cos(azi[i - 1]) + np.sin(inc[i]) * np.cos(azi[i]))
        east[i] = east[i - 1] + half * (np.sin(inc[i - 1]) * np.sin(azi[i - 1]) + np.sin(inc[i]) * np.sin(azi[i]))
        dogleg[i] = np.degrees(beta) / dmd * course_length if dmd > 0 else 0.0
    return {"tvd": tvd, "north": north, "east": east, "dogleg": dogleg}


def _tvd_between_stations(survey: Survey, md: np.ndarray) -> np.ndarray:
    """TVD at arbitrary MD: minimum-curvature arc from the station above."""
    out = np.full(md.shape, np.nan)
    s_md = survey.md
    inc = np.radians(survey.inclination)
    azi = np.radians(survey.azimuth)
    finite = np.isfinite(md)
    # Above the first station: straight at the first station's inclination.
    above = finite & (md <= s_md[0])
    out[above] = survey.tvd[0] - (s_md[0] - md[above]) * np.cos(inc[0])
    below = finite & (md >= s_md[-1])
    out[below] = survey.tvd[-1] + (md[below] - s_md[-1]) * np.cos(inc[-1])
    inside = finite & ~above & ~below
    if inside.any():
        idx = np.clip(np.searchsorted(s_md, md[inside], side="right") - 1, 0, s_md.size - 2)
        i0, i1 = idx, idx + 1
        cos_dl = np.cos(inc[i1] - inc[i0]) - np.sin(inc[i0]) * np.sin(inc[i1]) * (1 - np.cos(azi[i1] - azi[i0]))
        beta = np.arccos(np.clip(cos_dl, -1.0, 1.0))
        frac = (md[inside] - s_md[i0]) / (s_md[i1] - s_md[i0])
        # Interpolate the direction along the arc (slerp), then integrate the
        # vertical component of the partial arc.
        with np.errstate(divide="ignore", invalid="ignore"):
            t0 = np.stack([np.sin(inc[i0]) * np.cos(azi[i0]), np.sin(inc[i0]) * np.sin(azi[i0]), np.cos(inc[i0])])
            t1 = np.stack([np.sin(inc[i1]) * np.cos(azi[i1]), np.sin(inc[i1]) * np.sin(azi[i1]), np.cos(inc[i1])])
            small = beta < 1e-9
            sb = np.where(small, 1.0, np.sin(beta))
            w0 = np.where(small, 1.0 - frac, np.sin((1 - frac) * beta) / sb)
            w1 = np.where(small, frac, np.sin(frac * beta) / sb)
            tf = w0 * t0 + w1 * t1
            beta_p = frac * beta
            rf = np.where(beta_p < 1e-9, 1.0, 2.0 / np.where(beta_p < 1e-9, 1.0, beta_p) * np.tan(beta_p / 2.0))
        dmd = md[inside] - s_md[i0]
        out[inside] = survey.tvd[i0] + 0.5 * dmd * rf * (t0[2] + tf[2])
    return out


def read_survey(
    source: Any,
    *,
    columns: Optional[Dict[str, str]] = None,
    depth_unit: Optional[str] = None,
    sheet_name: Any = None,
) -> Survey:
    """Read a deviation survey and compute its trajectory.

    Parameters
    ----------
    source : path, DataFrame or Survey
        CSV/TSV/Excel table, a LAS file with MD, inclination and azimuth
        curves, or a DataFrame. Columns are found by name, ignoring case,
        spaces and units in brackets: MD (``MD``, ``Measured Depth``,
        ``DEPTH``), inclination (``INC``, ``Inclination``, ``DEVI``) and
        azimuth (``AZI``, ``Azimuth``, ``HAZI``).
    columns : dict, optional
        Explicit ``{"md": ..., "inclination": ..., "azimuth": ...}`` headers.
    depth_unit : str, optional
        ``"m"`` or ``"ft"``; read from a header such as ``MD (ft)`` or the
        LAS depth unit when not given.
    sheet_name : optional
        Excel sheet.

    Returns
    -------
    Survey
        Stations sorted by MD, with minimum-curvature TVD, north, east and
        dogleg severity, and a ``report`` of the columns and checks.
    """
    if isinstance(source, Survey):
        return source
    report: Dict[str, Any] = {"warnings": []}
    unit_hint = None
    if isinstance(source, (str, Path)) and Path(source).suffix.lower() == ".las":
        from petrophys.io.las_reader import _read_lasio

        las = _read_lasio(Path(source), None)
        table = las.df().reset_index()
        unit_hint = _depth_unit_from(las.curves[0].unit)
    elif isinstance(source, pd.DataFrame):
        table = source.copy()
    else:
        from petrophys.io.geology_reader import _read_table

        table = _read_table(source, sheet_name=sheet_name)
    found: Dict[str, Any] = {}
    units: Dict[str, str] = {}
    for column in table.columns:
        key, unit = _name_and_unit(column)
        for role, names in (("md", _MD), ("inclination", _INC), ("azimuth", _AZI), ("tvd", _TVD)):
            if key in names and role not in found:
                found[role] = column
                units[role] = unit
                break
    for role, header in (columns or {}).items():
        if header not in table.columns:
            raise ValueError(f"Survey column {header!r} not found; columns are {list(table.columns)}")
        found[role] = header
        units[role] = _name_and_unit(header)[1]
    if "azimuth" not in found and "md" in found and "inclination" in found:
        # Inclination-only surveys (e.g. a log DEVI curve) are common; TVD is
        # exact for a well in one vertical plane and close for small turns.
        table = table.assign(__azimuth=0.0)
        found["azimuth"] = "__azimuth"
        report["warnings"].append(
            "No azimuth column: azimuth taken as constant, exact for a well in one vertical plane"
        )
    missing = [role for role in ("md", "inclination", "azimuth") if role not in found]
    if missing:
        raise ValueError(
            f"Survey needs MD, inclination and azimuth columns; could not find {missing} "
            f"among {list(table.columns)}. Name them with columns={{...}}."
        )
    data = pd.DataFrame({role: pd.to_numeric(table[found[role]], errors="coerce") for role in ("md", "inclination", "azimuth")})
    data = data.dropna().sort_values("md").drop_duplicates("md")
    if len(data) < 2:
        raise ValueError("Survey needs at least two stations with MD, inclination and azimuth")
    unit = (
        _depth_unit_from(depth_unit) if depth_unit else _depth_unit_from(units.get("md", "")) or unit_hint
    )
    if depth_unit and unit is None:
        raise ValueError("depth_unit must be 'm' or 'ft'")
    if unit is None:
        report["warnings"].append("Survey depth unit not stated; taken as the well's depth unit")
    inc = data["inclination"].to_numpy()
    azi = data["azimuth"].to_numpy() % 360.0
    if np.any((inc < 0) | (inc > 180)):
        raise ValueError("Inclination must be between 0 and 180 degrees")
    if np.nanmax(inc) <= np.pi / 2 + 1e-6 and np.nanmax(inc) > 0 and np.nanmax(azi) <= 2 * np.pi + 1e-6:
        report["warnings"].append(
            "All inclinations and azimuths are below 1.6 and 6.3: they may be radians; degrees are expected"
        )
    md = data["md"].to_numpy()
    if md[0] > 0:
        report["warnings"].append(
            f"First station at MD {md[0]:g}: the hole above is taken as straight at its inclination"
        )
    trajectory = minimum_curvature(md, inc, azi, course_length=_course_length(unit))
    report.update({"columns": {role: str(found[role]) for role in found if found[role] != "__azimuth"},
                   "stations": int(md.size),
                   "max_inclination": float(np.max(inc)), "max_dogleg": float(np.max(trajectory["dogleg"]))})
    if "tvd" in found:
        given = pd.to_numeric(table.loc[data.index, found["tvd"]], errors="coerce").to_numpy()
        ok = np.isfinite(given)
        if ok.any():
            difference = float(np.max(np.abs(given[ok] - trajectory["tvd"][ok])))
            report["tvd_check"] = difference
            if difference > max(1.0, 0.001 * float(md[-1])):
                report["warnings"].append(
                    f"TVD in the file differs from minimum curvature by up to {difference:.1f}: "
                    "check the tie-in, the depth reference or the unit"
                )
    for message in report["warnings"]:
        warnings.warn(f"Survey: {message}", UserWarning, stacklevel=2)
    return Survey(
        md=md, inclination=inc, azimuth=azi, tvd=trajectory["tvd"], north=trajectory["north"],
        east=trajectory["east"], dogleg=trajectory["dogleg"], depth_unit=unit or "", report=report,
    )


def _item(well: Any, name: str) -> Tuple[Any, str]:
    """A LAS header item's value and unit, from ~Well or ~Parameter."""
    items = getattr(well.header, "las_well_items", None) or {}
    if name in items:
        return items[name].get("value"), str(items[name].get("unit") or "")
    get = getattr(well, "get_parameter", None)
    if callable(get):
        value = get(name)
        if value is not None:
            return value, str(get(f"{name}__UNIT") or "")
    return None, ""


def _number(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) and number > -999.0 else None


def depth_reference_elevation(well: Any) -> Tuple[Optional[float], Dict[str, Any]]:
    """Elevation of the log's zero depth above sea level, in the well's depth unit.

    From the LAS header: ``APD`` (depth reference above the permanent
    datum) plus ``EPD`` (permanent datum above sea level, 0 when ``PDAT`` is
    MSL); otherwise the elevation named by ``LMF`` (``EKB``, ``EDF`` or
    ``EGL``), else a nonzero ``EKB`` or ``EDF``. Units in the header are
    converted. Returns the elevation (None when the header has none) and a
    note of its source.
    """
    depth_unit = str(getattr(well, "depth_unit", "") or "")

    def converted(value: Any, unit: str) -> Optional[float]:
        number = _number(value)
        if number is None:
            return None
        source_unit = _depth_unit_from(unit) or depth_unit
        if source_unit == depth_unit or not source_unit:
            return number
        return number * _FT_PER_M if (source_unit, depth_unit) == ("m", "ft") else number * 0.3048

    lmf = str(_item(well, "LMF")[0] or "").strip().upper()
    apd, apd_unit = _item(well, "APD")
    if _number(apd) is not None:
        epd, epd_unit = _item(well, "EPD")
        pdat = str(_item(well, "PDAT")[0] or "").upper()
        base = converted(epd, epd_unit) or 0.0
        note = f"APD {apd} {apd_unit} + EPD {epd if epd is not None else 0}" + (f" ({pdat})" if pdat else "")
        return converted(apd, apd_unit) + base, {"source": note, "reference": lmf or "APD"}
    by_reference = {"KB": "EKB", "KELLYBUSHING": "EKB", "RKB": "EKB", "DF": "EDF", "DRILLFLOOR": "EDF",
                    "DRILLINGFLOOR": "EDF", "RT": "EKB", "GL": "EGL", "GROUNDLEVEL": "EGL"}
    key = by_reference.get(re.sub(r"[^A-Z]", "", lmf))
    for name in ([key] if key else []) + ["EKB", "EDF"]:
        value, unit = _item(well, name)
        number = converted(value, unit)
        if number is not None and (number != 0.0 or name == key):
            return number, {"source": f"{name} {value} {unit}".strip(), "reference": lmf or name[1:]}
    return None, {"source": "no elevation in the header", "reference": lmf}


def attach_survey(
    well: Any,
    survey: Any,
    *,
    elevation: Optional[float] = None,
    columns: Optional[Dict[str, str]] = None,
    sheet_name: Any = None,
) -> Survey:
    """Add ``TVD`` and ``TVDSS`` curves to ``well`` from a deviation survey.

    Parameters
    ----------
    well : Well
        Well in measured depth.
    survey : path, DataFrame, Survey or ``"vertical"``
        The survey (see :func:`read_survey`), or ``"vertical"`` to declare
        the hole vertical (TVD = MD).
    elevation : float, optional
        Elevation of the log's depth reference above sea level, in the
        well's depth unit; read from the header
        (:func:`depth_reference_elevation`) when not given. Without one,
        only ``TVD`` is added.
    columns : dict, optional
        Explicit ``{"md": ..., "inclination": ..., "azimuth": ...}`` headers
        of a survey table, as for :func:`read_survey`.
    sheet_name : optional
        Sheet of an Excel survey workbook, as for :func:`read_survey`.

    Returns
    -------
    Survey
        The survey the curves were computed from.

    Notes
    -----
    ``TVD`` is below the log's depth reference; ``TVDSS`` = TVD - elevation,
    positive below sea level. The survey is converted to the well's depth
    unit. The source and elevation are recorded in the well parameters
    ``tvd_source`` and ``depth_reference_elevation``.
    """
    depth = np.asarray(well.depth, dtype=float)
    unit = str(well.depth_unit)
    if isinstance(survey, str) and survey.strip().lower() == "vertical":
        tvd = depth.copy()
        trajectory = Survey(md=np.array([0.0, float(np.nanmax(depth))]), inclination=np.zeros(2),
                            azimuth=np.zeros(2), tvd=np.array([0.0, float(np.nanmax(depth))]),
                            north=np.zeros(2), east=np.zeros(2), dogleg=np.zeros(2), depth_unit=unit,
                            report={"vertical": True})
        source = "declared vertical"
    else:
        trajectory = read_survey(survey, columns=columns, sheet_name=sheet_name)
        factor = 1.0
        if trajectory.depth_unit and trajectory.depth_unit != unit:
            factor = _FT_PER_M if (trajectory.depth_unit, unit) == ("m", "ft") else 0.3048
            trajectory = Survey(
                md=trajectory.md * factor, inclination=trajectory.inclination, azimuth=trajectory.azimuth,
                tvd=trajectory.tvd * factor, north=trajectory.north * factor, east=trajectory.east * factor,
                dogleg=trajectory.dogleg / factor * _course_length(unit) / _course_length(trajectory.depth_unit),
                depth_unit=unit, report=trajectory.report,
            )
        tvd = trajectory.tvd_at(depth)
        beyond = depth > trajectory.md[-1] + 1e-6
        if beyond.any():
            warnings.warn(
                f"Survey ends at MD {trajectory.md[-1]:g}; {int(beyond.sum())} log samples below it "
                "are extrapolated at the last inclination",
                UserWarning, stacklevel=2,
            )
        source = f"deviation survey, {trajectory.md.size} stations"
    well.add_curve("TVD", tvd, unit, "True vertical depth below the log depth reference")
    note = None
    if elevation is None:
        elevation, note = depth_reference_elevation(well)
    if elevation is not None:
        well.add_curve("TVDSS", tvd - float(elevation), unit, "True vertical depth below sea level")
        well.set_parameter("depth_reference_elevation", float(elevation))
    if note and elevation is not None:
        source += f"; elevation from {note['source']}"
    elif elevation is not None:
        source += f"; elevation {float(elevation):g} {unit} given"
    well.set_parameter("tvd_source", source)
    if elevation is None:
        warnings.warn(
            "No depth-reference elevation (EKB/EDF/APD in the header, or elevation=): only TVD was added, not TVDSS",
            UserWarning, stacklevel=2,
        )
    return trajectory


__all__ = [
    "Survey",
    "attach_survey",
    "depth_reference_elevation",
    "minimum_curvature",
    "read_survey",
]
