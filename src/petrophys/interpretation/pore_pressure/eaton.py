"""Pore pressure from sonic or resistivity by Eaton's method.

Shales compact with burial as long as their water can escape. Where it
cannot, they stay under-compacted: slower and more conductive than the
normal compaction trend (NCT) for their depth. Eaton (1975) turns that
departure into pore pressure:

- sonic:        ``Pp = S - (S - Ph) * (DTn / DT)**3``
- resistivity:  ``Pp = S - (S - Ph) * (R / Rn)**1.2``

with S the overburden (vertical stress), Ph the hydrostatic pressure and
DTn, Rn the normal-trend values at that depth. The inputs that set the
answer are yours to state: the overburden density profile (water depth, air
gap and the density above the logged interval), the hydrostatic water
density, and the NCT, fitted on clean shale in a normally pressured section
(:func:`fit_normal_compaction_trend`) or given directly. The method reads
shale: :func:`eaton_from_well` takes the shale samples as ``shale=`` and
leaves the pressure blank elsewhere. Eaton's exponents are regional and should be calibrated to
measured pressures (RFT/MDT, kicks, mud weights).

Depths are true vertical depth below the rig floor (TVD RKB) in m or ft;
pressures in bar, psi, kPa or MPa; densities in g/cc.

References
----------
Eaton, B.A. (1975). The equation for geopressure prediction from well logs.
SPE 5544.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

#: Pressure (in the unit) per g/cc of density per depth unit.
_GRADIENT = {
    ("m", "bar"): 0.0980665, ("m", "kPa"): 9.80665, ("m", "MPa"): 0.00980665,
    ("m", "psi"): 1.42233, ("ft", "psi"): 0.433527, ("ft", "bar"): 0.0298907,
    ("ft", "kPa"): 2.98907, ("ft", "MPa"): 0.00298907,
}
EATON_EXPONENTS = {"sonic": 3.0, "resistivity": 1.2}


def _gradient(depth_unit: str, pressure_unit: str) -> float:
    key = (str(depth_unit), str(pressure_unit))
    if key not in _GRADIENT:
        raise ValueError("Use depth_unit 'm' or 'ft' and pressure_unit 'bar', 'psi', 'kPa' or 'MPa'")
    return _GRADIENT[key]


def hydrostatic_pressure(
    tvd: Any,
    *,
    water_density: float = 1.03,
    air_gap: float = 0.0,
    depth_unit: str = "m",
    pressure_unit: str = "bar",
) -> np.ndarray:
    """Hydrostatic pore pressure of a water column from sea level (``air_gap`` below the rig floor).

    Onshore, with ``air_gap`` the depth reference's height above ground, the
    column starts at ground level. ``water_density`` is the formation-water
    density in g/cc (about 1.03 for sea water, more for saline brines).
    """
    z = np.asarray(tvd, dtype=float) - air_gap
    return np.where(z > 0, water_density * _gradient(depth_unit, pressure_unit) * z, 0.0)


def overburden_pressure(
    tvd: Any,
    rhob: Any,
    *,
    water_depth: float = 0.0,
    air_gap: float = 0.0,
    seawater_density: float = 1.03,
    density_above_log: Optional[float] = None,
    depth_unit: str = "m",
    pressure_unit: str = "bar",
) -> np.ndarray:
    """Vertical stress (overburden) by integrating density from the surface.

    The column is the air gap (no weight), sea water to the mudline at
    ``air_gap + water_depth``, sediment of ``density_above_log`` from the
    mudline to the first valid RHOB, then the bulk density log (gaps
    interpolated). ``tvd`` must increase.

    Parameters
    ----------
    tvd, rhob : array-like
        True vertical depth below the rig floor and bulk density (g/cc).
    water_depth, air_gap : float
        In the depth unit; both 0 onshore with the rig floor at ground level.
    seawater_density : float
        Density of the sea water above the mudline (g/cc); default 1.03.
    density_above_log : float, optional
        Mean sediment density between mudline and log top (g/cc). Required
        when the log starts more than 1 % of its depth below the mudline;
        typical shallow sediment averages 1.8-2.1.
    depth_unit : {"m", "ft"}
        Unit of ``tvd``, ``water_depth`` and ``air_gap``; default ``"m"``.
    pressure_unit : {"bar", "psi", "kPa", "MPa"}
        Unit of the returned stress; default ``"bar"``.
    """
    z = np.asarray(tvd, dtype=float)
    rho = np.asarray(rhob, dtype=float)
    if z.shape != rho.shape:
        raise ValueError("tvd and rhob need the same length")
    if np.any(np.diff(z[np.isfinite(z)]) <= 0):
        raise ValueError("tvd must increase")
    g = _gradient(depth_unit, pressure_unit)
    mudline = air_gap + water_depth
    finite = np.isfinite(z) & np.isfinite(rho) & (rho > 0)
    if not finite.any():
        raise ValueError("No valid density samples")
    log_top = float(z[finite][0])
    gap = log_top - mudline
    if gap > 0.01 * max(log_top, 1.0) and density_above_log is None:
        raise ValueError(
            f"RHOB starts {gap:g} {depth_unit} below the mudline; give density_above_log for that interval"
        )
    rho_filled = np.interp(z, z[finite], rho[finite])
    above = seawater_density * g * water_depth + (density_above_log or 0.0) * g * max(gap, 0.0)
    # Trapezoidal integration of the log from its top.
    increments = np.concatenate([[0.0], 0.5 * (rho_filled[1:] + rho_filled[:-1]) * np.diff(z) * g])
    cumulative = np.cumsum(increments)
    start = int(np.argmax(finite))
    stress = above + cumulative - cumulative[start]
    shallow = z < log_top
    if shallow.any():
        sediment = np.clip(z[shallow] - mudline, 0.0, None) * (density_above_log or 0.0) * g
        water = np.clip(np.minimum(z[shallow], mudline) - air_gap, 0.0, None) * seawater_density * g
        stress[shallow] = water + sediment
    return stress


@dataclass
class NormalCompactionTrend:
    """Normal compaction trend of shale sonic or resistivity against depth below the mudline.

    - sonic: ``DTn = dt_matrix + (dt_mudline - dt_matrix) * exp(-c * (tvd - mudline))``
    - resistivity: ``log10(Rn) = intercept + slope * (tvd - mudline)``
    """

    kind: str
    mudline: float = 0.0
    dt_matrix: float = 55.0
    dt_mudline: float = 200.0
    c: float = 0.0005
    intercept: float = 0.0
    slope: float = 0.0
    n_samples: int = 0
    r_squared: float = float("nan")

    def __post_init__(self) -> None:
        if self.kind not in EATON_EXPONENTS:
            raise ValueError("kind must be 'sonic' or 'resistivity'")

    def normal(self, tvd: Any) -> np.ndarray:
        """Normal-trend sonic (us/ft) or resistivity (ohm.m) at ``tvd``."""
        z = np.clip(np.asarray(tvd, dtype=float) - self.mudline, 0.0, None)
        if self.kind == "sonic":
            return self.dt_matrix + (self.dt_mudline - self.dt_matrix) * np.exp(-self.c * z)
        return np.power(10.0, self.intercept + self.slope * z)


def fit_normal_compaction_trend(
    tvd: Any,
    values: Any,
    *,
    kind: str,
    mask: Optional[Any] = None,
    mudline: float = 0.0,
    dt_matrix: float = 55.0,
) -> NormalCompactionTrend:
    """Fit a normal compaction trend to shale sonic or resistivity.

    ``mask`` selects the samples that define it: clean shale (high VCL) in an
    interval known to be normally pressured. The sonic trend is fitted as
    ``ln(DT - dt_matrix)`` linear in depth below the mudline; the
    resistivity trend as ``log10(R)`` linear in depth.
    """
    if kind not in EATON_EXPONENTS:
        raise ValueError("kind must be 'sonic' or 'resistivity'")
    z = np.asarray(tvd, dtype=float) - mudline
    v = np.asarray(values, dtype=float)
    use = np.isfinite(z) & np.isfinite(v) & (z >= 0)
    if mask is not None:
        use &= np.asarray(mask, dtype=bool)
    with np.errstate(invalid="ignore", divide="ignore"):
        if kind == "sonic":
            use &= v > dt_matrix
            y = np.log(np.where(v > dt_matrix, v - dt_matrix, np.nan))
        else:
            use &= v > 0
            y = np.log10(np.where(v > 0, v, np.nan))
    if use.sum() < 10:
        raise ValueError(f"Too few trend samples ({int(use.sum())}; need 10)")
    slope, intercept = np.polyfit(z[use], y[use], 1)
    fitted = slope * z[use] + intercept
    total = np.sum((y[use] - y[use].mean()) ** 2)
    r2 = float(1.0 - np.sum((y[use] - fitted) ** 2) / total) if total > 0 else float("nan")
    if not np.isfinite(r2) or r2 < 0.3 or (kind == "sonic" and slope >= 0):
        import warnings

        warnings.warn(
            f"Normal compaction trend fits poorly (R2 {r2:.2f}, {int(use.sum())} samples"
            f"{', slowness not decreasing with depth' if kind == 'sonic' and slope >= 0 else ''}): "
            "select clean shale in a normally pressured section, or give the trend parameters",
            UserWarning, stacklevel=2,
        )
    if kind == "sonic":
        return NormalCompactionTrend(
            "sonic", mudline=float(mudline), dt_matrix=dt_matrix, dt_mudline=float(dt_matrix + np.exp(intercept)),
            c=float(-slope), n_samples=int(use.sum()), r_squared=r2,
        )
    return NormalCompactionTrend(
        "resistivity", mudline=float(mudline), intercept=float(intercept), slope=float(slope),
        n_samples=int(use.sum()), r_squared=r2,
    )


def eaton_pore_pressure(
    overburden: Any,
    hydrostatic: Any,
    observed: Any,
    normal: Any,
    *,
    kind: str,
    exponent: Optional[float] = None,
) -> np.ndarray:
    """Eaton pore pressure from overburden, hydrostatic and observed/normal-trend log values.

    Parameters
    ----------
    overburden, hydrostatic : array-like
        Vertical stress and hydrostatic pressure, same unit.
    observed, normal : array-like
        Shale sonic (us/ft) or resistivity (ohm.m) and its normal-trend value.
    kind : str
        ``"sonic"`` or ``"resistivity"``.
    exponent : float, optional
        Eaton exponent; 3.0 (sonic) and 1.2 (resistivity) by default.

    Returns
    -------
    numpy.ndarray
        Pore pressure in the unit of the inputs: hydrostatic on the trend,
        rising towards the overburden as the log departs from it in the
        under-compacted direction; NaN where an input is missing and where
        the formula falls below zero (a log far faster or more resistive
        than the shale trend, as in sands and carbonates), with a warning
        giving the count.

    Examples
    --------
    On the normal trend, pore pressure is hydrostatic:

    >>> float(eaton_pore_pressure(500.0, 300.0, 100.0, 100.0, kind="sonic"))
    300.0
    """
    pp, negative = _eaton(overburden, hydrostatic, observed, normal, kind=kind, exponent=exponent)
    if negative:
        import warnings

        warnings.warn(
            f"{negative} sample(s) gave a pore pressure below zero and were left blank: the log is far "
            "faster or more resistive than the shale trend there. Eaton's method applies to shale; "
            "evaluate it on shale samples only",
            UserWarning, stacklevel=2,
        )
    return pp


def _eaton(overburden, hydrostatic, observed, normal, *, kind, exponent=None):
    """Eaton pore pressure and the number of samples blanked for falling below zero."""
    if kind not in EATON_EXPONENTS:
        raise ValueError("kind must be 'sonic' or 'resistivity'")
    x = EATON_EXPONENTS[kind] if exponent is None else float(exponent)
    if not np.isfinite(x) or x <= 0:
        raise ValueError(f"The Eaton exponent must be positive (got {x:g})")
    s = np.asarray(overburden, dtype=float)
    ph = np.asarray(hydrostatic, dtype=float)
    obs = np.asarray(observed, dtype=float)
    nrm = np.asarray(normal, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = nrm / obs if kind == "sonic" else obs / nrm
        pp = np.where(np.isfinite(ratio) & (ratio > 0), s - (s - ph) * np.power(ratio, x), np.nan)
        negative = np.isfinite(pp) & (pp < 0)
    pp = np.where(negative, np.nan, pp)
    return pp, int(np.sum(negative))


def pressure_gradient(
    pressure: Any,
    tvd: Any,
    *,
    air_gap: float = 0.0,
    depth_unit: str = "m",
    pressure_unit: str = "bar",
) -> np.ndarray:
    """Equivalent density (g/cc, equivalent mud weight) of a pressure at depth, from sea level."""
    z = np.asarray(tvd, dtype=float) - air_gap
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(z > 0, np.asarray(pressure, dtype=float) / (_gradient(depth_unit, pressure_unit) * z), np.nan)


def calibrate_eaton_exponent(
    measured: Any,
    overburden: Any,
    hydrostatic: Any,
    observed: Any,
    normal: Any,
    *,
    kind: str,
) -> Tuple[float, float]:
    """Eaton exponent that best fits measured pore pressures.

    All arrays are at the measurement depths. Solves
    ``ln((S - Pp)/(S - Ph)) = x ln(ratio)`` by least squares (ratio
    ``DTn/DT`` or ``R/Rn``). Use pressures that represent the shale, e.g.
    formation-tester points in sands interbedded with it, and for
    ``observed`` the shale's log value at each measurement depth (the
    adjacent shale interpolated across the sand), not the sand's own
    reading; :meth:`PorePressureResult.calibrate` does this when the result
    has a shale mask. Returns the exponent and the RMS pressure misfit with
    it.

    Raises ``ValueError`` when the best fit is not a positive exponent: the
    pressures then do not rise where the shale departs from the trend in the
    under-compacted direction, so the NCT, the shale values or the choice of
    measurements need another look.
    """
    if kind not in EATON_EXPONENTS:
        raise ValueError("kind must be 'sonic' or 'resistivity'")
    pm, s, ph, obs, nrm = (np.asarray(v, dtype=float) for v in (measured, overburden, hydrostatic, observed, normal))
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = nrm / obs if kind == "sonic" else obs / nrm
        y = np.log((s - pm) / (s - ph))
        x = np.log(ratio)
    use = np.isfinite(x) & np.isfinite(y) & (np.abs(x) > 1e-6)
    if use.sum() < 2:
        no_log = int(np.sum(~np.isfinite(x)))
        bad = int(np.sum(np.isfinite(x) & ~np.isfinite(y)))
        raise ValueError(
            f"Need at least two usable measurements; of {pm.size}: {no_log} have no log value or trend "
            f"(outside the logged interval or missing), {bad} lie outside hydrostatic-overburden, "
            f"{int(np.sum(np.isfinite(x) & (np.abs(x) <= 1e-6)))} sit on the trend"
        )
    exponent = float(np.sum(x[use] * y[use]) / np.sum(x[use] ** 2))
    if not exponent > 0:
        with np.errstate(divide="ignore", invalid="ignore"):
            excess = float(np.nanmedian(pm[use] / ph[use]))
        under = int(np.sum(x[use] < 0))
        raise ValueError(
            f"The measurements fit an Eaton exponent of {exponent:.3g}, not a positive one ({int(use.sum())} usable "
            f"points, median measured/hydrostatic {excess:.3f}, log on the under-compacted side of the trend at "
            f"{under}). Eaton needs pressure to rise where the shale departs from the trend towards "
            f"under-compaction. Check the normal compaction trend and use the shale's log value at each "
            "measurement depth, not the reservoir's; normally pressured points do not constrain the exponent, "
            "keep the default then"
        )
    fitted, _ = _eaton(s[use], ph[use], obs[use], nrm[use], kind=kind, exponent=exponent)
    return exponent, float(np.sqrt(np.nanmean((fitted - pm[use]) ** 2)))


@dataclass
class PorePressureResult:
    """Eaton pore pressure along a well, with its inputs (pressures in ``pressure_unit``)."""

    tvd: np.ndarray
    overburden: np.ndarray
    hydrostatic: np.ndarray
    observed: np.ndarray
    normal: np.ndarray
    pore_pressure: np.ndarray
    #: Pore pressure as equivalent density from sea level (g/cc).
    pore_pressure_emw: np.ndarray
    kind: str = "sonic"
    exponent: float = float("nan")
    air_gap: float = 0.0
    water_depth: float = 0.0
    depth_unit: str = "m"
    pressure_unit: str = "bar"
    #: TVD below sea level, when the depth-reference elevation is known.
    tvdss: Optional[np.ndarray] = None
    notes: list = field(default_factory=list)
    #: Samples Eaton was evaluated on (the ``shale`` mask), or None when every sample was.
    shale: Optional[np.ndarray] = None

    def __repr__(self) -> str:
        valid = np.isfinite(self.pore_pressure_emw)
        span = (f"{np.nanmin(self.tvd[valid]):g}-{np.nanmax(self.tvd[valid]):g}" if valid.any() else "none")
        return (f"PorePressureResult({self.kind}, exponent {self.exponent:g}, TVD {span} {self.depth_unit}, "
                f"EMW p50 {np.nanmedian(self.pore_pressure_emw) if valid.any() else float('nan'):.2f} g/cc, "
                f"air gap {self.air_gap:g}, water depth {self.water_depth:g}"
                f"{'' if self.shale is None else f', {int(np.sum(self.shale))} shale samples'})")

    def at(self, tvd: Any, *, datum: str = "TVD") -> Dict[str, np.ndarray]:
        """Overburden, hydrostatic, observed, normal and pore pressure at ``tvd`` on ``datum`` (TVD or TVDSS).

        With a shale mask, ``pore_pressure`` is blank between shales and two
        more entries carry the shale trend across them: ``shale_observed``,
        the shale log value interpolated between the nearest shale samples
        above and below, and ``shale_pore_pressure``, Eaton from it with this
        result's exponent. These are the values to compare with a pressure
        measured in a sand between shales.
        """
        base = self.tvd if datum.upper() == "TVD" else self.tvdss
        if base is None:
            raise ValueError("No TVDSS: the depth-reference elevation is unknown")
        z = np.asarray(tvd, dtype=float)
        ok = np.isfinite(base)
        order = np.argsort(base[ok])
        out = {}
        for name, values in (("overburden", self.overburden), ("hydrostatic", self.hydrostatic),
                             ("observed", self.observed), ("normal", self.normal),
                             ("pore_pressure", self.pore_pressure)):
            out[name] = np.interp(z, base[ok][order], np.asarray(values, dtype=float)[ok][order],
                                  left=np.nan, right=np.nan)
        if self.shale is not None:
            observed = np.asarray(self.observed, dtype=float)
            use = ok & np.asarray(self.shale, dtype=bool) & np.isfinite(observed)
            if use.sum() >= 1:
                sort = np.argsort(base[use])
                out["shale_observed"] = np.interp(z, base[use][sort], observed[use][sort], left=np.nan, right=np.nan)
            else:
                out["shale_observed"] = np.full(z.shape, np.nan)
            out["shale_pore_pressure"], _ = _eaton(
                out["overburden"], out["hydrostatic"], out["shale_observed"], out["normal"],
                kind=self.kind, exponent=self.exponent,
            )
        return out

    def calibrate(self, tvd: Any, pressure: Any, *, datum: str = "TVD") -> Tuple[float, float]:
        """Fit the Eaton exponent to measured pressures (same unit) at ``tvd`` on ``datum``.

        With a shale mask the fit uses the shale log value across each
        measurement depth (``shale_observed`` from :meth:`at`); without one it
        uses the log at the measurement depth itself, usually the reservoir,
        and warns. Returns the exponent and RMS misfit, as
        :func:`calibrate_eaton_exponent`; rerun :func:`eaton_from_well` with
        ``exponent=`` to apply it.
        """
        values = self.at(tvd, datum=datum)
        if self.shale is None:
            import warnings

            warnings.warn(
                "Calibrating on the log value at the measurement depths, which are usually reservoir rather "
                "than shale; pass shale= to eaton_from_well to calibrate against the shale trend",
                UserWarning, stacklevel=2,
            )
            observed = values["observed"]
        else:
            observed = values["shale_observed"]
        return calibrate_eaton_exponent(
            pressure, values["overburden"], values["hydrostatic"], observed, values["normal"],
            kind=self.kind,
        )


def eaton_from_well(
    well: Any,
    nct: NormalCompactionTrend,
    *,
    curve: Optional[str] = None,
    tvd_curve: Optional[str] = None,
    vertical: bool = False,
    density_curve: str = "RHOB",
    water_depth: float = 0.0,
    air_gap: Optional[float] = None,
    density_above_log: Optional[float] = None,
    water_density: float = 1.03,
    seawater_density: float = 1.03,
    exponent: Optional[float] = None,
    shale: Optional[Any] = None,
    pressure_unit: str = "bar",
    add_curves: bool = False,
) -> PorePressureResult:
    """Eaton pore pressure along a well from its logs and stated inputs.

    Parameters
    ----------
    well : Well
        Well with sonic (``DTC``/``DT``) or deep resistivity, bulk density and
        TVD (``pp.read(..., survey=...)`` adds ``TVD``).
    nct : NormalCompactionTrend
        The normal trend; its ``kind`` decides sonic or resistivity, its
        mudline is on this well's TVD.
    curve : str, optional
        Log to use; ``DTC``/``DT`` or ``RDEP``/``RT`` by default.
    tvd_curve, vertical
        TVD below the log depth reference (``TVD`` when present), or declare
        the well vertical.
    density_curve : str
        Bulk density curve (g/cc) the overburden is integrated from;
        default ``RHOB``.
    water_depth : float
        Sea depth at the well (0 onshore), in the well's depth unit.
    air_gap : float, optional
        Height of the log depth reference above sea level offshore, above
        ground level onshore. Read from the header when not given: offshore
        the reference elevation (EKB/EDF/APD), onshore EKB - EGL.
    density_above_log, water_density, seawater_density, exponent
        As for :func:`overburden_pressure`, :func:`hydrostatic_pressure` and
        :func:`eaton_pore_pressure`.
    shale : array-like of bool, optional
        Shale samples, one per depth sample (e.g. ``well["VCL"].data > 0.6``).
        Eaton's method is a shale method: pore pressure is computed on these
        samples and left blank elsewhere, and :meth:`PorePressureResult.calibrate`
        interpolates the shale log across the measurement depths. Without it
        every sample is evaluated and the result is to be read in shales only.
    pressure_unit : {"bar", "psi", "kPa", "MPa"}
        Unit of the overburden, hydrostatic and pore pressures; default
        ``"bar"``.
    add_curves : bool
        Add ``OVERBURDEN``, ``PHYD``, ``PP_EATON`` and ``PP_EATON_EMW``.
    """
    from petrophys.io.survey_reader import _depth_unit_from, _item, _number, depth_reference_elevation

    unit = str(well.depth_unit)
    notes = []
    depth = np.asarray(well.depth, dtype=float)
    if tvd_curve is None and not vertical:
        if well.get("TVD") is None:
            raise ValueError("Pore pressure needs TVD: read the well with survey=..., pass tvd_curve, or vertical=True")
        tvd_curve = "TVD"
    tvd = depth
    if tvd_curve is not None:
        tvd_log = well.get(tvd_curve)
        tvd = np.asarray(tvd_log.data, dtype=float)
        tvd_unit = _depth_unit_from(tvd_log.unit) or unit
        if tvd_unit != unit:
            # A TVD curve in its own unit is converted to the well's depth unit.
            tvd = tvd * 0.3048 if (tvd_unit, unit) == ("ft", "m") else tvd / 0.3048
            notes.append(f"{tvd_curve} converted from {tvd_unit} to {unit}")
    elevation, source = depth_reference_elevation(well)
    if air_gap is not None and water_depth > 0:
        # Offshore the air gap is the depth reference's height above sea level.
        elevation = float(air_gap)
    if air_gap is None:
        if elevation is None:
            raise ValueError("The header has no depth-reference elevation (EKB/EDF/APD); pass air_gap=")
        if water_depth > 0:
            limit = 150.0 if unit == "m" else 492.0
            if elevation > limit:
                raise ValueError(
                    f"The header puts the depth reference {elevation:g} {unit} above sea level ({source['source']}), "
                    "too high for an offshore air gap: is the well onshore (water_depth=0)? Otherwise pass air_gap="
                )
            air_gap = elevation
            notes.append(f"Air gap {air_gap:g} {unit} from {source['source']} (offshore)")
        else:
            ground_value, ground_unit = _item(well, "EGL")
            ground = _number(ground_value)
            if ground is None:
                raise ValueError("Onshore, the air gap needs the ground elevation (EGL) in the header, or air_gap=")
            gl_unit = _depth_unit_from(ground_unit) or unit
            if gl_unit != unit:
                ground = ground / 0.3048 if (gl_unit, unit) == ("m", "ft") else ground * 0.3048
            air_gap = elevation - ground
            notes.append(f"Depth reference {air_gap:g} {unit} above ground (EKB - EGL), onshore")
    if curve is None:
        names = ("DTC", "DT", "DTCO") if nct.kind == "sonic" else ("RDEP", "RT", "ILD", "LLD")
        curve = next((n for n in names if well.get(n) is not None), None)
        if curve is None:
            raise ValueError(f"No {nct.kind} curve among {names}; pass curve=")
    observed = np.asarray(well.get(curve).data, dtype=float)
    density = well.get(density_curve)
    if density is None:
        raise ValueError(f"No density curve {density_curve!r} for the overburden")
    rhob = np.asarray(density.data, dtype=float)
    has_tvd = np.isfinite(tvd)
    s = np.full(tvd.shape, np.nan)
    s[has_tvd] = overburden_pressure(
        tvd[has_tvd], rhob[has_tvd], water_depth=water_depth, air_gap=air_gap, seawater_density=seawater_density,
        density_above_log=density_above_log, depth_unit=unit, pressure_unit=pressure_unit,
    )
    ph = hydrostatic_pressure(tvd, water_density=water_density, air_gap=air_gap, depth_unit=unit,
                              pressure_unit=pressure_unit)
    normal = nct.normal(tvd)
    x = EATON_EXPONENTS[nct.kind] if exponent is None else float(exponent)
    shale_mask = None
    evaluated = observed
    if shale is None:
        notes.append("No shale mask: Eaton evaluated on every sample; read it in shales only (pass shale=)")
    else:
        shale_mask = np.asarray(shale, dtype=bool).reshape(-1)
        if shale_mask.shape != depth.shape:
            raise ValueError(f"shale has {shale_mask.size} values; the well has {depth.size} depth samples")
        evaluated = np.where(shale_mask, observed, np.nan)
        notes.append(f"Eaton evaluated on {int(shale_mask.sum())} shale samples of {depth.size}; blank elsewhere")
    pp, negative = _eaton(s, ph, evaluated, normal, kind=nct.kind, exponent=x)
    if negative:
        import warnings

        message = (f"{negative} sample(s) gave a pore pressure below zero and were left blank: the log is far "
                   f"{'faster' if nct.kind == 'sonic' else 'more resistive'} than the shale trend there")
        notes.append(message)
        warnings.warn(message + ("" if shale_mask is not None else
                                 "; Eaton applies to shale, pass shale= to evaluate shale samples only"),
                      UserWarning, stacklevel=2)
    emw = pressure_gradient(pp, tvd, air_gap=air_gap, depth_unit=unit, pressure_unit=pressure_unit)
    result = PorePressureResult(
        tvd=tvd, overburden=s, hydrostatic=ph, observed=observed, normal=normal, pore_pressure=pp,
        pore_pressure_emw=emw, kind=nct.kind, exponent=x, air_gap=float(air_gap), water_depth=float(water_depth),
        depth_unit=unit, pressure_unit=pressure_unit,
        tvdss=None if elevation is None else tvd - elevation, notes=notes, shale=shale_mask,
    )
    if add_curves:
        well.add_curve("OVERBURDEN", s, pressure_unit, "Overburden (vertical stress)")
        well.add_curve("PHYD", ph, pressure_unit, "Hydrostatic pressure")
        well.add_curve("PP_EATON", pp, pressure_unit, f"Eaton pore pressure ({nct.kind}, exponent {x:g})")
        well.add_curve("PP_EATON_EMW", emw, "g/cc", "Eaton pore pressure, equivalent density")
    return result


__all__ = [
    "EATON_EXPONENTS",
    "NormalCompactionTrend",
    "PorePressureResult",
    "calibrate_eaton_exponent",
    "eaton_from_well",
    "eaton_pore_pressure",
    "fit_normal_compaction_trend",
    "hydrostatic_pressure",
    "overburden_pressure",
    "pressure_gradient",
]
