"""Empirical saturation-height functions fitted to log saturation above a known FWL.

Where core capillary-pressure data are missing or sparse, the log water
saturation of a well with a known free-water level (FWL) is itself the data:
fitting Sw against true vertical height above the FWL gives a function to
populate a model. Three common forms:

- ``"power"`` (Lambda-type): ``Sw = a * h**(-b)``
- ``"cuddy"`` (FOIL, Cuddy et al. 1993): ``BVW = Sw * phi = a * h**(-b)``,
  so ``Sw = a * h**(-b) / phi``; porosity enters through bulk volume water
- ``"skelt_harrison"`` (Harrison and Jing 2001):
  ``Sw = 1 - a * exp(-(b / (h + d))**c)``, sigmoidal through the transition
  zone; ``d`` absorbs an FWL offset when ``fit_offset`` is set

Height is vertical: pass a TVD curve, or declare the well vertical. The FWL
must be on the same datum and in the same unit as the TVD. Fit each rock
type (or porosity class) separately; one function over mixed rock smears
the transition zone.

References
----------
Cuddy, S., Allinson, G. and Steele, R. (1993). A simple, convincing model for
calculating water saturations in Southern North Sea gas fields. SPWLA 34th
Annual Logging Symposium.

Harrison, B. and Jing, X.D. (2001). Saturation height methods and their
impact on volumetric hydrocarbon in place estimates. SPE 71326.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

import numpy as np
from petrophys._internal.advice import Advice

SATURATION_HEIGHT_MODELS = ("power", "cuddy", "skelt_harrison")


@dataclass
class FittedSaturationHeight:
    """An empirical saturation-height function and its fit statistics."""

    model: str
    parameters: Dict[str, float]
    #: Unit of the heights the function was fitted on (the TVD unit).
    height_unit: str = ""
    n_samples: int = 0
    r_squared: float = float("nan")
    #: Root-mean-square Sw error of the fit (v/v).
    rmse_sw: float = float("nan")
    height_range: tuple = (float("nan"), float("nan"))
    notes: list = field(default_factory=list)

    def predict(self, height: Any, phi: Optional[Any] = None) -> np.ndarray:
        """Sw (0-1) at ``height`` above the FWL; ``phi`` is needed by the Cuddy form.

        Heights at or below the FWL give Sw = 1.
        """
        h = np.asarray(height, dtype=float)
        p = self.parameters
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            if self.model == "power":
                sw = p["a"] * np.power(np.where(h > 0, h, np.nan), -p["b"])
            elif self.model == "cuddy":
                if phi is None:
                    raise ValueError("The Cuddy function needs porosity")
                sw = p["a"] * np.power(np.where(h > 0, h, np.nan), -p["b"]) / np.asarray(phi, dtype=float)
            else:
                sw = 1.0 - p["a"] * np.exp(-np.power(p["b"] / (h + p["d"]), p["c"]))
        sw = np.where(np.isfinite(h) & (h <= 0), 1.0, sw)
        return np.clip(sw, 0.0, 1.0)


def _datum_curve(reference: str) -> str:
    """The curve name of a depth datum: TVDSS for sea-level datums, TVD for the rig reference."""
    key = "".join(ch for ch in reference.upper() if ch.isalnum())
    if key in {"TVDSS", "TVDMSL", "SSTVD", "TVDBMSL", "TVDSL", "SUBSEA", "TVDSUBSEA", "MSL"}:
        return "TVDSS"
    if key in {"TVD", "TVDRKB", "TVDKB", "TVDDF", "TVDRT", "TVDBRT"}:
        return "TVD"
    return reference.upper()


def height_above_fwl(tvd: Any, fwl: float) -> np.ndarray:
    """Vertical height above the free-water level, ``fwl - tvd`` (TVD increasing downward)."""
    return float(fwl) - np.asarray(tvd, dtype=float)


def fit_saturation_height(
    height: Any,
    sw: Any,
    *,
    model: str = "skelt_harrison",
    phi: Optional[Any] = None,
    fit_offset: bool = False,
    min_height: float = 0.0,
    height_unit: str = "",
) -> FittedSaturationHeight:
    """Fit an empirical saturation-height function to Sw versus height above the FWL.

    Parameters
    ----------
    height : array-like
        Vertical height above the FWL (see :func:`height_above_fwl`).
    sw : array-like
        Water saturation (v/v) at those heights.
    model : str
        ``"power"``, ``"cuddy"`` or ``"skelt_harrison"``.
    phi : array-like, optional
        Porosity (v/v); required by ``"cuddy"``.
    fit_offset : bool
        Skelt-Harrison only: fit the height offset ``d`` (an FWL shift)
        instead of holding it at 0 for a known FWL.
    min_height : float
        Leave out samples at or below this height (0 drops the water leg).
    height_unit : str
        Unit of ``height``, recorded with the fit.

    Returns
    -------
    FittedSaturationHeight
        Parameters, R-squared and RMS Sw error on the fitted samples;
        ``predict`` evaluates the function.
    """
    if model not in SATURATION_HEIGHT_MODELS:
        raise ValueError(f"model must be one of {SATURATION_HEIGHT_MODELS}")
    h = np.asarray(height, dtype=float).reshape(-1)
    s = np.asarray(sw, dtype=float).reshape(-1)
    p = None if phi is None else np.asarray(phi, dtype=float).reshape(-1)
    if model == "cuddy" and p is None:
        raise ValueError("The Cuddy function needs porosity (phi)")
    use = np.isfinite(h) & np.isfinite(s) & (h > min_height) & (s > 0) & (s <= 1.0)
    if p is not None:
        use &= np.isfinite(p) & (p > 0)
    if use.sum() < 5:
        raise ValueError(f"Too few samples above the FWL to fit ({int(use.sum())}; need 5)")
    h, s = h[use], s[use]
    p = p[use] if p is not None else None
    notes = []

    if model == "power":
        keep = s < 0.999  # samples clipped at Sw = 1 carry no slope information
        slope, intercept = np.polyfit(np.log(h[keep]), np.log(s[keep]), 1)
        parameters = {"a": float(np.exp(intercept)), "b": float(-slope)}
    elif model == "cuddy":
        bvw = s * p
        slope, intercept = np.polyfit(np.log(h), np.log(bvw), 1)
        parameters = {"a": float(np.exp(intercept)), "b": float(-slope)}
    else:
        from scipy.optimize import curve_fit

        def skelt(hh, a, b, c, d):
            return 1.0 - a * np.exp(-np.power(b / (hh + d), c))

        swirr_guess = float(np.clip(np.nanpercentile(s, 5), 0.02, 0.9))
        b0 = float(np.nanmedian(h))
        if fit_offset:
            start = (1.0 - swirr_guess, b0, 1.5, 0.0)
            bounds = ([0.0, 1e-6, 0.05, -float(np.min(h)) + 1e-6], [1.0, np.inf, 20.0, np.inf])
            function = skelt
        else:
            start = (1.0 - swirr_guess, b0, 1.5)
            bounds = ([0.0, 1e-6, 0.05], [1.0, np.inf, 20.0])

            def function(hh, a, b, c):
                return skelt(hh, a, b, c, 0.0)
        try:
            values, _ = curve_fit(function, h, s, p0=start, bounds=bounds, maxfev=20000)
        except RuntimeError as exc:
            raise ValueError(f"Skelt-Harrison fit did not converge: {exc}") from None
        parameters = {"a": float(values[0]), "b": float(values[1]), "c": float(values[2]),
                      "d": float(values[3]) if fit_offset else 0.0}
        at_bound = [name for name, value, low, high in zip("abcd", values, bounds[0], bounds[1])
                    if np.isclose(value, low, rtol=1e-6, atol=1e-9) or np.isclose(value, high, rtol=1e-6)]
        if at_bound:
            import warnings

            message = (f"Parameters {at_bound} reached their bounds: the data do not constrain the function "
                       "(check the FWL, the rock type mix and the height range)")
            notes.append(message)
            warnings.warn(f"Skelt-Harrison fit: {message}", UserWarning, stacklevel=2)
        notes.append(f"Irreducible Sw approached high above the FWL: {1.0 - parameters['a']:.3f}")

    fitted = FittedSaturationHeight(
        model=model, parameters=parameters, height_unit=height_unit, n_samples=int(h.size),
        height_range=(float(h.min()), float(h.max())), notes=notes,
    )
    predicted = fitted.predict(h, p)
    residual = predicted - s
    total = np.sum((s - s.mean()) ** 2)
    fitted.r_squared = float(1.0 - np.sum(residual**2) / total) if total > 0 else float("nan")
    fitted.rmse_sw = float(np.sqrt(np.mean(residual**2)))
    if not np.isfinite(fitted.r_squared) or fitted.r_squared < 0.2:
        import warnings

        message = (f"the {model} function explains little of Sw (R2 {fitted.r_squared:.2f}, RMS error "
                   f"{fitted.rmse_sw:.2f}) over heights {fitted.height_range[0]:.0f}-{fitted.height_range[1]:.0f} "
                   f"{height_unit}: check the FWL and its datum, and fit one rock type at a time")
        fitted.notes.append(message)
        warnings.warn(f"Saturation-height fit: {message}", UserWarning, stacklevel=2)
    return fitted


def fit_saturation_height_from_well(
    well: Any,
    fwl: Any,
    *,
    tvd_curve: Optional[str] = None,
    vertical: bool = False,
    model: str = "skelt_harrison",
    sw_curve: Sequence[str] = ("SW",),
    phi_curve: Sequence[str] = ("PHIE", "PHIT"),
    top: Optional[float] = None,
    bottom: Optional[float] = None,
    mask: Optional[Any] = None,
    fit_offset: bool = False,
) -> FittedSaturationHeight:
    """Fit a saturation-height function to a well's log Sw above a known FWL.

    Parameters
    ----------
    well : Well
        Interpreted well (Sw and porosity).
    fwl : float or PressureFWLResult
        Free-water level: a TVD on the datum and in the unit of
        ``tvd_curve`` (of the depth index when ``vertical``), or the result
        of :func:`~petrophys.interpretation.saturation.height.fit_pressure_fwl`,
        ``FormationPressureData.fit_fwl`` or ``infer_contacts``, whose datum
        (TVDSS, TVD_MSL) picks the well's ``TVDSS`` curve (added by
        ``pp.read(..., survey=...)``) and whose unit is converted.
    tvd_curve : str, optional
        True vertical depth curve. Required unless ``vertical`` is set or
        ``fwl`` is a pressure result.
    vertical : bool
        Declare the well vertical, so the depth index is TVD.
    model, fit_offset
        As for :func:`fit_saturation_height`.
    sw_curve, phi_curve : sequence of str
        Candidate curve names, first found is used.
    top, bottom : float, optional
        Measured-depth window (the reservoir interval).
    mask : array-like of bool, optional
        Samples to fit, e.g. one rock type or the net reservoir flag.
    """
    fwl_unit = None
    if hasattr(fwl, "fwl_depth"):
        reference = _datum_curve(str(getattr(fwl, "depth_reference", "") or ""))
        fwl_unit = str(getattr(fwl, "depth_unit", "") or "") or None
        if fwl.fwl_depth is None:
            raise ValueError(
                f"The pressure result gives no FWL (status {getattr(fwl, 'status', 'unknown')!r}): the contact "
                "inference is ambiguous; review its branches or label the station fluids and fit_fwl()"
            )
        fwl = float(fwl.fwl_depth)
        if tvd_curve is None and not vertical:
            if well.get(reference) is None:
                raise ValueError(Advice(
                    f"The FWL is on {reference}; the well has no {reference} curve. Read it with "
                    "pp.read(..., survey=...) (or attach_survey) so TVDSS is computed, or pass tvd_curve.",
                    ("survey", "tvd_curve")
                ))
            tvd_curve = reference
    if tvd_curve is None and not vertical:
        raise ValueError(Advice(
            "Height above the FWL is vertical: pass tvd_curve (e.g. 'TVDSS' from pp.read(..., survey=...)), "
            "or vertical=True for a vertical well", ("tvd_curve", "survey", "vertical")
        ))
    depth = np.asarray(well.depth, dtype=float)
    if tvd_curve is not None:
        curve = well.get(tvd_curve)
        if curve is None:
            raise ValueError(f"No curve {tvd_curve!r} on the well")
        tvd = np.asarray(curve.data, dtype=float)
        from petrophys.io.survey_reader import _depth_unit_from

        unit = _depth_unit_from(getattr(curve, "unit", "")) or str(well.depth_unit)
    else:
        tvd = depth
        unit = str(well.depth_unit)

    def first(names: Sequence[str]) -> Optional[np.ndarray]:
        for name in names:
            curve = well.get(name)
            if curve is not None:
                return np.asarray(curve.data, dtype=float)
        return None

    sw = first(sw_curve)
    if sw is None:
        raise ValueError(f"No Sw curve among {list(sw_curve)}")
    phi = first(phi_curve)
    if model == "cuddy" and phi is None:
        raise ValueError(f"The Cuddy function needs porosity; none of {list(phi_curve)} found")
    window = np.ones(depth.shape, dtype=bool)
    if top is not None:
        window &= depth >= top
    if bottom is not None:
        window &= depth <= bottom
    if mask is not None:
        window &= np.asarray(mask, dtype=bool)
    if fwl_unit is None and tvd_curve is not None and well.get("TVD") is not None and well.get("TVDSS") is not None:
        import warnings

        warnings.warn(
            f"A bare FWL number has no datum; it is read on {tvd_curve}. The well has TVD and TVDSS: "
            "pass the pressure FWL result (it carries its datum) or check the curve",
            UserWarning, stacklevel=2,
        )
    if fwl_unit and unit and fwl_unit != unit and {fwl_unit, unit} <= {"m", "ft"}:
        fwl = fwl / 0.3048 if (fwl_unit, unit) == ("m", "ft") else fwl * 0.3048
    if not (np.nanmin(tvd) - 0.2 * np.ptp(tvd[np.isfinite(tvd)]) <= fwl):
        import warnings

        warnings.warn(
            f"FWL {fwl:g} lies far above the logged TVD ({np.nanmin(tvd):g}-{np.nanmax(tvd):g}): "
            "check that both are on the same datum", UserWarning, stacklevel=2,
        )
    height = height_above_fwl(tvd, fwl)
    return fit_saturation_height(
        height[window], sw[window], model=model,
        phi=None if phi is None else phi[window], fit_offset=fit_offset, height_unit=unit,
    )


__all__ = [
    "SATURATION_HEIGHT_MODELS",
    "FittedSaturationHeight",
    "fit_saturation_height",
    "fit_saturation_height_from_well",
    "height_above_fwl",
]
