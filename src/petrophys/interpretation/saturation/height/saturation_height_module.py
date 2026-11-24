"""Saturation-height modeling.

Fits a saturation function to capillary-pressure / J-function core data and
predicts water saturation versus height above the free-water level (FWL) for the
log column. Two fitted models are provided:

  * J power-law:     Sw = a * J^b               (normalizes k/phi; basin-wide)
  * Brooks-Corey:    Sw = Swir + (1-Swir)*(Pd/Pc)^lambda   (single drainage curve)

The J-function model is the one applied back to logs, because J folds in each
depth's permeability and porosity.

References
----------
Brooks, R.H. and Corey, A.T. (1964). Hydraulic properties of porous media.
Hydrology Papers 3, Colorado State University.

Leverett, M.C. (1941). Capillary behavior in porous solids. *Transactions
of the AIME* 142, 152-169.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, Union
import warnings

import numpy as np

from petrophys.interpretation.saturation.height.capillary import (
    FLUID_SYSTEMS,
    leverett_j,
    pc_from_height,
)
from petrophys.model.units import normalize_depth_unit
from petrophys._internal.analysis_helpers import as_well, get_curve_data_depthwise

if TYPE_CHECKING:
    from petrophys.model.well import Well

ArrayLike = Union[float, np.ndarray]


@dataclass
class JFunctionModel:
    """Sw = a * J^b (optionally above an irreducible floor Swir).

    ``form`` is ``"power"`` (Sw = a * J^b) or ``"power_swir"``
    (Sw = Swir + (1 - Swir) * a * J^b). ``ift`` (dyne/cm) and ``theta``
    (degrees) are the interfacial tension and contact angle J is computed
    with, ``sg_water`` and ``sg_hc`` the specific gravities that convert
    height to capillary pressure, and ``r2`` the fit's R2 on Sw.
    """

    a: float
    b: float
    swir: float = 0.0
    form: str = "power"      # "power" or "power_swir"
    ift: float = 30.0
    theta: float = 30.0
    sg_water: float = 1.0
    sg_hc: float = 0.8
    r2: Optional[float] = None
    #: The samples the fit used: ``sw``, ``perm`` (mD) and ``phi``, with
    #: ``height`` (ft above the FWL) from :func:`fit_j_function_from_well` or
    #: ``j`` from :func:`~petrophys.interpretation.saturation.height.fit_j_function_from_core`.
    points: Optional[Dict[str, np.ndarray]] = field(default=None, repr=False)

    def _equation(self, digits: int = 4) -> str:
        """The fitted relation as text, in the model's ``form``."""
        core = f"{self.a:.{digits}f} * J^{self.b:.{digits}f}"
        if self.form == "power_swir":
            return f"Sw = {self.swir:.{digits}f} + (1 - {self.swir:.{digits}f}) * {core}"
        return f"Sw = {core}"

    def __repr__(self) -> str:
        n = "" if self.points is None else f", n={len(self.points['sw'])}"
        r2 = "" if self.r2 is None else f", R2={self.r2:.3f}"
        return f"JFunctionModel({self._equation()}{r2}{n})"

    def plot(self, perms: Sequence[float] = (1.0, 10.0, 100.0), ax: Any = None) -> Any:
        """Sw against height above the FWL: the fitted samples coloured by
        permeability and the model's curve for each of ``perms`` (mD), each at
        the median porosity of the samples near that permeability. Each curve
        has the colour of its permeability on the colour bar of the samples.

        A model fitted from core plugs is drawn as Sw against J, with the
        fitted curve. Returns the figure. A model without its fitted samples
        (``points``) raises ValueError.
        """
        import matplotlib.pyplot as plt
        from matplotlib import patheffects as path_effects

        from petrophys.visualization.style import plot_style

        if self.points is None:
            raise ValueError("plot() needs the fitted samples: fit with fit_j_function_from_well "
                             "or fit_j_function_from_core")
        from petrophys.visualization.base import Plot

        if "height" not in self.points:
            return self._plot_j(ax)
        height, sw, perm, phi = (self.points[key] for key in ("height", "sw", "perm", "phi"))
        with plot_style():
            if ax is None:
                fig, ax = plt.subplots(figsize=(6, 5))
            else:
                fig = ax.figure
            shown = ax.scatter(sw, height, c=np.log10(perm), s=7, lw=0, cmap="viridis")
            fig.colorbar(shown, ax=ax, label="log10 permeability [mD]")
            curve_height = np.linspace(0.5, float(np.nanmax(height)), 200)
            log_perm = np.log10(perm)
            outline = [path_effects.withStroke(linewidth=3.6, foreground="black")]
            for k in perms:
                near = np.abs(log_perm - np.log10(k)) <= 0.5
                p = float(np.median(phi[near])) if near.any() else float(np.median(phi))
                ax.plot(self.sw_from_height(curve_height, np.full_like(curve_height, k),
                                            np.full_like(curve_height, p)),
                        curve_height, color=shown.cmap(shown.norm(np.log10(k))), lw=2.0,
                        path_effects=outline, label=f"model, {k:g} mD, phi {p:.2f}")
            title = f"J-function fit, R2 = {self.r2:.2f}" if self.r2 is not None else "J-function fit"
            ax.set(xlim=(0, 1), xlabel="Sw [v/v]", ylabel="Height above FWL [ft]", title=title)
            ax.legend(loc="best", fontsize=8)
            return Plot(fig)

    def _plot_j(self, ax: Any = None) -> Any:
        import matplotlib.pyplot as plt

        from petrophys.visualization.base import Plot
        from petrophys.visualization.style import plot_style

        assert self.points is not None
        j, sw, perm = (self.points[key] for key in ("j", "sw", "perm"))
        with plot_style():
            if ax is None:
                fig, ax = plt.subplots(figsize=(6, 5))
            else:
                fig = ax.figure
            shown = ax.scatter(sw, j, c=np.log10(np.clip(perm, 1e-3, None)), s=5, lw=0, cmap="viridis")
            fig.colorbar(shown, ax=ax, label="log10 permeability [mD]")
            line = np.geomspace(max(float(np.nanmin(j)), 1e-4), float(np.nanmax(j)), 200)
            ax.plot(self.sw_from_j(line), line, color="black", lw=1.8,
                    label=self._equation(3))
            ax.set_yscale("log")
            title = f"J-function from core, R2 = {self.r2:.2f}" if self.r2 is not None else "J-function from core"
            ax.set(xlim=(0, 1), xlabel="Sw [v/v]", ylabel="Leverett J", title=title)
            ax.legend(loc="best", fontsize=8)
            return Plot(fig)

    def sw_from_j(self, j: ArrayLike) -> ArrayLike:
        """Water saturation (v/v) at Leverett J values.

        Parameters
        ----------
        j : float or array
            Leverett J-function, dimensionless.

        Returns
        -------
        float or array
            Sw between ``swir`` and 1; 1 where J is zero or negative (at or
            below the free-water level) and NaN where J is not finite.
        """
        jj = np.asarray(j, dtype=float)
        pos = np.isfinite(jj) & (jj > 0)
        at_fwl = np.isfinite(jj) & (jj <= 0)   # Pc=0 at/below the FWL -> fully water
        core = self.a * np.power(np.where(pos, jj, 1.0), self.b)
        if self.form == "power_swir":
            sw = self.swir + (1.0 - self.swir) * core
        else:
            sw = core
        sw = np.clip(sw, self.swir, 1.0)
        out = np.where(pos, sw, np.nan)
        out = np.where(at_fwl, 1.0, out)
        return float(out) if np.ndim(j) == 0 else out

    def sw_from_pc(self, pc_psi: ArrayLike, perm_md: ArrayLike, phi: ArrayLike) -> ArrayLike:
        """Water saturation (v/v) at a capillary pressure, for a rock's permeability and porosity.

        Parameters
        ----------
        pc_psi : float or array
            Capillary pressure at reservoir conditions, psi.
        perm_md : float or array
            Permeability, mD.
        phi : float or array
            Porosity, v/v.

        Returns
        -------
        float or array
            Sw from :meth:`sw_from_j` at the J of each sample, computed with
            the model's ``ift`` and ``theta``; NaN where the permeability or
            porosity is not positive.
        """
        j = leverett_j(pc_psi, perm_md, phi, ift=self.ift, theta=self.theta)
        return self.sw_from_j(j)

    def sw_from_height(self, height_ft: ArrayLike, perm_md: ArrayLike, phi: ArrayLike) -> ArrayLike:
        """Water saturation (v/v) at a height above the free-water level.

        Parameters
        ----------
        height_ft : float or array
            Height above the free-water level, ft; a negative height counts
            as zero.
        perm_md : float or array
            Permeability, mD.
        phi : float or array
            Porosity, v/v.

        Returns
        -------
        float or array
            Sw from :meth:`sw_from_pc` at the capillary pressure
            ``0.433 * (sg_water - sg_hc) * height_ft`` psi; 1 at and below the
            free-water level.
        """
        pc = pc_from_height(height_ft, sg_water=self.sg_water, sg_hc=self.sg_hc)
        return self.sw_from_pc(pc, perm_md, phi)


@dataclass
class BrooksCoreyModel:
    """Sw = Swir + (1-Swir)*(Pd/Pc)^lambda for Pc >= Pd (drainage)."""

    swir: float
    pd: float
    lam: float
    r2: Optional[float] = None

    def sw_from_pc(self, pc_psi: ArrayLike) -> ArrayLike:
        """Water saturation (v/v) at a capillary pressure.

        Parameters
        ----------
        pc_psi : float or array
            Capillary pressure, psi, in the fluid system of the fitted curve.

        Returns
        -------
        float or array
            Sw between ``swir`` and 1; 1 up to the entry pressure ``pd``,
            which includes zero and negative pressure (at or below the
            free-water level), and NaN where the pressure is not finite.
        """
        pc = np.asarray(pc_psi, dtype=float)
        above = np.isfinite(pc) & (pc > 0)
        ratio = np.where(above, np.minimum(self.pd / np.where(above, pc, 1.0), 1.0), 1.0)
        sw = self.swir + (1.0 - self.swir) * np.power(ratio, self.lam)
        sw = np.where(np.isfinite(pc), np.clip(sw, self.swir, 1.0), np.nan)
        return float(sw) if np.ndim(pc_psi) == 0 else sw


def _r2(y: np.ndarray, yhat: np.ndarray) -> float:
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def fit_j_function(
    j: ArrayLike,
    sw: ArrayLike,
    form: str = "power",
    swir: float = 0.0,
    ift: float = 30.0,
    theta: float = 30.0,
    sg_water: float = 1.0,
    sg_hc: float = 0.8,
) -> JFunctionModel:
    """Fit Sw = a*J^b (log-linear). ``form='power_swir'`` fits the reduced
    saturation (Sw-Swir)/(1-Swir) so the curve asymptotes to ``swir``."""
    if form not in ("power", "power_swir"):
        raise ValueError(f"form must be 'power' or 'power_swir'; got {form!r}")
    j = np.asarray(j, dtype=float)
    sw = np.asarray(sw, dtype=float)
    ok = np.isfinite(j) & np.isfinite(sw) & (j > 0) & (sw > 0) & (sw <= 1.0)
    if form == "power_swir":
        red = (sw - swir) / (1.0 - swir)
        ok = ok & (red > 0)
        x, y = np.log(j[ok]), np.log(red[ok])
    else:
        ok = ok & (sw > swir)
        x, y = np.log(j[ok]), np.log(sw[ok])
    if x.size < 3:
        raise ValueError("need >= 3 valid (J, Sw) points to fit")
    b, log_a = np.polyfit(x, y, 1)
    model = JFunctionModel(
        a=float(np.exp(log_a)), b=float(b), swir=float(swir), form=form,
        ift=ift, theta=theta, sg_water=sg_water, sg_hc=sg_hc,
    )
    model.r2 = _r2(sw[ok], model.sw_from_j(j[ok]))
    return model


def fit_brooks_corey(pc_psi: ArrayLike, sw: ArrayLike, swir: Optional[float] = None) -> BrooksCoreyModel:
    """Fit Brooks-Corey (Swir, Pd, lambda) to a drainage Pc curve.

    A given ``swir`` is held fixed and only Pd and lambda are fitted.
    """
    pc = np.asarray(pc_psi, dtype=float)
    sw = np.asarray(sw, dtype=float)
    ok = np.isfinite(pc) & np.isfinite(sw) & (pc > 0) & (sw > 0) & (sw <= 1.0)
    pc, sw = pc[ok], sw[ok]
    if pc.size < 4:
        raise ValueError("need >= 4 valid (Pc, Sw) points to fit Brooks-Corey")
    try:
        from scipy.optimize import curve_fit

        def model(pc_, swir_, pd_, lam_):
            ratio = np.minimum(pd_ / pc_, 1.0)
            return swir_ + (1.0 - swir_) * np.power(ratio, lam_)

        if swir is None:
            swir0 = float(np.min(sw))
            p0 = [swir0, float(np.min(pc)), 0.5]
            lower = [0.0, 1e-3, 1e-3]
            upper = [min(0.9, swir0 + 0.2), float(np.max(pc)), 10.0]
            popt, _ = curve_fit(model, pc, sw, p0=p0, bounds=(lower, upper), maxfev=20000)
            bc = BrooksCoreyModel(swir=float(popt[0]), pd=float(popt[1]), lam=float(popt[2]))
        else:
            swir0 = float(swir)
            popt, _ = curve_fit(
                lambda pc_, pd_, lam_: model(pc_, swir0, pd_, lam_), pc, sw,
                p0=[float(np.min(pc)), 0.5],
                bounds=([1e-3, 1e-3], [float(np.max(pc)), 10.0]), maxfev=20000,
            )
            bc = BrooksCoreyModel(swir=swir0, pd=float(popt[0]), lam=float(popt[1]))
    except Exception:
        # Fallback: fix Swir at min(Sw), linearize log((Sw-Swir)/(1-Swir)) vs log(Pc).
        swir0 = float(np.min(sw)) if swir is None else float(swir)
        red = (sw - swir0) / (1.0 - swir0)
        m = red > 0
        slope, intercept = np.polyfit(np.log(pc[m]), np.log(red[m]), 1)
        lam = -slope
        pd = float(np.exp(intercept / lam)) if lam > 0 else float(np.min(pc))
        bc = BrooksCoreyModel(swir=swir0, pd=pd, lam=lam)
    bc.r2 = _r2(sw, bc.sw_from_pc(pc))
    return bc


def saturation_height_from_logs(
    well: Any,
    model: JFunctionModel,
    fwl_depth: float,
    perm_curve: Union[str, Sequence[str]] = ("PERM_CC", "PERM", "PERM_CORE", "PERM_TIMUR"),
    phi_curve: Union[str, Sequence[str]] = ("PHIE_CC", "PHIE", "PHIT"),
    depth_increases_down: bool = True,
    tvd_curve: Optional[str] = None,
) -> np.ndarray:
    """Predict Sw vs depth from a fitted J-function model and an FWL depth.

    Height above the FWL is ``fwl_depth - depth`` (so shallower = higher = drier),
    converted from the depth unit (m or ft) to ft for the field-unit capillary
    relation Pc = 0.433 (SGw - SGhc) h. Below the FWL height is clipped to
    0 -> Sw = 1.

    Capillary height is vertical. ``depth`` is the ``tvd_curve`` when one is
    named, in that curve's unit (the well's depth unit when the curve has
    none). Otherwise it is the well's depth index, taken as true vertical
    depth, which holds for a vertical well only; a warning says so.
    ``fwl_depth`` must be in the same unit and on the same datum as that
    depth.

    Takes a well or an interpretation result. ``perm_curve`` (mD) and
    ``phi_curve`` (v/v) each name a curve or several in order of preference;
    at each depth the first of them with a value is used. A well with none
    of the named curves raises ValueError. Returns Sw (v/v) at every depth
    of the well.
    """
    well = as_well(well)
    height = _height_above_fwl(well, fwl_depth, tvd_curve, depth_increases_down, "saturation_height_from_logs")
    data = {}
    for role, value in (("permeability", perm_curve), ("porosity", phi_curve)):
        names = [value] if isinstance(value, str) else list(value)
        values, _, _ = get_curve_data_depthwise(well, names)
        if values is None:
            raise ValueError(f"No {role} curve: the well has none of {names}")
        data[role] = values
    sw = model.sw_from_height(height, data["permeability"], data["porosity"])
    return np.asarray(sw, dtype=float)


def _height_above_fwl(well: "Well", fwl_depth: float, tvd_curve: Optional[str],
                      depth_increases_down: bool, caller: str) -> np.ndarray:
    """Height above the FWL in ft, clipped at 0 below it."""
    if tvd_curve is not None:
        curve = well.get(tvd_curve)
        if curve is None or not np.any(np.isfinite(np.asarray(curve.data, dtype=float))):
            raise ValueError(f"TVD curve {tvd_curve!r} with finite samples is not on the well")
        depth = np.asarray(curve.data, dtype=float)
        unit = normalize_depth_unit(curve.unit, strict=False, default=well.depth_unit)
        if unit not in ("m", "ft"):
            unit = well.depth_unit
    else:
        warnings.warn(
            f"{caller}: no tvd_curve given, so the depth index is "
            "taken as true vertical depth, which holds for a vertical well only. "
            "Pass tvd_curve, on the datum of fwl_depth, for a deviated well.",
            UserWarning,
            stacklevel=3,
        )
        depth = np.asarray(well.depth if getattr(well, "depth", None) is not None else well._depth, dtype=float)
        unit = well.depth_unit
    height = (fwl_depth - depth) if depth_increases_down else (depth - fwl_depth)
    if unit == "m":
        height = height / 0.3048
    return np.clip(height, 0.0, None)


def fit_j_function_from_well(
    well: "Well",
    fwl_depth: float,
    sw_curve: Union[str, Sequence[str]] = ("SW_INPUT", "SW"),
    perm_curve: Union[str, Sequence[str]] = ("PERM_CC", "PERM", "PERM_CORE", "PERM_TIMUR"),
    phi_curve: Union[str, Sequence[str]] = ("PHIE_CC", "PHIE", "PHIT"),
    fluids: str = "oil_brine",
    sg_water: float = 1.0,
    sg_hc: float = 0.8,
    form: str = "power",
    swir: float = 0.0,
    top: Optional[float] = None,
    bottom: Optional[float] = None,
    tvd_curve: Optional[str] = None,
) -> JFunctionModel:
    """Fit a Leverett J-function to a well's log Sw above a free-water level.

    Each sample's capillary pressure comes from its height above
    ``fwl_depth`` (in the well's depth unit and datum) and the fluid
    densities, and its J from that pressure, permeability and porosity; the
    J-Sw power law is then fitted (:func:`fit_j_function`). ``fluids`` names
    the interfacial tension and contact angle (``"oil_brine"``,
    ``"gas_brine"``, ...). ``top`` and ``bottom`` bound the samples. The
    model keeps them, so ``model.plot()`` draws the fit; apply it with
    :func:`saturation_height_from_logs`. Takes a well or an interpretation
    result.

    ``sw_curve``, ``perm_curve`` and ``phi_curve`` each name a curve or
    several in order of preference; at each depth the first of them with a
    value is used. ``sg_water`` and ``sg_hc`` are the specific gravities of
    the water and the hydrocarbon, and ``form`` and ``swir`` are those of
    :func:`fit_j_function`. Height is measured along ``tvd_curve`` when one
    is named, with ``fwl_depth`` in that curve's unit and on its datum;
    otherwise the depth index is taken as true vertical depth, which holds
    for a vertical well only, and a warning says so. Unknown ``fluids``, a
    well with none of the named curves, or fewer than 3 usable samples above
    the free-water level raise ValueError.

    >>> model = fit_j_function_from_well(hw3, fwl_depth=7991)  # doctest: +SKIP
    >>> model.plot()                                           # doctest: +SKIP
    """
    well = as_well(well)
    if fluids not in FLUID_SYSTEMS:
        raise ValueError(f"fluids must be one of {sorted(FLUID_SYSTEMS)}; got {fluids!r}")
    ift, theta = FLUID_SYSTEMS[fluids]
    data: Dict[str, np.ndarray] = {}
    for role, value in (("sw", sw_curve), ("perm", perm_curve), ("phi", phi_curve)):
        names = [value] if isinstance(value, str) else list(value)
        values, _, _ = get_curve_data_depthwise(well, names)
        if values is None:
            raise ValueError(f"No {role} curve: the well has none of {names}")
        data[role] = np.asarray(values, dtype=float)
    height = _height_above_fwl(well, fwl_depth, tvd_curve, True, "fit_j_function_from_well")
    depth = np.asarray(well.depth, dtype=float)
    window = np.ones(depth.shape, dtype=bool)
    if top is not None:
        window &= depth >= top
    if bottom is not None:
        window &= depth <= bottom
    sw, perm, phi = data["sw"], data["perm"], data["phi"]
    use = (window & (height > 0) & np.isfinite(sw) & (sw > 0) & (sw <= 1)
           & np.isfinite(perm) & (perm > 0) & np.isfinite(phi) & (phi > 0))
    j = leverett_j(pc_from_height(height[use], sg_water=sg_water, sg_hc=sg_hc),
                   perm[use], phi[use], ift=ift, theta=theta)
    model = fit_j_function(j, sw[use], form=form, swir=swir, ift=ift, theta=theta,
                           sg_water=sg_water, sg_hc=sg_hc)
    model.points = {"height": height[use], "sw": sw[use], "perm": perm[use], "phi": phi[use]}
    return model
