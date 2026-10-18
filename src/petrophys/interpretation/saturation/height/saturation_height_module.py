"""Saturation-height modeling.

Fits a saturation function to capillary-pressure / J-function core data and
predicts water saturation versus height above the free-water level (FWL) for the
log column. Two fitted models are provided:

  * J power-law:     Sw = a * J^b               (normalizes k/phi; basin-wide)
  * Brooks-Corey:    Sw = Swir + (1-Swir)*(Pd/Pc)^lambda   (single drainage curve)

The J-function model is the one applied back to logs, because J folds in each
depth's permeability and porosity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union
import warnings

import numpy as np

from petrophys.interpretation.saturation.height.capillary import (
    leverett_j,
    pc_from_height,
)
from petrophys.model.units import normalize_depth_unit
from petrophys._internal.analysis_helpers import get_curve_data_depthwise

if TYPE_CHECKING:
    from petrophys.model.well import Well

ArrayLike = Union[float, np.ndarray]


@dataclass
class JFunctionModel:
    """Sw = a * J^b (optionally above an irreducible floor Swir)."""

    a: float
    b: float
    swir: float = 0.0
    form: str = "power"      # "power" or "power_swir"
    ift: float = 30.0
    theta: float = 30.0
    sg_water: float = 1.0
    sg_hc: float = 0.8
    r2: Optional[float] = None

    def sw_from_j(self, j: ArrayLike) -> ArrayLike:
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
        j = leverett_j(pc_psi, perm_md, phi, ift=self.ift, theta=self.theta)
        return self.sw_from_j(j)

    def sw_from_height(self, height_ft: ArrayLike, perm_md: ArrayLike, phi: ArrayLike) -> ArrayLike:
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
        pc = np.asarray(pc_psi, dtype=float)
        sw = np.full(pc.shape, np.nan) if pc.ndim else np.array(np.nan)
        ok = np.isfinite(pc) & (pc > 0)
        ratio = np.where(ok, np.minimum(self.pd / np.where(ok, pc, 1.0), 1.0), np.nan)
        sw = self.swir + (1.0 - self.swir) * np.power(ratio, self.lam)
        sw = np.clip(sw, self.swir, 1.0)
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
    well: "Well",
    model: JFunctionModel,
    fwl_depth: float,
    perm_curve=("PERM_CC", "PERM", "PERM_CORE", "PERM_TIMUR"),
    phi_curve=("PHIE_CC", "PHIE", "PHIT"),
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
    """
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
            "saturation_height_from_logs: no tvd_curve given, so the depth index is "
            "taken as true vertical depth, which holds for a vertical well only. "
            "Pass tvd_curve, on the datum of fwl_depth, for a deviated well.",
            UserWarning,
            stacklevel=2,
        )
        depth = np.asarray(well.depth if getattr(well, "depth", None) is not None else well._depth, dtype=float)
        unit = well.depth_unit
    perm, _, _ = get_curve_data_depthwise(well, list(perm_curve))
    phi, _, _ = get_curve_data_depthwise(well, list(phi_curve))
    if perm is None or phi is None:
        raise ValueError("permeability and porosity curves are required")
    height = (fwl_depth - depth) if depth_increases_down else (depth - fwl_depth)
    if unit == "m":
        height = height / 0.3048
    height = np.clip(height, 0.0, None)
    sw = model.sw_from_height(height, perm, phi)
    return np.asarray(sw, dtype=float)
