"""Laboratory capillary pressure (SCAL), and fluid contacts from it.

The steps follow common practice for special core analysis:

1. Read the plug capillary-pressure curves: Pc against wetting-phase
   saturation, with each plug's porosity and permeability
   (:func:`read_capillary_pressure`).
2. Describe each plug: for mercury injection, the Thomeer (1960) hyperbola
   gives the entry (displacement) pressure Pd, the pore geometrical factor G
   and the mercury bulk volume at infinite pressure Bv-infinity
   (:meth:`CapillaryPressureData.thomeer`).
3. Convert laboratory Pc to reservoir conditions by the ratio of
   ``sigma * cos(theta)`` (:meth:`CapillaryPressureData.to_reservoir`).
4. Normalise the plugs with the Leverett (1941) J-function and fit one J-Sw
   relation (:func:`fit_j_function_from_core`).
5. With that core model, find the free-water level (FWL) that best explains
   the log water saturation (:func:`estimate_fwl_from_logs`), and place the
   water contact, where the logs first read Sw = 1, above the FWL by the
   entry-pressure height (:func:`water_contact_from_fwl`).

References
----------
Leverett, M.C. (1941). Capillary behavior in porous solids. Trans. AIME 142.
Thomeer, J.H.M. (1960). Introduction of a pore geometrical factor defined by
the capillary pressure curve. JPT 12(3), 73-77.
Worthington, P.F. (2002). Application of saturation-height functions in
integrated reservoir description. AAPG Methods in Exploration 13.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np

from petrophys.interpretation.saturation.height.capillary import (
    FLUID_SYSTEMS,
    WATER_GRADIENT_PSI_PER_FT,
    convert_pc_lab_to_reservoir,
    leverett_j,
)
from petrophys.interpretation.saturation.height.saturation_height_module import (
    JFunctionModel,
    fit_j_function,
)

if TYPE_CHECKING:
    import pandas as pd

    from petrophys.model.well import Well

#: Column names read for each quantity, in order of preference (case-insensitive).
_COLUMNS = {
    "sample": ("SAMPLE", "SAMPLE_NO", "SAMPLE_ID", "PLUG", "PLUG_ID", "ID"),
    "pc": ("PCHG", "PC_HG", "PC", "PCAP", "PC_PSI", "PRESSURE"),
    "sw": ("SW", "SWN", "SW_FRAC", "SWT"),
    "porosity": ("POROSITY", "PHI", "POR", "CPOR", "PHIC"),
    "permeability": ("PERM", "PERMEABILITY", "K", "KAIR", "KH", "KH_CORE"),
    "depth": ("DEPTH", "DEPT", "MD", "PLUG_DEPTH"),
}


def _column(frame: "pd.DataFrame", given: Optional[str], role: str, required: bool = True) -> Optional[str]:
    if given is not None:
        if given not in frame.columns:
            raise KeyError(f"No column {given!r}; the table has {list(frame.columns)}")
        return given
    upper = {str(name).strip().upper().replace(" ", "_"): name for name in frame.columns}
    for candidate in _COLUMNS[role]:
        if candidate in upper:
            return str(upper[candidate])
    if required:
        raise KeyError(f"No {role} column found (looked for {', '.join(_COLUMNS[role])}); "
                       f"name it with {role}=; the table has {list(frame.columns)}")
    return None


def _fraction(values: np.ndarray) -> np.ndarray:
    """Percent to v/v when the values read as percent."""
    finite = values[np.isfinite(values)]
    return values / 100.0 if finite.size and float(np.nanmedian(finite)) > 1.0 else values


def _thomeer_bv(pc: np.ndarray, pd: float, g: float, bv_inf: float) -> np.ndarray:
    """Thomeer hyperbola: mercury bulk volume (% of bulk) at each Pc; zero up to Pd."""
    out = np.zeros_like(pc, dtype=float)
    above = pc > pd
    out[above] = bv_inf * np.exp(-g / np.log10(pc[above] / pd))
    return out


def fit_thomeer(pc_psi: Any, bv_percent: Any) -> Dict[str, float]:
    """Fit the Thomeer (1960) hyperbola to one mercury-injection curve.

    ``bv_percent`` is the bulk volume occupied by mercury, percent of bulk
    (porosity x non-wetting saturation x 100), at each injection pressure
    ``pc_psi``. Returns ``Pd`` (entry pressure, psi), ``G`` (pore geometrical
    factor), ``BV_INF`` (bulk volume at infinite pressure, %) and ``R2``. One
    pore system; a bimodal curve is described by its dominant system.
    """
    from scipy.optimize import least_squares

    pc = np.asarray(pc_psi, dtype=float)
    bv = np.asarray(bv_percent, dtype=float)
    ok = np.isfinite(pc) & np.isfinite(bv) & (pc > 0)
    pc, bv = pc[ok], bv[ok]
    entered = bv > 0.01 * max(float(bv.max()), 1e-9)
    if np.count_nonzero(entered) < 4:
        raise ValueError("need at least 4 points with mercury entered to fit a Thomeer hyperbola")
    bv_max = float(bv.max())
    pd0 = float(pc[entered].min())
    lower = (float(pc.min()) * 0.5, 0.005, bv_max)
    upper = (pd0 * 1.5, 5.0, bv_max * 3.0)
    start = (pd0 * 0.9, 0.3, bv_max * 1.05)

    def residual(params: np.ndarray) -> np.ndarray:
        return _thomeer_bv(pc, *params) - bv

    fit = least_squares(residual, x0=start, bounds=(lower, upper), x_scale="jac")
    pd, g, bv_inf = (float(value) for value in fit.x)
    ss_tot = float(np.sum((bv - bv.mean()) ** 2))
    r2 = 1.0 - float(np.sum(fit.fun ** 2)) / ss_tot if ss_tot > 0 else float("nan")
    return {"Pd": pd, "G": g, "BV_INF": bv_inf, "R2": r2}


@dataclass
class CapillaryPressureData:
    """Plug capillary-pressure curves: a row per measured point.

    ``table`` has ``SAMPLE``, ``PC`` (psi), ``SW`` (wetting-phase saturation,
    v/v), ``POROSITY`` (v/v), ``PERMEABILITY`` (mD) and, where known,
    ``DEPTH``. ``system`` names the fluids the Pc was measured with
    (``"air_mercury"``, ``"air_brine"``, ``"oil_brine"``, ``"gas_brine"``),
    and ``ift`` (dyne/cm) and ``theta`` (degrees) their interfacial tension
    and contact angle.
    """

    table: "pd.DataFrame"
    system: str = "air_mercury"
    ift: float = 480.0
    theta: float = 140.0

    def __repr__(self) -> str:
        pc = self.table["PC"]
        return (f"CapillaryPressureData({self.n_samples} samples, {len(self.table)} points, "
                f"{self.system} Pc {pc.min():.3g}-{pc.max():.3g} psi)")

    @property
    def n_samples(self) -> int:
        return int(self.table["SAMPLE"].nunique())

    @property
    def samples(self) -> "pd.DataFrame":
        """A row per plug: porosity, permeability (and depth) and its number of points."""
        columns = [name for name in ("DEPTH", "POROSITY", "PERMEABILITY") if name in self.table]
        frame = self.table.groupby("SAMPLE", sort=False)[columns].first()
        frame["POINTS"] = self.table.groupby("SAMPLE", sort=False).size()
        return frame

    def to_reservoir(self, fluids: str = "oil_brine", ift: Optional[float] = None,
                     theta: Optional[float] = None) -> "CapillaryPressureData":
        """The curves at reservoir conditions: Pc scaled by the ratio of
        ``|sigma cos(theta)|`` of the reservoir fluids (``fluids``, or ``ift`` and
        ``theta`` given) to that of the laboratory fluids."""
        if fluids not in FLUID_SYSTEMS:
            raise ValueError(f"fluids must be one of {sorted(FLUID_SYSTEMS)}; got {fluids!r}")
        res_ift, res_theta = FLUID_SYSTEMS[fluids]
        res_ift = res_ift if ift is None else float(ift)
        res_theta = res_theta if theta is None else float(theta)
        table = self.table.copy()
        table["PC"] = convert_pc_lab_to_reservoir(table["PC"].to_numpy(dtype=float),
                                                  self.ift, self.theta, res_ift, res_theta)
        return replace(self, table=table, system=fluids, ift=res_ift, theta=res_theta)

    def leverett_j(self) -> np.ndarray:
        """The Leverett J-function of every point (Pc, permeability, porosity, sigma cos theta)."""
        return np.asarray(leverett_j(self.table["PC"].to_numpy(dtype=float),
                                     self.table["PERMEABILITY"].to_numpy(dtype=float),
                                     self.table["POROSITY"].to_numpy(dtype=float),
                                     ift=self.ift, theta=self.theta), dtype=float)

    def thomeer(self) -> "pd.DataFrame":
        """A Thomeer hyperbola fitted to each plug: ``Pd`` (entry pressure, psi,
        in this data's fluid system), ``G``, ``BV_INF`` (% of bulk) and ``R2``,
        with the plug's porosity and permeability. Plugs that cannot be fitted
        are left out, and a warning names them."""
        import pandas as pd

        rows, failed = {}, []
        for sample, group in self.table.groupby("SAMPLE", sort=False):
            bv = 100.0 * group["POROSITY"].to_numpy(dtype=float) * (1.0 - group["SW"].to_numpy(dtype=float))
            try:
                rows[sample] = fit_thomeer(group["PC"].to_numpy(dtype=float), bv)
            except ValueError:
                failed.append(str(sample))
        if failed:
            warnings.warn(f"No Thomeer fit for {len(failed)} plugs: {', '.join(failed[:10])}"
                          + (" ..." if len(failed) > 10 else ""), UserWarning, stacklevel=2)
        frame = pd.DataFrame.from_dict(rows, orient="index")
        frame.index.name = "SAMPLE"
        return frame.join(self.samples[["POROSITY", "PERMEABILITY"]])

    def entry_pressure(self) -> "pd.Series":
        """Each plug's entry (displacement) pressure, psi, in this data's fluid
        system: the Thomeer Pd."""
        return self.thomeer()["Pd"].rename("ENTRY_PRESSURE")

    def plot(self, ax: Any = None) -> Any:
        """The curves, Pc (log) against Sw, coloured by permeability."""
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm

        from petrophys.visualization.style import plot_style

        with plot_style():
            if ax is None:
                fig, ax = plt.subplots(figsize=(6.5, 5))
            else:
                fig = ax.figure
            perms = self.samples["PERMEABILITY"].clip(lower=1e-3)
            norm = LogNorm(vmin=float(perms.min()), vmax=float(max(perms.max(), perms.min() * 10)))
            cmap = plt.get_cmap("viridis")
            for sample, group in self.table.groupby("SAMPLE", sort=False):
                shown = group[group["PC"] > 0]
                ax.plot(shown["SW"], shown["PC"], color=cmap(norm(max(perms[sample], 1e-3))), lw=0.8)
            ax.set_yscale("log")
            ax.set(xlim=(0, 1), xlabel="Sw (v/v)", ylabel=f"Pc, {self.system.replace('_', '-')} (psi)",
                   title=f"Capillary pressure, {self.n_samples} plugs")
            fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="Permeability (mD)")
            return fig


def read_capillary_pressure(
    source: Union[str, Path, "pd.DataFrame"],
    *,
    sample: Optional[str] = None,
    pc: Optional[str] = None,
    sw: Optional[str] = None,
    porosity: Optional[str] = None,
    permeability: Optional[str] = None,
    depth: Optional[str] = None,
    system: str = "air_mercury",
    ift: Optional[float] = None,
    theta: Optional[float] = None,
    sheet_name: Any = 0,
) -> CapillaryPressureData:
    """Read plug capillary-pressure curves from a CSV or Excel table (or a DataFrame).

    One row per measured point: the plug (``sample``), the capillary pressure
    in psi (``pc``), the wetting-phase saturation (``sw``), and the plug's
    ``porosity``, ``permeability`` (mD) and, optionally, ``depth``. Columns
    are found by common names (``Sample``, ``PcHg`` or ``Pc``, ``Sw``,
    ``Porosity``, ``Perm`` ...) unless named. Porosity and Sw in percent
    become v/v. ``system`` names the laboratory fluids, which set the
    interfacial tension and contact angle (``ift``, ``theta`` override them).
    """
    import pandas as pd

    if isinstance(source, pd.DataFrame):
        frame = source
    else:
        path = Path(source)
        frame = pd.read_excel(path, sheet_name=sheet_name) if path.suffix.lower() in (
            ".xlsx", ".xls", ".xlsm") else pd.read_csv(path)
    if system not in FLUID_SYSTEMS:
        raise ValueError(f"system must be one of {sorted(FLUID_SYSTEMS)}; got {system!r}")
    names = {
        "SAMPLE": _column(frame, sample, "sample"),
        "PC": _column(frame, pc, "pc"),
        "SW": _column(frame, sw, "sw"),
        "POROSITY": _column(frame, porosity, "porosity"),
        "PERMEABILITY": _column(frame, permeability, "permeability"),
        "DEPTH": _column(frame, depth, "depth", required=False),
    }
    table = pd.DataFrame({key: frame[name] for key, name in names.items() if name is not None})
    for key in ("PC", "SW", "POROSITY", "PERMEABILITY"):
        table[key] = pd.to_numeric(table[key], errors="coerce")
    table = table.dropna(subset=["PC", "SW", "POROSITY", "PERMEABILITY"]).reset_index(drop=True)
    table["SW"] = _fraction(table["SW"].to_numpy(dtype=float))
    table["POROSITY"] = _fraction(table["POROSITY"].to_numpy(dtype=float))
    lab_ift, lab_theta = FLUID_SYSTEMS[system]
    return CapillaryPressureData(table=table, system=system,
                                 ift=lab_ift if ift is None else float(ift),
                                 theta=lab_theta if theta is None else float(theta))


def fit_j_function_from_core(
    data: CapillaryPressureData,
    fluids: str = "oil_brine",
    *,
    sg_water: float = 1.0,
    sg_hc: float = 0.8,
    max_height_ft: float = 1000.0,
    form: str = "power",
    swir: float = 0.0,
    ift: Optional[float] = None,
    theta: Optional[float] = None,
) -> JFunctionModel:
    """Fit one Leverett J-function to the plugs' capillary-pressure curves.

    The curves are converted to reservoir ``fluids`` (``ift`` and ``theta``
    override the table's values), each point's J is computed from its Pc,
    permeability and porosity, and ``Sw = a * J^b`` is fitted (``form`` and
    ``swir`` as in :func:`fit_j_function`). Points where the non-wetting
    phase has not entered (Sw = 1) and points above ``max_height_ft`` of
    hydrocarbon column (Pc beyond what the reservoir sees) are left out.
    ``sg_water`` and ``sg_hc`` are the reservoir specific gravities the model
    converts height with. The model's ``plot()`` draws Sw against J.
    """
    reservoir = data if data.system == fluids and ift is None and theta is None else data.to_reservoir(
        fluids, ift=ift, theta=theta)
    j = reservoir.leverett_j()
    sw = reservoir.table["SW"].to_numpy(dtype=float)
    pc = reservoir.table["PC"].to_numpy(dtype=float)
    pc_max = WATER_GRADIENT_PSI_PER_FT * (sg_water - sg_hc) * max_height_ft
    use = np.isfinite(j) & (j > 0) & (sw > 0) & (sw < 0.999) & (pc <= pc_max)
    model = fit_j_function(j[use], sw[use], form=form, swir=swir, ift=reservoir.ift,
                           theta=reservoir.theta, sg_water=sg_water, sg_hc=sg_hc)
    model.points = {"j": j[use], "sw": sw[use],
                    "perm": reservoir.table["PERMEABILITY"].to_numpy(dtype=float)[use],
                    "phi": reservoir.table["POROSITY"].to_numpy(dtype=float)[use]}
    return model


def water_contact_from_fwl(
    fwl_depth: float,
    entry_pressure_psi: float,
    *,
    sg_water: float = 1.0,
    sg_hc: float = 0.8,
    depth_unit: str = "ft",
) -> float:
    """The depth of the water contact (where logs first read Sw = 1) above the FWL.

    Hydrocarbon enters the rock only where the capillary pressure exceeds
    the entry pressure, so the contact lies above the free-water level by
    ``h = Pd / (0.433 * (sg_water - sg_hc))`` ft. ``entry_pressure_psi`` is the
    entry pressure at reservoir conditions (``data.to_reservoir().entry_pressure()``,
    e.g. its median, or the value of the rock type at the contact). Depth
    increases downwards, in ``depth_unit`` (``"ft"`` or ``"m"``).
    """
    if sg_water <= sg_hc:
        raise ValueError("sg_water must exceed sg_hc")
    height_ft = float(entry_pressure_psi) / (WATER_GRADIENT_PSI_PER_FT * (sg_water - sg_hc))
    return float(fwl_depth) - (height_ft * 0.3048 if depth_unit == "m" else height_ft)


@dataclass
class FWLFromLogs:
    """The free-water level that best explains the log water saturation.

    ``fwl`` is the best depth, ``low`` and ``high`` its 95% range, ``rms`` the
    Sw misfit there, ``n`` the samples used, and ``profile`` the misfit at
    each candidate depth. Where the logs do not reach the transition zone the
    misfit keeps falling to the edge of the search: ``bound`` is then
    ``"deeper"`` (the FWL lies below ``low``; a well entirely in the oil leg)
    or ``"shallower"`` (above ``high``; a well in the water leg), and ``fwl``
    is NaN.
    """

    fwl: float
    low: float
    high: float
    rms: float
    n: int
    depth_unit: str
    profile: "pd.DataFrame" = field(repr=False)
    bound: Optional[str] = None
    _plot_data: Dict[str, Any] = field(default_factory=dict, repr=False)

    def __repr__(self) -> str:
        if self.bound == "deeper":
            where = f"FWL deeper than {self.low:.1f} {self.depth_unit} (no transition zone logged)"
        elif self.bound == "shallower":
            where = f"FWL shallower than {self.high:.1f} {self.depth_unit} (the well is in the water leg)"
        else:
            where = f"FWL {self.fwl:.1f} {self.depth_unit}, 95% range {self.low:.1f}-{self.high:.1f}"
        return f"FWLFromLogs({where}, RMS Sw misfit {self.rms:.3f}, n={self.n})"

    def plot(self) -> Any:
        """The misfit against candidate FWL, and log against model Sw at the best FWL."""
        import matplotlib.pyplot as plt

        from petrophys.visualization.style import plot_style

        with plot_style():
            fig, (ax_fit, ax_sw) = plt.subplots(1, 2, figsize=(10, 5.5), gridspec_kw={"width_ratios": [1, 1]})
            ax_fit.plot(self.profile["RMS"], self.profile["FWL"], color="black", lw=1.2)
            ax_fit.axhspan(self.low if np.isfinite(self.low) else self.profile["FWL"].min(),
                           self.high if np.isfinite(self.high) else self.profile["FWL"].max(),
                           color="tab:blue", alpha=0.15, label="95% range")
            best = self._plot_data["best"]
            ax_fit.axhline(best, color="tab:blue", lw=1.2,
                           label=f"FWL {best:.1f}" if self.bound is None else f"search edge {best:.1f}")
            ax_fit.invert_yaxis()
            ax_fit.set(xlabel="RMS Sw misfit", ylabel=f"Candidate FWL ({self.depth_unit})",
                       title="Misfit against FWL")
            ax_fit.legend(loc="lower right", fontsize=8)
            depth, log_sw, model_sw = (self._plot_data[key] for key in ("depth", "log_sw", "model_sw"))
            ax_sw.plot(log_sw, depth, ".", ms=2.5, color="0.45", label="log Sw")
            ax_sw.plot(model_sw, depth, ".", ms=2.5, color="tab:blue", label="model Sw")
            ax_sw.axhline(self._plot_data["best"], color="tab:blue", lw=1.0)
            ax_sw.invert_yaxis()
            ax_sw.set(xlim=(0, 1), xlabel="Sw (v/v)", ylabel=f"Depth ({self.depth_unit})",
                      title="Log and model at the best FWL")
            ax_sw.legend(loc="lower left", fontsize=8)
            fig.tight_layout()
            return fig


def _depth_and_unit(well: "Well", tvd_curve: Optional[str], caller: str) -> Tuple[np.ndarray, str]:
    from petrophys.model.units import normalize_depth_unit

    if tvd_curve is not None:
        curve = well.get(tvd_curve)
        if curve is None:
            raise ValueError(f"TVD curve {tvd_curve!r} is not on the well")
        unit = normalize_depth_unit(curve.unit, strict=False, default=well.depth_unit)
        return np.asarray(curve.data, dtype=float), unit if unit in ("m", "ft") else well.depth_unit
    warnings.warn(f"{caller}: no tvd_curve given, so the depth index is taken as true vertical "
                  "depth, which holds for a vertical well only.", UserWarning, stacklevel=3)
    return np.asarray(well.depth, dtype=float), well.depth_unit


def estimate_fwl_from_logs(
    well: Any,
    model: JFunctionModel,
    *,
    sw_curve: Union[str, Sequence[str]] = ("SW_INPUT", "SW"),
    perm_curve: Union[str, Sequence[str]] = ("PERM_CC", "PERM", "PERM_CORE", "PERM_TIMUR"),
    phi_curve: Union[str, Sequence[str]] = ("PHIE_CC", "PHIE", "PHIT"),
    top: Optional[float] = None,
    bottom: Optional[float] = None,
    search: Optional[Tuple[float, float]] = None,
    tvd_curve: Optional[str] = None,
) -> FWLFromLogs:
    """The free-water level at which a saturation-height model best matches the logs.

    The FWL is varied, and at each candidate depth the model (from core,
    :func:`fit_j_function_from_core`) predicts Sw at every sample from its
    height above the FWL, permeability and porosity; the candidate with the
    least RMS difference from the log Sw is the estimate. The 95% range comes
    from the misfit profile (an F-test on the summed squared misfit, with the
    sample count reduced for the correlation of neighbouring log samples).
    ``top`` and ``bottom`` bound the samples used, ``search`` the candidate
    depths (default: from the middle of the samples' depth span to one span
    below the deepest). Depths are in the well's depth unit, or the
    ``tvd_curve``'s. Takes a well or an interpretation result. Compare the
    estimate with a pressure FWL where one exists.
    """
    import pandas as pd
    from scipy.stats import f as f_dist

    from petrophys._internal.analysis_helpers import as_well, get_curve_data_depthwise

    well = as_well(well)
    data = {}
    for role, value in (("sw", sw_curve), ("perm", perm_curve), ("phi", phi_curve)):
        names = [value] if isinstance(value, str) else list(value)
        values, _, _ = get_curve_data_depthwise(well, names)
        if values is None:
            raise ValueError(f"No {role} curve: the well has none of {names}")
        data[role] = np.asarray(values, dtype=float)
    depth, unit = _depth_and_unit(well, tvd_curve, "estimate_fwl_from_logs")
    index_depth = np.asarray(well.depth, dtype=float)
    use = (np.isfinite(depth) & np.isfinite(data["sw"]) & (data["sw"] >= 0) & (data["sw"] <= 1)
           & np.isfinite(data["perm"]) & (data["perm"] > 0) & np.isfinite(data["phi"]) & (data["phi"] > 0))
    if top is not None:
        use &= index_depth >= top
    if bottom is not None:
        use &= index_depth <= bottom
    n = int(np.count_nonzero(use))
    if n < 20:
        raise ValueError(f"{n} samples with Sw, permeability and porosity; at least 20 are needed")
    z, sw, perm, phi = depth[use], data["sw"][use], data["perm"][use], data["phi"][use]
    span = float(z.max() - z.min()) or 1.0
    low_search, high_search = search if search is not None else (float(z.min()) + 0.5 * span, float(z.max()) + span)
    candidates = np.linspace(low_search, high_search, 401)
    to_ft = 1.0 / 0.3048 if unit == "m" else 1.0
    sse = np.empty(candidates.size)
    for i, fwl in enumerate(candidates):
        predicted = np.asarray(model.sw_from_height(np.clip(fwl - z, 0.0, None) * to_ft, perm, phi), dtype=float)
        sse[i] = float(np.sum((predicted - sw) ** 2))
    best = int(np.argmin(sse))
    predicted = np.asarray(model.sw_from_height(np.clip(candidates[best] - z, 0.0, None) * to_ft, perm, phi),
                           dtype=float)
    residual = predicted - sw
    order = np.argsort(z)
    r = residual[order]
    rho = float(np.corrcoef(r[:-1], r[1:])[0, 1]) if r.size > 2 and np.std(r) > 0 else 0.0
    rho = min(max(rho, 0.0), 0.99)
    n_eff = max(n * (1.0 - rho) / (1.0 + rho), 3.0)
    threshold = sse[best] * (1.0 + f_dist.ppf(0.95, 1, n_eff - 1.0) / (n_eff - 1.0))
    inside = candidates[sse <= threshold]
    low, high, fwl, bound = float(inside.min()), float(inside.max()), float(candidates[best]), None
    if best == candidates.size - 1:
        bound, fwl, high = "deeper", float("nan"), float("inf")
    elif best == 0:
        bound, fwl, low = "shallower", float("nan"), float("-inf")
    profile = pd.DataFrame({"FWL": candidates, "RMS": np.sqrt(sse / n)})
    return FWLFromLogs(
        fwl=fwl, low=low, high=high, rms=float(np.sqrt(sse[best] / n)), n=n, depth_unit=unit,
        profile=profile, bound=bound,
        _plot_data={"depth": z, "log_sw": sw, "model_sw": predicted, "best": float(candidates[best])},
    )
