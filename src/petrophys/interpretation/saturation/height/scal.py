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

from petrophys._internal.advice import Advice
import re
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


#: Pressure units read from a column header, as psi per unit.
_PC_UNITS = {"psi": 1.0, "psia": 1.0, "psig": 1.0, "kpa": 0.145038, "mpa": 145.038,
             "bar": 14.5038, "atm": 14.6959}


def _bare(name: Any) -> str:
    """A header without its unit in brackets, upper case, words joined by ``_``."""
    text = re.sub(r"[\(\[].*?[\)\]]", " ", str(name))
    return "_".join(re.findall(r"[A-Z0-9]+", text.upper()))


def _header_unit(name: Any) -> str:
    """The unit a header gives: in brackets (``Pc (kPa)``), else a pressure
    unit as its last word (``Pc_kPa``, ``PC_BAR``); lower case, "" for none."""
    match = re.search(r"[\(\[]\s*([^\)\]]+?)\s*[\)\]]", str(name))
    if match:
        return match.group(1).strip().lower()
    words = re.findall(r"[a-z0-9]+", str(name).lower())
    return words[-1] if len(words) > 1 and words[-1] in _PC_UNITS else ""


def _nonwetting(name: Any) -> bool:
    """Whether a saturation header names the non-wetting phase (mercury, gas, oil injected)."""
    words = set(_bare(name).split("_"))
    return bool(words & {"HG", "SHG", "MERCURY", "SNW", "NONWETTING", "NON"}) and not words & {"SW", "WATER"}


def _keyword_match(bare: str, role: str) -> bool:
    words = set(bare.split("_"))
    if role == "sample":
        return bool(words & {"SAMPLE", "PLUG"})
    if role == "depth":
        return "DEPTH" in words
    if role == "porosity":
        return bool(words & {"POROSITY", "PHI"})
    if role == "permeability":
        return bool(words & {"PERM", "PERMEABILITY", "KLINKENBERG"})
    if role == "pc":
        return bool(words & {"PC", "PCHG", "PRESSURE", "PCAP"})
    if role == "sw":
        return bool(words & {"SATURATION", "SAT", "SW", "SHG"})
    return False


def _column(frame: "pd.DataFrame", given: Optional[str], role: str, required: bool = True) -> Optional[str]:
    if given is not None:
        if given not in frame.columns:
            raise KeyError(f"No column {given!r}; the table has {list(frame.columns)}")
        return given
    bare = {_bare(name): name for name in frame.columns}
    for candidate in _COLUMNS[role]:
        if candidate in bare:
            return str(bare[candidate])
    matches = [name for key, name in bare.items() if _keyword_match(key, role)]
    if len(matches) == 1:
        return str(matches[0])
    if required:
        found = f" (several match: {matches})" if matches else ""
        raise KeyError(f"No single {role} column found{found}; name it with {role}=; "
                       f"the table has {list(frame.columns)}")
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
    pore system; a bimodal curve is described by its dominant system. Fewer
    than 4 points with mercury entered, no point with a positive pressure and
    a mercury volume, or the same mercury volume at every pressure (no curve
    to fit) raise ValueError.
    """
    from scipy.optimize import least_squares

    pc = np.asarray(pc_psi, dtype=float).reshape(-1)
    bv = np.asarray(bv_percent, dtype=float).reshape(-1)
    if pc.size != bv.size:
        raise ValueError(f"pc_psi has {pc.size} values and bv_percent {bv.size}; they must match")
    ok = np.isfinite(pc) & np.isfinite(bv) & (pc > 0)
    pc, bv = pc[ok], bv[ok]
    if pc.size == 0:
        raise ValueError("no point with a positive pressure and a mercury volume to fit a Thomeer hyperbola")
    if float(np.ptp(bv)) <= 1e-9 * max(abs(float(bv.max())), 1.0):
        raise ValueError(f"the mercury volume is {float(bv.max()):.3g}% at every pressure, so there is "
                         "no curve to fit a Thomeer hyperbola to")
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
        """The number of plugs."""
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
        """The curves, Pc (log) against Sw, coloured by permeability.

        Drawn on ``ax`` when given, else on a figure of its own; returns the
        figure.
        """
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
    pc_unit: Optional[str] = None,
    saturation: Optional[str] = None,
    sheet_name: Any = 0,
) -> CapillaryPressureData:
    """Read plug capillary-pressure curves from a CSV or Excel table (or a DataFrame).

    One row per measured point: the plug (``sample``), the capillary pressure
    (``pc``), the saturation (``sw``), and the plug's ``porosity``,
    ``permeability`` (mD) and, optionally, ``depth``. Columns are found by
    their names, with any unit in brackets ignored (``Plug No.``, ``Injection
    Pressure (psia)``, ``Helium Porosity (%)``, ``Klinkenberg Perm (mD)`` ...),
    unless named. Porosity and saturation in percent become v/v.

    ``pc_unit`` is the pressure unit (``psi``, ``kPa``, ``MPa``, ``bar``,
    ``atm``); by default the unit in the header, in brackets (``Pc (kPa)``)
    or as its last word (``Pc_kPa``, ``PC_BAR``), else psi. ``saturation`` is
    ``"wetting"`` (Sw) or ``"nonwetting"`` (mercury saturation, converted to
    Sw = 1 - SHg); by default a header naming mercury (``Hg Saturation``) is
    non-wetting. A saturation that rises with pressure in most plugs is not
    Sw, and is refused with a pointer to the other ``saturation`` setting.
    ``system`` names the laboratory fluids, which set the interfacial tension
    and contact angle (``ift``, ``theta`` override them). ``sheet_name`` is
    the sheet of an Excel workbook (default: the first).

    A plug named on its first row only (merged cells in a spreadsheet) owns
    the rows down to the next plug, and those rows take its porosity,
    permeability and depth where theirs are blank. Rows without a number for
    Pc, saturation, porosity or permeability are dropped.

    A named column that is absent, or a required column that cannot be
    found, raises KeyError; an unknown ``system``, ``pc_unit`` or
    ``saturation``, or a table with no usable row, raises ValueError.
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
    if frame.empty:
        raise ValueError("The capillary-pressure table has no rows")
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
    # A plug named on its first row only (merged cells) owns the rows below it.
    unnamed = table["SAMPLE"].isna() | (table["SAMPLE"].astype(str).str.strip() == "")
    if unnamed.any():
        table["SAMPLE"] = table["SAMPLE"].mask(unnamed).ffill()
        for key in ("POROSITY", "PERMEABILITY", "DEPTH"):
            if key in table:
                table[key] = table[key].where(~unnamed | table[key].notna(),
                                              table.groupby("SAMPLE")[key].ffill())
    table = table.dropna(subset=["SAMPLE", "PC", "SW", "POROSITY", "PERMEABILITY"]).reset_index(drop=True)
    if table.empty:
        raise ValueError(
            "No row of the table has a plug name and a number for Pc, saturation, porosity and "
            f"permeability (columns read: {', '.join(str(name) for name in names.values() if name is not None)})"
        )
    unit = (pc_unit or _header_unit(names["PC"]) or "psi").strip().lower()
    if unit not in _PC_UNITS:
        raise ValueError(Advice(f"Unknown pressure unit {unit!r}; give pc_unit= as one of {sorted(_PC_UNITS)}",
            ('pc_unit',)))
    table["PC"] = table["PC"] * _PC_UNITS[unit]
    kind = saturation or ("nonwetting" if _nonwetting(names["SW"]) else "wetting")
    if kind not in ("wetting", "nonwetting"):
        raise ValueError(f'saturation must be "wetting" or "nonwetting"; got {saturation!r}')
    table["SW"] = _fraction(table["SW"].to_numpy(dtype=float))
    if kind == "nonwetting":
        table["SW"] = 1.0 - table["SW"]
    table["POROSITY"] = _fraction(table["POROSITY"].to_numpy(dtype=float))
    rising = [
        np.corrcoef(group["PC"], group["SW"])[0, 1] > 0
        for _, group in table.groupby("SAMPLE") if len(group) > 2 and group["SW"].std() > 0
    ]
    if rising and np.mean(rising) > 0.5:
        count = f"{int(np.sum(rising))} of {len(rising)} plugs"
        if kind == "nonwetting":
            why = 'saturation="nonwetting" was given' if saturation else "its header names the non-wetting phase"
            raise ValueError(
                f"The saturation in {names['SW']!r} was read as non-wetting ({why}) and converted to "
                f"Sw = 1 - S, which rises with capillary pressure in {count}: the column falls with "
                'pressure, so it is the wetting-phase Sw already. Pass saturation="wetting".'
            )
        raise ValueError(
            f"The saturation in {names['SW']!r} rises with capillary pressure in "
            f"{count}, so it is not the wetting-phase Sw: "
            'mercury (non-wetting) saturation? Pass saturation="nonwetting".'
        )
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

    The curves are converted to reservoir ``fluids`` (``ift``, dyne/cm, and
    ``theta``, degrees, override that fluid system's interfacial tension and
    contact angle), each point's J is computed from its Pc,
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
    increases downwards, in ``depth_unit``: feet or metres, as any spelling
    of them (``"ft"``, ``"F"``, ``"m"``, ``"M"``, ``"metres"``). A
    ``depth_unit`` that is neither raises ValueError.
    """
    from petrophys.model.units import normalize_depth_unit

    if sg_water <= sg_hc:
        raise ValueError("sg_water must exceed sg_hc")
    unit = normalize_depth_unit(depth_unit, strict=True)
    height_ft = float(entry_pressure_psi) / (WATER_GRADIENT_PSI_PER_FT * (sg_water - sg_hc))
    return float(fwl_depth) - (height_ft * 0.3048 if unit == "m" else height_ft)


@dataclass
class FWLFromLogs:
    """The free-water level that best explains the log water saturation.

    ``fwl`` is the best depth, ``low`` and ``high`` its 95% range, ``rms`` the
    Sw misfit there, ``r2`` the share of the log Sw variance the model
    explains there, ``n`` the samples used, and ``profile`` the misfit at
    each candidate depth (columns ``FWL`` and ``RMS``). Depths are in
    ``depth_unit``. ``search`` is the shallowest and deepest candidate depth
    and ``sample_depths`` the shallowest and deepest sample used.

    Where the least misfit is at an edge of the search the FWL is not
    resolved, ``fwl`` is NaN and ``bound`` says what is known. ``"deeper"``:
    the misfit keeps falling to the bottom of the search, so the FWL lies
    below ``low`` (``high`` is infinite); where the search reaches below the
    deepest sample the samples are all above the transition zone.
    ``"shallower"``: the least misfit is at the top of the search, so the FWL
    is at or above that depth and above ``high`` (``low`` is minus infinity);
    where the search starts at or above the shallowest sample the samples
    are all in the water leg.

    ``range_cut`` tells whether the 95% range of a resolved FWL is cut by the
    search: ``"top"``, ``"bottom"`` or ``"both"`` where the range reaches
    that edge of the search and may extend beyond it, else ``""``.
    """

    fwl: float
    low: float
    high: float
    rms: float
    n: int
    depth_unit: str
    profile: "pd.DataFrame" = field(repr=False)
    bound: Optional[str] = None
    r2: float = float("nan")
    _plot_data: Dict[str, Any] = field(default_factory=dict, repr=False)
    search: Tuple[float, float] = (float("nan"), float("nan"))
    sample_depths: Tuple[float, float] = (float("nan"), float("nan"))
    range_cut: str = ""

    def __repr__(self) -> str:
        unit = self.depth_unit
        if self.bound == "deeper":
            where = f"FWL deeper than {self.low:.1f} {unit}"
            if self.search[1] >= self.sample_depths[1]:
                where += " (no transition zone logged)"
        elif self.bound == "shallower":
            where = f"FWL at or above the top of the search, {self.search[0]:.1f} {unit}"
            if self.search[0] <= self.sample_depths[0]:
                where += " (the samples used are in the water leg)"
        else:
            where = f"FWL {self.fwl:.1f} {unit}, 95% range {self.low:.1f}-{self.high:.1f}"
            if self.range_cut:
                edge = "top and bottom" if self.range_cut == "both" else self.range_cut
                where += f" (cut by the {edge} of the search)"
        return f"FWLFromLogs({where}, RMS Sw misfit {self.rms:.3f}, R2 {self.r2:.2f}, n={self.n})"

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
    depths as ``(shallowest, deepest)`` (default: from the shallowest sample
    used, less 1/400 of the samples' depth span, to one span below the
    deepest). Takes a well
    or an interpretation result. ``r2`` is how much of the log Sw the model
    explains at the best FWL. Compare the estimate with a pressure FWL where
    one exists.

    ``sw_curve``, ``perm_curve`` and ``phi_curve`` each name a curve or
    several in order of preference; at each depth the first of them with a
    value is used. Height is measured along ``tvd_curve`` when one is named;
    otherwise the depth index is taken as true vertical depth, with a
    warning. ``search`` and the depths of the result are in the unit and on
    the datum of that depth: the ``tvd_curve``'s when one is named, else the
    well's depth index. ``top`` and ``bottom`` are always depths on the
    well's depth index (measured depth), also when ``tvd_curve`` is named.

    Where the least misfit is at the top or the bottom of the search the FWL
    is not resolved and the result's ``bound`` says on which side it lies;
    where the 95% range reaches an edge of the search its ``range_cut`` says
    so (see :class:`FWLFromLogs`). Widen ``search`` in either case.

    A well with none of the named curves, or fewer than 20 samples with Sw,
    permeability and porosity, raises ValueError, as does a ``search`` that
    is not two finite depths with the shallower first.
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
    if search is None:
        # From just above the shallowest sample, in steps of 3/800 of the span.
        low_search, high_search, count = float(z.min()) - span / 400.0, float(z.max()) + span, 535
    else:
        count = 401
        try:
            low_search, high_search = (float(value) for value in search)
        except (TypeError, ValueError):
            raise ValueError(f"search must be two depths, (shallowest, deepest); got {search!r}") from None
        if not (np.isfinite(low_search) and np.isfinite(high_search) and low_search < high_search):
            raise ValueError("search must be two finite depths with the shallower first, "
                             f"(shallowest, deepest) in {unit}; got {search!r}")
    candidates = np.linspace(low_search, high_search, count)
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
    range_cut = ""
    if best == candidates.size - 1:
        bound, fwl, high = "deeper", float("nan"), float("inf")
    elif best == 0:
        bound, fwl, low = "shallower", float("nan"), float("-inf")
    else:
        cut_top, cut_bottom = bool(sse[0] <= threshold), bool(sse[-1] <= threshold)
        range_cut = "both" if cut_top and cut_bottom else "top" if cut_top else "bottom" if cut_bottom else ""
        if range_cut:
            edge = "top and bottom" if range_cut == "both" else range_cut
            warnings.warn(f"estimate_fwl_from_logs: the 95% range of the FWL reaches the {edge} of the "
                          f"search ({low_search:.1f}-{high_search:.1f} {unit}) and may extend beyond it; "
                          "widen search= to see the whole range.", UserWarning, stacklevel=2)
    ss_tot = float(np.sum((sw - sw.mean()) ** 2))
    r2 = 1.0 - float(sse[best]) / ss_tot if ss_tot > 0 else float("nan")
    profile = pd.DataFrame({"FWL": candidates, "RMS": np.sqrt(sse / n)})
    return FWLFromLogs(
        fwl=fwl, low=low, high=high, rms=float(np.sqrt(sse[best] / n)), n=n, depth_unit=unit, r2=r2,
        profile=profile, bound=bound, search=(float(low_search), float(high_search)),
        sample_depths=(float(z.min()), float(z.max())), range_cut=range_cut,
        _plot_data={"depth": z, "log_sw": sw, "model_sw": predicted, "best": float(candidates[best])},
    )
