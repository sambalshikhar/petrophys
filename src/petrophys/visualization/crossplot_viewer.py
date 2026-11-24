"""
Crossplot visualization.
"""

from petrophys.visualization.base import Plot
from petrophys.visualization.style import _styled_plot, display_unit, save_figure

from typing import TYPE_CHECKING, Any, Optional, Tuple, Union
import warnings

if TYPE_CHECKING:
    from petrophys.model.well import Well

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

import numpy as np
from petrophys._internal.validation import warn_invalid
from petrophys.visualization._data import paired_values, note_outside_axes


class CrossPlot(Plot):
    """
    Crossplot creator for petrophysics analysis.

    Parameters
    ----------
    figsize : tuple
        Figure size
    """

    def __init__(self, figsize: Tuple[float, float] = (10, 8)):
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib is required for visualization")

        self.figsize = figsize
        Plot.__init__(self)
        # Names and colours of a class curve colouring the points (set by crossplot).
        self._categories: Tuple[Any, Any] = (None, None)

    @_styled_plot
    def create(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        color_data: Optional[np.ndarray] = None,
        x_label: str = "X",
        y_label: str = "Y",
        color_label: str = "Color",
        title: str = "Crossplot",
        x_log: bool = False,
        y_log: bool = False,
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
        cmap: str = "viridis",
        marker_size: float = 5,
        alpha: float = 0.7
    ) -> Tuple[Figure, Axes]:
        """
        Create the crossplot.

        Parameters
        ----------
        x_data : np.ndarray
            X-axis data
        y_data : np.ndarray
            Y-axis data
        color_data : np.ndarray, optional
            Data for color coding
        x_label : str
            X-axis label
        y_label : str
            Y-axis label
        color_label : str
            Colorbar label
        title : str
            Plot title
        x_log : bool
            Logarithmic X-axis
        y_log : bool
            Logarithmic Y-axis
        xlim : tuple, optional
            X-axis limits
        ylim : tuple, optional
            Y-axis limits
        cmap : str
            Colormap name
        marker_size : float
            Marker size
        alpha : float
            Marker transparency

        Returns
        -------
        tuple
            (Figure, Axes)
        """
        x_data, y_data, color_data = paired_values(x_data, y_data, color_data)
        self.fig, self.ax = plt.subplots(figsize=self.figsize)

        # Remove NaN values; points without a colour value are drawn grey.
        invalid = np.isinf(x_data) | np.isinf(y_data)
        if x_log:
            invalid |= np.isfinite(x_data) & (x_data <= 0)
        if y_log:
            invalid |= np.isfinite(y_data) & (y_data <= 0)
        warn_invalid(invalid, "Crossplot input", "cannot be represented on the selected axes",
                     "Nonfinite and nonpositive logarithmic-axis points are omitted.")
        mask = np.isfinite(x_data) & np.isfinite(y_data) & ~invalid

        x = x_data[mask]
        y = y_data[mask]

        from petrophys.visualization.style import draw_categories

        labels, colors = getattr(self, "_categories", (None, None))
        if color_data is not None and draw_categories(self.ax, x_data[mask], y_data[mask], color_data[mask],
                                                       color_label, size=marker_size, alpha=alpha,
                                                       labels=labels, colors=colors):
            pass
        elif color_data is not None and np.any(np.isfinite(color_data[mask])):
            c = color_data[mask]
            colormap = plt.get_cmap(cmap).with_extremes(bad="0.6")
            scatter = self.ax.scatter(x, y, c=c, s=marker_size, alpha=alpha,
                                     cmap=colormap, edgecolors='none',
                                     plotnonfinite=True)
            cbar = self.fig.colorbar(scatter, ax=self.ax)
            if cbar.solids is not None:  # opaque, whatever the points' transparency
                cbar.solids.set_alpha(1.0)
            cbar.set_label(color_label)
            from petrophys.visualization._data import depth_downward

            depth_downward(cbar, color_label)
        else:
            self.ax.scatter(x, y, s=marker_size, alpha=alpha, edgecolors='none')

        if x_log:
            self.ax.set_xscale('log')
        if y_log:
            self.ax.set_yscale('log')

        if xlim:
            self.ax.set_xlim(xlim)
        if ylim:
            self.ax.set_ylim(ylim)

        self.ax.set_xlabel(x_label)
        self.ax.set_ylabel(y_label)
        self.ax.set_title(title)
        self.ax.grid(True, alpha=0.3)
        note_outside_axes(self.ax, x, y)

        return self.fig, self.ax

    def add_line(
        self,
        x: Union[np.ndarray, list],
        y: Union[np.ndarray, list],
        color: str = "red",
        linestyle: str = "-",
        linewidth: float = 1,
        label: Optional[str] = None
    ) -> None:
        """Add a line to the crossplot."""
        if self.ax is None:
            raise RuntimeError("Call create() first")

        self.ax.plot(x, y, color=color, linestyle=linestyle,
                    linewidth=linewidth, label=label)

        if label:
            self.ax.legend()

    def add_matrix_points(
        self,
        matrix_lines: dict
    ) -> None:
        """
        Add matrix mineral end points.

        Parameters
        ----------
        matrix_lines : dict
            Dictionary with mineral names and (x, y) positions
        """
        if self.ax is None:
            raise RuntimeError("Call create() first")

        for mineral, (x, y) in matrix_lines.items():
            self.ax.plot(x, y, 'ko', markersize=10)
            self.ax.annotate(mineral, (x, y), textcoords="offset points",
                           xytext=(5, 5), fontsize=9)

    @_styled_plot
    def show(self) -> None:
        """Display the plot."""
        if self.fig is None:
            raise RuntimeError("Call create() first")
        self.fig.tight_layout()
        plt.show()

    @_styled_plot
    def save(self, filename: str, dpi: int = 150) -> None:
        """Save the plot to file."""
        if self.fig is None:
            raise RuntimeError("Call create() first")
        self.fig.tight_layout()
        save_figure(self.fig, filename, dpi=dpi)


@_styled_plot
def _plain_crossplot(
    well: "Well",
    x_curve: str,
    y_curve: str,
    color_curve: Optional[str] = None,
    depth_range: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 8),
    **kwargs
) -> CrossPlot:
    """
    Create a crossplot from well data.

    Parameters
    ----------
    well : Well
        Well object
    x_curve : str
        Name of X-axis curve
    y_curve : str
        Name of Y-axis curve
    color_curve : str, optional
        Name of curve for color coding
    depth_range : tuple, optional
        (top, bottom) depth range to include
    title : str, optional
        Plot title
    figsize : tuple
        Figure size
    **kwargs
        Additional arguments passed to CrossPlot.create()

    Returns
    -------
    CrossPlot
        The created crossplot
    """
    x_c = well.get(x_curve)
    y_c = well.get(y_curve)

    if x_c is None:
        raise ValueError(f"Curve '{x_curve}' not found")
    if y_c is None:
        raise ValueError(f"Curve '{y_curve}' not found")

    x_data = x_c.data
    y_data = y_c.data

    color_data = None
    color_label = ""
    if color_curve and well.get(color_curve) is None and str(color_curve).upper() in {"DEPTH", "DEPT", "MD"}:
        # Colour by the depth index itself.
        color_data = np.asarray(well.depth, dtype=float)
        unit = display_unit(getattr(well, "depth_unit", ""))
        color_label = f"Depth [{unit}]" if unit else "Depth"
    elif color_curve:
        c_c = well.get(color_curve)
        if c_c is not None:
            color_data = c_c.data
            unit = display_unit(c_c.unit)
            color_label = f"{color_curve} [{unit}]" if unit else color_curve
        else:
            warnings.warn(
                f"Colour curve '{color_curve}' not found; plotting without colour.",
                UserWarning,
                stacklevel=2,
            )

    # Apply depth range filter
    if depth_range:
        depth = well.depth
        mask = (depth >= depth_range[0]) & (depth <= depth_range[1])
        x_data = x_data[mask]
        y_data = y_data[mask]
        if color_data is not None:
            color_data = color_data[mask]

    if title is None:
        title = f"{x_curve} vs {y_curve}"

    plot = CrossPlot(figsize=figsize)
    if color_curve and well.get(color_curve) is not None:
        # A class curve's own names and colours (HFU: "no unit", the flow-unit plot's colours).
        custom = getattr(getattr(well.get(color_curve), "metadata", None), "custom", None) or {}
        plot._categories = (custom.get("category_labels"), custom.get("category_colors"))
    plot.create(
        x_data, y_data,
        color_data=color_data,
        x_label=f"{x_curve} [{display_unit(x_c.unit)}]" if display_unit(x_c.unit) else x_curve,
        y_label=f"{y_curve} [{display_unit(y_c.unit)}]" if display_unit(y_c.unit) else y_curve,
        color_label=color_label,
        title=title,
        **kwargs
    )

    return plot


#: Chart recognised from a pair of standard curve names (either order).
_POROSITY = {"PHIE", "PHIT", "PHIE_CC", "PHI", "PHIT_GAS_CORR"}
_RESISTIVITY = {"RT", "RDEP", "RD", "ILD", "LLD", "RT_INV", "AT90", "M2R9"}
_CHART_PAIRS = [
    ("nd", {"NPHI", "PHIN"}, {"RHOB"}),
    ("pef", {"PEF", "PE"}, {"RHOB"}),
    ("mn", {"N"}, {"M"}),
    ("mid", {"DTMAA", "DT_MAA"}, {"RHOMAA", "RHO_MAA"}),
    ("pickett", _POROSITY, _RESISTIVITY),
    ("buckles", _POROSITY, {"SW", "SWT", "SWE", "SW_ARCHIE"}),
    ("flow_units", {"PHI_Z", "PHIZ"}, {"RQI"}),
    ("poro_perm", {"PHIC", "CPOR", "CORE_POROSITY", "POROSITY"},
     {"KH_CORE", "K_CORE", "PERM_CORE", "CPERM", "KAIR", "CORE_PERMEABILITY", "PERMEABILITY"}),
]
_CHARTS = {"plain", "nd", "pef", "mn", "mid", "pickett", "hingle", "buckles", "flow_units", "poro_perm"}
_AUTO = object()


def _standard(name: str) -> str:
    from petrophys.preprocessing.standardization.alias_map import get_standard_mnemonic

    upper = str(name).strip().upper()
    standard = str(get_standard_mnemonic(upper) or upper).upper()
    return standard if standard else upper


def _chart_for(x: str, y: str):
    """The chart a pair of curve names asks for, and whether the pair was given swapped."""
    sx, sy = _standard(x), _standard(y)
    for chart, first, second in _CHART_PAIRS:
        if (x.upper() in first or sx in first) and (y.upper() in second or sy in second):
            return chart, False
        if (y.upper() in first or sy in first) and (x.upper() in second or sx in second):
            return chart, True
    return "plain", False


def crossplot(
    data,
    x: str,
    y: str,
    color: Optional[str] = _AUTO,  # type: ignore[assignment]
    *,
    top: Optional[float] = None,
    bottom: Optional[float] = None,
    chart: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    **options,
):
    """Crossplot two curves, with the chart their pair calls for.

    The pair is recognised by its standard names, in either order, so a
    well's own names work too (``TNPH`` is a neutron):

    =========================================  ==========================================
    ``x``, ``y``                               chart
    =========================================  ==========================================
    ``"NPHI"``, ``"RHOB"``                     density-neutron, with sandstone, limestone
                                               and dolomite lines (porosity 0-40 %)
    ``"PEF"``, ``"RHOB"``                      density-PEF, with quartz, calcite and
                                               dolomite lines and the anhydrite point
    ``"N"``, ``"M"``                           M-N, from NPHI, RHOB and DT, with the
                                               quartz-calcite-dolomite triangle
    ``"DTMAA"``, ``"RHOMAA"``                  matrix identification (MID)
    porosity, deep resistivity                 Pickett: log-log, lines of constant Sw
    (``"PHIE"``, ``"RDEP"``)                   (``chart="hingle"``: Hingle)
    porosity, ``"SW"``                         Buckles: lines of constant BVW
    ``"PHI_Z"``, ``"RQI"``                     hydraulic flow units, lines of constant FZI
    core porosity, core permeability           poro-perm, with the transform fitted to the
    (``"PHIC"``, ``"KH_CORE"``)                plugs (``transform=`` your own, or None)
    any other pair                             points, coloured by ``color``
    =========================================  ==========================================

    Parameters
    ----------
    data : Well, interpretation result, CoreData or flow-unit result
        What to plot. Core data (:func:`core_data`) and a flow-unit result
        draw their own chart; a mapping ``{title: data}`` draws one panel each,
        side by side on shared axes (poro-perm: ``{title: (core, transform)}``).
    x, y : str
        Curve names: the well's own or standard names.
    color : str, optional
        Curve to colour the points by (``"DEPTH"`` for depth). Default: the
        chart's own (GR on lithology charts, VCL on Pickett and Hingle, PERM
        on Buckles, depth on poro-perm); ``None`` for none.
    top, bottom : float, optional
        Depth window, in the well's depth unit.
    chart : str, optional
        Force a chart: ``"plain"``, ``"nd"``, ``"pef"``, ``"mn"``, ``"mid"``,
        ``"pickett"``, ``"hingle"``, ``"buckles"``, ``"flow_units"`` or
        ``"poro_perm"``.
    title, figsize : optional
        Title and size of the figure.
    **options
        The chart's own settings: ``rw``, ``a``, ``m``, ``n``, ``vcl_max`` for
        Pickett and Hingle; ``transform`` for poro-perm; ``x_log``, ``y_log``
        (default: log for resistivity and permeability), ``xlim``, ``ylim``
        and the like for a plain crossplot.

    Returns
    -------
    CrossPlot
        ``.fig`` and ``.ax``; ``.show()``, ``.save(path)``. At the end of a
        notebook cell it shows its figure.

    See Also
    --------
    petrophys.visualization.charts : each chart by name, with its own
        defaults: ``pickett_plot(result)``, ``nd_plot(well)``, ...

    Examples
    --------
    >>> import petrophys as pp
    >>> well = pp.read(pp.sample("30_6-5"))
    >>> pp.crossplot(well, "NPHI", "RHOB", top=2841, bottom=2958)   # doctest: +SKIP
    >>> pp.crossplot(pp.interpret(well), "PHIE", "RDEP", top=3223, bottom=3556)   # doctest: +SKIP
    """
    from collections.abc import Mapping

    from petrophys.model.core import CoreData
    from petrophys.visualization import charts

    if chart is not None and chart not in _CHARTS:
        raise ValueError(f"chart must be one of {sorted(_CHARTS)}; got {chart!r}")
    found, swapped = _chart_for(str(x), str(y))
    kind = chart or found
    if swapped and kind != "plain":
        x, y = y, x
    if isinstance(data, Mapping) and kind != "poro_perm":
        raise ValueError("A mapping of data draws side-by-side poro-perm panels; pass one well otherwise")
    if isinstance(data, CoreData) and kind not in {"poro_perm", "flow_units"}:
        kind = "poro_perm"
    common = {"title": title, "figsize": figsize}
    window = {"top": top, "bottom": bottom}
    colour = {} if color is _AUTO else {"color": color}

    # A pair with a chart of its own: that chart, as charts.<name>_plot draws it.
    if kind == "poro_perm":
        if not isinstance(data, (Mapping, CoreData)):
            from petrophys.interpretation.core_analysis import core_data as _core_data

            data = _core_data(data, porosity_curves=[str(x)], perm_curves=[str(y)])
        return charts.poro_perm_plot(data, **({"transform": options.pop("transform")} if "transform" in options
                                              else {}), **common)
    if kind == "flow_units" and not hasattr(data, "curves"):
        return charts.flow_unit_plot(data, **common, **options)
    if kind in {"nd", "pef", "mn"}:
        draw = {"nd": charts.nd_plot, "pef": charts.pef_plot, "mn": charts.mn_plot}[kind]
        return draw(data, **window, **colour, **common, **options)
    if kind == "mid":
        return charts.mid_plot(data, **window, **colour, **common, **options)
    if kind in {"pickett", "hingle"}:
        draw = charts.pickett_plot if kind == "pickett" else charts.hingle_plot
        return draw(data, str(x), str(y), **window, **colour, **common, **options)
    if kind == "buckles":
        return charts.buckles_plot(data, str(x), str(y), **window, **colour, **common, **options)

    from petrophys._internal.analysis_helpers import as_well

    # Resistivity and permeability on log axes, unless told otherwise.
    logs = _RESISTIVITY | {"PERM", "PERM_CC", "KH_CORE", "K", "KTIM", "KSDR", "PERM_TIMUR", "PERM_COATES"}
    options.setdefault("x_log", _standard(x) in logs or str(x).upper() in logs)
    options.setdefault("y_log", _standard(y) in logs or str(y).upper() in logs)
    depth_range = None if top is None and bottom is None else (
        -np.inf if top is None else float(top), np.inf if bottom is None else float(bottom))
    plot = _plain_crossplot(as_well(data), str(x), str(y), color_curve=None if color is _AUTO else color,
                            depth_range=depth_range, title=title, figsize=figsize or (8, 7), **options)
    if title and plot.ax is not None:
        plot.ax.set_title(title)
    return plot
