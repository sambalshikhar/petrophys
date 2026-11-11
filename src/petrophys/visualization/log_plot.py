"""
Basic log plot functionality.
"""

from petrophys.visualization.style import _styled_plot, save_figure

from typing import TYPE_CHECKING, Optional, List, Dict, Tuple, Union
import numpy as np

if TYPE_CHECKING:
    from petrophys.model.well import Well

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


from petrophys.preprocessing.standardization.alias_map import get_standard_mnemonic

#: Scales a curve takes when no xlim is given: limestone-compatible density and
#: neutron (RHOB 1.95-2.95 g/cc against NPHI 0.45 to -0.15 v/v), as on log prints.
_DEFAULT_SCALES: Dict[str, Tuple[float, float]] = {
    "RHOB": (1.95, 2.95),
    "NPHI": (0.45, -0.15),
}


def _fit_x_to_window(ax: "Axes", top: float, bottom: float) -> None:
    """Scale the x axis to the curves drawn between ``top`` and ``bottom``."""
    low, high = sorted((float(top), float(bottom)))
    values = []
    for line in ax.lines:
        x = np.asarray(line.get_xdata(), dtype=float)
        y = np.asarray(line.get_ydata(), dtype=float)
        if x.shape != y.shape:
            continue
        keep = np.isfinite(x) & np.isfinite(y) & (y >= low) & (y <= high)
        if ax.get_xscale() == "log":
            keep &= x > 0
        values.append(x[keep])
    values = np.concatenate(values) if values else np.array([])
    if values.size == 0:
        return
    lo, hi = float(values.min()), float(values.max())
    if ax.get_xscale() == "log":
        ax.set_xlim(lo / 1.25, hi * 1.25)
    else:
        pad = 0.05 * (hi - lo) or 0.05 * abs(hi) or 1.0
        ax.set_xlim(lo - pad, hi + pad)


class LogPlot:
    """
    Log plot creator for well data visualization.

    Parameters
    ----------
    well : Well
        Well object with curve data
    figsize : tuple
        Figure size (width, height) in inches
    """

    def __init__(self, well: "Well", figsize: Tuple[float, float] = (12, 10)):
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib is required for visualization")

        self.well = well
        self.figsize = figsize
        self.fig: Optional[Figure] = None
        self.axes: List[Axes] = []
        # Curves are drawn in standard units (percent porosity as v/v), as
        # petrophys.plot draws them; the caller's well is unchanged.
        from petrophys.visualization.log_viewer import _display_well

        self._shown = _display_well(well, interpreted=True)
        self._fixed_xlim: set = set()
        self._twins: Dict[Tuple[int, str], Axes] = {}

    def _curve(self, curve_name: str):
        """The curve named, or found by its standard name as petrophys.plot finds it (RDEP finds RT)."""
        from petrophys.visualization.log_viewer import _resolve_curve_name_for_well

        resolved = _resolve_curve_name_for_well(self._shown, curve_name)
        if resolved is None:
            raise ValueError(
                f"Curve '{curve_name}' not found by its own name or as a standard name; "
                f"the well has {list(self._shown.curves)}"
            )
        return self._shown.get(resolved)

    @_styled_plot
    def create_figure(self, n_tracks: int = 3) -> Tuple[Figure, List[Axes]]:
        """
        Create figure with specified number of tracks.

        Parameters
        ----------
        n_tracks : int
            Number of log tracks

        Returns
        -------
        tuple
            (Figure, list of Axes)
        """
        self.fig, self.axes = plt.subplots(
            1, n_tracks,
            figsize=self.figsize,
            sharey=True
        )
        self._track_units = {}
        self._fixed_xlim = set()
        self._twins = {}

        if n_tracks == 1:
            self.axes = [self.axes]

        # Set depth axis
        for ax in self.axes:
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3)
        unit = getattr(self.well, "depth_unit", "") or ""
        self.axes[0].set_ylabel(f"Depth [{unit}]" if unit else "Depth")

        return self.fig, self.axes

    @_styled_plot
    def add_curve(
        self,
        track: int,
        curve_name: str,
        color: str = "black",
        linewidth: float = 0.5,
        linestyle: str = "-",
        log_scale: bool = False,
        xlim: Optional[Tuple[float, float]] = None,
        label: Optional[str] = None
    ) -> None:
        """
        Add a curve to a track.

        Parameters
        ----------
        track : int
            Track index (0-based)
        curve_name : str
            Name of the curve
        color : str
            Line color
        linewidth : float
            Line width
        linestyle : str
            Line style
        log_scale : bool
            Use logarithmic x-axis
        xlim : tuple, optional
            X-axis limits
        label : str, optional
            Legend label
        """
        if self.fig is None:
            raise RuntimeError("Call create_figure() first")

        curve = self._curve(curve_name)
        depth = self._shown.depth
        data = curve.data
        if label is None:
            label = curve_name if curve.mnemonic.upper() == curve_name.upper() else f"{curve.mnemonic} ({curve_name})"

        # Track the units seen per track. A curve whose unit differs from the
        # track's first curve (NPHI in v/v beside RHOB in g/cc) gets its own
        # scale on a second axis, labelled at the top, as log headers show it.
        equiv = {"v/v", "m3/m3", "frac", "fraction", "dec", "decimal"}
        unit = (curve.unit or "").strip()
        canonical = "v/v" if unit.lower() in equiv else unit
        if not hasattr(self, "_track_units"):
            self._track_units = {}
        info = self._track_units.setdefault(track, {"units": [], "names": {}})
        base = self.axes[track]
        if not info["units"] or canonical == info["units"][0]:
            ax = base
        else:
            ax = self._twins.get((track, canonical))
            if ax is None:
                ax = base.twiny()
                ax.grid(False)
                self._twins[(track, canonical)] = ax
        if canonical not in info["units"]:
            info["units"].append(canonical)
        info["names"].setdefault(canonical, []).append(curve.mnemonic)

        ax.plot(data, depth, color=color, linewidth=linewidth,
                linestyle=linestyle, label=label)

        if log_scale:
            ax.set_xscale('log')

        if xlim is None:
            xlim = _DEFAULT_SCALES.get(str(get_standard_mnemonic(curve.mnemonic)).upper())
        if xlim:
            ax.set_xlim(xlim)
            self._fixed_xlim.add(id(ax))

        # One curve on an axis -> "NAME [unit]"; several curves of one unit -> "[unit]".
        names = info["names"][canonical]
        text = f"{names[0]} [{unit}]" if len(names) == 1 else f"[{unit}]"
        if ax is base:
            ax.set_xlabel(text)
        else:
            ax.set_xlabel(text, color=color)
            ax.tick_params(axis="x", colors=color)
        # One legend per track, listing the curves of all its scales.
        twins = [twin for (index, _), twin in self._twins.items() if index == track]
        handles, labels = [], []
        for each in [base, *twins]:
            found_handles, found_labels = each.get_legend_handles_labels()
            handles += found_handles
            labels += found_labels
            if each.get_legend() is not None:
                each.get_legend().remove()
        (twins[-1] if twins else base).legend(handles, labels, loc='upper right', fontsize=8)

    def add_fill(
        self,
        track: int,
        curve_name: str,
        fill_value: float = 0,
        color: str = "yellow",
        alpha: float = 0.3,
        where: str = "right"
    ) -> None:
        """
        Add fill between curve and a value.

        Parameters
        ----------
        track : int
            Track index
        curve_name : str
            Name of the curve
        fill_value : float
            Value to fill to
        color : str
            Fill color
        alpha : float
            Fill transparency
        where : str
            'left' or 'right' of fill_value
        """
        if self.fig is None:
            raise RuntimeError("Call create_figure() first")

        curve = self._curve(curve_name)
        depth = self._shown.depth
        data = curve.data

        ax = self.axes[track]

        if where == "right":
            ax.fill_betweenx(depth, fill_value, data,
                            where=data > fill_value,
                            color=color, alpha=alpha)
        else:
            ax.fill_betweenx(depth, data, fill_value,
                            where=data < fill_value,
                            color=color, alpha=alpha)

    def set_depth_range(
        self,
        top: Optional[float] = None,
        bottom: Optional[float] = None
    ) -> None:
        """
        Set depth range for all tracks.

        Parameters
        ----------
        top : float, optional
            Top depth
        bottom : float, optional
            Bottom depth
        """
        if self.fig is None:
            raise RuntimeError("Call create_figure() first")

        if top is None:
            top = self.well.start_depth
        if bottom is None:
            bottom = self.well.stop_depth

        for ax in self.axes:
            ax.set_ylim(bottom, top)
        for ax in [*self.axes, *getattr(self, "_twins", {}).values()]:
            if id(ax) not in getattr(self, "_fixed_xlim", set()):
                _fit_x_to_window(ax, top, bottom)

    @_styled_plot
    def add_title(self, title: str) -> None:
        """Add title to the figure."""
        if self.fig is None:
            raise RuntimeError("Call create_figure() first")

        well_name = self.well.name or "Unknown Well"
        # The well name is added once: a title that already names the well keeps it.
        text = title if well_name in str(title) else f"{title}\n{well_name}"
        self.fig.suptitle(text, fontsize=12)

    @_styled_plot
    def show(self) -> None:
        """Display the figure."""
        if self.fig is None:
            raise RuntimeError("Call create_figure() first")

        self.fig.tight_layout()
        plt.show()

    @_styled_plot
    def save(self, filename: str, dpi: int = 150) -> None:
        """Save the figure to file."""
        if self.fig is None:
            raise RuntimeError("Call create_figure() first")

        self.fig.tight_layout()
        save_figure(self.fig, filename, dpi=dpi)


@_styled_plot
def create_log_plot(
    well: "Well",
    curves: List[Dict],
    depth_range: Optional[Tuple[float, float]] = None,
    title: str = "Log Plot",
    figsize: Tuple[float, float] = (12, 10)
) -> LogPlot:
    """
    Create a log plot with specified curves.

    Parameters
    ----------
    well : Well
        Well object
    curves : list
        List of curve specifications, each dict with:
        - track: int
        - name: str
        - color: str (optional)
        - log_scale: bool (optional)
        - xlim: tuple (optional)
    depth_range : tuple, optional
        (top, bottom) depth range
    title : str
        Plot title
    figsize : tuple
        Figure size

    Returns
    -------
    LogPlot
        The created plot object

    Example
    -------
    >>> curves = [
    ...     {"track": 0, "name": "GR", "color": "green"},
    ...     {"track": 1, "name": "RHOB", "color": "red"},
    ...     {"track": 1, "name": "NPHI", "color": "blue"},
    ...     {"track": 2, "name": "RT", "color": "black", "log_scale": True},
    ... ]
    >>> plot = create_log_plot(well, curves)
    >>> plot.show()
    """
    # Determine number of tracks
    if not curves:
        raise ValueError("create_log_plot requires at least one curve spec")
    n_tracks = max(c.get("track", 0) for c in curves) + 1

    plot = LogPlot(well, figsize=figsize)
    plot.create_figure(n_tracks)

    for curve_spec in curves:
        plot.add_curve(
            track=curve_spec.get("track", 0),
            curve_name=curve_spec["name"],
            color=curve_spec.get("color", "black"),
            linewidth=curve_spec.get("linewidth", 0.5),
            linestyle=curve_spec.get("linestyle", "-"),
            log_scale=curve_spec.get("log_scale", False),
            xlim=curve_spec.get("xlim"),
            label=curve_spec.get("label")
        )

    if depth_range:
        plot.set_depth_range(depth_range[0], depth_range[1])
    else:
        plot.set_depth_range()

    plot.add_title(title)

    return plot
