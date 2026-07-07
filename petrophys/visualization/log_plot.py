"""
Basic log plot functionality.
"""

from typing import Optional, List, Dict, Tuple, Union
import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


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

        if n_tracks == 1:
            self.axes = [self.axes]

        # Set depth axis
        for ax in self.axes:
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3)

        return self.fig, self.axes

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

        curve = self.well.get(curve_name)
        if curve is None:
            raise ValueError(f"Curve '{curve_name}' not found")

        depth = self.well.depth
        data = curve.data

        ax = self.axes[track]
        ax.plot(data, depth, color=color, linewidth=linewidth,
                linestyle=linestyle, label=label or curve_name)

        if log_scale:
            ax.set_xscale('log')

        if xlim:
            ax.set_xlim(xlim)

        # On a multi-curve track the legend names the curves; keep the axis
        # label from silently becoming whichever curve was added last. Track
        # the units seen per track: one curve -> "NAME [unit]"; several curves
        # sharing a unit -> "[unit]"; genuinely mixed units -> blank.
        # Fraction-porosity spellings count as the same unit.
        equiv = {"v/v", "m3/m3", "frac", "fraction", "dec", "decimal"}
        unit = (curve.unit or "").strip()
        canonical = "v/v" if unit.lower() in equiv else unit
        if not hasattr(self, "_track_units"):
            self._track_units = {}
        info = self._track_units.setdefault(track, {"n": 0, "units": set()})
        info["n"] += 1
        info["units"].add(canonical)
        if info["n"] == 1:
            ax.set_xlabel(f"{curve_name} [{unit}]")
        elif len(info["units"]) == 1:
            ax.set_xlabel(f"[{next(iter(info['units']))}]")
        else:
            ax.set_xlabel("")
        ax.legend(loc='upper right', fontsize=8)

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

        curve = self.well.get(curve_name)
        if curve is None:
            raise ValueError(f"Curve '{curve_name}' not found")

        depth = self.well.depth
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

    def add_title(self, title: str) -> None:
        """Add title to the figure."""
        if self.fig is None:
            raise RuntimeError("Call create_figure() first")

        well_name = self.well.name or "Unknown Well"
        self.fig.suptitle(f"{title}\n{well_name}", fontsize=12)

    def show(self) -> None:
        """Display the figure."""
        if self.fig is None:
            raise RuntimeError("Call create_figure() first")

        plt.tight_layout()
        plt.show()

    def save(self, filename: str, dpi: int = 150) -> None:
        """Save the figure to file."""
        if self.fig is None:
            raise RuntimeError("Call create_figure() first")

        plt.tight_layout()
        self.fig.savefig(filename, dpi=dpi, bbox_inches='tight')


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
