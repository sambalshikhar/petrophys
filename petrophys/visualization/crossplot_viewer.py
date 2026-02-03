"""
Crossplot visualization.
"""

from typing import Optional, Tuple, Union

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    import matplotlib.colors as mcolors
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

import numpy as np


class CrossPlot:
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
        self.fig: Optional[Figure] = None
        self.ax: Optional[Axes] = None

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
        self.fig, self.ax = plt.subplots(figsize=self.figsize)

        # Remove NaN values
        mask = ~np.isnan(x_data) & ~np.isnan(y_data)
        if color_data is not None:
            mask &= ~np.isnan(color_data)

        x = x_data[mask]
        y = y_data[mask]

        if color_data is not None:
            c = color_data[mask]
            scatter = self.ax.scatter(x, y, c=c, s=marker_size, alpha=alpha,
                                     cmap=cmap, edgecolors='none')
            cbar = self.fig.colorbar(scatter, ax=self.ax)
            cbar.set_label(color_label)
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

    def show(self) -> None:
        """Display the plot."""
        if self.fig is None:
            raise RuntimeError("Call create() first")
        plt.tight_layout()
        plt.show()

    def save(self, filename: str, dpi: int = 150) -> None:
        """Save the plot to file."""
        if self.fig is None:
            raise RuntimeError("Call create() first")
        plt.tight_layout()
        self.fig.savefig(filename, dpi=dpi, bbox_inches='tight')


def create_crossplot(
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
    if color_curve:
        c_c = well.get(color_curve)
        if c_c is not None:
            color_data = c_c.data
            color_label = f"{color_curve} [{c_c.unit}]"

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
    plot.create(
        x_data, y_data,
        color_data=color_data,
        x_label=f"{x_curve} [{x_c.unit}]",
        y_label=f"{y_curve} [{y_c.unit}]",
        color_label=color_label,
        title=title,
        **kwargs
    )

    return plot


def create_density_neutron_crossplot(
    well: "Well",
    depth_range: Optional[Tuple[float, float]] = None,
    figsize: Tuple[float, float] = (10, 8)
) -> CrossPlot:
    """
    Create a standard density-neutron crossplot.

    Parameters
    ----------
    well : Well
        Well object
    depth_range : tuple, optional
        Depth range to include
    figsize : tuple
        Figure size

    Returns
    -------
    CrossPlot
        The crossplot
    """
    plot = create_crossplot(
        well,
        x_curve="NPHI",
        y_curve="RHOB",
        color_curve="GR",
        depth_range=depth_range,
        title="Density-Neutron Crossplot",
        figsize=figsize,
        xlim=(-0.05, 0.45),
        ylim=(2.95, 1.95)  # Inverted for standard display
    )

    # Add matrix lines
    matrix_points = {
        "Sandstone": (0.0, 2.65),
        "Limestone": (0.0, 2.71),
        "Dolomite": (0.02, 2.87),
    }
    plot.add_matrix_points(matrix_points)

    return plot
