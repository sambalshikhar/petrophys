"""
Pickett plot for Archie parameter determination.
"""

from petrophys.visualization.style import _styled_plot, save_figure

from typing import TYPE_CHECKING, Optional, Tuple
import warnings

if TYPE_CHECKING:
    from petrophys.model.well import Well

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    from matplotlib.ticker import FormatStrFormatter, NullFormatter
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

import numpy as np


class PickettPlot:
    """
    Pickett crossplot for determining Archie parameters.

    Log(Rt) vs Log(PHIE) with Sw isolines.
    """

    def __init__(self, figsize: Tuple[float, float] = (10, 8)):
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib is required for visualization")

        self.figsize = figsize
        self.fig: Optional[Figure] = None
        self.ax: Optional[Axes] = None

    @_styled_plot
    def create(
        self,
        phie: np.ndarray,
        rt: np.ndarray,
        rw: float,
        color_data: Optional[np.ndarray] = None,
        a: float = 1.0,
        m: float = 2.0,
        n: float = 2.0,
        sw_lines: list = [0.2, 0.4, 0.6, 0.8, 1.0],
        title: str = "Pickett Plot",
        color_label: str = "Depth"
    ) -> Tuple[Figure, Axes]:
        """
        Create the Pickett plot.

        Parameters
        ----------
        phie : np.ndarray
            Effective porosity
        rt : np.ndarray
            True resistivity
        rw : float
            Formation water resistivity
        color_data : np.ndarray, optional
            Data for color coding (e.g., depth)
        a : float
            Tortuosity factor
        m : float
            Cementation exponent
        n : float
            Saturation exponent
        sw_lines : list
            Sw values for iso-saturation lines
        title : str
            Plot title
        color_label : str
            Colorbar label

        Returns
        -------
        tuple
            (Figure, Axes)
        """
        self.fig, self.ax = plt.subplots(figsize=self.figsize)

        # Remove NaN and invalid values
        mask = ~np.isnan(phie) & ~np.isnan(rt) & (phie > 0) & (rt > 0)
        phie_valid = phie[mask]
        rt_valid = rt[mask]

        # Plot data points
        if color_data is not None and np.any(np.isfinite(color_data[mask])):
            color_valid = color_data[mask]
            # Points without a colour value are drawn grey, not dropped.
            colormap = plt.get_cmap('viridis').with_extremes(bad="0.6")
            scatter = self.ax.scatter(
                phie_valid, rt_valid,
                c=color_valid, s=5, alpha=0.7,
                cmap=colormap, edgecolors='none', plotnonfinite=True
            )
            cbar = self.fig.colorbar(scatter, ax=self.ax)
            cbar.set_label(color_label)
        else:
            self.ax.scatter(phie_valid, rt_valid, s=5, alpha=0.7, edgecolors='none')

        # Add Sw isolines
        phie_range = np.logspace(-2, 0, 100)  # 0.01 to 1.0

        for i, sw in enumerate(sorted(sw_lines)):
            # Archie equation: Rt = a * Rw / (PHIE^m * Sw^n)
            rt_line = a * rw / (phie_range ** m * sw ** n)
            self.ax.plot(phie_range, rt_line, 'k--', linewidth=0.5, alpha=0.7)
            # Each label sits further along its line than the one before, so
            # the labels of closely spaced lines step apart instead of stacking.
            label_idx = min(len(phie_range) // 3 + 8 * i, len(phie_range) - 1)
            self.ax.annotate(
                f'Sw={sw:.0%}',
                (phie_range[label_idx], rt_line[label_idx]),
                fontsize=8, alpha=0.7,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.6, pad=0.5),
            )

        # Set log scale
        self.ax.set_xscale('log')
        self.ax.set_yscale('log')

        # Set limits. The porosity axis spans less than two decades, so
        # matplotlib would label minor ticks and the labels would collide.
        self.ax.set_xlim(0.01, 0.5)
        self.ax.set_ylim(0.1, 1000)
        self.ax.set_xticks([0.01, 0.02, 0.05, 0.1, 0.2, 0.5])
        self.ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
        self.ax.xaxis.set_minor_formatter(NullFormatter())

        # Labels
        self.ax.set_xlabel('Effective Porosity (PHIE)')
        self.ax.set_ylabel('True Resistivity (Rt) [ohm.m]')
        self.ax.set_title(f'{title}\na={a}, m={m}, n={n}, Rw={rw:.3f}')
        self.ax.grid(True, alpha=0.3, which='both')

        return self.fig, self.ax

    def add_trend_line(
        self,
        phie: np.ndarray,
        rt: np.ndarray,
        color: str = "red",
        label: Optional[str] = None,
        method: str = "rma",
    ) -> Tuple[float, float]:
        """
        Add a trend line and calculate m (slope).

        The line is the fit of
        :func:`petrophys.interpretation.electrical.pickett_analysis` on the
        same samples (finite, Rt > 0, PHIE > 0.01): by default the reduced
        major axis of log Rt and log PHIE.

        Parameters
        ----------
        phie : np.ndarray
            Porosity data
        rt : np.ndarray
            Resistivity data
        color : str
            Line color
        label : str, optional
            Line label
        method : {"rma", "ols"}, default "rma"
            Reduced major axis or least squares of log Rt on log PHIE.

        Returns
        -------
        tuple
            (slope m, intercept)
        """
        from petrophys.interpretation.electrical.cementation.pickett_analysis import (
            fit_pickett_line,
            pickett_samples,
        )

        if self.ax is None:
            raise RuntimeError("Call create() first")

        phie = np.asarray(phie, dtype=float)
        rt = np.asarray(rt, dtype=float)
        mask = pickett_samples(rt, phie)
        if int(np.count_nonzero(mask)) < 2:
            raise ValueError("add_trend_line needs >= 2 valid (phie > 0.01, rt > 0) points")
        fit = fit_pickett_line(np.log10(phie[mask]), np.log10(rt[mask]), method=method)
        m_calc = fit["m"]
        intercept = fit["intercept"]

        # Plot trend line
        phie_range = np.logspace(-2, 0, 100)
        rt_trend = 10 ** (intercept - m_calc * np.log10(phie_range))

        line_label = label or f'm = {m_calc:.2f}'
        self.ax.plot(phie_range, rt_trend, color=color, linewidth=2, label=line_label)
        self.ax.legend()

        return m_calc, intercept

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
def create_pickett_plot(
    well: "Well",
    rw: float,
    phie_curve: str = "PHIE",
    rt_curve: Optional[str] = None,
    color_curve: Optional[str] = None,
    depth_range: Optional[Tuple[float, float]] = None,
    a: float = 1.0,
    m: float = 2.0,
    n: float = 2.0,
    figsize: Tuple[float, float] = (10, 8)
) -> PickettPlot:
    """
    Create a Pickett plot from well data.

    Parameters
    ----------
    well : Well
        Well object
    rw : float
        Formation water resistivity
    phie_curve : str
        Name of porosity curve
    rt_curve : str, optional
        Name of resistivity curve (default: RT, else RDEP)
    color_curve : str, optional
        Name of curve for color coding
    depth_range : tuple, optional
        (top, bottom) depth range
    a : float
        Tortuosity factor
    m : float
        Cementation exponent
    n : float
        Saturation exponent
    figsize : tuple
        Figure size

    Returns
    -------
    PickettPlot
        The created plot
    """
    phie_c = well.get(phie_curve)
    rt_names = [rt_curve] if rt_curve is not None else ["RT", "RDEP"]
    rt_c = next((well.get(name) for name in rt_names if well.get(name) is not None), None)

    if phie_c is None:
        raise ValueError(f"Curve '{phie_curve}' not found")
    if rt_c is None:
        raise ValueError(f"Curve {' or '.join(repr(name) for name in rt_names)} not found")

    phie = phie_c.data
    rt = rt_c.data

    color_data = None
    color_label = "Value"
    if color_curve:
        c_c = well.get(color_curve)
        if c_c is not None:
            color_data = c_c.data
            color_label = f"{color_curve} [{c_c.unit}]"
        else:
            warnings.warn(
                f"Colour curve '{color_curve}' not found; plotting without colour.",
                UserWarning,
                stacklevel=2,
            )
    else:
        # Default to depth
        color_data = well.depth
        depth_unit = getattr(well, "depth_unit", "") or ""
        color_label = f"Depth [{depth_unit}]" if depth_unit else "Depth"

    # Apply depth range
    if depth_range:
        depth = well.depth
        mask = (depth >= depth_range[0]) & (depth <= depth_range[1])
        phie = phie[mask]
        rt = rt[mask]
        if color_data is not None:
            color_data = color_data[mask]

    well_name = well.name or "Unknown Well"
    title = f"Pickett Plot - {well_name}"
    if depth_range:
        # Name the window, so a water-leg check reads as one.
        unit = f" {well.depth_unit}" if getattr(well, "depth_unit", None) else ""
        title += f" ({min(depth_range):g}-{max(depth_range):g}{unit})"

    plot = PickettPlot(figsize=figsize)
    plot.create(
        phie, rt, rw,
        color_data=color_data,
        a=a, m=m, n=n,
        title=title,
        color_label=color_label
    )

    return plot
