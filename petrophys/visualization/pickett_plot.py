"""
Pickett plot for Archie parameter determination.
"""

from typing import Optional, Tuple

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
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
        if color_data is not None:
            color_valid = color_data[mask]
            scatter = self.ax.scatter(
                phie_valid, rt_valid,
                c=color_valid, s=5, alpha=0.7,
                cmap='viridis', edgecolors='none'
            )
            cbar = self.fig.colorbar(scatter, ax=self.ax)
            cbar.set_label(color_label)
        else:
            self.ax.scatter(phie_valid, rt_valid, s=5, alpha=0.7, edgecolors='none')

        # Add Sw isolines
        phie_range = np.logspace(-2, 0, 100)  # 0.01 to 1.0

        for sw in sw_lines:
            # Archie equation: Rt = a * Rw / (PHIE^m * Sw^n)
            rt_line = a * rw / (phie_range ** m * sw ** n)
            self.ax.plot(phie_range, rt_line, 'k--', linewidth=0.5, alpha=0.7)
            # Label the line
            label_idx = len(phie_range) // 3
            self.ax.annotate(
                f'Sw={sw:.0%}',
                (phie_range[label_idx], rt_line[label_idx]),
                fontsize=8, alpha=0.7
            )

        # Set log scale
        self.ax.set_xscale('log')
        self.ax.set_yscale('log')

        # Set limits
        self.ax.set_xlim(0.01, 0.5)
        self.ax.set_ylim(0.1, 1000)

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
        label: Optional[str] = None
    ) -> Tuple[float, float]:
        """
        Add a trend line and calculate m (slope).

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

        Returns
        -------
        tuple
            (slope m, intercept)
        """
        if self.ax is None:
            raise RuntimeError("Call create() first")

        # Remove invalid values
        mask = ~np.isnan(phie) & ~np.isnan(rt) & (phie > 0) & (rt > 0)
        log_phie = np.log10(phie[mask])
        log_rt = np.log10(rt[mask])

        # Linear regression in log space
        coeffs = np.polyfit(log_phie, log_rt, 1)
        m_calc = -coeffs[0]  # Negative because of Archie equation
        intercept = coeffs[1]

        # Plot trend line
        phie_range = np.logspace(-2, 0, 100)
        rt_trend = 10 ** (intercept - m_calc * np.log10(phie_range))

        line_label = label or f'm = {m_calc:.2f}'
        self.ax.plot(phie_range, rt_trend, color=color, linewidth=2, label=line_label)
        self.ax.legend()

        return m_calc, intercept

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


def create_pickett_plot(
    well: "Well",
    rw: float,
    phie_curve: str = "PHIE",
    rt_curve: str = "RT",
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
    rt_curve : str
        Name of resistivity curve
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
    rt_c = well.get(rt_curve)

    if phie_c is None:
        raise ValueError(f"Curve '{phie_curve}' not found")
    if rt_c is None:
        raise ValueError(f"Curve '{rt_curve}' not found")

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
        # Default to depth
        color_data = well.depth
        color_label = "Depth"

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

    plot = PickettPlot(figsize=figsize)
    plot.create(
        phie, rt, rw,
        color_data=color_data,
        a=a, m=m, n=n,
        title=title,
        color_label=color_label
    )

    return plot
