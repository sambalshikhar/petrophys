"""
Triple combo log plot.
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


class TripleComboPlot:
    """
    Standard triple combo log plot.

    Track 1: GR, Caliper
    Track 2: Resistivity (log scale)
    Track 3: Density, Neutron
    """

    def __init__(self, well: "Well", figsize: Tuple[float, float] = (14, 12)):
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib is required for visualization")

        self.well = well
        self.figsize = figsize
        self.fig: Optional[Figure] = None
        self.axes = []

    def create(
        self,
        depth_range: Optional[Tuple[float, float]] = None,
        gr_range: Tuple[float, float] = (0, 150),
        rt_range: Tuple[float, float] = (0.2, 2000),
        rhob_range: Tuple[float, float] = (1.95, 2.95),
        nphi_range: Tuple[float, float] = (0.45, -0.15)
    ) -> Tuple[Figure, list]:
        """
        Create the triple combo plot.

        Parameters
        ----------
        depth_range : tuple, optional
            (top, bottom) depth range
        gr_range : tuple
            GR display range
        rt_range : tuple
            Resistivity display range
        rhob_range : tuple
            Density display range
        nphi_range : tuple
            Neutron display range (reversed for limestone scale)

        Returns
        -------
        tuple
            (Figure, list of Axes)
        """
        self.fig, self.axes = plt.subplots(1, 3, figsize=self.figsize, sharey=True)

        depth = self.well.depth

        # Set depth range
        if depth_range:
            top, bottom = depth_range
        else:
            top = self.well.start_depth or np.nanmin(depth)
            bottom = self.well.stop_depth or np.nanmax(depth)

        # Track 1: GR and Caliper
        ax1 = self.axes[0]
        self._plot_track1(ax1, gr_range)

        # Track 2: Resistivity
        ax2 = self.axes[1]
        self._plot_track2(ax2, rt_range)

        # Track 3: Density-Neutron
        ax3 = self.axes[2]
        self._plot_track3(ax3, rhob_range, nphi_range)

        # Common settings
        for ax in self.axes:
            ax.set_ylim(bottom, top)
            ax.grid(True, alpha=0.3)

        self.axes[0].set_ylabel("Depth")

        # Title
        well_name = self.well.name or "Unknown Well"
        self.fig.suptitle(f"Triple Combo - {well_name}", fontsize=14)

        plt.tight_layout()

        return self.fig, self.axes

    def _plot_track1(self, ax: "Axes", gr_range: Tuple[float, float]) -> None:
        """Plot GR and Caliper track."""
        depth = self.well.depth

        # GR
        gr_curve = self.well.get("GR")
        if gr_curve is not None:
            ax.plot(gr_curve.data, depth, color='green', linewidth=0.5, label='GR')
            ax.fill_betweenx(depth, gr_range[0], gr_curve.data,
                            alpha=0.3, color='yellow')
            ax.set_xlim(gr_range)

        # Caliper (secondary axis)
        cali_curve = self.well.get("CALI")
        if cali_curve is not None:
            ax2 = ax.twiny()
            ax2.plot(cali_curve.data, depth, color='black', linewidth=0.5, label='CALI')
            ax2.set_xlabel("CALI [in]", fontsize=8)
            ax2.tick_params(axis='x', labelsize=8)

            # Bit size
            bs_curve = self.well.get("BS")
            if bs_curve is not None:
                ax2.plot(bs_curve.data, depth, color='black', linewidth=1,
                        linestyle='--', label='BS')

        ax.set_xlabel("GR [API]", fontsize=8)
        ax.legend(loc='upper right', fontsize=8)

    def _plot_track2(self, ax: "Axes", rt_range: Tuple[float, float]) -> None:
        """Plot resistivity track."""
        depth = self.well.depth

        # Deep resistivity
        for name in ["RT", "LLD", "ILD", "RDEP", "AT90"]:
            curve = self.well.get(name)
            if curve is not None:
                ax.plot(curve.data, depth, color='red', linewidth=0.5, label=name)
                break

        # Medium resistivity
        for name in ["RMED", "ILM", "LLS", "AT60"]:
            curve = self.well.get(name)
            if curve is not None:
                ax.plot(curve.data, depth, color='blue', linewidth=0.5, label=name)
                break

        # Shallow resistivity
        for name in ["RSHA", "MSFL", "SFL", "AT10"]:
            curve = self.well.get(name)
            if curve is not None:
                ax.plot(curve.data, depth, color='green', linewidth=0.5, label=name)
                break

        ax.set_xscale('log')
        ax.set_xlim(rt_range)
        ax.set_xlabel("Resistivity [ohm.m]", fontsize=8)
        ax.legend(loc='upper right', fontsize=8)

    def _plot_track3(
        self,
        ax: "Axes",
        rhob_range: Tuple[float, float],
        nphi_range: Tuple[float, float]
    ) -> None:
        """Plot density-neutron track."""
        depth = self.well.depth

        # Density
        rhob_curve = self.well.get("RHOB")
        if rhob_curve is not None:
            ax.plot(rhob_curve.data, depth, color='red', linewidth=0.5, label='RHOB')
            ax.set_xlim(rhob_range)

        # Neutron (secondary axis)
        nphi_curve = self.well.get("NPHI")
        if nphi_curve is not None:
            ax2 = ax.twiny()
            ax2.plot(nphi_curve.data, depth, color='blue', linewidth=0.5, label='NPHI')
            ax2.set_xlim(nphi_range)
            ax2.set_xlabel("NPHI [v/v]", fontsize=8)

            # Crossover fill
            if rhob_curve is not None:
                # Normalize for comparison
                rhob_norm = (rhob_curve.data - rhob_range[0]) / (rhob_range[1] - rhob_range[0])
                nphi_norm = (nphi_curve.data - nphi_range[0]) / (nphi_range[1] - nphi_range[0])

                # Fill crossover (gas indicator)
                ax.fill_betweenx(
                    depth,
                    rhob_curve.data,
                    rhob_range[0] + (nphi_norm * (rhob_range[1] - rhob_range[0])),
                    where=rhob_norm < nphi_norm,
                    alpha=0.3,
                    color='yellow',
                    label='Crossover'
                )

        ax.set_xlabel("RHOB [g/cc]", fontsize=8)
        ax.legend(loc='upper right', fontsize=8)

    def show(self) -> None:
        """Display the plot."""
        if self.fig is None:
            raise RuntimeError("Call create() first")
        plt.show()

    def save(self, filename: str, dpi: int = 150) -> None:
        """Save the plot to file."""
        if self.fig is None:
            raise RuntimeError("Call create() first")
        self.fig.savefig(filename, dpi=dpi, bbox_inches='tight')


def create_triple_combo(
    well: "Well",
    depth_range: Optional[Tuple[float, float]] = None,
    figsize: Tuple[float, float] = (14, 12)
) -> TripleComboPlot:
    """
    Create a triple combo plot.

    Parameters
    ----------
    well : Well
        Well object with log data
    depth_range : tuple, optional
        (top, bottom) depth range
    figsize : tuple
        Figure size

    Returns
    -------
    TripleComboPlot
        The created plot
    """
    plot = TripleComboPlot(well, figsize=figsize)
    plot.create(depth_range=depth_range)
    return plot
