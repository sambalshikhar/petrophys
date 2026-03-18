"""
Interpretation plot with petrophysical results.
"""

from typing import Optional, Tuple, List

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

import numpy as np


class InterpretationPlot:
    """
    Full interpretation plot with raw logs and computed results.

    Tracks:
    1. GR/Caliper
    2. Resistivity
    3. Density/Neutron
    4. VCL
    5. Porosity (PHIT, PHIE)
    6. Saturation (Sw, Sxo)
    7. Permeability
    8. Net Pay Flags
    """

    def __init__(self, well: "Well", figsize: Tuple[float, float] = (20, 14)):
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib is required for visualization")

        self.well = well
        self.figsize = figsize
        self.fig: Optional[Figure] = None
        self.axes: List[Axes] = []

    def create(
        self,
        depth_range: Optional[Tuple[float, float]] = None
    ) -> Tuple[Figure, List[Axes]]:
        """
        Create the interpretation plot.

        Parameters
        ----------
        depth_range : tuple, optional
            (top, bottom) depth range

        Returns
        -------
        tuple
            (Figure, list of Axes)
        """
        n_tracks = 8
        self.fig, self.axes = plt.subplots(
            1, n_tracks,
            figsize=self.figsize,
            sharey=True
        )

        depth = self.well.depth

        # Set depth range
        if depth_range:
            top, bottom = depth_range
        else:
            top = self.well.start_depth or np.nanmin(depth)
            bottom = self.well.stop_depth or np.nanmax(depth)

        # Plot each track
        self._plot_track1_gr(self.axes[0])
        self._plot_track2_resistivity(self.axes[1])
        self._plot_track3_density_neutron(self.axes[2])
        self._plot_track4_vcl(self.axes[3])
        self._plot_track5_porosity(self.axes[4])
        self._plot_track6_saturation(self.axes[5])
        self._plot_track7_permeability(self.axes[6])
        self._plot_track8_flags(self.axes[7])

        # Common settings
        for ax in self.axes:
            ax.set_ylim(bottom, top)
            ax.grid(True, alpha=0.3)

        self.axes[0].set_ylabel("Depth")

        # Title
        well_name = self.well.name or "Unknown Well"
        self.fig.suptitle(f"Interpretation - {well_name}", fontsize=14)

        plt.tight_layout()

        return self.fig, self.axes

    def _plot_track1_gr(self, ax: "Axes") -> None:
        """GR track."""
        depth = self.well.depth

        gr = self.well.get("GR")
        if gr is not None:
            ax.plot(gr.data, depth, 'g-', linewidth=0.5, label='GR')
            ax.fill_betweenx(depth, 0, gr.data, alpha=0.3, color='green')
            ax.set_xlim(0, 150)

        ax.set_xlabel("GR [API]")
        ax.legend(loc='upper right', fontsize=7)

    def _plot_track2_resistivity(self, ax: "Axes") -> None:
        """Resistivity track."""
        depth = self.well.depth

        for names, color in [(["RT", "RDEP", "LLD", "ILD"], "red"),
                              (["RMED", "ILM", "LLS"], "blue"),
                              (["RSHA", "MSFL", "SFL", "RXO"], "green")]:
            for name in names:
                curve = self.well.get(name)
                if curve is not None:
                    ax.plot(curve.data, depth, color=color, linewidth=0.5, label=name)
                    break

        ax.set_xscale('log')
        ax.set_xlim(0.2, 2000)
        ax.set_xlabel("Resistivity [ohm.m]")
        ax.legend(loc='upper right', fontsize=7)

    def _plot_track3_density_neutron(self, ax: "Axes") -> None:
        """Density-Neutron track."""
        depth = self.well.depth

        rhob = self.well.get("RHOB")
        if rhob is not None:
            ax.plot(rhob.data, depth, 'r-', linewidth=0.5, label='RHOB')
            ax.set_xlim(1.95, 2.95)

        nphi = self.well.get("NPHI")
        if nphi is not None:
            ax2 = ax.twiny()
            ax2.plot(nphi.data, depth, 'b-', linewidth=0.5, label='NPHI')
            ax2.set_xlim(0.45, -0.15)
            ax2.set_xlabel("NPHI [v/v]", fontsize=8)

        ax.set_xlabel("RHOB [g/cc]")
        ax.legend(loc='upper right', fontsize=7)

    def _plot_track4_vcl(self, ax: "Axes") -> None:
        """VCL track."""
        depth = self.well.depth

        vcl = self.well.get("VCL")
        if vcl is not None:
            ax.plot(vcl.data, depth, 'brown', linewidth=0.5, label='VCL')
            ax.fill_betweenx(depth, 0, vcl.data, alpha=0.5, color='brown')

        ax.set_xlim(0, 1)
        ax.set_xlabel("VCL [v/v]")
        ax.legend(loc='upper right', fontsize=7)

    def _plot_track5_porosity(self, ax: "Axes") -> None:
        """Porosity track."""
        depth = self.well.depth

        phit = self.well.get("PHIT")
        if phit is not None:
            ax.plot(phit.data, depth, 'b-', linewidth=0.5, label='PHIT')

        phie = self.well.get("PHIE")
        if phie is not None:
            ax.plot(phie.data, depth, 'r-', linewidth=0.5, label='PHIE')
            ax.fill_betweenx(depth, 0, phie.data, alpha=0.3, color='yellow')

        ax.set_xlim(0, 0.4)
        ax.set_xlabel("Porosity [v/v]")
        ax.legend(loc='upper right', fontsize=7)

    def _plot_track6_saturation(self, ax: "Axes") -> None:
        """Saturation track."""
        depth = self.well.depth

        sw = self.well.get("SW")
        if sw is not None:
            ax.plot(sw.data, depth, 'b-', linewidth=0.5, label='Sw')
            ax.fill_betweenx(depth, sw.data, 1, alpha=0.3, color='green',
                            label='Hydrocarbon')

        sxo = self.well.get("SXO")
        if sxo is not None:
            ax.plot(sxo.data, depth, 'c--', linewidth=0.5, label='Sxo')

        ax.set_xlim(0, 1)
        ax.set_xlabel("Saturation [v/v]")
        ax.legend(loc='upper right', fontsize=7)

    def _plot_track7_permeability(self, ax: "Axes") -> None:
        """Permeability track."""
        depth = self.well.depth

        perm = self.well.get("PERM")
        if perm is not None:
            ax.plot(perm.data, depth, 'k-', linewidth=0.5, label='Perm')

        ax.set_xscale('log')
        ax.set_xlim(0.01, 10000)
        ax.set_xlabel("Perm [mD]")
        ax.legend(loc='upper right', fontsize=7)

    def _plot_track8_flags(self, ax: "Axes") -> None:
        """Net pay flags track."""
        depth = self.well.depth

        # Net Pay
        net_pay = self.well.get("NET_PAY")
        if net_pay is not None:
            ax.fill_betweenx(depth, 0, net_pay.data * 0.33,
                            alpha=0.8, color='green', label='Net Pay')

        # Net Reservoir
        net_res = self.well.get("NET_RES")
        if net_res is not None:
            ax.fill_betweenx(depth, 0.33, 0.33 + net_res.data * 0.33,
                            alpha=0.6, color='yellow', label='Net Res')

        # Net Rock
        net_rock = self.well.get("NET_ROCK")
        if net_rock is not None:
            ax.fill_betweenx(depth, 0.66, 0.66 + net_rock.data * 0.34,
                            alpha=0.4, color='orange', label='Net Rock')

        ax.set_xlim(0, 1)
        ax.set_xticks([])
        ax.set_xlabel("Flags")
        ax.legend(loc='upper right', fontsize=7)

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


def create_interpretation_plot(
    well: "Well",
    depth_range: Optional[Tuple[float, float]] = None,
    figsize: Tuple[float, float] = (20, 14)
) -> InterpretationPlot:
    """
    Create an interpretation plot.

    Parameters
    ----------
    well : Well
        Well object with raw and computed curves
    depth_range : tuple, optional
        (top, bottom) depth range
    figsize : tuple
        Figure size

    Returns
    -------
    InterpretationPlot
        The created plot
    """
    plot = InterpretationPlot(well, figsize=figsize)
    plot.create(depth_range=depth_range)
    return plot
