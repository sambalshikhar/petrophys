import os

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pytest

pytest.importorskip("matplotlib")

import matplotlib.pyplot as plt

from petrophys import Well
from petrophys.visualization import create_interpretation_plot


def _interpretation_well() -> Well:
    depth = np.arange(1000.0, 1005.0, 0.5)
    well = Well("INTERP-PLOT")
    well.depth = depth
    well.depth_unit = "ft"
    well.add_curve("GR", np.linspace(40.0, 90.0, depth.size), "API", "Gamma Ray")
    well.add_curve("RDEP", np.geomspace(2.0, 50.0, depth.size), "ohm.m", "Deep Resistivity")
    well.add_curve("RHOB", np.linspace(2.65, 2.35, depth.size), "g/cc", "Density")
    well.add_curve("NPHI", np.linspace(0.08, 0.28, depth.size), "v/v", "Neutron")
    well.add_curve("VCL", np.linspace(0.15, 0.45, depth.size), "v/v", "Clay Volume")
    well.add_curve("PHIT", np.linspace(0.24, 0.12, depth.size), "v/v", "Total Porosity")
    well.add_curve("PHIE", np.linspace(0.20, 0.08, depth.size), "v/v", "Effective Porosity")
    well.add_curve("SW", np.linspace(0.25, 0.85, depth.size), "v/v", "Water Saturation")
    well.add_curve("SXO", np.linspace(0.55, 0.95, depth.size), "v/v", "Flushed Zone Saturation")
    well.add_curve("PERM", np.geomspace(0.1, 100.0, depth.size), "mD", "Permeability")
    return well


def test_interpretation_plot_uses_cpi_axis_conventions_without_hc_fill():
    plot = create_interpretation_plot(_interpretation_well(), figsize=(10, 5))
    try:
        porosity_ax = plot.axes[4]
        saturation_ax = plot.axes[5]

        assert porosity_ax.get_xlim() == (0.4, 0.0)
        assert saturation_ax.get_xlim() == (1.0, 0.0)

        legend = saturation_ax.get_legend()
        labels = [text.get_text() for text in legend.get_texts()]
        assert labels == ["Sw", "Sxo"]
        assert "Hydrocarbon" not in labels
    finally:
        plt.close(plot.fig)


def test_interpretation_plot_marks_apparent_porosity_labels():
    well = _interpretation_well()
    well.get("PHIT").metadata.description = "Apparent total porosity, density-derived from PHID"
    well.get("PHIT").metadata.custom["porosity_apparent"] = True
    well.get("PHIE").metadata.description = "Apparent effective porosity from density-derived PHIT"
    well.get("PHIE").metadata.custom["porosity_apparent"] = True

    plot = create_interpretation_plot(well, figsize=(10, 5))
    try:
        legend = plot.axes[4].get_legend()
        labels = [text.get_text() for text in legend.get_texts()]
        assert "PHIT app" in labels
        assert "PHIE app" in labels
    finally:
        plt.close(plot.fig)
