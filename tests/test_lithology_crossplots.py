"""Tests for the industry lithology crossplots (N-D, M-N, MID)."""

import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from petrophys.model import Well
from petrophys.visualization import (
    create_nd_crossplot, create_mn_plot, create_mid_plot,
    create_buckles_plot, create_hingle_plot,
)


def _well(n=120):
    rng = np.random.RandomState(0)
    w = Well("XPLOT")
    w.depth = np.arange(n, dtype=float)
    w.add_curve("NPHI", rng.uniform(0.0, 0.30, n), "v/v", "Neutron")
    w.add_curve("RHOB", rng.uniform(2.2, 2.75, n), "g/cm3", "Density")
    w.add_curve("DTC", rng.uniform(50.0, 90.0, n), "us/ft", "Sonic")
    w.add_curve("GR", rng.uniform(20.0, 120.0, n), "gAPI", "Gamma Ray")
    w.add_curve("PHIE", rng.uniform(0.02, 0.30, n), "v/v", "Eff porosity")
    w.add_curve("SW", rng.uniform(0.2, 1.0, n), "v/v", "Water sat")
    w.add_curve("PERM", rng.uniform(0.1, 1000.0, n), "mD", "Permeability")
    w.add_curve("RDEP", rng.uniform(1.0, 200.0, n), "ohm.m", "Deep res")
    w.add_curve("VCL", rng.uniform(0.0, 0.6, n), "v/v", "Clay")
    return w


def test_nd_crossplot_has_matrix_lines_and_inverted_density():
    fig = create_nd_crossplot(_well(), color_curve="GR")
    ax = fig.axes[0]
    labels = ax.get_legend_handles_labels()[1]
    assert {"Sandstone", "Limestone", "Dolomite"}.issubset(set(labels))
    # density axis inverted (high density at the bottom)
    ylo, yhi = ax.get_ylim()
    assert ylo > yhi
    plt.close(fig)


def test_mn_plot_renders_with_mineral_points():
    fig = create_mn_plot(_well(), color_curve="GR")
    ax = fig.axes[0]
    texts = [t.get_text() for t in ax.texts]
    assert "Quartz" in texts and "Calcite" in texts and "Dolomite" in texts
    plt.close(fig)


def test_mid_plot_renders():
    fig = create_mid_plot(_well(), color_curve="GR")
    ax = fig.axes[0]
    texts = [t.get_text() for t in ax.texts]
    assert "Sandstone" in texts and "Dolomite" in texts
    plt.close(fig)


def test_mn_plot_requires_sonic():
    w = _well()
    w.remove_curve("DTC")
    with pytest.raises(ValueError):
        create_mn_plot(w)


def test_buckles_plot_renders():
    fig = create_buckles_plot(_well(), color_curve="PERM")
    ax = fig.axes[0]
    assert any("BVW" in t.get_text() for t in ax.texts)
    plt.close(fig)


def test_hingle_plot_renders():
    fig = create_hingle_plot(_well(), rw=0.05, color_curve="VCL")
    ax = fig.axes[0]
    assert any("Sw=" in t.get_text() for t in ax.texts)
    plt.close(fig)


def test_buckles_requires_sw():
    w = _well()
    w.remove_curve("SW")
    with pytest.raises(ValueError):
        create_buckles_plot(w)
