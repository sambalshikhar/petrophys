import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

import petrophys as pp
import petrophys.visualization as viz
from petrophys.interpretation.core_analysis import core_data_from_well
from petrophys.model import Well
from petrophys.visualization import (
    create_bar_comparison,
    create_heatmap,
    create_line_comparison,
    create_multiwell_overlay,
)


def _mesh_edges(ax):
    mesh = ax.collections[0]
    coords = mesh.get_coordinates()
    return coords[0, :, 0], coords[:, 0, 1]


def test_heatmap_centres_cells_on_their_coordinates():
    fig, ax = create_heatmap(np.arange(6.0).reshape(2, 3), x=[0.0, 1.0, 2.0], y=[100.0, 101.0])
    x_edges, y_edges = _mesh_edges(ax)
    np.testing.assert_allclose(x_edges, [-0.5, 0.5, 1.5, 2.5])
    np.testing.assert_allclose(y_edges, [99.5, 100.5, 101.5])
    plt.close(fig)


def test_heatmap_log_spaced_bins_get_log_symmetric_cells():
    t2 = np.array([1.0, 10.0, 100.0, 1000.0])
    fig, ax = create_heatmap(np.ones((2, 4)), x=t2, y=[1.0, 2.0])
    x_edges, _ = _mesh_edges(ax)
    np.testing.assert_allclose(np.log10(x_edges), [-0.5, 0.5, 1.5, 2.5, 3.5])
    plt.close(fig)


def test_heatmap_accepts_edges_and_rejects_wrong_lengths():
    fig, ax = create_heatmap(np.ones((2, 3)), x=[0.0, 1.0, 2.0, 3.0], y=[0.0, 5.0, 10.0])
    x_edges, y_edges = _mesh_edges(ax)
    np.testing.assert_allclose(x_edges, [0.0, 1.0, 2.0, 3.0])
    np.testing.assert_allclose(y_edges, [0.0, 5.0, 10.0])
    plt.close(fig)
    with pytest.raises(ValueError, match="give 3 cell centres or 4 cell edges"):
        create_heatmap(np.ones((2, 3)), x=[0.0, 1.0], y=[0.0, 1.0])


def _well(name, offset):
    well = Well(name)
    well.depth = np.arange(1000.0, 1010.0)
    well.add_curve("GR", np.linspace(20, 120, 10) + offset, "gAPI", "Gamma ray")
    return well


@pytest.mark.parametrize("calls", [1, 2, 3])
def test_depth_axis_points_down_however_often_a_chart_draws_on_it(calls):
    fig, ax = plt.subplots()
    for i in range(calls):
        create_multiwell_overlay([_well(f"W{i}", i)], "GR", ax=ax)
        create_line_comparison([1, 2], {"a": [3, 4]}, invert_yaxis=True, ax=ax)
        create_heatmap(np.ones((2, 2)), invert_yaxis=True, ax=ax)
    assert ax.yaxis_inverted()
    plt.close(fig)


def test_horizontal_bars_keep_first_category_on_top_on_shared_axes():
    fig, ax = plt.subplots()
    create_bar_comparison(["a", "b"], [1, 2], horizontal=True, ax=ax)
    create_bar_comparison(["a", "b"], [2, 3], horizontal=True, ax=ax)
    assert ax.yaxis_inverted()
    plt.close(fig)


def test_core_data_from_well_returns_saturation_as_fraction():
    well = pp.read(pp.sample("30_6-5"))
    with pytest.warns(UserWarning, match="SWC: inferred percent core saturation"):
        core = core_data_from_well(well)
    assert np.nanmax(core.sw) <= 1.0 and np.nanmax(core.so) <= 1.0
    assert core.metadata["core_sw_normalization"]["reason"] == "inferred_percent"
    # The source curve is untouched.
    assert np.nanmax(well.get("SWC").data) > 1.0


def test_core_data_from_well_keeps_declared_fractions():
    well = Well("FRACTION-CORE")
    well.depth = np.array([1000.0, 1001.0])
    well.add_curve("PHIC", [0.2, 0.25], "v/v", "Core porosity")
    well.add_curve("SWC", [0.30, 0.45], "v/v", "Core water saturation")
    core = core_data_from_well(well)
    np.testing.assert_allclose(core.sw, [0.30, 0.45])
    assert core.metadata["core_sw_normalization"]["reason"] == "fraction"


def test_core_plot_helpers_are_public():
    for name in ("plot_calibration", "plot_poro_perm", "plot_core_on_logs",
                 "GEOLINK_LITHOLOGY_CODE_LABELS"):
        assert name in viz.__all__ and hasattr(viz, name)


def test_core_filter_keeps_non_array_metadata_whole():
    from petrophys.model.core import CoreData

    core = CoreData(
        depth=np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        porosity=np.linspace(0.1, 0.2, 6),
        metadata={"well_name": "ABCDEF", "record": {k: k for k in "abcdef"},
                  "plug_id": np.arange(6)},
    )
    kept = core.filter_by_depth(2.0, 4.0)
    assert kept.metadata["well_name"] == "ABCDEF"
    assert kept.metadata["record"] == {k: k for k in "abcdef"}
    np.testing.assert_array_equal(kept.metadata["plug_id"], [1, 2, 3])
