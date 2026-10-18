"""Regression tests for visualization defects fixed before release."""

import os
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from defusedxml import ElementTree as ET

import petrophys.visualization.log_viewer as lv
from petrophys.model.well import Well


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _well(n=10, start=1000.0, step=1.0):
    w = Well("T")
    w.depth = start + step * np.arange(n, dtype=float)
    return w


def _fill_spans(ax):
    """(min, max) depth of every fill_betweenx polygon on ``ax``."""
    spans = []
    for coll in ax.collections:
        for path in coll.get_paths():
            v = path.vertices
            if len(v):
                spans.append((float(v[:, 1].min()), float(v[:, 1].max())))
    return spans


# --- log viewer -----------------------------------------------------------

def test_display_well_converts_percent_saturation_and_porosity_labels():
    w = _well(20, step=0.5)
    w.add_curve("SW", np.full(20, 40.0), "%", "")
    w.add_curve("PHID", np.full(20, 20.0), "%", "")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        display = lv._display_well(w)
    np.testing.assert_allclose(display.get("SW").data, 0.40)
    assert display.get("PHID").unit == "v/v"
    np.testing.assert_allclose(display.get("PHID").data, 0.20)
    # The caller's well is untouched.
    assert w.get("SW").unit == "%" and w.get("SW").data[0] == 40.0


def test_depth_curve_only_well_keeps_metres():
    d = 2000.0 + 0.5 * np.arange(20)
    w = Well("D")
    w.add_curve("DEPTH", d, "m", "")
    w.add_curve("GR", np.full(20, 50.0), "gAPI", "")
    display = lv._display_well(w)
    np.testing.assert_allclose(display.get("DEPTH").data, d)
    assert display.get("DEPTH").unit == "m"


def test_reversed_depth_range_keeps_depth_increasing_down():
    w = _well(10)
    w.add_curve("GR", np.linspace(20, 120, 10), "gAPI", "")
    fig = lv.create_log_viewer(w, template_defaults="raw", depth_range=(1008, 1002))
    assert fig.axes[0].get_ylim() == (1008.0, 1002.0)


def test_template_zero_values_are_kept_and_nan_int_uses_default():
    track = lv._parse_track(ET.fromstring('<track left="0.45" right="0"/>'))
    assert track.right == 0.0
    curve = lv._parse_curve(
        ET.fromstring('<curve curve_name="A" alpha="0" line_width="0" fill_alpha="0"/>')
    )
    assert (curve.alpha, curve.line_width, curve.fill_alpha) == (0.0, 0.0, 0.0)
    assert lv._as_int("nan", 5) == 5 and lv._as_int("inf", 5) == 5


def test_lithology_style_match_is_word_bounded():
    assert lv._lookup_lithology_style("Basalt", lv.LITHOLOGY_LABEL_COLORS, None) != \
        lv.LITHOLOGY_LABEL_COLORS["halite"]
    assert lv._lookup_lithology_style("Rock salt", lv.LITHOLOGY_LABEL_COLORS, None) == \
        lv.LITHOLOGY_LABEL_COLORS["salt"]


def test_lithology_mapping_table_edits_are_reread(tmp_path, monkeypatch):
    table = tmp_path / "codes.csv"
    table.write_text("code,lithology\n1,First Label\n")
    monkeypatch.setenv("PETROPHYS_LITHOLOGY_MAPPING", str(table))
    lv._load_lithology_mapping_tables.cache_clear()
    try:
        assert lv._default_geolink_lithology_mapping()[1] == "First Label"
        table.write_text("code,lithology\n1,Edited Label\n")
        stat = table.stat()
        os.utime(table, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10_000_000))
        assert lv._default_geolink_lithology_mapping()[1] == "Edited Label"
    finally:
        lv._load_lithology_mapping_tables.cache_clear()


def test_sample_cell_fill_covers_single_samples():
    depth = 1000.0 + 0.5 * np.arange(10)
    active = np.zeros(10, dtype=bool)
    active[3] = True
    active[6:9] = True
    fig, ax = plt.subplots()
    edges, mask = lv._sample_cell_fill(depth, active)
    ax.fill_betweenx(edges, 0, 1, where=mask)
    spans = sorted(_fill_spans(ax))
    assert spans == [(1001.25, 1001.75), (1002.75, 1004.25)]


def test_flag_track_draws_single_sample_flag():
    w = _well(10, step=0.5)
    flag = np.zeros(10)
    flag[3] = 1.0
    w.add_curve("NET_PAY", flag, "", "")
    viewer = lv.LogViewer(w, template_defaults="petrophysics")
    viewer.create()
    spans = [s for ax in viewer.axes for s in _fill_spans(ax)]
    assert (1001.25, 1001.75) in spans


def _xover_template(tmp_path):
    xml = tmp_path / "xover.xml"
    xml.write_text(
        '<template><track display_name="NPHI_RHOB">'
        '<curve curve_name="NPHI" left="0.45" right="-0.15" left_crossover="RHOB"/>'
        '<curve curve_name="RHOB" left="1.95" right="2.95"/>'
        "</track></template>"
    )
    return xml


@pytest.mark.parametrize("rhob, expect_fill", [(2.55, False), (2.10, True)])
def test_crossover_fill_uses_other_curve_scale(tmp_path, rhob, expect_fill):
    w = _well(10, step=0.5)
    w.add_curve("NPHI", np.full(10, 0.30), "v/v", "")
    w.add_curve("RHOB", np.full(10, rhob), "g/cm3", "")
    viewer = lv.LogViewer(w, template_xml_path=_xover_template(tmp_path))
    viewer.create()
    xs = [
        path.vertices[:, 0]
        for ax in viewer.axes for coll in ax.collections for path in coll.get_paths()
        if len(path.vertices)
    ]
    if expect_fill:
        assert xs
        lo, hi = min(x.min() for x in xs), max(x.max() for x in xs)
        assert lo == pytest.approx(0.15) and hi == pytest.approx(0.25)
    else:
        assert not xs


def test_mineral_stack_leaves_gap_for_missing_volume_and_labels_phit():
    n = 6
    w = _well(n, step=0.5)
    vcl = np.full(n, 0.2)
    vcl[1] = np.nan
    w.add_curve("VCLAY", vcl, "v/v", "")
    w.add_curve("VQUAR", np.full(n, 0.5), "v/v", "")
    w.add_curve("PHIT", np.full(n, 0.3), "v/v", "")
    viewer = lv.LogViewer(w, template_defaults="minerals")
    viewer.create()
    # No boundary line is drawn at the depth of the missing clay volume.
    boundaries = [
        np.asarray(line.get_xdata(), float)
        for line in viewer.axes[0].lines if line.get_color() == "0.32"
    ]
    assert len(boundaries) == 3
    for x in boundaries:
        assert np.isnan(x[1]) and np.all(np.isfinite(np.delete(x, 1)))
    texts = [t.get_text() for ax in viewer.header_axes for t in ax.texts]
    assert any("PHIT" in t for t in texts)
    assert not any(t.strip() == "PHIE" for t in texts)


def test_colormap_fill_honours_colour_value_range(tmp_path):
    xml = tmp_path / "cmap.xml"
    xml.write_text(
        '<template><track display_name="GR" left="0" right="150">'
        '<curve curve_name="GR" left="0" right="150" fill="left" '
        'fill_color_map="viridis" right_color_value="300"/>'
        "</track></template>"
    )
    w = _well(10)
    w.add_curve("GR", np.linspace(0.0, 150.0, 10), "gAPI", "")
    viewer = lv.LogViewer(w, template_xml_path=xml)
    viewer.create()
    colors = [
        coll.get_facecolor()[0][:3]
        for ax in viewer.axes for coll in ax.collections if len(coll.get_facecolor())
    ]
    cmap = plt.get_cmap("viridis")
    np.testing.assert_allclose(colors[-1], cmap(0.5)[:3], atol=1e-6)


# --- lithology crossplots ---------------------------------------------------

def test_mn_plot_converts_percent_and_us_per_m():
    from petrophys.visualization.lithology_crossplots import create_mn_plot

    w = _well(3, start=0.0)
    w.add_curve("RHOB", [2.65, 2.368, 2.87], "g/cm3", "")
    w.add_curve("NPHI", [-2.0, 20.0, 2.0], "%", "")
    w.add_curve("DTC", [55.5 / 0.3048, 75.88 / 0.3048, 43.5 / 0.3048], "us/m", "")
    offsets = np.asarray(create_mn_plot(w, color_curve=None).axes[0].collections[0].get_offsets())
    np.testing.assert_allclose(offsets[:, 0], [0.618, 0.585, 0.524], atol=2e-3)
    np.testing.assert_allclose(offsets[:, 1], [0.809, 0.827, 0.778], atol=2e-3)


def test_crossplot_points_with_missing_colour_are_drawn():
    from petrophys.visualization.lithology_crossplots import (
        create_buckles_plot,
        create_nd_crossplot,
    )

    w = _well(4, start=0.0)
    w.add_curve("NPHI", [0.1, 0.2, 0.25, 0.15], "v/v", "")
    w.add_curve("RHOB", [2.4, 2.3, 2.2, 2.5], "g/cc", "")
    w.add_curve("GR", [40.0, np.nan, np.nan, 60.0], "gAPI", "")
    sc = create_nd_crossplot(w).axes[0].collections[0]
    assert not np.ma.getmaskarray(sc.get_offsets()).any()

    b = _well(4, start=0.0)
    b.add_curve("PHIE", [0.2, 0.1, 0.15, 0.25], "v/v", "")
    b.add_curve("SW", [0.3, 0.6, 0.5, 0.2], "v/v", "")
    b.add_curve("PERM", [100.0, 0.0, np.nan, 5.0], "mD", "")
    sc = create_buckles_plot(b).axes[0].collections[0]
    assert len(sc.get_offsets()) == 4
    assert not np.ma.getmaskarray(sc.get_offsets()).any()


def test_nd_crossplot_reports_points_outside_axes():
    from petrophys.visualization.lithology_crossplots import create_nd_crossplot

    w = _well(4, start=0.0)
    w.add_curve("NPHI", [0.1, 0.5, 0.6, 0.15], "v/v", "")
    w.add_curve("RHOB", [2.4, 2.3, 2.2, 2.5], "g/cc", "")
    texts = [t.get_text() for t in create_nd_crossplot(w, color_curve=None).axes[0].texts]
    assert "2 of 4 points outside the plotted range" in texts


def test_crossplot_viewer_keeps_points_with_missing_colour_and_warns_missing_curve():
    from petrophys.visualization.crossplot_viewer import create_crossplot

    w = _well(4, start=0.0)
    w.add_curve("NPHI", [0.1, 0.2, 0.25, 0.15], "v/v", "")
    w.add_curve("RHOB", [2.4, 2.3, 2.2, 2.5], "g/cc", "")
    w.add_curve("GR", [40.0, np.nan, np.nan, 60.0], "gAPI", "")
    plot = create_crossplot(w, "NPHI", "RHOB", color_curve="GR")
    assert len(plot.ax.collections[0].get_offsets()) == 4
    with pytest.warns(UserWarning, match="GRX"):
        create_crossplot(w, "NPHI", "RHOB", color_curve="GRX")


def test_pickett_warns_on_missing_colour_curve():
    from petrophys.visualization.pickett_plot import create_pickett_plot

    w = _well(4, start=0.0)
    w.add_curve("PHIE", [0.1, 0.2, 0.25, 0.15], "v/v", "")
    w.add_curve("RDEP", [10.0, 2.0, 5.0, 20.0], "ohm.m", "")
    with pytest.warns(UserWarning, match="VCLX"):
        create_pickett_plot(w, rw=0.05, rt_curve="RDEP", color_curve="VCLX")


# --- other plots ------------------------------------------------------------

def test_saturation_height_legend_matches_power_swir_form():
    from petrophys.interpretation.saturation.height.saturation_height_module import (
        JFunctionModel,
    )
    from petrophys.visualization.saturation_height_plot import create_saturation_height_plot

    model = JFunctionModel(a=0.3, b=-0.5, swir=0.1, form="power_swir")
    fig = create_saturation_height_plot(model, j=[0.5, 1.0], sw=[0.5, 0.4], data_label="SW_INPUT")
    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert "SW_INPUT" in labels
    assert any(label.startswith("Sw = 0.1 + 0.9") for label in labels)


def test_zone_flags_draw_single_sample_and_full_runs():
    from petrophys.visualization.zone_flags import create_net_pay_summary_plot, plot_zone_flags

    w = _well(10)
    pay = np.zeros(10)
    pay[2] = 1.0
    res = np.zeros(10)
    res[5:8] = 1.0
    w.add_curve("NET_PAY", pay, "", "")
    w.add_curve("NET_RES", res, "", "")
    fig, ax = plot_zone_flags(w, ["NET_PAY", "NET_RES"])
    spans = sorted(_fill_spans(ax))
    assert spans == [(1001.5, 1002.5), (1004.5, 1007.5)]
    assert ax.get_ylabel() == f"Depth [{w.depth_unit}]"
    fig, axes = create_net_pay_summary_plot(w)
    assert (1001.5, 1002.5) in _fill_spans(axes[1])


def test_batch_export_rejects_missing_input_dir(tmp_path):
    from petrophys.visualization.batch_export import export_interpretation_plots

    with pytest.raises(NotADirectoryError):
        export_interpretation_plots(tmp_path / "missing", tmp_path / "out")


def test_log_plot_labels_depth_axis():
    from petrophys.visualization.log_plot import LogPlot

    w = _well(5)
    _, axes = LogPlot(w).create_figure(2)
    assert axes[0].get_ylabel() == f"Depth [{w.depth_unit}]"


def test_core_on_logs_handles_missing_core_depth_and_maps_corecal():
    from petrophys.model.core import CoreData
    from petrophys.visualization.core import plot_core_on_logs

    w = _well(20)
    w.add_curve("PHIE_CORECAL", np.full(20, 0.2), "v/v", "")
    core = CoreData(depth=np.array([1005.0, 1010.0]), porosity=np.array([0.21, 0.19]))
    fig, axes = plot_core_on_logs(w, core, curves=["PHIE_CORECAL"])
    assert any(c.get_label() == "Core PHI" for c in axes[0].collections)

    nodepth = CoreData(depth=np.array([np.nan]), porosity=np.array([0.2]))
    fig, axes = plot_core_on_logs(w, nodepth, curves=["PHIE_CORECAL"], depth_range=(1000, 1019))
    assert "Core Data on Logs" in fig._suptitle.get_text()
