"""Regression tests for the model-layer fixes (curves, wells, zones, tops, units, tables)."""

import warnings

import numpy as np
import pandas as pd
import pytest

import petrophys as pp
from petrophys._internal.benchmark import historical_benchmark
from petrophys.model.class_tables import CLASS_TABLES, Bands
from petrophys.model.core import CoreData
from petrophys.model.curve import Curve
from petrophys.model.formation_tops import FormationTops, label_depths
from petrophys.model.log_data import Zone, align_to_depth
from petrophys.model.units import temperature_to_fahrenheit, temperature_unit
from petrophys.model.well import Well


@pytest.fixture(scope="module")
def well_path():
    return pp.sample("30_6-5")


def _well(path):
    return pp.read(path)


# --- Curve ------------------------------------------------------------------

def test_curve_numpy_protocol_with_dtype_and_copy(well_path):
    from petrophys.interpretation.porosity import density_porosity

    w = _well(well_path)
    gr = w["GR"]
    np.testing.assert_array_equal(np.asarray(gr, dtype=float), gr.data, strict=False)
    copied = np.array(gr, copy=True)
    assert copied is not gr.data and np.array_equal(copied, gr.data, equal_nan=True)
    np.testing.assert_allclose(
        density_porosity(w["RHOB"]), density_porosity(w["RHOB"].data), equal_nan=True
    )


def test_numpy_reductions_on_curve():
    c = Curve("X", [10.0, np.nan, 30.0, 40.0, np.nan])
    assert np.mean(c) == pytest.approx(np.nanmean([10, 30, 40]))
    assert np.std(c) == pytest.approx(np.nanstd([10, 30, 40]))
    assert np.min(c) == 10.0 and np.max(c) == 40.0
    assert c.mean() == pytest.approx(np.nanmean([10, 30, 40]))


def test_curve_arithmetic_units(well_path):
    w = _well(well_path)
    rhob, rt = w["RHOB"], w["RT"] if "RT" in w else w[w.curves[0]]
    assert (rhob + 1.0).unit == rhob.unit
    assert (2.0 * rhob).unit == rhob.unit
    assert (rhob / rhob).unit == ""
    assert (rhob * rhob).unit == ""
    assert (rhob ** 2).unit == ""
    assert (1.0 / rt).unit == ""
    with historical_benchmark():
        assert (1.0 / rt).unit == rt.unit
        assert (rhob ** 2).unit == rhob.unit


def test_curve_interpolate_uses_depth_and_does_not_extrapolate():
    c = Curve("X", [0.0, np.nan, 10.0], depth=[0.0, 1.0, 10.0])
    np.testing.assert_allclose(c.interpolate().data, [0.0, 1.0, 10.0])
    out = Curve("L", [np.nan, 1, np.nan, 3, np.nan, np.nan]).interpolate().data
    np.testing.assert_allclose(out, [np.nan, 1, 2, 3, np.nan, np.nan])
    with historical_benchmark():
        old = Curve("L", [np.nan, 1, np.nan, 3, np.nan, np.nan]).interpolate().data
    np.testing.assert_allclose(old, [np.nan, 1, 2, 3, 3, 3])


def test_curve_setters_check_length():
    c = Curve("GR", [1, 2, 3, 4, 5], depth=[1, 2, 3, 4, 5])
    with pytest.raises(ValueError):
        c.data = [1, 2]
    with pytest.raises(ValueError):
        c.depth = [1, 2]
    c.data = [5, 4, 3, 2, 1]
    assert len(c) == 5


# --- Well -------------------------------------------------------------------

def test_rename_curve_refuses_to_overwrite(well_path):
    w = _well(well_path)
    n, rhob_mean = w.n_curves, np.nanmean(w["RHOB"].data)
    with pytest.raises(ValueError):
        w.rename_curve("GR", "RHOB")
    assert w.n_curves == n and np.nanmean(w["RHOB"].data) == rhob_mean
    assert w.rename_curve("GR", "GR_NEW") and "GR_NEW" in w


def test_well_depth_setter_updates_curves(well_path):
    w = _well(well_path)
    w.depth = w.depth / 0.3048
    w.depth_unit = "ft"
    assert w["GR"].depth[0] == pytest.approx(w.top)
    assert w["GR"].depth_unit == "ft"
    assert len(w["GR"].depth_slice(9300, 9600)) == w.depth_slice(9300, 9600).n_samples
    with pytest.raises(ValueError):
        w.depth = w.depth[:10]


def test_setitem_copies_and_checks_depth(well_path):
    w = _well(well_path)
    w["GRCOPY"] = w["GR"]
    assert w["GR"] is not w["GRCOPY"] and w["GR"].mnemonic == "GR"
    bad = Curve("X", np.zeros(w.n_samples), depth=np.arange(w.n_samples, dtype=float))
    with pytest.raises(ValueError):
        w["X"] = bad
    w["Y"] = Curve("Y", np.ones(w.n_samples))
    np.testing.assert_array_equal(w["Y"].depth, w.depth)


def test_from_dataframe_index_case_and_collisions():
    df = pd.DataFrame({"DEPTH": [1.0, 2, 3], "GR": [1, 2, 3]})
    w = Well.from_dataframe(df.set_index("DEPTH"), depth_unit="m")
    np.testing.assert_array_equal(w.depth, [1.0, 2, 3])
    assert w.curves == ["GR"]
    w = Well.from_dataframe(pd.DataFrame({"Depth": [1.0, 2], "GR": [1, 2]}), depth_unit="m")
    assert w.curves == ["GR"] and w.depth is not None
    with pytest.raises(ValueError):
        Well.from_dataframe(pd.DataFrame({"DEPTH": [1.0, 2], "gr": [1, 1], "GR": [5, 5]}), depth_unit="m")
    w = Well.from_dataframe(pd.DataFrame({"DEPTH": [1.0, 2, 3], "GR": ["1", "bad", "3"]}), depth_unit="m")
    assert w.get_parameter("COERCED_NON_NUMERIC_VALUES") == {"GR": ["bad"]}
    w = Well.from_dataframe(pd.DataFrame({"DEPTH": [1.0, 2], 0: [1.0, 2.0]}), depth_unit="m")
    assert w.curves == ["0"]


def test_to_dataframe_rejects_missing_curve(well_path):
    w = _well(well_path)
    with pytest.raises(KeyError, match="GRR"):
        w.to_dataframe(["GR", "GRR"])
    assert w.to_dataframe(["GR"]).columns.tolist() == ["DEPTH", "GR"]


# --- log_data ---------------------------------------------------------------

def test_align_to_depth_does_not_bridge_null_gaps():
    out = align_to_depth(np.array([0.0, 10, np.nan, np.nan, 40, 50]), np.arange(6.0), np.array([2.5, 0.5, 4.0]))
    assert np.isnan(out[0]) and out[1] == pytest.approx(5.0) and out[2] == pytest.approx(40.0)
    out = align_to_depth(np.r_[1.0, np.full(998, np.nan), 2.0], np.arange(1000.0), np.array([500.0]))
    assert np.isnan(out[0])
    out = align_to_depth(np.array([0.0, 10.0, 20.0]), np.array([0.0, np.nan, 2.0]), np.array([1.0]))
    assert out[0] == pytest.approx(10.0)
    with pytest.raises(ValueError):
        align_to_depth(np.arange(3.0), np.arange(3.0), np.arange(3.0), method="cubic")


def test_zone_rejects_inverted_depths():
    with pytest.raises(ValueError):
        Zone("R", 150, 100)
    assert Zone("R", 100, 150).thickness == 50


# --- formation tops ---------------------------------------------------------

def test_clip_to_well_converts_units(well_path):
    w = _well(well_path)
    assert w.depth_unit == "m"
    tops = FormationTops.from_dict([
        {"name": "A", "top_depth": 1000.0 / 0.3048, "bottom_depth": 2000.0 / 0.3048,
         "metadata": {"depth_unit": "ft"}},
    ])
    clipped = tops.clip_to_well(w)
    assert len(clipped) == 1
    assert clipped[0].top_depth == pytest.approx(1000.0)
    assert clipped[0].metadata["depth_unit"] == "m"


def test_label_depths_warns_on_missing_bottom():
    table = pd.DataFrame({"name": ["A", "B"], "top": [100.0, 200.0], "bottom": [200.0, np.nan]})
    with pytest.warns(UserWarning, match="'B'"):
        assert label_depths([150.0, 250.0], table).tolist() == ["A", ""]


def test_tops_csv_is_lossless_and_numpy_index(tmp_path):
    t = FormationTops.from_dict([{"name": "A", "top_depth": 2841.125, "bottom_depth": 2865.375}])
    path = tmp_path / "t.csv"
    t.to_csv(path)
    back = pd.read_csv(path)
    assert back["top_depth"].iloc[0] == 2841.125 and back["bottom_depth"].iloc[0] == 2865.375
    assert t[np.int64(0)].name == "A"


# --- units, core, class tables ----------------------------------------------

def test_temperature_symbols():
    assert [temperature_unit(x) for x in ["℃", "℉", "oC", "oF", "degC"]] == ["C", "F", "C", "F", "C"]
    assert temperature_to_fahrenheit(100.0, "℃") == pytest.approx(212.0)
    assert temperature_to_fahrenheit(100.0, "oC") == pytest.approx(212.0)
    assert temperature_unit("deg C of formation") == "C"


def test_core_to_dataframe_keeps_per_plug_columns():
    c = CoreData.from_dataframe(pd.DataFrame({"DEPTH": [1.0, 2.0], "CPOR": [0.2, 0.3], "KV": [5.0, 7.0]}))
    df = c.to_dataframe()
    assert list(df.columns) == ["DEPTH", "CPOR", "KV"]
    np.testing.assert_array_equal(df["KV"], [5.0, 7.0])


def test_bands_labels_include_edge_only_label():
    b = Bands("LOW", ((">=", 1.0, "AT"), (">", 1.0, "HIGH")))
    assert b.labels == ("LOW", "AT", "HIGH")
    b = Bands("POOR", ((">", 1.0, "FAIR"), (">", 10.0, "GOOD")))
    assert b.labels == ("POOR", "FAIR", "GOOD")


def test_reservoir_quality_describe_families():
    t = CLASS_TABLES["RESERVOIR_QUALITY"]
    text = t.describe("clastic")
    assert "Clastic rock" in text and "Carbonate rock" not in text
    with pytest.raises(ValueError):
        t.describe(["sandstone"])
    assert "Carbonate rock" in t.describe([])
