"""Regression tests for lithology and borehole-imaging fixes."""

import warnings

import numpy as np
import pytest

import petrophys
from petrophys._internal.benchmark import historical_benchmark
from petrophys.interpretation.borehole_imaging import BoreholeImageData
from petrophys.interpretation.borehole_imaging import fmi as F
from petrophys.interpretation.borehole_imaging.borehole_image import (
    _collect_flattened_curve_groups,
)
from petrophys.interpretation.lithology import load_lithology_csv, mn_crossplot_data
from petrophys.interpretation.lithology import matrix_inference as MI
from petrophys.interpretation.lithology.matrix_properties import (
    get_matrix_density,
    get_mineral_properties,
)


def test_lithology_csv_with_spaces_after_commas(tmp_path):
    path = tmp_path / "lith.csv"
    path.write_text("top, bottom, lithology\n1000, 1100, limestone\n1100, 1200, sandstone\n")
    intervals = load_lithology_csv(path)
    assert [(i.top, i.bottom, i.lithology) for i in intervals] == [
        (1000.0, 1100.0, "carbonate"),
        (1100.0, 1200.0, "clastic"),
    ]


def test_lithology_csv_warns_on_unparseable_rows_and_rejects_reversed(tmp_path):
    path = tmp_path / "lith.csv"
    path.write_text('top,bottom,lithology\n"1,100",1200,sandstone\n1200,1300,shale\n')
    with pytest.warns(Warning, match="Skipped 1 row"):
        intervals = load_lithology_csv(path)
    assert len(intervals) == 1

    path.write_text("top,bottom,lithology\n1300,1200,sandstone\n")
    with pytest.raises(ValueError, match="below bottom"):
        load_lithology_csv(path)


def test_property_name_honoured_for_matrix_fallback_names():
    assert get_matrix_density("sandstone") == pytest.approx(2.65)
    assert get_mineral_properties("ls", "density") == pytest.approx(2.71)
    assert get_matrix_density("quartz") == pytest.approx(2.65)
    with pytest.raises(KeyError):
        get_mineral_properties("quartz", "not_a_property")


def test_mn_undefined_near_fluid_density_is_nan():
    out = mn_crossplot_data(
        np.array([0.5, 0.5, 0.2]), np.array([1.0, 0.995, 2.4]), np.array([150.0, 150.0, 70.0])
    )
    assert np.all(np.isnan(out["m"][:2])) and np.all(np.isnan(out["n"][:2]))
    assert np.isfinite(out["m"][2]) and np.isfinite(out["n"][2])


def test_navigation_length_must_match_image_depth():
    data = BoreholeImageData.from_arrays([0.0, 1.0], np.ones((2, 1)), channel_names=["FMPA"])
    with pytest.raises(ValueError, match="one value per image depth"):
        F.with_navigation(data, {"P1AZ": [0.0, 90.0, 180.0, 270.0]})
    ok = F.with_navigation(data, {"P1AZ": [0.0, 90.0]})
    assert F.orient_to_azimuth(ok, n_azimuth=4).image.shape == (2, 4)


def test_from_arrays_rejects_mismatched_channel_metadata():
    with pytest.raises(ValueError, match="one sample count per channel"):
        BoreholeImageData.from_arrays(
            [0.0], np.arange(16.0)[None], channel_names=list(F.ARRAY_ORDER)
        )


def test_explicit_prefixes_do_not_fall_back_to_generic_groups():
    names = ["FMPA_0", "FMPA_1", "AT10", "AT20"]
    assert _collect_flattened_curve_groups(names, ["ZZZ"]) == []
    assert _collect_flattened_curve_groups(names, None)[0][0] == "FMPA"


def test_equalize_dynamic_window_converted_for_feet():
    image = np.random.default_rng(0).normal(size=(40, 5))
    depth_m = np.arange(40) * 0.1
    metric = F.equalize_dynamic(image, depth_m, 0.4)
    feet = F.equalize_dynamic(image, depth_m / 0.3048, 0.4, depth_unit="ft")
    np.testing.assert_allclose(metric, feet)


def test_highside_image_not_labelled_with_compass_ticks():
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    depth = np.arange(4) * 0.1
    data = BoreholeImageData.from_arrays(depth, np.ones((4, 24)), channel_names=["FMPA"])
    data = F.with_navigation(data, {"RB": np.zeros(4), "P1AZ": np.zeros(4)})
    fig, axes = plt.subplots(1, 2)
    try:
        F.plot_fmi(F.orient_to_azimuth(data, reference="highside"), ax=axes[0], equalize="none")
        F.plot_fmi(F.orient_to_azimuth(data, reference="north"), ax=axes[1], equalize="none")
        assert axes[0].get_xlabel() == "Angle from high side [deg]"
        assert "N" not in [t.get_text() for t in axes[0].get_xticklabels()]
        assert axes[1].get_xlabel() == "Azimuth [deg]"
        assert [t.get_text() for t in axes[1].get_xticklabels()][0] == "N"
    finally:
        plt.close(fig)


def _well_with_solver_curves():
    well = petrophys.read(petrophys.sample("30_6-5"))
    n = well.get("RHOB").data.size
    for name in ("VQTZ", "VCAL", "VDOL"):
        if well.get(name) is not None:
            well.remove_curve(name)
    well.add_curve("VQTZ", np.full(n, 0.1), unit="v/v")
    well.add_curve("VCAL", np.full(n, 0.6), unit="v/v")
    well.add_curve("VDOL", np.full(n, 0.0), unit="v/v")
    return well


def test_resolve_from_minerals_reads_solver_calcite_curve():
    well = _well_with_solver_curves()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = MI.resolve_from_minerals(well, warn=False)
        assert res is not None and res.matrix == "limestone"
        with historical_benchmark():
            assert MI.resolve_from_minerals(well, warn=False) is None
