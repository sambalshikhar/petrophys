"""Regression tests for core-analysis fixes (gap bridging, loaders, edge cases)."""

import math
import warnings

import lasio
import numpy as np
import pytest

from petrophys import Well
from petrophys._internal.benchmark import historical_benchmark
from petrophys.interpretation.core_analysis import (
    CoreImageData,
    DigitalCoreFacies,
    correct_porosity_with_grain_density,
    interpolate_log_at_core,
    wentworth_class,
)
from petrophys.interpretation.core_analysis.depth_matching import _scan
from petrophys.interpretation.core_analysis.mechanical import CoreMechanical
from petrophys.model.core import CoreData


def _gap_well():
    well = Well("GAP")
    well.depth = np.arange(1000.0, 1100.0, 0.5)
    phi = 0.10 + 0.002 * (well.depth - 1000.0)
    phi[(well.depth > 1030) & (well.depth < 1070)] = np.nan
    well.add_curve("PHIE", phi, "v/v")
    return well


def test_interpolate_log_at_core_does_not_bridge_nan_gap():
    core = CoreData(depth=np.array([1050.0, 1010.25, 1020.0]), porosity=np.full(3, 0.2))
    out = interpolate_log_at_core(_gap_well(), core, "PHIE")
    assert np.isnan(out[0])
    np.testing.assert_allclose(out[1:], [0.1205, 0.14])


def test_interpolate_log_at_core_benchmark_keeps_old_bridging():
    core = CoreData(depth=np.array([1050.0]), porosity=np.array([0.2]))
    with historical_benchmark():
        out = interpolate_log_at_core(_gap_well(), core, "PHIE")
    np.testing.assert_allclose(out, [0.2])


def test_depth_shift_scan_ignores_plugs_in_log_gap():
    depth = np.arange(0.0, 101.0)
    values = depth.copy()
    values[40:61] = np.nan
    _, counts = _scan(depth, values, np.array([10.0, 20.0, 50.0]),
                      np.array([1.0, 2.0, 3.0]), np.array([0.0]), 1, False)
    assert counts[0] == 2


def test_grain_density_correction_finite_beyond_gden_plugs():
    well = Well("G")
    well.depth = np.arange(1005.0, 1046.0, 10.0)
    well.add_curve("RHOB", np.full(well.depth.size, 2.30), "g/cm3")
    core = CoreData(depth=np.array([1010.0, 1020, 1030, 1040]),
                    grain_density=np.array([np.nan, 2.70, 2.70, np.nan]))
    phid = correct_porosity_with_grain_density(well, core)
    np.testing.assert_allclose(phid, [0.35 / 1.65, 0.35 / 1.65, 0.4 / 1.7, 0.35 / 1.65, 0.35 / 1.65])


def _write_image_las(path, sat):
    las = lasio.LASFile()
    las.append_curve("DEPT", 3815.0 + 0.01 * np.arange(3), unit="M")
    las.append_curve("E_MBWP", np.full(3, 150.0))
    las.append_curve("E_MHPX", np.full(3, 200.0), unit="DEG")
    las.append_curve("E_MSPX", np.asarray(sat, dtype=float))
    las.append_curve("E_MVPX", np.full(3, 180.0))
    las.write(str(path), version=2.0)


def test_image_saturation_value_scaled_by_255(tmp_path):
    for tag, sat in (("a", [30, 255, 100]), ("b", [30, 60, 40])):
        path = tmp_path / f"img_{tag}.las"
        _write_image_las(path, sat)
        image = CoreImageData.from_las(path)
        assert image.saturation[0] == pytest.approx(30 / 255)
        assert image.value[0] == pytest.approx(180 / 255)


def test_rgb_to_hsv_missing_samples_are_nan():
    image = CoreImageData(depth=np.arange(2.0), r=np.array([np.nan, 200.0]),
                          g=np.array([np.nan, 100.0]), b=np.array([np.nan, 50.0]))
    hsv = image.rgb_to_hsv()
    assert np.isnan(hsv["hue"][0]) and np.isnan(hsv["saturation"][0])
    assert hsv["hue"][1] == pytest.approx(20.0)
    assert hsv["saturation"][1] == pytest.approx(0.75)


def test_facies_from_las_null_rows_are_zero(tmp_path):
    path = tmp_path / "facies.las"
    path.write_text(
        "~Version\nVERS.   2.0 :\nWRAP.   NO  :\n~Well\nSTRT.M  1000.0 :\n"
        "STOP.M  1000.5 :\nSTEP.M  0.1    :\nNULL.   -999.25 :\nWELL.   DEMO   :\n"
        "~Curve\nDEPT.M     : depth\nFACIES.    : facies code\n~A\n"
        "1000.0 -999.25\n1000.1 1\n1000.2 2\n1000.3 3\n1000.4 -999.25\n1000.5 2\n"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = DigitalCoreFacies.from_las(path)
    assert result.facies.tolist() == [0, 1, 2, 3, 0, 2]
    assert result.n_facies == 3


def test_facies_predict_all_nan_returns_unknown():
    classifier = DigitalCoreFacies(n_facies=2, n_init=2)
    classifier.fit(np.arange(10.0), {"a": np.arange(10.0)})
    assert classifier.predict({"a": np.array([np.nan, np.nan])}).tolist() == [0, 0]


def test_wentworth_class_nan_and_negative_unknown():
    assert wentworth_class(math.nan) == "Unknown"
    assert wentworth_class(-5.0) == "Unknown"
    assert wentworth_class(0.0) == "Clay"
    assert wentworth_class(300.0) == "Medium Sand"


def _gapped_mechanical():
    depth = np.r_[np.arange(0, 1, 0.01), np.arange(6, 7, 0.01)]
    return CoreMechanical(depth=depth, ucs=np.r_[np.full(100, 10.0), np.full(100, 50.0)])


def test_scale_average_uses_depth_window_across_gap():
    smoothed = _gapped_mechanical().scale_average(0.30)
    np.testing.assert_allclose(smoothed.ucs[[99, 100]], [10.0, 50.0])
    assert smoothed.n_samples == 200


def test_scale_average_regular_sampling_matches_index_window():
    depth = np.arange(0, 2, 0.01)
    ucs = np.sin(depth * 7.0) * 10 + 30
    smoothed = CoreMechanical(depth=depth, ucs=ucs).scale_average(0.30)
    expected = [ucs[max(0, i - 15):i + 16].mean() for i in range(ucs.size)]
    np.testing.assert_allclose(smoothed.ucs, expected)


def test_scale_average_step_m_resamples():
    depth = np.arange(0, 2, 0.01)
    smoothed = CoreMechanical(depth=depth, ucs=np.full(depth.size, 5.0)).scale_average(0.30, step_m=0.30)
    np.testing.assert_allclose(smoothed.depth, 0.30 * np.arange(7))
    np.testing.assert_allclose(smoothed.ucs, 5.0)
