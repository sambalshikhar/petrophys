import numpy as np
import pytest
import warnings

from petrophys.model import Well
from petrophys.model.curve import Curve
from petrophys.model.log_data import DepthIndex
from petrophys.interpretation.core_analysis.calibration import CalibrationResult
from petrophys.interpretation.core_analysis.color_analysis import CoreImageData
from petrophys.model.core import CoreData
from petrophys.interpretation.core_analysis.grain_size import GrainSizeData
from petrophys.interpretation.core_analysis.mechanical import CoreMechanical
from petrophys.visualization.core import plot_calibration
from petrophys.interpretation.geochemistry.xrf import XRFData
from petrophys.evaluation.intervals import Formation, FormationTops
from petrophys.interpretation.nmr.nmr_data import NMRData


def test_curve_statistics_return_nan_for_empty_curve():
    curve = Curve("EMPTY", [])

    assert curve.valid_count() == 0
    assert curve.null_count() == 0
    assert np.isnan(curve.min())
    assert np.isnan(curve.max())
    assert np.isnan(curve.mean())
    assert np.isnan(curve.std())
    assert np.isnan(curve.median())
    assert np.isnan(curve.percentile(50))
    assert np.isnan(curve.describe()["min"])


def test_curve_statistics_return_nan_for_all_nan_curve_without_warnings():
    curve = Curve("ALLNAN", [np.nan, np.nan])

    assert curve.valid_count() == 0
    assert curve.null_count() == 2
    assert np.isnan(curve.min())
    assert np.isnan(curve.max())
    assert np.isnan(curve.mean())
    assert np.isnan(curve.std())
    assert np.isnan(curve.median())
    assert np.isnan(curve.percentile(50))
    assert np.isnan(curve.describe()["p75"])


def test_well_top_bottom_return_none_for_empty_or_all_nan_depth():
    empty = Well("EMPTY-DEPTH")
    empty.depth = []
    assert empty.top is None
    assert empty.bottom is None

    all_nan = Well("NAN-DEPTH")
    all_nan.depth = [np.nan, np.nan]
    assert all_nan.top is None
    assert all_nan.bottom is None


def test_depth_index_and_specialist_data_bounds_handle_empty_or_all_nan_depth():
    cases = [
        DepthIndex(np.array([])),
        DepthIndex(np.array([np.nan, np.nan])),
        CoreData(depth=np.array([])),
        CoreData(depth=np.array([np.nan, np.nan])),
        NMRData(depth=np.array([]), mphi=np.array([])),
        NMRData(depth=np.array([np.nan, np.nan]), mphi=np.array([np.nan, np.nan])),
        XRFData(depth=np.array([]), elements={}),
        XRFData(depth=np.array([np.nan, np.nan]), elements={}),
        GrainSizeData(depth=np.array([]), d50=np.array([])),
        GrainSizeData(depth=np.array([np.nan, np.nan]), d50=np.array([np.nan, np.nan])),
        CoreMechanical(depth=np.array([]), ucs=np.array([])),
        CoreMechanical(depth=np.array([np.nan, np.nan]), ucs=np.array([np.nan, np.nan])),
        CoreImageData(depth=np.array([])),
        CoreImageData(depth=np.array([np.nan, np.nan])),
    ]

    for item in cases:
        assert item.top is None
        assert item.bottom is None
        repr(item)


def test_formation_tops_clip_to_well_returns_empty_for_missing_depth_range():
    tops = FormationTops([Formation("A", 100.0, 200.0)])

    empty = Well("EMPTY-DEPTH")
    empty.depth = []
    assert len(tops.clip_to_well(empty)) == 0

    all_nan = Well("NAN-DEPTH")
    all_nan.depth = [np.nan, np.nan]
    assert len(tops.clip_to_well(all_nan)) == 0


def test_specialist_indices_preserve_nan_for_all_nan_inputs_without_warnings():
    image = CoreImageData(
        depth=np.array([1.0, 2.0]),
        grey_level=np.array([np.nan, np.nan]),
        grey_std=np.array([np.nan, np.nan]),
        r=np.array([np.nan, np.nan]),
        g=np.array([np.nan, np.nan]),
        b=np.array([np.nan, np.nan]),
    )
    xrf = XRFData(
        depth=np.array([1.0, 2.0]),
        elements={"Si": np.array([np.nan, np.nan]), "Al": np.array([np.nan, np.nan])},
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        brightness = image.brightness_index()
        color = image.color_index()
        texture = image.texture_index()
        proxies = xrf.mineral_proxies()

    assert not any(issubclass(item.category, RuntimeWarning) for item in caught)
    assert np.isnan(brightness).all()
    assert np.isnan(color).all()
    assert np.isnan(texture).all()
    assert np.isnan(proxies["clay_index"]).all()


def test_core_mechanical_smoothing_handles_nonfinite_depth_without_warnings():
    mechanical = CoreMechanical(
        depth=np.array([np.nan, np.nan]),
        ucs=np.array([np.nan, np.nan]),
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        smoothed = mechanical.scale_average(window_m=1.0)

    assert smoothed is mechanical
    assert not any(issubclass(item.category, RuntimeWarning) for item in caught)


def test_empty_calibration_plot_still_builds_axes():
    pytest.importorskip("matplotlib")
    result = CalibrationResult(
        property_name="Empty",
        n_points=0,
        r_squared=np.nan,
        bias=np.nan,
        rmse=np.nan,
        slope=np.nan,
        intercept=np.nan,
        core_values=np.array([]),
        log_values=np.array([]),
    )

    fig, ax = plot_calibration(result)

    assert fig is ax.figure
