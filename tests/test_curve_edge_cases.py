import numpy as np

from petrophys.base import Well
from petrophys.base.curve import Curve


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
