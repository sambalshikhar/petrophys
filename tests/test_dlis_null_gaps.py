"""Merged DLIS frames keep absent-value runs as NaN instead of interpolating across them."""
import numpy as np

from petrophys.io.dlis_reader import _resample_to_depth


def test_null_run_is_not_interpolated():
    depth = np.arange(1000.0, 1010.5, 0.5)
    gr = np.linspace(50.0, 120.0, depth.size)
    gr[3:18] = np.nan                                  # 1001.5-1008.5 m absent
    out = _resample_to_depth(depth, gr, depth)
    assert np.isnan(out[3:18]).all()
    assert np.allclose(out[:3], gr[:3]) and np.allclose(out[18:], gr[18:])


def test_resampling_between_neighbours_still_interpolates():
    depth = np.arange(1000.0, 1005.0, 0.5)
    values = depth * 2.0
    target = np.arange(1000.0, 1004.6, 0.25)
    assert np.allclose(_resample_to_depth(depth, values, target), target * 2.0)
