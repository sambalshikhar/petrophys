"""Well.select: a well restricted to any set of samples (a lithology group is not one depth range)."""
import numpy as np
import pytest

from petrophys.model.well import Well


def _well():
    w = Well(name="W")
    w.depth = np.arange(1000.0, 1010.0, 1.0)
    w.add_curve("VCL", np.linspace(0.0, 0.9, 10), "v/v", "clay")
    w.add_curve("GR", np.arange(10.0) * 10, "gAPI", "gamma")
    w.set_parameter("RW", 0.05)
    return w


def test_select_mask_index_and_slice():
    w = _well()
    sand = w.select(w.get("VCL").data < 0.3)
    assert list(sand.depth) == [1000.0, 1001.0, 1002.0] and list(sand.get("GR").data) == [0.0, 10.0, 20.0]
    assert sand.get_parameter("RW") == 0.05 and sand.depth_unit == w.depth_unit
    picked = w.select(np.array([1, 4, 8]))
    assert list(picked.depth) == [1001.0, 1004.0, 1008.0]
    assert list(w.select(slice(2, 4)).get("VCL").data) == pytest.approx([0.2, 0.3])


def test_select_copies_and_checks_the_mask():
    w = _well()
    sub = w.select(np.ones(10, dtype=bool))
    sub.set_parameter("RW", 1.0)
    assert w.get_parameter("RW") == 0.05                       # parameters copied, not shared
    with pytest.raises(ValueError, match="mask has 3 samples"):
        w.select(np.ones(3, dtype=bool))


def test_depth_slice_unchanged():
    w = _well()
    s = w.depth_slice(1002.0, 1005.0)
    assert list(s.depth) == [1002.0, 1003.0, 1004.0, 1005.0] and list(s.get("GR").data) == [20.0, 30.0, 40.0, 50.0]
