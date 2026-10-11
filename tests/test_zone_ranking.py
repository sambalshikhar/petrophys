"""Best and weakest zone of a property, in its better direction, leaving out zones in bad hole."""
import numpy as np
import pytest

from petrophys.evaluation.summaries import best_and_weakest, property_direction


def test_direction():
    assert property_direction("PERM_CC") == "higher" and property_direction("bvw") == "lower"
    assert property_direction("GR") is None


def test_ranking_leaves_out_bad_hole_and_breaks_ties_in_zone_order():
    assert best_and_weakest([0.2, 0.1, 0.3], "higher", bad_hole=[0.0, 0.0, 0.9]) == \
        {"best": 0, "weakest": 1, "basis": "good_hole"}
    assert best_and_weakest([0.5, 0.2, np.nan, 0.5], "lower") == {"best": 1, "weakest": 0, "basis": "good_hole"}


def test_all_in_bad_hole_and_too_few():
    assert best_and_weakest([0.2, 0.1], "higher", bad_hole=[0.8, 0.9]) == {"best": 0, "weakest": 1,
                                                                          "basis": "all_in_bad_hole"}
    assert best_and_weakest([0.2, 0.1], "higher", bad_hole=[0.0, 0.9])["basis"] == "too_few"
    assert best_and_weakest([0.2], "higher")["best"] is None
    with pytest.raises(ValueError, match="direction"):
        best_and_weakest([0.2, 0.3], "better")
