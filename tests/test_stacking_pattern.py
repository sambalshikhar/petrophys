"""Vertical stacking patterns of zone values, base to top."""
import numpy as np

from petrophys.evaluation.intervals import MIN_STEP, stacking_pattern


def test_packages_and_abrupt_steps():
    p = stacking_pattern([90.0, 70.0, 50.0, 52.0, 85.0], "GR")
    assert p["threshold"] == 4.0                                   # 10% of the 40 API range, above the 3 API floor
    assert p["steps"] == ["", "cleaner", "cleaner", "no change", "dirtier"]
    assert [(q["start"], q["end"], q["kind"]) for q in p["packages"]] == [(0, 2, "cleaning_up"), (3, 4, "abrupt_increase")]


def test_floor_threshold_and_missing_values():
    p = stacking_pattern([0.30, 0.31, np.nan, 0.50], "VCL")
    assert p["threshold"] == MIN_STEP["VCL"] == 0.03                    # 10% of the 0.2 range is below the floor
    assert p["steps"][1] == "no change" and p["steps"][2] == "" and p["steps"][3] == ""   # no step across a gap
    q = stacking_pattern([60.0, 61.0], "GR")
    assert q["threshold"] == 3.0 and q["packages"] == []
    assert stacking_pattern([1.0, 2.0, 3.0], min_change=0.5)["packages"][0]["kind"] == "dirtying_up"
