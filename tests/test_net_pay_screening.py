"""Screening confidence of flagged intervals and the order in which the net-pay cutoffs remove rock."""
import numpy as np
import pytest

from petrophys.evaluation.net_pay import cutoff_attrition, grade_interval, lithology_in, majority_label

ROCK = dict(non_reservoir_lithologies=["SHALE", "COAL", "ANHYDRITE"], marginal_lithologies=["SILT", "SILTSTONE"])


def test_head_noun_names_the_rock():
    assert lithology_in("CALCAREOUS SHALE", ["SHALE"]) and not lithology_in("SHALY SANDSTONE", ["SHALE"])
    assert lithology_in("silt", ["SILT"]) and not lithology_in("", ["SILT"])


def test_majority_label_ignores_missing_and_breaks_ties():
    assert majority_label(["SAND", "SILT", "SILT", "NO_LITHOLOGY_DATA"]) == "SILT"
    assert majority_label(["SILT", "SAND"]) == "SAND"                      # tie: alphabetical
    assert majority_label(["NO_LITHOLOGY_DATA", ""]) == "UNKNOWN"


@pytest.mark.parametrize("lith,bh,thick,want", [
    ("SANDSTONE", 0.1, 3.0, ("higher", [])),
    ("CALCAREOUS SHALE", None, 3.0, ("low", ["calcareous shale is not reservoir rock"])),
    ("SILT", 0.9, 0.5, ("low", ["silt is marginal reservoir rock", "bad hole 90%", "thinner than 1.0 m"])),
    ("SAND", 0.5, 1.0, ("higher", [])),                                  # at the limits: not beyond them
])
def test_grade_interval(lith, bh, thick, want):
    assert grade_interval(lith, bh, thick, **ROCK) == want


def test_cutoff_attrition_in_order():
    steps = cutoff_attrition([1, 1, 1, 1, 0, 0], [1, 1, 1, 0, 0, 0], [1, 0, 0, 0, 0, 0])
    assert [s["cutoff"] for s in steps] == ["shale_volume", "porosity", "saturation"]
    assert [round(s["fails"], 4) for s in steps] == [round(2 / 6, 4), 0.25, round(2 / 3, 4)]
    assert [s["tested"] for s in steps] == [6, 4, 3]
    assert [s["cutoff"] for s in cutoff_attrition([0, 0], [0, 0], [0, 0])] == ["shale_volume"]   # no rock left
    assert cutoff_attrition([], [], []) == []
    assert cutoff_attrition(np.array([1.0, np.nan]), [1, 0], [0, 0])[0]["fails"] == 0.5         # NaN is not set
