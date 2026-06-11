import numpy as np
import pytest

from petrophys.base import Well
from petrophys.base.log_data import DepthIndex
from petrophys.summation import calculate_net_gross, calculate_zone_summary
from petrophys.summation.averages import calculate_statistics
from petrophys.summation.formation_analysis import analyze_formations
from petrophys.formation.intervals import Formation
from petrophys.workflow.quick_look import QuickLook


def test_depth_helpers_handle_descending_and_nonfinite_depths():
    well = Well("DESCENDING")
    well.depth = np.array([100.0, np.nan, 99.0, 98.0])

    assert well.top == 98.0
    assert well.bottom == 100.0
    assert well.step == 1.0

    index = DepthIndex(np.array([np.nan, 100.0, 98.0]))
    nearest_index, nearest_depth = index.find_nearest(99.0)

    assert nearest_index in {1, 2}
    assert nearest_depth in {98.0, 100.0}

    with pytest.raises(ValueError, match="empty/non-finite DepthIndex"):
        DepthIndex(np.array([np.nan, np.nan])).resample(0.5)


def test_summation_uses_positive_sample_thickness_for_descending_depth():
    well = Well("DESCENDING-SUMMARY")
    well.depth = np.array([100.0, 99.0, 98.0])
    well.add_curve("NET_ROCK", np.array([1.0, 1.0, 1.0]))
    well.add_curve("NET_RES", np.array([1.0, 0.0, 1.0]))
    well.add_curve("NET_PAY", np.array([1.0, 0.0, 1.0]))

    ntg = calculate_net_gross(well, top_depth=98.0, bottom_depth=100.0)
    summary = calculate_zone_summary(well, top_depth=98.0, bottom_depth=100.0)

    assert ntg.gross_thickness == 3.0
    assert ntg.net_pay_thickness == 2.0
    assert ntg.ntg_pay == pytest.approx(2.0 / 3.0)
    assert summary.gross_thickness == 3.0
    assert summary.net_pay == 2.0
    assert summary.ntg == pytest.approx(2.0 / 3.0)


def test_default_depth_range_errors_cleanly_for_all_nan_depth():
    well = Well("NO-FINITE-DEPTH")
    well.depth = np.array([np.nan, np.nan])
    well.add_curve("PHIE", np.array([0.10, 0.12]))

    with pytest.raises(ValueError, match="finite depth range"):
        calculate_statistics(well, "PHIE")


def test_formation_analysis_returns_empty_for_missing_well_depth_range():
    well = Well("NO-FORMATION-RANGE")
    well.depth = np.array([np.nan, np.nan])

    result = analyze_formations(well, [Formation("A", 100.0, 200.0)])

    assert result.formations == []
    assert result.summaries == []


def test_quick_look_closes_potential_pay_zone_at_last_sample():
    well = Well("PAY-TO-TD")
    well.depth = np.array([100.0, 101.0, 102.0, 103.0])
    well.add_curve("GR", np.array([35.0, 40.0, 42.0, 45.0]), "API", "Gamma Ray")
    well.add_curve("RT", np.array([20.0, 25.0, 30.0, 35.0]), "ohm.m", "Resistivity")

    result = QuickLook().analyze(well)

    assert len(result.potential_pay_zones) == 1
    assert result.potential_pay_zones[0]["top"] == 100.0
    assert result.potential_pay_zones[0]["bottom"] == 103.0
