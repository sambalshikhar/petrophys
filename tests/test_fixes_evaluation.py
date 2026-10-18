"""Regression tests for the evaluation-group fixes."""

import warnings

import numpy as np
import pytest

import petrophys as pp
from petrophys import Well
from petrophys._internal.benchmark import historical_benchmark


def _well(depth, unit="m", **curves):
    well = Well("FIX-EVAL")
    well.depth = np.asarray(depth, dtype=float)
    well.depth_unit = unit
    for name, data in curves.items():
        well.add_curve(name, np.asarray(data, dtype=float), "", name)
    return well


def test_calculate_mhi_nonpositive_sxo_is_nan():
    from petrophys.evaluation.mhi import calculate_mhi

    out = calculate_mhi(np.array([0.0, -0.1, 0.5]), np.array([0.3, 0.3, 0.3]))
    assert np.isnan(out[:2]).all() and out[2] == pytest.approx(0.6)
    assert np.isnan(calculate_mhi(0.0, 0.3))
    with historical_benchmark():
        assert calculate_mhi(0.0, 0.3) == pytest.approx(30.0)


def test_saturation_analysis_no_mhi_from_fifth_root_sxo():
    from petrophys.evaluation.mhi import MhiModule

    well = pp.interpret(pp.read(pp.sample("HW-3"))).well
    assert well.get("MHI") is None
    sw = np.asarray(well.get("SW").data)
    assert np.allclose(np.asarray(well.get("SXO").data), np.clip(sw, 0, 1) ** 0.2, equal_nan=True)
    assert not MhiModule.diagnostic_sxo_mask(well).any()
    result = pp.SaturationAnalysis().analyze(well)
    assert well.get("MHI") is None
    labels = set(result.sections["hydrocarbon_movability_invasion_characteristics"].raw_labels)
    assert "HIGH_MOVEABLE_HC" not in labels


def test_lithology_pef_28_dolomite_not_mismatch():
    from petrophys.evaluation.diagnostics import LithologyAnalysis

    well = _well([1.0, 2.0], PEF=[2.8, 3.0], RHOB=[2.85, 2.85], LITHOLOGY_GEOLINK=[15, 15])
    result = LithologyAnalysis().analyze(well, add_curves=False)
    assert list(result.sections["carbonate_analysis"].raw_labels) == ["DOLOMITE_CONFIRMED"] * 2
    with historical_benchmark():
        result = LithologyAnalysis().analyze(well, add_curves=False)
    assert result.sections["carbonate_analysis"].raw_labels[0] == "CARBONATE_LOG_MISMATCH"


def test_compare_saturation_models_infinite_rt_is_nan():
    from petrophys.evaluation.sensitivity import compare_saturation_models

    r = compare_saturation_models(
        rt=np.array([np.nan, 10, np.inf, 0, 5.0]), phie=np.array([0.22, 0.2, 0.18, 0.15, 0.24]),
        vcl=np.array([0.1, 0.2, 0.3, 0.35, 0.05]), depth=np.arange(5.0), rw=0.07, rsh=3,
    )
    for model, sw in r.sw.items():
        assert np.isnan(sw[2]), model
        assert np.isfinite(sw[1]), model


def test_stacking_pattern_flat_is_no_change():
    from petrophys.evaluation.intervals.sequence import stacking_pattern

    p = stacking_pattern([50.0, 50.0, 50.0])
    assert p["steps"] == ["", "no change", "no change"] and p["packages"] == []


def test_grade_interval_depth_unit():
    from petrophys.evaluation.net_pay import grade_interval

    assert grade_interval("SANDSTONE", 0.1, 2.0, min_thickness=3.0, depth_unit="ft") == (
        "low", ["thinner than 3.0 ft"])
    assert grade_interval("SANDSTONE", 0.1, 2.0, min_thickness=3.0)[1] == ["thinner than 3.0 m"]


def test_flow_roles_float32_nan_unknown():
    from petrophys.evaluation.flow_units.roles import flow_roles

    out = flow_roles(np.array([np.nan, 30.0], dtype=np.float32), [1, 1])
    assert out["zones"][0]["role"] is None and out["strongest"] == 1


def test_best_and_weakest_unknown_bad_hole_kept():
    from petrophys.evaluation.summaries import best_and_weakest

    out = best_and_weakest([0.4, 0.1, 0.3], "higher", bad_hole=[np.nan, 0, 0])
    assert out == {"best": 0, "weakest": 1, "basis": "good_hole"}
    assert best_and_weakest([0.4, 0.1, 0.3], "higher", bad_hole=[None, 0, 0.9])["best"] == 0


def test_weighted_averages_missing_weight_curve_warns():
    from petrophys.evaluation.summaries.averages import calculate_weighted_averages

    well = _well(np.arange(4.0), SW=[0.2, 0.4, 0.6, 0.8], PHIE=[0.1, 0.2, 0.3, 0.4])
    with pytest.warns(UserWarning, match="PHIEE"):
        calculate_weighted_averages(well, ["SW"], weight_curve="PHIEE", net_pay_curve=None)


def test_net_gross_missing_flags_nan():
    from petrophys.evaluation.summaries.net_gross import calculate_net_gross
    from petrophys.evaluation.summaries.zone_summary import calculate_zone_summary

    well = _well(np.arange(100.0, 110.0), NET_PAY=np.ones(10))
    with pytest.warns(UserWarning, match="NET_RES"):
        r = calculate_net_gross(well)
    assert r.net_pay_thickness == pytest.approx(10.0)
    assert np.isnan(r.net_rock_thickness) and np.isnan(r.ntg_rock) and np.isnan(r.ntg_pay_res)
    zs = calculate_zone_summary(well, 100.0, 109.0)
    assert np.isnan(zs.net_rock) and zs.net_pay == pytest.approx(10.0)
    with historical_benchmark():
        assert calculate_zone_summary(well, 100.0, 109.0).net_rock == 0.0


def test_adjacent_zones_do_not_double_count_boundary():
    from petrophys.evaluation.intervals import IntervalAnalyzer
    from petrophys.evaluation.summaries.net_gross import calculate_ntg_by_zone
    from petrophys.evaluation.summaries.zone_summary import summarize_zones
    from petrophys.model.intervals import Formation

    flags = np.ones(10)
    well = _well(np.arange(100.0, 110.0), NET_PAY=flags, NET_RES=flags, NET_ROCK=flags)
    zones = [{"name": "A", "top": 100.0, "bottom": 104.0}, {"name": "B", "top": 104.0, "bottom": 109.0}]
    assert [z.gross_thickness for z in summarize_zones(well, zones)] == pytest.approx([4.0, 6.0])
    assert [r.gross_thickness for r in calculate_ntg_by_zone(well, zones).values()] == pytest.approx([4.0, 6.0])
    summaries = IntervalAnalyzer(well).analyze_all([Formation("A", 100.0, 104.0), Formation("B", 104.0, 109.0)])
    assert [s.zone_summary.gross_thickness for s in summaries] == pytest.approx([4.0, 6.0])


def test_hcpv_from_well_reversed_window_and_coverage():
    from petrophys.evaluation.volumetrics.hydrocarbon_pore_volume import calculate_hcpv_from_well

    phie = np.array([0.2, np.nan, 0.2, 0.2])
    well = _well(np.arange(4.0), NET_PAY=np.ones(4), PHIE=phie, SW=np.full(4, 0.3))
    with pytest.raises(ValueError, match="below"):
        calculate_hcpv_from_well(well, 1.0, top_depth=3.0, bottom_depth=1.0)
    with pytest.warns(UserWarning, match="1 of 4"):
        calculate_hcpv_from_well(well, 1.0)
