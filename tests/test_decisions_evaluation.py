"""Tests for the evaluation decisions in release_checks/METHOD_DECISIONS.md.

Each decision changes a result outside the benchmark reproduction only; the
old behaviour is asserted inside ``historical_benchmark()``.
"""

import numpy as np
import pytest

from petrophys import Well
from petrophys._internal.benchmark import historical_benchmark

G = 9.81


def _well(depth, unit="m", **curves):
    well = Well("DECISION-EVAL")
    well.depth = np.asarray(depth, dtype=float)
    well.depth_unit = unit
    for name, data in curves.items():
        well.add_curve(name, np.asarray(data, dtype=float), "", name)
    return well


def _numerical(well, **kwargs):
    from petrophys.evaluation.diagnostics import NumericalStatisticalAnalysis

    return NumericalStatisticalAnalysis().analyze(well, add_curves=False, **kwargs).derived_curves


# ---------------------------------------------------------------------------
# S_V from the surface, PP without an implied seawater column, absolute SH_MIN
# ---------------------------------------------------------------------------
def test_sv_integrates_from_surface_over_a_constant_density_column():
    depth = np.arange(2000.0, 2010.0, 0.5)
    well = _well(depth, RHOB=np.full(20, 2.4), DTC=np.full(20, 80.0), DTS=np.full(20, 140.0))
    curves = _numerical(well)

    # 2.3 g/cc from the surface to the log top at 2000 m, then 2.4 g/cc every 0.5 m.
    expected_sv = (2300.0 * G * 2000.0 + 2400.0 * G * 0.5 * np.arange(20)) / 1e6
    np.testing.assert_allclose(curves["S_V"].data, expected_sv)
    # Fresh-water gradient from the datum, no seawater term.
    np.testing.assert_allclose(curves["PP_HYDROSTATIC"].data, 1000.0 * G * depth / 1e6)
    sv, pp, sh_min = (curves[name].data for name in ("S_V", "PP_HYDROSTATIC", "SH_MIN"))
    assert np.all(sv > pp)
    assert np.all((sh_min > pp) & (sh_min < sv))
    assert "increment" not in curves["S_V"].description

    with historical_benchmark():
        old = _numerical(well)
    np.testing.assert_allclose(old["S_V"].data, 2400.0 * G * 0.5 * np.arange(1, 21) / 1e6)
    assert old["PP_HYDROSTATIC"].data[0] == pytest.approx((1000.0 * 2000.0 + 1025.0 * 100.0) * G / 1e6)
    assert old["SH_MIN"].data[0] < old["PP_HYDROSTATIC"].data[0]
    assert "increment" in old["S_V"].description


def test_sv_and_pp_share_the_water_depth():
    depth = np.array([2000.0, 2000.5])
    well = _well(depth, RHOB=np.full(2, 2.4))
    curves = _numerical(well, water_depth_m=100.0, rho_overburden=2.1)

    # 100 m of 1025 kg/m3 seawater, 1900 m of 2.1 g/cc sediment, then the log.
    assert curves["S_V"].data[0] == pytest.approx((1025.0 * 100.0 + 2100.0 * 1900.0) * G / 1e6)
    assert curves["S_V"].data[1] == pytest.approx(curves["S_V"].data[0] + 2400.0 * G * 0.5 / 1e6)
    # Water gradient with the seawater excess over fresh water in the top 100 m.
    assert curves["PP_HYDROSTATIC"].data[0] == pytest.approx((1000.0 * 2000.0 + 25.0 * 100.0) * G / 1e6)


def test_sv_interpolates_internal_rhob_gaps():
    depth = np.array([1000.0, 1001.0, 1002.0, 1003.0, 1004.0])
    well = _well(depth, RHOB=[2.0, np.nan, np.nan, 3.0, 3.0])
    sv = _numerical(well)["S_V"].data

    top = 2300.0 * G * 1000.0 / 1e6
    assert sv[0] == pytest.approx(top)
    # Linear ramp 2.0 -> 3.0 g/cc over 3 m integrates to 2.5 * 3 = 7.5 g/cc.m.
    assert sv[3] == pytest.approx(top + 7.5 * 1000.0 * G / 1e6)
    assert sv[1] == pytest.approx(top + (2.0 + 7.0 / 3.0) / 2.0 * 1000.0 * G / 1e6)
    assert sv[4] == pytest.approx(sv[3] + 3000.0 * G / 1e6)
    assert np.all(np.isfinite(sv))

    with historical_benchmark():
        old = _numerical(well)["S_V"].data
    assert np.isnan(old[1]) and np.isnan(old[2])


def test_sv_is_nan_outside_the_density_log():
    well = _well(np.arange(1000.0, 1005.0), RHOB=[np.nan, 2.4, 2.4, 2.4, np.nan])
    sv = _numerical(well)["S_V"].data
    assert np.isnan(sv[0]) and np.isnan(sv[4])
    assert sv[1] == pytest.approx(2300.0 * G * 1001.0 / 1e6)


# ---------------------------------------------------------------------------
# Trend-zone gradient threshold per metre
# ---------------------------------------------------------------------------
def test_trend_gradient_threshold_is_per_metre():
    from petrophys.evaluation.diagnostics import NumericalStatisticalAnalysis

    depth = np.arange(8000.0, 8020.0, 0.5)
    # 0.05 API/ft is 0.164 API/m: above the 0.1 per metre threshold.
    well = _well(depth, "ft", GR=0.05 * depth)
    section = "log_gradient_trend_analysis"
    result = NumericalStatisticalAnalysis().analyze(well, add_curves=False)
    assert set(result.sections[section].raw_labels) == {"MODERATE_TREND_ZONE"}
    assert "per metre" in result.sections[section].calculation_thresholds

    with historical_benchmark():
        old = NumericalStatisticalAnalysis().analyze(well, add_curves=False)
    assert set(old.sections[section].raw_labels) == {"STABLE_ZONE"}

    metres = _well(depth * 0.3048, "m", GR=0.05 * depth)
    assert set(
        NumericalStatisticalAnalysis().analyze(metres, add_curves=False).sections[section].raw_labels
    ) == {"MODERATE_TREND_ZONE"}


# ---------------------------------------------------------------------------
# Zone HCPV and BVW: summed and thickness-weighted
# ---------------------------------------------------------------------------
def _zone_well():
    # Midpoint cells: 1, 1, 1.5, 2, 2 (sum 7.5); pay cells 1, 1, 1.5, 2 (sum 5.5).
    return _well(
        [100.0, 101.0, 102.0, 104.0, 106.0],
        NET_PAY=[1, 1, 1, 0, 1],
        PHIE=[0.2, 0.3, 0.1, 0.25, 0.2],
        SW=[0.5, 0.2, 0.4, 0.3, 1.0],
    )


def test_zone_hcpv_is_a_thickness_weighted_sum():
    from petrophys.evaluation.summaries.zone_summary import calculate_zone_summary

    zone = calculate_zone_summary(_zone_well(), 100.0, 106.0)
    assert zone.net_pay == pytest.approx(5.5)
    # sum(PHIE * (1 - SW) * h) = 0.1*1 + 0.24*1 + 0.06*1.5 + 0*2
    assert zone.hcpv == pytest.approx(0.43)
    # sum(PHIE * SW * h) / sum(h) = (0.1 + 0.06 + 0.06 + 0.4) / 5.5
    assert zone.bvw == pytest.approx(0.62 / 5.5)
    assert zone.avg_phie == pytest.approx(0.2) and zone.avg_sw == pytest.approx(0.525)

    with historical_benchmark():
        old = calculate_zone_summary(_zone_well(), 100.0, 106.0)
    assert old.hcpv == pytest.approx(0.2 * (1.0 - 0.525) * 5.5)
    assert old.bvw == pytest.approx(0.2 * 0.525)


def test_formation_hcpv_uses_the_zone_sum():
    from petrophys.evaluation.summaries.formation_analysis import analyze_formations
    from petrophys.model.intervals import Formation

    result = analyze_formations(_zone_well(), [Formation("A", 100.0, 106.0)])
    assert result.summaries[0].zone_summary.hcpv == pytest.approx(0.43)


def test_zone_summary_prefers_core_corrected_curves():
    from petrophys.evaluation.summaries.zone_summary import calculate_zone_summary

    well = _well(
        np.arange(100.0, 104.0),
        NET_PAY=np.ones(4), PHIE=np.full(4, 0.2), PHIE_CC=[0.3, 0.3, np.nan, 0.3],
        SW=np.full(4, 0.5), PERM=np.full(4, 10.0), PERM_CC=np.full(4, 100.0),
    )
    zone = calculate_zone_summary(well, 100.0, 103.0)
    # PHIE_CC where it exists, PHIE at the gap: (0.3 + 0.3 + 0.2 + 0.3) / 4
    assert zone.avg_phie == pytest.approx(0.275)
    assert zone.hcpv == pytest.approx(0.275 * 0.5 * 4.0)
    assert zone.avg_perm == pytest.approx(100.0)

    with historical_benchmark():
        old = calculate_zone_summary(well, 100.0, 103.0)
    assert old.avg_phie == pytest.approx(0.2) and old.avg_perm == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# calculate_hcpv_from_well: net pay over samples with finite PHIE and SW, PHIE_CC
# ---------------------------------------------------------------------------
def test_hcpv_from_well_counts_net_pay_with_finite_inputs():
    from petrophys.evaluation.volumetrics.hydrocarbon_pore_volume import calculate_hcpv_from_well

    well = _well([5000.0, 5001.0, 5002.0, 5003.0], "ft",
                 NET_PAY=np.ones(4), PHIE=[0.2, np.nan, 0.2, 0.2], SW=np.full(4, 0.3))
    with pytest.warns(UserWarning, match="1 of 4"):
        result = calculate_hcpv_from_well(well, area_acres=1.0)
    assert result.net_pay_ft == pytest.approx(3.0)
    assert result.hcpv_acft == pytest.approx(3.0 * 0.2 * 0.7)

    with historical_benchmark(), pytest.warns(UserWarning, match="1 of 4"):
        old = calculate_hcpv_from_well(well, area_acres=1.0)
    assert old.net_pay_ft == pytest.approx(4.0)


def test_hcpv_from_well_prefers_phie_cc():
    from petrophys.evaluation.volumetrics.hydrocarbon_pore_volume import calculate_hcpv_from_well

    well = _well([5000.0, 5001.0], "ft", NET_PAY=np.ones(2), PHIE=np.full(2, 0.2),
                 PHIE_CC=np.full(2, 0.3), SW=np.full(2, 0.3))
    assert calculate_hcpv_from_well(well, area_acres=1.0).avg_porosity == pytest.approx(0.3)
    with historical_benchmark():
        assert calculate_hcpv_from_well(well, area_acres=1.0).avg_porosity == pytest.approx(0.2)

    only_cc = _well([5000.0, 5001.0], "ft", NET_PAY=np.ones(2), PHIE_CC=np.full(2, 0.3), SW=np.full(2, 0.3))
    assert calculate_hcpv_from_well(only_cc, area_acres=1.0).avg_porosity == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# Pay averages and BvwModule prefer PHIE_CC / PERM_CC; BVW is not overwritten
# ---------------------------------------------------------------------------
def test_pay_averages_prefer_core_corrected_curves():
    from petrophys.evaluation.summaries.averages import calculate_pay_averages

    well = _well(np.arange(100.0, 104.0), NET_PAY=np.ones(4), PHIE=np.full(4, 0.2),
                 PHIE_CC=[0.3, 0.3, np.nan, 0.3], PERM=np.full(4, 10.0), PERM_CC=np.full(4, 100.0))
    averages = calculate_pay_averages(well)
    assert averages["PHIE"] == pytest.approx(0.275) and averages["PERM"] == pytest.approx(100.0)
    assert set(averages) == {"PHIE", "SW", "VCL", "PERM", "BVW"}
    with historical_benchmark():
        old = calculate_pay_averages(well)
    assert old["PHIE"] == pytest.approx(0.2) and old["PERM"] == pytest.approx(10.0)


def test_bvw_module_prefers_phie_cc_and_keeps_an_existing_bvw():
    from petrophys.evaluation.bvw import BvwModule

    well = _well(np.arange(100.0, 102.0), PHIE=np.full(2, 0.2), PHIE_CC=np.full(2, 0.3),
                 SW=np.full(2, 0.5), BVW=np.full(2, 0.123))
    np.testing.assert_allclose(BvwModule().calculate(well), 0.15)
    BvwModule().add_to_well(well)
    np.testing.assert_allclose(well.get("BVW").data, 0.123)

    with historical_benchmark():
        np.testing.assert_allclose(BvwModule().calculate(well), 0.10)
        BvwModule().add_to_well(well)
    np.testing.assert_allclose(well.get("BVW").data, 0.10)

    fresh = _well(np.arange(100.0, 102.0), PHIE=np.full(2, 0.2), SW=np.full(2, 0.5))
    BvwModule().add_to_well(fresh)
    np.testing.assert_allclose(fresh.get("BVW").data, 0.10)


# ---------------------------------------------------------------------------
# NetPayModule resolves lithology labels through cutoff_family
# ---------------------------------------------------------------------------
def test_net_pay_module_uses_cutoff_family():
    from petrophys.evaluation.net_pay.net_pay_module import NetPayModule
    from petrophys.model.constants import DefaultParameters

    well = _well([1000.0, 1000.5])
    phie, vcl, sw = np.array([0.06, 0.06]), np.array([0.1, 0.1]), np.array([0.65, 0.65])
    labels = np.array(["limestone", "sandstone"], dtype=object)

    per_depth = NetPayModule(lithology=labels).calculate(well, vcl=vcl, phie=phie, sw=sw)
    assert per_depth.net_pay.tolist() == [True, False]
    assert NetPayModule(lithology="chalk").phie_cutoff == DefaultParameters.PHIE_CUTOFF_CARBONATE
    assert NetPayModule(lithology="Dolomite").sw_cutoff == DefaultParameters.SW_CUTOFF_CARBONATE
    assert NetPayModule(lithology="sandstone").phie_cutoff == DefaultParameters.PHIE_CUTOFF_CLASTIC

    with historical_benchmark():
        old = NetPayModule(lithology=labels).calculate(well, vcl=vcl, phie=phie, sw=sw)
        assert NetPayModule(lithology="chalk").phie_cutoff == DefaultParameters.PHIE_CUTOFF_CLASTIC
    assert old.net_pay.tolist() == [False, False]
