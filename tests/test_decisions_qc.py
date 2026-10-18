"""Quality-control decisions of release_checks/METHOD_DECISIONS.md.

DCAL is a differential caliper (-5 to +5 in), value-range QC converts
recognised units before screening and sets +/-inf to NaN, WellboreQC takes
its roughness threshold per metre and applies the washout threshold, and the
saturation-height plot labels the points 'data'. Inside
``historical_benchmark`` the benchmark packages' behaviour is kept.
"""

import numpy as np
import pytest

import petrophys as pp
from petrophys._internal.benchmark import historical_benchmark
from petrophys.preprocessing.quality import WellboreQC
from petrophys.preprocessing.quality.value_ranges import (
    VALUE_RANGE_RULES,
    apply_value_range_qc,
    clean_curve_values,
    rule_for_curve,
    value_range_qc_note,
)


def _well(name="QC", n=4, unit="m"):
    well = pp.Well(name)
    well.depth = np.arange(1000.0, 1000.0 + 0.5 * n, 0.5)
    well.depth_unit = unit
    return well


# --- DCAL -----------------------------------------------------------------------


def test_dcal_has_a_differential_caliper_rule():
    canonical, rule = rule_for_curve("DCAL")
    assert canonical == "DCAL"
    assert (rule.low, rule.high, rule.action) == (-5.0, 5.0, "nan")
    assert "DCAL" not in VALUE_RANGE_RULES["CALI"].aliases
    with historical_benchmark():
        assert rule_for_curve("DCAL")[0] == "CALI"


def test_dcal_screening_keeps_the_differential_caliper():
    if "16_7-1" in pp.list_samples():
        well = pp.read(pp.sample("16_7-1"))
        assert well.get("DCAL") is not None
    else:
        well = _well(n=6)
        well.add_curve("DCAL", np.array([-6.0, -1.0, 0.0, 0.5, 2.0, 7.0]), "in", "Differential caliper")
    dcal = np.array(well.get("DCAL").data, dtype=float)
    finite = int(np.count_nonzero(np.isfinite(dcal)))
    expected = int(np.count_nonzero(np.isfinite(dcal) & ((dcal < -5.0) | (dcal > 5.0))))

    report = apply_value_range_qc(well, curve_names=["DCAL"])
    assert report[0]["canonical_curve"] == "DCAL"
    assert report[0]["changed"] == expected
    assert report[0]["changed"] < 0.05 * finite

    well.get("DCAL").data[:] = dcal
    with historical_benchmark():
        report = apply_value_range_qc(well, curve_names=["DCAL"])
    assert report[0]["canonical_curve"] == "CALI"
    assert report[0]["changed"] == int(np.count_nonzero(np.isfinite(dcal) & (dcal < 4.0)))


# --- units and infinities -------------------------------------------------------


def test_recognised_units_are_converted_before_screening():
    well = _well()
    well.add_curve("NPHI", np.array([-20.0, 10.0, 30.0, 70.0]), "pu", "Neutron")
    well.add_curve("CALI", np.array([50.0, 215.9, 1100.0, np.nan]), "mm", "Caliper")
    well.add_curve("DT", np.array([50.0, 200.0, 900.0, 1000.0]), "us/m", "Sonic")
    well.add_curve("SW", np.array([-10.0, 50.0, 120.0, 80.0]), "%", "Water saturation")
    well.add_curve("GR", np.array([-5.0, 50.0, 150.0, 500.0]), "gAPI", "Gamma ray")

    report = {item["curve"]: item for item in apply_value_range_qc(well)}

    np.testing.assert_allclose(well.get("NPHI").data, [np.nan, 10.0, 30.0, np.nan], equal_nan=True)
    np.testing.assert_allclose(well.get("CALI").data, [np.nan, 215.9, np.nan, np.nan], equal_nan=True)
    np.testing.assert_allclose(well.get("DT").data, [np.nan, 200.0, 900.0, np.nan], equal_nan=True)
    np.testing.assert_allclose(well.get("SW").data, [0.0, 50.0, 100.0, 80.0])
    np.testing.assert_allclose(well.get("GR").data, [np.nan, 50.0, 150.0, np.nan], equal_nan=True)
    assert (report["NPHI"]["low"], report["NPHI"]["high"], report["NPHI"]["unit"]) == (-15.0, 60.0, "pu")
    assert report["CALI"]["unit"] == "mm" and report["CALI"]["low"] == pytest.approx(101.6, abs=0.01)
    assert report["DT"]["unit"] == "us/m" and report["DT"]["high"] == pytest.approx(984.25, abs=0.01)
    assert (report["SW"]["low"], report["SW"]["high"], report["SW"]["unit"]) == (0.0, 100.0, "%")
    assert "unit" not in report["GR"]
    assert "NPHI set NaN=2 (<-15.0:1, >60.0:1 pu)" in value_range_qc_note(list(report.values()))


def test_standard_and_unrecognised_units_are_screened_as_they_are():
    cleaned, item = clean_curve_values("NPHI", np.array([0.1, 0.7]), unit="v/v")
    np.testing.assert_allclose(cleaned, [0.1, np.nan], equal_nan=True)
    assert "unit" not in item and (item["low"], item["high"]) == (-0.15, 0.6)
    cleaned, item = clean_curve_values("NPHI", np.array([0.1, 0.7]), unit="frac")
    assert "unit" not in item
    cleaned, item = clean_curve_values("NPHI", np.array([10.0, 70.0]), unit="banana")
    np.testing.assert_allclose(cleaned, [np.nan, np.nan])
    cleaned, item = clean_curve_values("NPHI", np.array([10.0, 70.0]))
    np.testing.assert_allclose(cleaned, [np.nan, np.nan])


def test_the_benchmark_screens_every_curve_in_the_rule_unit():
    well = _well()
    well.add_curve("NPHI", np.array([-20.0, 10.0, 30.0, 70.0]), "pu", "Neutron")
    with historical_benchmark():
        report = apply_value_range_qc(well)
    np.testing.assert_allclose(well.get("NPHI").data, [np.nan] * 4, equal_nan=True)
    assert report[0]["changed"] == 4 and "unit" not in report[0]


def test_infinities_become_nan_under_clip_and_nan_rules():
    well = _well()
    well.add_curve("SW", np.array([-np.inf, 0.5, np.inf, 1.5]), "v/v", "Water saturation")
    well.add_curve("GR", np.array([np.inf, 50.0, -np.inf, 150.0]), "gAPI", "Gamma ray")
    report = {item["curve"]: item for item in apply_value_range_qc(well)}
    np.testing.assert_allclose(well.get("SW").data, [np.nan, 0.5, np.nan, 1.0], equal_nan=True)
    np.testing.assert_allclose(well.get("GR").data, [np.nan, 50.0, np.nan, 150.0], equal_nan=True)
    assert (report["SW"]["changed"], report["SW"]["below"], report["SW"]["above"]) == (3, 1, 2)
    assert (report["GR"]["changed"], report["GR"]["below"], report["GR"]["above"]) == (2, 1, 1)


def test_the_benchmark_leaves_infinities():
    well = _well()
    well.add_curve("SW", np.array([-np.inf, 0.5, np.inf, 1.5]), "v/v", "Water saturation")
    well.add_curve("GR", np.array([np.inf, 50.0, -np.inf, 150.0]), "gAPI", "Gamma ray")
    with historical_benchmark():
        report = {item["curve"]: item for item in apply_value_range_qc(well)}
    np.testing.assert_allclose(well.get("SW").data, [-np.inf, 0.5, np.inf, 1.0])
    np.testing.assert_allclose(well.get("GR").data, [np.inf, 50.0, -np.inf, 150.0])
    assert report["SW"]["changed"] == 1 and report["GR"]["changed"] == 0


# --- WellboreQC -----------------------------------------------------------------


def _caliper_well(unit, cali, bs=8.5):
    well = _well("BH", n=cali.size, unit="m")
    if unit == "ft":
        well.depth = well.depth * 3.28084
        well.depth_unit = "ft"
    well.add_curve("CALI", cali, "in", "Caliper")
    well.add_curve("BS", np.full(cali.size, bs), "in", "Bit size")
    return well


def _rugose_caliper():
    cali = np.full(40, 8.6)
    cali[15:25] += np.abs(np.random.default_rng(1).normal(0.0, 1.5, 10))
    return cali


def test_roughness_threshold_is_per_metre_whatever_the_depth_unit():
    in_m = WellboreQC().analyze(_caliper_well("m", _rugose_caliper()), add_curves=False)
    in_ft = WellboreQC().analyze(_caliper_well("ft", _rugose_caliper()), add_curves=False)
    flags_m = in_m.derived_curves["BAD_HOLE"].data
    flags_ft = in_ft.derived_curves["BAD_HOLE"].data
    assert np.nansum(flags_m) > 0
    np.testing.assert_array_equal(flags_ft, flags_m)
    np.testing.assert_allclose(
        in_ft.derived_curves["CALI_ROUGHNESS"].data,
        in_m.derived_curves["CALI_ROUGHNESS"].data,
        rtol=1e-5,
        atol=1e-9,
    )
    assert in_m.derived_curves["CALI_ROUGHNESS"].unit == "in/m"
    assert "in/m" in in_m.sections["bad_hole_detection"].calculation_thresholds


def test_the_benchmark_takes_the_roughness_per_logged_depth_unit():
    with historical_benchmark():
        in_m = WellboreQC().analyze(_caliper_well("m", _rugose_caliper()), add_curves=False)
        in_ft = WellboreQC().analyze(_caliper_well("ft", _rugose_caliper()), add_curves=False)
    assert np.nansum(in_m.derived_curves["BAD_HOLE"].data) > 0
    assert np.nansum(in_ft.derived_curves["BAD_HOLE"].data) == 0
    np.testing.assert_allclose(
        in_ft.derived_curves["CALI_ROUGHNESS"].data,
        in_m.derived_curves["CALI_ROUGHNESS"].data * 0.3048,
        rtol=1e-5,
        atol=1e-9,
    )
    assert in_m.derived_curves["CALI_ROUGHNESS"].unit == ""
    assert "caliper_washout" not in in_m.metadata["qc_summary"]["detection_method_counts"]


def test_a_smooth_overgauge_hole_is_a_washout():
    well = _caliper_well("m", np.full(30, 11.0))
    result = WellboreQC().analyze(well, add_curves=False)
    assert np.all(result.derived_curves["BAD_HOLE"].data == 1.0)
    summary = result.metadata["qc_summary"]
    assert summary["bad_hole_percentage"] == 100.0
    assert summary["detection_method_counts"] == {
        "caliper_roughness": 0,
        "drho_fallback": 0,
        "caliper_washout": 30,
    }
    assert "washout CALI - BS > 1.5 in" in result.sections["bad_hole_detection"].calculation_thresholds

    in_gauge = WellboreQC().analyze(_caliper_well("m", np.full(30, 9.5)), add_curves=False)
    assert np.all(in_gauge.derived_curves["BAD_HOLE"].data == 0.0)
    assert in_gauge.metadata["qc_summary"]["detection_method_counts"]["caliper_washout"] == 0

    with historical_benchmark():
        result = WellboreQC().analyze(well, add_curves=False)
    assert np.all(result.derived_curves["BAD_HOLE"].data == 0.0)
    assert result.metadata["qc_summary"]["detection_method_counts"] == {
        "caliper_roughness": 30,
        "drho_fallback": 0,
    }


def test_the_washout_threshold_is_the_parameter():
    well = _caliper_well("m", np.full(30, 11.0))
    result = WellboreQC(washout_threshold=3.0).analyze(well, add_curves=False)
    assert np.all(result.derived_curves["BAD_HOLE"].data == 0.0)


# --- saturation-height plot -----------------------------------------------------


def test_saturation_height_points_are_labelled_data_by_default():
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    from petrophys.interpretation.saturation.height.saturation_height_module import fit_j_function
    from petrophys.visualization import create_saturation_height_plot

    j = np.geomspace(0.2, 20, 40)
    sw = np.clip(0.3 * j ** -0.4, 0.05, 1.0)
    model = fit_j_function(j, sw)
    fig = create_saturation_height_plot(model, j=j, sw=sw)
    labels = [text.get_text() for text in fig.axes[0].get_legend().get_texts()]
    assert "data" in labels and "core data" not in labels
    plt.close(fig)
    fig = create_saturation_height_plot(model, j=j, sw=sw, data_label="SW_INPUT (reference)")
    labels = [text.get_text() for text in fig.axes[0].get_legend().get_texts()]
    assert "SW_INPUT (reference)" in labels
    plt.close(fig)
