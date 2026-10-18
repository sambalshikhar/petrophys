"""Regression tests for reporting and quality-control fixes."""

from __future__ import annotations

import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import petrophys as pp
from petrophys._internal.benchmark import historical_benchmark
from petrophys.io.las_reader import read_las
from petrophys.reporting import reference_export as rx


def _well(name="T-1", depth=None, unit="m"):
    well = pp.Well(name)
    well.depth = np.arange(1000.0, 1010.0, 0.5) if depth is None else np.asarray(depth, dtype=float)
    well.depth_unit = unit
    return well


# --- reference_export ---------------------------------------------------------


def test_format_value_keeps_small_values():
    assert rx._format_value(3.44e-7) == "3.44e-07"
    assert rx._format_value(-4e-7) == "-4e-07"
    assert rx._format_value(3.456789e-6) == "3.45679e-06"
    assert rx._format_value(1907.72) == "1907.72"
    assert rx._format_value(0.0) == "0"
    with historical_benchmark():
        assert rx._format_value(3.44e-7) == "0"


def test_las_export_converts_percent_saturation_and_writes_decreasing_header(tmp_path):
    well = _well(depth=[1002.0, 1001.0, 1000.0])
    well.add_curve("SW", np.array([25.0, 50.0, np.nan]), "%", "SW")
    well.add_curve("VSH", np.array([10.0, 20.0, 30.0]), "%", "VSH")
    well.add_curve("PERM", np.array([3.44e-7, 0.5, 2000.0]), "mD", "PERM")
    out = rx.export_example_format_las(well, tmp_path / "out.las")
    text = out.read_text().splitlines()
    values = {line[:4]: line.split()[2] for line in text if line.startswith(("STRT", "STOP", "STEP"))}
    assert values == {"STRT": "1002", "STOP": "1000", "STEP": "-1"}
    back = read_las(out)
    np.testing.assert_allclose(back.get("SW").data, [0.25, 0.5, np.nan])
    np.testing.assert_allclose(back.get("VCL").data, [0.1, 0.2, 0.3])
    np.testing.assert_allclose(back.get("PERM").data, [3.44e-7, 0.5, 2000.0])


def test_temperature_converters_parse_whole_words_and_kelvin():
    to_f, to_c = rx._convert_temperature_to_fahrenheit, rx._convert_temperature_to_celsius
    np.testing.assert_allclose(to_f(np.array([100.0]), "Celsius (ref)"), [212.0])
    np.testing.assert_allclose(to_f(np.array([100.0]), "deg C (from log)"), [212.0])
    np.testing.assert_allclose(to_f(np.array([100.0]), "K"), [-279.67])
    np.testing.assert_allclose(to_f(np.array([100.0]), "degF"), [100.0])
    np.testing.assert_allclose(to_c(np.array([212.0]), "K"), [-61.15])
    np.testing.assert_allclose(to_c(np.array([212.0]), "degF"), [100.0])
    with historical_benchmark():
        np.testing.assert_allclose(to_f(np.array([100.0]), "K"), [100.0])


def test_tops_with_one_well_spelled_differently_raise(tmp_path):
    tops = tmp_path / "tops.csv"
    tops.write_text("well,formation,top\nWell-A,Upper Fm.,1000\nWELL-A,Middle Fm.,1500\nwell-a ,Lower Fm.,2000\n")
    with pytest.raises(ValueError, match="spells one well"):
        rx._load_excel_formation_segments(tops)


def test_tops_prefix_fallback_does_not_take_another_wells_tops(tmp_path):
    tops = tmp_path / "tops.csv"
    tops.write_text("well,formation,top\n15/9-19 S,Hugin Fm.,3600\n15/9-19 S,Sleipner Fm.,3700\n16/1-2,Draupne Fm.,2000\n")
    by_well = rx._load_excel_formation_segments(tops)
    assert rx._formation_segments_for_well(Path("15_9-1.las"), _well("15/9-1"), by_well) == []
    # A well named without its "S" suffix still finds its own tops.
    segments = rx._formation_segments_for_well(Path("15_9-19.las"), _well("15/9-19"), by_well)
    assert [segment["name"] for segment in segments] == ["Hugin Fm.", "Sleipner Fm."]


def test_formation_names_colliding_after_sanitizing_share_a_key():
    assert rx._formation_file_key("LKC/G", False) == rx._formation_file_key("lkc_g", False)
    assert rx._formation_file_key("LKC/G", True) != rx._formation_file_key("LKC_G", True)


def test_lithology_workbook_rejects_fractional_and_duplicate_codes(tmp_path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("openpyxl")
    path = tmp_path / "codes.xlsx"
    columns = ["Attribute", "Lithology Attribute", "Abbreviation"]
    pd.DataFrame([["Sandstone", "Sand", 1], ["Shale", "Shale", 2.6]], columns=columns).to_excel(path, index=False)
    with pytest.raises(ValueError, match="not an integer"):
        rx._load_lithology_mapping_workbook(path)
    pd.DataFrame([["Sandstone", "Sand", 1], ["Dolomite", "Dolo", 1]], columns=columns).to_excel(path, index=False)
    with pytest.raises(ValueError, match="defined twice"):
        rx._load_lithology_mapping_workbook(path)
    pd.DataFrame([["Sandstone", "Sand", 1], ["Shale", "Shale", 2]], columns=columns).to_excel(path, index=False)
    assert sorted(rx._load_lithology_mapping_workbook(path)) == [1, 2]


def test_temp_and_group_code_notes_match_the_calculation():
    well = _well(unit="m")
    well.add_curve("TEMP", np.linspace(80.0, 90.0, well.n_samples), "degF", "TEMP")
    step = SimpleNamespace(
        method="linear_gradient_from_BHT",
        parameters={"surface_temp_f": 68.0, "bht_f": 250.0, "gradient_f_per_100ft": 1.6},
    )
    note = rx._compact_curve_note("TEMP", "computed", "", step, well)
    assert "DEPT_ft/100 (DEPT in ft from depth 0)" in note and "minDEPT" not in note
    assert "minDEPT" in rx._compact_curve_note("TEMP", "computed", "", step, well, benchmark=True)
    note = rx._compact_curve_note("GT_LITHOLOGY_GROUP_CODE", "computed", "", None, well)
    assert "no_lithology_data=0" in note and "special_lithology=14" in note


@pytest.fixture(scope="module")
def run_30_6_5():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        well, config = rx._read_context_input_well(Path(pp.sample("30_6-5")))
        original = set(well.curves)
        return original, rx._run_context_workflow(well, config)


def test_raw_logs_not_reported_as_derived(run_30_6_5):
    original, result = run_30_6_5
    derived = set(rx._derived_logs_computed(result.well, result, original))
    assert not derived & {"GR", "RHOB", "NPHI", "CALI", "SP", "RDEP", "DTC", "PEF", "BS"}
    assert {"PHIE", "SW", "VCL"} <= derived


def test_extra_curve_qc_does_not_leak_into_next_context(run_30_6_5):
    from petrophys.preprocessing.quality.value_ranges import VALUE_RANGE_QC_PARAMETER, apply_value_range_qc

    original, result = run_30_6_5
    group = result.well.depth_slice(float(result.well.top), float(result.well.top) + 50.0)
    apply_value_range_qc(group, rules=rx._context_value_range_rules(result))
    n = group.n_samples
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rx._build_reference_json_payload(
            group, result, Path("30_6-5.las"), original,
            extra_statistics_curves={"POISSON_RATIO": SimpleNamespace(data=np.full(n, 0.7), unit="")},
            derived_logs_override=[], status_targets_override=[],
        )
        rx._build_reference_json_payload(
            group, result, Path("30_6-5.las"), original,
            extra_statistics_curves={"COMPOSITE_CLAY": SimpleNamespace(data=np.full(n, 0.5), unit="")},
            derived_logs_override=[], status_targets_override=[],
        )
    curves = [item["curve"] for item in group.get_parameter(VALUE_RANGE_QC_PARAMETER)]
    assert "POISSON_RATIO" not in curves


# --- context export -----------------------------------------------------------


def test_package_overwrite_raises_value_error(tmp_path):
    source = Path(pp.sample("HW-3")).parent
    with pytest.raises(ValueError, match="single-file"):
        pp.export_context(str(source), tmp_path / "new", layout="formation_lithology", overwrite=True)
    with pytest.raises(ValueError, match="folder of well files"):
        pp.export_context(None, tmp_path / "new", layout="formation_lithology")


# --- quality control ----------------------------------------------------------


def _bad_hole_well(header_bs):
    well = _well()
    n = well.n_samples
    cali = np.full(n, 8.6)
    cali[5:8] = 11.0
    well.add_curve("CALI", cali, "in", "CALI")
    well.add_curve("RHOB", np.full(n, 2.45), "g/cc", "RHOB")
    well.header.bit_size = header_bs
    return well


@pytest.mark.parametrize("header_bs", [-999.25, 215.9])
def test_implausible_header_bit_size_is_not_used(header_bs):
    from petrophys.preprocessing.quality import detect_washout, generate_bad_hole_flags

    flags = generate_bad_hole_flags(_bad_hole_well(header_bs))
    assert not flags["washout"].any() and not flags["large_hole"].any()
    assert detect_washout(_bad_hole_well(header_bs)).statistics["washout_percent"] == 0.0
    assert int(generate_bad_hole_flags(_bad_hole_well(8.5))["washout"].sum()) == 3


def test_washout_thickness_uses_depth_increments():
    from petrophys.preprocessing.quality import WashoutDetection

    well = _bad_hole_well(8.5)
    well.depth = np.array([1000.0 + 0.5 * i if i < 10 else 1005.0 + 1.0 * (i - 9) for i in range(well.n_samples)])
    stats = WashoutDetection().detect(well).statistics
    assert stats["washout_thickness"] == pytest.approx(1.5)  # samples 5-7, 0.5 m apart
    well.get("CALI").data[:] = 11.0
    stats = WashoutDetection().detect(well).statistics
    assert stats["washout_thickness"] == pytest.approx(float(np.sum(np.abs(np.gradient(well.depth)))))


def test_wellbore_qc_reports_unknown_without_qc_curves():
    from petrophys.preprocessing.quality import WellboreQC

    well = _well()
    well.add_curve("GR", np.full(well.n_samples, 50.0), "API", "GR")
    summary = WellboreQC().analyze(well, add_curves=False).metadata["qc_summary"]
    assert summary["bad_hole_percentage"] is None


def test_good_data_mask_requires_present_curves_any_case():
    from petrophys.preprocessing.quality import generate_bad_hole_flags
    from petrophys.preprocessing.quality.bad_hole.curve_masking import get_good_data_mask

    well = _bad_hole_well(8.5)
    flags = generate_bad_hole_flags(well)
    assert get_good_data_mask(well, flags, ["rhob"]).sum() == 17
    assert get_good_data_mask(well, flags, ["RHOB", "NOPE"]).sum() == 0


def test_rule_for_curve_sxo_prefix_with_custom_rules():
    from petrophys.preprocessing.quality.value_ranges import ValueRangeRule, rule_for_curve

    assert rule_for_curve("SXO-1", {"GR": ValueRangeRule(0, 200, "nan", ("GR",))}) is None
    assert rule_for_curve("SXO-1")[0] == "SXO"


# --- normalization ------------------------------------------------------------


def test_histogram_functions_return_floats_for_integer_input():
    from petrophys.preprocessing.normalization import histogram_matching
    from petrophys.preprocessing.normalization.histogram_matching import histogram_equalization

    reference = np.array([10.0, 20, 30, 40, 50])
    np.testing.assert_allclose(
        histogram_matching(np.array([1, 2, 3, 4, 5]), reference),
        histogram_matching(np.array([1.0, 2, 3, 4, 5]), reference),
    )
    np.testing.assert_allclose(histogram_equalization(np.array([0, 1, 2, 10])), [0.0, 10 / 3, 20 / 3, 10.0])


def test_normalization_method_and_infinite_samples():
    from petrophys.preprocessing.normalization import normalize_log
    from petrophys.preprocessing.normalization.normalization_module import NormalizationModule

    with pytest.raises(ValueError, match="z-score"):
        NormalizationModule(method="z-score")
    NormalizationModule(method="percentile")
    result = normalize_log(np.array([1.0, 2, 3, np.inf]), 0, 1)
    np.testing.assert_allclose(result[:3], normalize_log(np.array([1.0, 2, 3]), 0, 1))
    assert np.isnan(result[3])
