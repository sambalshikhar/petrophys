import warnings

import numpy as np
import pytest

from petrophys import Well, sample
from petrophys.io import read_las
from petrophys.model.curve import Curve
from petrophys.preprocessing import CurveStandardizer, standardize_core_saturation, standardize_curves
from petrophys.preprocessing.standardization import standardize_units


@pytest.mark.parametrize("unit", ["%", " PCT ", "percent", "percentage", "pu"])
def test_explicit_percent_converts_even_below_one(unit):
    original = Curve("SWC", [0, 0.5, 1, 13.7, 84.6, 100, np.nan], unit)
    converted = standardize_core_saturation(original)
    np.testing.assert_allclose(converted.data, [0, .005, .01, .137, .846, 1, np.nan])
    assert converted.unit == "v/v"
    assert original.unit == unit
    assert original.data[3] == 13.7
    assert "core_saturation_normalization" not in original.metadata.custom


@pytest.mark.parametrize("unit", ["frac", "fraction", "v/v", "", "m3/m3"])
def test_mislabeled_percent_inference_and_idempotence(unit):
    original = Curve("SWC", [13.7, 84.6, 0, np.nan, -999.25], unit)
    with pytest.warns(UserWarning, match="inferred percent"):
        converted = standardize_core_saturation(original)
    np.testing.assert_allclose(converted.data, [.137, .846, 0, np.nan, np.nan])
    evidence = converted.metadata.custom["core_saturation_normalization"]
    assert evidence["factor"] == .01
    assert evidence["original_unit"] == unit
    with warnings.catch_warnings(record=True) as recorded:
        repeated = standardize_core_saturation(converted)
    assert recorded == []
    np.testing.assert_allclose(repeated.data, converted.data)
    assert repeated.metadata.custom["core_saturation_normalization"] == evidence


@pytest.mark.parametrize("unit", ["frac", "v/v", ""])
def test_fractional_core_is_unchanged(unit):
    original = Curve("SWC", [0, .137, .846, 1, np.nan], unit)
    with warnings.catch_warnings(record=True) as recorded:
        converted = standardize_core_saturation(original)
    assert recorded == []
    np.testing.assert_allclose(converted.data, original.data)


@pytest.mark.parametrize("values", [[.2, .4, 25], [.2, 25, 75], [25]])
def test_ambiguous_scale_never_rescales_good_fractional_values(values):
    original = Curve("SWC", values, "frac")
    with pytest.warns(UserWarning, match="core saturation"):
        converted = standardize_core_saturation(original)
    expected = np.asarray(values, dtype=float)
    expected[expected > 1] = np.nan
    np.testing.assert_allclose(converted.data, expected)
    assert converted.metadata.custom["core_saturation_normalization"]["factor"] == 1


def test_explicit_override_resolves_ambiguous_or_mislabeled_units():
    original = Curve("SWC", [.5, 25], "frac")
    converted = standardize_core_saturation(original, source_unit="%")
    np.testing.assert_allclose(converted.data, [.005, .25])
    with pytest.warns(UserWarning, match="masked"):
        fractional = standardize_core_saturation(original, source_unit="v/v")
    np.testing.assert_allclose(fractional.data, [.5, np.nan])


def test_percent_distribution_can_include_small_percentages():
    curve = Curve("SOC", [.6, 3, 7, 15, 16.9], "frac")
    with pytest.warns(UserWarning, match="inferred percent"):
        converted = standardize_core_saturation(curve)
    np.testing.assert_allclose(converted.data, [.006, .03, .07, .15, .169])


def test_small_fractional_overshoots_do_not_trigger_percentage_inference():
    curve = Curve("SWC", [1.01, 1.05, 1.1], "frac")
    with pytest.warns(UserWarning, match="core saturation"):
        converted = standardize_core_saturation(curve)
    assert converted.metadata.custom["core_saturation_normalization"]["factor"] == 1


def test_invalid_values_are_masked_not_clipped_and_missing_values_survive():
    original = Curve("SOC", [25, 75, -999.25, -1, 101, np.inf, np.nan], "%")
    with pytest.warns(UserWarning, match="masked 3"):
        converted = standardize_core_saturation(original)
    np.testing.assert_allclose(converted.data, [.25, .75, np.nan, np.nan, np.nan, np.nan, np.nan])
    assert np.isnan(standardize_core_saturation(Curve("SWC", [np.nan], "")).data).all()
    with pytest.raises(ValueError, match="source_unit"):
        standardize_core_saturation(Curve("SWC", [25], "mg/L"))


@pytest.mark.parametrize("mnemonic", ["SWC", "SW_CORE", "CORE_SW", "SWC:1", "SOC", "SG_CORE"])
def test_all_standardization_entry_points_handle_core_aliases(mnemonic):
    original = Curve(mnemonic, [25, 75], "%")
    np.testing.assert_allclose(standardize_units(original).data, [.25, .75])
    converted = CurveStandardizer(rename_curves=False).standardize_curve(original)
    assert converted.mnemonic == mnemonic
    np.testing.assert_allclose(converted.data, [.25, .75])
    well = Well("CORE")
    well.depth = np.array([1000., 1001.])
    well.add_curve(mnemonic, [25, 75], "%")
    standardize_curves(well, rename_curves=False, convert_units=False)
    np.testing.assert_allclose(well.get(mnemonic).data, [25, 75])
    standardize_curves(well, rename_curves=False)
    np.testing.assert_allclose(well.get(mnemonic).data, [.25, .75])


def test_inplace_normalization():
    curve = Curve("SWC", [25, 75], "%")
    assert standardize_core_saturation(curve, inplace=True) is curve
    np.testing.assert_allclose(curve.data, [.25, .75])


def test_bundled_las_raw_viewer_core_points_are_on_scale_without_mutating_well():
    from petrophys.visualization import create_log_viewer

    well = read_las(sample("30_6-5"))
    original = well.get("SWC").data.copy()
    viewer = create_log_viewer(well, depth_range=(2800, 3000), figsize=(18, 9))
    try:
        plotted = [
            collection for ax in viewer.axes for collection in ax.collections
            if collection.get_label().startswith("Sw Core")
        ]
        assert len(plotted) == 1
        xy = plotted[0].get_offsets()
        assert len(xy) == 74
        assert np.all((xy[:, 0] >= 0) & (xy[:, 0] <= 1))
        np.testing.assert_allclose(np.sort(1 - xy[:, 0]), np.sort(original[np.isfinite(original)] / 100))
        assert any("SWC: inferred percent" in message for message in viewer.warnings)
        np.testing.assert_allclose(well.get("SWC").data, original)
        assert well.get("SWC").unit == "frac"
        assert "core_saturation_normalization" not in well.get("SWC").metadata.custom
    finally:
        viewer.close()


def test_dynamic_workflow_normalizes_real_core_and_records_evidence():
    from petrophys.workflow import DynamicWorkflowConfig, run_dynamic_workflow

    well = read_las(sample("30_6-5"))
    with pytest.warns(UserWarning, match="inferred percent"):
        result = run_dynamic_workflow(well, DynamicWorkflowConfig(rw=.07))
    core = result.well.get("SWC")
    assert np.nanmin(core.data) == pytest.approx(.137)
    assert np.nanmax(core.data) == pytest.approx(.846)
    oil = result.well.get("SOC")
    assert np.isfinite(oil.data).sum() == 44
    assert np.nanmin(oil.data) == pytest.approx(.006)
    assert np.nanmax(oil.data) == pytest.approx(.169)
    step = next(step for step in result.steps_executed if step.step == "normalization")
    evidence = step.parameters["core_saturation_normalization"]["SWC"]
    assert evidence["factor"] == .01
    assert evidence["reason"] == "inferred_percent"
