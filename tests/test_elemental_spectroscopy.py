import numpy as np
import pytest

from petrophys import Well
from petrophys.interpretation.geochemistry import (
    has_elemental_spectroscopy_inputs,
    interpret_elemental_spectroscopy,
    oxide_closure,
)
from petrophys.workflow import DynamicWorkflow, DynamicWorkflowConfig, WorkflowStep


def _well(n=3):
    well = Well("ELEMENTAL")
    well.depth = np.arange(2000.0, 2000.0 + n)
    well.depth_unit = "m"
    return well


def test_slb_walk2_dry_weights_and_matching_uncertainty_are_resolved():
    well = _well()
    well.add_curve(
        "DWAL_WALK2",
        [0.05, 0.06, 0.07],
        "kgf/kgf",
        "Dry Weight Fraction Aluminum (SpectroLith WALK2 Model)",
    )
    well.add_curve(
        "DWSI_WALK2",
        [0.25, 0.30, 0.35],
        "kgf/kgf",
        "Dry Weight Fraction Silicon (SpectroLith WALK2 Model)",
    )
    well.add_curve(
        "DWCA_WALK2",
        [0.10, 0.12, 0.14],
        "kgf/kgf",
        "Dry Weight Fraction Calcium (SpectroLith WALK2 Model)",
    )
    well.add_curve(
        "DWAL_SIG_ACQ",
        [0.9, 0.9, 0.9],
        "kgf/kgf",
        "Statistical Uncertainty in Dry Weight Aluminum in Acquisition",
    )
    well.add_curve(
        "DWAL_SIG_WALK2",
        [0.005, 0.006, 0.007],
        "kgf/kgf",
        "Statistical Uncertainty in Dry Weight Aluminum (SpectroLith WALK2 Model)",
    )

    result = interpret_elemental_spectroscopy(well)

    assert set(result.dry_weight_fractions) == {"AL", "CA", "SI"}
    assert result.sources["ELEM_AL"] == "DWAL_WALK2"
    assert result.sources["ELEM_AL_SIGMA"] == "DWAL_SIG_WALK2"
    np.testing.assert_allclose(well.get("ELEM_AL_SIGMA").data, [0.005, 0.006, 0.007])
    np.testing.assert_allclose(well.get("ELEM_RATIO_SI_AL").data, [5.0, 5.0, 5.0])
    assert not result.warnings


def test_baker_dry_weights_and_halliburton_corrected_yields_stay_separate():
    well = _well(2)
    well.add_curve("WFAL", [5.0, 10.0], "%", "Aluminum dry weight fraction")
    well.add_curve("WFSI", [25.0, 30.0], "%", "Silicon dry weight fraction")
    well.add_curve(
        "CFEC", [0.12, 0.14], "", "GEM capture iron corrected relative yield"
    )
    well.add_curve(
        "CSIC", [0.30, 0.32], "", "GEM capture silicon corrected relative yield"
    )

    result = interpret_elemental_spectroscopy(well)

    np.testing.assert_allclose(result.dry_weight_fractions["AL"], [0.05, 0.10])
    assert "CAPTURE_FE" in result.relative_yields
    assert "CAPTURE_SI" in result.relative_yields
    assert "FE" not in result.dry_weight_fractions
    assert well.get("ELEM_CAPTURE_FE").unit == "relative_yield"


def test_description_fallback_supports_vendor_products_without_known_mnemonics():
    well = _well(2)
    well.add_curve(
        "GSI_A17",
        [6.0, 8.0],
        "%",
        "Aluminum elemental concentration, dry weight",
    )
    well.add_curve(
        "GSI_M42",
        [0.05, 0.06],
        "",
        "Capture magnesium relative yield",
    )

    result = interpret_elemental_spectroscopy(well)

    np.testing.assert_allclose(result.dry_weight_fractions["AL"], [0.06, 0.08])
    assert result.sources["ELEM_AL"] == "GSI_A17"
    assert "CAPTURE_MG" in result.relative_yields


def test_calc_caliper_is_not_mislabeled_as_gem_aluminum():
    well = _well(2)
    well.add_curve("CALC", [8.5, 8.6], "in", "Corrected caliper")

    assert not has_elemental_spectroscopy_inputs(well)
    assert not interpret_elemental_spectroscopy(well).relative_yields

    gem_well = _well(2)
    gem_well.add_curve(
        "CALC",
        [0.02, 0.03],
        "",
        "GEM corrected aluminum relative yield",
    )
    result = interpret_elemental_spectroscopy(gem_well)
    assert "GEM_AL" in result.relative_yields


def test_raw_spectral_bins_are_not_treated_as_processed_elements():
    well = _well(2)
    well.add_curve("S1SP[1]", [100.0, 120.0], "count", "Gamma spectrum energy bin")
    well.add_curve("A1TD[1]", [10.0, 11.0], "count", "Neutron time-distribution bin")

    assert not has_elemental_spectroscopy_inputs(well)


def test_combined_nickel_chromium_yield_is_not_labeled_as_pure_nickel():
    well = _well(2)
    well.add_curve(
        "CNC_20",
        [0.01, 0.02],
        "",
        "Capture Nickel-Chromium Relative Yield (20 Elemental Standards Processing)",
    )

    result = interpret_elemental_spectroscopy(well)

    assert "CAPTURE_NI_CR" in result.relative_yields
    assert "CAPTURE_NI" not in result.relative_yields


def test_oxide_closure_requires_calibration_and_closes_to_unity():
    result = oxide_closure(
        {"SI": np.array([0.4, 0.2]), "CA": np.array([0.2, 0.3])},
        sensitivities={"SI": 2.0, "CA": 1.0},
        oxide_factors={"SI": 2.0, "CA": 1.5},
    )

    np.testing.assert_allclose(result.closure, [1.0, 1.0])
    assert np.all(result.element_fractions["SI"] >= 0.0)
    with pytest.raises(ValueError, match="Missing sensitivities"):
        oxide_closure(
            {"SI": np.array([0.4]), "CA": np.array([0.2])},
            sensitivities={"SI": 2.0},
            oxide_factors={"SI": 2.0, "CA": 1.5},
        )


def test_mass_balance_flags_excess_without_forcing_incomplete_sum_to_one():
    well = _well(2)
    well.add_curve("WFAL", [0.2, 0.8], "fraction", "Aluminum dry weight fraction")
    well.add_curve("WFSI", [0.3, 0.5], "fraction", "Silicon dry weight fraction")

    interpret_elemental_spectroscopy(well)

    np.testing.assert_allclose(well.get("ELEM_MEASURED_SUM").data, [0.5, 1.3])
    np.testing.assert_allclose(
        well.get("ELEM_MASS_BALANCE_EXCESS_FLAG").data, [0.0, 1.0]
    )


def test_dynamic_workflow_records_elemental_sources_and_outputs():
    well = _well(2)
    well.add_curve("WFSI", [25.0, 30.0], "%", "Silicon dry weight fraction")
    workflow = DynamicWorkflow(DynamicWorkflowConfig(standardize_input=False))

    result = workflow.run_selected_steps(
        well,
        [WorkflowStep.ELEMENTAL_SPECTROSCOPY],
    )

    trace = next(step for step in result.steps_executed if step.step == "elemental_spectroscopy")
    assert trace.success
    assert trace.input_curves == ["WFSI"]
    assert "ELEM_SI" in trace.output_curves
    assert trace.parameters["dry_weight_elements"] == ["SI"]
