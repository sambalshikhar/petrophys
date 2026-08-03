import numpy as np

from petrophys import Well
from petrophys.interpretation.geochemistry import (
    has_spectral_gamma_inputs,
    interpret_spectral_gamma,
)
from petrophys.workflow import DynamicWorkflow, DynamicWorkflowConfig, WorkflowStep


def _well(n=4):
    well = Well("SPECTRAL")
    well.depth = np.arange(1000.0, 1000.0 + n)
    well.depth_unit = "m"
    return well


def test_slb_hngs_products_are_standardized_and_reconstructed():
    well = _well()
    well.add_curve("POTA", [1.0, 2.0, 3.0, 4.0], "%", "Potassium Concentration")
    well.add_curve("THOR", [2.0, 4.0, 6.0, 8.0], "ppm", "Thorium Concentration")
    well.add_curve("URAN", [1.0, 1.5, 2.0, 2.5], "ppm", "Uranium Concentration")
    well.add_curve("SGR", [40.0, 60.0, 80.0, 100.0], "gAPI", "Spectroscopy Gamma Ray")
    well.add_curve("CGR", [30.0, 45.0, 60.0, 75.0], "gAPI", "Computed Gamma Ray")

    result = interpret_spectral_gamma(well)

    assert result.sources == {
        "POTASSIUM": "POTA",
        "THORIUM": "THOR",
        "URANIUM": "URAN",
        "SGR": "SGR",
        "CGR": "CGR",
    }
    np.testing.assert_allclose(well.get("POTASSIUM").data, [1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(well.get("TH_K_RATIO").data, [2.0, 2.0, 2.0, 2.0])
    expected_cgr = 16.32 * well.get("POTA").data + 3.93 * well.get("THOR").data
    np.testing.assert_allclose(well.get("CGR_CALC").data, expected_cgr)
    assert well.get("POTASSIUM").metadata.source == "spectral_gamma"


def test_baker_hughes_aliases_convert_units_without_accepting_kth_ratio():
    well = _well(2)
    well.add_curve("K40", [0.01, 0.02], "fraction", "Potassium content")
    well.add_curve("TH232", [0.001, 0.002], "%", "Thorium content")
    well.add_curve("U238", [1e-6, 2e-6], "g/g", "Uranium content")
    well.add_curve("GRSL", [55.0, 65.0], "API", "Gamma ray from spectrum")
    well.add_curve("KTH", [1.2, 1.3], "ratio", "Potassium thorium ratio")

    result = interpret_spectral_gamma(well)

    np.testing.assert_allclose(result.measurements["POTASSIUM"], [1.0, 2.0])
    np.testing.assert_allclose(result.measurements["THORIUM"], [10.0, 20.0])
    np.testing.assert_allclose(result.measurements["URANIUM"], [1.0, 2.0])
    assert result.sources["SGR"] == "GRSL"
    assert "CGR" not in result.measurements


def test_ambiguous_short_names_require_physical_evidence():
    well = _well(2)
    well.add_curve("K", [10.0, 20.0], "mD", "Permeability")
    well.add_curve("TH", [80.0, 90.0], "us/ft", "Compressional slowness")
    well.add_curve("TOTA", [1.0, 2.0], "", "Total acquisition channel")

    assert not has_spectral_gamma_inputs(well)
    result = interpret_spectral_gamma(well)
    assert not result.measurements
    assert well.get("POTASSIUM") is None


def test_halliburton_grkt_is_recognized_as_uranium_free_gamma():
    well = _well(2)
    well.add_curve("GRTO", [80.0, 90.0], "API", "Total spectral gamma ray")
    well.add_curve("GRKT", [50.0, 60.0], "API", "Potassium-thorium gamma ray")
    well.add_curve("POTA", [2.0, 2.5], "%", "Formation potassium concentration")
    well.add_curve("THOR", [8.0, 9.0], "ppm", "Formation thorium concentration")
    well.add_curve("URAN", [3.0, 4.0], "ppm", "Formation uranium concentration")

    result = interpret_spectral_gamma(well)

    assert result.sources["SGR"] == "GRTO"
    assert result.sources["CGR"] == "GRKT"


def test_description_fallback_supports_unknown_vendor_mnemonics():
    well = _well(2)
    well.add_curve(
        "MSG_TOTAL_QC",
        [70.0, 80.0],
        "API",
        "Spectral gamma ray total, environmentally corrected",
    )
    well.add_curve(
        "RAD_K_01",
        [1.5, 2.0],
        "%",
        "Formation potassium concentration",
    )

    result = interpret_spectral_gamma(well)

    assert result.sources["SGR"] == "MSG_TOTAL_QC"
    assert result.sources["POTASSIUM"] == "RAD_K_01"


def test_dynamic_workflow_uses_uranium_free_gamma_for_generic_vcl():
    well = _well(5)
    well.add_curve("POTA", np.linspace(0.5, 3.0, 5), "%", "Potassium Concentration")
    well.add_curve("THOR", np.linspace(1.0, 12.0, 5), "ppm", "Thorium Concentration")

    workflow = DynamicWorkflow(DynamicWorkflowConfig(standardize_input=False))
    result = workflow.run_selected_steps(
        well,
        [WorkflowStep.SPECTRAL_GAMMA, WorkflowStep.VCLAY],
    )

    traces = {step.step: step for step in result.steps_executed}
    assert traces["spectral_gamma"].success
    assert traces["vclay"].parameters["gr_curve"] == "CGR_CALC"
    assert well.get("VCL") is not None


def test_vendor_benchmark_profile_does_not_add_new_specialist_steps():
    well = _well(30)
    well.add_curve("POTA", np.linspace(1.0, 3.0, 30), "%", "Potassium Concentration")
    well.add_curve("THOR", np.linspace(2.0, 6.0, 30), "ppm", "Thorium Concentration")
    well.add_curve(
        "LITHOLOGY_GEOLINK",
        np.ones(30),
        "code",
        "Vendor lithology code",
    )
    config = DynamicWorkflowConfig(
        standardize_input=False,
        interpretation_profile="vendor_benchmark",
    )

    result = DynamicWorkflow(config).run(well)

    executed = {step.step for step in result.steps_executed}
    assert "spectral_gamma" not in executed
    assert "elemental_spectroscopy" not in executed
    assert well.get("CGR_CALC") is None
