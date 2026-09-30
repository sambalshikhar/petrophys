from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from petrophys import Well
from petrophys.evaluation.diagnostics import (
    HydrocarbonAnalysis,
    LithologyAnalysis,
    NumericalStatisticalAnalysis,
    PermeabilityAnalysis,
    PorosityAnalysis,
    SaturationAnalysis,
)
from petrophys.reporting.reference_export import (
    _build_lithology_group_segments,
    _build_reference_bad_hole_qc,
    _build_reference_derived_log_status,
    _build_reference_json_payload,
    _build_reference_statistics,
    _category_derived_logs_computed,
    _compact_curve_note,
    _curve_value_summary,
    _formation_segments_for_well,
    _input_log_display_list,
    _load_excel_formation_segments,
    _missing_curve_reason,
    _validate_single_well_tops_scope,
)
from petrophys._internal.analysis_helpers import CategoryAnalysisResult, DerivedCurve, get_curve_data
from petrophys.workflow import DynamicWorkflow, DynamicWorkflowConfig
from petrophys.workflow import run_comprehensive_analysis
from petrophys.workflow.zoned_batch import _finalize_export_curves


def _build_analysis_well() -> Well:
    depth = np.arange(1000.0, 1005.0, 0.5)
    well = Well("ANALYSIS-1")
    well.depth = depth
    well.depth_unit = "ft"
    well.add_curve("DEPT", depth, "ft", "Depth")
    well.add_curve("GR", np.array([45, 48, 52, 55, 60, 85, 95, 110, 120, 130]), "API", "Gamma Ray")
    well.add_curve("RHOB", np.array([2.18, 2.20, 2.22, 2.25, 2.30, 2.45, 2.50, 2.58, 2.62, 2.66]), "g/cc", "Bulk Density")
    well.add_curve("NPHI", np.array([0.08, 0.09, 0.10, 0.11, 0.12, 0.22, 0.24, 0.28, 0.30, 0.32]), "v/v", "Neutron Porosity")
    well.add_curve("PHIN", np.array([0.08, 0.09, 0.10, 0.11, 0.12, 0.22, 0.24, 0.28, 0.30, 0.32]), "v/v", "Corrected Neutron Porosity")
    well.add_curve("PHID", np.array([0.25, 0.24, 0.23, 0.22, 0.20, 0.12, 0.10, 0.07, 0.05, 0.03]), "v/v", "Density Porosity")
    well.add_curve("PHIS", np.array([0.14, 0.14, 0.14, 0.14, 0.13, 0.10, 0.09, 0.07, 0.06, 0.05]), "v/v", "Sonic Porosity")
    well.add_curve("PHIT", np.array([0.22, 0.21, 0.20, 0.19, 0.18, 0.12, 0.10, 0.08, 0.06, 0.04]), "v/v", "Total Porosity")
    well.add_curve("PHIE", np.array([0.20, 0.19, 0.18, 0.17, 0.16, 0.10, 0.09, 0.07, 0.05, 0.03]), "v/v", "Effective Porosity")
    well.add_curve("VCL", np.array([0.10, 0.12, 0.15, 0.18, 0.20, 0.35, 0.40, 0.55, 0.65, 0.75]), "v/v", "Clay Volume")
    well.add_curve("RDEP", np.array([120.0, 110.0, 100.0, 80.0, 50.0, 20.0, 12.0, 6.0, 3.0, 1.5]), "ohm.m", "Deep Resistivity")
    well.add_curve("RXO", np.array([60.0, 55.0, 52.0, 48.0, 35.0, 18.0, 10.0, 5.5, 3.2, 1.4]), "ohm.m", "Flushed Resistivity")
    well.add_curve("SW", np.array([0.12, 0.15, 0.18, 0.22, 0.30, 0.45, 0.55, 0.70, 0.82, 0.92]), "v/v", "Water Saturation")
    well.add_curve("SXO", np.array([0.55, 0.58, 0.60, 0.65, 0.75, 0.80, 0.88, 0.95, 0.98, 1.00]), "v/v", "Flushed Water Saturation")
    well.add_curve("NET_PAY", np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0]), "", "Net Pay Flag")
    well.add_curve("PERM", np.array([200.0, 150.0, 80.0, 30.0, 8.0, 2.0, 0.8, 0.2, 0.05, 0.01]), "mD", "Permeability")
    well.add_curve("DTC", np.array([80.0, 82.0, 84.0, 86.0, 88.0, 95.0, 100.0, 110.0, 125.0, 135.0]), "us/ft", "Compressional Sonic")
    well.add_curve("DTS", np.array([150.0, 152.0, 134.0, 136.0, 140.0, 160.0, 170.0, 185.0, 210.0, 225.0]), "us/ft", "Shear Sonic")
    well.add_curve("PEF", np.array([2.2, 2.3, 2.4, 2.6, 2.8, 4.6, 4.8, 5.0, 3.0, 2.0]), "", "PEF")
    well.add_curve("DRHO", np.array([0.02, 0.03, 0.02, 0.01, 0.00, -0.02, -0.04, -0.06, 0.12, 0.16]), "g/cc", "Density Correction")
    well.add_curve("CALI", np.array([8.5, 8.6, 8.4, 9.4, 7.9, 8.5, 8.4, 9.2, 8.0, 8.6]), "in", "Caliper")
    well.add_curve("BS", np.array([8.5, 8.5, 8.5, 8.5, 8.5, 8.5, 8.5, 8.5, 8.5, 8.5]), "in", "Bit Size")
    well.add_curve("BAD_HOLE", np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0]), "", "Bad Hole Flag")
    well.add_curve("LITHOLOGY_GEOLINK", np.array([1, 1, 1, 1, 10, 9, 9, 15, 7, 21]), "", "Lithology Code")
    return well


def _build_dynamic_workflow_well() -> Well:
    depth = np.arange(9000.0, 9005.0, 0.5)
    well = Well("WORKFLOW-1")
    well.depth = depth
    well.depth_unit = "ft"
    well.add_curve("DEPT", depth, "ft", "Depth")
    well.add_curve("GR", np.array([35, 38, 42, 45, 50, 60, 75, 85, 95, 110]), "API", "Gamma Ray")
    well.add_curve("RHOB", np.array([2.20, 2.22, 2.24, 2.26, 2.30, 2.36, 2.42, 2.48, 2.55, 2.60]), "g/cc", "Bulk Density")
    well.add_curve("NPHI", np.array([0.09, 0.10, 0.11, 0.12, 0.14, 0.18, 0.22, 0.26, 0.28, 0.30]), "v/v", "Neutron Porosity")
    well.add_curve("DTC", np.array([78, 80, 82, 84, 86, 92, 98, 105, 115, 125]), "us/ft", "Compressional Sonic")
    well.add_curve("RDEP", np.array([90, 85, 78, 60, 40, 22, 12, 7, 4, 2]), "ohm.m", "Deep Resistivity")
    well.add_curve("RXO", np.array([45, 42, 40, 34, 28, 18, 9, 5, 3, 1.8]), "ohm.m", "Flushed Resistivity")
    well.add_curve("CALI", np.array([8.5, 8.5, 8.6, 8.5, 8.4, 8.5, 8.6, 8.7, 8.5, 8.5]), "in", "Caliper")
    well.add_curve("BS", np.array([8.5] * 10), "in", "Bit Size")
    well.add_curve("DRHO", np.array([0.01] * 10), "g/cc", "Density Correction")
    return well


def test_curve_value_summary_rounds_statistics_to_four_decimals():
    well = Well("STAT-ROUNDING")
    well.depth = np.arange(5.0)
    well.add_curve(
        "PERM",
        np.array([
            4.440630585851518,
            25.63451407314591,
            28.846711749460937,
            29.674369069951126,
            1.23456789,
        ]),
        "mD",
        "Permeability",
    )

    summary = _curve_value_summary(well, "PERM")

    assert summary["min"] == 1.2346
    assert summary["p50"] == 25.6345
    assert summary["max"] == 29.6744
    assert summary["finite_percent"] == 100


def test_curve_alias_selection_skips_present_but_nonfinite_primary_curve():
    well = Well("ALIAS-FINITE")
    well.depth = np.arange(4.0)
    well.add_curve("RDEP", np.full(4, np.nan), "ohm.m", "Deep Resistivity")
    well.add_curve("RT", np.array([10.0, 11.0, 12.0, 13.0]), "ohm.m", "True Resistivity")

    data, mnemonic = get_curve_data(well, ["RDEP", "RT"])
    raw_data, raw_mnemonic = get_curve_data(well, ["RDEP", "RT"], require_finite=False)

    assert mnemonic == "RT"
    np.testing.assert_allclose(data, [10.0, 11.0, 12.0, 13.0])
    assert raw_mnemonic == "RDEP"
    assert np.isnan(raw_data).all()


def test_curve_alias_selection_combines_partial_aliases_depthwise():
    well = Well("ALIAS-PARTIAL")
    well.depth = np.arange(4.0)
    well.add_curve("RDEP", np.array([np.nan, 11.0, np.nan, 13.0]), "ohm.m", "Deep Resistivity")
    well.add_curve("RT", np.array([10.0, 99.0, 12.0, 99.0]), "ohm.m", "True Resistivity")

    data, mnemonic = get_curve_data(well, ["RDEP", "RT"])

    assert mnemonic == "RDEP+RT"
    np.testing.assert_allclose(data, [10.0, 11.0, 12.0, 13.0])


def _build_core_corrected_priority_well() -> Well:
    well = Well("CORE-PRIORITY")
    depth = np.arange(4.0)
    well.depth = depth
    well.depth_unit = "m"
    well.add_curve("DEPT", depth, "m", "Depth")
    well.add_curve("PHIT", np.full(4, 0.04), "v/v", "Low Total Porosity")
    well.add_curve("PHIE", np.full(4, 0.05), "v/v", "Low Effective Porosity")
    well.add_curve("PHIE_CC", np.full(4, 0.22), "v/v", "Core-Calibrated Effective Porosity")
    well.add_curve("PHID", np.full(4, 0.25), "v/v", "Density Porosity")
    well.add_curve("PHIN", np.full(4, 0.10), "v/v", "Neutron Porosity")
    well.add_curve("PHIS", np.full(4, 0.08), "v/v", "Sonic Porosity")
    well.add_curve("PERM", np.full(4, 0.01), "mD", "Low Empirical Permeability")
    well.add_curve("PERM_CC", np.full(4, 2000.0), "mD", "Core-Calibrated Permeability")
    well.add_curve("VCL", np.full(4, 0.10), "v/v", "Clay Volume")
    well.add_curve("SW", np.full(4, 0.50), "v/v", "Water Saturation")
    well.add_curve("RDEP", np.full(4, 20.0), "ohm.m", "Deep Resistivity")
    well.add_curve("RXO", np.full(4, 30.0), "ohm.m", "Flushed Resistivity")
    well.add_curve("LITHOLOGY_GEOLINK", np.full(4, 1.0), "", "Lithology")
    well.add_curve("GR", np.full(4, 45.0), "API", "Gamma Ray")
    well.add_curve("PEF", np.full(4, 2.2), "", "PEF")
    well.add_curve("RHOB", np.full(4, 2.25), "g/cc", "Bulk Density")
    well.add_curve("NPHI", np.full(4, 0.10), "v/v", "Neutron Porosity")
    well.add_curve("BAD_HOLE", np.zeros(4), "", "Bad Hole")
    well.add_curve("NET_PAY", np.ones(4), "", "Net Pay")
    return well


def test_category_analyses_prioritize_core_corrected_porosity_and_perm():
    well = _build_core_corrected_priority_well()

    perm_result = PermeabilityAnalysis().analyze(well, add_curves=False)
    assert np.all(
        perm_result.sections["permeability_distribution_analysis"].raw_labels
        == "EXCELLENT_PERMEABILITY"
    )
    assert "PERM_CC" in perm_result.sections["permeability_distribution_analysis"].methodology

    poro_result = PorosityAnalysis().analyze(well, add_curves=False)
    assert np.all(
        poro_result.sections["total_porosity_quality_distribution"].raw_labels
        == "EXCELLENT_POROSITY"
    )
    assert "PHIE_CC" in poro_result.sections["total_porosity_quality_distribution"].methodology
    assert "PERM_CC" in poro_result.sections["porosity_quality_assessment_characteristics"].methodology

    hydrocarbon_result = HydrocarbonAnalysis().analyze(well, add_curves=False)
    assert np.all(
        hydrocarbon_result.sections["reservoir_quality_for_hydrocarbons"].raw_labels
        == "EXCELLENT_RESERVOIR"
    )
    assert "PHIE_CC" in hydrocarbon_result.sections["reservoir_quality_for_hydrocarbons"].methodology
    assert "PERM_CC" in hydrocarbon_result.sections["reservoir_quality_for_hydrocarbons"].methodology

    lithology_result = LithologyAnalysis().analyze(well, add_curves=False)
    assert np.all(
        lithology_result.sections["lithology_petrophysics_relationships"].raw_labels
        == "EXCELLENT_SANDSTONE_RESERVOIR"
    )
    assert "PHIE_CC" in lithology_result.sections["lithology_petrophysics_relationships"].methodology
    assert "PERM_CC" in lithology_result.sections["lithology_petrophysics_relationships"].methodology


def test_saturation_and_numerical_analysis_document_core_corrected_sources():
    well = _build_core_corrected_priority_well()

    saturation_result = SaturationAnalysis().analyze(well, add_curves=False)
    bvw = saturation_result.derived_curves["BVW"].data
    np.testing.assert_allclose(bvw, np.full(4, 0.11))
    assert "PHIE_CC" in saturation_result.sections["bulk_volume_water_porosity_integration"].methodology

    numerical_result = NumericalStatisticalAnalysis().analyze(well, add_curves=False)
    methodology = numerical_result.sections["log_correlation_consistency_evaluation"].methodology
    assert "PHIE_CC" in methodology
    assert "PERM_CC" in methodology


def test_numerical_analysis_skips_all_nan_curves_without_runtime_warnings():
    import warnings

    well = _build_analysis_well()
    well.add_curve("ALL_NAN", np.full(well.n_samples, np.nan), "", "Missing data")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        NumericalStatisticalAnalysis().analyze(well, add_curves=False)

    assert not any(issubclass(item.category, RuntimeWarning) for item in caught)


def test_numerical_deviation_threshold_ignores_floating_point_noise():
    values = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    z_score = abs((values[0] - np.mean(values)) / np.std(values))
    assert z_score > 3.0
    assert np.isclose(z_score, 3.0, rtol=0.0, atol=1e-12)

    well = Well("NUMERICAL-THRESHOLD")
    well.depth = np.arange(values.size, dtype=float)
    well.add_curve("VANHY", values, "v/v", "Anhydrite volume")
    well.add_curve("NET_PAY", np.zeros(values.size), "", "Net pay flag")

    result = NumericalStatisticalAnalysis().analyze(well, add_curves=False)
    labels = result.sections["statistical_deviation_detection"].raw_labels

    assert labels[0] == "NORMAL_ZONE"


def test_numerical_deviation_count_ignores_z2_floating_point_noise():
    values = np.array([1e-12, 0.0, 0.0, 0.0, 0.0])
    z_score = abs((values[0] - np.mean(values)) / np.std(values))
    assert z_score > 2.0
    assert np.isclose(z_score, 2.0, rtol=0.0, atol=1e-12)

    well = Well("NUMERICAL-Z2-THRESHOLD")
    well.depth = np.arange(values.size, dtype=float)
    for mnemonic in ("TEST_A", "TEST_B", "TEST_C"):
        well.add_curve(mnemonic, values, "", "Threshold test")
    well.add_curve("NET_PAY", np.zeros(values.size), "", "Net pay flag")

    result = NumericalStatisticalAnalysis().analyze(well, add_curves=False)
    labels = result.sections["statistical_deviation_detection"].raw_labels

    assert labels[0] == "NORMAL_ZONE"


def test_electrofacies_is_reproducible_and_independent_of_row_order():
    well = _build_analysis_well()

    np.random.seed(17)
    expected = NumericalStatisticalAnalysis._electrofacies(well, well.n_samples)
    np.random.seed(938)
    repeated = NumericalStatisticalAnalysis._electrofacies(well, well.n_samples)

    order = np.array([7, 2, 9, 0, 5, 1, 8, 3, 6, 4])
    reordered = Well("ANALYSIS-REORDERED")
    reordered.depth = well.depth[order]
    reordered.depth_unit = well.depth_unit
    for mnemonic in ("RHOB", "NPHI", "GR", "RDEP"):
        curve = well.get(mnemonic)
        reordered.add_curve(
            mnemonic,
            curve.data[order],
            curve.unit,
            curve.metadata.description,
        )
    reordered_labels = NumericalStatisticalAnalysis._electrofacies(
        reordered,
        reordered.n_samples,
    )
    restored_labels = np.empty_like(reordered_labels)
    restored_labels[order] = reordered_labels

    np.testing.assert_array_equal(repeated, expected)
    np.testing.assert_array_equal(restored_labels, expected)


def test_electrofacies_handles_one_unique_log_response_as_one_class():
    well = Well("ANALYSIS-CONSTANT")
    well.depth = np.arange(4.0)
    well.add_curve("GR", np.full(4, 50.0), "API", "Gamma Ray")
    well.add_curve("RHOB", np.full(4, 2.4), "g/cc", "Bulk Density")

    labels = NumericalStatisticalAnalysis._electrofacies(well, well.n_samples)

    np.testing.assert_array_equal(labels, np.zeros(4))


def test_rock_quality_module_prioritizes_core_corrected_inputs():
    from petrophys.evaluation.rock_quality.rock_quality_module import RockQualityModule

    well = _build_core_corrected_priority_well()

    result = RockQualityModule().calculate(well)

    np.testing.assert_allclose(result.rqi, np.full(4, 0.0314 * np.sqrt(2000.0 / 0.22)))
    np.testing.assert_allclose(result.phi_z, np.full(4, 0.22 / (1.0 - 0.22)))


def test_net_pay_module_prioritizes_core_corrected_porosity():
    from petrophys.evaluation.net_pay.net_pay_module import NetPayModule

    well = Well("NET-PAY-CORE")
    well.depth = np.arange(3.0)
    well.add_curve("PHIE", np.full(3, 0.02), "v/v", "Low Effective Porosity")
    well.add_curve("PHIE_CC", np.full(3, 0.20), "v/v", "Core-Calibrated Effective Porosity")
    well.add_curve("VCL", np.full(3, 0.10), "v/v", "Clay Volume")
    well.add_curve("SW", np.full(3, 0.40), "v/v", "Water Saturation")

    result = NetPayModule(phie_cutoff=0.08).calculate(well)

    assert np.all(result.net_reservoir)
    assert np.all(result.net_pay)


def test_hydrocarbon_analysis_adds_requested_curves_and_targets():
    well = _build_analysis_well()

    result = HydrocarbonAnalysis().analyze(well, add_curves=True)
    payload = result.to_dict()["HYDROCARBON_ANALYSIS"]

    assert well.get("CROSSOVER_FLAG") is not None
    assert well.get("VP_VS_RATIO") is not None
    assert well.get("PRIME_GAS_TARGET") is not None
    assert well.get("PRIME_OIL_TARGET") is not None
    assert well.get("NET_PAY") is not None
    np.testing.assert_array_equal(
        well["CROSSOVER_FLAG"].data,
        np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=float),
    )
    assert np.nanmax(well["PRIME_GAS_TARGET"].data) == 1.0
    assert np.nanmax(well["PRIME_OIL_TARGET"].data) == 1.0
    assert "PRIME_GAS_TARGET" in payload["integrated_hydrocarbon_potential"]


def test_input_lithology_is_not_reported_as_missing_litho_fallback():
    well = _build_analysis_well()
    original_curves = set(well.curves)
    workflow_result = SimpleNamespace(
        base_result=SimpleNamespace(steps_executed=[], steps_skipped=[]),
        zone_results=[],
    )

    statuses = _build_reference_derived_log_status(well, workflow_result, original_curves)

    assert "LITHO" not in {item["mnemonic"] for item in statuses}


def test_rmf_missing_reason_does_not_assume_non_obm_mud():
    well = _build_analysis_well()
    reason = _missing_curve_reason(
        "RMF",
        well,
        SimpleNamespace(),
        step_results={},
        skipped_steps=set(),
        computed_curves=set(),
        original_curves=set(well.curves),
    )

    assert "mud type was not OBM" not in reason
    assert "mud type was not provided or was unknown" in reason


def test_hydrocarbon_analysis_rejects_implausible_shear_slowness_for_vpvs():
    well = _build_analysis_well()
    well.add_curve("DTS", np.full(well.n_samples, 4.0e7), "us/ft", "Bad Shear Sonic")

    HydrocarbonAnalysis().analyze(well, add_curves=True)

    assert well.get("VP_VS_RATIO") is None
    assert well.get("PRIME_GAS_TARGET") is None


def test_hydrocarbon_analysis_does_not_use_dte_as_shear_sonic():
    well = _build_analysis_well()
    del well._curves["DTS"]
    well.add_curve("DTE", np.full(well.n_samples, 4.0e7), "us/ft", "Non-shear DTE telemetry")

    HydrocarbonAnalysis().analyze(well, add_curves=True)

    assert well.get("VP_VS_RATIO") is None
    assert well.get("PRIME_GAS_TARGET") is None


def test_input_log_display_list_excludes_core_only_curves():
    curves = {
        "BS",
        "DTC",
        "DT",
        "DTE",
        "CALI",
        "DCAL",
        "GDEN",
        "KH_CORE",
        "KV_CORE",
        "PHIC",
        "SWC",
        "SOC",
        "LITHOLOGY_GEOLINK",
    }

    displayed = _input_log_display_list(curves)

    assert displayed == ["CALI", "DTC", "DTE", "lithology"]


def test_reference_statistics_use_canonical_names_and_skip_raw_lithology_aliases():
    well = Well("STATS-CANONICAL")
    well.depth = np.arange(3.0)
    well.add_curve("DEPT", well.depth, "ft", "Depth")
    well.add_curve("TEMP", np.array([100.123456, 101.0, 102.0]), "degF", "Temperature")
    well.add_curve("TMPC", np.array([37.0, 38.0, 39.0]), "degC", "Legacy Celsius temperature alias")
    well.add_curve("PHI_Z", np.array([0.1, 0.2, 0.3]), "", "Phi z alias")
    well.add_curve("LITHOLOGY_GEOLINK", np.array([1.0, 1.0, 2.0]), "", "Lithology")
    well.add_curve("CALI", np.array([8.5, 8.6, 8.7]), "in", "Caliper")
    well.add_curve("DCAL", np.array([0.1, 0.2, 0.3]), "in", "Caliper delta")
    well.add_curve("DT", np.array([80.0, 81.0, 82.0]), "us/ft", "Sonic alias")
    well.add_curve("DTC", np.array([79.0, 80.0, 81.0]), "us/ft", "Sonic")
    well.add_curve("BAD_HOLE", np.array([0.0, 1.0, 0.0]), "", "Internal bad-hole flag")

    stats = _build_reference_statistics(
        well,
        extra_derived_curves={
            "ND_SEPARATION": DerivedCurve(np.array([0.01, 0.02, 0.03]), "", "ND separation"),
            "COMPOSITE_CLAY": DerivedCurve(np.array([0.0, 0.5, 1.0]), "", "Composite clay"),
        },
    )

    assert "TEMP_overall" in stats
    assert stats["TEMP_overall"]["minimum"] == 100.1235
    assert "PHI_NORM_overall" in stats
    assert "ND_SEPARATION_overall" in stats
    assert "COMPOSITE_CLAY_overall" in stats
    assert "COMPOSITE_CLAY_INDICATOR_overall" in stats
    assert "TMPC_overall" not in stats
    assert "PHI_Z_overall" not in stats
    assert "LITHOLOGY_GEOLINK_overall" not in stats
    assert "DCAL_overall" not in stats
    assert "DT_overall" not in stats
    assert "BAD_HOLE_overall" in stats
    assert "BADHOLEFLAG_overall" not in stats


def test_category_derived_logs_include_full_workflow_outputs_not_raw_inputs():
    well = Well("DERIVED-LIST")
    well.depth = np.arange(3.0)
    well.add_curve("DEPT", well.depth, "ft", "Depth")
    well.add_curve("GR", np.array([40.0, 45.0, 50.0]), "API", "Gamma Ray")
    well.add_curve("VCL", np.array([0.10, 0.15, 0.20]), "v/v", "Clay Volume")
    original_curves = set(well.curves)
    well.add_curve("PHIE", np.array([0.12, 0.14, 0.16]), "v/v", "Effective Porosity")
    well.add_curve("PERM", np.array([10.0, 20.0, 30.0]), "mD", "Permeability")
    well.add_curve("BAD_HOLE", np.array([0.0, 1.0, 0.0]), "", "Internal bad-hole flag")

    workflow_result = SimpleNamespace(
        base_result=SimpleNamespace(
            steps_executed=[
                SimpleNamespace(step="normalization", output_curves=sorted(original_curves)),
                SimpleNamespace(step="vclay", output_curves=["VCL"]),
                SimpleNamespace(step="porosity", output_curves=["PHIE"]),
                SimpleNamespace(step="permeability", output_curves=["PERM"]),
                SimpleNamespace(step="bad_hole", output_curves=["BAD_HOLE"]),
            ],
            steps_skipped=[],
        ),
        zone_results=[],
    )
    category_result = CategoryAnalysisResult(
        category_key="SATURATION_ANALYSIS",
        sections={},
        derived_curves={
            "BVW_EFFICIENCY": DerivedCurve(np.array([0.2, 0.3, 0.4]), "", "BVW efficiency"),
            "GR": DerivedCurve(np.array([1.0, 1.0, 1.0]), "API", "Raw gamma ray duplicate"),
        },
    )

    computed = _category_derived_logs_computed(
        "saturation",
        well,
        workflow_result,
        original_curves,
        category_result,
    )

    assert "PHIE" in computed
    assert "PERM" in computed
    assert "VCL" not in computed
    assert "BVW_EFFICIENCY" in computed
    assert "BAD_HOLE" in computed
    assert "BADHOLEFLAG" not in computed
    assert "GR" not in computed


def test_bad_hole_qc_reports_bs_absent_without_synthetic_bit_size_fallback():
    well = Well("BAD-HOLE-NO-BS")
    well.depth = np.arange(3.0)
    well.depth_unit = "ft"
    well.add_curve("DEPT", well.depth, "ft", "Depth")
    well.add_curve("CALI", np.array([8.5, 10.2, 8.6]), "in", "Caliper")
    well.add_curve("DRHO", np.array([0.01, 0.02, 0.20]), "g/cc", "Density correction")

    workflow_result = SimpleNamespace(
        base_result=SimpleNamespace(
            steps_executed=[
                SimpleNamespace(
                    step="bad_hole",
                    parameters={
                        "bit_size_used": None,
                        "bit_size_source": None,
                    },
                    output_curves=["BAD_HOLE"],
                )
            ],
            steps_skipped=[],
            warnings=[],
            errors=[],
        ),
        zone_results=[],
    )

    qc = _build_reference_bad_hole_qc(well, workflow_result)

    assert qc["log_availability"]["BS"] is False
    assert qc["parameters_used"]["bit_size_used_in"] is None
    assert "bit_size_source" not in qc["parameters_used"]
    assert "bs_curve_present" not in qc["parameters_used"]
    assert "DRHO" in qc["parameters_used"]["washout_formula"]
    assert "skipped" in qc["parameters_used"]["large_hole_formula"]


def test_bad_hole_qc_reports_bs_present_when_bs_curve_exists():
    well = Well("BAD-HOLE-WITH-BS")
    well.depth = np.arange(3.0)
    well.depth_unit = "ft"
    well.add_curve("DEPT", well.depth, "ft", "Depth")
    well.add_curve("CALI", np.array([8.5, 12.8, 8.6]), "in", "Caliper")
    well.add_curve("BS", np.array([8.5, 8.5, 8.5]), "in", "Bit size")

    workflow_result = SimpleNamespace(
        base_result=SimpleNamespace(
            steps_executed=[
                SimpleNamespace(
                    step="bad_hole",
                    parameters={
                        "bit_size_used": 8.5,
                        "bit_size_source": "BS_curve",
                    },
                    output_curves=["BAD_HOLE"],
                )
            ],
            steps_skipped=[],
            warnings=[],
            errors=[],
        ),
        zone_results=[],
    )

    qc = _build_reference_bad_hole_qc(well, workflow_result)

    assert qc["log_availability"]["BS"] is True
    assert "bit_size_source" not in qc["parameters_used"]
    assert "bs_curve_present" not in qc["parameters_used"]


def test_reference_json_omits_derived_log_status_by_default():
    well = Well("PAYLOAD-STATUS")
    well.depth = np.arange(3.0)
    well.add_curve("DEPT", well.depth, "ft", "Depth")
    well.add_curve("TEMP", np.array([104.0, 105.8, 107.6]), "degF", "Temperature")
    workflow_result = SimpleNamespace(
        base_result=SimpleNamespace(steps_executed=[], steps_skipped=[], warnings=[], errors=[]),
        zone_results=[],
    )

    payload = _build_reference_json_payload(
        well,
        workflow_result,
        Path("PAYLOAD-STATUS.las"),
        original_curves={"DEPT"},
    )

    assert "derived_log_status" not in payload["analysis_results"]["data_completeness"]


def test_saturation_analysis_derives_so_from_sw():
    well = Well("SAT-SO")
    well.depth = np.arange(3.0)
    well.add_curve("SW", np.array([0.2, 0.65, np.nan]), "v/v", "Water Saturation")

    result = SaturationAnalysis().analyze(well)

    assert "SO" in result.derived_curves
    np.testing.assert_allclose(
        result.derived_curves["SO"].data,
        np.array([0.8, 0.35, np.nan]),
        equal_nan=True,
    )
    assert result.derived_curves["SO"].unit == "frac"


def test_reference_json_primary_completeness_counts_alias_metadata_and_processed_curves():
    well = Well("PAYLOAD-COMPLETENESS")
    well.depth = np.arange(3.0)
    well.header.bit_size = 17.5
    well.add_curve("DEPT", well.depth, "ft", "Depth")
    well.add_curve("TEMP", np.array([150.0, 151.0, 152.0]), "degF", "Temperature")
    well.add_curve("RMF", np.array([0.005, 0.006, 0.007]), "ohm.m", "Mud filtrate")
    well.add_curve("RW", np.array([0.08, 0.09, 0.10]), "ohm.m", "Formation water")
    workflow_result = SimpleNamespace(
        base_result=SimpleNamespace(steps_executed=[], steps_skipped=[], warnings=[], errors=[]),
        zone_results=[],
    )

    payload = _build_reference_json_payload(
        well,
        workflow_result,
        Path("PAYLOAD-COMPLETENESS.las"),
        original_curves={"DEPT", "RSHAL"},
    )

    missing = set(payload["analysis_results"]["data_completeness"]["missing_primary_logs"])
    assert "BS" in missing
    assert "RSHA" not in missing
    assert "TEMP" not in missing
    assert "RMF" not in missing
    assert "RW" not in missing
    assert "RXO" in missing


def test_reference_json_sets_core_flags_when_core_corrected_curves_used():
    well = Well("PAYLOAD-CORE")
    well.depth = np.arange(3.0)
    well.add_curve("DEPT", well.depth, "ft", "Depth")
    well.add_curve("PHIE", np.array([0.12, 0.14, 0.16]), "v/v", "Effective Porosity")
    well.add_curve("PERM", np.array([10.0, 20.0, 40.0]), "mD", "Permeability")
    well.add_curve("PHIE_CC", np.array([0.13, 0.15, 0.17]), "v/v", "Core-Calibrated Porosity")
    well.add_curve("PERM_CC", np.array([12.0, 24.0, 48.0]), "mD", "Core-Calibrated Permeability")
    workflow_result = SimpleNamespace(
        base_result=SimpleNamespace(steps_executed=[], steps_skipped=[], warnings=[], errors=[]),
        zone_results=[],
    )

    payload = _build_reference_json_payload(
        well,
        workflow_result,
        Path("PAYLOAD-CORE.las"),
        original_curves={"DEPT", "PHIE", "PERM"},
        derived_logs_override=["PHIE_CC", "PERM_CC"],
    )

    results = payload["analysis_results"]
    assert results["core_phi_present"] is True
    assert results["core_k_present"] is True


def test_excel_tops_loader_includes_member_tops(tmp_path):
    tops_file = tmp_path / "tops.xlsx"
    pd.DataFrame(
        {
            "well": ["WELL-1", "WELL-1", "WELL-1", "WELL-1"],
            "formation": [
                "ROGALAND GP. TOP",
                "HERMOD MB. TOP",
                "BALDER FM. TOP",
                "IGNORE UNIT TOP",
            ],
            "md_top": [1000.0, 1100.0, 1200.0, 1300.0],
        }
    ).to_excel(tops_file, index=False)

    tops = _load_excel_formation_segments(tops_file)
    names = [segment["display_name"] for segment in tops["WELL-1"]]

    assert "ROGALAND GP. TOP" in names
    assert "HERMOD MB. TOP" in names
    assert "BALDER FM. TOP" in names
    assert "IGNORE UNIT TOP" not in names


def test_tops_loader_accepts_generic_single_well_csv(tmp_path):
    tops_file = tmp_path / "formation_tops.csv"
    pd.DataFrame(
        {
            "formation": ["Topeka", "Lecompton", "Oread"],
            "top": [3782.0, 3884.0, 3926.0],
            "depth_unit": ["ft", "ft", "ft"],
        }
    ).to_csv(tops_file, index=False)

    tops = _load_excel_formation_segments(tops_file)
    segments = _formation_segments_for_well(
        Path("1046123714.las"),
        Well("Morgan No. 1-17"),
        tops,
    )

    assert [segment["display_name"] for segment in segments] == [
        "TOPEKA",
        "LECOMPTON",
        "OREAD",
    ]
    assert [segment["bottom_depth"] for segment in segments] == [
        3884.0,
        3926.0,
        None,
    ]

    _validate_single_well_tops_scope(tops, [Path("1046123714.las")])
    with pytest.raises(ValueError, match="no 'well' column"):
        _validate_single_well_tops_scope(
            tops,
            [Path("1046123714.las"), Path("another-well.las")],
        )


def test_missing_curve_reason_names_category_input_requirements():
    well = _build_analysis_well()
    del well._curves["PEF"]

    reason = _missing_curve_reason(
        "PEF_CLAY_INDICATOR",
        well,
        SimpleNamespace(),
        step_results={},
        skipped_steps=set(),
        computed_curves=set(),
        original_curves=set(well.curves),
    )

    assert reason == "required input curves unavailable: PEF"


def test_missing_lithology_group_uses_readable_group_name():
    well = Well("MISSING-LITHO")
    well.depth = np.array([1000.0, 1001.0, 1002.0, 1003.0])
    well.add_curve("DEPT", well.depth, "ft", "Depth")
    well.add_curve("LITHOLOGY_GEOLINK", np.array([np.nan, np.nan, np.nan, np.nan]), "", "Lithology")

    segments = _build_lithology_group_segments(well, minimum_group_thickness=0.0)

    assert len(segments) == 1
    assert segments[0]["lithology"] == "NO_LITHOLOGY_DATA"
    assert segments[0]["group_name"] == "NO_LITHOLOGY_DATA_GROUP_1"


def test_computed_lithology_is_used_when_geolink_lithology_is_absent():
    well = Well("COMPUTED-LITHO")
    well.depth = np.arange(1000.0, 1008.0)
    well.add_curve("DEPT", well.depth, "ft", "Depth")
    well.add_curve(
        "LITHO",
        np.array([1.0, 1.0, 2.0, 2.0, 8.0, 8.0, 0.0, 0.0]),
        "",
        "Computed lithology",
    )

    segments = _build_lithology_group_segments(well, minimum_group_thickness=0.0)

    assert [segment["lithology"] for segment in segments] == [
        "SANDSTONE",
        "LIMESTONE",
        "MIXED",
        "UNKNOWN",
    ]


def test_geolink_lithology_remains_authoritative_when_computed_lithology_exists():
    well = Well("GEOLINK-LITHO")
    well.depth = np.arange(1000.0, 1006.0)
    well.add_curve("DEPT", well.depth, "ft", "Depth")
    well.add_curve(
        "LITHOLOGY_GEOLINK",
        np.array([1.0, 1.0, np.nan, np.nan, 15.0, 15.0]),
        "",
        "GeoLink lithology",
    )
    well.add_curve("LITHO", np.full(well.n_samples, 2.0), "", "Computed lithology")

    segments = _build_lithology_group_segments(well, minimum_group_thickness=0.0)

    assert [segment["lithology"] for segment in segments] == [
        "SAND",
        "NO_LITHOLOGY_DATA",
        "DOLOMITE",
    ]


def test_export_retains_computed_lithology_only_for_fallback_grouping():
    fallback_well = Well("COMPUTED-LITHO")
    fallback_well.depth = np.arange(1000.0, 1002.0)
    fallback_well.add_curve("DEPT", fallback_well.depth, "ft", "Depth")
    fallback_well.add_curve("LITHO", np.array([1.0, 2.0]), "", "Computed lithology")

    _finalize_export_curves(
        fallback_well,
        {"DEPT"},
        retain_computed_lithology=True,
    )

    assert fallback_well.get("LITHO") is not None

    geolink_well = Well("GEOLINK-LITHO")
    geolink_well.depth = np.arange(1000.0, 1002.0)
    geolink_well.add_curve("DEPT", geolink_well.depth, "ft", "Depth")
    geolink_well.add_curve("LITHOLOGY_GEOLINK", np.array([1.0, 15.0]), "", "GeoLink lithology")
    geolink_well.add_curve("LITHO", np.array([2.0, 2.0]), "", "Computed lithology")

    _finalize_export_curves(
        geolink_well,
        {"DEPT", "LITHOLOGY_GEOLINK"},
        retain_computed_lithology=True,
    )

    assert geolink_well.get("LITHO") is None


def test_vpvs_status_skips_when_dts_is_unavailable():
    well = _build_analysis_well()
    del well._curves["DTS"]
    well.add_curve("DTE", np.full(well.n_samples, 250.0), "us/ft", "Non-shear DTE telemetry")
    workflow_result = SimpleNamespace(
        base_result=SimpleNamespace(steps_executed=[], steps_skipped=[]),
        zone_results=[],
    )

    statuses = _build_reference_derived_log_status(
        well,
        workflow_result,
        original_curves=set(well.curves),
        extra_targets=["VP_VS_RATIO"],
    )
    vpvs = next(item for item in statuses if item["mnemonic"] == "VP_VS_RATIO")

    assert vpvs["status"] == "missing"
    assert vpvs["reason"] == "DTS had no finite samples in this segment"


def test_sw_missing_note_lists_blocking_inputs():
    well = _build_analysis_well()
    for curve in ["PHIE", "SW"]:
        del well._curves[curve]
    well.add_curve("RW", np.full(well.n_samples, 0.03), "ohm.m", "Formation water resistivity")
    step = SimpleNamespace(
        step="saturation",
        method="archie",
        parameters={
            "sw_model_requested": "auto",
            "a": 1.0,
            "m": 2.0,
            "n": 2.0,
        },
    )

    reason = _missing_curve_reason(
        "SW",
        well,
        SimpleNamespace(),
        step_results={"saturation": step},
        skipped_steps=set(),
        computed_curves=set(),
        original_curves=set(well.curves),
    )
    note = _compact_curve_note("SW", "missing", reason, step, well)

    assert "selected effective porosity" in reason
    assert "RDEP/RT=present" in note
    assert "PHIE=missing in this segment" in note
    assert "RW=present" in note


def test_missing_invasion_note_distinguishes_present_but_nonfinite_shallow_curve():
    well = _build_analysis_well()
    del well._curves["RXO"]
    well.add_curve("RSHA", np.full(well.n_samples, np.nan), "ohm.m", "Empty shallow resistivity")

    reason = _missing_curve_reason(
        "INVASION_RATIO",
        well,
        SimpleNamespace(),
        step_results={},
        skipped_steps=set(),
        computed_curves=set(),
        original_curves=set(well.curves),
    )

    assert reason == "shallow resistivity curve was present but had no finite samples; requires finite RSHA, RESS, or RXO"


def test_hydrocarbon_crossover_requires_clean_reservoir_quality_and_good_hole():
    well = _build_analysis_well()

    HydrocarbonAnalysis().analyze(well, add_curves=True)

    expected = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=float)
    np.testing.assert_array_equal(well["CROSSOVER_FLAG"].data, expected)


def test_hydrocarbon_crossover_is_nan_without_bad_hole_curve():
    well = _build_analysis_well()
    del well._curves["BAD_HOLE"]

    HydrocarbonAnalysis().analyze(well, add_curves=True)

    assert well.get("CROSSOVER_FLAG") is None


def test_rock_quality_leaves_invalid_phi_and_perm_as_nan():
    from petrophys.evaluation.rock_quality import calculate_rock_quality

    result = calculate_rock_quality(
        perm_k=np.array([100.0, 0.0, 10.0, np.nan]),
        phi_e=np.array([0.20, 0.20, 0.0, 0.15]),
    )

    assert np.isfinite(result["FZI"][0])
    assert np.isnan(result["RQI"][1])
    assert np.isnan(result["phi_z"][2])
    assert np.isnan(result["FZI"][2])
    assert np.isnan(result["FZI"][3])


def test_invasion_ratio_rejects_tiny_rt_spikes():
    from petrophys.interpretation.electrical.invasion import InvasionModule, calculate_invasion_ratio

    ratio = calculate_invasion_ratio(
        rxo=np.array([1.0, 1.0, 50.0, 1.0, 500.0]),
        rt=np.array([1.0, 0.1, 0.1, 0.0, 1.0]),
    )

    assert ratio[0] == 1.0
    assert ratio[1] == 10.0
    assert np.isnan(ratio[2])
    assert np.isnan(ratio[3])
    assert np.isnan(ratio[4])

    well = Well("INVASION-ALIAS-FINITE")
    well.depth = np.arange(4.0)
    well.add_curve("RSHA", np.full(4, np.nan), "ohm.m", "Empty shallow resistivity")
    well.add_curve("RXO", np.array([2.0, 4.0, 6.0, 8.0]), "ohm.m", "Finite flushed resistivity")
    well.add_curve("RDEP", np.full(4, 2.0), "ohm.m", "Deep resistivity")

    result = InvasionModule().analyze(well)

    assert result.statistics["rxo_curve"] == "RXO"
    np.testing.assert_allclose(result.ratio, [1.0, 2.0, 3.0, 4.0])


def test_multi_mineral_fraction_constraint_enforces_available_volume():
    from petrophys.interpretation.lithology.multi_mineral_solver import _constrain_mineral_fractions

    constrained = _constrain_mineral_fractions(np.array([0.5, 0.4, 0.2]), 0.65)

    assert np.all(constrained >= 0.0)
    assert np.sum(constrained) <= 0.65 + 1e-12


def test_multi_mineral_solver_broadcasts_scalar_inputs_across_rows():
    from petrophys.interpretation.lithology.multi_mineral_solver import MultiMineralSolver

    result = MultiMineralSolver().solve(
        rhob=np.array([2.30, 2.40, 2.50]),
        nphi=np.array([0.20, 0.22, 0.24]),
        vcl=0.10,
        phie=0.15,
        pef=3.0,
    )

    assert result["quartz"].shape == (3,)
    assert result["vcl"].shape == (3,)
    np.testing.assert_allclose(result["vcl"], np.array([0.10, 0.10, 0.10]))
    assert np.all(np.isfinite(result["residual"]))


def test_multi_mineral_solver_uses_pef_dtc_when_neutron_missing():
    from petrophys.interpretation.lithology.multi_mineral_solver import MultiMineralSolver

    result = MultiMineralSolver().solve(
        rhob=np.array([2.45, 2.58, 2.70]),
        nphi=np.array([np.nan, np.nan, np.nan]),
        vcl=np.array([0.20, 0.25, 0.15]),
        phie=np.array([0.12, 0.08, 0.05]),
        pef=np.array([2.4, 3.0, 4.2]),
        dt=np.array([78.0, 72.0, 65.0]),
    )

    solved = np.asarray(result["solved"], dtype=bool)
    assert np.all(solved)
    assert np.all(np.isfinite(result["quartz"]))
    assert np.all(np.isfinite(result["calcite"]))
    assert np.all(np.isfinite(result["dolomite"]))


def test_multi_mineral_add_to_well_does_not_require_neutron_when_pef_dtc_present():
    from petrophys.interpretation.lithology.multi_mineral_solver import MultiMineralSolver

    depth = np.array([1000.0, 1000.5, 1001.0])
    well = Well("NO-NEUTRON-MINERALS")
    well.depth = depth
    well.add_curve("DEPT", depth, "ft", "Depth")
    well.add_curve("RHOB", np.array([2.45, 2.58, 2.70]), "g/cc", "Bulk Density")
    well.add_curve("VCL", np.array([0.20, 0.25, 0.15]), "v/v", "Clay Volume")
    well.add_curve("PHIE", np.array([0.12, 0.08, 0.05]), "v/v", "Effective Porosity")
    well.add_curve("PEF", np.array([2.4, 3.0, 4.2]), "b/e", "Photoelectric Factor")
    well.add_curve("DTC", np.array([78.0, 72.0, 65.0]), "us/ft", "Compressional Slowness")

    MultiMineralSolver().add_to_well(well)

    for curve in ("VQTZ", "VCAL", "VDOL", "VANH"):
        assert well.get(curve) is not None
        assert np.all(np.isfinite(well.get(curve).data))
    assert well.get("MMQC") is not None
    assert well.get("VRESI") is None


def test_multi_mineral_add_to_well_requires_some_discriminator_curve():
    from petrophys.interpretation.lithology.multi_mineral_solver import MultiMineralSolver

    depth = np.array([1000.0, 1000.5])
    well = Well("NO-MINERAL-DISCRIMINATOR")
    well.depth = depth
    well.add_curve("RHOB", np.array([2.45, 2.58]), "g/cc", "Bulk Density")
    well.add_curve("VCL", np.array([0.20, 0.25]), "v/v", "Clay Volume")
    well.add_curve("PHIE", np.array([0.12, 0.08]), "v/v", "Effective Porosity")

    with pytest.raises(ValueError, match="NPHI/PHIN, PEF, or DTC/DT"):
        MultiMineralSolver().add_to_well(well)


def test_multi_mineral_solver_rejects_incompatible_lengths_cleanly():
    from petrophys.interpretation.lithology.multi_mineral_solver import MultiMineralSolver

    with pytest.raises(ValueError, match="nphi length"):
        MultiMineralSolver().solve(
            rhob=np.array([2.30, 2.40, 2.50]),
            nphi=np.array([0.20, 0.22]),
            vcl=0.10,
            phie=0.15,
        )


def test_run_comprehensive_analysis_returns_all_categories_and_curves():
    well = _build_analysis_well()

    result = run_comprehensive_analysis(well, add_curves=True, store_results=True)
    payload = result.to_dict()

    expected_keys = {
        "HYDROCARBON_ANALYSIS",
        "COMPREHENSIVE_LITHOLOGY_ANALYSIS",
        "POROSITY_AND_PORE_SYSTEMS",
        "PERMEABILITY_AND_FLOW_UNITS",
        "SATURATION_AND_FLUID_DISTRIBUTION",
        "NUMERICAL_AND_STATISTICAL_ANALYSIS",
        "WELLBORE_QUALITY_CONTROL",
    }
    assert expected_keys.issubset(payload.keys())
    assert well.get("PHIT_GAS_CORR") is not None
    assert well.get("BAD_HOLE") is not None
    assert well.get("BADHOLEFLAG") is None
    assert well.get("ACOUSTIC_IMPEDANCE") is not None
    assert well.get_result("HYDROCARBON_ANALYSIS") is not None


def test_numerical_elastic_properties_use_us_per_ft_conversion_and_require_dts():
    depth = np.array([1000.0, 1001.0])
    well = Well("ELASTIC-1")
    well.depth = depth
    well.depth_unit = "ft"
    well.add_curve("RHOB", np.array([2.30, 2.30]), "g/cc", "Bulk Density")
    well.add_curve("DTC", np.array([100.0, 100.0]), "us/ft", "Compressional Sonic")
    well.add_curve("DTS", np.array([200.0, 200.0]), "us/ft", "Shear Sonic")

    NumericalStatisticalAnalysis().analyze(well, add_curves=True)

    assert np.isclose(well.get("ACOUSTIC_IMPEDANCE").data[0], 7010400.0)
    assert np.isclose(well.get("POISSON_RATIO").data[0], 1.0 / 3.0)
    assert np.isclose(well.get("YOUNG_MOD").data[0], 14.2451328, rtol=1e-6)

    no_dts = Well("ELASTIC-NO-DTS")
    no_dts.depth = depth
    no_dts.depth_unit = "ft"
    no_dts.add_curve("RHOB", np.array([2.30, 2.30]), "g/cc", "Bulk Density")
    no_dts.add_curve("DTC", np.array([100.0, 100.0]), "us/ft", "Compressional Sonic")

    NumericalStatisticalAnalysis().analyze(no_dts, add_curves=True)

    assert no_dts.get("ACOUSTIC_IMPEDANCE") is not None
    assert no_dts.get("POISSON_RATIO") is None
    assert no_dts.get("YOUNG_MOD") is None
    assert no_dts.get("BRITTLENESS_INDEX") is None


def test_dynamic_workflow_computes_bvw_mhi_and_net_pay_from_canonical_saturation_curves():
    well = _build_dynamic_workflow_well()
    config = DynamicWorkflowConfig(
        bht_f=220.0,
        td_ft=9005.0,
        rw=0.05,
        rmf_at_meas_temp=0.2,
        matrix_type="sandstone",
        lithology="clastic",
        sw_model="archie",
        perm_method="timur",
    )

    result = DynamicWorkflow(config).run(well)
    executed = {step.step for step in result.steps_executed if getattr(step, "success", False)}

    assert {"bvw", "mhi", "net_pay"}.issubset(executed)
    assert well.get("SW") is not None
    assert well.get("SXO") is not None
    assert well.get("SWA") is None
    assert well.get("SWXOA") is None
    assert well.get("BVW") is not None
    assert well.get("MHI") is not None
    assert well.get("NET_PAY") is not None


def test_saturation_auto_routes_by_vcl_into_one_mixed_sw_curve():
    from petrophys.interpretation.saturation.archie import archie
    from petrophys.interpretation.saturation.simandoux import simandoux
    from petrophys.interpretation.saturation.sw_module import SaturationModule

    depth = np.array([1000.0, 1001.0, 1002.0, 1003.0])
    rt = np.array([80.0, 60.0, 12.0, 8.0])
    phie = np.array([0.22, 0.20, 0.18, 0.16])
    vcl = np.array([0.05, 0.15, 0.45, 0.65])

    well = Well("SAT-MIXED-1")
    well.depth = depth
    well.depth_unit = "m"
    well.add_curve("DEPT", depth, "m", "Depth")
    well.add_curve("RDEP", rt, "ohm.m", "Deep Resistivity")
    well.add_curve("PHIE", phie, "v/v", "Effective Porosity")
    well.add_curve("VCL", vcl, "v/v", "Clay Volume")

    result = SaturationModule(a=1.0, m=2.0, n=2.0, model="auto", vcl_cutoff=0.3).calculate(
        well,
        rw=0.05,
        rcl=5.0,
        salinity=35000.0,
        lithology="clastic",
    )

    expected = np.empty_like(rt)
    expected[:2] = archie(rt[:2], 0.05, phie[:2], 1.0, 2.0, 2.0)
    expected[2:] = simandoux(rt[2:], 0.05, phie[2:], vcl[2:], 5.0, 1.0, 2.0, 2.0)

    assert result.sw.shape == depth.shape
    assert result.model == "archie+simandoux"
    assert result.model_counts == {"archie": 2, "simandoux": 2}
    assert np.allclose(result.sw, expected)


def test_temperature_profile_uses_bht_or_fallback_gradient():
    from petrophys.interpretation.temperature import calculate_temperature_profile

    well = Well("TEMP-1")
    depth = np.array([0.0, 500.0, 1000.0])
    well.depth = depth
    well.depth_unit = "m"
    well.set_parameter("FMT", 80.0)
    well.set_parameter("FMT__UNIT", "C")
    well.set_parameter("FMTD", 1000.0)
    well.set_parameter("FMTD__UNIT", "m")
    well.add_curve("DEPT", depth, "m", "Depth")

    fallback = calculate_temperature_profile(
        well,
        surface_temp=20.0,
        surface_temp_unit="C",
    )
    assert fallback.source == "fallback_geothermal_gradient"
    assert np.allclose(fallback.temperature, np.array([68.0, 95.0, 122.0]))

    bht_c_well = Well("TEMP-BHT-C")
    bht_c_well.depth = depth
    bht_c_well.depth_unit = "ft"
    bht_c_well.header.bht = 100.0
    bht_c_well.header.td = 1000.0
    bht_c_well.set_parameter("BHT__UNIT", "C")
    bht_c_well.add_curve("DEPT", depth, "ft", "Depth")

    bht_c = calculate_temperature_profile(
        bht_c_well,
        surface_temp=68.0,
        surface_temp_unit="F",
    )
    assert bht_c.source == "linear_gradient_from_BHT"
    assert np.allclose(bht_c.temperature, np.array([68.0, 140.0, 212.0]))


def test_dynamic_workflow_fallback_temperature_and_corrected_rm_curve():
    well = _build_dynamic_workflow_well()
    well.set_parameter("FMT", 80.0)
    well.set_parameter("FMT__UNIT", "C")
    well.set_parameter("FMTD", float(well.depth[-1]))
    well.set_parameter("FMTD__UNIT", "ft")
    config = DynamicWorkflowConfig(
        rm=0.6,
        rm_meas_temp_f=75.0,
        rw=0.05,
        matrix_type="sandstone",
        lithology="clastic",
    )

    result = DynamicWorkflow(config).run(well)
    temp_step = next(step for step in result.steps_executed if step.step == "temperature")

    assert temp_step.method == "fallback_geothermal_gradient"
    assert well.get("TEMP") is not None
    assert well.get("RM") is not None
    assert well.get("RMF") is not None
    assert np.nanmax(well["RM"].data) < 0.6
    assert np.allclose(well["RMF"].data, well["RM"].data * 0.75)


def test_dynamic_workflow_converts_celsius_bht_metadata_for_temperature_profile():
    well = _build_dynamic_workflow_well()
    well.header.bht = 100.0
    well.header.td = float(well.depth[-1])
    well.set_parameter("BHT__UNIT", "C")

    config = DynamicWorkflowConfig(
        rm=0.6,
        rw=0.05,
        matrix_type="sandstone",
        lithology="clastic",
    )

    result = DynamicWorkflow(config).run(well)
    temp_step = next(step for step in result.steps_executed if step.step == "temperature")
    rmf_step = next(step for step in result.steps_executed if step.step == "rmf")

    assert temp_step.method == "linear_gradient_from_BHT"
    assert np.isclose(temp_step.parameters["reference_temperature_f"], 212.0)
    assert np.isclose(well["TEMP"].data[-1], 212.0)
    assert np.isclose(rmf_step.parameters["formation_temp_max_f"], 212.0)


def test_archie_coefficients_cover_lithology_nomenclature():
    from petrophys.evaluation.diagnostics.lithology_analysis import GROUND_TRUTH_GROUPS, GROUND_TRUTH_NAMES
    from petrophys.model.constants import ArchieParameterDefaults, DefaultParameters
    from petrophys.interpretation.electrical.cementation import default_archie_params

    labels = set(GROUND_TRUTH_GROUPS.keys()) | set(GROUND_TRUTH_NAMES.values())
    labels.update({
        "SAND",
        "SHALY_SAND",
        "CALCAREOUS_SANDSTONE",
        "SANDY_LIMESTONE",
        "Cross Bedded Sst",
        "Dolomite2",
        "Kaīnite",
        "Aeolian Sandstone",
        "Calcareous Debris Flow",
        "Plutonic Rock",
        "Volcanic Tuff",
        "Porous Limestone",
    })

    for label in labels:
        coeffs = ArchieParameterDefaults.get(label)
        assert coeffs.a > 0
        assert coeffs.m > 0
        assert coeffs.n > 0

    sand = ArchieParameterDefaults.get("SAND")
    assert sand.lithology == "sandstone"
    assert (sand.a, sand.m, sand.n) == (0.81, 2.0, 2.0)
    assert ArchieParameterDefaults.get("Kaīnite").lithology == "evaporite"
    assert ArchieParameterDefaults.get("Dolomite2").lithology == "dolomite"
    assert DefaultParameters.ARCHIE_BY_LITHOLOGY["limestone"] == {"a": 1.0, "m": 2.0, "n": 2.0}

    legacy = default_archie_params("unconsolidated")
    assert (legacy.a, legacy.m, legacy.n) == (0.62, 2.15, 2.0)
