import numpy as np
import pytest

from petrophys import Well
from petrophys.exceptions import DataQualityWarning
from petrophys.preprocessing.standardization import standardize_curves
from petrophys.preprocessing.standardization.alias_map import get_standard_mnemonic
from petrophys.workflow import DynamicWorkflowConfig, run_dynamic_workflow
import petrophys.workflow.zoned_batch as zoned_batch


def _alias_override(base_name, value, unit="", profile="current"):
    overrides = {}
    injected = {}
    unmapped = []
    zoned_batch._apply_parameter_alias(
        base_name, value, unit, overrides, injected, unmapped,
        calculation_profile=profile,
    )
    return overrides, injected, unmapped


def test_parameter_aliases_convert_temperature_depth_salinity_and_booleans():
    cases = [
        ("CLFL", 90.0, "g/L", "rmf_chlorides_mg_l", 90000.0),
        ("CLFL", 30000.0, "mg/l", "rmf_chlorides_mg_l", 30000.0),
        ("MFST", 20.0, "degC", "rmf_meas_temp_f", 68.0),
        ("MST", 60.0, "degF", "rm_meas_temp_f", 60.0),
        ("SAL", 35.0, "g/l", "salinity_ppm", 35000.0),
        ("WATSAL", 35000.0, "mg/L", "salinity_ppm", 35000.0),
        ("WSAL", 3.5, "%", "salinity_ppm", 35000.0),
        ("RWT", 25.0, "C", "rw_meas_temp_f", 77.0),
        ("RMFT", 25.0, "C", "rmf_meas_temp_f", 77.0),
        ("BHT", 100.0, "C", "bht_f", 212.0),
        ("TD", 3000.0, "m", "td_ft", 9842.52),
    ]

    for base_name, value, unit, expected_key, expected_value in cases:
        overrides, injected, unmapped = _alias_override(base_name, value, unit)
        assert injected == {}
        assert unmapped == []
        assert expected_key in overrides
        assert np.isclose(overrides[expected_key], expected_value)

    overrides, _, unmapped = _alias_override("APPLY_HC_CORRECTION", "yes")
    assert overrides["apply_hc_correction"] is True
    assert unmapped == []

    overrides, _, unmapped = _alias_override("LITHOLOGY", "carbonate")
    assert overrides["lithology_domain"] == "carbonate"
    overrides, _, unmapped = _alias_override("PP_DOMAIN", "carbonate")
    assert overrides["lithology_domain"] == "carbonate"
    assert unmapped == []


def test_benchmark_profile_reads_clfl_as_salinity_and_skips_mud_sample_temperatures():
    for unit in ("kppm", "g/L"):
        overrides, _, unmapped = _alias_override("CLFL", 90.0, unit, "benchmark")
        assert overrides == {"rmf_salinity_ppm": 90000.0}
        assert unmapped == []
    for name in ("MFST", "MST"):
        overrides, _, unmapped = _alias_override(name, 20.0, "degC", "benchmark")
        assert overrides == {} and unmapped == [name]


def test_unknown_parameter_aliases_are_reported_as_unmapped():
    overrides, injected, unmapped = _alias_override("NOT_A_WORKFLOW_PARAMETER", 123.0)

    assert overrides == {}
    assert injected == {}
    assert unmapped == ["NOT_A_WORKFLOW_PARAMETER"]


def test_zoned_parameter_extraction_converts_depths_and_alias_units():
    well = Well("ZONED-ALIASES")
    well.depth = np.array([0.0, 500.0, 1000.0, 1500.0])
    well.depth_unit = "ft"
    well.set_parameter("ZTOP_Z1", 100.0)
    well.set_parameter("ZTOP_Z1__UNIT", "m")
    well.set_parameter("ZBOT_Z1", 300.0)
    well.set_parameter("ZBOT_Z1__UNIT", "m")
    well.set_parameter("CLFL_Z1", 80.0)
    well.set_parameter("CLFL_Z1__UNIT", "kppm")
    well.set_parameter("SAL_Z1", 35.0)
    well.set_parameter("SAL_Z1__UNIT", "g/l")
    well.set_parameter("BHT_Z1", 100.0)
    well.set_parameter("BHT_Z1__UNIT", "C")
    well.set_parameter("APPLY_HC_CORRECTION_Z1", "true")

    zones = zoned_batch.extract_zoned_parameters(well)

    assert len(zones) == 1
    zone = zones[0]
    assert np.isclose(zone.top, 328.084)
    assert np.isclose(zone.bottom, 984.252)
    # 80 kppm of chloride is 131,882 ppm NaCl; its brine (McCain 1991) weighs
    # (62.368 + 0.438603*13.1882 + 1.60074e-3*13.1882**2) / 62.428 = 1.09616 g/cm3.
    assert np.isclose(zone.config_overrides["rmf_chlorides_mg_l"], 80000.0 * 1.09616, rtol=1e-5)
    assert "rmf_salinity_ppm" not in zone.config_overrides
    benchmark = zoned_batch.extract_zoned_parameters(well, calculation_profile="benchmark")
    assert benchmark[0].config_overrides["rmf_salinity_ppm"] == 80000.0
    assert zone.config_overrides["salinity_ppm"] == 35000.0
    assert zone.config_overrides["bht_f"] == 212.0
    assert zone.config_overrides["apply_hc_correction"] is True


def test_curve_standardization_renames_aliases_and_corrects_units():
    well = Well("STANDARDIZE-ALIASES")
    well.depth = np.array([0.0, 1.0])
    well.add_curve("ILD", np.array([10.0, 20.0]), "ohm.ft", "Deep resistivity")
    well.add_curve("ZDEN", np.array([2200.0, 2400.0]), "kg/m3", "Density")
    well.add_curve("TNPH", np.array([20.0, 35.0]), "pu", "Neutron")
    well.add_curve("CAL", np.array([216.0, 254.0]), "mm", "Caliper")
    well.add_curve("RSHAL", np.array([1.5, 2.5]), "ohm.m", "Shallow resistivity")
    well.add_curve(
        "PHIE_CORECAL", np.array([0.12, 0.18]), "v/v", "Legacy core-calibrated porosity"
    )
    well.add_curve("PHIE_D", np.array([0.10, 0.20]), "m3/m3", "Effective porosity")
    well.add_curve("VSH", np.array([0.30, 0.40]), "m3/m3", "Shale volume")

    standardized = standardize_curves(well, rename_curves=True, convert_units=True)

    assert standardized.get("RDEP") is not None
    assert standardized.get("RHOB") is not None
    assert standardized.get("NPHI") is not None
    assert standardized.get("CALI") is not None
    assert standardized.get("RSHA") is not None
    assert standardized.get("PHIE_CC") is not None
    assert standardized.get("PHIE") is not None
    assert standardized.get("VCL") is not None
    assert standardized.get("PHIE_CORECAL") is None
    np.testing.assert_allclose(standardized["RDEP"].data, np.array([3.048, 6.096]))
    np.testing.assert_allclose(standardized["RHOB"].data, np.array([2.2, 2.4]))
    np.testing.assert_allclose(standardized["NPHI"].data, np.array([0.20, 0.35]))
    np.testing.assert_allclose(
        standardized["CALI"].data, np.array([8.5039416, 10.0000054])
    )
    np.testing.assert_allclose(standardized["RSHA"].data, np.array([1.5, 2.5]))
    np.testing.assert_allclose(standardized["PHIE_CC"].data, np.array([0.12, 0.18]))
    np.testing.assert_allclose(standardized["PHIE"].data, np.array([0.10, 0.20]))
    np.testing.assert_allclose(standardized["VCL"].data, np.array([0.30, 0.40]))
    assert standardized["PHIE"].unit == "v/v"
    assert standardized["VCL"].unit == "v/v"


def test_legacy_compensated_neutron_and_density_correction_aliases():
    well = Well("LEGACY-CDL-CNL")
    well.depth = np.array([4000.0, 4000.5, 4001.0])
    well.add_curve("CNLS", np.array([12.0, 18.0, 24.0]), "", "CN Limestone Porosity")
    well.add_curve("RHOC", np.array([0.02, 0.04, 0.06]), "", "CDL Density Correction")

    standardized = standardize_curves(well, rename_curves=True, convert_units=True)

    np.testing.assert_allclose(standardized["NPHI"].data, [0.12, 0.18, 0.24])
    np.testing.assert_allclose(standardized["DRHO"].data, [0.02, 0.04, 0.06])
    assert "CNLS" not in standardized
    assert "RHOC" not in standardized


def test_lasio_duplicate_suffixes_standardize_by_base_mnemonic():
    assert get_standard_mnemonic("GR:1") == "GR"
    assert get_standard_mnemonic("TNPH:2") == "NPHI"

    well = Well("DUPLICATE-MNEMONICS")
    well.depth = np.array([1000.0, 1000.5])
    well.add_curve("GR:1", [40.0, 50.0], "gAPI")
    well.add_curve("TNPH:1", [0.20, 0.25], "m3/m3")

    standardized = standardize_curves(well, rename_curves=True)

    assert "GR" in standardized
    assert "NPHI" in standardized


def test_dynamic_workflow_adds_working_aliases_without_losing_vendor_curves():
    well = Well("VENDOR-ALIASES")
    well.depth = np.arange(1000.0, 1003.0)
    well.add_curve("GR:1", [40.0, 60.0, 80.0], "gAPI")
    well.add_curve("TNPH:1", [0.18, 0.20, 0.22], "m3/m3")
    well.add_curve("RHO8", [2.35, 2.30, 2.25], "g/cm3")
    well.add_curve("PEFZ", [2.0, 2.2, 2.4], "")

    result = run_dynamic_workflow(
        well,
        DynamicWorkflowConfig(rw=0.05, standardize_input=True),
    )

    assert result.success
    assert "GR:1" in well
    assert "TNPH:1" in well
    assert "RHO8" in well
    assert "GR" in well
    assert "NPHI" in well
    assert "RHOB" in well
    assert "PHIT" in well


def test_low_quality_dlis_alias_is_not_promoted_to_canonical_curve():
    well = Well("UNCALIBRATED")
    well.depth = np.array([1000.0, 1001.0])
    curve = well.add_curve(
        "RVDRU",
        [10.0, 20.0],
        "ohm.m",
        "Deep Resistivity Uncalibrated",
    )
    curve.metadata.custom["standardization_priority"] = -200

    standardized = standardize_curves(well, rename_curves=True)

    assert "RVDRU" in standardized
    assert "RDEP" not in standardized


def test_uncalibrated_resistivity_mnemonics_are_not_canonical_aliases():
    well = Well("UNCALIBRATED-LAS")
    well.depth = np.array([1000.0, 1001.0])
    well.add_curve(
        "RVDRU",
        [10.0, 20.0],
        "ohm.m",
        "Deep Resistivity Uncalibrated",
    )
    well.add_curve(
        "RVSRU",
        [5.0, 8.0],
        "ohm.m",
        "Shallow Resistivity Uncalibrated",
    )

    standardized = standardize_curves(well, rename_curves=True)

    assert get_standard_mnemonic("RVDRU") == "RVDRU"
    assert get_standard_mnemonic("RVSRU") == "RVSRU"
    assert "RVDRU" in standardized
    assert "RVSRU" in standardized
    assert "RDEP" not in standardized
    assert "RSHA" not in standardized


def test_dte_is_not_treated_as_shear_sonic():
    well = Well("STANDARDIZE-DTE")
    well.depth = np.array([0.0, 1.0, 2.0])
    well.add_curve("DTE", np.array([1.5e7, 2.0e7, 4.0e7]), "us/ft", "DTE telemetry")

    standardized = standardize_curves(well, rename_curves=True, convert_units=True)

    assert standardized.get("DTS") is None
    assert standardized.get("DTE") is not None
    assert np.all(np.isnan(standardized["DTE"].data))


def test_dcal_is_not_treated_as_absolute_caliper():
    well = Well("STANDARDIZE-DCAL")
    well.depth = np.array([0.0, 1.0])
    well.add_curve("DCAL", np.array([-0.5, 0.2]), "in", "Differential caliper")

    standardized = standardize_curves(well, rename_curves=True, convert_units=True)

    assert standardized.get("CALI") is None
    assert standardized.get("DCAL") is not None


def test_phin_is_not_treated_as_raw_nphi_alias():
    well = Well("STANDARDIZE-PHIN")
    well.depth = np.array([0.0, 1.0])
    well.add_curve(
        "PHIN", np.array([0.20, 0.65]), "v/v", "Lithology-corrected neutron porosity"
    )

    standardized = standardize_curves(well, rename_curves=True, convert_units=True)

    assert standardized.get("NPHI") is None
    assert standardized.get("PHIN") is not None
    np.testing.assert_allclose(
        standardized["PHIN"].data, np.array([0.20, np.nan]), equal_nan=True
    )


def test_sonic_standardization_masks_implausible_dtc_and_dts_values():
    well = Well("STANDARDIZE-SONIC-SCREEN")
    well.depth = np.array([0.0, 1.0, 2.0, 3.0])
    well.add_curve(
        "DTC", np.array([80.0, 10.0, 320.0, 120.0]), "us/ft", "Compressional sonic"
    )
    well.add_curve("DTS", np.array([150.0, 15.0, 4.0e7, 500.0]), "us/ft", "Shear sonic")

    standardized = standardize_curves(well, rename_curves=True, convert_units=True)

    np.testing.assert_allclose(
        standardized["DTC"].data,
        np.array([80.0, np.nan, np.nan, 120.0]),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        standardized["DTS"].data,
        np.array([150.0, np.nan, np.nan, 500.0]),
        equal_nan=True,
    )


def test_standardization_converts_percent_like_nphi_and_masks_outliers():
    well = Well("STANDARDIZE-NPHI-PERCENT")
    well.depth = np.array([0.0, 1.0, 2.0, 3.0])
    well.add_curve("NPHI", np.array([20.0, 35.0, 55.0, 153041.0]), "m3/m3", "Neutron")

    standardized = standardize_curves(well, rename_curves=True, convert_units=True)

    np.testing.assert_allclose(
        standardized["NPHI"].data,
        np.array([0.20, 0.35, 0.55, np.nan]),
        equal_nan=True,
    )


def _sonic_sp_well(dtc) -> Well:
    well = Well("SONIC-SP")
    well.depth = 1000.0 + 0.5 * np.arange(6)
    well.depth_unit = "m"
    well.add_curve("GR", np.full(6, 40.0), "API", "Gamma ray")
    well.add_curve("DTC", np.asarray(dtc, dtype=float), "us/ft", "Compressional sonic")
    well.add_curve("SP", np.array([-6183.0, 80.0, 75.0, 70.0, 60.0, 85.0]), "mV", "SP")
    return well


def test_current_interpretation_screens_impossible_sonic_and_sp():
    # 20.9 us/ft is faster than any rock matrix (dolomite, the fastest, at
    # 26,000 ft/s or 38.5 us/ft); 16/7-1 logs its salt at 20.9, 0.3048 times
    # halite's 67 us/ft. An SP of -6183 mV is no formation potential. The
    # "current" standardization removes both; the default screening, which the
    # context export uses, keeps them.
    dtc = [20.9, 20.9, 20.9, 20.9, 90.0, 100.0]
    with pytest.warns(DataQualityWarning) as caught:
        result = run_dynamic_workflow(_sonic_sp_well(dtc), DynamicWorkflowConfig(rw=0.05))

    nan = np.nan
    np.testing.assert_allclose(result.well["DTC"].data, [nan] * 4 + [90.0, 100.0], equal_nan=True)
    np.testing.assert_allclose(result.well["SP"].data, [nan, 80.0, 75.0, 70.0, 60.0, 85.0], equal_nan=True)
    # The workflow repeats the standardizer's warning word for word, so a
    # caller that collects both, such as the MCP server, shows it once.
    message = "DTC: 4 of 6 samples are outside the plausible DTC range 38.5 to 300 us/ft and were removed."
    assert message in result.warnings
    assert message in [str(item.message) for item in caught]
    # Porosity runs on the sonic that is left.
    phis = result.well["PHIS"].data
    assert np.isnan(phis[:4]).all() and np.isfinite(phis[4:]).all()

    default = standardize_curves(_sonic_sp_well(dtc), rename_curves=True, convert_units=True)
    np.testing.assert_allclose(default["DTC"].data, dtc)
    assert default["SP"].data[0] == -6183.0
    benchmark = run_dynamic_workflow(_sonic_sp_well(dtc), DynamicWorkflowConfig(rw=0.05, calculation_profile="benchmark"))
    np.testing.assert_allclose(benchmark.well["DTC"].data, dtc)


def test_a_wholly_impossible_sonic_skips_porosity_without_errors():
    with pytest.warns(DataQualityWarning, match="DTC: 6 of 6 samples"):
        result = run_dynamic_workflow(_sonic_sp_well([20.9] * 6), DynamicWorkflowConfig(rw=0.05))
    assert "porosity" in result.steps_skipped
    assert not result.errors


def test_the_mcp_summary_lists_a_range_screening_warning_once():
    pytest.importorskip("mcp")
    from petrophys.mcp.serialize import interpretation_summary

    with pytest.warns(DataQualityWarning) as caught:
        result = run_dynamic_workflow(_sonic_sp_well([20.9] * 4 + [90.0, 100.0]), DynamicWorkflowConfig(rw=0.05))
    summary = interpretation_summary(
        result, "interpretation-1", "SONIC-SP", captured_warnings=[str(item.message) for item in caught]
    )
    assert sum("outside the plausible DTC range" in warning for warning in summary["warnings"]) == 1
