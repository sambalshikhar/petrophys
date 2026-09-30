import numpy as np

from petrophys import Well
from petrophys.preprocessing.standardization import standardize_curves
from petrophys.preprocessing.standardization.alias_map import get_standard_mnemonic
from petrophys.workflow import DynamicWorkflowConfig, run_dynamic_workflow
import petrophys.workflow.zoned_batch as zoned_batch


def _alias_override(base_name, value, unit=""):
    overrides = {}
    injected = {}
    unmapped = []
    zoned_batch._apply_parameter_alias(
        base_name, value, unit, overrides, injected, unmapped
    )
    return overrides, injected, unmapped


def test_parameter_aliases_convert_temperature_depth_salinity_and_booleans():
    cases = [
        ("CLFL", 90.0, "kppm", "rmf_salinity_ppm", 90000.0),
        ("CLFL", 90.0, "g/L", "rmf_salinity_ppm", 90000.0),
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
    assert overrides["lithology"] == "carbonate"
    assert unmapped == []


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
    assert zone.config_overrides["rmf_salinity_ppm"] == 80000.0
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
        DynamicWorkflowConfig(rw=0.05),
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
