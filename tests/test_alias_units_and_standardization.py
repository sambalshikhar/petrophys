import numpy as np

from petrophys import Well
from petrophys.qaqc.standardization import standardize_curves
import petrophys.workflow.zoned_batch as zoned_batch


def _alias_override(base_name, value, unit=""):
    overrides = {}
    injected = {}
    unmapped = []
    zoned_batch._apply_parameter_alias(base_name, value, unit, overrides, injected, unmapped)
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
    well.add_curve("PHIE_CORECAL", np.array([0.12, 0.18]), "v/v", "Legacy core-calibrated porosity")

    standardized = standardize_curves(well, rename_curves=True, convert_units=True)

    assert standardized.get("RDEP") is not None
    assert standardized.get("RHOB") is not None
    assert standardized.get("NPHI") is not None
    assert standardized.get("CALI") is not None
    assert standardized.get("PHIE_CC") is not None
    assert standardized.get("PHIE_CORECAL") is None
    np.testing.assert_allclose(standardized["RDEP"].data, np.array([3.048, 6.096]))
    np.testing.assert_allclose(standardized["RHOB"].data, np.array([2.2, 2.4]))
    np.testing.assert_allclose(standardized["NPHI"].data, np.array([0.20, 0.35]))
    np.testing.assert_allclose(standardized["CALI"].data, np.array([8.5039416, 10.0000054]))
    np.testing.assert_allclose(standardized["PHIE_CC"].data, np.array([0.12, 0.18]))


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
    well.add_curve("PHIN", np.array([0.20, 0.65]), "v/v", "Lithology-corrected neutron porosity")

    standardized = standardize_curves(well, rename_curves=True, convert_units=True)

    assert standardized.get("NPHI") is None
    assert standardized.get("PHIN") is not None
    np.testing.assert_allclose(standardized["PHIN"].data, np.array([0.20, np.nan]), equal_nan=True)


def test_sonic_standardization_masks_implausible_dtc_and_dts_values():
    well = Well("STANDARDIZE-SONIC-SCREEN")
    well.depth = np.array([0.0, 1.0, 2.0, 3.0])
    well.add_curve("DTC", np.array([80.0, 10.0, 320.0, 120.0]), "us/ft", "Compressional sonic")
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
