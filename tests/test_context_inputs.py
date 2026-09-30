"""Standardized well-parameter and core-table inputs."""

import numpy as np
import pandas as pd
import pytest

import petrophys as pp


def _well():
    well = pp.Well("HW-31")
    well.depth = np.array([1000.0, 1000.5, 1001.0, 1001.5])
    well.depth_unit = "m"
    return well


def test_well_parameters_select_well_and_convert_declared_units():
    table = pd.DataFrame(
        {
            "well": ["HW-30", "HW-31"],
            "lithology_domain": ["clastic", "carbonate"],
            "rw": [0.05, 0.07],
            "rw_reference_temperature": [75.0, 25.0],
            "rw_temperature_unit": ["F", "C"],
            "bht": [200.0, 100.0],
            "bht_unit": ["F", "C"],
            "td": [9000.0, 3000.0],
            "depth_unit": ["ft", "m"],
            "gr_clean": [20.0, 20.0],
            "gr_shale": [100.0, 100.0],
        }
    )
    options = pp.read_well_parameters(table, well_name="HW-31")
    assert options["lithology_domain"] == "carbonate"
    assert options["rw"] == pytest.approx(0.07)
    assert options["rw_meas_temp_f"] == pytest.approx(77.0)
    assert options["bht_f"] == pytest.approx(212.0)
    assert options["td_ft"] == pytest.approx(3000.0 / 0.3048)
    assert options["gr_min"] == 20.0
    assert options["gr_max"] == 100.0


def test_well_parameters_reject_unknown_columns_and_duplicate_wells():
    with pytest.raises(ValueError, match="Unsupported"):
        pp.read_well_parameters(pd.DataFrame({"well": ["A"], "mystery": [1]}))
    duplicate = pd.DataFrame({"well": ["A", "A"], "rw": [0.05, 0.06]})
    with pytest.raises(ValueError, match="duplicate"):
        pp.read_well_parameters(duplicate, well_name="A")


def test_core_table_aligns_depths_and_infers_mislabeled_percent_saturation():
    well = _well()
    table = pd.DataFrame(
        {
            "well": ["HW-31", "HW-31"],
            "depth": [1000.0, 1001.0],
            "depth_unit": ["m", "m"],
            "core_porosity": [12.0, 18.0],
            "porosity_unit": ["percent", "percent"],
            "core_permeability": [0.1, 0.2],
            "permeability_unit": ["D", "D"],
            "core_water_saturation": [20.0, 80.0],
            "core_oil_saturation": [80.0, 20.0],
            "saturation_unit": ["frac", "frac"],
        }
    )
    with pytest.warns(UserWarning, match="inferred percent"):
        report = pp.attach_core_measurements(well, table, well_name="HW-31")
    np.testing.assert_allclose(well.get("PHIC").data[[0, 2]], [0.12, 0.18])
    np.testing.assert_allclose(well.get("KH_CORE").data[[0, 2]], [100.0, 200.0])
    np.testing.assert_allclose(well.get("SWC").data[[0, 2]], [0.2, 0.8])
    np.testing.assert_allclose(well.get("SOC").data[[0, 2]], [0.8, 0.2])
    assert report["curves"] == ["KH_CORE", "PHIC", "SOC", "SWC"]


def test_core_table_averages_duplicate_plugs_at_one_log_depth():
    well = _well()
    table = pd.DataFrame(
        {
            "well": ["HW-31", "HW-31"],
            "depth": [1000.49, 1000.51],
            "depth_unit": ["m", "m"],
            "core_porosity": [0.1, 0.2],
            "porosity_unit": ["fraction", "fraction"],
        }
    )
    report = pp.attach_core_measurements(well, table, well_name="HW-31")
    assert well.get("PHIC").data[1] == pytest.approx(0.15)
    assert report["duplicate_log_rows_averaged"] == 1


def test_packaged_context_input_templates_are_discoverable():
    for name in ("well_parameters_template", "core_measurements_template"):
        assert pp.sample(name).is_file()
