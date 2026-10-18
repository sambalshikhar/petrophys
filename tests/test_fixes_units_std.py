"""Regression tests for unit-standardization and imputation fixes."""

import warnings

import numpy as np
import pytest

import petrophys
from petrophys._internal.benchmark import historical_benchmark
from petrophys.io import read_las
from petrophys.model.well import Well
from petrophys.preprocessing.standardization import standardize_curves, standardize_units
from petrophys.preprocessing.standardization.unit_converter import (
    benchmark_unit_table,
    detect_porosity_unit,
    get_conversion_factor,
    standardize_porosity_units,
    standardize_resistivity_units,
)


@pytest.mark.parametrize(
    "source, target, factor",
    [
        ("K/M3", "g/cm3", 0.001),
        ("G/C3", "g/cm3", 1.0),
        ("US/F", "us/ft", 1.0),
        ("usec/ft", "us/ft", 1.0),
        ("µs/m", "us/ft", 0.3048),
        ("μs/m", "us/ft", 0.3048),
        ("ft3/ft3", "v/v", 1.0),
        ("dec", "v/v", 1.0),
        ("frac", "v/v", 1.0),
    ],
)
def test_common_las_unit_spellings_convert(source, target, factor):
    assert get_conversion_factor(source, target) == pytest.approx(factor)


def test_benchmark_keeps_old_unit_table():
    for context in (benchmark_unit_table, historical_benchmark):
        with context():
            assert get_conversion_factor("K/M3", "g/cm3") is None
            assert get_conversion_factor("ft3/ft3", "v/v") is None


def test_rhob_in_k_m3_is_converted_not_masked():
    well = Well("las20")
    well.depth = np.arange(3.0)
    well.add_curve("RHOB", np.array([2550.0, 2450.0, 2400.0]), "K/M3")
    standardize_curves(well)
    assert well["RHOB"].unit == "g/cm3"
    np.testing.assert_allclose(well["RHOB"].data, [2.55, 2.45, 2.40])


def test_standardize_units_blank_unit_warns_and_returns_curve():
    kgs = read_las(petrophys.sample("1046123714"))
    original = kgs["CNLS"].data.copy()
    with pytest.warns(UserWarning, match="left unconverted"):
        result = standardize_units(kgs["CNLS"])
    np.testing.assert_array_equal(result.data, original)


def test_explicit_vv_porosity_not_rescaled_by_spike():
    data, unit = standardize_porosity_units(np.array([0.20, 0.25, 0.30, 1.05]), "v/v")
    np.testing.assert_allclose(data, [0.20, 0.25, 0.30, 1.05])
    assert unit == "v/v"
    data, _ = standardize_porosity_units(np.array([20.0, 25.0, 30.0]), "")
    np.testing.assert_allclose(data, [0.20, 0.25, 0.30])
    assert detect_porosity_unit(np.array([0.2, 0.25, 1.01])) == "v/v"


def test_unknown_resistivity_unit_keeps_its_label():
    with pytest.warns(UserWarning, match="mS/m"):
        data, unit = standardize_resistivity_units(np.array([10.0]), "mS/m")
    assert unit == "mS/m"
    np.testing.assert_allclose(data, [10.0])


def test_percent_core_swc_labelled_frac_is_converted_outside_the_benchmark():
    well = read_las(petrophys.sample("30_6-5"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        standardize_curves(well)
    swc = well["SWC"]
    assert swc.unit == "v/v"
    assert np.nanmax(swc.data) == pytest.approx(0.846)

    # The benchmark packages left the percent values under the "frac" label.
    well = read_las(petrophys.sample("30_6-5"))
    with historical_benchmark(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        standardize_curves(well)
    swc = well["SWC"]
    assert swc.unit == "frac"
    assert np.nanmax(swc.data) == pytest.approx(84.6)


def test_ft3_ft3_porosity_relabelled_vv():
    hw = read_las(petrophys.sample("HW-3"))
    original = hw["NPHI"].data.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        standardize_curves(hw)
    assert hw["NPHI"].unit == "v/v"
    finite = np.isfinite(hw["NPHI"].data)
    np.testing.assert_allclose(hw["NPHI"].data[finite], original[finite])


def test_imputer_accepts_ft3_ft3_nphi():
    from petrophys.preprocessing.imputation import impute_missing_curve

    hw = read_las(petrophys.sample("HW-3"))
    data = hw["NPHI"].data.copy()
    data[100:120] = np.nan
    hw["NPHI"].data = data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = impute_missing_curve(hw, "NPHI")
    assert result.missing_before == 20
    assert result.imputed_count == 20


def test_imputer_leaves_finite_out_of_range_values():
    from petrophys.preprocessing.imputation import load_pretrained_imputer

    well = read_las(petrophys.sample("15_9-14"))
    data = well["RHOB"].data.copy()
    index = np.where(np.isfinite(data))[0]
    data[index[5000:5010]] = np.nan
    data[index[8000]] = 0.9
    data[index[8001]] = 3.6
    well["RHOB"].data = data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = load_pretrained_imputer().impute_missing(well, "RHOB")
    assert well["RHOB"].data[index[8000]] == 0.9
    assert well["RHOB"].data[index[8001]] == 3.6
    assert result.missing_before == int(np.isnan(data).sum())
