"""Implausible inputs are reported, never dropped silently."""

import warnings

import numpy as np
import pytest

import petrophys as pp
from petrophys import Well
from petrophys.preprocessing.standardization import standardize_curves


def _density_well(values, unit="g/cc"):
    depth = np.arange(1000.0, 1000.0 + len(values), 1.0)
    well = Well("DENSITY")
    well.depth = depth
    well.depth_unit = "m"
    well.add_curve("DEPT", depth, "m", "Depth")
    well.add_curve("GR", np.full(depth.size, 60.0), "API", "Gamma ray")
    well.add_curve("RHOZ", np.asarray(values, dtype=float), unit, "Density")
    return well


def test_mislabelled_unit_raises_data_quality_warning_with_hint():
    well = _density_well(np.full(50, 2400.0))

    with pytest.warns(pp.DataQualityWarning, match="1000x too large, check the unit label 'g/cc'"):
        standardize_curves(well, rename_curves=False, convert_units=True)

    report = well.get("RHOZ").metadata.custom["range_screening"]
    assert (report["masked"], report["finite"], report["standard_curve"]) == (50, 50, "RHOB")


def test_isolated_spikes_are_recorded_without_a_warning():
    values = np.full(50, 2.4)
    values[:3] = 9.0
    well = _density_well(values)

    with warnings.catch_warnings():
        warnings.simplefilter("error", pp.DataQualityWarning)
        standardize_curves(well, rename_curves=False, convert_units=True)

    assert well.get("RHOZ").metadata.custom["range_screening"]["masked"] == 3


def test_interpretation_reports_removed_curves():
    well = _density_well(np.full(50, 2400.0))

    with pytest.warns(pp.DataQualityWarning):
        result = pp.interpret(well, rw=0.07)

    assert any("RHOZ" in warning and "unit label" in warning for warning in result.warnings)
    normalization = next(s for s in result.steps_executed if s.step == "normalization")
    assert normalization.parameters["range_screening"]["RHOZ"]["fraction"] == 1.0


def _logged_well(top=2000.0, base=3000.0, unit="m"):
    depth = np.arange(top, base + 1.0, 1.0)
    well = Well("LOGGED")
    well.depth = depth
    well.depth_unit = unit
    well.add_curve("GR", np.full(depth.size, 60.0), "API", "Gamma ray")
    return well


def test_geology_outside_the_logged_interval_warns():
    with pytest.warns(pp.DataQualityWarning, match="None of the 2 geology rows overlaps"):
        table = pp.read_formations({"A": 100.0, "B": 50000.0}, well=_logged_well(), depth_unit="m")
    assert table.attrs["import_report"]["outside_logged_interval"] == 2


def test_geology_in_the_wrong_unit_gets_a_unit_hint():
    feet_well = _logged_well(6562.0, 9843.0, unit="ft")
    with pytest.warns(pp.DataQualityWarning, match="depths fit if they are in m"):
        pp.read_formations({"A": 2100.0, "B": 2800.0}, well=feet_well)


def test_partially_covered_geology_is_recorded_without_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error", pp.DataQualityWarning)
        table = pp.read_formations({"Above": 500.0, "Inside": 2500.0}, well=_logged_well(), depth_unit="m")
    assert table.attrs["import_report"]["outside_logged_interval"] == 1


def test_unrecognized_lithology_labels_are_reported(tmp_path):
    from petrophys.interpretation.lithology import load_lithology_csv

    path = tmp_path / "lithology.csv"
    path.write_text("top,bottom,lithology\n100,200,Sandstone\n200,300,Granite\n")

    with pytest.warns(pp.DataQualityWarning, match=r"\['Granite'\].*treated as clastic"):
        intervals = load_lithology_csv(path)

    assert [interval.lithology for interval in intervals] == ["clastic", "clastic"]


def test_scale_hint_only_when_most_of_the_curve_is_implausible():
    values = np.full(50, 2.4)
    values[:3] = 2400.0
    well = _density_well(values)
    standardize_curves(well, rename_curves=False, convert_units=True)
    assert well.get("RHOZ").metadata.custom["range_screening"]["hint"] == ""


def test_interpret_fails_fast_when_data_quality_warnings_are_errors():
    import warnings

    import petrophys as pp

    well = pp.read(pp.sample("15_9-14"))
    with warnings.catch_warnings():
        warnings.simplefilter("error", pp.DataQualityWarning)
        with pytest.raises(pp.DataQualityWarning, match="DTE"):
            pp.interpret(well, rw=0.05)
