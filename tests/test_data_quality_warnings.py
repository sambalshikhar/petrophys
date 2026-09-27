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
