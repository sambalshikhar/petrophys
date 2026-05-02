import numpy as np

from petrophys.base import Well
from petrophys.io.example_format_export import _synchronize_post_qc_dependent_curves
from petrophys.qaqc.value_range_qc import (
    VALUE_RANGE_QC_PARAMETER,
    apply_value_range_qc,
    value_range_qc_note,
)


def test_value_range_qc_sets_out_of_range_values_to_nan():
    well = Well("QC-RANGES")
    well.depth = np.arange(4.0)
    well.add_curve("PHIT", np.array([-0.01, 0.20, 0.46, 0.45]), "v/v", "Total Porosity")
    well.add_curve("RMF", np.array([0.005, 0.20, 12.0, 1.0]), "ohm.m", "Mud Filtrate")

    report = apply_value_range_qc(well)

    np.testing.assert_allclose(well.get("PHIT").data, np.array([np.nan, 0.20, np.nan, 0.45]), equal_nan=True)
    np.testing.assert_allclose(well.get("RMF").data, np.array([np.nan, 0.20, np.nan, 1.0]), equal_nan=True)
    assert well.get_parameter(VALUE_RANGE_QC_PARAMETER) == report
    assert any(item["curve"] == "PHIT" and item["changed"] == 2 for item in report)
    assert any(item["curve"] == "RMF" and item["changed"] == 2 for item in report)


def test_value_range_qc_clips_saturation_aliases():
    well = Well("QC-SAT")
    well.depth = np.arange(4.0)
    well.add_curve("SWA", np.array([-0.2, 0.2, 1.2, np.nan]), "v/v", "Water Saturation")
    well.add_curve("SWXOA", np.array([-0.1, 0.4, 1.1, 0.8]), "v/v", "Flushed Saturation")

    report = apply_value_range_qc(well)

    np.testing.assert_allclose(well.get("SWA").data, np.array([0.0, 0.2, 1.0, np.nan]), equal_nan=True)
    np.testing.assert_allclose(well.get("SWXOA").data, np.array([0.0, 0.4, 1.0, 0.8]), equal_nan=True)
    note = value_range_qc_note(report)
    assert "SWA clipped=2" in note
    assert "SWXOA clipped=2" in note


def test_context_dependent_curves_are_resynchronized_after_range_qc():
    well = Well("QC-DEPENDENTS")
    well.depth = np.arange(2.0)
    well.add_curve("PHIE", np.array([0.20, 0.50]), "v/v", "Effective Porosity")
    well.add_curve("SW", np.array([0.50, 1.00]), "v/v", "Water Saturation")
    well.add_curve("PERM", np.array([100.0, 100.0]), "mD", "Permeability")
    well.add_curve("BVW", np.array([0.10, 0.50]), "v/v", "Stale Bulk Volume Water")

    apply_value_range_qc(well)
    _synchronize_post_qc_dependent_curves(well)

    np.testing.assert_allclose(well.get("PHIE").data, np.array([0.20, np.nan]), equal_nan=True)
    np.testing.assert_allclose(well.get("BVW").data, np.array([0.10, np.nan]), equal_nan=True)
    assert np.isfinite(well.get("RQI").data[0])
    assert np.isnan(well.get("RQI").data[1])
    assert np.isfinite(well.get("FZI").data[0])
    assert np.isnan(well.get("FZI").data[1])
