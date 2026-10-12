import warnings

import numpy as np

from petrophys.evaluation.diagnostics import NumericalStatisticalAnalysis
from petrophys.model import Well
from petrophys.reporting.reference_export import (
    _CONTEXT_VALUE_RANGE_RULES,
    _build_reference_statistics,
    _synchronize_post_qc_dependent_curves,
)
from petrophys.preprocessing.quality.value_ranges import (
    VALUE_RANGE_QC_PARAMETER,
    apply_value_range_qc,
    value_range_qc_note,
)


def test_value_range_qc_sets_out_of_range_values_to_nan():
    well = Well("QC-RANGES")
    well.depth = np.arange(4.0)
    well.add_curve("PHIT", np.array([-0.01, 0.20, 0.46, 0.45]), "v/v", "Total Porosity")
    well.add_curve("RMF", np.array([0.0005, 0.005, 12.0, 1.0]), "ohm.m", "Mud Filtrate")
    well.add_curve("SO", np.array([-0.2, 0.4, 1.2, 1.0]), "v/v", "Oil Saturation")

    report = apply_value_range_qc(well)

    np.testing.assert_allclose(well.get("PHIT").data, np.array([np.nan, 0.20, np.nan, 0.45]), equal_nan=True)
    np.testing.assert_allclose(well.get("RMF").data, np.array([np.nan, 0.005, np.nan, 1.0]), equal_nan=True)
    np.testing.assert_allclose(well.get("SO").data, np.array([0.0, 0.4, 1.0, 1.0]))
    assert well.get_parameter(VALUE_RANGE_QC_PARAMETER) == report
    assert any(item["curve"] == "PHIT" and item["changed"] == 2 for item in report)
    assert any(item["curve"] == "RMF" and item["changed"] == 2 for item in report)
    assert any(item["curve"] == "SO" and item["changed"] == 2 for item in report)


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


def _segment_after_qc(phie, sw, perm, **workflow_curves):
    """A context segment after the exporter's value-range QC.

    ``workflow_curves`` hold dependent curves as the workflow computed them,
    from the porosity and permeability before that QC.
    """
    well = Well("QC-SEGMENT")
    well.depth = np.arange(float(len(phie)))
    well.add_curve("PHIE", np.array(phie, dtype=float), "v/v", "Effective Porosity")
    well.add_curve("SW", np.array(sw, dtype=float), "v/v", "Water Saturation")
    well.add_curve("PERM", np.array(perm, dtype=float), "mD", "Permeability")
    for name, values in workflow_curves.items():
        well.add_curve(name, np.array(values, dtype=float), "", name)
    apply_value_range_qc(well, rules=_CONTEXT_VALUE_RANGE_RULES)
    return well


def _segment_without_porosity():
    # PHIE 0.50 and 0.60 exceed the 0.45 limit: QC removes all porosity.
    # Before it, PHI_Z = 0.5/0.5 and 0.6/0.4, RQI = 0.0314*sqrt(PERM/PHIE),
    # FZI = RQI/PHI_Z and BVW = PHIE*SW.
    return _segment_after_qc(
        [0.50, 0.60], [0.5, 0.5], [100.0, 400.0],
        PHI_Z=[1.0, 1.5], RQI=[0.4441, 0.8107], FZI=[0.4441, 0.5405], BVW=[0.25, 0.30],
    )


def _segment_without_permeability():
    # PERM 5e-5 and 0 mD are below the 1e-4 mD limit: QC removes all
    # permeability. PHIE 0.50 at the second depth exceeds 0.45.
    return _segment_after_qc(
        [0.20, 0.50], [0.5, 0.4], [5e-5, 0.0],
        PHI_Z=[0.25, 1.0], PHI_NORM=[0.25, 1.0], RQI=[0.0005, 0.0003], FZI=[0.002, 0.0003],
        BVW=[0.10, 0.20],
    )


def _segment_losing_one_porosity_sample():
    # PHIE 0.50 at the second depth exceeds 0.45; SW and PERM are kept at both.
    return _segment_after_qc(
        [0.20, 0.50], [0.5, 0.5], [100.0, 100.0],
        PHI_Z=[0.25, 1.0], RQI=[0.7021, 0.4441], FZI=[2.8085, 0.4441], BVW=[0.10, 0.25],
    )


def _data_completeness(well):
    with warnings.catch_warnings():
        # Without VCL the analyzer's net-pay fallback warns; completeness ignores it.
        warnings.simplefilter("ignore", UserWarning)
        result = NumericalStatisticalAnalysis().analyze(well, add_curves=False)
    return result.derived_curves["DATA_COMPLETENESS"].data


def test_segment_without_porosity_after_qc_has_no_porosity_dependents():
    well = _segment_without_porosity()

    _synchronize_post_qc_dependent_curves(well)

    for name in ("PHI_Z", "RQI", "FZI", "BVW"):
        assert np.isnan(well.get(name).data).all(), name
    assert well.get("PHI_NORM") is None
    stats = _build_reference_statistics(well)
    assert stats["PERM_overall"]["maximum"] == 400.0
    assert not {"PHI_NORM_overall", "RQI_overall", "FZI_overall", "BVW_overall"} & set(stats)


def test_benchmark_segment_without_porosity_keeps_the_workflow_dependents():
    well = _segment_without_porosity()

    _synchronize_post_qc_dependent_curves(well, benchmark=True)

    np.testing.assert_allclose(well.get("PHI_Z").data, [1.0, 1.5])
    np.testing.assert_allclose(well.get("RQI").data, [0.4441, 0.8107])
    np.testing.assert_allclose(well.get("FZI").data, [0.4441, 0.5405])
    np.testing.assert_allclose(well.get("BVW").data, [0.25, 0.30])
    # The eval package reports PHI_NORM from the pre-QC PHI_Z.
    assert _build_reference_statistics(well)["PHI_NORM_overall"]["maximum"] == 1.5


def test_segment_without_permeability_after_qc_keeps_phi_norm_but_not_rqi_or_fzi():
    well = _segment_without_permeability()

    _synchronize_post_qc_dependent_curves(well)

    # PHI_NORM = 0.20 / (1 - 0.20) = 0.25 and BVW = 0.20 * 0.5 = 0.10.
    np.testing.assert_allclose(well.get("PHI_NORM").data, [0.25, np.nan])
    np.testing.assert_allclose(well.get("PHI_Z").data, [0.25, np.nan])
    np.testing.assert_allclose(well.get("BVW").data, [0.10, np.nan])
    assert np.isnan(well.get("RQI").data).all()
    assert np.isnan(well.get("FZI").data).all()
    stats = _build_reference_statistics(well)
    assert stats["PHI_NORM_overall"]["minimum"] == stats["PHI_NORM_overall"]["maximum"] == 0.25
    assert not {"RQI_overall", "FZI_overall", "PERM_overall"} & set(stats)


def test_benchmark_segment_without_permeability_keeps_phi_norm_rqi_and_fzi():
    well = _segment_without_permeability()

    _synchronize_post_qc_dependent_curves(well, benchmark=True)

    np.testing.assert_allclose(well.get("BVW").data, [0.10, np.nan])
    np.testing.assert_allclose(well.get("PHI_NORM").data, [0.25, 1.0])
    np.testing.assert_allclose(well.get("PHI_Z").data, [0.25, 1.0])
    np.testing.assert_allclose(well.get("RQI").data, [0.0005, 0.0003])
    stats = _build_reference_statistics(well)
    assert stats["PHI_NORM_overall"]["maximum"] == 1.0
    assert stats["RQI_overall"]["maximum"] == 0.0005


def test_phi_z_follows_the_post_qc_porosity_and_no_rejected_sample_counts_as_data():
    well = _segment_losing_one_porosity_sample()

    _synchronize_post_qc_dependent_curves(well)

    # RQI = 0.0314 * sqrt(100 / 0.20) = 0.70212 um; FZI = 0.70212 / 0.25 = 2.80850 um.
    np.testing.assert_allclose(well.get("RQI").data, [0.70212, np.nan], rtol=1e-4)
    np.testing.assert_allclose(well.get("FZI").data, [2.80850, np.nan], rtol=1e-4)
    np.testing.assert_allclose(well.get("PHI_NORM").data, [0.25, np.nan])
    np.testing.assert_array_equal(well.get("PHI_Z").data, well.get("PHI_NORM").data)
    # Eight curves (PHIE, SW, PERM, PHI_Z, RQI, FZI, BVW and the added
    # PHI_NORM); at the second depth only SW and PERM hold data.
    np.testing.assert_allclose(_data_completeness(well), [1.0, 2 / 8])


def test_benchmark_keeps_the_pre_qc_phi_z_in_data_completeness():
    well = _segment_losing_one_porosity_sample()

    _synchronize_post_qc_dependent_curves(well, benchmark=True)

    np.testing.assert_allclose(well.get("PHI_Z").data, [0.25, 1.0])
    # SW, PERM and the pre-QC PHI_Z of 1.0 at the second depth.
    np.testing.assert_allclose(_data_completeness(well), [1.0, 3 / 8])
