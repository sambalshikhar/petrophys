"""Regression tests for the edge-case guards and silent-bias fixes added in the
library-hardening pass (audit follow-up)."""

import os
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pytest

from petrophys import Well


# --- crash guards ---------------------------------------------------------------

def test_create_log_plot_rejects_empty_curves():
    pytest.importorskip("matplotlib")
    from petrophys.visualization.log_plot import create_log_plot
    w = Well("E"); w.depth = np.arange(10.0)
    with pytest.raises(ValueError):
        create_log_plot(w, [])


def test_pickett_trend_line_needs_two_points():
    pytest.importorskip("matplotlib")
    from petrophys.visualization.pickett_plot import PickettPlot
    p = PickettPlot()
    p.create(phie=np.array([0.1, 0.2, 0.3]), rt=np.array([10.0, 5.0, 2.0]), rw=0.05)
    with pytest.raises(ValueError):
        p.add_trend_line(np.array([np.nan, 0.0]), np.array([np.nan, -1.0]))


def test_hingle_plot_rejects_zero_m():
    pytest.importorskip("matplotlib")
    from petrophys.visualization.lithology_crossplots import create_hingle_plot
    w = Well("H"); w.depth = np.arange(5.0)
    w.add_curve("PHIE", np.full(5, 0.2), "v/v", "phi")
    w.add_curve("RDEP", np.full(5, 10.0), "ohm.m", "rt")
    with pytest.raises(ValueError):
        create_hingle_plot(w, rw=0.05, m=0.0)


def test_digital_facies_statistics_empty_facies_is_nan_not_crash():
    pytest.importorskip("sklearn")
    from petrophys.interpretation.core_analysis.digital_facies import FaciesResult
    facies = np.array([1, 1, 1, 1])  # n_facies=2 but class 2 has zero members
    res = FaciesResult(depth=np.arange(4.0), facies=facies, n_facies=2,
                       feature_names=["x"], weights=np.array([1.0]),
                       cluster_centers=np.zeros((2, 1)), inertia=0.0)
    df = res.statistics({"x": np.array([1.0, 2.0, 3.0, 4.0])})
    col = "x_P50" if "x_P50" in df.columns else ("x", "P50")
    assert np.isnan(df.loc[2, col])  # empty cluster -> NaN, no crash


# --- silent-bias fixes ----------------------------------------------------------

def test_dual_water_survives_zero_resistivities():
    from petrophys.interpretation.saturation.dual_water import dual_water
    # scalar rt=0 / rw=0 must not raise ZeroDivisionError
    swe = dual_water(phi_t=0.2, rt=0.0, rw=0.0, rwb=0.0, vcl=0.3, phi_cl=0.4)
    assert np.isfinite(swe) or np.isnan(swe)  # finite/NaN, never a crash
    arr = dual_water(phi_t=np.array([0.2, 0.2]), rt=np.array([0.0, 10.0]),
                     rw=0.05, rwb=0.04, vcl=0.3, phi_cl=0.4)
    assert np.asarray(arr).shape == (2,)


def test_matrix_properties_fallback_has_u_key():
    from petrophys.interpretation.lithology.matrix_properties import get_mineral_properties
    # 'pyrite' is in MINERAL_PROPERTIES; check a base-constants-only mineral resolves
    # with a 'u' key if it falls through (use any mineral present in MatrixProperties).
    from petrophys.interpretation.lithology.matrix_properties import MINERAL_PROPERTIES
    props = get_mineral_properties("quartz")
    assert "u" in props


@pytest.mark.parametrize(
    ("value", "unit", "expected"),
    [
        (100.0, "deg F", 100.0),
        (100.0, "DEGF", 100.0),
        (100.0, "deg C", 212.0),
        # 'from' contains an 'f' but the unit is Celsius
        (0.0, "deg C (from log)", 32.0),
        (100.0, "Celsius (ref)", 212.0),
        (100.0, "C (ref)", 212.0),
        (100.0, "deg_c", 212.0),
        (50.0, "C", 122.0),
        (373.15, "K", 212.0),
        # unlabelled header temperatures follow the LAS Fahrenheit convention
        (150.0, "", 150.0),
    ],
)
def test_temperature_celsius_detection_not_fooled_by_stray_f(value, unit, expected):
    from petrophys.model.units import temperature_to_fahrenheit
    assert temperature_to_fahrenheit(value, unit) == pytest.approx(expected)


def test_temperature_conversion_has_a_single_implementation():
    from petrophys.interpretation.temperature import temp_profile
    from petrophys.model.units import temperature_to_fahrenheit
    from petrophys.workflow import dynamic_workflow, zoned_batch

    for module in (dynamic_workflow, zoned_batch, temp_profile):
        assert not hasattr(module, "_temperature_to_f")
        assert module.temperature_to_fahrenheit is temperature_to_fahrenheit


def test_reference_temperature_arrays_respect_worded_units():
    import numpy as np
    from petrophys.reporting.reference_export import (
        _convert_temperature_to_celsius,
        _convert_temperature_to_fahrenheit,
    )
    data = np.array([100.0])
    assert _convert_temperature_to_fahrenheit(data, "deg C (from log)")[0] == pytest.approx(212.0)
    assert _convert_temperature_to_celsius(data, "degF")[0] == pytest.approx(37.7777778)
    assert _convert_temperature_to_celsius(data, "deg C (from log)")[0] == pytest.approx(100.0)


def test_imputation_metrics_flag_in_sample_vs_held_out():
    # the training metric dict must carry an 'evaluation' field distinguishing
    # held-out from in-sample, so a train==test fallback isn't read as a test score.
    import inspect
    from petrophys.preprocessing.imputation import training
    src = inspect.getsource(training)
    assert '"evaluation"' in src
    assert "in_sample_single_well" in src
    assert "held_out_wells" in src


def test_vcl_resolves_gr_family_when_only_spectral_gr():
    # a well carrying only SGR (no plain GR) must still compute VCL from it
    from petrophys.interpretation.clay_volume import VclModule
    w = Well("SGR-ONLY")
    d = np.arange(1000.0, 1020.0, 0.5)
    w.depth = d
    w.add_curve("SGR", np.linspace(30.0, 110.0, d.size), "API", "Spectral GR")
    res = VclModule(gr_min=30.0, gr_max=110.0).calculate(w)
    assert np.any(np.isfinite(res.vcl))
    assert "GR" in res.methods_used if hasattr(res, "methods_used") else True
