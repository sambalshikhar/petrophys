"""Calculations behind the GeoLink eval context benchmark.

The full reproduction of the benchmark needs the GeoLink LAS files and is
verified outside the test suite with scripts/generate_benchmark_context.py.
"""
import numpy as np
import pytest

from petrophys import Well
from petrophys.evaluation.diagnostics import SaturationAnalysis
from petrophys.evaluation.net_pay.flags import generate_net_flags
from petrophys.evaluation.net_pay.net_pay_module import NetPayModule
from petrophys.preprocessing.quality.bad_hole.flag_generator import generate_bad_hole_flags
from petrophys.preprocessing.standardization import standardize_curves
from petrophys.reporting import reference_export
from petrophys.workflow import DynamicWorkflow, DynamicWorkflowConfig
from petrophys.workflow.dynamic_workflow import _profile_setting
from petrophys.workflow.zoned_batch import run_dynamic_workflow_with_zones


def _well(n_samples=40, name="BENCHMARK-TEST") -> Well:
    well = Well(name)
    well.depth = np.arange(8000.0, 8000.0 + 0.5 * n_samples, 0.5)
    well.depth_unit = "ft"
    well.add_curve("GR", np.linspace(25.0, 125.0, n_samples), "API", "Gamma Ray")
    well.add_curve("RHOB", np.linspace(2.20, 2.62, n_samples), "g/cc", "Bulk Density")
    well.add_curve("NPHI", np.linspace(0.08, 0.32, n_samples), "v/v", "Neutron")
    well.add_curve("PEF", np.linspace(2.0, 4.5, n_samples), "b/e", "PEF")
    well.add_curve("RDEP", np.linspace(50.0, 2.0, n_samples), "ohm.m", "Deep Resistivity")
    well.add_curve("CALI", np.full(n_samples, 8.7), "in", "Caliper")
    return well


def _config(**settings):
    return DynamicWorkflowConfig(
        rw=0.05,
        skip_steps=["temperature", "rmf", "invasion", "mhi"],
        **settings,
    )


def _run(well=None, **settings):
    result = DynamicWorkflow(_config(**settings)).run(well if well is not None else _well())
    assert result.success
    return result


def _step(result, name):
    return next(step for step in result.steps_executed if step.step == name)


def test_defaults():
    config = DynamicWorkflowConfig(calculation_profile="geolink")

    assert (config.gr_min, config.gr_max) == (None, None)
    assert config.matrix_type == "sandstone"
    # Left unset, the profile-dependent settings take the eval package's values.
    assert (config.vcl_cutoff, config.sw_cutoff, config.phie_cutoff) == (None, None, None)
    assert (_profile_setting(config, "vcl_cutoff"), _profile_setting(config, "sw_cutoff")) == (0.3, 0.7)
    assert _profile_setting(config, "standardize_input") is False
    assert config.mineral_solver_version == 1


def test_core_saturation_is_converted_outside_the_benchmark_only():
    from petrophys._internal.benchmark import historical_benchmark

    def well():
        well = Well("CORE")
        well.depth = np.arange(1000.0, 1002.0, 0.5)
        well.depth_unit = "m"
        well.add_curve("SWC", np.array([45.0, 60.0, np.nan, 30.0]), "frac", "Core water saturation")
        return well

    with pytest.warns(UserWarning, match="SWC: inferred percent"):
        standardized = standardize_curves(well())
    assert standardized.get("SWC").unit == "v/v"
    assert np.nanmax(standardized.get("SWC").data) == pytest.approx(0.60)

    # The benchmark packages kept the percent values under the "frac" label.
    with historical_benchmark():
        standardized = standardize_curves(well())
    assert standardized.get("SWC").unit == "frac"
    assert np.nanmax(standardized.get("SWC").data) == pytest.approx(60.0)


def test_collision_keeps_the_curve_with_more_valid_samples():
    well = Well("SONIC")
    well.depth = np.arange(1000.0, 1002.0, 0.5)
    well.depth_unit = "m"
    well.add_curve("DTC", np.array([80.0, 81.0, 82.0, 83.0]), "us/ft", "Compressional")
    well.add_curve("DT", np.array([90.0, np.nan, 92.0, 93.0]), "us/ft", "Sonic")

    from petrophys._internal.benchmark import historical_benchmark

    with historical_benchmark():
        standardized = standardize_curves(well)

    assert "DTC" in standardized.curves and "DT" not in standardized.curves
    assert standardized.get("DTC").data[0] == 80.0


def test_washout_needs_a_caliper_and_ignores_pad_calipers():
    well = Well("WASHOUT")
    well.depth = np.arange(1000.0, 1002.0, 0.5)
    well.depth_unit = "m"
    well.add_curve("DRHO", np.array([0.0, 0.3, 0.0, 0.0]), "g/cc", "Density correction")
    well.add_curve("HD1", np.array([8.6, 8.6, 12.0, 8.6]), "in", "Pad caliper")

    assert not generate_bad_hole_flags(well, bit_size=8.5)["washout"].any()

    well.add_curve("CALI", np.array([8.6, 8.6, 12.0, 8.6]), "in", "Caliper")
    assert generate_bad_hole_flags(well, bit_size=8.5)["washout"].tolist() == [False, True, True, False]


def test_an_8_5_in_bit_size_is_assumed_without_one():
    params = _step(_run(calculation_profile="geolink"), "bad_hole").parameters

    assert (params["bit_size_source"], params["bit_size_used"]) == ("fallback_8.5", 8.5)


def test_net_flags_use_strict_cutoffs():
    flags = generate_net_flags(np.array([0.3, 0.2]), np.array([0.1, 0.1]), np.array([0.5, 0.5]))

    assert flags["net_rock"].tolist() == [False, True]
    assert not generate_net_flags(np.array([0.2]), np.array([0.08]), np.array([0.5]))["net_reservoir"][0]
    assert not generate_net_flags(np.array([0.2]), np.array([0.1]), np.array([0.7]))["net_pay"][0]


def test_carbonate_cutoffs_need_the_family_label():
    well = _well(n_samples=1)
    phie, vcl, sw = np.array([0.06]), np.array([0.1]), np.array([0.5])

    from petrophys._internal.benchmark import historical_benchmark

    limestone = NetPayModule(lithology=np.array(["limestone"], dtype=object))
    carbonate = NetPayModule(lithology=np.array(["carbonate"], dtype=object))

    # Outside the benchmark the limestone label takes the carbonate family
    # (tests/test_decisions_evaluation.py).
    with historical_benchmark():
        assert not limestone.calculate(well, vcl=vcl, phie=phie, sw=sw).net_reservoir[0]
        assert carbonate.calculate(well, vcl=vcl, phie=phie, sw=sw).net_reservoir[0]


def test_saturation_analysis_does_not_derive_oil_saturation():
    well = _well()
    well.add_curve("SW", np.linspace(0.2, 1.0, well.n_samples), "v/v", "Water saturation")

    assert "SO" not in SaturationAnalysis().analyze(well, add_curves=False).derived_curves


def test_interpreted_input_porosity_is_recalculated():
    well = _well()
    well.add_curve("PHIE", np.full(well.n_samples, 0.123), "v/v", "Interpreted effective porosity")

    _run(well)

    assert not np.allclose(well.get("PHIE").data, 0.123)
    assert well.get("PHIE_CALC") is None


def test_swirr_is_kept_from_before_permeability():
    steps = [step.step for step in _run().steps_executed]

    assert "permeability" in steps
    assert "permeability_finalize" not in steps


def test_header_parameters_override_default_settings_only():
    def rho_fl(**settings):
        well = _well()
        well.set_parameter("RHO_FL", 1.1)
        result = run_dynamic_workflow_with_zones(well, _config(**settings))
        return _step(result.base_result, "porosity").parameters["rho_fl"], result.warnings

    assert rho_fl()[0] == 1.1
    value, warnings = rho_fl(rho_fl=1.05)
    assert value == 1.05
    assert any("config value kept" in warning for warning in warnings)


def test_export_finds_the_raw_folder_next_to_processed_logs(tmp_path):
    processed = tmp_path / "eval_processed"
    processed.mkdir()
    (tmp_path / "eval").mkdir()

    assert reference_export._raw_input_dir_for_processed(processed) == tmp_path / "eval"


def test_value_range_qc_accepts_another_rule_set():
    from petrophys.preprocessing.quality.value_ranges import ValueRangeRule, apply_value_range_qc

    def well():
        w = Well("RANGES")
        w.depth = np.arange(1000.0, 1001.5, 0.5)
        w.depth_unit = "m"
        w.add_curve("SP", np.array([-999.0, 10.0, 20.0]), "mV", "SP")
        w.add_curve("RMF", np.array([0.005, 0.05, 0.5]), "ohm.m", "Rmf")
        return w

    default = well()
    apply_value_range_qc(default)
    custom = well()
    apply_value_range_qc(custom, rules={"RMF": ValueRangeRule(0.01, 10.0, "nan", ("RMF",))})

    assert np.isnan(default.get("SP").data[0]) and default.get("RMF").data[0] == 0.005
    assert custom.get("SP").data[0] == -999.0 and np.isnan(custom.get("RMF").data[0])
