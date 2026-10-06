"""LAS metadata inputs for the unified workflow."""

import numpy as np
import pytest

import petrophys as pp
from petrophys.workflow.zoned_batch import lithology_domain_from_metadata


def _well():
    well = pp.Well("HW-31")
    well.depth = np.array([1000.0, 1000.5, 1001.0, 1001.5])
    well.depth_unit = "m"
    return well


def test_lithology_domain_reads_round_trip_las_metadata():
    well = _well()
    assert lithology_domain_from_metadata(well) is None
    well.set_parameter("PP_DOMAIN", "Carbonate")
    assert lithology_domain_from_metadata(well) == "carbonate"
    well.set_parameter("PP_DOMAIN", "unknown")
    with pytest.raises(ValueError, match="PP_DOMAIN"):
        lithology_domain_from_metadata(well)


def test_interpret_uses_las_workflow_parameters_and_explicit_options_win():
    well = _well()
    well.set_parameter("A", 0.62)
    well.set_parameter("M", 2.15)
    well.set_parameter("N", 2.05)
    well.set_parameter("PP_DOMAIN", "carbonate")

    result = pp.interpret(well)
    assert result.config.a == pytest.approx(0.62)
    assert result.config.m == pytest.approx(2.15)
    assert result.config.n == pytest.approx(2.05)
    assert result.config.lithology_domain == "carbonate"

    overridden = pp.interpret(well, a=1.0, lithology_domain="clastic")
    assert overridden.config.a == pytest.approx(1.0)
    assert overridden.config.lithology_domain == "clastic"


def test_interpret_preserves_native_las_rw_provenance():
    well = _well()
    well.add_curve("GR", np.array([20.0, 30.0, 40.0, 50.0]), "API")
    well.set_parameter("RW", 0.07)
    well.set_parameter("RWT", 230.0)
    well.set_parameter("RWT__UNIT", "degF")

    result = pp.interpret(well)
    rw_step = next(step for step in result.steps_executed if step.step == "rw")

    assert result.config.rw is None
    assert rw_step.parameters["source_priority"] == "las.RW"
    assert rw_step.parameters["rw_at_measured_temp"] == pytest.approx(0.07)
    assert rw_step.parameters["measured_temp_f"] == pytest.approx(230.0)


@pytest.mark.parametrize("method", ["linear", "larionov_tertiary", "steiber"])
def test_gr_equation_from_las_metadata_is_used_by_api_and_context(method):
    from petrophys.reporting.reference_export import _context_workflow_config
    from petrophys.workflow import run_dynamic_workflow_with_zones

    well = _well()
    well.add_curve("GR", np.array([20.0, 40.0, 60.0, 100.0]), "API")
    well.set_parameter("GR_METHOD", method)
    well.set_parameter("GR_MIN", 20.0)
    well.set_parameter("GR_MAX", 100.0)
    igr = np.array([0.0, 0.25, 0.5, 1.0])
    expected = {
        "linear": igr,
        "larionov_tertiary": np.clip(0.083 * (2 ** (3.7 * igr) - 1), 0, 1),
        "steiber": igr / (3 - 2 * igr),
    }[method]

    result = pp.interpret(well)
    np.testing.assert_allclose(result.well["VCL"].data, expected)
    config = _context_workflow_config(well=well)
    packaged = run_dynamic_workflow_with_zones(well, config)
    np.testing.assert_allclose(packaged.well["VCL"].data, expected)
    assert next(s for s in packaged.base_result.steps_executed if s.step == "vclay").method == method

    overridden = pp.interpret(well, gr_method="linear")
    np.testing.assert_allclose(overridden.well["VCL"].data, igr)


def test_vcl_trace_reports_the_endpoints_used_in_the_equation():
    from petrophys.reporting.reference_export import _curve_route_reason, _vcl_curve_note

    well = _well()
    well.add_curve("GR", np.array([0.0, 40.0, 100.0, 160.0]), "API")
    result = pp.interpret(well, gr_min=0.0, gr_max=160.0)
    trace = next(s for s in result.steps_executed if s.step == "vclay")

    assert trace.parameters["gr_min_used"] == 0.0
    assert trace.parameters["gr_max_used"] == 160.0
    np.testing.assert_allclose(result.well["VCL"].data, [0, 0.25, 0.625, 1])
    assert "IGR=clip((GR-0)/(160-0),0,1)" in _vcl_curve_note(trace)
    reason = _curve_route_reason("VCL", trace, result.well)
    assert "GRmin=0 API, GRmax=160 API" in reason


def test_las_auto_gr_endpoints_are_computed_from_samples(tmp_path):
    from petrophys.io import export_las
    from petrophys.reporting.reference_export import _read_context_input_well
    from petrophys.workflow import run_dynamic_workflow_with_zones

    well = _well()
    well.add_curve("GR", np.array([0.0, 40.0, 100.0, 160.0]), "API")
    well.set_parameter("GR_MIN", "AUTO")
    well.set_parameter("GR_MAX", "auto")
    path = tmp_path / "well.las"
    export_las(well, path)
    loaded = pp.read(path)
    result = pp.interpret(loaded)
    expected = [0, 34 / 145, 94 / 145, 1]
    np.testing.assert_allclose(result.well["VCL"].data, expected)
    trace = next(s for s in result.steps_executed if s.step == "vclay")
    assert trace.parameters["gr_min_used"] == pytest.approx(6.0)
    assert trace.parameters["gr_max_used"] == pytest.approx(151.0)

    context_well, config = _read_context_input_well(path)
    assert config.gr_min is None and config.gr_max is None
    packaged = run_dynamic_workflow_with_zones(context_well, config)
    np.testing.assert_allclose(packaged.well["VCL"].data, expected)

    overridden = pp.interpret(loaded, gr_min=10.0, gr_max=150.0)
    np.testing.assert_allclose(overridden.well["VCL"].data, [0, 30 / 140, 90 / 140, 1])


def test_las_cutoffs_override_lithology_defaults():
    well = _well()
    phi = np.array([0.06, 0.09, 0.05, 0.1])
    well.add_curve("LITHOLOGY_GEOLINK", np.full(4, 9.0), "")
    well.add_curve("GR", np.zeros(4), "API")
    well.add_curve("RHOB", 2.71 - 1.71 * phi, "g/cc")
    well.add_curve("NPHI", phi, "v/v")
    well.add_curve("RDEP", np.full(4, 10.0), "ohm.m")
    well.set_parameter("VCL_CUTOFF", 0.3)
    well.set_parameter("PHIE_CUTOFF", 0.08)
    well.set_parameter("SW_CUTOFF", 0.7)
    result = pp.interpret(well).raise_for_errors()
    assert result.config.vcl_cutoff == 0.3
    assert result.config.phie_cutoff == 0.08
    assert result.config.sw_cutoff == 0.7
    np.testing.assert_array_equal(result.well["NET_RES"].data, [0, 1, 0, 1])
    np.testing.assert_array_equal(result.well["RESFLAG"].data, [0, 1, 0, 1])
    overridden = pp.interpret(well, phie_cutoff=0.045)
    np.testing.assert_array_equal(overridden.well["NET_RES"].data, [1, 1, 1, 1])
